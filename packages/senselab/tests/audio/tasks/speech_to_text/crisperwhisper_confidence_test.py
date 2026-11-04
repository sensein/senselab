"""CrisperWhisper confidence: each window's final ids, scored from the decode's own logits or encoder output."""

import importlib
import zlib
from typing import Any

import numpy
import pytest
from senselab.audio.tasks.speech_to_text import crisperwhisper as cw
from senselab.audio.tasks.speech_to_text.token_confidence import word_prefix
from senselab.audio.tasks.speech_to_text.window_confidence import (
    Decode,
    Window,
    capture_windows,
    ct2_window_logits,
    spliced_block,
    window_confidence,
)

VOCAB = 6
EOT = 5
PROMPT = [4, 0]
GEN = [1, 2, 3, EOT]
PIECES = {1: " he", 2: "llo", 3: " there", EOT: "<|endoftext|>"}
SPECIAL = {EOT, 4}
DECODED = {
    "clean": [GEN],
    "fallback": [[3, 3, 3, 3, EOT]],
    "recovery": [[1, 2, EOT]],
    "repair": [[1, 2, 2, 2, 2, EOT], [3, EOT]],
}
"""Each case's decode segments; `repair` rewinds to `[1, 2]` and re-decodes; `fallback` and `recovery` are then
replaced by the library with `GEN`, from a sampled decode and from a decode past the early stop."""
KEEP = 2
"""Where `repair` rewinds."""


def row(prefix: list[int]) -> numpy.ndarray:
    """The raw logits a model gives the position after `prefix`, a function of the prefix alone (causality)."""
    return numpy.random.default_rng(zlib.crc32(bytes(prefix))).normal(size=VOCAB).astype(numpy.float32)


def teacher_logprob(prompt: list[int], ids: list[int]) -> list[float]:
    """Teacher-forced log-probability of each of `ids` after `prompt`."""
    out = []
    for j, token in enumerate(ids):
        logits = row(prompt + ids[:j]).astype(numpy.float64)
        out.append(float(logits[token] - logits.max() - numpy.log(numpy.exp(logits - logits.max()).sum())))
    return out


def _expected(ids: list[int] = GEN) -> tuple[float, list[float]]:
    lp = teacher_logprob(PROMPT, ids)
    return float(numpy.exp(numpy.mean(lp[:3]))), [float(numpy.exp(lp[0] + lp[1])), float(numpy.exp(lp[2]))]


class Tokenizer:
    """Decodes `PIECES`."""

    def decode(self, ids: list[int]) -> str:
        """Return the text of `ids`."""
        return "".join(PIECES[i] for i in ids)


class Word:
    """One timed word of a library result."""

    def __init__(self, text: str, start: float, end: float) -> None:
        """Hold `text` from `start` to `end` seconds."""
        self.word, self.text, self.start, self.end = text, text, start, end


class Result:
    """A library result: the two words `GEN` decodes to."""

    words = [Word("hello", 0.0, 0.4), Word("there", 0.4, 0.8)]


class Chunk:
    """One window's share of a long-form result: the text of the words that survive the merge."""

    def __init__(self, text: str) -> None:
        """Hold the surviving `text`."""
        self.text = text


class MergedResult(Result):
    """A long-form result whose only window kept `hello`: `there` is re-covered by the next window."""

    words = [Word("hello", 0.0, 0.4)]
    chunks: list[Chunk] = [Chunk("hello")]


class Encoded:
    """A ct2 encoder output of `features`."""

    def __init__(self, features: object) -> None:
        """Hold the encoded `features`."""
        self.features = features


class CT2Model:
    """A ct2 Whisper: `encode`, and `prefill` + `forward_batch` over `row`; records what each pass was fed."""

    def __init__(self) -> None:
        """Record nothing yet."""
        self.encodes = 0
        self.fed: list[Any] = []

    def encode(self, features: object) -> Encoded:
        """Count one encoder pass."""
        self.encodes += 1
        return Encoded(features)

    def prefill(self, source: object, prompt: list[int]) -> tuple[list[int], numpy.ndarray]:
        """Return the prompt as the state and one row per prompt token, the last predicting the next token."""
        self.fed.append((type(source).__name__, list(prompt)))
        stale = numpy.full((len(prompt) - 1, VOCAB), 50.0, dtype=numpy.float32)
        return list(prompt), numpy.concatenate([stale, row(list(prompt))[None]])[None]

    def forward_batch(self, state: list[int], ids: list[int]) -> numpy.ndarray:
        """Return the rows predicting the token after each of `ids`."""
        self.fed.append(list(ids))
        return numpy.stack([row(state + ids[: i + 1]) for i in range(len(ids))])[None]


class HFModel:
    """A transformers Whisper: `generate` follows a script and returns `row` logits; a forward is teacher forcing."""

    def __init__(self) -> None:
        """Record nothing yet."""
        self.script: list[tuple[list[int], bool]] = []
        self.flags: list[bool] = []
        self.fed: list[list[int]] = []

    def generate(self, features: object, decoder_input_ids: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """Return the next scripted ids after the prefix, with one logits row per step (and one past them if asked)."""
        torch = importlib.import_module("torch")
        ids, extra = self.script.pop(0)
        prefix = decoder_input_ids[0].tolist()
        self.flags.append(bool(kwargs.get("output_logits")))
        sequences = torch.tensor([prefix + ids])
        if not kwargs.get("return_dict_in_generate"):
            return sequences
        logits = tuple(torch.from_numpy(row(prefix + ids[:k]))[None] for k in range(len(ids) + extra))
        if not kwargs.get("output_logits"):
            logits = ()
        return type("Out", (), {"sequences": sequences, "logits": logits, "cross_attentions": ()})()

    def __call__(self, input_features: object, decoder_input_ids: Any, use_cache: bool) -> Any:  # noqa: ANN401
        """Teacher-force `decoder_input_ids`: one `row` per position."""
        torch = importlib.import_module("torch")
        fed = decoder_input_ids[0].tolist()
        self.fed.append(fed)
        logits = numpy.stack([row(fed[: i + 1]) for i in range(len(fed))])
        return type("Out", (), {"logits": torch.from_numpy(logits)[None]})()


class Engine:
    """A crisperwhisper engine of either backend; a window decode runs the case's segments through the model."""

    all_special_ids = SPECIAL
    tokenizer = Tokenizer()
    device = "cpu"

    def __init__(self, kind: str, case: str) -> None:
        """Hold the model of the `kind` backend."""
        self.kind, self.case, self.plan = kind, case, DECODED[case]
        self.model: Any = CT2Model() if kind == "ct2" else HFModel()
        if kind == "ct2":
            self.generate_greedy_with_attention = self._segment

    def _segment(self, encoded: object, prompt: list[int], **kwargs: Any) -> tuple[str, list[int]]:  # noqa: ANN401
        return "state", list(self._planned.pop(0))

    def _generate(self, features: object, prefix: list[int], ids: list[int], extra: bool = False) -> list[int]:
        torch = importlib.import_module("torch")
        self.model.script.append((ids, extra))
        out = self.model.generate(features, decoder_input_ids=torch.tensor([prefix]), max_new_tokens=8)
        return [int(t) for t in out[0].tolist()[len(prefix) :]]

    def _decode(self, features: object, prompt: list[int], segments: list[list[int]]) -> list[int]:
        if self.kind == "ct2":
            encoded = self.model.encode(features)
            self._planned = [list(s) for s in segments]
            ids = self.generate_greedy_with_attention(encoded, prompt, max_new_tokens=8)[1]
            if len(segments) > 1:
                ids = ids[:KEEP] + self.generate_greedy_with_attention(encoded, prompt + ids[:KEEP])[1]
            return ids
        ids = self._generate(features, prompt, segments[0])
        if len(segments) > 1:
            ids = ids[:KEEP] + self._generate(features, prompt + ids[:KEEP], segments[1])
        return ids

    def generate_with_repair(self, features: object, prompt_tokens: list[int], **kwargs: Any) -> list[int]:  # noqa: ANN401
        """Return the window's decode."""
        return self._decode(features, prompt_tokens, self.plan)

    def generate_with_repair_and_attention(
        self,
        features: object,
        prompt_tokens: list[int],
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[list[int], str]:
        """Return the window's decode and its attention."""
        return self._decode(features, prompt_tokens, self.plan), "attention"

    def generate_sampled(self, features: object, prompt_tokens: list[int]) -> list[int]:
        """Return a sampled decode: `GEN`; a ct2 sample runs outside the captured calls."""
        return list(GEN) if self.kind == "ct2" else self._generate(features, prompt_tokens, GEN)

    def greedy_stops_and_decode(self, features: object, prompt_tokens: list[int]) -> list[int]:
        """Return a decode past the early stop, without its end token; transformers gives a step past the ids."""
        return GEN[:-1] if self.kind == "ct2" else self._generate(features, prompt_tokens, GEN[:-1], extra=True)

    def cross_attention_for_tokens(self, features: object, prompt_tokens: list[int], gen_ids: list[int]) -> str:
        """Return the attention of `gen_ids`."""
        return "recaptured"


class CrisperWhisper:
    """The library model: one window, its ids replaced by `GEN` after fallback or early-EOT recovery."""

    def __init__(self, kind: str, case: str, result: type[Result] = Result) -> None:
        """Hold the engine the library would build, and the `result` it returns."""
        self._engine = Engine(kind, case)
        self._result = result

    def transcribe(self, path: str, **kwargs: Any) -> Result:  # noqa: ANN401
        """Decode one window, replacing its ids where the library would, and return its words."""
        engine, features = self._engine, object()
        ids, _ = engine.generate_with_repair_and_attention(features, PROMPT, max_length=8)
        if engine.case == "fallback":
            ids = engine.generate_sampled(features, PROMPT)
        elif engine.case == "recovery":
            ids = [*engine.greedy_stops_and_decode(features, PROMPT), EOT]
        if engine.case in ("fallback", "recovery"):
            assert engine.cross_attention_for_tokens(features, PROMPT, ids) == "recaptured"
        assert ids == GEN
        return self._result()


def _library(kind: str, model: CrisperWhisper) -> cw.Library:
    library = cw.Library.__new__(cw.Library)
    library.engine, library.model = kind, model
    return library


def _transcribe(kind: str, case: str, merged: bool = False, confidence: bool = True) -> tuple[cw.Decoded, Any]:
    library = _library(kind, CrisperWhisper(kind, case, MergedResult if merged else Result))
    return library.transcribe(numpy.zeros(160), "en", 8, confidence), library.model._engine.model


@pytest.mark.env
@pytest.mark.parametrize("case", list(DECODED))
@pytest.mark.parametrize("kind", ["ct2", "transformers"])
def test_confidence_is_of_the_final_ids_from_the_decodes_own_computation(kind: str, case: str) -> None:
    """Both engines score the window's final ids to the teacher-forced values, with no second encoder pass.

    transformers reads them from the logits of the decode's own calls (a repair spliced at the rewind, a recovery's
    end token from the step past its ids) and teacher-forces nothing; ct2 runs one decoder pass on the encoder
    output its decode made.
    """
    pytest.importorskip("torch")
    decoded, model = _transcribe(kind, case)
    mean, words = _expected()
    assert decoded.mean == pytest.approx(mean, rel=1e-5)
    assert decoded.confidences == pytest.approx(words, rel=1e-5)
    assert decoded.encode_reused == 1
    if kind == "ct2":
        assert model.encodes == 1 and model.fed == [("Encoded", PROMPT), GEN[:-1]]
    else:
        assert model.fed == [] and all(model.flags)


@pytest.mark.env
def test_the_decodes_logits_equal_teacher_forcing_where_both_are_computable() -> None:
    """The log-probabilities read from a repaired decode's calls equal a teacher-forced forward of the final ids."""
    torch = pytest.importorskip("torch")
    library = _library("transformers", CrisperWhisper("transformers", "repair"))
    model = library.model._engine.model
    decoded = library.transcribe(numpy.zeros(160), "en", 8, True)
    out = model(input_features=None, decoder_input_ids=torch.tensor([PROMPT + GEN[:-1]]), use_cache=False)
    forced = torch.log_softmax(out.logits[0, -len(GEN) :].double(), dim=-1)[range(len(GEN)), GEN].tolist()
    assert decoded.mean == pytest.approx(float(numpy.exp(numpy.mean(forced[:3]))), rel=1e-5)
    assert decoded.confidences == pytest.approx(
        [float(numpy.exp(forced[0] + forced[1])), float(numpy.exp(forced[2]))], rel=1e-5
    )


@pytest.mark.logic
def test_a_repaired_window_is_the_earlier_call_up_to_the_rewind_then_the_later_one() -> None:
    """Ids before the rewind come from the first call, the escape and after from the second; a tail row ends it."""
    features = object()
    tail = [0.0] * VOCAB
    tail[EOT] = -0.5
    first = Decode(features, PROMPT, [1, 2, 2, 2], [-0.1, -0.2, -9.0, -9.0])
    second = Decode(features, PROMPT + [1, 2], [3], [-0.3], tail)
    block = spliced_block(Window(features, PROMPT, GEN), [first, second], SPECIAL)
    assert block is not None and block["token_logprob"] == [-0.1, -0.2, -0.3, -0.5]
    assert block["avg_logprob"] == pytest.approx(-0.2)


@pytest.mark.logic
def test_a_window_no_call_covers_is_not_spliced() -> None:
    """An id no call on the window's features predicted leaves the window without a block."""
    features = object()
    covered = Decode(features, PROMPT, [1, 2], [-0.1, -0.2])
    assert spliced_block(Window(features, PROMPT, GEN), [covered], SPECIAL) is None
    assert spliced_block(Window(object(), PROMPT, [1, 2]), [covered], SPECIAL) is None


@pytest.mark.env
@pytest.mark.parametrize("kind", ["ct2", "transformers"])
def test_overlap_dropped_ids_not_scored(kind: str) -> None:
    """Ids of words the window merge drops are not scored, in the mean or in any word's confidence."""
    pytest.importorskip("torch")
    decoded, model = _transcribe(kind, "clean", merged=True)
    lp = teacher_logprob(PROMPT, GEN)
    assert decoded.mean == pytest.approx(float(numpy.exp(numpy.mean(lp[:2]))), rel=1e-5)
    assert decoded.confidences == pytest.approx([float(numpy.exp(lp[0] + lp[1]))], rel=1e-5)
    if kind == "ct2":
        assert model.fed == [("Encoded", PROMPT), GEN[:1]]
    assert (decoded.windows, decoded.chunks) == (1, 1)


@pytest.mark.logic
def test_word_prefix_ends_after_the_last_kept_words_own_tokens() -> None:
    """The prefix of one word is `he` + `llo`; of none is empty; of every word is all the ids, end token included."""
    decode = Tokenizer().decode
    assert [word_prefix(GEN, n, lambda t: decode([t]), SPECIAL) for n in (0, 1, 2)] == [0, 2, len(GEN)]


@pytest.mark.env
@pytest.mark.parametrize("kind", ["ct2", "transformers"])
def test_chunks_that_do_not_match_the_windows_leave_the_decode_unscored(kind: str) -> None:
    """A result with more chunks than captured windows cannot say which ids survive, so none is scored."""
    pytest.importorskip("torch")

    class Lost(MergedResult):
        chunks: list[Chunk] = [Chunk("hello"), Chunk("there")]

    decoded = _library(kind, CrisperWhisper(kind, "clean", Lost)).transcribe(numpy.zeros(160), "en", 8, True)
    assert decoded.mean is None and decoded.confidences == [None] and decoded.encode_reused == 0


class TwoWindows(CrisperWhisper):
    """A long-form library model: two windows decoded, and a result that carries no chunks."""

    def transcribe(self, path: str, **kwargs: Any) -> Result:  # noqa: ANN401
        """Decode two windows and return the words of the result."""
        for _ in range(2):
            self._engine.generate_with_repair_and_attention(object(), PROMPT, max_length=8)
        return self._result()


@pytest.mark.env
@pytest.mark.parametrize("kind", ["ct2", "transformers"])
def test_a_multi_window_result_without_matching_chunks_is_unscored(kind: str) -> None:
    """Two windows and no chunk text cannot say which ids survive the merge, so the decode is not scored."""
    pytest.importorskip("torch")
    library = _library(kind, TwoWindows(kind, "clean"))
    decoded = library.transcribe(numpy.zeros(160), "en", 8, True)
    assert decoded.mean is None and decoded.confidences == [None, None]
    assert library.model._engine.model.fed == []


@pytest.mark.env
def test_a_word_count_that_disagrees_with_the_kept_words_leaves_the_mean_unscored() -> None:
    """When the kept-word counts and the result's words differ, no word is scored and the mean is None too."""
    pytest.importorskip("torch")

    class Miscounted(Result):
        chunks: list[Chunk] = [Chunk("hello")]

    decoded = _library("ct2", CrisperWhisper("ct2", "clean", Miscounted)).transcribe(numpy.zeros(160), "en", 8, True)
    assert decoded.mean is None and decoded.confidences == [None, None]


class LongFormLibrary(CrisperWhisper):
    """A long-form library model: window one keeps only `hello`, window two, the final one, is whole."""

    def __init__(self, kind: str) -> None:
        """Build an engine whose first window decodes `hello there`, the second `llo there`, no leading space."""
        super().__init__(kind, "clean")

    def transcribe(self, path: str, **kwargs: Any) -> Result:  # noqa: ANN401
        """Decode two windows and return the words and chunks the merge keeps."""
        for segment in ([1, 2, 3, EOT], [2, 3, EOT]):
            self._engine.plan = [segment]
            self._engine.generate_with_repair_and_attention(object(), PROMPT, max_length=8)

        class Merged(Result):
            words = [Word("hello", 0.0, 0.4), Word("llo", 0.5, 0.6), Word("there", 0.6, 0.9)]
            chunks: list[Chunk] = [Chunk("hello"), Chunk("llo there")]

        return Merged()


@pytest.mark.env
@pytest.mark.parametrize("kind", ["ct2", "transformers"])
def test_a_non_final_window_is_cut_at_the_merge_and_each_window_is_grouped_alone(kind: str) -> None:
    """Words past the merge's cut are not scored, and a window's first piece without a space starts its own word."""
    pytest.importorskip("torch")
    decoded = _library(kind, LongFormLibrary(kind)).transcribe(numpy.zeros(160), "en", 8, True)
    first, second = teacher_logprob(PROMPT, [1, 2]), teacher_logprob(PROMPT, [2, 3])
    assert decoded.confidences == pytest.approx(
        [float(numpy.exp(sum(first))), float(numpy.exp(second[0])), float(numpy.exp(second[1]))], rel=1e-5
    )
    assert decoded.mean == pytest.approx(float(numpy.exp((numpy.mean(first) + numpy.mean(second)) / 2)), rel=1e-5)
    assert decoded.encode_reused == 2


@pytest.mark.env
def test_windows_keep_decode_order_their_encoder_output_and_the_engine_is_left_as_it_was() -> None:
    """A replacement updates its own window in place and keeps its encoder output; the wrappers are removed on exit."""
    pytest.importorskip("torch")
    engine = Engine("ct2", "fallback")
    first, second = object(), object()
    with capture_windows(engine) as windows:
        assert engine.generate_with_repair(first, PROMPT) == DECODED["fallback"][0]
        assert engine.generate_with_repair_and_attention(second, [4], max_length=8) == (
            DECODED["fallback"][0],
            "attention",
        )
        assert engine.cross_attention_for_tokens(first, PROMPT, GEN) == "recaptured"
    assert [(w.features, w.prompt_tokens, w.ids) for w in windows] == [
        (first, PROMPT, GEN),
        (second, [4], DECODED["fallback"][0]),
    ]
    assert [type(w.encoded).__name__ for w in windows] == ["Encoded", "Encoded"]
    assert windows[0].encoded.features is first
    assert engine.generate_greedy_with_attention == engine._segment


@pytest.mark.logic
def test_two_modes_on_shared_features_are_two_windows() -> None:
    """Two prompts decoded on one `features` stay two windows; a replacement updates only its own prompt's."""
    engine = Engine("ct2", "fallback")
    shared, verbatim, intended = object(), [4, 0], [4, 1]
    with capture_windows(engine) as windows:
        engine.generate_with_repair(shared, verbatim)
        engine.generate_with_repair(shared, intended)
        engine.cross_attention_for_tokens(shared, intended, GEN)
    assert [(w.features, w.prompt_tokens, w.ids) for w in windows] == [
        (shared, verbatim, DECODED["fallback"][0]),
        (shared, intended, GEN),
    ]
    assert windows[0].encoded is not windows[1].encoded


@pytest.mark.env
def test_special_ids_are_scored_but_left_out_of_the_mean() -> None:
    """The end token has a log-probability in the block and no weight in `avg_logprob`."""
    pytest.importorskip("torch")
    engine = Engine("ct2", "clean")
    [block] = window_confidence(engine, [Window("f", PROMPT, GEN)], ct2_window_logits, SPECIAL)
    assert block["tokens"] == GEN and len(block["token_logprob"]) == len(GEN)
    assert block["avg_logprob"] == pytest.approx(float(numpy.mean(block["token_logprob"][:3])))
    assert block["avg_logprob"] != pytest.approx(float(numpy.mean(block["token_logprob"])))


@pytest.mark.logic
def test_ct2_window_logits_use_the_recorded_encoder_output_and_refuse_a_short_row_count() -> None:
    """The decoder pass reads the recorded encoder output (the mel only without one); too few rows is an error."""
    engine = Engine("ct2", "clean")
    expected = numpy.stack([row(PROMPT + GEN[:j]) for j in range(len(GEN))])
    assert ct2_window_logits(engine, Window("mel", PROMPT, GEN, Encoded("mel"))).tolist() == expected.tolist()
    ct2_window_logits(engine, Window("mel", PROMPT, GEN))
    assert [fed[0] for fed in engine.model.fed[::2]] == ["Encoded", "str"]

    class Short(CT2Model):
        def forward_batch(self, state: list[int], ids: list[int]) -> numpy.ndarray:
            return super().forward_batch(state, ids[:-1])

    engine.model = Short()
    with pytest.raises(ValueError, match="logit rows"):
        ct2_window_logits(engine, Window("mel", PROMPT, GEN))


@pytest.mark.env
@pytest.mark.parametrize("kind", ["ct2", "transformers"])
def test_confidence_off_records_and_scores_nothing(kind: str) -> None:
    """Off asks no call for logits and runs no decoder pass, leaving every confidence none; on is the scored decode."""
    pytest.importorskip("torch")
    off, model = _transcribe(kind, "clean", confidence=False)
    assert model.fed == [] and not any(getattr(model, "flags", []))
    assert off.mean is None and set(off.confidences) <= {None} and off.confidence_s == 0.0 and off.windows == 1
    assert off.encode_reused == 0
    on, _ = _transcribe(kind, "clean")
    assert on.mean is not None and on.words == off.words and on.confidence_s > 0.0
