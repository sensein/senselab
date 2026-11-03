"""CrisperWhisper confidence: both engines score each window's final ids, teacher-forced, under one definition."""

import importlib
from typing import Any

import numpy
import pytest
from senselab.audio.tasks.speech_to_text import crisperwhisper as cw
from senselab.audio.tasks.speech_to_text.window_confidence import (
    Window,
    capture_windows,
    ct2_teacher_logits,
    window_confidence,
)

VOCAB = 6
EOT = 5
PROMPT = [4, 0]
GEN = [1, 2, 3, EOT]
PIECES = {1: " he", 2: "llo", 3: " there", EOT: "<|endoftext|>"}
LOGITS = numpy.random.default_rng(7).normal(size=(len(GEN), VOCAB)).astype(numpy.float32)
LOGITS[3, EOT] = -9.0
DECODED = {"clean": GEN, "fallback": [3, 3, 3, 3, EOT], "recovery": [1, 2, EOT]}
"""What the window decode returns; on `fallback` and `recovery` the library then replaces it with `GEN`."""


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


class CT2Model:
    """Teacher forcing over `LOGITS`, row `k` predicting position `k`."""

    def __init__(self) -> None:
        """Record nothing fed yet."""
        self.fed: list[list[int]] = []

    def prefill(self, features: object, prompt: list[int]) -> tuple[str, numpy.ndarray]:
        """Return a state and one row per prompt token, the last predicting the first generated token."""
        self.fed.append(list(prompt))
        stale = numpy.full((len(prompt) - 1, VOCAB), 50.0, dtype=numpy.float32)
        return "state", numpy.concatenate([stale, LOGITS[:1]])[None]

    def forward_batch(self, state: str, ids: list[int]) -> numpy.ndarray:
        """Return the rows predicting the token after each of `ids`."""
        self.fed.append(list(ids))
        return LOGITS[1 : 1 + len(ids)][None]


class HFModel:
    """A transformers Whisper model: one teacher-forced forward over `LOGITS`."""

    def __init__(self) -> None:
        """Record nothing fed yet."""
        self.fed: list[list[int]] = []

    def __call__(self, input_features: object, decoder_input_ids: Any, use_cache: bool) -> Any:  # noqa: ANN401
        """Return logits whose last `len(GEN)` positions are `LOGITS`."""
        torch = importlib.import_module("torch")
        fed = decoder_input_ids[0].tolist()
        self.fed.append(fed)
        stale = numpy.full((len(fed) - len(GEN), VOCAB), 50.0, dtype=numpy.float32)
        logits = torch.from_numpy(numpy.concatenate([stale, LOGITS[: len(fed) - len(PROMPT) + 1]])[None])
        return type("Out", (), {"logits": logits})()


class Engine:
    """A crisperwhisper engine of either backend: window decodes return `DECODED[case]`."""

    all_special_ids = {EOT, 4}
    tokenizer = Tokenizer()
    device = "cpu"

    def __init__(self, kind: str, case: str) -> None:
        """Hold the model of the `kind` backend."""
        self.model: Any = CT2Model() if kind == "ct2" else HFModel()
        self.case = case

    def generate_with_repair(self, features: object, prompt_tokens: list[int], **kwargs: Any) -> list[int]:  # noqa: ANN401
        """Return the window's decode."""
        return list(DECODED[self.case])

    def generate_with_repair_and_attention(
        self,
        features: object,
        prompt_tokens: list[int],
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[list[int], str]:
        """Return the window's decode and its attention."""
        return list(DECODED[self.case]), "attention"

    def cross_attention_for_tokens(self, features: object, prompt_tokens: list[int], gen_ids: list[int]) -> str:
        """Return the attention of `gen_ids`."""
        return "recaptured"


class CrisperWhisper:
    """The library model: one window, its ids replaced by `GEN` after fallback or early-EOT recovery."""

    def __init__(self, kind: str, case: str) -> None:
        """Hold the engine the library would build."""
        self._engine = Engine(kind, case)

    def transcribe(self, path: str, **kwargs: Any) -> Result:  # noqa: ANN401
        """Decode one window, replacing its ids where the library would, and return its words."""
        features = object()
        ids, _ = self._engine.generate_with_repair_and_attention(features, PROMPT, max_length=8)
        if ids != GEN:
            assert self._engine.cross_attention_for_tokens(features, PROMPT, GEN) == "recaptured"
        return Result()


def _transcribe(kind: str, case: str) -> tuple[cw.Decoded, list[list[int]]]:
    library = cw.Library.__new__(cw.Library)
    library.engine, library.model = kind, CrisperWhisper(kind, case)
    return library.transcribe(numpy.zeros(160), "en", 8), library.model._engine.model.fed


def _expected() -> tuple[float, list[float]]:
    shifted = LOGITS - LOGITS.max(axis=1, keepdims=True)
    logprob = shifted[range(len(GEN)), GEN] - numpy.log(numpy.exp(shifted).sum(axis=1))
    return float(numpy.exp(logprob[:3].mean())), [float(numpy.exp(logprob[:2].sum())), float(numpy.exp(logprob[2]))]


@pytest.mark.env
@pytest.mark.parametrize("case", list(DECODED))
@pytest.mark.parametrize("kind", ["ct2", "transformers"])
def test_confidence_is_of_the_final_ids_on_both_engines(kind: str, case: str) -> None:
    """Each engine scores the ids the window finally emits, after fallback or recovery, once, with the same values."""
    pytest.importorskip("torch")
    decoded, fed = _transcribe(kind, case)
    mean, words = _expected()
    assert decoded.mean == pytest.approx(mean, rel=1e-5)
    assert decoded.confidences == pytest.approx(words, rel=1e-5)
    assert fed == ([PROMPT, GEN[:-1]] if kind == "ct2" else [PROMPT + GEN[:-1]])


@pytest.mark.logic
def test_windows_keep_decode_order_and_the_engine_is_left_as_it_was() -> None:
    """A replacement updates its own window in place; the wrappers pass results through and are removed on exit."""
    engine = Engine("ct2", "fallback")
    first, second = object(), object()
    with capture_windows(engine) as windows:
        assert engine.generate_with_repair(first, PROMPT) == DECODED["fallback"]
        assert engine.generate_with_repair_and_attention(second, [4], max_length=8) == ([3, 3, 3, 3, EOT], "attention")
        assert engine.cross_attention_for_tokens(first, PROMPT, GEN) == "recaptured"
    assert windows == [Window(first, PROMPT, GEN), Window(second, [4], DECODED["fallback"])]
    assert not vars(engine).keys() - {"model", "case"}


@pytest.mark.logic
def test_two_modes_on_shared_features_are_two_windows() -> None:
    """Two prompts decoded on one `features` stay two windows; a replacement updates only its own prompt's."""
    engine = Engine("ct2", "fallback")
    shared, verbatim, intended = object(), [4, 0], [4, 1]
    with capture_windows(engine) as windows:
        engine.generate_with_repair(shared, verbatim)
        engine.generate_with_repair(shared, intended)
        engine.cross_attention_for_tokens(shared, intended, GEN)
    assert windows == [Window(shared, verbatim, DECODED["fallback"]), Window(shared, intended, GEN)]


@pytest.mark.env
def test_special_ids_are_scored_but_left_out_of_the_mean() -> None:
    """The end token has a log-probability in the block and no weight in `avg_logprob`."""
    pytest.importorskip("torch")
    engine = Engine("ct2", "clean")
    [block] = window_confidence(engine, [Window("f", PROMPT, GEN)], ct2_teacher_logits, engine.all_special_ids)
    assert block["tokens"] == GEN and len(block["token_logprob"]) == len(GEN)
    assert block["avg_logprob"] == pytest.approx(float(numpy.mean(block["token_logprob"][:3])))
    assert block["avg_logprob"] != pytest.approx(float(numpy.mean(block["token_logprob"])))


@pytest.mark.logic
def test_ct2_teacher_logits_refuse_a_row_count_other_than_the_tokens() -> None:
    """A model that returns too few rows is an error, not a shorter score."""
    engine = Engine("ct2", "clean")
    assert ct2_teacher_logits(engine, Window("f", PROMPT, GEN)).tolist() == LOGITS.tolist()
    with pytest.raises(ValueError, match="logit rows"):
        ct2_teacher_logits(engine, Window("f", PROMPT, [*GEN, 1]))
