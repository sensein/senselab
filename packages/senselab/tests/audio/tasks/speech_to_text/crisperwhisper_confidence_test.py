"""CrisperWhisper confidence: both engines score the decoded tokens from raw logits under one definition."""

import importlib
from typing import Any

import numpy
import pytest
from senselab.audio.tasks.speech_to_text import crisperwhisper as cw
from senselab.audio.tasks.speech_to_text.token_confidence import (
    capture_ct2_token_confidence,
    ct2_teacher_logits,
)

VOCAB = 6
EOT = 5
PROMPT = [4, 0]
GEN = [1, 2, 3, EOT]
PIECES = {1: " he", 2: "llo", 3: " there", EOT: "<|endoftext|>"}
LOGITS = numpy.random.default_rng(7).normal(size=(len(GEN), VOCAB)).astype(numpy.float32)
LOGITS[3, EOT] = -9.0


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
    """A library result: two words, as `GEN` decodes."""

    words = [Word("hello", 0.0, 0.4), Word("there", 0.4, 0.8)]


class CT2Model:
    """Teacher forcing over `LOGITS`: `prefill` predicts `GEN[0]`, `forward_batch` the rest."""

    def __init__(self) -> None:
        """Record nothing fed yet."""
        self.fed: list[list[int]] = []

    def prefill(self, features: object, prompt: list[int]) -> tuple[str, numpy.ndarray]:
        """Return a state and one row per prompt token, the last predicting `GEN[0]`."""
        self.fed.append(list(prompt))
        stale = numpy.full((len(prompt) - 1, VOCAB), 50.0, dtype=numpy.float32)
        return "state", numpy.concatenate([stale, LOGITS[:1]])[None]

    def forward_batch(self, state: str, ids: list[int]) -> numpy.ndarray:
        """Return the rows predicting the tokens after each of `ids`."""
        self.fed.append(list(ids))
        return LOGITS[1 : 1 + len(ids)][None]


class CT2Engine:
    """The library's ct2 engine: its two window decodes both return `GEN`."""

    all_special_ids = {EOT, 4}
    tokenizer = Tokenizer()

    def __init__(self) -> None:
        """Hold a fresh `CT2Model`."""
        self.model = CT2Model()

    def generate_with_repair(self, features: object, prompt_tokens: list[int], **kwargs: Any) -> list[int]:  # noqa: ANN401
        """Return `GEN`."""
        return list(GEN)

    def generate_with_repair_and_attention(
        self,
        features: object,
        prompt_tokens: list[int],
        **kwargs: Any,  # noqa: ANN401
    ) -> tuple[list[int], str]:
        """Return `GEN` and its attention."""
        return list(GEN), "attention"


class HFModel:
    """A transformers model whose generate returns `GEN` after `PROMPT` with `LOGITS`."""

    def generate(self, **kwargs: Any) -> Any:  # noqa: ANN401
        """Return the dict output the capture asks for."""
        torch = importlib.import_module("torch")
        assert kwargs["output_logits"] and kwargs["return_dict_in_generate"]
        return {
            "sequences": torch.tensor([PROMPT + GEN]),
            "logits": tuple(torch.from_numpy(row[None].copy()) for row in LOGITS),
        }


class TransformersEngine:
    """The library's transformers engine."""

    all_special_ids = {EOT, 4}
    tokenizer = Tokenizer()
    model = HFModel()


class CrisperWhisper:
    """The library model: one window, decoded the way each engine's longform path decodes it."""

    def __init__(self, engine: str) -> None:
        """Hold the `engine` the library would build."""
        self.kind = engine
        self._engine: Any = CT2Engine() if engine == "ct2" else TransformersEngine()

    def transcribe(self, path: str, **kwargs: Any) -> Result:  # noqa: ANN401
        """Decode one window and return its words."""
        if self.kind == "ct2":
            self._engine.generate_with_repair_and_attention("features", PROMPT, max_length=8)
        else:
            self._engine.model.generate(input_features="features")
        return Result()


def _host(view: object) -> numpy.ndarray:
    return numpy.asarray(view)


def _library(engine: str) -> cw.Library:
    library = cw.Library.__new__(cw.Library)
    library.engine, library.host, library.model = engine, _host, CrisperWhisper(engine)
    return library


def _expected() -> tuple[float, list[float]]:
    shifted = LOGITS - LOGITS.max(axis=1, keepdims=True)
    logprob = shifted[range(len(GEN)), GEN] - numpy.log(numpy.exp(shifted).sum(axis=1))
    return float(numpy.exp(logprob[:3].mean())), [float(numpy.exp(logprob[:2].sum())), float(numpy.exp(logprob[2]))]


@pytest.mark.logic
def test_ct2_reports_the_confidence_the_transformers_engine_reports_from_the_same_logits() -> None:
    """Given identical raw logits, ct2 fills the mean and every word confidence with the transformers values."""
    ct2, transformers = (
        _library("ct2").transcribe(numpy.zeros(160), "en", 8),
        _library("transformers").transcribe(numpy.zeros(160), "en", 8),
    )
    mean, words = _expected()
    assert ct2.mean is not None and None not in ct2.confidences
    assert ct2.mean == pytest.approx(transformers.mean) and ct2.mean == pytest.approx(mean, rel=1e-5)
    assert ct2.confidences == pytest.approx(transformers.confidences)
    assert ct2.confidences == pytest.approx(words, rel=1e-5)


@pytest.mark.logic
def test_ct2_scoring_excludes_the_engines_special_ids_and_leaves_the_engine_as_it_was() -> None:
    """The special end token is scored but left out of the mean; the decode returns unchanged, unwrapped on exit."""
    engine = CT2Engine()
    with capture_ct2_token_confidence(engine, special_token_ids=engine.all_special_ids, host=numpy.asarray) as blocks:
        assert engine.generate_with_repair("features", PROMPT) == GEN
    [block] = blocks
    assert block["tokens"] == GEN and len(block["token_logprob"]) == len(GEN)
    assert block["avg_logprob"] == pytest.approx(float(numpy.mean(block["token_logprob"][:3])))
    assert block["avg_logprob"] != pytest.approx(float(numpy.mean(block["token_logprob"])))
    assert engine.model.fed == [PROMPT, GEN[:-1]] and not vars(engine).keys() - {"model"}


@pytest.mark.logic
def test_teacher_logits_refuse_a_row_count_other_than_the_tokens() -> None:
    """A model that returns too few rows is an error, not a shorter score."""
    model = CT2Model()
    assert ct2_teacher_logits(model, "f", PROMPT, GEN, numpy.asarray).tolist() == LOGITS.tolist()
    with pytest.raises(ValueError, match="logit rows"):
        ct2_teacher_logits(model, "f", PROMPT, [*GEN, 1], numpy.asarray)
