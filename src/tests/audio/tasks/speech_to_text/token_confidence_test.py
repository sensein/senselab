"""Token-confidence extraction tests (T026) — pure math over synthetic logits.

These exercise the real tensors, no mocked models: a hand-built logit sequence has
a known entropy and log-probability, so the assertions pin the arithmetic rather
than an implementation detail. The pipeline-capture seam is covered separately
with a stand-in ``generate`` because the alternative is a Whisper download.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from senselab.audio.tasks.speech_to_text.token_confidence import (
    capture_token_confidence,
    merge_confidence_blocks,
    token_confidence_from_logits,
    whisper_token_ids,
)


def _uniform_logits(vocab: int) -> torch.Tensor:
    """One step of a perfectly flat distribution → entropy = log(vocab)."""
    return torch.zeros(1, vocab)


def _peaked_logits(vocab: int, token_id: int, magnitude: float = 30.0) -> torch.Tensor:
    """One step that is all but deterministic on ``token_id`` → entropy ≈ 0."""
    logits = torch.full((1, vocab), -magnitude)
    logits[0, token_id] = magnitude
    return logits


# ── Entropy ───────────────────────────────────────────────────────────


def test_uniform_distribution_yields_max_entropy() -> None:
    """A flat softmax over V tokens has entropy exactly log(V) nats."""
    vocab = 8
    out = token_confidence_from_logits(logits=[_uniform_logits(vocab)], sequences=torch.tensor([[3]]))
    assert out[0]["token_entropy"] == [pytest.approx(math.log(vocab), abs=1e-5)]


def test_peaked_distribution_yields_near_zero_entropy() -> None:
    """A near-deterministic step carries almost no entropy."""
    out = token_confidence_from_logits(logits=[_peaked_logits(8, 5)], sequences=torch.tensor([[5]]))
    assert out[0]["token_entropy"][0] == pytest.approx(0.0, abs=1e-6)


def test_entropy_reported_per_generated_token() -> None:
    """One entropy per decoding step, in order."""
    logits = [_peaked_logits(8, 1), _uniform_logits(8), _peaked_logits(8, 2)]
    out = token_confidence_from_logits(logits=logits, sequences=torch.tensor([[1, 4, 2]]))
    entropies = out[0]["token_entropy"]
    assert len(entropies) == 3
    assert entropies[0] == pytest.approx(0.0, abs=1e-6)
    assert entropies[1] == pytest.approx(math.log(8), abs=1e-5)
    assert entropies[2] == pytest.approx(0.0, abs=1e-6)


# ── avg_logprob ───────────────────────────────────────────────────────


def test_avg_logprob_of_confident_sequence_near_zero() -> None:
    """Picking the peaked token every step → log p ≈ 0."""
    logits = [_peaked_logits(8, 1), _peaked_logits(8, 2)]
    out = token_confidence_from_logits(logits=logits, sequences=torch.tensor([[1, 2]]))
    assert out[0]["avg_logprob"] == pytest.approx(0.0, abs=1e-6)


def test_avg_logprob_of_uniform_sequence_is_negative_log_vocab() -> None:
    """Under a flat distribution every chosen token has log p = -log(V)."""
    vocab = 4
    logits = [_uniform_logits(vocab), _uniform_logits(vocab)]
    out = token_confidence_from_logits(logits=logits, sequences=torch.tensor([[0, 3]]))
    assert out[0]["avg_logprob"] == pytest.approx(-math.log(vocab), abs=1e-5)


def test_special_tokens_excluded_from_avg_logprob() -> None:
    """Forced/special ids don't dilute the transcript's own confidence."""
    vocab = 4
    # Step 0 is a special token drawn from a flat distribution; step 1 is real
    # and fully confident. Skipping the special leaves avg_logprob ≈ 0.
    logits = [_uniform_logits(vocab), _peaked_logits(vocab, 2)]
    out = token_confidence_from_logits(
        logits=logits,
        sequences=torch.tensor([[0, 2]]),
        special_token_ids={0},
    )
    assert out[0]["avg_logprob"] == pytest.approx(0.0, abs=1e-6)
    # Entropy is still reported for every step — only avg_logprob filters.
    assert len(out[0]["token_entropy"]) == 2


def test_all_special_tokens_yields_none_avg_logprob() -> None:
    """No real tokens → no honest avg_logprob to report."""
    out = token_confidence_from_logits(
        logits=[_uniform_logits(4)],
        sequences=torch.tensor([[0]]),
        special_token_ids={0},
    )
    assert out[0]["avg_logprob"] is None


# ── no_speech_prob ────────────────────────────────────────────────────


def test_no_speech_prob_read_from_first_step() -> None:
    """Matches Whisper's definition: softmax(first-step logits)[<|nospeech|>]."""
    vocab = 4
    first = torch.tensor([[0.0, 0.0, 0.0, 0.0]])  # flat → each prob = 0.25
    out = token_confidence_from_logits(
        logits=[first, _peaked_logits(vocab, 1)],
        sequences=torch.tensor([[2, 1]]),
        no_speech_token_id=3,
    )
    assert out[0]["no_speech_prob"] == pytest.approx(0.25, abs=1e-5)


def test_no_speech_prob_none_without_token_id() -> None:
    """Non-Whisper backends have no <|nospeech|> token → None, not a guess."""
    out = token_confidence_from_logits(logits=[_uniform_logits(4)], sequences=torch.tensor([[1]]))
    assert out[0]["no_speech_prob"] is None


# ── Batch + alignment ─────────────────────────────────────────────────


def test_batch_items_scored_independently() -> None:
    """Each batch row gets its own confidence block."""
    vocab = 8
    step0 = torch.cat([_peaked_logits(vocab, 1), _uniform_logits(vocab)], dim=0)
    out = token_confidence_from_logits(logits=[step0], sequences=torch.tensor([[1], [4]]))
    assert len(out) == 2
    assert out[0]["token_entropy"][0] == pytest.approx(0.0, abs=1e-6)
    assert out[1]["token_entropy"][0] == pytest.approx(math.log(vocab), abs=1e-5)


def test_forced_prefix_tokens_ignored_when_sequences_longer_than_logits() -> None:
    """Encoder-decoder sequences carry a forced prefix; only the tail is scored.

    ``sequences`` here is 4 long but only 2 decoding steps were recorded, so the
    scored tokens must be the final 2 — mirroring HF's transition-score alignment.
    """
    vocab = 8
    logits = [_peaked_logits(vocab, 6), _peaked_logits(vocab, 7)]
    out = token_confidence_from_logits(logits=logits, sequences=torch.tensor([[1, 2, 6, 7]]))
    # If alignment were wrong the chosen ids would be 1,2 (flat → very negative).
    assert out[0]["avg_logprob"] == pytest.approx(0.0, abs=1e-6)


def test_empty_logits_returns_none_fields() -> None:
    """A generate call that produced no scored steps degrades to None."""
    out = token_confidence_from_logits(logits=[], sequences=torch.tensor([[1, 2]]))
    assert out[0]["token_entropy"] is None
    assert out[0]["avg_logprob"] is None


# ── Pipeline capture seam ─────────────────────────────────────────────


class _FakeGenerateOutput(dict):
    """Stands in for a transformers ModelOutput (dict-like with attributes)."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.__dict__.update(kwargs)


class _FakeModel:
    """Minimal model whose ``generate`` returns logits alongside sequences."""

    def __init__(self) -> None:
        self.seen_kwargs: dict[str, object] = {}

    def generate(self, **kwargs: object) -> _FakeGenerateOutput:
        self.seen_kwargs = kwargs
        return _FakeGenerateOutput(
            sequences=torch.tensor([[4]]),
            logits=(_peaked_logits(8, 4),),
        )


class _FakePipe:
    def __init__(self) -> None:
        self.model = _FakeModel()


def test_capture_requests_logits_and_collects_confidence() -> None:
    """The seam turns on logit output and harvests one block per generate call."""
    pipe = _FakePipe()
    with capture_token_confidence(pipe, no_speech_token_id=None) as captured:
        pipe.model.generate(inputs=torch.zeros(1, 4))
    assert pipe.model.seen_kwargs["output_logits"] is True
    assert pipe.model.seen_kwargs["return_dict_in_generate"] is True
    assert len(captured) == 1
    assert captured[0]["avg_logprob"] == pytest.approx(0.0, abs=1e-6)


def test_capture_restores_original_generate() -> None:
    """The wrapper is removed on exit — no lasting mutation of the pipeline.

    Asserted behaviorally (does generate still inject ``output_logits``?) rather
    than by object speaker, since attribute access rebuilds a bound method each
    time and an speaker check could never hold.
    """
    pipe = _FakePipe()
    with capture_token_confidence(pipe, no_speech_token_id=None):
        pipe.model.generate(inputs=torch.zeros(1, 4))
        assert pipe.model.seen_kwargs["output_logits"] is True
    pipe.model.seen_kwargs = {}
    pipe.model.generate(inputs=torch.zeros(1, 4))
    assert "output_logits" not in pipe.model.seen_kwargs


def test_capture_tolerates_backend_without_logits() -> None:
    """A backend that ignores output_logits yields no confidence, not a crash."""

    class _SilentModel:
        """Returns a bare tensor, as a backend that ignores ``output_logits`` would."""

        def __init__(self) -> None:
            self.seen_kwargs: dict[str, object] = {}

        def generate(self, **kwargs: object) -> torch.Tensor:
            self.seen_kwargs = kwargs
            return torch.tensor([[1, 2]])

    pipe = _FakePipe()
    pipe.model = _SilentModel()  # type: ignore[assignment]
    with capture_token_confidence(pipe, no_speech_token_id=None) as captured:
        pipe.model.generate(inputs=torch.zeros(1, 4))
    assert captured == []


# ── Real Whisper shapes (long-form / return_timestamps="word") ────────
#     Verified against transformers 5.5.4 + openai/whisper-tiny: the top-level
#     generate output carries only ('sequences', 'token_timestamps', 'segments'),
#     and each segments[batch][seg]["result"] holds 1-D logits (vocab,) and 1-D
#     sequences (seq_len,) — not the batched 2-D shapes.


def test_one_dimensional_logits_and_sequences_supported() -> None:
    """Long-form per-segment results come back unbatched; treat them as batch of 1."""
    vocab = 8
    out = token_confidence_from_logits(
        logits=[_peaked_logits(vocab, 5).squeeze(0)],  # (vocab,)
        sequences=torch.tensor([5]),  # (seq_len,)
    )
    assert len(out) == 1
    assert out[0]["avg_logprob"] == pytest.approx(0.0, abs=1e-6)
    assert out[0]["token_entropy"][0] == pytest.approx(0.0, abs=1e-6)


def test_one_dimensional_alignment_drops_forced_prefix() -> None:
    """The unbatched path must apply the same trailing-token alignment."""
    vocab = 8
    out = token_confidence_from_logits(
        logits=[_peaked_logits(vocab, 6).squeeze(0), _peaked_logits(vocab, 7).squeeze(0)],
        sequences=torch.tensor([1, 2, 6, 7]),
    )
    assert out[0]["avg_logprob"] == pytest.approx(0.0, abs=1e-6)


def test_merge_confidence_blocks_combines_segments() -> None:
    """Entropies concatenate, log-probs average, no-speech takes the max."""
    merged = merge_confidence_blocks(
        [
            {"token_entropy": [0.0, 1.0], "avg_logprob": -0.2, "no_speech_prob": 0.1},
            {"token_entropy": [2.0], "avg_logprob": -0.4, "no_speech_prob": 0.7},
        ]
    )
    assert merged["token_entropy"] == [0.0, 1.0, 2.0]
    assert merged["avg_logprob"] == pytest.approx(-0.3)
    assert merged["no_speech_prob"] == pytest.approx(0.7)


def test_merge_confidence_blocks_all_none() -> None:
    """Nothing to merge → all-None block rather than zeros."""
    merged = merge_confidence_blocks([{"token_entropy": None, "avg_logprob": None, "no_speech_prob": None}])
    assert merged == {"token_entropy": None, "avg_logprob": None, "no_speech_prob": None}


class _LongFormModel:
    """Mimics Whisper's long-form output: no top-level logits, results under segments."""

    def __init__(self) -> None:
        self.seen_kwargs: dict[str, object] = {}

    def generate(self, **kwargs: object) -> _FakeGenerateOutput:
        self.seen_kwargs = kwargs
        segment = {
            "result": {
                "sequences": torch.tensor([1, 2, 4]),
                "logits": [_peaked_logits(8, 4).squeeze(0)],
            }
        }
        other = {
            "result": {
                "sequences": torch.tensor([1, 2, 6]),
                "logits": [_uniform_logits(8).squeeze(0)],
            }
        }
        return _FakeGenerateOutput(
            sequences=torch.tensor([[1, 2, 4, 6]]),
            token_timestamps=torch.zeros(1, 4),
            segments=[[segment, other]],
        )


def test_capture_extracts_from_longform_segments() -> None:
    """One merged block per batch item, harvested out of the segments nesting."""
    pipe = _FakePipe()
    pipe.model = _LongFormModel()  # type: ignore[assignment]
    with capture_token_confidence(pipe, no_speech_token_id=None) as captured:
        pipe.model.generate(inputs=torch.zeros(1, 4))
    assert len(captured) == 1, "one batch item → one merged confidence block"
    entropies = captured[0]["token_entropy"]
    assert entropies is not None and len(entropies) == 2  # one per segment step
    assert entropies[0] == pytest.approx(0.0, abs=1e-6)  # peaked segment
    assert entropies[1] == pytest.approx(math.log(8), abs=1e-5)  # uniform segment


# ── <|nospeech|> id resolution ────────────────────────────────────────
#     Vocab spelling differs across Whisper releases: openai/whisper-tiny uses
#     "<|nocaptions|>" (50362) and maps "<|nospeech|>" to the *unk* id (50257).
#     Verified directly against the checkpoint's tokenizer.


class _Tokenizer:
    def __init__(self, mapping: dict[str, int], unk_token_id: int | None = 50257) -> None:
        self._mapping = mapping
        self.unk_token_id = unk_token_id
        self.all_special_ids = [1, 2, 3]

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._mapping.get(token, self.unk_token_id)


class _TokenizerPipe:
    def __init__(self, tokenizer: object, generation_config: object = None) -> None:
        self.tokenizer = tokenizer
        self.model = SimpleNamespace(generation_config=generation_config)


def test_nospeech_id_falls_back_to_nocaptions_spelling() -> None:
    """whisper-tiny's real layout: <|nospeech|> is unk, <|nocaptions|> is the token."""
    pipe = _TokenizerPipe(_Tokenizer({"<|nocaptions|>": 50362}))
    no_speech_id, special_ids = whisper_token_ids(pipe)
    assert no_speech_id == 50362
    assert special_ids == {1, 2, 3}


def test_nospeech_id_prefers_canonical_spelling() -> None:
    """When both exist, the canonical <|nospeech|> wins."""
    pipe = _TokenizerPipe(_Tokenizer({"<|nospeech|>": 50361, "<|nocaptions|>": 50362}))
    assert whisper_token_ids(pipe)[0] == 50361


def test_nospeech_id_prefers_generation_config() -> None:
    """An explicit generation_config value beats tokenizer lookup."""
    pipe = _TokenizerPipe(
        _Tokenizer({"<|nocaptions|>": 50362}),
        generation_config=SimpleNamespace(no_speech_token_id=99),
    )
    assert whisper_token_ids(pipe)[0] == 99


def test_nospeech_id_none_when_no_spelling_matches() -> None:
    """A non-Whisper tokenizer yields None rather than the unk id."""
    pipe = _TokenizerPipe(_Tokenizer({}))
    assert whisper_token_ids(pipe)[0] is None


def test_nospeech_id_none_without_tokenizer() -> None:
    """CTC/other backends without a tokenizer degrade to (None, None)."""
    assert whisper_token_ids(SimpleNamespace(tokenizer=None)) == (None, None)


# ── Return-type preservation (regression: CI cpu-tests) ───────────────
#     Injecting return_dict_in_generate=True changes what generate hands back.
#     Pipelines that never asked for a dict then treat a ModelOutput as a tensor
#     ("'ModelOutput' object has no attribute 'dtype'"), so the seam must restore
#     the type the caller would have received.


def test_capture_preserves_tensor_return_type() -> None:
    """Caller didn't request a dict → it must still receive the bare tensor."""
    pipe = _FakePipe()
    with capture_token_confidence(pipe, no_speech_token_id=None) as captured:
        result = pipe.model.generate(inputs=torch.zeros(1, 4))
    assert isinstance(result, torch.Tensor), "pipeline must not be handed a ModelOutput"
    assert result.tolist() == [[4]]
    assert len(captured) == 1, "confidence is still harvested"


def test_capture_preserves_dict_when_caller_requested_it() -> None:
    """Caller asked for dict output (Whisper word timestamps) → keep the dict."""
    pipe = _FakePipe()
    with capture_token_confidence(pipe, no_speech_token_id=None) as captured:
        result = pipe.model.generate(inputs=torch.zeros(1, 4), return_token_timestamps=True)
    assert not isinstance(result, torch.Tensor)
    assert "sequences" in result
    assert len(captured) == 1


def test_capture_leaves_unrecognized_result_untouched() -> None:
    """A stand-in (Mock-like) generate result passes through unchanged."""

    class _OpaqueModel:
        def generate(self, **kwargs: object) -> str:
            return "opaque"

    pipe = _FakePipe()
    pipe.model = _OpaqueModel()  # type: ignore[assignment]
    with capture_token_confidence(pipe, no_speech_token_id=None):
        assert pipe.model.generate(inputs=None) == "opaque"


def test_special_ids_ignored_when_not_iterable() -> None:
    """A Mock tokenizer must not blow up id resolution (CI regression)."""

    class _MockLike:
        unk_token_id = 0
        all_special_ids = object()  # not iterable, as Mock attributes are

        def convert_tokens_to_ids(self, token: str) -> object:
            return object()

    no_speech_id, special_ids = whisper_token_ids(_TokenizerPipe(_MockLike()))
    assert no_speech_id is None
    assert special_ids is None
