"""Per-axis aggregator tests (T016) — happy path + edge cases for each axis."""

from __future__ import annotations

import math
from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.aggregate import (
    aggregate_asr,
    aggregate_speaker,
    aggregate_speech_presence,
)

# ── Presence axis ─────────────────────────────────────────────────────


def test_speech_presence_uniform_agreement_zero_uncertainty() -> None:
    """All voters fully committed (native_confidence=1 or absent) → p_voice=1 → uncertainty 0."""
    votes = {f"m{i}": {"speaks": True, "native_confidence": None} for i in range(4)}
    assert aggregate_speech_presence(votes) == pytest.approx(0.0, abs=1e-6)


def test_speech_presence_uniform_agreement_with_partial_confidence() -> None:
    """All voters say True but with native_confidence=0.8 → p_voice=0.8 → uncertainty 0.4."""
    votes = {f"m{i}": {"speaks": True, "native_confidence": 0.8} for i in range(4)}
    assert aggregate_speech_presence(votes) == pytest.approx(1.0 - abs(2 * 0.8 - 1), abs=1e-6)


def test_speech_presence_50_50_split_saturates_to_one() -> None:
    """A confidence-weighted 50/50 split gives uncertainty 1.0."""
    votes = {
        "m0": {"speaks": True, "native_confidence": None},
        "m1": {"speaks": False, "native_confidence": None},
    }
    assert aggregate_speech_presence(votes) == pytest.approx(1.0)


def test_speech_presence_three_to_one_split_yields_half() -> None:
    """3 True + 1 False (equal weights) → p_voice=0.75 → 1-|2·0.75-1| = 0.5."""
    votes = {
        "m0": {"speaks": True},
        "m1": {"speaks": True},
        "m2": {"speaks": True},
        "m3": {"speaks": False},
    }
    assert aggregate_speech_presence(votes) == pytest.approx(0.5, abs=1e-6)


def test_speech_presence_uniform_silence_with_high_confidence() -> None:
    """All voters say False with confidence 0.9 → p_voice = 1-0.9 = 0.1 → uncertainty 0.2."""
    votes = {f"m{i}": {"speaks": False, "native_confidence": 0.9} for i in range(3)}
    p = 1 - 0.9
    assert aggregate_speech_presence(votes) == pytest.approx(1.0 - abs(2 * p - 1), abs=1e-6)


def test_speech_presence_uniform_silence_no_native_conf_zero_uncertainty() -> None:
    """All voters say False with no confidence → p_voice = 0 → uncertainty 0."""
    votes = {f"m{i}": {"speaks": False, "native_confidence": None} for i in range(3)}
    assert aggregate_speech_presence(votes) == pytest.approx(0.0, abs=1e-6)


def test_speech_presence_native_confidence_pulls_p_voice() -> None:
    """A YAMNet vote True with conf=0.99 plus a binary False vote → p_voice = (0.99 + 0)/2 = 0.495."""
    votes: dict[str, dict[str, Any]] = {
        "yamnet": {"speaks": True, "native_confidence": 0.99},
        "binary_dissenter": {"speaks": False, "native_confidence": None},
    }
    p = (0.99 + 0.0) / 2
    assert aggregate_speech_presence(votes) == pytest.approx(1.0 - abs(2 * p - 1), abs=1e-6)


def test_speech_presence_single_contributor_uses_native_confidence() -> None:
    """One contributor + native_confidence=0.7, speaks=True → p_voice=0.7 → 1-|0.4|=0.6."""
    votes = {"m0": {"speaks": True, "native_confidence": 0.7}}
    assert aggregate_speech_presence(votes) == pytest.approx(1.0 - abs(2 * 0.7 - 1), abs=1e-6)


def test_speech_presence_no_contributors_returns_none() -> None:
    """Presence no contributors returns none."""
    assert aggregate_speech_presence({}) is None


# ── Identity axis ─────────────────────────────────────────────────────


def test_speaker_low_same_label_uncertainty_means_confirmed_speaker() -> None:
    """All ``(diar, emb)`` pairs report low calibrated same-label uncertainty.

    Calibrated uncertainty 0 means the audio cosine distance was at or below the
    same-speaker floor — the diar model's "same speaker" claim is confirmed.
    """
    votes: dict[str, dict[str, Any]] = {
        "pyannote": {"speaker_label": "SPEAKER_00", "cluster_id": "S0", "speaker_changed_from_prev": False},
        "pyannote::ecapa": {"same_label_uncertainty": 0.05, "change_inconsistency_uncertainty": None},
        "sortformer": {"speaker_label": "speaker_0", "cluster_id": "S0", "speaker_changed_from_prev": False},
        "sortformer::ecapa": {"same_label_uncertainty": 0.07, "change_inconsistency_uncertainty": None},
    }
    assert aggregate_speaker(votes, raw_vs_enh=None, aggregator="min") == pytest.approx(0.07)


def test_speaker_high_same_label_uncertainty_means_audio_refutes_model() -> None:
    """High calibrated same-label uncertainty on any pair drives ``min`` (worst-case)."""
    votes = {
        "pyannote::ecapa": {"same_label_uncertainty": 0.05, "change_inconsistency_uncertainty": None},
        "sortformer::ecapa": {"same_label_uncertainty": 0.85, "change_inconsistency_uncertainty": None},
    }
    assert aggregate_speaker(votes, raw_vs_enh=None, aggregator="min") == pytest.approx(0.85)


def test_speaker_first_bucket_with_no_prior_drops_signals() -> None:
    """``same_label_uncertainty=None`` (no prior to validate) → dropped from aggregator."""
    votes = {
        "pyannote::ecapa": {"same_label_uncertainty": None, "change_inconsistency_uncertainty": None},
        "sortformer::ecapa": {"same_label_uncertainty": None, "change_inconsistency_uncertainty": None},
    }
    assert aggregate_speaker(votes, raw_vs_enh=None, aggregator="min") is None


def test_speaker_raw_vs_enh_signal_only_appears_when_provided() -> None:
    """raw_vs_enh None → not a sub-signal; raw_vs_enh True → 1.0 contribution."""
    votes = {
        "pyannote::ecapa": {"same_label_uncertainty": 0.1, "change_inconsistency_uncertainty": None},
    }
    assert aggregate_speaker(votes, raw_vs_enh=None, aggregator="min") == pytest.approx(0.1)
    assert aggregate_speaker(votes, raw_vs_enh=True, aggregator="min") == pytest.approx(1.0)


def test_speaker_change_inconsistency_uncertainty_aggregated() -> None:
    """``change_inconsistency_uncertainty`` is folded alongside same-label uncertainty."""
    votes = {
        "pyannote::ecapa": {"same_label_uncertainty": None, "change_inconsistency_uncertainty": 0.4},
    }
    assert aggregate_speaker(votes, raw_vs_enh=None, aggregator="mean") == pytest.approx(0.4)


def test_speaker_cross_diar_disagreement_aggregated() -> None:
    """Cross-diar-model label disagreement contributes to the bucket score."""
    votes = {
        "__cross_diar_label_disagreement__": {"value": 0.5, "n_pairs": 2, "n_disagree": 1},
    }
    assert aggregate_speaker(votes, raw_vs_enh=None, aggregator="mean") == pytest.approx(0.5)


def test_speaker_no_signals_returns_none() -> None:
    """No calibrated uncertainties and no raw_vs_enh → None."""
    votes = {
        "pyannote": {"speaker_label": "SPEAKER_00", "cluster_id": "S0", "speaker_changed_from_prev": False},
    }
    assert aggregate_speaker(votes, raw_vs_enh=None, aggregator="min") is None


# ── Utterance axis ────────────────────────────────────────────────────


def test_asr_identical_phoneme_seqs_zero_distance() -> None:
    """All phoneme sources matching → all pairwise distances = 0 → uncertainty 0."""
    votes: dict[str, dict[str, Any]] = {
        "asr_a": {"text": "hello world", "phoneme_sequence": ["hh", "eh", "l", "ow"], "avg_logprob": None},
        "asr_b": {"text": "hello world", "phoneme_sequence": ["hh", "eh", "l", "ow"], "avg_logprob": None},
        "__pairwise_phoneme_distances__": {
            "pairs": {"asr_a|asr_b": 0.0},
            "n_sources": 2,
            "sources": ["asr_a", "asr_b"],
        },
    }
    assert aggregate_asr(votes, aggregator="min") == pytest.approx(0.0)


def test_asr_pairwise_distances_collapse_via_weighted_mean() -> None:
    """Pairwise distances are folded into a single weighted-mean sub-signal.

    With no per-source confidences, every source gets neutral weight 0.5,
    so each pair gets joint weight 0.25 and the weighted mean degenerates
    to the unweighted mean across pairs.
    """
    votes = {
        "__pairwise_phoneme_distances__": {
            "pairs": {
                "asr_a|asr_b": 0.1,
                "asr_a|asr_c": 0.5,
                "asr_b|asr_c": 0.3,
            },
            "n_sources": 3,
            "sources": ["asr_a", "asr_b", "asr_c"],
        },
    }
    expected_mean = (0.1 + 0.5 + 0.3) / 3
    assert aggregate_asr(votes, aggregator="min") == pytest.approx(expected_mean, abs=1e-6)


def test_asr_pairwise_weighted_by_per_source_confidence() -> None:
    """Per-source confidences weight pair contributions: low-confidence sources count less."""
    votes = {
        "__pairwise_phoneme_distances__": {
            "pairs": {
                "asr_a|asr_b": 0.1,
                "asr_a|asr_c": 0.5,
                "asr_b|asr_c": 0.5,
            },
            "n_sources": 3,
            "sources": ["asr_a", "asr_b", "asr_c"],
            "per_source_confidence": {
                "asr_a": 0.9,
                "asr_b": 0.9,
                "asr_c": 0.1,  # low-confidence source — its pairs should weigh less
            },
        },
    }
    # weights: a-b → 0.81, a-c → 0.09, b-c → 0.09. Total weight = 0.99.
    # weighted sum = 0.81·0.1 + 0.09·0.5 + 0.09·0.5 = 0.171
    # mean = 0.171 / 0.99 ≈ 0.173
    expected = (0.81 * 0.1 + 0.09 * 0.5 + 0.09 * 0.5) / (0.81 + 0.09 + 0.09)
    assert aggregate_asr(votes, aggregator="min") == pytest.approx(expected, abs=1e-6)


def test_asr_disjoint_phoneme_seqs_high_distance() -> None:
    """Pairs with no phonemes in common → distance ~1.0."""
    votes = {
        "__pairwise_phoneme_distances__": {
            "pairs": {"asr_a|asr_b": 1.0},
            "n_sources": 2,
            "sources": ["asr_a", "asr_b"],
        },
    }
    result = aggregate_asr(votes, aggregator="min")
    assert result is not None and result >= 0.9


def test_asr_drops_empty_side_pairs() -> None:
    """A source with no phonemes is excluded from the pairwise grid by the harvester.

    The aggregator only sees pairs that survived the harvester's drop, so it
    just averages over present pairs.
    """
    votes = {
        "__pairwise_phoneme_distances__": {
            "pairs": {"whisper|granite": 0.0},  # qwen3 (empty) absent from grid
            "n_sources": 2,
            "sources": ["whisper", "granite"],
        },
    }
    assert aggregate_asr(votes, aggregator="min") == pytest.approx(0.0)


def test_asr_only_avg_logprob_drives_when_no_pairs() -> None:
    """Single-ASR bucket with avg_logprob and no pairwise grid → 1 − exp(avg_logprob)."""
    votes = {
        "whisper": {"text": "hello", "phoneme_sequence": ["hh", "eh", "l", "ow"], "avg_logprob": -0.2},
    }
    result = aggregate_asr(votes, aggregator="min")
    assert result is not None and result == pytest.approx(1 - math.exp(-0.2), abs=1e-6)


def test_asr_ppg_contributes_via_pairwise() -> None:
    """PPG-vs-ASR distance enters the pairwise grid as ``__ppg__|<asr>``."""
    votes: dict[str, dict[str, Any]] = {
        "whisper": {"text": "hello", "phoneme_sequence": ["hh", "eh", "l", "ow"], "avg_logprob": None},
        "__pairwise_phoneme_distances__": {
            "pairs": {"__ppg__|whisper": 0.5},
            "n_sources": 2,
            "sources": ["__ppg__", "whisper"],
        },
    }
    assert aggregate_asr(votes, aggregator="min") == pytest.approx(0.5)


def test_asr_no_signal_returns_none() -> None:
    """No pairwise grid, no avg_logprob → None."""
    votes: dict[str, dict[str, Any]] = {"asr_a": {"text": "", "phoneme_sequence": [], "avg_logprob": None}}
    assert aggregate_asr(votes, aggregator="min") is None


# ── Utterance: token entropy, calibration, coupling (T027) ────────────
#     Feature 20260722-175022 (FR-017 / FR-018 / FR-019).


def _pairs(distance: float) -> dict[str, Any]:
    """A single-pair phoneme-distance block, the dominant asr sub-signal."""
    return {"__pairwise_phoneme_distances__": {"pairs": {"a|b": distance}, "per_source_confidence": {}}}


def test_asr_token_entropy_adds_sub_signal() -> None:
    """A high token entropy raises asr uncertainty under the mean aggregator."""
    without = aggregate_asr({**_pairs(0.0)}, aggregator="mean")
    with_entropy = aggregate_asr(
        {**_pairs(0.0), "m": {"text": "hi", "token_entropy": 3.0}},
        aggregator="mean",
    )
    assert without is not None and with_entropy is not None
    assert with_entropy > without


def test_asr_token_entropy_none_falls_back_exactly() -> None:
    """token_entropy=None reproduces today's value bit-for-bit (SC-008)."""
    baseline = aggregate_asr(
        {**_pairs(0.4), "m": {"text": "hi", "avg_logprob": -0.2}},
        aggregator="mean",
    )
    with_none = aggregate_asr(
        {**_pairs(0.4), "m": {"text": "hi", "avg_logprob": -0.2, "token_entropy": None}},
        aggregator="mean",
    )
    assert with_none == baseline


def test_asr_zero_token_entropy_is_full_confidence() -> None:
    """Zero entropy contributes a 0.0 uncertainty sub-signal, not None."""
    result = aggregate_asr({"m": {"text": "hi", "token_entropy": 0.0}}, aggregator="mean")
    assert result == pytest.approx(0.0, abs=1e-9)


def test_asr_token_entropy_saturates_at_reference() -> None:
    """Entropy at/above the reference scale saturates the sub-signal at 1.0."""
    huge = aggregate_asr({"m": {"text": "hi", "token_entropy": 500.0}}, aggregator="mean")
    assert huge == pytest.approx(1.0, abs=1e-9)


def test_asr_token_entropy_averaged_across_models() -> None:
    """Per-model entropies are averaged before normalization."""
    both = aggregate_asr(
        {"a": {"token_entropy": 0.0, "text": "x"}, "b": {"token_entropy": 3.0, "text": "x"}},
        aggregator="mean",
    )
    single = aggregate_asr({"a": {"token_entropy": 1.5, "text": "x"}}, aggregator="mean")
    assert both is not None and single is not None  # guard against a trivial None == None pass
    assert both == pytest.approx(single, abs=1e-9)


def test_asr_token_entropy_accepts_per_token_list() -> None:
    """A per-token list is collapsed to its mean."""
    as_list = aggregate_asr({"a": {"token_entropy": [0.0, 3.0], "text": "x"}}, aggregator="mean")
    as_mean = aggregate_asr({"a": {"token_entropy": 1.5, "text": "x"}}, aggregator="mean")
    assert as_list is not None and as_mean is not None  # guard against a trivial None == None pass
    assert as_list == pytest.approx(as_mean, abs=1e-9)


# ── Calibration (FR-018) ──────────────────────────────────────────────


def test_default_temperature_preserves_legacy_avg_logprob_mapping() -> None:
    """T=1 (default) keeps the historical 1 - exp(avg_logprob) exactly (SC-008)."""
    votes: dict[str, Any] = {"m": {"text": "hi", "avg_logprob": -0.5}}
    assert aggregate_asr(votes, aggregator="mean") == pytest.approx(1.0 - math.exp(-0.5), abs=1e-9)


def test_temperature_above_one_softens_reported_confidence() -> None:
    """A higher temperature flattens confidence → lower reported uncertainty."""
    votes: dict[str, Any] = {"m": {"text": "hi", "avg_logprob": -1.0}}
    sharp = aggregate_asr(votes, aggregator="mean")
    soft = aggregate_asr(votes, aggregator="mean", calibration={"temperature": {"asr": 2.0}})
    assert sharp is not None and soft is not None
    # exp(-1/2) > exp(-1) → higher confidence → lower uncertainty.
    assert soft < sharp
    assert soft == pytest.approx(1.0 - math.exp(-0.5), abs=1e-9)
