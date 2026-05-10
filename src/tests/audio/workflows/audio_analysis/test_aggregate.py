"""Per-axis aggregator tests (T016) — happy path + edge cases for each axis."""

from __future__ import annotations

import math

import pytest

from senselab.audio.workflows.audio_analysis.aggregate import (
    aggregate_identity,
    aggregate_presence,
    aggregate_utterance,
)

# ── Presence axis ─────────────────────────────────────────────────────


def test_presence_uniform_agreement_zero_uncertainty() -> None:
    """All voters fully committed (native_confidence=1 or absent) → p_voice=1 → uncertainty 0."""
    votes = {f"m{i}": {"speaks": True, "native_confidence": None} for i in range(4)}
    assert aggregate_presence(votes) == pytest.approx(0.0, abs=1e-6)


def test_presence_uniform_agreement_with_partial_confidence() -> None:
    """All voters say True but with native_confidence=0.8 → p_voice=0.8 → uncertainty 0.4."""
    votes = {f"m{i}": {"speaks": True, "native_confidence": 0.8} for i in range(4)}
    assert aggregate_presence(votes) == pytest.approx(1.0 - abs(2 * 0.8 - 1), abs=1e-6)


def test_presence_50_50_split_saturates_to_one() -> None:
    """A confidence-weighted 50/50 split gives uncertainty 1.0."""
    votes = {
        "m0": {"speaks": True, "native_confidence": None},
        "m1": {"speaks": False, "native_confidence": None},
    }
    assert aggregate_presence(votes) == pytest.approx(1.0)


def test_presence_three_to_one_split_yields_half() -> None:
    """3 True + 1 False (equal weights) → p_voice=0.75 → 1-|2·0.75-1| = 0.5."""
    votes = {
        "m0": {"speaks": True},
        "m1": {"speaks": True},
        "m2": {"speaks": True},
        "m3": {"speaks": False},
    }
    assert aggregate_presence(votes) == pytest.approx(0.5, abs=1e-6)


def test_presence_uniform_silence_with_high_confidence() -> None:
    """All voters say False with confidence 0.9 → p_voice = 1-0.9 = 0.1 → uncertainty 0.2."""
    votes = {f"m{i}": {"speaks": False, "native_confidence": 0.9} for i in range(3)}
    p = 1 - 0.9
    assert aggregate_presence(votes) == pytest.approx(1.0 - abs(2 * p - 1), abs=1e-6)


def test_presence_uniform_silence_no_native_conf_zero_uncertainty() -> None:
    """All voters say False with no confidence → p_voice = 0 → uncertainty 0."""
    votes = {f"m{i}": {"speaks": False, "native_confidence": None} for i in range(3)}
    assert aggregate_presence(votes) == pytest.approx(0.0, abs=1e-6)


def test_presence_native_confidence_pulls_p_voice() -> None:
    """A YAMNet vote True with conf=0.99 plus a binary False vote → p_voice = (0.99 + 0)/2 = 0.495."""
    votes = {
        "yamnet": {"speaks": True, "native_confidence": 0.99},
        "binary_dissenter": {"speaks": False, "native_confidence": None},
    }
    p = (0.99 + 0.0) / 2
    assert aggregate_presence(votes) == pytest.approx(1.0 - abs(2 * p - 1), abs=1e-6)


def test_presence_single_contributor_uses_native_confidence() -> None:
    """One contributor + native_confidence=0.7, speaks=True → p_voice=0.7 → 1-|0.4|=0.6."""
    votes = {"m0": {"speaks": True, "native_confidence": 0.7}}
    assert aggregate_presence(votes) == pytest.approx(1.0 - abs(2 * 0.7 - 1), abs=1e-6)


def test_presence_no_contributors_returns_none() -> None:
    """Presence no contributors returns none."""
    assert aggregate_presence({}) is None


# ── Identity axis ─────────────────────────────────────────────────────


def test_identity_low_same_label_uncertainty_means_confirmed_speaker() -> None:
    """All ``(diar, emb)`` pairs report low calibrated same-label uncertainty.

    Calibrated uncertainty 0 means the audio cosine distance was at or below the
    same-speaker floor — the diar model's "same speaker" claim is confirmed.
    """
    votes = {
        "pyannote": {"speaker_label": "SPEAKER_00", "cluster_id": "S0", "speaker_changed_from_prev": False},
        "pyannote::ecapa": {"same_label_uncertainty": 0.05, "change_inconsistency_uncertainty": None},
        "sortformer": {"speaker_label": "speaker_0", "cluster_id": "S0", "speaker_changed_from_prev": False},
        "sortformer::ecapa": {"same_label_uncertainty": 0.07, "change_inconsistency_uncertainty": None},
    }
    assert aggregate_identity(votes, raw_vs_enh=None, aggregator="min") == pytest.approx(0.07)


def test_identity_high_same_label_uncertainty_means_audio_refutes_model() -> None:
    """High calibrated same-label uncertainty on any pair drives ``min`` (worst-case)."""
    votes = {
        "pyannote::ecapa": {"same_label_uncertainty": 0.05, "change_inconsistency_uncertainty": None},
        "sortformer::ecapa": {"same_label_uncertainty": 0.85, "change_inconsistency_uncertainty": None},
    }
    assert aggregate_identity(votes, raw_vs_enh=None, aggregator="min") == pytest.approx(0.85)


def test_identity_first_bucket_with_no_prior_drops_signals() -> None:
    """``same_label_uncertainty=None`` (no prior to validate) → dropped from aggregator."""
    votes = {
        "pyannote::ecapa": {"same_label_uncertainty": None, "change_inconsistency_uncertainty": None},
        "sortformer::ecapa": {"same_label_uncertainty": None, "change_inconsistency_uncertainty": None},
    }
    assert aggregate_identity(votes, raw_vs_enh=None, aggregator="min") is None


def test_identity_raw_vs_enh_signal_only_appears_when_provided() -> None:
    """raw_vs_enh None → not a sub-signal; raw_vs_enh True → 1.0 contribution."""
    votes = {
        "pyannote::ecapa": {"same_label_uncertainty": 0.1, "change_inconsistency_uncertainty": None},
    }
    assert aggregate_identity(votes, raw_vs_enh=None, aggregator="min") == pytest.approx(0.1)
    assert aggregate_identity(votes, raw_vs_enh=True, aggregator="min") == pytest.approx(1.0)


def test_identity_change_inconsistency_uncertainty_aggregated() -> None:
    """``change_inconsistency_uncertainty`` is folded alongside same-label uncertainty."""
    votes = {
        "pyannote::ecapa": {"same_label_uncertainty": None, "change_inconsistency_uncertainty": 0.4},
    }
    assert aggregate_identity(votes, raw_vs_enh=None, aggregator="mean") == pytest.approx(0.4)


def test_identity_cross_diar_disagreement_aggregated() -> None:
    """Cross-diar-model label disagreement contributes to the bucket score."""
    votes = {
        "__cross_diar_label_disagreement__": {"value": 0.5, "n_pairs": 2, "n_disagree": 1},
    }
    assert aggregate_identity(votes, raw_vs_enh=None, aggregator="mean") == pytest.approx(0.5)


def test_identity_no_signals_returns_none() -> None:
    """No calibrated uncertainties and no raw_vs_enh → None."""
    votes = {
        "pyannote": {"speaker_label": "SPEAKER_00", "cluster_id": "S0", "speaker_changed_from_prev": False},
    }
    assert aggregate_identity(votes, raw_vs_enh=None, aggregator="min") is None


# ── Utterance axis ────────────────────────────────────────────────────


def test_utterance_identical_phoneme_seqs_zero_distance() -> None:
    """All phoneme sources matching → all pairwise distances = 0 → uncertainty 0."""
    votes = {
        "asr_a": {"text": "hello world", "phoneme_sequence": ["hh", "eh", "l", "ow"], "avg_logprob": None},
        "asr_b": {"text": "hello world", "phoneme_sequence": ["hh", "eh", "l", "ow"], "avg_logprob": None},
        "__pairwise_phoneme_distances__": {
            "pairs": {"asr_a|asr_b": 0.0},
            "n_sources": 2,
            "sources": ["asr_a", "asr_b"],
        },
    }
    assert aggregate_utterance(votes, aggregator="min") == pytest.approx(0.0)


def test_utterance_pairwise_distances_collapse_via_weighted_mean() -> None:
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
    assert aggregate_utterance(votes, aggregator="min") == pytest.approx(expected_mean, abs=1e-6)


def test_utterance_pairwise_weighted_by_per_source_confidence() -> None:
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
    assert aggregate_utterance(votes, aggregator="min") == pytest.approx(expected, abs=1e-6)


def test_utterance_disjoint_phoneme_seqs_high_distance() -> None:
    """Pairs with no phonemes in common → distance ~1.0."""
    votes = {
        "__pairwise_phoneme_distances__": {
            "pairs": {"asr_a|asr_b": 1.0},
            "n_sources": 2,
            "sources": ["asr_a", "asr_b"],
        },
    }
    result = aggregate_utterance(votes, aggregator="min")
    assert result is not None and result >= 0.9


def test_utterance_drops_empty_side_pairs() -> None:
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
    assert aggregate_utterance(votes, aggregator="min") == pytest.approx(0.0)


def test_utterance_only_avg_logprob_drives_when_no_pairs() -> None:
    """Single-ASR bucket with avg_logprob and no pairwise grid → 1 − exp(avg_logprob)."""
    votes = {
        "whisper": {"text": "hello", "phoneme_sequence": ["hh", "eh", "l", "ow"], "avg_logprob": -0.2},
    }
    result = aggregate_utterance(votes, aggregator="min")
    assert result is not None and result == pytest.approx(1 - math.exp(-0.2), abs=1e-6)


def test_utterance_ppg_contributes_via_pairwise() -> None:
    """PPG-vs-ASR distance enters the pairwise grid as ``__ppg__|<asr>``."""
    votes = {
        "whisper": {"text": "hello", "phoneme_sequence": ["hh", "eh", "l", "ow"], "avg_logprob": None},
        "__pairwise_phoneme_distances__": {
            "pairs": {"__ppg__|whisper": 0.5},
            "n_sources": 2,
            "sources": ["__ppg__", "whisper"],
        },
    }
    assert aggregate_utterance(votes, aggregator="min") == pytest.approx(0.5)


def test_utterance_no_signal_returns_none() -> None:
    """No pairwise grid, no avg_logprob → None."""
    votes = {"asr_a": {"text": "", "phoneme_sequence": [], "avg_logprob": None}}
    assert aggregate_utterance(votes, aggregator="min") is None
