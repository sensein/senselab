"""Amplitude-invariance probe: verdict derivation and the regression guard (T012/T013/T016).

The probe answers one research question — does either scene classifier normalize input
level as part of its own inference? Measurement says **no, for both**, which is why
detection cannot rely on gain and why an explicit noise floor is needed instead.

The pure derivation functions are tested here against synthetic per-gain results, so the
logic is verified without touching a model. The regression guard at the bottom is the part
that keeps the finding true: a model or dependency upgrade that changes level handling
should fail CI rather than silently altering background categorization (FR-017b).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from senselab.audio.tasks.classification.level_probe import (
    GAIN_RANGE_MIN_DB,
    AmplitudeInvarianceVerdict,
    detect_floor_signature,
    label_stability,
    max_score_delta,
    top_k_labels,
    validate_gain_range,
    verdict_from_sweep,
)


def _win(labels: list[str], scores: list[float], start: float = 0.0, end: float = 1.0) -> dict[str, Any]:
    """One classification window in the shape ``classify_audios`` emits."""
    return {"start": start, "end": end, "labels": labels, "scores": scores}


# ── top-k extraction ──────────────────────────────────────────────────


def test_top_k_labels_preserves_descending_order() -> None:
    """Senselab pre-sorts labels/scores descending, so index order is rank order."""
    w = _win(["Speech", "Conversation", "Music"], [0.9, 0.05, 0.01])
    assert top_k_labels(w, 2) == ("Speech", "Conversation")


def test_top_k_labels_tolerates_fewer_labels_than_k() -> None:
    """A short window may carry fewer than k labels; that is not an error."""
    assert top_k_labels(_win(["Speech"], [0.9]), 5) == ("Speech",)


def test_top_k_labels_of_malformed_window_is_empty() -> None:
    """A non-dict or empty window contributes nothing rather than raising."""
    assert top_k_labels({}, 3) == ()
    assert top_k_labels(None, 3) == ()


# ── label stability (FR-013) ──────────────────────────────────────────


def test_identical_results_are_perfectly_stable() -> None:
    """An amplitude-invariant classifier would score 1.0 at every gain."""
    ref = [_win(["Speech", "Music"], [0.9, 0.1])]
    assert label_stability(ref, ref, k=2) == pytest.approx(1.0)


def test_reordered_top_k_is_unstable() -> None:
    """Label identity migrating with level is the failure this measures."""
    ref = [_win(["Speech", "Music"], [0.9, 0.1])]
    cand = [_win(["Music", "Speech"], [0.9, 0.1])]
    assert label_stability(ref, cand, k=2) < 1.0


def test_disjoint_top_k_is_zero_stability() -> None:
    """No shared top-k label means no stability at all."""
    ref = [_win(["Speech"], [0.9])]
    cand = [_win(["Waterfall"], [0.9])]
    assert label_stability(ref, cand, k=1) == pytest.approx(0.0)


def test_stability_averages_across_windows() -> None:
    """One window matching and one not gives 0.5, not a pass or a fail."""
    ref = [_win(["Speech"], [0.9]), _win(["Music"], [0.9])]
    cand = [_win(["Speech"], [0.9]), _win(["Waterfall"], [0.9])]
    assert label_stability(ref, cand, k=1) == pytest.approx(0.5)


def test_stability_of_empty_input_is_none() -> None:
    """No windows means no measurement — not a stability of zero."""
    assert label_stability([], [], k=5) is None


def test_stability_compares_only_overlapping_windows() -> None:
    """Windowing can differ in count at the tail; compare what both produced."""
    ref = [_win(["Speech"], [0.9]), _win(["Speech"], [0.9])]
    cand = [_win(["Speech"], [0.9])]
    assert label_stability(ref, cand, k=1) == pytest.approx(1.0)


# ── score delta ───────────────────────────────────────────────────────


def test_max_score_delta_is_zero_for_identical_results() -> None:
    """Identical runs differ by nothing."""
    ref = [_win(["Speech", "Music"], [0.9, 0.1])]
    assert max_score_delta(ref, ref) == pytest.approx(0.0)


def test_max_score_delta_tracks_the_largest_per_label_change() -> None:
    """Compared per label, so a reordering does not masquerade as a small change."""
    ref = [_win(["Speech", "Music"], [0.9, 0.1])]
    cand = [_win(["Speech", "Music"], [0.5, 0.2])]
    assert max_score_delta(ref, cand) == pytest.approx(0.4)


def test_max_score_delta_counts_a_dropped_label_as_its_full_score() -> None:
    """A label vanishing from the top-k is a change of its whole score, not zero."""
    ref = [_win(["Speech", "Music"], [0.9, 0.3])]
    cand = [_win(["Speech"], [0.9])]
    assert max_score_delta(ref, cand) == pytest.approx(0.3)


# ── floor signature (FR-020d) ─────────────────────────────────────────


def test_floor_signature_detected_from_repeated_response() -> None:
    """A classifier saturating to a fixed pattern below its floor is detectable."""
    silence_like = [_win(["Silence", "Music"], [0.437, 0.350])] * 3
    sig = detect_floor_signature(silence_like)
    assert sig is not None
    assert sig["Silence"] == pytest.approx(0.437, abs=1e-3)
    assert sig["Music"] == pytest.approx(0.350, abs=1e-3)


def test_floor_signature_is_none_when_response_varies() -> None:
    """Real content produces varying output; only a fixed response is a floor response."""
    varied = [_win(["Speech"], [0.9]), _win(["Music"], [0.4]), _win(["Bird"], [0.2])]
    assert detect_floor_signature(varied) is None


def test_floor_signature_does_not_require_silence_to_dominate() -> None:
    """A co-occurring label can clear a threshold the silence label does not.

    `Silence` peaks at 0.437 while `Music` reaches 0.350, so keying a floor check on the
    silence score alone would miss this response entirely.
    """
    sig = detect_floor_signature([_win(["Silence", "Music"], [0.437, 0.350])] * 4)
    assert sig is not None and max(sig, key=lambda k: sig[k]) == "Silence"
    assert sig["Music"] > 0.3, "the co-occurring label is what a naive silence gate misses"


def test_floor_signature_of_empty_input_is_none() -> None:
    """No windows is no evidence of saturation."""
    assert detect_floor_signature([]) is None


# ── gain range (SC-005) ───────────────────────────────────────────────


def test_gain_range_below_minimum_rejected() -> None:
    """A verdict is only meaningful across a wide enough range to see the floor."""
    with pytest.raises(ValueError, match="30"):
        validate_gain_range([-5.0, 0.0, 5.0])


def test_gain_range_at_minimum_accepted() -> None:
    """The minimum span itself is sufficient."""
    lo, hi = validate_gain_range([-20.0, 0.0, 10.0])
    assert hi - lo == pytest.approx(GAIN_RANGE_MIN_DB)


def test_gain_range_requires_unity() -> None:
    """Stability is measured *against* unity gain, so it has to be probed."""
    with pytest.raises(ValueError, match="unity|0 dB"):
        validate_gain_range([-40.0, -20.0, -10.0])


def test_gain_range_needs_at_least_two_points() -> None:
    """One point cannot show a trend."""
    with pytest.raises(ValueError, match="at least"):
        validate_gain_range([0.0])


# ── verdict assembly (FR-014, FR-015) ─────────────────────────────────


def _sweep_level_sensitive() -> dict[float, list[dict[str, Any]]]:
    return {
        -40.0: [_win(["Silence"], [1.0])],
        -20.0: [_win(["Speech"], [0.4])],
        0.0: [_win(["Speech"], [0.99])],
        10.0: [_win(["Speech synthesizer", "Speech"], [0.49, 0.46])],
    }


def _sweep_invariant() -> dict[float, list[dict[str, Any]]]:
    ref = [_win(["Speech", "Conversation"], [0.9, 0.05])]
    return {-40.0: ref, -20.0: ref, 0.0: ref, 10.0: ref}


def test_verdict_is_level_sensitive_when_labels_move() -> None:
    """Any movement in labels or scores means the classifier is level-sensitive."""
    v = verdict_from_sweep("yamnet", window_length_s=0.96, per_gain=_sweep_level_sensitive())
    assert v.verdict == "level_sensitive"


def test_verdict_is_self_normalizing_only_when_nothing_moves() -> None:
    """The bar is high on purpose: this verdict would overturn a measured finding."""
    v = verdict_from_sweep("hypothetical", window_length_s=1.0, per_gain=_sweep_invariant())
    assert v.verdict == "self_normalizing"


def test_verdict_records_window_length_for_attribution() -> None:
    """A verdict belongs to one classifier at one window length (FR-015)."""
    v = verdict_from_sweep("ast", window_length_s=10.24, per_gain=_sweep_level_sensitive())
    assert v.classifier == "ast"
    assert v.window_length_s == pytest.approx(10.24)


def test_verdict_records_the_gain_range_it_holds_over() -> None:
    """A verdict is only claimed over the range actually probed."""
    v = verdict_from_sweep("yamnet", window_length_s=0.96, per_gain=_sweep_level_sensitive())
    assert v.gain_range_db == (-40.0, 10.0)


def test_verdict_low_level_floor_is_the_quietest_collapsing_gain() -> None:
    """The floor is where the classifier stops reporting content, not where it wobbles."""
    v = verdict_from_sweep("yamnet", window_length_s=0.96, per_gain=_sweep_level_sensitive())
    assert v.low_level_floor_db is not None
    assert v.low_level_floor_db <= -20.0


def test_verdict_rejects_a_sweep_without_unity() -> None:
    """Stability is measured against unity, so a sweep omitting it is unusable."""
    sweep = {k: v for k, v in _sweep_level_sensitive().items() if k != 0.0}
    with pytest.raises(ValueError, match="unity|0 dB"):
        verdict_from_sweep("yamnet", window_length_s=0.96, per_gain=sweep)


def test_verdict_serializes_to_the_contract_shape() -> None:
    """contracts/level-verdicts.md — every field a consumer reads must be present."""
    v = verdict_from_sweep(
        "ast",
        window_length_s=10.24,
        per_gain=_sweep_level_sensitive(),
        floor_mechanism="fixed dataset-level affine normalization; log(float32 eps) floor",
        mechanism_source="transformers/.../feature_extraction_...py:75-77,113,156",
    )
    doc = v.to_json()
    for key in (
        "classifier",
        "window_length_s",
        "verdict",
        "gain_range_db",
        "label_stability",
        "score_delta_max",
        "low_level_floor_db",
        "floor_mechanism",
        "mechanism_source",
    ):
        assert key in doc, f"missing {key}"


def test_mechanism_source_is_required_for_a_published_verdict() -> None:
    """An empirical verdict must be corroborated against code (FR-016)."""
    v = AmplitudeInvarianceVerdict(
        classifier="ast",
        window_length_s=10.24,
        verdict="level_sensitive",
        gain_range_db=(-40.0, 10.0),
        label_stability={},
        score_delta_max={},
    )
    with pytest.raises(ValueError, match="mechanism_source"):
        v.require_corroboration()


# ── regression guard (FR-017b, T016) ──────────────────────────────────
#
# Pinned from the measured audit. These are findings, not preferences: if either
# classifier starts reporting `self_normalizing`, the probe is wrong or the model
# changed — both warrant investigation rather than a quiet threshold edit.

BASELINE_PATH = Path(__file__).parent / "data" / "level_verdicts_baseline.json"


def _baseline() -> dict[str, Any]:
    return json.loads(BASELINE_PATH.read_text())


def test_baseline_exists_and_covers_both_classifiers() -> None:
    """The guard is only as good as the recording it defends."""
    classifiers = _baseline()["classifiers"]
    assert len(classifiers) == 2
    assert any("ast" in name for name in classifiers)
    assert "yamnet" in classifiers


def test_both_classifiers_measured_level_sensitive() -> None:
    """Neither self-normalizes. The expected asymmetry did not hold.

    Both were expected to differ — the long-window model was thought unlikely to
    self-amplify and the short-window one a plausible candidate. Measurement says both are
    level-sensitive, and they fail *differently*: the short-window model collapses to a
    silence verdict at reduced gain, while the long-window one holds its top label further
    down but churns its whole top-k list at every non-unity gain.
    """
    verdicts = {name: c["verdict"] for name, c in _baseline()["classifiers"].items()}
    assert set(verdicts.values()) == {"level_sensitive"}, verdicts


def test_unity_gain_is_perfectly_stable_against_itself() -> None:
    """Sanity check on the measurement, not on the model."""
    for name, c in _baseline()["classifiers"].items():
        assert c["unity_stability"] == pytest.approx(1.0), name


def test_short_window_classifier_has_a_collapse_floor() -> None:
    """Its floor is the most reliable level diagnostic either classifier exposes.

    Monotone and source-independent, so it doubles as a tripwire (FR-042).
    """
    yamnet = _baseline()["classifiers"]["yamnet"]
    assert yamnet["low_level_floor_db"] is not None
    assert yamnet["has_floor_signature"] is True


def test_long_window_classifier_moves_substantially_with_gain() -> None:
    """Even without a collapse floor, its scores are not level-comparable (FR-020e)."""
    ast = next(c for name, c in _baseline()["classifiers"].items() if "ast" in name)
    assert ast["max_delta_at_min_gain"] > 0.3


def test_baseline_records_the_score_function_it_was_measured_under() -> None:
    """A floor response measured under one output transform does not describe another.

    The transform changed with the FR-017c fix, and the long-window model's silence
    response changed with it — so a baseline without this field would be uninterpretable.
    """
    from senselab.audio.workflows.audio_analysis.sound_sources import AUDIOSET_SCORE_FUNCTION

    assert _baseline()["score_function"] == AUDIOSET_SCORE_FUNCTION
