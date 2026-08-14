"""The rule that turns a measured accuracy curve into a declared ceiling."""

import importlib.util
import sys
from pathlib import Path

import pytest

# `scripts/` is deliberately not an importable package -- pyproject sets
# `pythonpath = ["src"]`, so the repo root is not on sys.path and
# `from scripts.speaker_ceiling.derive import ...` raises ModuleNotFoundError. Load by file
# location instead, which is the convention analyze_audio_test.py already uses for the same reason.
_DERIVE = Path(__file__).resolve().parents[3] / "scripts" / "speaker_ceiling" / "derive.py"
_spec = importlib.util.spec_from_file_location("speaker_ceiling_derive_under_test", _DERIVE)
assert _spec is not None and _spec.loader is not None, f"could not load {_DERIVE}"
_derive = importlib.util.module_from_spec(_spec)
sys.modules["speaker_ceiling_derive_under_test"] = _derive
_spec.loader.exec_module(_derive)

DEFAULT_ACCURACY_THRESHOLD = _derive.DEFAULT_ACCURACY_THRESHOLD
derive_ceiling = _derive.derive_ceiling
derive_structural_bound = _derive.derive_structural_bound
exact_count_accuracy = _derive.exact_count_accuracy
format_structural_bound_evidence = _derive.format_structural_bound_evidence


def test_accuracy_counts_only_exact_matches() -> None:
    """Reporting 3 speakers when there are 4 is wrong, not partially right.

    A near-miss metric would let a backend that systematically undercounts look
    capable at high speaker counts, which is the exact error this probe exists to
    detect.
    """
    assert exact_count_accuracy([4, 4, 3, 4], true_k=4) == pytest.approx(0.75)


def test_a_refusal_counts_against_accuracy_but_is_not_a_wrong_answer() -> None:
    """None means the backend refused or crashed on that session.

    It cannot count as correct, but the caller still needs to distinguish it from
    a wrong number when reading the confusion — so it is preserved as None in the
    predictions and simply fails the exact-match test here.
    """
    assert exact_count_accuracy([2, None, 2, 2], true_k=2) == pytest.approx(0.75)


def test_empty_predictions_are_zero_not_a_crash() -> None:
    """A cell with no completed sessions scores zero, and the caller refuses on it."""
    assert exact_count_accuracy([], true_k=1) == 0.0


def test_ceiling_is_the_last_k_before_the_first_failure() -> None:
    """A backend good to 4 and poor at 5 has a ceiling of 4."""
    curve = {1: 1.0, 2: 1.0, 3: 0.95, 4: 0.85, 5: 0.30, 6: 0.10}
    assert derive_ceiling(curve) == 4


def test_a_later_recovery_does_not_raise_the_ceiling() -> None:
    """A ceiling a backend intermittently exceeds is not a ceiling.

    Scoring well at k=6 after failing k=4 means the k=6 successes are not
    dependable, so the honest ceiling is still 3.
    """
    curve = {1: 1.0, 2: 1.0, 3: 0.9, 4: 0.20, 5: 0.10, 6: 0.95}
    assert derive_ceiling(curve) == 3


def test_failing_the_smallest_count_yields_none() -> None:
    """A backend that cannot even do k=1 has no measurable ceiling.

    None here means 'the probe established nothing', which is the same meaning
    None already carries in DiarizationCapabilities.max_speakers.
    """
    assert derive_ceiling({1: 0.4, 2: 0.9, 3: 0.9}) is None


def test_the_default_threshold_is_the_documented_judgement() -> None:
    """The 80% is a judgement, not a measurement, so it is pinned here too.

    The profile records it beside the curve it was applied to precisely so a reader
    can disagree with it; a silent change to this constant would invalidate every
    ceiling already derived under it.
    """
    assert DEFAULT_ACCURACY_THRESHOLD == 0.8


def test_the_threshold_boundary_is_inclusive() -> None:
    """Exactly at the threshold counts as meeting it.

    Chosen so a curve that scores precisely 0.8 -- 16 of 20 sessions, the sample size
    this probe uses -- is not rejected by a floating-point hair.
    """
    assert derive_ceiling({1: 0.8, 2: 0.8, 3: 0.79}) == 2


def test_a_gap_in_the_curve_stops_the_ceiling_there() -> None:
    """A missing k cannot be assumed to have passed.

    The probe refuses on incomplete cells rather than emitting one, so a gap reaching
    this function at all means something upstream let it through; treating the gap as
    a pass would silently overstate the ceiling.
    """
    assert derive_ceiling({1: 1.0, 2: 1.0, 4: 1.0}) == 2


# --- derive_structural_bound / format_structural_bound_evidence -----------------------------
#
# These answer "how large a count can the backend emit at all", independent of
# derive_ceiling's "how large a count can it count correctly". The seed-17 probe's own
# k=8 confusions are used directly as fixtures here (see the task-4 brief): Sortformer
# 20x "4", child-adult 20x "2", and the other four backends spread across multiple
# values with no single accumulation point.


def test_a_backend_that_collapses_to_one_value_below_true_k_has_a_structural_bound() -> None:
    """Sortformer's measured k=8 confusion: every session predicted exactly 4.

    A checkpoint literally named `diar_sortformer_4spk` claimed 4; this is what
    confirms it structurally rather than by name alone.
    """
    assert derive_structural_bound({"4": 20}, true_k=8) == 4


def test_a_backend_confined_to_two_talkers_has_a_structural_bound_of_two() -> None:
    """The child-adult classifier's measured k=8 confusion: every session counted 2."""
    assert derive_structural_bound({"2": 20}, true_k=8) == 2


def test_a_spread_of_predicted_counts_is_not_a_structural_bound() -> None:
    """Pyannote and DiariZen's measured k=8 confusion: predictions spanned 5..8.

    Still trying to track the true count, however inaccurately -- not the same failure
    mode as a hard cap, and must not be reported as one.
    """
    assert derive_structural_bound({"5": 4, "6": 5, "7": 6, "8": 5}, true_k=8) is None


def test_overshooting_the_true_count_is_not_a_structural_bound() -> None:
    """MOSS and VibeVoice's measured k=8 confusions ranged well past the true count (12, 16).

    A backend that predicts *more* than what is present is not exhibiting a ceiling at
    all -- there is no plateau, just an unreliable count.
    """
    assert derive_structural_bound({"6": 2, "9": 10, "12": 8}, true_k=8) is None


def test_a_uniform_correct_score_is_not_evidence_of_a_ceiling() -> None:
    """Every session nailing the true count is an accuracy result, not a structural cap.

    A backend that gets k=8 perfectly right has given no evidence it would fail at k=9;
    reporting `max_speakers=8` here would fabricate a ceiling from a success.
    """
    assert derive_structural_bound({"8": 20}, true_k=8) is None


def test_a_uniform_overcount_at_the_boundary_is_not_a_ceiling_either() -> None:
    """A single value at or above true_k is excluded the same way a perfect score is.

    `bound >= true_k` covers both: nothing in a uniform 9-out-of-8 says the backend
    could not also reach 10.
    """
    assert derive_structural_bound({"9": 20}, true_k=8) is None


def test_a_cell_with_only_refusals_has_no_structural_bound() -> None:
    """A universal refusal says nothing about what the backend can emit, not that it emits 0."""
    assert derive_structural_bound({"refused": 20}, true_k=8) is None


def test_refusals_do_not_prevent_detecting_a_bound_among_completed_sessions() -> None:
    """A few refusals alongside a consistent completed count still show the plateau."""
    assert derive_structural_bound({"4": 15, "refused": 5}, true_k=8) == 4


def test_saturation_evidence_matches_the_probes_own_wording() -> None:
    """The exact string the task-4 brief gives for Sortformer's confirmed ceiling."""
    evidence = format_structural_bound_evidence({"4": 20}, true_k=8, probe_label="probe seed-17")
    assert evidence == "measured: saturates at 4 on 20/20 k=8 sessions (probe seed-17)"


def test_no_saturation_evidence_reports_the_highest_observed_count() -> None:
    """The exact string the task-4 brief gives for Pyannote's unbounded result."""
    evidence = format_structural_bound_evidence({"5": 4, "6": 5, "7": 6, "8": 5}, true_k=8, probe_label="probe seed-17")
    assert evidence == "measured: no saturation, emits up to 8 (probe seed-17)"


def test_no_saturation_evidence_reports_overshoot_past_true_k() -> None:
    """MOSS overshot to 12 at k=8; the evidence string must say 12, not 8."""
    evidence = format_structural_bound_evidence({"6": 2, "9": 10, "12": 8}, true_k=8, probe_label="probe seed-17")
    assert evidence == "measured: no saturation, emits up to 12 (probe seed-17)"


def test_all_refusals_evidence_says_so_rather_than_inventing_a_maximum() -> None:
    """Nothing was observed to report a maximum of, so the string must not claim one."""
    evidence = format_structural_bound_evidence({"refused": 20}, true_k=8, probe_label="probe seed-17")
    assert evidence == "measured: no completed sessions at k=8 (probe seed-17)"
