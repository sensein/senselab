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
exact_count_accuracy = _derive.exact_count_accuracy


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
