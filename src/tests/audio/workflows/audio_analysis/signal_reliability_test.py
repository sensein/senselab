"""Per-signal reliability, measured by perturbation, entering the aggregation.

A sub-signal's own uncertainty is evidence about how much its vote should count. Without
it, aggregation treats every signal as equally trustworthy — and under max-doubt a single
unreliable signal decides the axis outright, which is how a saturated embedding check came
to outvote unanimous diarizer agreement on a real recording.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.aggregators import apply_aggregator
from senselab.audio.workflows.audio_analysis.reliability import (
    reliability_from_stability,
    signal_stability,
)

# ── weighting inside the aggregator ───────────────────────────────────


def test_an_unreliable_signal_cannot_decide_the_axis_alone() -> None:
    """The observed failure: one saturated signal against unanimous agreement.

    Its doubt is attenuated toward the corroborated answer rather than taken at face value.
    """
    values: list[float | None] = [1.0, 0.0, 0.0]
    unweighted = apply_aggregator(values, "min")
    weighted = apply_aggregator(values, "min", weights=[0.05, 1.0, 1.0])
    assert unweighted == pytest.approx(1.0)
    assert weighted is not None and weighted < 0.2


def test_a_reliable_signal_keeps_its_full_objection() -> None:
    """Down-weighting must not silence a signal that has earned its vote."""
    assert apply_aggregator([1.0, 0.0], "min", weights=[1.0, 1.0]) == pytest.approx(1.0)


def test_an_unreliable_signal_is_attenuated_not_erased() -> None:
    """A lone dissenter stays visible at the lowest weight the reliability model can assign.

    Zeroing it would delete the only evidence that anything objected — the same mistake as
    letting it dominate, in the other direction. The floor lives in the reliability model
    rather than here: the aggregator honours whatever weight it is handed, so a caller that
    genuinely wants a signal excluded can say so.
    """
    from senselab.audio.workflows.audio_analysis.reliability import MIN_RELIABILITY

    weighted = apply_aggregator([1.0, 0.0], "min", weights=[MIN_RELIABILITY, 1.0])
    assert weighted is not None and weighted > 0.0


def test_an_explicit_zero_weight_excludes_a_signal() -> None:
    """The aggregator does not second-guess its caller; the floor is applied upstream."""
    assert apply_aggregator([1.0, 0.0], "min", weights=[0.0, 1.0]) == pytest.approx(0.0)


def test_weights_are_optional() -> None:
    """Unweighted aggregation must be unchanged, so existing outputs stay reproducible."""
    values: list[float | None] = [0.2, 0.8, 0.5]
    for name in ("min", "mean", "harmonic_mean", "disagreement_weighted"):
        assert apply_aggregator(values, name) == apply_aggregator(values, name, weights=None)


def test_mismatched_weights_are_refused() -> None:
    """Silently recycling or truncating weights would misattribute reliability."""
    with pytest.raises(ValueError, match="weights"):
        apply_aggregator([0.1, 0.2, 0.3], "min", weights=[1.0, 1.0])


def test_a_dropped_signal_drops_its_weight_with_it() -> None:
    """Weights follow their sub-signals out.

    ``None`` sub-signals are removed before aggregation; their weights must go too,
    or every later weight is applied to the wrong signal.
    """
    assert apply_aggregator([None, 1.0], "min", weights=[1.0, 0.05]) == apply_aggregator([1.0], "min", weights=[0.05])


# ── reliability measured from perturbation ────────────────────────────


def _harvest(pass_label: str, values: dict[str, float]) -> object:
    """A minimal harvest carrying one identity bucket with the given sub-signal values."""
    from types import SimpleNamespace

    votes = {k: {"same_label_uncertainty": v} for k, v in values.items()}
    return SimpleNamespace(
        pass_label=pass_label,
        identity_votes=[{"start": 0.0, "end": 0.5, "votes": votes}],
        presence_votes=[],
        utterance_votes=[],
    )


def test_a_signal_that_answers_the_same_under_perturbation_is_reliable() -> None:
    """Agreement with itself under perturbation is what earns weight.

    Raw and enhanced are the same recording under a transform, so a signal's two answers
    are already a stability sample — the same argument the speaker-count posterior uses.
    """
    stab = signal_stability(
        {"raw_16k": _harvest("raw_16k", {"a": 0.2}), "enhanced_16k": _harvest("enhanced_16k", {"a": 0.2})},
        axis="identity",
    )
    assert reliability_from_stability(stab)["a"] == pytest.approx(1.0)


def test_a_signal_that_flips_under_perturbation_is_not() -> None:
    """A signal that contradicts itself on the same audio has not earned its weight."""
    stab = signal_stability(
        {"raw_16k": _harvest("raw_16k", {"a": 0.0}), "enhanced_16k": _harvest("enhanced_16k", {"a": 1.0})},
        axis="identity",
    )
    assert reliability_from_stability(stab)["a"] < 0.1


def test_reliability_never_reaches_zero() -> None:
    """Mirrors the influence gate's floor.

    With few perturbation points the measure is
    coarse, and a hard zero would erase a claim rather than down-weight it.
    """
    stab = signal_stability(
        {"raw_16k": _harvest("raw_16k", {"a": 0.0}), "enhanced_16k": _harvest("enhanced_16k", {"a": 1.0})},
        axis="identity",
    )
    assert reliability_from_stability(stab)["a"] > 0.0


def test_a_single_pass_yields_no_reliability_claim() -> None:
    """One observation is not a stability sample.

    Reporting 1.0 would assert reliability
    that was never measured; the signal keeps full weight by default instead.
    """
    stab = signal_stability({"raw_16k": _harvest("raw_16k", {"a": 0.3})}, axis="identity")
    assert stab == {}


def test_signals_are_scored_independently() -> None:
    """One unstable signal must not drag down a steady one measured alongside it."""
    stab = signal_stability(
        {
            "raw_16k": _harvest("raw_16k", {"steady": 0.2, "flipper": 0.0}),
            "enhanced_16k": _harvest("enhanced_16k", {"steady": 0.2, "flipper": 1.0}),
        },
        axis="identity",
    )
    rel = reliability_from_stability(stab)
    assert rel["steady"] > rel["flipper"]


# The declared derivation gate's tests lived here. The gate is gone: a source kind written
# into policy encodes a judgement from whichever recording motivated it, and that judgement
# was wrong about the very model it named on a second recording. Its replacement is measured
# — perturbation stability x physical support — and is tested in ``signal_support_test.py``.
