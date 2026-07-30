"""Mutual-influence guards (T073-T079, FR-011b to FR-011h).

Signals influence one another iteratively toward convergence. That is more powerful than
one-directional reporting and it has failure modes one-directional flow does not, so the
guards land before any influence path is enabled.

The one that matters most is **self-confirmation**: if a signal is revised and its
uncertainty is then recomputed from the revised value, uncertainty falls *because* it was
overwritten, not because evidence arrived. A loop that cannot tell those apart converges on
its own edits and reports high confidence in them. :class:`ResolutionKind` exists to make
the distinction structural rather than a matter of care.

Two more, both cheap to get wrong:

- A **derived** signal agreeing with its parent is not corroboration — it is the same
  computation counted twice.
- **Oscillation** between two interpretations must terminate and say so, rather than
  emitting whichever state the last round happened to produce.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.adaptive.influence import (
    InfluenceWeight,
    effective_weight,
    resolve_influence,
)
from senselab.audio.workflows.audio_analysis.adaptive.provenance import (
    RESOLUTION_KINDS,
    RevisionRecord,
    classify_resolution,
    revision_log_entry,
)

GATES = {"independent": 1.0, "derived": 0.4}


# ── uncertainty gating (T073, FR-011b) ────────────────────────────────


def test_certain_signal_keeps_its_full_weight() -> None:
    """Zero uncertainty means no attenuation."""
    assert effective_weight(1.0, uncertainty=0.0, derivation_gate=1.0) == pytest.approx(1.0)


def test_uncertain_signal_is_attenuated() -> None:
    """A signal that does not trust itself may not move others as far."""
    assert effective_weight(1.0, uncertainty=0.75, derivation_gate=1.0) == pytest.approx(0.25)


def test_fully_uncertain_signal_is_floored_not_erased() -> None:
    """A dissenting claim stays visible while being unable to win.

    A hard zero looked right until a real run produced it: with two perturbation points
    normalised entropy is binary, so a source that disagreed with itself across raw and
    enhanced audio carried *exactly* zero weight and vanished from the posterior. The
    floor keeps the claim readable — its support is still recorded — without letting it
    influence the outcome.
    """
    weight = effective_weight(1.0, uncertainty=1.0, derivation_gate=1.0)
    assert 0.0 < weight <= 0.05


def test_the_floor_can_be_disabled_for_a_hard_zero() -> None:
    """Callers with many perturbation points can opt into full silencing."""
    assert effective_weight(1.0, uncertainty=1.0, derivation_gate=1.0, min_gate=0.0) == pytest.approx(0.0)


def test_a_floored_source_still_loses_to_a_confident_one() -> None:
    """The floor must not resurrect a claim into contention."""
    floored = effective_weight(1.0, uncertainty=1.0, derivation_gate=1.0)
    confident = effective_weight(1.0, uncertainty=0.1, derivation_gate=1.0)
    assert confident > 10 * floored


def test_weight_is_monotone_in_uncertainty() -> None:
    """More uncertainty never buys more influence."""
    weights = [effective_weight(1.0, uncertainty=u, derivation_gate=1.0) for u in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert weights == sorted(weights, reverse=True)


def test_exponent_sharpens_the_gate() -> None:
    """A higher exponent punishes uncertainty harder, without changing the endpoints."""
    soft = effective_weight(1.0, uncertainty=0.5, derivation_gate=1.0, exponent=1.0)
    hard = effective_weight(1.0, uncertainty=0.5, derivation_gate=1.0, exponent=3.0)
    assert hard < soft


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_uncertainty_outside_unit_range_rejected(bad: float) -> None:
    """An out-of-range value would silently produce a bad weight."""
    with pytest.raises(ValueError, match="uncertainty"):
        effective_weight(1.0, uncertainty=bad, derivation_gate=1.0)


# ── derivation gating (T074, FR-011c / SC-030) ────────────────────────


def test_derived_signal_is_gated_below_independent() -> None:
    """Agreement with a parent is not corroboration."""
    ind = resolve_influence("pyannote", base_weight=1.0, uncertainty=0.0, kind="independent", gates=GATES)
    der = resolve_influence("embedding_silhouette", base_weight=1.0, uncertainty=0.0, kind="derived", gates=GATES)
    assert der.effective_weight < ind.effective_weight


def test_derived_gate_must_be_below_independent() -> None:
    """A configuration that equalizes them defeats the guard, so it is rejected."""
    with pytest.raises(ValueError, match="derived"):
        resolve_influence(
            "x", base_weight=1.0, uncertainty=0.0, kind="derived", gates={"independent": 1.0, "derived": 1.0}
        )


def test_unknown_source_kind_rejected() -> None:
    """Every source declares whether it observes independently (FR-007)."""
    with pytest.raises(ValueError, match="kind"):
        resolve_influence("x", base_weight=1.0, uncertainty=0.0, kind="peer", gates=GATES)


def test_derived_signal_cannot_outweigh_a_certain_independent_one() -> None:
    """SC-030: a derived signal alone cannot drive a revision an independent one contradicts.

    Even at perfect self-confidence the derived signal stays below an independent signal
    that is merely reasonably confident.
    """
    der = resolve_influence("embedding_silhouette", base_weight=1.0, uncertainty=0.0, kind="derived", gates=GATES)
    ind = resolve_influence("pyannote", base_weight=1.0, uncertainty=0.5, kind="independent", gates=GATES)
    assert der.effective_weight < ind.effective_weight


def test_influence_weight_records_both_gates_separately() -> None:
    """So an audit can see *why* a weight was small."""
    w = resolve_influence("s", base_weight=1.0, uncertainty=0.5, kind="derived", gates=GATES)
    assert isinstance(w, InfluenceWeight)
    assert w.uncertainty_gate == pytest.approx(0.5)
    assert w.derivation_gate == pytest.approx(0.4)
    assert w.effective_weight == pytest.approx(0.2)


def test_influence_weight_serializes() -> None:
    """The contract shape a consumer reads."""
    doc = resolve_influence("s", base_weight=1.0, uncertainty=0.25, kind="independent", gates=GATES).to_json()
    for key in ("signal", "base_weight", "uncertainty_gate", "derivation_gate", "effective_weight"):
        assert key in doc


# ── self-confirmation guard (T075, FR-011d / SC-027) ──────────────────


def test_uncertainty_drop_after_a_revision_is_not_improvement() -> None:
    """The decisive guard. A value overwritten and then re-measured proves nothing."""
    kind = classify_resolution(
        before_uncertainty=0.8, after_uncertainty=0.2, was_revised=True, independent_evidence=False
    )
    assert kind == "revision"


def test_uncertainty_drop_from_new_evidence_is_improvement() -> None:
    """A genuine learning event is reported as one."""
    kind = classify_resolution(
        before_uncertainty=0.8, after_uncertainty=0.2, was_revised=False, independent_evidence=True
    )
    assert kind == "new_evidence"


def test_revision_with_independent_support_counts_as_new_evidence() -> None:
    """A revision *corroborated* from outside is a genuine improvement."""
    kind = classify_resolution(
        before_uncertainty=0.8, after_uncertainty=0.2, was_revised=True, independent_evidence=True
    )
    assert kind == "new_evidence"


def test_no_drop_is_unresolved() -> None:
    """Uncertainty that did not fall resolved nothing."""
    kind = classify_resolution(
        before_uncertainty=0.8, after_uncertainty=0.8, was_revised=False, independent_evidence=False
    )
    assert kind == "unresolved"


def test_uncertainty_rising_is_unresolved_not_improvement() -> None:
    """A rise is never an improvement."""
    kind = classify_resolution(
        before_uncertainty=0.2, after_uncertainty=0.7, was_revised=False, independent_evidence=True
    )
    assert kind == "unresolved"


def test_resolution_kinds_are_the_declared_three() -> None:
    """No fourth kind may appear."""
    assert set(RESOLUTION_KINDS) == {"new_evidence", "revision", "unresolved"}


# ── revision attribution (T077, FR-011g / SC-026) ─────────────────────


def _record(**kwargs: object) -> RevisionRecord:
    base = {
        "round": 2,
        "quantity": "speakers.S0.span",
        "before": [0.08, 1.60],
        "after": [0.08, 4.84],
        "caused_by": "identity_repair",
        "effective_weight": 0.42,
        "resolution_kind": "new_evidence",
        "evidence": {"change_point_prominence": 0.71},
    }
    base.update(kwargs)
    return RevisionRecord(**base)  # type: ignore[arg-type]


def test_revision_records_cause_round_and_evidence() -> None:
    """Every field an audit needs to retrace a change."""
    doc = _record().to_json()
    for key in ("round", "quantity", "before", "after", "caused_by", "effective_weight", "resolution_kind", "evidence"):
        assert key in doc, f"missing {key}"


def test_revision_without_a_cause_rejected() -> None:
    """An unattributed state change is exactly what FR-011g forbids."""
    with pytest.raises(ValueError, match="caused_by"):
        _record(caused_by="")


def test_revision_with_an_unknown_resolution_kind_rejected() -> None:
    """An unrecognized kind would bypass both downstream checks."""
    with pytest.raises(ValueError, match="resolution_kind"):
        _record(resolution_kind="improved")


def test_revision_log_entry_is_ordered_and_stable() -> None:
    """Deterministic key order, so byte-identical output survives serialization."""
    a = revision_log_entry(_record())
    b = revision_log_entry(_record())
    assert list(a) == list(b)


def test_revision_flagged_as_revision_is_not_reported_as_confidence_gain() -> None:
    """End-to-end shape of the guard: the flag travels with the record."""
    rec = _record(resolution_kind="revision")
    assert rec.to_json()["resolution_kind"] == "revision"
    assert rec.improves_confidence() is False


def test_new_evidence_record_does_report_a_confidence_gain() -> None:
    """The positive case of the guard."""
    assert _record(resolution_kind="new_evidence").improves_confidence() is True


# ── determinism (T078, FR-011f / SC-029) ──────────────────────────────


def test_influence_resolution_is_deterministic() -> None:
    """Same inputs, same weight — no dict-ordering or float-accumulation drift."""
    kwargs = {"base_weight": 0.7, "uncertainty": 0.33, "kind": "derived", "gates": GATES}
    first = resolve_influence("s", **kwargs).effective_weight  # type: ignore[arg-type]
    for _ in range(50):
        assert resolve_influence("s", **kwargs).effective_weight == first  # type: ignore[arg-type]


def test_signal_ordering_is_stable_regardless_of_input_order() -> None:
    """Iteration order must not depend on dict insertion order (FR-011f)."""
    from senselab.audio.workflows.audio_analysis.adaptive.influence import ordered_signals

    assert ordered_signals({"b": 1, "a": 2, "c": 3}) == ordered_signals({"c": 3, "a": 2, "b": 1})


def test_ordered_signals_is_sorted() -> None:
    """A fixed order, independent of how the mapping was built."""
    from senselab.audio.workflows.audio_analysis.adaptive.influence import ordered_signals

    assert ordered_signals({"z": 1, "a": 1, "m": 1}) == ["a", "m", "z"]
