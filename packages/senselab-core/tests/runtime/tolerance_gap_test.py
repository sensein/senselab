"""The tolerance judge on non-finite and large finite differences."""

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError
from senselab_core.runtime.tolerance import (
    AxisSpread,
    Coverage,
    Derivation,
    FieldTolerance,
    ToleranceProfile,
    attained_coverage,
    numeric_gap,
    required_items,
    within,
)


def _profile(tolerance: float) -> ToleranceProfile:
    field = FieldTolerance(
        path="x", kind="number", spreads=(AxisSpread(axis="host", spread=tolerance, pairs=1),), tolerance=tolerance
    )
    derivation = Derivation(shas=(), jobs=(), split="calibration", items=("a",), contexts=("c",))
    return ToleranceProfile(task="t", backend="b", derivation=derivation, fields=(field,), findings=())


@pytest.mark.logic
@pytest.mark.parametrize(
    ("a", "b", "gap"),
    [
        ([math.inf, 1.0], [math.inf, 1.0], 0.0),
        ([-math.inf], [-math.inf], 0.0),
        ([math.inf], [-math.inf], math.inf),
        ([math.inf], [1.0], math.inf),
        ([math.nan], [math.nan], math.inf),
        ([math.nan], [0.0], math.inf),
    ],
)
def test_non_finite_gap(a: list[float], b: list[float], gap: float) -> None:
    """An equal infinity differs by 0; every other non-finite pair differs by `inf`, never NaN."""
    assert numeric_gap(a, b) == gap


@pytest.mark.logic
def test_matched_infinity_does_not_hide_a_difference() -> None:
    """A matched infinity before a finite difference of 98 still fails a tolerance of 0.1."""
    assert within({"x": [math.inf, 1.0]}, {"x": [math.inf, 99.0]}, _profile(0.1)) == {
        "x": "difference 98 exceeds tolerance 0.1"
    }
    assert within({"x": [math.inf, 1.0]}, {"x": [math.inf, 1.05]}, _profile(0.1)) == {}


@pytest.mark.logic
@given(
    st.lists(st.one_of(st.floats(allow_nan=True), st.none()), max_size=5),
    st.floats(-1e6, 1e6),
    st.floats(1.0, 1e6),
)
def test_a_large_finite_difference_always_fails(prefix: list[float | None], x: float, delta: float) -> None:
    """Whatever matched values precede it, a finite difference above the tolerance is a disagreement."""
    a, b = [*prefix, x], [*prefix, x + delta]
    assert "x" in within({"x": a}, {"x": b}, _profile(0.5))


@pytest.mark.logic
def test_coverage_is_stated_from_the_item_count() -> None:
    """The packaged 0.95/0.95 bound needs 59 items; 12 items state 0.779 coverage, and a misstated one is refused."""
    assert required_items(0.95, 0.95) == 59 and required_items(0.99, 0.95) == 299
    assert attained_coverage(59, 0.95) >= 0.95 > attained_coverage(58, 0.95)
    thin = Coverage.of(12)
    assert (thin.target, thin.confidence, thin.required_items) == (0.95, 0.95, 59)
    assert round(thin.attained, 3) == 0.779
    items = tuple(f"i{n}" for n in range(12))
    derivation = Derivation(shas=(), jobs=(), split="calibration", items=items, contexts=("c",), coverage=thin)
    assert derivation.coverage == thin
    with pytest.raises(ValidationError, match="do not state"):
        Derivation(shas=(), jobs=(), split="calibration", items=items[:3], contexts=("c",), coverage=thin)
    with pytest.raises(ValidationError, match="needs 59"):
        Coverage(target=0.95, confidence=0.95, required_items=12, attained=thin.attained)
