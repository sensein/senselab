"""Time anchors, clock maps and spans."""

from fractions import Fraction

import numpy
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from senselab_core.kernel.time import Span, TimeAnchor, TimeMap


@pytest.mark.logic
def test_span_rejects_negative_duration() -> None:
    """A span ending before it starts is refused."""
    with pytest.raises(ValidationError):
        Span(start_ns=10, end_ns=5, clock_id="c")


@pytest.mark.logic
def test_span_allows_zero_duration_and_numpy_integers() -> None:
    """An instant is a span of zero length; numpy integers are accepted as plain ints."""
    s = Span(start_ns=numpy.int64(5), end_ns=5, clock_id="c")
    assert s.start_ns == s.end_ns == 5 and type(s.start_ns) is int


@pytest.mark.logic
@pytest.mark.parametrize("bad", [1.5, 1.0, True, "1", None])
def test_span_times_are_integer_nanoseconds_only(bad: object) -> None:
    """Floats, bools and strings are not nanoseconds."""
    with pytest.raises(ValidationError):
        Span(start_ns=bad, end_ns=10, clock_id="c")


@pytest.mark.logic
def test_span_needs_a_clock() -> None:
    """An empty clock id is refused."""
    with pytest.raises(ValidationError):
        Span(start_ns=0, end_ns=1, clock_id="")


@pytest.mark.logic
def test_span_is_frozen() -> None:
    """Spans are immutable."""
    s = Span(start_ns=0, end_ns=1, clock_id="c")
    with pytest.raises(ValidationError):
        s.end_ns = 2  # type: ignore[misc]


@pytest.mark.logic
def test_timemap_offset_and_drift() -> None:
    """A map adds the offset and the drift accumulated since the source origin."""
    m = TimeMap(source="a", target="b", offset_ns=1_000, drift_ppm=10.0)
    assert m.map(1_000_000_000) == 1_000_000_000 + 1_000 + 10_000


@pytest.mark.logic
def test_timemap_is_exact_over_long_recordings() -> None:
    """Drift over a year of nanoseconds is computed without float rounding."""
    year_ns = 365 * 86_400 * 10**9 + 7
    m = TimeMap(source="a", target="b", offset_ns=-3, drift_ppm=0.5)
    assert m.map(year_ns) == year_ns - 3 + (year_ns * 5 + 5_000_000) // 10_000_000


@pytest.mark.logic
def test_timemap_carries_format_and_refuses_bad_values() -> None:
    """A map is a formatted artifact; non-finite drift and negative uncertainty are refused."""
    assert TimeMap(source="a", target="b", offset_ns=0).format == "timemap@1.0"
    with pytest.raises(ValidationError):
        TimeMap(source="a", target="b", offset_ns=0, drift_ppm=float("nan"))
    with pytest.raises(ValidationError):
        TimeMap(source="a", target="b", offset_ns=0, uncertainty_ns=-1)
    with pytest.raises(ValidationError):
        TimeMap(source="a", target="b", offset_ns=0.5)


@pytest.mark.logic
def test_time_anchor_absolute_start_is_optional() -> None:
    """An anchor without an absolute start ties nothing to wall time."""
    assert TimeAnchor(clock_id="dev", absolute_start_ns=None).absolute_start_ns is None
    assert TimeAnchor(clock_id="dev", absolute_start_ns=1_700_000_000_000_000_000).clock_id == "dev"
    with pytest.raises(ValidationError):
        TimeAnchor(clock_id="dev", absolute_start_ns=1.5)


@pytest.mark.logic
def test_timemap_maps_integer_nanoseconds_only() -> None:
    """Mapping a float or bool time is refused."""
    m = TimeMap(source="a", target="b", offset_ns=0)
    for bad in (1.5, True):
        with pytest.raises(TypeError):
            m.map(bad)  # type: ignore[arg-type]


@pytest.mark.logic
def test_nanoseconds_bounded_to_int64() -> None:
    """Nanoseconds are int64: the extremes validate, one past either refuses."""
    assert Span(start_ns=-(2**63), end_ns=2**63 - 1, clock_id="c").end_ns == 2**63 - 1
    for bad in (2**63, 2**200):
        with pytest.raises(ValidationError):
            Span(start_ns=0, end_ns=bad, clock_id="c")
    with pytest.raises(ValidationError):
        Span(start_ns=-(2**63) - 1, end_ns=0, clock_id="c")
    with pytest.raises(ValidationError):
        TimeAnchor(clock_id="c", absolute_start_ns=2**64)


@pytest.mark.logic
@pytest.mark.parametrize("drift", [-1e6, 1e6, -3e6, 2e6])
def test_a37_drift_cannot_reverse_or_stop_time(drift: float) -> None:
    """A-37: a drift of +/-1e6 ppm or beyond (which stops or reverses time) is refused."""
    with pytest.raises(ValidationError):
        TimeMap(source="a", target="b", offset_ns=0, drift_ppm=drift)


@pytest.mark.logic
@settings(max_examples=300, deadline=None)
@given(
    offset=st.integers(-(2**62), 2**62),
    drift=st.floats(-1e6, 1e6, exclude_min=True, exclude_max=True),
    t=st.integers(-(2**61), 2**61),
    step=st.integers(1, 10**12),
)
def test_timemap_never_reverses_time(offset: int, drift: float, t: int, step: int) -> None:
    """A map with drift inside its bounds never reverses time.

    The exact map is strictly increasing; rounded to integer nanoseconds it is non-decreasing, and
    strictly increasing for any two times whose exact images lie more than 1 ns apart.
    """
    m = TimeMap(source="a", target="b", offset_ns=offset, drift_ppm=drift)
    assert m.map(t) <= m.map(t + step)
    if step * (1 + Fraction(drift) / 1_000_000) > 1:
        assert m.map(t) < m.map(t + step)
