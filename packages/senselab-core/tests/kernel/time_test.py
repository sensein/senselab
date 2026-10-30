"""Time anchors, clock maps and spans."""

import numpy
import pytest
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
