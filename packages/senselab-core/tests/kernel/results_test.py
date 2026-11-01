"""Annotations, embeddings, intervals and events."""

import numpy
import pytest
from pydantic import ValidationError
from senselab_core.kernel.results import Annotation, Embedding, Events, Intervals
from senselab_core.kernel.time import Span

SPAN = Span(start_ns=0, end_ns=1_000, clock_id="c")


@pytest.mark.logic
def test_annotation_carries_producer_and_guideline() -> None:
    """An annotation names who produced it, under which guideline, with optional uncertainty."""
    a = Annotation(span=SPAN, label="speech", value=None, score=0.9, producer="human:rater", guideline="vad@1.2")
    assert (a.producer, a.guideline, a.uncertainty, a.format) == ("human:rater", "vad@1.2", None, "annotation@1.0")
    b = Annotation(span=SPAN, label=None, value=120.0, score=None, producer="model:x", uncertainty=3.5)
    assert b.uncertainty == 3.5 and b.guideline is None


@pytest.mark.logic
def test_annotation_refuses_bad_values() -> None:
    """A label or a value is required, the producer is named, and numbers are finite."""
    with pytest.raises(ValidationError):
        Annotation(span=SPAN, label=None, value=None, score=None, producer="m")
    with pytest.raises(ValidationError):
        Annotation(span=SPAN, label="x", value=None, score=None, producer="")
    for field in ("value", "score", "uncertainty"):
        with pytest.raises(ValidationError):
            Annotation.model_validate(
                {"span": SPAN, "label": "x", "score": None, "value": None, "producer": "m"} | {field: float("nan")}
            )
    with pytest.raises(ValidationError):
        Annotation(span=SPAN, label="x", value=None, score=None, producer="m", uncertainty=-1.0)


@pytest.mark.logic
def test_annotation_round_trips_through_json() -> None:
    """The JSON form reads back to an equal annotation."""
    a = Annotation(span=SPAN, label="speech", value=None, score=0.5, producer="agent:y", guideline="g@1.0")
    assert Annotation.model_validate_json(a.model_dump_json()) == a


@pytest.mark.logic
def test_embedding_is_a_finite_vector_in_a_named_space() -> None:
    """An embedding is a non-empty 1-D finite vector with its space."""
    e = Embedding(vector=numpy.ones(4, dtype=numpy.float32), space="model@abc")
    assert e.format == "embedding@1.0" and not e.vector.flags.writeable
    for bad in (numpy.ones((2, 2)), numpy.array([]), numpy.array([1.0, numpy.nan]), numpy.array(["a"])):
        with pytest.raises(ValidationError):
            Embedding(vector=bad, space="s")
    with pytest.raises(ValidationError):
        Embedding(vector=numpy.ones(2), space="")


@pytest.mark.logic
def test_intervals_are_columnar_spans() -> None:
    """Columns have equal length, integer times and no negative durations."""
    iv = Intervals(
        start_ns=numpy.array([0, 10]), end_ns=numpy.array([5, 10]), label=numpy.array(["a", "b"]), clock_id="c"
    )
    assert iv.format == "intervals@1.0" and iv.start_ns.dtype == numpy.int64
    for start, end, label in (
        ([0, 10], [5, 9], ["a", "b"]),
        ([0, 10], [5], ["a", "b"]),
        ([0, 10], [5, 11], ["a"]),
        ([0.0, 1.0], [5, 11], ["a", "b"]),
    ):
        with pytest.raises(ValidationError):
            Intervals(start_ns=numpy.array(start), end_ns=numpy.array(end), label=numpy.array(label), clock_id="c")


@pytest.mark.logic
def test_events_are_columnar_instants() -> None:
    """Events have one time and label per row and an optional value column of the same length."""
    ev = Events(time_ns=numpy.array([3, 1]), label=numpy.array(["x", "y"]), value=None, clock_id="c")
    assert ev.format == "events@1.0" and ev.value is None
    assert (
        Events(time_ns=numpy.array([1]), label=numpy.array(["x"]), value=numpy.array([2.0]), clock_id="c").value
        is not None
    )
    for value in (numpy.array([1.0]), numpy.array([[1.0, 2.0]])):
        with pytest.raises(ValidationError):
            Events(time_ns=numpy.array([1, 2]), label=numpy.array(["x", "y"]), value=value, clock_id="c")
    with pytest.raises(ValidationError):
        Events(time_ns=numpy.array([1, 2]), label=numpy.array(["x"]), value=None, clock_id="c")
