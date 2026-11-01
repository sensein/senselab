"""Annotations, embeddings, intervals and events."""

import numpy
import pytest
from pydantic import ValidationError
from senselab_core.kernel.results import (
    Annotation,
    Embedding,
    Events,
    Intervals,
    Metric,
    TextSpans,
    Transcript,
    TranscriptNode,
    Trials,
)
from senselab_core.kernel.signals import PlainText
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


def _word(text: str, start: object, end: object, clock: str = "c", confidence: float | None = None) -> TranscriptNode:
    return TranscriptNode.model_validate(
        {"text": text, "span": {"start_ns": start, "end_ns": end, "clock_id": clock}, "confidence": confidence}
    )


@pytest.mark.logic
def test_transcript_word_spans_are_ns() -> None:
    """Word spans are integer nanoseconds on one clock; confidence is optional and lies in [0, 1]."""
    words = (_word("hello", 0, 400_000_000, confidence=0.9), _word("there", numpy.int64(500_000_000), 900_000_000))
    t = Transcript(segments=(TranscriptNode(text="hello there", children=words),))
    leaf = t.segments[0].children[1]
    assert leaf.span is not None and type(leaf.span.start_ns) is int and leaf.confidence is None
    assert t.format == "transcript@1.0"
    for start in (0.5, "0", True):
        with pytest.raises(ValidationError):
            _word("x", start, 10)
    with pytest.raises(ValidationError):
        _word("x", 0, 10, confidence=1.5)
    with pytest.raises(ValidationError, match="clock"):
        Transcript(segments=(TranscriptNode(text="a b", children=(_word("a", 0, 1), _word("b", 1, 2, "d"))),))
    assert Transcript(segments=(TranscriptNode(text="untimed"),)).segments[0].span is None


@pytest.mark.logic
def test_text_spans_within_text_bounds() -> None:
    """Character spans lie in `[0, length]` of the text they name, and their columns agree in length."""
    text = PlainText(content="call 555 0100")
    spans = TextSpans.over(text, start=[5], end=[13], label=["phone"], score=[0.8])
    assert (spans.text, spans.length, spans.format) == (text.identity, 13, "text-spans@1.0")
    assert text.content[spans.start[0] : spans.end[0]] == "555 0100"
    for start, end in (([5], [14]), ([-1], [3]), ([4], [3])):
        with pytest.raises(ValidationError):
            TextSpans.over(text, start=start, end=end, label=["phone"])
    with pytest.raises(ValidationError, match="length"):
        TextSpans.over(text, start=[0, 1], end=[1], label=["a"])
    none = numpy.array([], int)
    assert TextSpans.over(text, start=none, end=none, label=numpy.array([], str)).score is None


@pytest.mark.logic
def test_trials_pair_embeddings_with_a_label() -> None:
    """Trials pair row i of `first` with row i of `second`, labelled same or different, in one space."""
    a = numpy.array([[1.0, 0.0], [0.0, 1.0]])
    trials = Trials(first=a, second=a[::-1], same=[True, False], space="m")
    assert trials.same.dtype == numpy.bool_ and trials.format == "trials@1.0"
    for bad in ({"second": a[:1]}, {"same": [1, 0]}, {"same": [True]}, {"first": numpy.full((2, 2), numpy.nan)}):
        with pytest.raises(ValidationError):
            Trials.model_validate({"first": a, "second": a, "same": [True, False], "space": "m", **bad})


@pytest.mark.logic
def test_metric_names_value_unit_and_items() -> None:
    """A metric is a finite value with its unit over at least one item."""
    m = Metric(name="wer", value=0.125, unit="1", items=8)
    assert (m.name, m.value, m.unit, m.items, m.format) == ("wer", 0.125, "1", 8, "metric@1.0")
    for bad in ({"value": float("inf")}, {"items": 0}, {"items": 1.0}, {"unit": ""}):
        with pytest.raises(ValidationError):
            Metric.model_validate({"name": "wer", "value": 0.1, "unit": "1", "items": 1, **bad})
