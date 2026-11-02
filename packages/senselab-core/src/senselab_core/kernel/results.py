"""Result types: annotations, embeddings, intervals, events, transcripts, text spans, trials and metrics."""

from collections.abc import Iterator
from typing import Annotated

import numpy
from pydantic import BeforeValidator, Field, Strict, model_validator

from senselab_core.kernel._base import (
    FiniteFloat,
    Frozen,
    LabelArray,
    NonEmptyStr,
    NsArray,
    as_int64_array,
    as_label_array,
    as_numeric_array,
    require_finite,
)
from senselab_core.kernel.formats import FormatTag, Formatted
from senselab_core.kernel.identity import TextIdentity
from senselab_core.kernel.signals import Text
from senselab_core.kernel.time import Span


def _finite_vector(v: object) -> numpy.ndarray:
    a = as_numeric_array(v, 1)
    if a.size == 0:
        raise ValueError("an embedding is a non-empty vector of finite values")
    return require_finite(a, "an embedding")


def _finite_column(v: object) -> numpy.ndarray:
    return require_finite(as_numeric_array(v, 1), "a score column")


def _finite_matrix(v: object) -> numpy.ndarray:
    a = as_numeric_array(v, 2)
    if a.shape[1] == 0:
        raise ValueError("embedding rows are non-empty")
    return require_finite(a, "embeddings")


def _bool_column(v: object) -> numpy.ndarray:
    a = as_label_array(v)
    if a.dtype.kind != "b":
        raise ValueError(f"labels must be a 1-D bool array, got {a.dtype}")
    return a


def _check_lengths(**columns: numpy.ndarray | None) -> None:
    lengths = {name: len(c) for name, c in columns.items() if c is not None}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"columns differ in length: {lengths}")


class Annotation(Formatted):
    """A `label` or a `value` over `span`, with an optional `score`, from `producer`.

    `producer` names a human, an agent or a model; `guideline` is the annotation guideline version a
    human or agent followed; `uncertainty` is a non-negative spread in the units of `value`.
    """

    format: FormatTag = "annotation@1.0"

    span: Span
    label: str | None
    value: FiniteFloat | None
    score: FiniteFloat | None
    producer: NonEmptyStr
    guideline: str | None = None
    uncertainty: Annotated[FiniteFloat, Field(ge=0)] | None = None

    @model_validator(mode="after")
    def _label_or_value(self) -> "Annotation":
        if self.label is None and self.value is None:
            raise ValueError("an annotation needs a label or a value")
        return self


class Embedding(Formatted):
    """A finite 1-D `vector` in `space`, the resolved identity of the model that produced it."""

    format: FormatTag = "embedding@1.0"

    vector: Annotated[numpy.ndarray, BeforeValidator(_finite_vector)]
    space: NonEmptyStr


class Intervals(Formatted):
    """Columnar labelled spans `[start_ns[i], end_ns[i])` on `clock_id`."""

    format: FormatTag = "intervals@1.0"

    start_ns: NsArray
    end_ns: NsArray
    label: LabelArray
    clock_id: NonEmptyStr

    @model_validator(mode="after")
    def _check_columns(self) -> "Intervals":
        _check_lengths(start_ns=self.start_ns, end_ns=self.end_ns, label=self.label)
        if numpy.any(self.end_ns < self.start_ns):
            raise ValueError("an interval ends before it starts")
        return self


class Events(Formatted):
    """Columnar labelled instants `time_ns[i]` on `clock_id`, with an optional numeric `value` per event."""

    format: FormatTag = "events@1.0"

    time_ns: NsArray
    label: LabelArray
    value: Annotated[numpy.ndarray, BeforeValidator(lambda v: as_numeric_array(v, 1))] | None
    clock_id: NonEmptyStr

    @model_validator(mode="after")
    def _check_columns(self) -> "Events":
        _check_lengths(time_ns=self.time_ns, label=self.label, value=self.value)
        return self


Probability = Annotated[FiniteFloat, Field(ge=0, le=1)]
"""A finite float in `[0, 1]`."""

OffsetArray = Annotated[numpy.ndarray, BeforeValidator(lambda v: as_int64_array(v, "character offsets"))]
"""1-D frozen int64 character offsets."""


class TranscriptNode(Frozen):
    """A stretch of a transcript: its `text`, its `span` if timed, an optional `confidence`, and its `children`.

    `children` are the stretches it divides into (a segment into words), in order.
    """

    text: str
    span: Span | None = None
    confidence: Probability | None = None
    children: tuple["TranscriptNode", ...] = ()

    def nodes(self) -> Iterator["TranscriptNode"]:
        """Yield this node and every node below it, depth first, in order."""
        yield self
        for child in self.children:
            yield from child.nodes()


class Transcript(Formatted):
    """A tree of transcript nodes: `segments`, in order, each dividing into words; every span is on one clock."""

    format: FormatTag = "transcript@1.0"

    segments: tuple[TranscriptNode, ...]

    @model_validator(mode="after")
    def _one_clock(self) -> "Transcript":
        clocks = {n.span.clock_id for s in self.segments for n in s.nodes() if n.span is not None}
        if len(clocks) > 1:
            raise ValueError(f"a transcript's spans lie on one clock, got {sorted(clocks)}")
        return self


class TextSpans(Formatted):
    """Columnar labelled character spans `[start[i], end[i])` over the text whose content identity is `text`.

    Offsets count code points of the text's canonical content, of `length` code points, with
    `0 <= start <= end <= length`; `score` is an optional finite score per span.
    """

    format: FormatTag = "text-spans@1.0"

    text: TextIdentity
    length: Annotated[int, Strict(), Field(ge=0)]
    start: OffsetArray
    end: OffsetArray
    label: LabelArray
    score: Annotated[numpy.ndarray, BeforeValidator(_finite_column)] | None = None

    @model_validator(mode="after")
    def _check_columns(self) -> "TextSpans":
        _check_lengths(start=self.start, end=self.end, label=self.label, score=self.score)
        if numpy.any(self.start < 0) or numpy.any(self.end < self.start) or numpy.any(self.end > self.length):
            raise ValueError(f"a span lies outside [0, {self.length}] or ends before it starts")
        return self

    @classmethod
    def over(cls, text: Text, *, start: object, end: object, label: object, score: object = None) -> "TextSpans":
        """Return the spans over `text`, naming its identity and length."""
        return cls.model_validate(
            {"text": text.identity, "length": len(text.content), "start": start, "end": end, "label": label}
            | ({} if score is None else {"score": score})
        )


class Trials(Formatted):
    """Pairs of embeddings in `space`: row i of `first` against row i of `second`, `same[i]` true for one source."""

    format: FormatTag = "trials@1.0"

    first: Annotated[numpy.ndarray, BeforeValidator(_finite_matrix)]
    second: Annotated[numpy.ndarray, BeforeValidator(_finite_matrix)]
    same: Annotated[numpy.ndarray, BeforeValidator(_bool_column)]
    space: NonEmptyStr

    @model_validator(mode="after")
    def _check_pairs(self) -> "Trials":
        if self.first.shape != self.second.shape or self.same.shape != self.first.shape[:1]:
            raise ValueError(f"pairs of {self.first.shape} and {self.second.shape} with {self.same.shape} labels")
        return self


class Metric(Formatted):
    """The value of metric `name`, in `unit`, computed over `items` items."""

    format: FormatTag = "metric@1.0"

    name: NonEmptyStr
    value: FiniteFloat
    unit: NonEmptyStr
    items: Annotated[int, Strict(), Field(ge=1)]
