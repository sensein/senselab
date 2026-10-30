"""Result types: per-item annotations and embeddings, columnar intervals and events."""

from typing import Annotated

import numpy
from pydantic import BeforeValidator, Field, model_validator

from senselab_core.kernel._base import FiniteFloat, LabelArray, NonEmptyStr, NsArray, as_numeric_array
from senselab_core.kernel.formats import FormatTag, Formatted
from senselab_core.kernel.time import Span


def _finite_vector(v: object) -> numpy.ndarray:
    a = as_numeric_array(v, 1)
    if a.size == 0 or not numpy.all(numpy.isfinite(a)):
        raise ValueError("an embedding is a non-empty vector of finite values")
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
