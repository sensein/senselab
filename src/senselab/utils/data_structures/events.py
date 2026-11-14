"""Behaviour events: `BehaviorEvent`, `EventTrack` and `AnnotationSchema`.

A `BehaviorEvent` is a labelled time interval `[onset_s, offset_s)` in seconds from the start of a
clip. An `EventTrack` holds the events one source produced for one clip under one schema. An
`AnnotationSchema` names the labels a track may use, and loads from a YAML file.

A label for a whole clip is an event spanning the clip; `whole_clip_event` builds one.

Example:
    >>> schema = AnnotationSchema(schema_id="motion", task_type="segmentation", labels=["reach", "wave"])
    >>> track = EventTrack(schema_id="motion", events=[whole_clip_event("wave", 4.0, "annotator_a")])
    >>> track.check(schema)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictStr, StringConstraints, field_validator, model_validator

ConfidenceKind = Literal["vote", "prob", "logit", "none"]
TaskType = Literal["classification", "counting", "description", "segmentation"]

PREVALENCE_TOLERANCE = 1e-9

# A finite number: ints are accepted and stored as floats; bools, strings, NaN and infinities are refused.
FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
# A string that is either absent (`None`) or non-empty, so "" and `None` cannot both mean "missing".
NonEmptyStr = Annotated[str, StringConstraints(strict=True, min_length=1)]


class BehaviorEvent(BaseModel):
    """One labelled interval `[onset_s, offset_s)`.

    Attributes:
        onset_s: Start, in seconds from the start of the clip; inclusive.
        offset_s: End, in seconds; exclusive, so it must be greater than `onset_s`.
        label: A label from the track's `AnnotationSchema`.
        confidence: The producer's confidence, or `None` when it gives none.
        confidence_kind: What `confidence` is: a vote fraction, a probability or a logit; `none` exactly
            when `confidence` is `None`.
        source: The producing model's agent id, or an annotator's role label (never a person's name).
        subject_ref: Optional opaque subject key, non-empty when given; never a path.
        track_id: Optional id of the tracked individual within the clip, non-empty when given.

    Times and the confidence are finite numbers; an int is accepted, a bool or a string is not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    onset_s: FiniteFloat
    offset_s: FiniteFloat
    label: StrictStr
    confidence: Optional[FiniteFloat] = None
    confidence_kind: ConfidenceKind = "none"
    source: StrictStr
    subject_ref: Optional[NonEmptyStr] = None
    track_id: Optional[NonEmptyStr] = None

    @model_validator(mode="after")
    def _check_interval_and_confidence(self) -> "BehaviorEvent":
        if self.onset_s < 0:
            raise ValueError("onset_s must be >= 0")
        if self.onset_s >= self.offset_s:
            raise ValueError("onset_s must be < offset_s (intervals are half-open)")
        if (self.confidence is None) != (self.confidence_kind == "none"):
            raise ValueError("confidence_kind must be 'none' exactly when confidence is None")
        return self


class AnnotationSchema(BaseModel):
    """The labels a set of tracks may use.

    Attributes:
        schema_id: Identifier that tracks refer to.
        task_type: `classification`, `counting`, `description` or `segmentation`.
        labels: The allowed labels, unique, always strings.
        background_label: Optional label for "none of the labels"; allowed in tracks besides `labels`.
        prompt_template: Optional prompt text for model annotators.
        reference_prevalence: Optional caller-supplied label distribution (for example the training
            set's), over a subset of `labels` and summing to 1; reports show it beside the realised one.

    Unknown keys are refused. A caller whose schema files carry extra metadata maps them to these
    fields (or drops the rest) before constructing the schema.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_id: StrictStr
    task_type: TaskType
    labels: List[StrictStr]
    background_label: Optional[StrictStr] = None
    prompt_template: Optional[StrictStr] = None
    reference_prevalence: Optional[Dict[StrictStr, FiniteFloat]] = None

    @model_validator(mode="after")
    def _check_labels_and_prevalence(self) -> "AnnotationSchema":
        if not self.labels:
            raise ValueError("labels must not be empty")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")
        if self.reference_prevalence is not None:
            foreign = sorted(set(self.reference_prevalence) - set(self.labels))
            if foreign:
                raise ValueError(f"reference_prevalence keys not in labels: {foreign}")
            values = list(self.reference_prevalence.values())
            if any(not (0.0 <= v <= 1.0) for v in values):
                raise ValueError("reference_prevalence values must be in [0, 1]")
            if abs(math.fsum(values) - 1.0) > PREVALENCE_TOLERANCE:
                raise ValueError("reference_prevalence values must sum to 1")
        return self

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "AnnotationSchema":
        """Load a schema from a YAML mapping.

        Args:
            path: The YAML file.

        Returns:
            The schema.

        Raises:
            ValueError: If the file is not a mapping, or if a label, the background label or a
                reference-prevalence key loaded as a YAML boolean (an unquoted `yes`, `no`, `on`, `off`,
                `true` or `false`) or as another non-string.
        """
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("an annotation schema file must hold a mapping")
        labels = data.get("labels")
        if isinstance(labels, list):
            for i, label in enumerate(labels):
                _require_yaml_string(label, f"labels[{i}]")
        if data.get("background_label") is not None:
            _require_yaml_string(data["background_label"], "background_label")
        prevalence = data.get("reference_prevalence")
        if isinstance(prevalence, dict):
            for key in prevalence:
                _require_yaml_string(key, f"reference_prevalence key {key!r}")
        return cls(**data)


def _require_yaml_string(value: object, where: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{where} loaded as a YAML boolean; quote it")
    if not isinstance(value, str):
        raise ValueError(f"{where} is not a string")


class EventTrack(BaseModel):
    """The events one source produced for one clip under one schema, sorted by onset, offset, label.

    Attributes:
        schema_id: The `AnnotationSchema` the labels come from.
        events: The events; stored as a tuple sorted by `(onset_s, offset_s, label)`.
        duration_s: Optional clip duration; no event may end after it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_id: StrictStr
    events: Tuple[BehaviorEvent, ...]
    duration_s: Optional[FiniteFloat] = None

    @field_validator("events")
    @classmethod
    def _sort_events(cls, events: Tuple[BehaviorEvent, ...]) -> Tuple[BehaviorEvent, ...]:
        return tuple(sorted(events, key=lambda e: (e.onset_s, e.offset_s, e.label)))

    @model_validator(mode="after")
    def _check_duration(self) -> "EventTrack":
        if self.duration_s is not None:
            if self.duration_s <= 0:
                raise ValueError("duration_s must be > 0")
            late = sum(e.offset_s > self.duration_s for e in self.events)
            if late:
                raise ValueError(f"{late} events end after duration_s")
        return self

    def check(self, schema: AnnotationSchema) -> None:
        """Check the track against a schema.

        Args:
            schema: The schema the track claims to follow.

        Raises:
            ValueError: If the schema ids differ, or an event's label is neither a schema label nor
                the schema's background label.
        """
        if self.schema_id != schema.schema_id:
            raise ValueError(f"track schema_id {self.schema_id!r} != schema {schema.schema_id!r}")
        allowed = set(schema.labels)
        if schema.background_label is not None:
            allowed.add(schema.background_label)
        foreign = sorted({e.label for e in self.events} - allowed)
        if foreign:
            raise ValueError(f"labels not in schema {schema.schema_id!r}: {foreign}")


def whole_clip_event(label: str, duration_s: float, source: str) -> BehaviorEvent:
    """An event labelling a whole clip: `[0, duration_s)` with no confidence.

    Args:
        label: The clip's label.
        duration_s: The clip's duration in seconds.
        source: The producing model's agent id or an annotator's role label.

    Returns:
        The event.
    """
    return BehaviorEvent(onset_s=0.0, offset_s=duration_s, label=label, source=source)
