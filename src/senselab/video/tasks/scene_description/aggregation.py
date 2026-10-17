"""Per-video summaries of per-frame scene descriptions and image metrics."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel

from senselab.video.tasks.scene_description.cv_metrics import FrameMetrics
from senselab.video.tasks.scene_description.schema import DISTRACTIONS, SceneDescription

CATEGORICAL_FIELDS: Tuple[str, ...] = (
    "layout",
    "any_other_people_present",
    "primary_camera_on",
    "primary_setting",
    "primary_background_type",
    "primary_lighting",
    "primary_head_framing",
)
NUMERIC_FIELDS: Tuple[str, ...] = ("n_tiles", "n_camera_on", "n_people_visible")
CHANGE_FIELDS: Tuple[str, ...] = ("layout", "n_tiles", "any_other_people_present")

Value = Union[str, int, float, bool]


class CategoricalSummary(BaseModel):
    """Distribution of one categorical field over frames.

    Attributes:
        mode: The most frequent value (ties go to the value seen first).
        consistency: Fraction of frames whose value equals ``mode``.
        counts: Frames per value.
        n: Frames with a value.
    """

    mode: Optional[Value] = None
    consistency: float = 0.0
    counts: Dict[str, int] = {}
    n: int = 0


class NumericSummary(BaseModel):
    """Distribution of one numeric field over frames.

    Attributes:
        mean: Mean value.
        median: Median value.
        min: Smallest value.
        max: Largest value.
        consistency: Fraction of frames equal to the most frequent value.
        n: Frames with a value.
    """

    mean: Optional[float] = None
    median: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    consistency: float = 0.0
    n: int = 0


class ChangePoint(BaseModel):
    """A change of a field's value between two consecutive described frames.

    Attributes:
        timestamp_s: Time of the first frame with the new value.
        previous_timestamp_s: Time of the last frame with the old value.
        field: The field that changed.
        before: The old value.
        after: The new value.
    """

    timestamp_s: float
    previous_timestamp_s: float
    field: str
    before: Value
    after: Value


class VideoSceneSummary(BaseModel):
    """The per-video aggregate.

    Attributes:
        n_frames: Frames sampled.
        n_parsed: Frames with a valid description.
        parse_rate: ``n_parsed / n_frames``.
        categorical: Summaries of :data:`CATEGORICAL_FIELDS`.
        numeric: Summaries of :data:`NUMERIC_FIELDS`.
        distractions: Fraction of described frames in which any tile lists each distraction.
        metrics: Summaries of each numeric ``FrameMetrics`` field over all frames.
        change_points: Changes in :data:`CHANGE_FIELDS` between consecutive described frames.
    """

    n_frames: int
    n_parsed: int
    parse_rate: float
    categorical: Dict[str, CategoricalSummary]
    numeric: Dict[str, NumericSummary]
    distractions: Dict[str, float]
    metrics: Dict[str, NumericSummary]
    change_points: List[ChangePoint]


def frame_fields(description: SceneDescription) -> Dict[str, Value]:
    """Flatten a description into the frame-level fields the summary aggregates.

    ``primary_*`` fields come from the first (largest) tile; counts are summed over tiles.

    Args:
        description: One frame's description.

    Returns:
        Field name to value; ``primary_*`` fields are absent when there are no tiles.
    """
    tiles = description.tiles
    fields: Dict[str, Value] = {
        "layout": description.layout,
        "n_tiles": description.n_tiles,
        "n_camera_on": sum(t.camera_on for t in tiles),
        "n_people_visible": sum(t.n_people_visible for t in tiles),
        "any_other_people_present": any(t.other_people_present for t in tiles),
    }
    if tiles:
        p = tiles[0]
        fields.update(
            primary_camera_on=p.camera_on,
            primary_setting=p.setting,
            primary_background_type=p.background_type,
            primary_lighting=p.lighting,
            primary_head_framing=p.head_framing,
        )
    return fields


def summarise_categorical(values: Sequence[Value]) -> CategoricalSummary:
    """Mode, consistency and counts of a sequence of values.

    Args:
        values: One value per frame.

    Returns:
        The summary; empty when ``values`` is empty.
    """
    if not values:
        return CategoricalSummary()
    counts = Counter(values)
    mode, n_mode = counts.most_common(1)[0]
    return CategoricalSummary(
        mode=mode,
        consistency=n_mode / len(values),
        counts={str(k): v for k, v in counts.items()},
        n=len(values),
    )


def summarise_numeric(values: Sequence[float]) -> NumericSummary:
    """Mean, median, range and consistency of a sequence of numbers.

    Args:
        values: One value per frame.

    Returns:
        The summary; empty when ``values`` is empty.
    """
    if not values:
        return NumericSummary()
    arr = np.asarray(values, dtype=np.float64)
    n_mode = Counter(values).most_common(1)[0][1]
    return NumericSummary(
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        min=float(arr.min()),
        max=float(arr.max()),
        consistency=n_mode / len(values),
        n=len(values),
    )


def change_points(
    timestamps: Sequence[float], fields: Sequence[Dict[str, Any]], names: Sequence[str] = CHANGE_FIELDS
) -> List[ChangePoint]:
    """Changes of each named field between consecutive frames.

    Args:
        timestamps: Frame times, in order.
        fields: Frame-level fields per frame (see :func:`frame_fields`).
        names: Fields to track.

    Returns:
        Change points ordered by time, then by field.
    """
    points: List[ChangePoint] = []
    for i in range(1, len(fields)):
        for name in names:
            before, after = fields[i - 1].get(name), fields[i].get(name)
            if before is not None and after is not None and before != after:
                points.append(
                    ChangePoint(
                        timestamp_s=timestamps[i],
                        previous_timestamp_s=timestamps[i - 1],
                        field=name,
                        before=before,
                        after=after,
                    )
                )
    return points


def aggregate(
    timestamps: Sequence[float],
    descriptions: Sequence[Optional[SceneDescription]],
    metrics: Sequence[Optional[FrameMetrics]],
) -> VideoSceneSummary:
    """Summarise a video's frames.

    Args:
        timestamps: Frame times, in order.
        descriptions: The valid description per frame, or None where parsing failed.
        metrics: Image metrics per frame, or None where not computed.

    Returns:
        The per-video summary.
    """
    if not (len(timestamps) == len(descriptions) == len(metrics)):
        raise ValueError("timestamps, descriptions and metrics must have one entry per frame")
    parsed = [(t, frame_fields(d), d) for t, d in zip(timestamps, descriptions) if d is not None]
    fields = [f for _, f, _ in parsed]
    categorical = {name: summarise_categorical([f[name] for f in fields if name in f]) for name in CATEGORICAL_FIELDS}
    numeric = {name: summarise_numeric([float(f[name]) for f in fields if name in f]) for name in NUMERIC_FIELDS}
    distractions = {
        item: (sum(any(item in t.distractions for t in d.tiles) for _, _, d in parsed) / len(parsed)) if parsed else 0.0
        for item in DISTRACTIONS
        if item != "none"
    }
    present = [m for m in metrics if m is not None]
    metric_summary = {
        name: summarise_numeric([float(getattr(m, name)) for m in present])
        for name, field in FrameMetrics.model_fields.items()
        if field.annotation in (int, float)
    }
    n = len(timestamps)
    return VideoSceneSummary(
        n_frames=n,
        n_parsed=len(parsed),
        parse_rate=len(parsed) / n if n else 0.0,
        categorical=categorical,
        numeric=numeric,
        distractions=distractions,
        metrics=metric_summary,
        change_points=change_points([t for t, _, _ in parsed], fields),
    )
