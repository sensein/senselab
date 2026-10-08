"""Which transcript words are the task's own content because they fall on the task's own events.

A non-lexical family (a syllable train, a breath, a cough, a held vowel) has its events read by the task
layer. A transcript word whose timing hull lies on those events is a recogniser's reading of the task,
whatever it spells, and is never PII. The families, the readings and the overlap rule are in
``data/task_content.yaml``; the design is ``specs/20261007-task-events-in-background/design.md``
("Task content in the transcript").
"""

from __future__ import annotations

import functools
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from senselab.audio.workflows.triage.nodes.common import find_measurement, word_hull
from senselab.audio.workflows.triage.routing_analysis import families as family_sets
from senselab.utils.prov_store import Entity, ProvStore

TASK_CONTENT_PATH = Path(__file__).parent / "data" / "task_content.yaml"

Span = tuple[float, float]


@functools.cache
def task_content_parameters() -> dict[str, Any]:
    """The parameters of ``data/task_content.yaml``.

    Returns:
        The parsed mapping.
    """
    return dict(yaml.safe_load(TASK_CONTENT_PATH.read_text()) or {})


def readings_for(family: str | None) -> tuple[str, ...]:
    """The task readings whose events are a family's own content.

    Args:
        family: The declared family, or None.

    Returns:
        The measurement names, in the data file's order; empty for a lexical, undeclared or unknown family.
    """
    if not family:
        return ()
    for set_name, names in (task_content_parameters().get("families") or {}).items():
        if family in getattr(family_sets, str(set_name), frozenset()):
            return tuple(str(name) for name in names or ())
    return ()


def reading_events(attributes: Mapping[str, Any], decisions: Sequence[str]) -> list[Span]:
    """The events one stored task reading read the task as.

    Args:
        attributes: The reading measurement's attributes.
        decisions: The evidence decisions that read the task.

    Returns:
        A phonation reading's holds where it found one; otherwise the evidence's events where its
        decision is one of ``decisions``; empty otherwise.
    """
    if "holds" in attributes:
        if not attributes.get("found"):
            return []
        return [(float(start), float(end)) for start, end in attributes.get("holds") or ()]
    evidence = (attributes.get("reading") or {}).get("evidence") or {}
    if evidence.get("decision") not in decisions:
        return []
    return [(float(event["start_s"]), float(event["end_s"])) for event in evidence.get("events") or ()]


def task_events(store: ProvStore, family: str | None) -> list[Span]:
    """The events the task layer read of the declared family's own task.

    Args:
        store: The provenance store.
        family: The declared family.

    Returns:
        Every event of the family's readings, in time order; empty for a lexical family or where no
        reading read the task.
    """
    decisions = tuple(str(d) for d in task_content_parameters().get("decisions") or ())
    events: list[Span] = []
    for name in readings_for(family):
        measurement = find_measurement(store, name)
        if measurement is not None:
            events.extend(reading_events(measurement.attributes, decisions))
    return sorted(events)


def _merged(spans: Sequence[Span], pad_s: float) -> list[Span]:
    merged: list[list[float]] = []
    for start, end in sorted((start - pad_s, end + pad_s) for start, end in spans):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def share_inside(hull: Span, spans: Sequence[Span], pad_s: float) -> float:
    """How much of a word's timing hull lies inside the events, each widened by ``pad_s``.

    Args:
        hull: The word's timing hull.
        spans: The events.
        pad_s: The widening either side.

    Returns:
        The covered fraction of the hull; for a zero-length hull, 1.0 where it lies inside an event and
        0.0 otherwise.
    """
    widened = _merged(spans, pad_s)
    low, high = hull
    if high <= low:
        return 1.0 if any(start <= low <= end for start, end in widened) else 0.0
    covered = sum(max(0.0, min(high, end) - max(low, start)) for start, end in widened)
    return covered / (high - low)


def task_content_ids(words: Sequence[Entity], events: Sequence[Span]) -> set[str]:
    """The words that are the task's own content: those lying on its events.

    Args:
        words: The consensus words.
        events: The task events (:func:`task_events`).

    Returns:
        The ids of the timed words whose hull's share inside the widened events is at least
        ``overlap.min_share``; empty where there are no events.
    """
    if not events:
        return set()
    overlap = task_content_parameters()["overlap"]
    pad_s, min_share = float(overlap["pad_s"]), float(overlap["min_share"])
    return {
        word.id
        for word in words
        if word.extent is not None and share_inside(word_hull(word), events, pad_s) >= min_share
    }
