"""QUALITY's join: what the recording-level background found, read against each task span.

The branches place the task spans and events; BACKGROUND reads faults, activity and the floor; the
residual reads another voice. This module relates them by time overlap: a finding that overlaps or
abuts a task span is in the task, anything else is outside it. It also re-reads the raw stream's
task events on the enhanced stream, and says whether anything was captured at all.

Every parameter is in ``data/quality_join.yaml``; the design is
``specs/20261007-task-events-in-background/design.md`` ("Unit C step 2").
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from senselab.audio.workflows.triage.task_events import GenericView, event_snr_db

QUALITY_JOIN_PATH = Path(__file__).parent / "data" / "quality_join.yaml"
QUALITY_JOIN = "quality_join"
"""QUALITY's measurement of the join."""

OTHER_VOICE = "other_voice"
FAULT_KINDS = ("shutoff", "gate", "dropout", "clip", "discontinuity")
INSIDE = "in"
OUTSIDE = "out"

Span = tuple[float, float]


@functools.cache
def quality_join_parameters() -> dict[str, Any]:
    """The parameters of ``data/quality_join.yaml``.

    Returns:
        The parsed mapping.
    """
    return dict(yaml.safe_load(QUALITY_JOIN_PATH.read_text()) or {})


def touches(span: Span, task_spans: Sequence[Span], abut_s: float) -> bool:
    """Whether a finding overlaps a task span or lies within ``abut_s`` of one.

    Args:
        span: The finding.
        task_spans: The task spans.
        abut_s: How close a finding may end before or start after a span and still touch it.

    Returns:
        True where some task span reaches within ``abut_s`` of the finding.
    """
    start, end = span
    return any(end >= a - abut_s and start <= b + abut_s for a, b in task_spans)


def split_by_task(
    spans: Sequence[Span], task_spans: Sequence[Span], abut_s: float, min_s: float = 0.0
) -> dict[str, list[list[float]]]:
    """Findings split into those touching a task span (and lasting ``min_s``) and the rest.

    Args:
        spans: The findings.
        task_spans: The task spans.
        abut_s: :func:`touches`'s tolerance.
        min_s: The shortest finding that counts as in the task; a shorter one touching it is outside.

    Returns:
        ``{"in": [...], "out": [...]}``, each a list of ``[start, end]``.
    """
    inside: list[list[float]] = []
    outside: list[list[float]] = []
    for start, end in spans:
        row = [round(float(start), 4), round(float(end), 4)]
        if touches((start, end), task_spans, abut_s) and end - start >= min_s:
            inside.append(row)
        else:
            outside.append(row)
    return {INSIDE: inside, OUTSIDE: outside}


def faults_against(
    faults: Mapping[str, Sequence[Sequence[float]]], task_spans: Sequence[Span], p: dict[str, Any]
) -> dict[str, dict[str, list[list[float]]]]:
    """BACKGROUND's fault spans split by whether they fall in a task span.

    Args:
        faults: The ``faults`` record of the ``background_model`` measurement, kind to spans.
        task_spans: The task spans.
        p: The parameters.

    Returns:
        Per fault kind of :data:`FAULT_KINDS`, :func:`split_by_task` under its ``min_s`` and ``abut_s``.
    """
    out: dict[str, dict[str, list[list[float]]]] = {}
    for kind in FAULT_KINDS:
        q = p["faults"][kind]
        spans = [(float(a), float(b)) for a, b in faults.get(kind) or ()]
        out[kind] = split_by_task(spans, task_spans, float(q.get("abut_s", p["abut_s"])), float(q["min_s"]))
    return out


def drop_levels(shutoffs: Sequence[Sequence[float]], raw: GenericView | None, p: dict[str, Any]) -> list[list[Any]]:
    """Each dead stretch with the signal's level over the floor just before it dropped.

    Args:
        shutoffs: The dead stretches BACKGROUND read as shutoffs.
        raw: The plain stream's background, or None where it is not stored.
        p: The parameters.

    Returns:
        ``[start, end, level]`` per stretch: the level over the floor in the ``faults.shutoff.before_s``
        before it, or None without the background.
    """
    before_s = float(p["faults"]["shutoff"]["before_s"])
    out: list[list[Any]] = []
    for a, b in shutoffs:
        level = None if raw is None else event_snr_db(raw, float(a) - before_s, float(a) - raw.hop_s)
        out.append([round(float(a), 4), round(float(b), 4), None if level is None else round(level, 2)])
    return out


def separate_gates(drops: Sequence[Sequence[Any]], p: dict[str, Any]) -> tuple[list[Span], list[Span]]:
    """BACKGROUND's dead stretches split into shutoffs that cut a sounding signal and gates it had decayed into.

    Args:
        drops: :func:`drop_levels`'s rows.
        p: The parameters.

    Returns:
        ``(cut, gated)``: a stretch is a cut where the signal stood ``faults.shutoff.sounding_db`` over
        the floor just before it dropped, else a gate. A stretch whose level was not read is a cut.
    """
    sounding = float(p["faults"]["shutoff"]["sounding_db"])
    cut: list[Span] = []
    gated: list[Span] = []
    for a, b, level in drops:
        (cut if level is None or level >= sounding else gated).append((float(a), float(b)))
    return cut, gated


def deciding_kinds(split: Mapping[str, Mapping[str, Sequence[Any]]], section: Mapping[str, Any]) -> list[str]:
    """The kinds whose findings fall in a task span and whose parameters let them decide.

    Args:
        split: Kind to ``{"in": ..., "out": ...}``.
        section: The parameters' section for those kinds.

    Returns:
        The kinds, in the section's order.
    """
    return [kind for kind, q in section.items() if q.get("decides") and (split.get(kind) or {}).get(INSIDE)]


def stream_agreement(
    events: Sequence[Span], raw: GenericView, enhanced: GenericView | None, p: dict[str, Any]
) -> dict[str, Any]:
    """The raw stream's task events re-read on the enhanced stream, each over its own stream's floor.

    Args:
        events: The task events, in time order.
        raw: The plain stream's background.
        enhanced: The enhanced stream's background, or None where the stream is not stored.
        p: The parameters.

    Returns:
        ``standing_n`` (events standing ``streams.snr_low_db`` over the raw floor), ``lost_n`` (of those,
        the ones under it on the enhanced stream), ``lost_fraction``, and ``disagree`` where at least
        ``streams.lost_fraction`` of them are lost; ``disagree`` is None where nothing could be compared.
    """
    q = p["streams"]
    low = float(q["snr_low_db"])
    if enhanced is None:
        return {"standing_n": 0, "lost_n": 0, "lost_fraction": None, "disagree": None}
    standing = [(a, b) for a, b in events if event_snr_db(raw, a, b) >= low]
    lost = [(a, b) for a, b in standing if event_snr_db(enhanced, a, b) < low]
    fraction = len(lost) / len(standing) if standing else None
    return {
        "standing_n": len(standing),
        "lost_n": len(lost),
        "lost_fraction": None if fraction is None else round(fraction, 3),
        "disagree": None if fraction is None else fraction >= float(q["lost_fraction"]),
    }


def join_record(
    *,
    task_spans: Sequence[Span],
    event_kind: str | None,
    faults: Mapping[str, Sequence[Sequence[float]]],
    other_voice: Sequence[Span] | None,
    streams: Mapping[str, Any] | None,
    plain_active_s: float | None,
    enhanced_active_s: float | None,
    level_rel_db: float | None,
    p: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The join's record: findings split by the task spans, what decides, and whether anything was captured.

    Args:
        task_spans: The task spans, from the branches.
        event_kind: The task events' kind (``breath``, ``cough``, ``phonation``), or None.
        faults: The ``background_model`` measurement's ``faults``.
        other_voice: Spans of another voice over the task's surroundings, or None where it was not read.
        streams: :func:`stream_agreement`'s record, or None.
        plain_active_s: Seconds of activity on the plain stream, or None where BACKGROUND read none.
        enhanced_active_s: Seconds of activity on the enhanced stream, or None.
        level_rel_db: The recording's active level against its session's, or None.
        p: The parameters; ``data/quality_join.yaml`` when None.

    Returns:
        The record QUALITY writes.
    """
    p = p or quality_join_parameters()
    fault_split = faults_against(faults, task_spans, p)
    interference: dict[str, dict[str, list[list[float]]]] = {}
    if other_voice is not None:
        interference[OTHER_VOICE] = split_by_task(other_voice, task_spans, float(p["abut_s"]))
    agreement = dict(streams or {})
    disagree = bool(agreement.get("disagree")) and event_kind in tuple(p["streams"]["decides"] or ())
    no_activity = plain_active_s is not None and plain_active_s <= 0.0 and not (enhanced_active_s or 0.0) > 0.0
    level_max = float(p["capture"]["level_rel_db_max"])
    return {
        "task_spans": [[round(a, 4), round(b, 4)] for a, b in task_spans],
        "event_kind": event_kind,
        "faults": fault_split,
        "faults_in_task": deciding_kinds(fault_split, p["faults"]),
        "interference": interference,
        "interference_in_task": deciding_kinds(interference, p["interference"]),
        "streams": agreement,
        "streams_disagree": disagree,
        "plain_active_s": plain_active_s,
        "enhanced_active_s": enhanced_active_s,
        "no_activity": no_activity,
        "level_rel_db": level_rel_db,
        "quiet_vs_session": level_rel_db is not None and level_rel_db <= level_max,
        "level_rel_db_max": level_max,
        "fault_min_s": {kind: float(q["min_s"]) for kind, q in p["faults"].items()},
        "lost_fraction_max": float(p["streams"]["lost_fraction"]),
    }
