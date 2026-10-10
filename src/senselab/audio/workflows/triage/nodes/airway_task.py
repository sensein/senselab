"""AIRWAY's task measures: the breath train and the cough onsets, written to the store.

A declared airway family named in ``data/airway_event_requirements.yaml`` is measured here, inside
the AIRWAY branch, and VERDICT reads only what this module writes. The design is in
``specs/20261006-airway-move/design.md``.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from senselab.audio.workflows.triage.background_model import BACKGROUND_MODEL
from senselab.audio.workflows.triage.breath_pattern import (
    BreathPattern,
    breath_in_review,
    breath_pattern_of,
)
from senselab.audio.workflows.triage.cough_pattern import CoughPattern, cough_pattern_of, in_cough_review_band
from senselab.audio.workflows.triage.nodes.branches import Expectation
from senselab.audio.workflows.triage.nodes.common import mint_live, write_measurement
from senselab.audio.workflows.triage.task_events import REVIEW
from senselab.audio.workflows.triage.vocabulary import SUPERSEDES, TASK_EXTENT_SPAN_ROLE
from senselab.utils.prov_store import ProvStore

AIRWAY_EVENT_REQUIREMENTS_PATH = Path(__file__).parents[1] / "data" / "airway_event_requirements.yaml"

BREATH_READING = "airway_breath_reading"
"""The measurement holding AIRWAY's breathing-pattern reading, or the inputs it lacked."""

COUGH_READING = "airway_cough_reading"
"""The measurement holding AIRWAY's cough-onset reading, or the inputs it lacked."""


@functools.cache
def _airway_event_requirements() -> dict[str, str]:
    """``data/airway_event_requirements.yaml``: per declared airway family, the event kind it is decided on."""
    document = yaml.safe_load(AIRWAY_EVENT_REQUIREMENTS_PATH.read_text()) or {}
    return {
        str(family): str(kind) for kind, families in document.items() if kind != "version" for family in families or ()
    }


def required_event(declared_family: str | None) -> str | None:
    """The event kind a declared airway family is decided on, or None.

    Args:
        declared_family: The task family the recording declares.

    Returns:
        ``breath`` or ``cough`` for a family ``data/airway_event_requirements.yaml`` lists under it;
        None otherwise.
    """
    return _airway_event_requirements().get(declared_family or "")


def breath_attributes(read: BreathPattern | tuple[str, ...]) -> dict[str, Any]:
    """The breath measurement's attributes: the reading and its derived counts, or what was absent.

    Args:
        read: What :func:`~senselab.audio.workflows.triage.breath_pattern.breath_pattern_of` returned.

    Returns:
        ``absent`` naming the missing inputs; or the task evidence's ``decision`` and the breaths its
        cluster counts (``breaths``), ``review``, the old measure's ``pattern``, ``events_n`` and
        ``vetoed_by`` (reported, deciding nothing), and the full ``reading``.
    """
    if not isinstance(read, BreathPattern):
        return {"absent": list(read)}
    phases = len(read.evidence.events) if read.evidence is not None else None
    return {
        "absent": [] if read.evidence is not None else [BACKGROUND_MODEL],
        "decision": read.evidence.decision if read.evidence is not None else None,
        "breaths": None if phases is None else int(np.floor(phases / 2 + 0.5)),
        "pattern": read.pattern,
        "events_n": read.events_n,
        "vetoed_by": read.veto.vetoed_by if read.veto is not None else None,
        "train_breaths": read.train.breaths if read.train is not None else None,
        "review": breath_in_review(read),
        "reading": read.record(),
    }


def cough_attributes(read: CoughPattern | tuple[str, ...], instructed: int | None) -> dict[str, Any]:
    """The cough measurement's attributes: the reading and its derived counts, or what was absent.

    Args:
        read: What :func:`~senselab.audio.workflows.triage.cough_pattern.cough_pattern_of` returned.
        instructed: The instructed cough count, or None where the task names none.

    Returns:
        ``absent`` naming the missing inputs; or ``onsets_n``, ``review`` (the strict/lenient band, or
        the task evidence's ``review`` decision) and the full ``reading``.
    """
    if not isinstance(read, CoughPattern):
        return {"absent": list(read)}
    reviewed = read.evidence is not None and read.evidence.decision == REVIEW
    return {
        "absent": [],
        "onsets_n": read.onsets_n,
        "review": in_cough_review_band(read, instructed or 1) or reviewed,
        "reading": read.record(),
    }


def settle_task_extent(store: ProvStore, activity: str, software: str, extent: Mapping[str, Any] | None) -> str | None:
    """Write the measure's task extent as the task-extent span that stands, retiring any earlier one.

    The span carries ``supersedes``, naming the live task-extent spans that do not, which stay live;
    readers take it in their place (:func:`~senselab.audio.workflows.triage.vocabulary.standing_task_extents`).
    With no extent, an earlier superseding span is retired and the branches' spans stand again.

    Args:
        store: The provenance store.
        activity: AIRWAY's activity.
        software: The software agent.
        extent: The reading's ``extent`` record, or None.

    Returns:
        The standing span's id, or None where no extent was written.
    """
    spans = [
        span
        for span in store.entities("span")
        if span.attributes.get("role") == TASK_EXTENT_SPAN_ROLE
        and span.extent is not None
        and not store.is_invalidated(span.id)
    ]
    ours = [span for span in spans if span.attributes.get(SUPERSEDES) is not None]
    branches = sorted(span.id for span in spans if span.attributes.get(SUPERSEDES) is None)
    kept: str | None = None
    if extent is not None:
        kept = mint_live(
            store,
            prov_type="span",
            extent=(float(extent["start_s"]), float(extent["end_s"])),
            attributes={
                "family": "airway",
                "role": TASK_EXTENT_SPAN_ROLE,
                "extent_from": str(extent["source"]),
                "phases": extent.get("phases"),
                "breaths": extent.get("breaths"),
                SUPERSEDES: branches,
            },
        )
        if kept not in {span.id for span in ours}:
            store.was_generated_by(kept, activity)
            store.was_attributed_to(kept, software)
            for branch_id in branches:
                store.was_derived_from(kept, branch_id)
    for span in ours:
        if span.id != kept:
            store.was_invalidated_by(span.id, activity)
    return kept


def measure_airway_task(
    store: ProvStore,
    activity: str,
    software: str,
    *,
    family: str,
    expectation: Expectation,
    sampling_hz: float,
    language: str | None,
    run_dir: Path,
) -> list[str]:
    """Measure a declared airway task and write the reading and the standing extent.

    Args:
        store: The provenance store.
        activity: AIRWAY's activity.
        software: The software agent.
        family: The declared airway family.
        expectation: Its ``AIRWAY_EXPECTATIONS`` row.
        sampling_hz: The conditioned stream's sampling rate.
        language: The recording's declared language, or None.
        run_dir: The run directory the derivatives' sidecar paths are relative to.

    Returns:
        The ids written: the reading measurement and the standing extent span where one was placed.
        Empty for a family no measure decides.
    """
    needed = required_event(family)
    if needed not in ("breath", "cough"):
        return []
    instructed = expectation.required_count.value if expectation.required_count is not None else None
    breath = cough = None
    if needed == "breath":
        breath = breath_pattern_of(
            store, run_dir, sampling_hz=sampling_hz, language=language, family=family, instructed=instructed
        )
        name, attributes = BREATH_READING, breath_attributes(breath)
    else:
        cough = cough_pattern_of(store, run_dir, sampling_hz=sampling_hz, language=language, instructed=instructed)
        name, attributes = COUGH_READING, cough_attributes(cough, instructed)
    written = [write_measurement(store, activity, software, name=name, signal="plain", attributes=attributes)]
    reading = attributes.get("reading") or {}
    kept = settle_task_extent(store, activity, software, reading.get("extent"))
    if kept is not None:
        written.append(kept)
    return written
