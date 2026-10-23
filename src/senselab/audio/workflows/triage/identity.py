"""Who and what a recording is: its participant, session and task, declared once and recorded on the run.

A caller declares a recording's identity in its hint's ``metadata`` under :data:`PARTICIPANT_KEY`,
:data:`SESSION_KEY` and :data:`TASK_TOKEN_KEY`. ADMIT resolves it once with :func:`identity_of` and writes
it onto the ``recording`` stream entity it already creates (:meth:`RecordingIdentity.as_attributes`), so
a reader that holds only the store sees the declaration too.

Whatever the hint leaves out is read off the recording's file stem, exactly as before this module: the
``task-`` entity through :func:`~senselab.audio.workflows.triage.routing_analysis.families.task_id_of`,
and the session as an adjacent ``sub-<label>_ses-<label>`` pair. A store written before ADMIT recorded an
identity is read the same way, off its recording's stem, so earlier runs replay unchanged.

See ``specs/20261022-triage-dataset-profile/plan.md``, step 1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.routing_analysis.families import UNKNOWN_TASK, task_family, task_id_of
from senselab.utils.prov_store import Entity, ProvStore

PARTICIPANT_KEY = "participant_id"
"""The hint ``metadata`` key a caller declares the participant under, used as given (e.g. ``sub-01``)."""

SESSION_KEY = "session_id"
"""The hint ``metadata`` key a caller declares the session under, used as given (e.g. ``ses-02``)."""

TASK_TOKEN_KEY = "task_token"
"""The hint ``metadata`` key a caller declares the task id under; lower-cased, trailing index intact."""

DECLARED_PARTICIPANT = "declared_participant"
DECLARED_SESSION = "declared_session"
DECLARED_TASK_ID = "declared_task_id"
DECLARED_FROM = "declared_from"
"""The ``recording`` stream attributes ADMIT records the resolved identity under."""

FROM_HINT = "hint"
FROM_STEM = "stem"
"""Where a field of the identity came from."""

RECORDING_STREAM = "recording"
ADMIT_NODE = "ADMIT"

_PARTICIPANT_AND_SESSION = re.compile(r"(sub-[^_]+)_(ses-[^_]+)")


@dataclass(frozen=True)
class RecordingIdentity:
    """A recording's participant, session and task, and where each was read from.

    Attributes:
        participant: The participant, e.g. ``sub-01``, or None where nothing names one.
        session: The session, e.g. ``ses-02``, or None where nothing names one.
        task_id: The lower-cased task id with its trailing index, or :data:`UNKNOWN_TASK`.
        sources: :data:`FROM_HINT` or :data:`FROM_STEM` for each of ``participant``, ``session`` and
            ``task_id``.
    """

    participant: str | None
    session: str | None
    task_id: str
    sources: Mapping[str, str] = field(default_factory=dict)

    @property
    def task_family(self) -> str:
        """The family the task id collapses into: its trailing numeric segments removed."""
        return task_family(self.task_id)

    @property
    def session_key(self) -> str | None:
        """``<participant>_<session>``, the key sessions are grouped by, or None unless both are named."""
        return f"{self.participant}_{self.session}" if self.participant and self.session else None

    def as_attributes(self) -> dict[str, Any]:
        """The attributes ADMIT writes onto the ``recording`` stream entity."""
        return {
            DECLARED_PARTICIPANT: self.participant,
            DECLARED_SESSION: self.session,
            DECLARED_TASK_ID: self.task_id,
            DECLARED_FROM: dict(self.sources),
        }


def session_key_of(stem: str) -> str | None:
    """The ``sub-<label>_ses-<label>`` pair a file stem names, or None.

    Args:
        stem: A recording's file stem.

    Returns:
        The adjacent participant and session labels, or None where the stem names no such pair.
    """
    found = _PARTICIPANT_AND_SESSION.search(stem)
    return f"{found.group(1)}_{found.group(2)}" if found else None


def identity_from_stem(stem: str) -> RecordingIdentity:
    """The identity a file stem alone names: the reading every caller made before hints declared one.

    The participant and session are taken only from an adjacent ``sub-<label>_ses-<label>`` pair, so
    :attr:`RecordingIdentity.session_key` equals :func:`session_key_of` for every stem.

    Args:
        stem: A recording's file stem.

    Returns:
        The identity, every field from :data:`FROM_STEM`.
    """
    found = _PARTICIPANT_AND_SESSION.search(stem)
    return RecordingIdentity(
        participant=found.group(1) if found else None,
        session=found.group(2) if found else None,
        task_id=task_id_of(stem),
        sources={"participant": FROM_STEM, "session": FROM_STEM, "task_id": FROM_STEM},
    )


def _declared(hint: AudioHints | None, key: str) -> str | None:
    value = (hint.metadata or {}).get(key) if hint is not None else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def identity_of(stem: str, hint: AudioHints | None = None) -> RecordingIdentity:
    """A recording's identity: each field from the hint where it declares one, otherwise off the stem.

    Args:
        stem: The recording's file stem.
        hint: What the recording was declared to contain, or None.

    Returns:
        The identity, with each field's source.
    """
    parsed = identity_from_stem(stem)
    participant = _declared(hint, PARTICIPANT_KEY)
    session = _declared(hint, SESSION_KEY)
    token = _declared(hint, TASK_TOKEN_KEY)
    return RecordingIdentity(
        participant=participant or parsed.participant,
        session=session or parsed.session,
        task_id=token.lower() if token else parsed.task_id,
        sources={
            "participant": FROM_HINT if participant else FROM_STEM,
            "session": FROM_HINT if session else FROM_STEM,
            "task_id": FROM_HINT if token else FROM_STEM,
        },
    )


def recording_entity(store: ProvStore) -> Entity | None:
    """The latest live ``recording`` stream entity, the one ADMIT writes, or None.

    Args:
        store: The provenance store.

    Returns:
        The entity, or None when ADMIT wrote none.
    """
    found = [
        entity
        for entity in store.entities("stream")
        if entity.attributes.get("name") == RECORDING_STREAM and not store.is_invalidated(entity.id)
    ]
    return found[-1] if found else None


def recorded_identity(store: ProvStore) -> RecordingIdentity | None:
    """The identity ADMIT recorded, or, for a store written before it recorded one, the stem's reading.

    Args:
        store: The provenance store.

    Returns:
        The identity, or None when the store holds no ``recording`` stream.
    """
    entity = recording_entity(store)
    if entity is None:
        return None
    attributes = entity.attributes
    stem = Path(str(attributes.get("path") or "")).stem
    if DECLARED_TASK_ID not in attributes:
        return identity_from_stem(stem)
    sources = attributes.get(DECLARED_FROM)
    return RecordingIdentity(
        participant=attributes.get(DECLARED_PARTICIPANT),
        session=attributes.get(DECLARED_SESSION),
        task_id=str(attributes.get(DECLARED_TASK_ID) or UNKNOWN_TASK),
        sources=dict(sources) if isinstance(sources, Mapping) else {},
    )


def task_id_candidates(store: ProvStore, hint: AudioHints | None) -> list[str]:
    """Every task id the hint and the store carry, best carrier first, trailing index intact.

    The order is the hint's own token, then a task id ADMIT recorded from a hint, then the recording's
    stem. Where ADMIT refused the recording and wrote no stream, the stem of the ``audio_file`` its
    activity was run with stands in.

    Args:
        store: The provenance store.
        hint: The recording's hints, or None.

    Returns:
        The candidate task ids, lower-cased, in the order they should be tried.
    """
    found: list[str] = []
    token = _declared(hint, TASK_TOKEN_KEY)
    if token:
        found.append(token.lower())
    with_path = [
        entity
        for entity in store.entities("stream")
        if entity.attributes.get("name") == RECORDING_STREAM
        and entity.attributes.get("path")
        and not store.is_invalidated(entity.id)
    ]
    if with_path:
        attributes = with_path[-1].attributes
        sources = attributes.get(DECLARED_FROM)
        recorded = attributes.get(DECLARED_TASK_ID)
        if isinstance(sources, Mapping) and sources.get("task_id") == FROM_HINT and isinstance(recorded, str):
            found.append(recorded)
        found.append(task_id_of(Path(str(attributes["path"])).stem))
    else:
        admitted = [a for a in store.activities(ADMIT_NODE) if a.parameters.get("audio_file")]
        if admitted:
            found.append(task_id_of(Path(str(admitted[-1].parameters["audio_file"])).stem))
    return found


def declared_session_key(store: ProvStore, stem: str | None = None) -> str | None:
    """The session key the recording declares: a hint's, as ADMIT recorded it, else the stem's.

    Args:
        store: The provenance store.
        stem: The stem to read where no hint declared the participant or session; None reads the
            ``recording`` stream's own path.

    Returns:
        ``<participant>_<session>``, or None unless both are named.
    """
    identity = recorded_identity(store)
    if identity is not None and FROM_HINT in (identity.sources.get("participant"), identity.sources.get("session")):
        return identity.session_key
    if stem is None:
        entity = recording_entity(store)
        stem = Path(str(entity.attributes.get("path") or "")).stem if entity is not None else ""
    return session_key_of(stem)
