"""Who and what a recording is: its participant, session and task, declared once and recorded on the run.

A caller declares a recording's identity in its hint's ``metadata`` under :data:`PARTICIPANT_KEY`,
:data:`SESSION_KEY` and :data:`TASK_TOKEN_KEY`. ADMIT resolves it once with :func:`identity_of` and writes
it onto the ``recording`` stream entity it already creates (:meth:`RecordingIdentity.as_attributes`), so
a reader that holds only the store sees the declaration too.

Whatever the hint leaves out is read off the recording's file stem, the one place triage reads a file name
for identity: the ``task-`` entity through
:func:`~senselab.audio.workflows.triage.routing_analysis.families.task_id_of`, and the ``sub-`` and ``ses-``
entities, each on its own. A store written before ADMIT recorded an identity has none: the readers here return
None for it, and a replay records one.

Corpus passes read the same record. The corpus table (``recording_vectors``) carries each run's recorded
identity as columns (:func:`identity_columns`), a replay manifest copies them, and a pass groups its rows by
them (:func:`row_identity`, :func:`session_group`, :func:`participant_group`). A pass that opens a store
compares its row with the record (:func:`identity_disagreement`), so a stale manifest is refused rather than
grouped by an identity the run no longer records.

See ``specs/20261022-triage-dataset-profile/plan.md``, steps 1 and 3c.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

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

ROW_PARTICIPANT = "participant"
ROW_SESSION = "session"
ROW_TASK = "task"
"""The corpus-table and manifest columns a recording's identity travels in."""

ROW_SOURCES = {"participant": "participant_from", "session": "session_from", "task_id": "task_from"}
"""The column each identity field's source travels in, keyed by :class:`RecordingIdentity` field."""

RECORDING_STREAM = "recording"
ADMIT_NODE = "ADMIT"

_UNUSABLE_LABEL = re.compile(r"[/\\]|^\.{1,2}$")

SESSION_SEPARATOR = "/"
"""Joins a session key's participant and session; no label holds it (:func:`check_label`), so a key names one pair."""


class IdentityLabelError(ValueError):
    """A participant or session label that cannot name a folder or a key: empty, ``.``, ``..``, or with a slash."""


class ManifestWithoutIdentity(ValueError):
    """A corpus-table or manifest row that carries no identity columns; rebuild it from a current corpus table."""


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
        """``<participant>/<session>``, the key one recording occasion is grouped by, or None unless both are named.

        A session is identified by its participant and its label together, whatever the label's namespace: a label
        unique within its participant (BIDS ``ses-1``) and one unique across the dataset both name one occasion so.
        """
        return f"{self.participant}{SESSION_SEPARATOR}{self.session}" if self.participant and self.session else None

    def as_attributes(self) -> dict[str, Any]:
        """The attributes ADMIT writes onto the ``recording`` stream entity."""
        return {
            DECLARED_PARTICIPANT: self.participant,
            DECLARED_SESSION: self.session,
            DECLARED_TASK_ID: self.task_id,
            DECLARED_FROM: dict(self.sources),
        }


def _entity(stem: str, prefix: str) -> str | None:
    return next((chunk for chunk in stem.split("_") if chunk.startswith(prefix) and chunk != prefix), None)


def identity_from_stem(stem: str) -> RecordingIdentity:
    """The identity a file stem alone names: the ``sub-``, ``ses-`` and ``task-`` entities, each on its own.

    Args:
        stem: A recording's file stem.

    Returns:
        The identity, every field from :data:`FROM_STEM`; a field the stem does not name is None (the task
        :data:`UNKNOWN_TASK`).
    """
    return RecordingIdentity(
        participant=_entity(stem, "sub-"),
        session=_entity(stem, "ses-"),
        task_id=task_id_of(stem),
        sources={"participant": FROM_STEM, "session": FROM_STEM, "task_id": FROM_STEM},
    )


def check_label(label: str | None, what: str) -> str | None:
    """A participant or session label, refused where it could not name one folder, file or group key.

    A session key joins participant and session with :data:`SESSION_SEPARATOR`, a slash, so a label may not hold one.

    Args:
        label: The label, or None.
        what: ``participant`` or ``session``, for the message.

    Returns:
        The label, unchanged.

    Raises:
        IdentityLabelError: If the label is empty, ``.`` or ``..``, or holds a slash.
    """
    if label is not None and (not label or _UNUSABLE_LABEL.search(label)):
        raise IdentityLabelError(f"the {what} label {label!r} cannot name a folder or a key: no slash, '.' or '..'")
    return label


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

    Raises:
        IdentityLabelError: If the participant or session cannot name a folder (:func:`check_label`).
    """
    parsed = identity_from_stem(stem)
    participant = _declared(hint, PARTICIPANT_KEY)
    session = _declared(hint, SESSION_KEY)
    token = _declared(hint, TASK_TOKEN_KEY)
    return RecordingIdentity(
        participant=check_label(participant or parsed.participant, "participant"),
        session=check_label(session or parsed.session, "session"),
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


def admit_record(store: ProvStore) -> tuple[Mapping[str, Any], str] | None:
    """What ADMIT recorded about the recording, and the path it recorded it for.

    Args:
        store: The provenance store.

    Returns:
        The latest live ``recording`` stream's attributes and its ``path``; where ADMIT refused the recording
        and wrote no stream, its activity's parameters and ``audio_file``; None when ADMIT left neither.
    """
    entity = recording_entity(store)
    return admit_choice(
        entity.attributes if entity is not None else None, [a.parameters for a in store.activities(ADMIT_NODE)]
    )


def admit_choice(
    stream: Mapping[str, Any] | None, admitted: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], str] | None:
    """Which of ADMIT's records speaks for the recording, for any reader of a store.

    Args:
        stream: The latest live ``recording`` stream's attributes, or None where ADMIT wrote none.
        admitted: Every ADMIT activity's parameters, in store order.

    Returns:
        The stream's attributes and ``path``; else the last activity naming an ``audio_file``, with it; else None.
    """
    if stream is not None:
        return stream, str(stream.get("path") or "")
    named = [parameters for parameters in admitted if parameters.get("audio_file")]
    return (named[-1], str(named[-1]["audio_file"])) if named else None


def recorded_identity(store: ProvStore) -> RecordingIdentity | None:
    """The identity ADMIT recorded: on the ``recording`` stream, or on its activity where it refused the recording.

    Args:
        store: The provenance store.

    Returns:
        The identity, or None where ADMIT recorded none (a store written before it recorded identity).
    """
    record = admit_record(store)
    return identity_from_attributes(record[0]) if record is not None else None


def identity_from_attributes(attributes: Mapping[str, Any]) -> RecordingIdentity | None:
    """The identity a ``recording`` stream's attributes, or ADMIT's activity parameters, record.

    Args:
        attributes: The stream entity's attributes, or the activity's parameters.

    Returns:
        The identity, or None where they record none.
    """
    if DECLARED_TASK_ID not in attributes:
        return None
    sources = attributes.get(DECLARED_FROM)
    return RecordingIdentity(
        participant=attributes.get(DECLARED_PARTICIPANT),
        session=attributes.get(DECLARED_SESSION),
        task_id=str(attributes.get(DECLARED_TASK_ID) or UNKNOWN_TASK),
        sources=dict(sources) if isinstance(sources, Mapping) else {},
    )


def declared_task_id(store: ProvStore, stem: str = "") -> str:
    """The task id every stage reads: the one ADMIT recorded, else the stem's.

    Args:
        store: The provenance store.
        stem: The stem to read where ADMIT recorded no task (a store from before it recorded identity);
            empty reads the stem of the path ADMIT recorded (:func:`admit_record`).

    Returns:
        The lower-cased task id with its trailing index, or :data:`UNKNOWN_TASK`.
    """
    record = admit_record(store)
    if record is not None:
        attributes, path = record
        recorded = attributes.get(DECLARED_TASK_ID)
        if isinstance(recorded, str):
            return recorded
        stem = stem or Path(path).stem
    return task_id_of(stem) if stem else UNKNOWN_TASK


def declared_session_key(store: ProvStore) -> str | None:
    """The session key ADMIT recorded.

    Args:
        store: The provenance store.

    Returns:
        ``<participant>_<session>``, or None unless the recorded identity names both.
    """
    identity = recorded_identity(store)
    return identity.session_key if identity is not None else None


# ------------------------------------------------------------------------------ corpus rows


def identity_columns(identity: RecordingIdentity | None) -> dict[str, str | None]:
    """The corpus-table columns an identity is written as: each field and its source.

    Args:
        identity: The recording's identity, or None where its run records none.

    Returns:
        ``participant``, ``session``, ``task`` and one ``*_from`` column per field (:data:`ROW_SOURCES`); every
        column None for a run that records no identity.
    """
    columns: dict[str, str | None] = {
        ROW_PARTICIPANT: identity.participant if identity else None,
        ROW_SESSION: identity.session if identity else None,
        ROW_TASK: identity.task_id if identity else None,
    }
    for name, column in ROW_SOURCES.items():
        columns[column] = identity.sources.get(name) if identity else None
    return columns


def _label(value: object) -> str | None:
    return str(value) if isinstance(value, str) and value else None


def row_identity(row: Mapping[str, Any]) -> RecordingIdentity | None:
    """The identity a corpus-table or manifest row carries.

    Args:
        row: The row, carrying :func:`identity_columns`.

    Returns:
        The row's identity, or None where the row's run records none (every ``*_from`` column empty).

    Raises:
        ManifestWithoutIdentity: If the row carries none of the identity columns.
    """
    if not all(key in row for key in (ROW_PARTICIPANT, ROW_SESSION, ROW_TASK)):
        raise ManifestWithoutIdentity(
            f"row {row.get('stem')!r} carries no identity columns; rebuild the manifest from a current corpus table"
        )
    sources = {name: str(row[column]) for name, column in ROW_SOURCES.items() if _label(row.get(column))}
    if not sources:
        return None
    return RecordingIdentity(
        participant=_label(row.get(ROW_PARTICIPANT)),
        session=_label(row.get(ROW_SESSION)),
        task_id=_label(row.get(ROW_TASK)) or UNKNOWN_TASK,
        sources=sources,
    )


def session_group(identity: RecordingIdentity, stem: str) -> str:
    """The key a corpus pass groups a recording's session by: the recordings expected to share its conditions.

    Args:
        identity: The recording's identity.
        stem: Its stem.

    Returns:
        :attr:`RecordingIdentity.session_key`; a recording that does not name both participant and session is its
        own group, keyed by its stem, sharing its conditions with no other recording.
    """
    return identity.session_key or stem


def undeclared_session(participant: str) -> str:
    """The key a participant's recordings that declare no session are pooled under, for comparisons by speaker.

    It is not a session: no session-level check reads it (:func:`is_undeclared_session`).

    Args:
        participant: The participant.

    Returns:
        ``<participant>/``, which no declared session key equals.
    """
    return f"{participant}{SESSION_SEPARATOR}"


def is_undeclared_session(key: str) -> bool:
    """Whether a session key is :func:`undeclared_session`'s.

    Args:
        key: A session key.

    Returns:
        True for a participant's pool of recordings that declare no session.
    """
    return key.endswith(SESSION_SEPARATOR)


def participant_group(identity: RecordingIdentity, stem: str) -> str:
    """The key a corpus pass groups a recording's participant by.

    Args:
        identity: The recording's identity.
        stem: Its stem.

    Returns:
        The participant; a recording that names none is its own group (:func:`session_group`).
    """
    return identity.participant or session_group(identity, stem)


def row_groups(row: Mapping[str, Any]) -> tuple[str, str]:
    """The participant and session groups a corpus pass files a manifest row under.

    Args:
        row: The row, carrying :func:`identity_columns`.

    Returns:
        ``(participant, session)`` (:func:`participant_group`, :func:`session_group`); a row whose run records no
        identity is its own group in both, keyed by its stem.

    Raises:
        ManifestWithoutIdentity: If the row carries no identity columns.
    """
    stem = str(row.get("stem") or "")
    identity = row_identity(row)
    if identity is None:
        return stem, stem
    return participant_group(identity, stem), session_group(identity, stem)


def identity_disagreement(row: RecordingIdentity | None, recorded: RecordingIdentity | None) -> list[str]:
    """What a manifest row and the run's record disagree on.

    Args:
        row: The row's identity (:func:`row_identity`), or None where the row names none.
        recorded: The run's recorded identity (:func:`recorded_identity`), or None where it records none.

    Returns:
        ``participant``, ``session`` and/or ``task_id``, in that order; ``recorded identity`` where either side
        has none. Empty when they agree.
    """
    if row is None or recorded is None:
        return ["recorded identity"]
    return [name for name in ("participant", "session", "task_id") if getattr(row, name) != getattr(recorded, name)]
