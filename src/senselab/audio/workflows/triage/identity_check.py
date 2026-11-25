"""Whether a corpus's declared identities hold together, checked before a run or over its corpus table.

Every recording declares a participant, a session and a task (:mod:`~senselab.audio.workflows.triage.identity`),
and the corpus passes group runs by them. :func:`check_identities` reads one
:class:`IdentityRecord` per recording and finds what would make that grouping wrong:

* :data:`SESSION_UNDER_TWO_PARTICIPANTS` -- a session identifier named under two participants, where the identifiers
  are declared unique across the corpus (:data:`SESSION_SCOPE_GLOBAL`). Corpus passes key a session by its
  participant and its identifier together, so nothing is merged; it is reported because the passes that compare a
  speaker within a session expect one participant per session.
* :data:`DUPLICATE_IDENTITY` -- one participant, session and task id (trailing index included) declared by two
  recordings. A recording that does not name both a participant and a session is keyed by its stem instead: it is
  its own session, so repeating a task outside a session is no duplicate. Refused.
* :data:`UNREADABLE_IDENTITY` -- a recording whose identity could not be built (its hint raised, or it declares a
  label that cannot name a folder). Refused.
* :data:`DUPLICATE_RECORDING_ID` -- before a run, two recordings sharing one id (the hint's ``recording_id``, else the
  path under ``--corpus-root``, else the stem; plan step 9b). The tables and review pages join on it, so this is
  refused.
* :data:`DUPLICATE_STEM` -- after a run, one recording id under more than one run directory (a recording run twice,
  or tables concatenated across builds). Reported, not refused.
* :data:`TASK_NOT_DESCRIBED` -- a recording whose declared task the dataset's task list does not describe
  (:func:`~senselab.audio.workflows.triage.task_catalogue.resolve_task`), one finding per key: a task the list lacks,
  a task described only inline in the hint (recorded, but no branch reads an inline description yet, so each branch
  takes its out-of-family mode), a recording that declares no task, or a finished run that records none. Nothing is
  read off a file name.
* :data:`STEM_DISAGREES` -- a BIDS-like stem whose entities name another participant, session or task than the one
  declared. The declared one wins.
* :data:`NO_RECORDED_IDENTITY` -- a finished run that records no identity.

The report (:class:`IdentityReport`) lists the stems and places each finding involves, counts per check, and
whether any finding is refused; :meth:`IdentityReport.as_dict` is its JSON form. ``scripts/check_identity.py`` builds
the records from a recording list and its hints, or from a corpus table.

See ``specs/20261022-triage-dataset-profile/plan.md``, step 3c.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, NamedTuple

from senselab.audio.workflows.triage.identity import RecordingIdentity
from senselab.audio.workflows.triage.routing_analysis.families import UNKNOWN_TASK
from senselab.audio.workflows.triage.task_catalogue import FROM_HINT_DESCRIPTION, FROM_TASK_LIST, NO_TASK

SESSION_SCOPE_PARTICIPANT = "participant"
SESSION_SCOPE_GLOBAL = "global"
SESSION_SCOPES = (SESSION_SCOPE_PARTICIPANT, SESSION_SCOPE_GLOBAL)
"""The namespace a corpus's session identifiers are unique in: their participant (as BIDS ``ses-1``), or the corpus.

It is a fact about whoever minted the identifiers, so the check takes it as an argument; it changes no triage output.
"""

SESSION_UNDER_TWO_PARTICIPANTS = "session_under_two_participants"
DUPLICATE_IDENTITY = "duplicate_identity"
UNREADABLE_IDENTITY = "unreadable_identity"
DUPLICATE_STEM = "duplicate_stem"
DUPLICATE_RECORDING_ID = "duplicate_recording_id"
TASK_NOT_DESCRIBED = "task_not_described"
STEM_DISAGREES = "stem_disagrees"
NO_RECORDED_IDENTITY = "no_recorded_identity"
"""The checks, by the name a finding and the report's counts carry."""

REFUSE = "refuse"
REPORT = "report"
"""A finding's level: a refused one fails the check, a reported one is listed."""

CHECK_LEVELS: dict[str, str] = {
    SESSION_UNDER_TWO_PARTICIPANTS: REPORT,
    DUPLICATE_IDENTITY: REFUSE,
    UNREADABLE_IDENTITY: REFUSE,
    DUPLICATE_STEM: REPORT,
    DUPLICATE_RECORDING_ID: REFUSE,
    TASK_NOT_DESCRIBED: REPORT,
    STEM_DISAGREES: REPORT,
    NO_RECORDED_IDENTITY: REPORT,
}
"""Every check and its level, in the order the report lists them."""

NO_TASK_DECLARED = "(no task declared)"
NO_TASK_RECORDED = "(no task recorded)"
"""The :data:`TASK_NOT_DESCRIBED` keys of a recording whose hint names no task, and of a run from before ADMIT recorded
the task."""

HINT_ONLY_SUFFIX = " (described only in the hint)"
"""Appended to the :data:`TASK_NOT_DESCRIBED` key of a task described inline in its hint and absent from the list."""

DESCRIBED = frozenset({FROM_TASK_LIST})
"""The task sources the branches analyse a recording against: an entry of the task list. A hint's inline description
is recorded but read by no branch yet, so it is reported (:data:`HINT_ONLY_SUFFIX`)."""

_FIELDS = ("participant", "session", "task_id")


class IdentityRecord(NamedTuple):
    """One recording as the check reads it.

    Attributes:
        stem: The recording's file stem.
        location: Where it is: the run directory (a corpus table's ``run_dir``) or the source path.
        identity: Its declared or recorded identity, or None where its run records none.
        stem_identity: What its stem names, as the dataset's file-name reader reads it (senselab reads no file
            name, plan step 6a), or None to skip :data:`STEM_DISAGREES` for it.
        error: Why its identity could not be built, or None.
        task: The task its hint declares, by name, or None.
        task_source: What describes that task (``ResolvedTask.source``: the task list, the hint, neither, or no task
            declared), or None where the run records no task.
    """

    stem: str
    location: str
    identity: RecordingIdentity | None
    stem_identity: RecordingIdentity | None = None
    error: str | None = None
    task: str | None = None
    task_source: str | None = None


@dataclass(frozen=True)
class IdentityFinding:
    """One finding: what it is, the key it is about, and every recording it involves.

    Attributes:
        check: The check's name, a key of :data:`CHECK_LEVELS`.
        key: What the finding is about: a session label, an identity, a stem or a task.
        stems: The recordings' stems, one per recording involved.
        locations: Their locations, in the same order.
        detail: Anything further the check names, e.g. the participants a session falls under.
    """

    check: str
    key: str
    stems: tuple[str, ...]
    locations: tuple[str, ...]
    detail: Mapping[str, Any] = field(default_factory=dict)

    @property
    def level(self) -> str:
        """:data:`REFUSE` or :data:`REPORT`."""
        return CHECK_LEVELS[self.check]

    def as_dict(self) -> dict[str, Any]:
        """The finding's JSON form."""
        return {
            "check": self.check,
            "level": self.level,
            "key": self.key,
            "stems": list(self.stems),
            "locations": list(self.locations),
            **({"detail": dict(self.detail)} if self.detail else {}),
        }


@dataclass(frozen=True)
class IdentityReport:
    """What :func:`check_identities` found over one corpus.

    Attributes:
        session_scope: The scope the corpus was checked under.
        recordings: How many records were checked.
        findings: Every finding, grouped by check in :data:`CHECK_LEVELS` order.
    """

    session_scope: str
    recordings: int
    findings: tuple[IdentityFinding, ...]

    @property
    def counts(self) -> dict[str, int]:
        """How many findings each check made; every check is named, zero included."""
        counts = dict.fromkeys(CHECK_LEVELS, 0)
        for finding in self.findings:
            counts[finding.check] += 1
        return counts

    @property
    def recordings_involved(self) -> dict[str, int]:
        """How many distinct recordings each check's findings involve; every check is named."""
        involved: dict[str, set[tuple[str, str]]] = {check: set() for check in CHECK_LEVELS}
        for finding in self.findings:
            involved[finding.check].update(zip(finding.stems, finding.locations))
        return {check: len(each) for check, each in involved.items()}

    @property
    def refused(self) -> bool:
        """Whether any finding is refused."""
        return any(finding.level == REFUSE for finding in self.findings)

    def as_dict(self) -> dict[str, Any]:
        """The report's JSON form: scope, counts, levels, whether refused, and every finding."""
        return {
            "session_scope": self.session_scope,
            "recordings": self.recordings,
            "refused": self.refused,
            "levels": dict(CHECK_LEVELS),
            "counts": self.counts,
            "recordings_involved": self.recordings_involved,
            "findings": [finding.as_dict() for finding in self.findings],
        }


def undescribed_task(record: IdentityRecord) -> str | None:
    """The :data:`TASK_NOT_DESCRIBED` key a recording falls under, or None where its task is described.

    Args:
        record: The recording.

    Returns:
        The task's name where the task list lacks it, with :data:`HINT_ONLY_SUFFIX` where the hint describes it
        inline; :data:`NO_TASK_DECLARED` where the hint names none, :data:`NO_TASK_RECORDED` where the run records no
        task; None for a task the list describes.
    """
    if record.task_source in DESCRIBED:
        return None
    if record.task_source is None:
        return NO_TASK_RECORDED
    if record.task_source == FROM_HINT_DESCRIPTION:
        return f"{record.task or '(unnamed)'}{HINT_ONLY_SUFFIX}"
    if record.task_source == NO_TASK or not record.task:
        return NO_TASK_DECLARED
    return record.task


def identity_key(record: IdentityRecord) -> tuple[str | None, ...]:
    """The key two recordings may not share (:data:`DUPLICATE_IDENTITY`).

    Args:
        record: The recording.

    Returns:
        ``(participant, session, task_id)``; ``("stem", stem)`` for a recording whose identity does not name both a
        participant and a session, or that has none.
    """
    identity = record.identity
    if identity is None or identity.participant is None or identity.session is None:
        return ("stem", record.stem)
    return (identity.participant, identity.session, identity.task_id)


def _finding(check: str, key: str, records: Iterable[IdentityRecord], **detail: Any) -> IdentityFinding:  # noqa: ANN401
    chosen = list(records)
    return IdentityFinding(
        check=check,
        key=key,
        stems=tuple(record.stem for record in chosen),
        locations=tuple(record.location for record in chosen),
        detail=detail,
    )


def _stem_disagreement(identity: RecordingIdentity, parsed: RecordingIdentity) -> dict[str, dict[str, str | None]]:
    """The fields a stem names that the declared identity names otherwise."""
    named = {
        "participant": parsed.participant,
        "session": parsed.session,
        "task_id": None if parsed.task_id == UNKNOWN_TASK else parsed.task_id,
    }
    return {
        name: {"declared": getattr(identity, name), "stem": value}
        for name, value in named.items()
        if value is not None and getattr(identity, name) != value
    }


def check_identities(
    records: Iterable[IdentityRecord | tuple[Any, ...]],
    *,
    session_scope: str,
    before_run: bool = False,
) -> IdentityReport:
    """Check a corpus's identities.

    Args:
        records: One :class:`IdentityRecord` (or a tuple of its fields) per recording.
        session_scope: :data:`SESSION_SCOPES`: whether the corpus's session identifiers are unique within their
            participant or across the corpus.
        before_run: Whether the records are the corpus's recordings (``--recordings``), where a repeated id is two
            files colliding (:data:`DUPLICATE_RECORDING_ID`, refused), rather than finished runs, where it is one
            recording run more than once (:data:`DUPLICATE_STEM`, reported).

    Returns:
        The report: every finding, its counts, and whether any is refused.

    Raises:
        ValueError: If ``session_scope`` is not one of :data:`SESSION_SCOPES`.
    """
    if session_scope not in SESSION_SCOPES:
        raise ValueError(f"session_scope must be one of {', '.join(SESSION_SCOPES)}; received {session_scope!r}")
    rows = [record if isinstance(record, IdentityRecord) else IdentityRecord(*record) for record in records]

    sessions: dict[str, dict[str, list[IdentityRecord]]] = defaultdict(lambda: defaultdict(list))
    by_key: dict[tuple[str | None, ...], list[IdentityRecord]] = defaultdict(list)
    by_stem: dict[str, list[IdentityRecord]] = defaultdict(list)
    by_task: dict[str, list[IdentityRecord]] = defaultdict(list)
    findings: dict[str, list[IdentityFinding]] = {check: [] for check in CHECK_LEVELS}

    for record in rows:
        by_stem[record.identity.id_of(record.stem) if record.identity is not None else record.stem].append(record)
        if record.error is not None:
            findings[UNREADABLE_IDENTITY].append(
                _finding(UNREADABLE_IDENTITY, record.stem, [record], error=record.error)
            )
            continue
        by_key[identity_key(record)].append(record)
        identity = record.identity
        if identity is None:
            findings[NO_RECORDED_IDENTITY].append(_finding(NO_RECORDED_IDENTITY, record.stem, [record]))
            continue
        if identity.participant is not None and identity.session is not None:
            sessions[identity.session][identity.participant].append(record)
        undescribed = undescribed_task(record)
        if undescribed is not None:
            by_task[undescribed].append(record)
        if record.stem_identity is not None:
            disagree = _stem_disagreement(identity, record.stem_identity)
            if disagree:
                findings[STEM_DISAGREES].append(_finding(STEM_DISAGREES, record.stem, [record], fields=disagree))

    if session_scope == SESSION_SCOPE_GLOBAL:
        for session, participants in sorted(sessions.items()):
            if len(participants) > 1:
                findings[SESSION_UNDER_TWO_PARTICIPANTS].append(
                    _finding(
                        SESSION_UNDER_TWO_PARTICIPANTS,
                        session,
                        (record for name in sorted(participants) for record in participants[name]),
                        participants=sorted(participants),
                    )
                )
    for key, chosen in by_key.items():
        if len(chosen) > 1:
            label = key[1] if key[0] == "stem" else "/".join(str(part) for part in key)
            findings[DUPLICATE_IDENTITY].append(_finding(DUPLICATE_IDENTITY, str(label), chosen))
    repeated = DUPLICATE_RECORDING_ID if before_run else DUPLICATE_STEM
    for recording_id, chosen in by_stem.items():
        if len({record.location for record in chosen}) > 1:
            findings[repeated].append(_finding(repeated, recording_id, chosen))
    for task, chosen in sorted(by_task.items()):
        findings[TASK_NOT_DESCRIBED].append(_finding(TASK_NOT_DESCRIBED, task, chosen))

    return IdentityReport(
        session_scope=session_scope,
        recordings=len(rows),
        findings=tuple(finding for check in CHECK_LEVELS for finding in findings[check]),
    )
