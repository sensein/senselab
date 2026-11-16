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
* :data:`DUPLICATE_STEM` -- one stem in more than one place. Corpus passes key runs by place and stem, so this is
  reported, not refused.
* :data:`UNKNOWN_TASK_FAMILY` -- a task whose family no branch's expectation table knows (:func:`triage_families`).
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
from senselab.audio.workflows.triage.nodes.branches import EXPECTATIONS, VOICE_EXPECTATIONS_PENDING_DECLARATION
from senselab.audio.workflows.triage.routing_analysis.families import UNKNOWN_TASK

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
UNKNOWN_TASK_FAMILY = "unknown_task_family"
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
    UNKNOWN_TASK_FAMILY: REPORT,
    STEM_DISAGREES: REPORT,
    NO_RECORDED_IDENTITY: REPORT,
}
"""Every check and its level, in the order the report lists them."""

_FIELDS = ("participant", "session", "task_id")


class IdentityRecord(NamedTuple):
    """One recording as the check reads it.

    Attributes:
        stem: The recording's file stem.
        location: Where it is: the run directory (a corpus table's ``run_dir``) or the source path.
        identity: Its declared or recorded identity, or None where its run records none.
        stem_identity: What its stem alone names (:func:`~senselab.audio.workflows.triage.identity.identity_from_stem`),
            or None to skip :data:`STEM_DISAGREES` for it.
        error: Why its identity could not be built, or None.
    """

    stem: str
    location: str
    identity: RecordingIdentity | None
    stem_identity: RecordingIdentity | None = None
    error: str | None = None


@dataclass(frozen=True)
class IdentityFinding:
    """One finding: what it is, the key it is about, and every recording it involves.

    Attributes:
        check: The check's name, a key of :data:`CHECK_LEVELS`.
        key: What the finding is about: a session label, an identity, a stem or a task family.
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


def triage_families() -> frozenset[str]:
    """Every task family triage knows: the keys of each branch's expectation table.

    Returns:
        The families of :data:`~senselab.audio.workflows.triage.nodes.branches.EXPECTATIONS` and of the VOICE
        families awaiting a declaration.
    """
    known = {family for table in EXPECTATIONS.values() for family in table}
    return frozenset(known | set(VOICE_EXPECTATIONS_PENDING_DECLARATION))


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
    known_families: Iterable[str],
) -> IdentityReport:
    """Check a corpus's identities.

    Args:
        records: One :class:`IdentityRecord` (or a tuple of its fields) per recording.
        session_scope: :data:`SESSION_SCOPES`: whether the corpus's session identifiers are unique within their
            participant or across the corpus.
        known_families: The task families triage knows (:func:`triage_families`).

    Returns:
        The report: every finding, its counts, and whether any is refused.

    Raises:
        ValueError: If ``session_scope`` is not one of :data:`SESSION_SCOPES`.
    """
    if session_scope not in SESSION_SCOPES:
        raise ValueError(f"session_scope must be one of {', '.join(SESSION_SCOPES)}; received {session_scope!r}")
    known = frozenset(known_families)
    rows = [record if isinstance(record, IdentityRecord) else IdentityRecord(*record) for record in records]

    sessions: dict[str, dict[str, list[IdentityRecord]]] = defaultdict(lambda: defaultdict(list))
    by_key: dict[tuple[str | None, ...], list[IdentityRecord]] = defaultdict(list)
    by_stem: dict[str, list[IdentityRecord]] = defaultdict(list)
    by_family: dict[str, list[IdentityRecord]] = defaultdict(list)
    findings: dict[str, list[IdentityFinding]] = {check: [] for check in CHECK_LEVELS}

    for record in rows:
        by_stem[record.stem].append(record)
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
        if identity.task_family not in known:
            by_family[identity.task_family].append(record)
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
    for stem, chosen in by_stem.items():
        if len({record.location for record in chosen}) > 1:
            findings[DUPLICATE_STEM].append(_finding(DUPLICATE_STEM, stem, chosen))
    for family, chosen in sorted(by_family.items()):
        findings[UNKNOWN_TASK_FAMILY].append(_finding(UNKNOWN_TASK_FAMILY, family, chosen))

    return IdentityReport(
        session_scope=session_scope,
        recordings=len(rows),
        findings=tuple(finding for check in CHECK_LEVELS for finding in findings[check]),
    )
