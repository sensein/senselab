"""The triage graph's shared vocabulary and the file-level fold.

A branch reports and VERDICT decides. The three branches and QUALITY write a :class:`BranchReport` —
task conformance and typed deviations, no outcome — and the spans they proposed; every decision about
the recording is made here. The fold's rules, the two axes, the two grounds for a discard, the
agreement and hint tables and what each input contributes are in
``specs/20260817-triage-workflow-dag/verdict.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Literal, Mapping, Sequence, TypeVar

from senselab.audio.workflows.triage.decision import (
    ANNOTATION,
    DISCARD,
    PASS,
    REVIEW,
    WITHHOLD,
    EvidenceItem,
    item,
    reasons_of,
    records,
)

GRAPH_ORDER = (
    "ADMIT",
    "PREPROCESS",
    "SESSION",
    "BACKGROUND",
    "TAXONOMY",
    "routing",
    "AIRWAY",
    "SPEECH",
    "VOICE",
    "QUALITY",
    "REDACT",
    "REVIEW",
    "VERDICT",
)
"""The nodes the runner drives, in the order it drives them. VERDICT folds the twelve before it.

SESSION reads every member of the recording's BIDS session after PREPROCESS has read each one's own
floor, and BACKGROUND reads the recording against that floor before routing and the branches."""

QUALITY = "QUALITY"
"""The terminal node every recording reaches, whatever routed. A graph edge, never a branch."""

BRANCHES = ("AIRWAY", "SPEECH", "VOICE")
"""The branches routing selects among; each is the authority on its own kind and no other."""

RULESET_ROUTING = "ruleset_routing"
"""The measurement ``routing`` writes the family taxonomy ruleset's reading of a recording into.

``specs/20260912-ruleset-in-pipeline/design.md`` holds its attributes.
"""

PII_SCAN = "pii_scan"
"""The measurement SPEECH writes its PII scan's own record of itself into.

``specs/20260919-pii-against-the-stimulus/design.md`` holds the declined path's attributes.
"""

SCANNED = "scanned"
"""The :data:`PII_SCAN` key a declined scan carries, and a scan that ran does not."""

SECOND_OPINION_ANSWERS = "second_opinion_answers"
"""The measurement SECOND_OPINION writes a decision model's probabilities into.

``specs/20261003-clef-second-opinion/design.md`` holds its attributes.
"""

REDACTION_LLM_ANNOTATION = "redaction_llm_annotation"
"""The measurement REDACT writes its optional LLM re-read's summary into.

``specs/20260817-triage-workflow-dag/llm-check.md`` holds its attributes.
"""

ROUTED = "routed"
DECLINED = "declined"
UNAVAILABLE = "unavailable"
UNGATED = "ungated"
UNJUDGED = "unjudged"

BRANCH_ROUTE_STATES = (ROUTED, DECLINED, UNAVAILABLE, UNGATED, UNJUDGED)
"""What the ruleset made of one branch: a gate fired, every gate was silent, none could be read, the
branch configures no gate at all and the ruleset never looked, or ROUTING recorded no decision for
the branch and so never judged it."""

EMPTY = "empty"
UNEXPLAINED = "unexplained"
UNREADABLE = "unreadable"

FILE_ROUTE_STATES = (ROUTED, EMPTY, UNEXPLAINED, UNREADABLE)
"""What the ruleset made of the whole recording, mirroring
:class:`~senselab.audio.workflows.triage.routing_analysis.ruleset.RouteState`."""


class Outcome(Enum):
    """What a node concluded."""

    PASS = "pass"
    FLAG = "flag"
    FAIL = "fail"


class Triage(Enum):
    """What should happen to this recording. The file axis; a node's ``Outcome`` is not one of these."""

    PASS = "pass"
    REVIEW = "review"
    DISCARD = "discard"


class RunStatus(Enum):
    """Whether the pipeline finished what this recording's decision needs.

    ``incomplete`` names a recording the pipeline still owes something -- a missing derivative, a node
    that did not finish, a configuration the fold cannot read; :data:`OPERATIONAL_GROUND_KEYS` names
    those grounds. Its verdict is :attr:`Triage.REVIEW` with the reason ``not_measured``.
    """

    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class KindState(Enum):
    """Whether a kind is in the recording."""

    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"


class RunState(Enum):
    """Whether a node ran at all."""

    COMPLETED = "completed"
    SKIPPED = "skipped"
    ERRORED = "errored"


class Release(Enum):
    """Which artefact of this recording may be handed on.

    Three values over one question, read only where the verdict is not ``discard``: the recording as
    recorded, only REDACT's redacted copy, or neither because the redaction policy holds it. A discard
    carries no release value, and neither does a recording whose release the graph could not assess
    (its verdict is ``review``, reason ``not_measured``).
    ``specs/20260924-which-artefact-is-releasable/design.md`` holds the vocabulary and what each
    value permits; ``specs/20260817-triage-workflow-dag/verdict.md`` holds the fold.
    """

    AS_IS = "as_is"
    REDACTED = "redacted"
    WITHHELD = "withheld"


NO_LEXICAL_WORD = "SPEECH ran and the consensus transcript carries no lexical word"
NOTHING_BEYOND_STIMULUS = "no lexical word lies outside what the task asked for, so the scan was declined"
SCAN_FOUND_NOTHING = "the scan ran over the transcript and found nothing to redact"
NON_LEXICAL_TASK = "the ruleset declined SPEECH because the task carries no lexical content"
REVIEWER_UNMASKED_ALL = (
    "the redaction reviewer unmasked the words it named and no content word was left masked, "
    "so the original is released"
)
NO_CONTENT_MASKED = "every mask REDACT planned hid only non-content words, so the original is released"
FINDINGS_ARE_TASK_CONTENT = (
    "every finding REDACT read is content the task itself declares, so REDACT masked nothing and the original "
    "is released"
)
REVIEWER_CLEARED_UNMASKED = (
    "REDACT's re-scan still read a finding it had planned no mask over, and the reviewer read the original as "
    "clean and proposed nothing to hide, so the original is released"
)

RELEASE_WITHOUT_REDACTION_GROUNDS = (
    NO_LEXICAL_WORD,
    NOTHING_BEYOND_STIMULUS,
    SCAN_FOUND_NOTHING,
    NON_LEXICAL_TASK,
    REVIEWER_UNMASKED_ALL,
    NO_CONTENT_MASKED,
    FINDINGS_ARE_TASK_CONTENT,
    REVIEWER_CLEARED_UNMASKED,
)
"""Which reading cleared the recording. One stands behind every :attr:`Release.AS_IS`."""

NO_TRANSCRIPT = "SPEECH did not run, so nothing read the recording for content a redaction would remove"
SPEECH_UNREAD = "SPEECH left no lexical count, so whether the recording carries redactable content is unknown"
REDACTION_OWED = "the scan found content to redact and REDACT left no verdict over it"
SCAN_UNRECORDED = "SPEECH read lexical words and recorded no scan either way"

RELEASE_UNKNOWN_GROUNDS = (NO_TRANSCRIPT, SPEECH_UNREAD, REDACTION_OWED, SCAN_UNRECORDED)
"""Why the graph could not tell. One of these stands behind every release the graph could not assess."""

REVIEWER_PROPOSED_REDACTION = "the redaction reviewer proposed hiding more and the policy lets that withhold"

UNPLACED_PLACED = "placed"
"""A detector finding the fold could not place, placed by a reviewer ``redact`` entry of its family."""

UNPLACED_CLEARED = "cleared"
"""A detector finding the fold could not place, where the reviewer read the original as clean."""

UNPLACED_OPEN = "open"
"""A detector finding the fold could not place, which a reading neither placed nor cleared."""

UNPLACED_UNREAD = "unread"
"""A detector finding the fold could not place, where no reviewer read the recording."""

DOMINANT_SPEAKER_GATE = "dominant_speaker_share_min"
"""The gate whose failure is diarization's own reading of another speaker in the task extent."""

UNPLACED_FINDING_UNREAD = (
    "a detector finding could not be placed on the transcript's words and no reviewer read the recording"
)

REDACT_VERIFY_FOUND = (
    "REDACT's re-scan of its redacted transcript still read identifying content and no reading cleared it"
)

REDACT_UNRESOLVED = "REDACT did not resolve its redaction, so neither copy may be handed on"

RELEASE_WITHHELD_GROUNDS = (
    REVIEWER_PROPOSED_REDACTION,
    UNPLACED_FINDING_UNREAD,
    REDACT_VERIFY_FOUND,
    REDACT_UNRESOLVED,
)
"""Why the redaction policy withholds a recording. One stands behind every :attr:`Release.WITHHELD`."""

REVIEWER_CLEARED_RESCAN = (
    "REDACT's re-scan still read a finding, and the reviewer read the original as clean and proposed nothing to hide"
)

REVIEWER_UNMASKED_SOME = (
    "the redaction reviewer unmasked the words it named; the copy keeps the content words still masked"
)

MASKS_TRIMMED_TO_CONTENT = "REDACT's masks were trimmed to the content words they hide; the copy keeps those"

POLICY_MASKS_ADDED = (
    "the redaction policy masked words it always masks -- a date element, an age, a state -- beside REDACT's "
    "masks; the copy keeps those"
)

POLICY_MASKS_ONLY = (
    "the scan found nothing to redact, and the redaction policy masked words it always masks -- a date element, "
    "an age, a state; the copy keeps those"
)
DISCARDED = "the recording was discarded, so no artefact of it is released"
"""The release ground of every discard; a discard carries no release value."""

RELEASE_WITH_REDACTION_GROUNDS = (
    REVIEWER_CLEARED_RESCAN,
    REVIEWER_UNMASKED_SOME,
    MASKS_TRIMMED_TO_CONTENT,
    POLICY_MASKS_ADDED,
    POLICY_MASKS_ONLY,
)
"""Why a redacted copy is released other than as REDACT itself planned and passed it."""


@dataclass(frozen=True)
class RedactionEvidence:
    """What the store says about whether this recording carried anything a redaction could remove.

    Attributes:
        lexical_words_n: How many lexical words SPEECH read off the consensus transcript, or None
            where it left no report to say.
        scanned: True where SPEECH scanned the transcript, False where it declined to, and None
            where it recorded no scan either way.
        findings_n: How many live ``pii`` findings the store holds.
        rescan_survivors: The categories REDACT's re-scan still read after its one re-plan, its
            ``unremediable``. Empty where REDACT did not run, passed, or could not complete a scan.
        masks_n: How many masks REDACT planned over the recording.
        masks_final_n: How many masks stand once the reviewer's unmasks and the content-word trim
            are applied; see :func:`~senselab.audio.workflows.triage.nodes.redact.mask_plan`.
        masks_changed: Whether any planned mask lost a word.
        reviewer_unmasked_n: How many words the reviewer's ``release`` entries unmasked.
        policy_masks_n: How many masks the redaction policy itself placed, over words no detector
            finding covered.
        person_names_masked_n: How many words of a person's name stay masked.
        name_release_proposed: The reviewer's ``release`` quotes naming a person's name it may not
            release on its own.
        reviewer_requested_n: How many reviewer ``redact`` entries propose hiding more than the masks hide.
    """

    lexical_words_n: int | None = None
    scanned: bool | None = None
    findings_n: int = 0
    rescan_survivors: tuple[str, ...] = ()
    masks_n: int = 0
    masks_final_n: int = 0
    masks_changed: bool = False
    reviewer_unmasked_n: int = 0
    policy_masks_n: int = 0
    person_names_masked_n: int = 0
    name_release_proposed: tuple[str, ...] = ()
    reviewer_requested_n: int = 0


@dataclass(frozen=True)
class TaskEvidence:
    """What the store says about whether the declared task was performed at all.

    Attributes:
        owning_branches: The branches whose expectations own the declared family; empty where the
            recording declares no family, or one no branch owns.
        duration_s: The recording's duration, or None where ADMIT recorded none.
        minimum_duration_s: The shortest recording the declared family can occupy, or None.
        event_tokens_n: How many bracketed ASR event tokens name the declared airway family's own
            event (``[cough]`` in a cough task); 0 for every other family.
        owner_absent_inputs: The instruments, derivatives or gate readings an owning branch needed to
            look for the task and did not have; empty where every owner could look.
        required_event: The event kind the declared airway family is decided on (``breath``), from
            ``data/airway_event_requirements.yaml``; None for every other family.
        events_found_n: How many events of the family's own kind the owning AIRWAY branch detected,
            or None where it reported no count.
        event_kind: The declared airway family's own event kind, or None.
        instructed_count: The count the declared airway family's instruction spoke, or None.
        breath_mode: ``sustained`` or ``counted`` for a breath family decided on the breathing-pattern
            measure (``data/breath_pattern.yaml``); None for every other family.
        breath_pattern: The measure's pattern, one of ``breath_pattern.PATTERNS``, or None where it
            could not be read.
        breath_events_n: How many breath events the measure found, or None.
        breath_vetoed_by: The veto that says what the measure found is not breathing, one of
            ``breath_pattern.VETO_*`` (``breath_pattern.breath_veto_of``), or None.
        breath_train_breaths: The breaths the task evidence's dominant cluster counts (its events over
            two, rounded half up), or None where it was not read.
        breath_decision: The task evidence's decision, ``present``, ``review`` or ``absent``
            (``task_events.decide``), or None where it was not read.
        breath_review: Whether the task evidence leaves the breath reading for review.
        breath_reading: The measure's full reading, for the verdict record; empty where none.
        cough_mode: ``counted`` or ``performed`` for a cough family decided on the cough-onset measure
            (``data/cough_pattern.yaml``), by whether its instruction speaks a count; None otherwise.
        cough_onsets_n: How many coughs the measure found, or None where it could not be read.
        cough_review: Whether the decision differs inside the measure's review band
            (``data/cough_pattern.yaml``, ``review``).
        cough_reading: The measure's full reading, for the verdict record; empty where none.
        quality: QUALITY's join of the background against the task spans
            (``nodes.quality.measure_join``); empty where QUALITY wrote none.
        contest_events_min: The events AIRWAY's own detector must have found for a measure's
            no-event discard to be contested instead (``data/discard_contested.yaml``); None where
            no discard of this family is contested.
        contest_rise_db_min: The breath-train rise a breath family's no-event discard needs to be
            contested (``data/discard_contested.yaml``, ``breath``); None for any other family.
        breath_train_rise_db: The breath train's median burst rise over the floor, or None.
        voice_mode: ``sustained`` or ``glide`` for a declared voice family decided on VOICE's
            phonation reading; None for every other family.
        voice_found: Whether the reading found a phonation attempt, or None where it was not read.
        voice_mismatch: The glide's ``task_mismatch`` description, or None.
        voice_review: Why the reading is left for review; empty where it is not.
        voice_outside_speech: The speech-like runs read outside the extent.
        voice_reading: The reading's record, for the verdict record; empty where none.
        ddk_mode: ``syllable`` or ``cycle`` for a declared syllable-repetition family decided on SPEECH's
            task-layer reading (``ddk_task``); None for every other family.
        ddk_decision: The reading's decision, ``present``, ``review`` or ``absent``, or None where it
            was not read.
        ddk_reading: The reading's record, for the verdict record; empty where none.
    """

    owning_branches: tuple[str, ...] = ()
    duration_s: float | None = None
    minimum_duration_s: float | None = None
    event_tokens_n: int = 0
    owner_absent_inputs: tuple[str, ...] = ()
    required_event: str | None = None
    events_found_n: int | None = None
    event_kind: str | None = None
    instructed_count: int | None = None
    breath_mode: str | None = None
    breath_pattern: str | None = None
    breath_events_n: int | None = None
    breath_vetoed_by: str | None = None
    breath_train_breaths: int | None = None
    breath_decision: str | None = None
    breath_review: bool = False
    breath_reading: dict[str, Any] = field(default_factory=dict)
    cough_mode: str | None = None
    cough_onsets_n: int | None = None
    cough_review: bool = False
    cough_reading: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    contest_events_min: int | None = None
    contest_rise_db_min: float | None = None
    breath_train_rise_db: float | None = None
    voice_mode: str | None = None
    voice_found: bool | None = None
    voice_mismatch: str | None = None
    voice_review: tuple[str, ...] = ()
    voice_outside_speech: tuple[dict[str, Any], ...] = ()
    voice_reading: dict[str, Any] = field(default_factory=dict)
    ddk_mode: str | None = None
    ddk_decision: str | None = None
    ddk_reading: dict[str, Any] = field(default_factory=dict)


UNMEASURABLE = "unmeasurable"
ACOUSTICALLY_EMPTY = "acoustically_empty"
TOO_SHORT_FOR_TASK = "too_short_for_task"
"""Discard ground: the recording is far shorter than its declared task can take, a truncated or aborted capture."""
NO_BREATH_CAPTURED = "no_breath_captured"
"""Discard ground: the breath train of a breath task counted no breath."""
BREATH_SUSTAINED = "sustained"
BREATH_COUNTED = "counted"


def breath_present(evidence: TaskEvidence) -> bool:
    """Whether a breath family's task events found breathing.

    Args:
        evidence: The task evidence.

    Returns:
        True where the task evidence decided ``present`` or ``review``; False where it decided
        ``absent`` or was not read.
    """
    return evidence.breath_decision in ("present", "review")


def breath_conforms(evidence: TaskEvidence) -> bool:
    """Whether a breath family's measure says the task was performed as instructed.

    Args:
        evidence: The task evidence.

    Returns:
        :func:`breath_present`, and for a counted family at least the instructed number of breath-train
        breaths.
    """
    if not breath_present(evidence):
        return False
    if evidence.breath_mode == BREATH_COUNTED and evidence.instructed_count is not None:
        return (evidence.breath_train_breaths or 0) >= evidence.instructed_count
    return True


def breath_shortfall(evidence: TaskEvidence) -> str | None:
    """The detected-against-instructed count where a counted breath family found too few breaths.

    Args:
        evidence: The task evidence.

    Returns:
        ``"detected B breaths (P phases in the breath task) where M were instructed"`` where breaths
        were present but too few were counted (:func:`breath_conforms`); None otherwise.
    """
    breaths = evidence.breath_train_breaths or 0
    if (
        evidence.breath_mode != BREATH_COUNTED
        or evidence.instructed_count is None
        or not breath_present(evidence)
        or breaths >= evidence.instructed_count
    ):
        return None
    read = evidence.breath_reading.get("evidence") or {}
    return (
        f"detected {breaths} breaths ({read.get('events_n', 0)} phases in the breath task) "
        f"where {evidence.instructed_count} were instructed"
    )


def breath_review_reading(evidence: TaskEvidence) -> str:
    """The task evidence a low-confidence breath review flag carries.

    Args:
        evidence: The task evidence.

    Returns:
        ``"N events, best SNR s dB (why)"``, with ``none`` for a reading absent.
    """
    read = evidence.breath_reading.get("evidence") or {}
    return (
        f"{read.get('events_n', 0)} events, best SNR {read.get('best_snr_db', 'none')} dB ({read.get('why', 'none')})"
    )


COUGH_COUNTED = "counted"
COUGH_PERFORMED = "performed"


def cough_present(evidence: TaskEvidence) -> bool:
    """Whether a cough family's measure found at least one cough.

    Args:
        evidence: The task evidence.

    Returns:
        True where the measure was read and found a cough onset.
    """
    return evidence.cough_mode is not None and (evidence.cough_onsets_n or 0) > 0


def cough_conforms(evidence: TaskEvidence) -> bool:
    """Whether a cough family's measure says the task was performed as instructed.

    Args:
        evidence: The task evidence.

    Returns:
        :func:`cough_present`, and for a counted family at least the instructed number of coughs.
    """
    if not cough_present(evidence):
        return False
    if evidence.cough_mode == COUGH_COUNTED and evidence.instructed_count is not None:
        return (evidence.cough_onsets_n or 0) >= evidence.instructed_count
    return True


def cough_shortfall(evidence: TaskEvidence) -> str | None:
    """The detected-against-instructed count where a counted cough family found too few coughs.

    Args:
        evidence: The task evidence.

    Returns:
        ``"detected N coughs where M were instructed"`` where coughs were found but fewer than the
        instruction asked; None otherwise.
    """
    if evidence.cough_mode != COUGH_COUNTED or not cough_present(evidence) or cough_conforms(evidence):
        return None
    return f"detected {evidence.cough_onsets_n} coughs where {evidence.instructed_count} were instructed"


def cough_review_reading(evidence: TaskEvidence) -> str:
    """The cough readings a low-confidence review flag carries.

    Args:
        evidence: The task evidence.

    Returns:
        ``"N coughs (S strict, L lenient)"``.
    """
    reading = evidence.cough_reading
    return (
        f"{evidence.cough_onsets_n} coughs ({reading.get('onsets_strict_n', 'none')} strict, "
        f"{reading.get('onsets_lenient_n', 'none')} lenient)"
    )


def ddk_present(evidence: TaskEvidence) -> bool:
    """Whether a syllable-repetition family's task events found the train.

    Args:
        evidence: The task evidence.

    Returns:
        True where the reading decided ``present`` or ``review``.
    """
    return evidence.ddk_decision in ("present", "review")


def ddk_shortfall(evidence: TaskEvidence) -> str | None:
    """The events found against the instructed count, where a counted syllable family found fewer.

    Args:
        evidence: The task evidence.

    Returns:
        ``"detected N <unit>s where M were instructed"`` where the train was present and fewer task
        events than the instruction asked were read; None otherwise.
    """
    notes = evidence.ddk_reading.get("annotations") or {}
    found, required = notes.get("events_n"), notes.get("required_count")
    if not ddk_present(evidence) or found is None or required is None or found >= required:
        return None
    return f"detected {found} {evidence.ddk_mode}s where {required} were instructed"


def ddk_review_reading(evidence: TaskEvidence) -> str:
    """The task-layer readings a syllable-train review flag carries.

    Args:
        evidence: The task evidence.

    Returns:
        ``"N events, C clear, identity I (why)"``.
    """
    reading = evidence.ddk_reading
    inputs = reading.get("inputs") or {}
    return (
        f"{reading.get('events_n', 0)} {evidence.ddk_mode} events, {inputs.get('clear_free_n', 0)} clear, "
        f"identity {reading.get('identity', 'none')} ({reading.get('why', 'none')})"
    )


DECLARED_TASK_ABSENT = "declared_task_absent"
"""Discard ground: the branch owning the declared task ran and found none of it, whatever another branch found."""
NO_COUGH_CAPTURED = "no_cough_captured"
"""Discard ground: a cough task holds no cough onset."""
NO_PHONATION_CAPTURED = "no_phonation_captured"
"""Discard ground: a voice task holds nothing over the noise floor."""
NO_SYLLABLE_TRAIN_CAPTURED = "no_syllable_train_captured"
"""Discard ground: a syllable-repetition task holds no syllable over the noise floor."""
SYLLABLE_TRAIN_NOT_TARGET = "syllable_train_not_target"
"""Discard ground: a syllable-repetition task's events are another activity, not the declared syllables."""
DISCARD_GROUNDS = (
    UNMEASURABLE,
    TOO_SHORT_FOR_TASK,
    ACOUSTICALLY_EMPTY,
    NO_BREATH_CAPTURED,
    NO_COUGH_CAPTURED,
    NO_PHONATION_CAPTURED,
    NO_SYLLABLE_TRAIN_CAPTURED,
    SYLLABLE_TRAIN_NOT_TARGET,
    DECLARED_TASK_ABSENT,
)
"""Every ground a file discards on, in the order the fold tries them; an operational flag (``rerun``) is
tried between the third and the fourth, since a missing derivative may be why the task was not found."""
EVENT_ABSENT_GROUNDS = {"breath": NO_BREATH_CAPTURED, "cough": NO_COUGH_CAPTURED, "phonation": NO_PHONATION_CAPTURED}
"""The discard ground for each required event kind of ``data/airway_event_requirements.yaml``."""
TASK_MISMATCH = "task_mismatch"
"""Flag ground: the declared airway task's own events were detected, but not in the pattern it asked for."""
BREATH_REVIEW_LOW_CONFIDENCE = "breath_review_low_confidence"
"""Flag ground: a breath task's breathing was kept, on a breath train weak or irregular enough to review."""
COUGH_REVIEW_LOW_CONFIDENCE = "cough_review_low_confidence"
"""Flag ground: a cough task's coughs were kept, on a count whose decision differs inside the review band."""
VOICE_REVIEW_LOW_CONFIDENCE = "voice_review_low_confidence"
"""Flag ground: a voice task's phonation reading differs inside its review band."""
DDK_REVIEW_WEAK_EVENTS = "ddk_review_weak_events"
"""Flag ground: a syllable task's train stands, but too few of its events are clear of the background."""
DDK_REVIEW_IDENTITY = "ddk_review_identity"
"""Flag ground: a syllable task's train stands, but its identity against the template is ambiguous."""
STREAMS_DISAGREE = "streams_disagree"
"""Flag ground: the enhanced stream loses the task events the raw stream stands on."""
SPEECH_OUTSIDE_TASK = "speech_outside_task"
"""Annotation: speech-like runs outside a voice task's phonation extent, with the words over them."""
DISCARD_CONTESTED = "discard_contested"
"""Flag ground: an airway measure found none of its event, and AIRWAY's own event detector found the task's."""

AGREE = "agree"
MISMATCH = "mismatch"
RESOLVED = "resolved"
NOT_RUN = "not_run"

CLAIMED_AND_FOUND = "claimed_and_found"
CLAIMED_NOT_FOUND = "claimed_not_found"
FOUND_UNCLAIMED = "found_unclaimed"
NO_CLAIM = "no_claim"

UNDETERMINED: Literal["UNDETERMINED"] = "UNDETERMINED"
"""A conformance question that was not answered: nothing asked one, or nothing could answer it."""

Conformance = bool | Literal["UNDETERMINED"]
"""Whether what was asked for happened. Mirrors ``branches.Done``, which is where it comes from."""

TASK = "task"
"""A conformance about the instruction the recording declares. What the three branches report."""

STORE_ASSERTIONS = "store_assertions"
"""A conformance about the store's own records. What QUALITY reports; never about a participant."""

CONFORMANCE_REFERENTS = (TASK, STORE_ASSERTIONS)
"""What a conformance can be about. Named so a corpus count never mixes the two."""

TASK_NOT_CONFORMED = "reported that what the instruction asked for did not happen"
NOTHING_READ = "read none of the words the stimulus asked for"
"""The conformance ground for a read-aloud task whose alignment realised no expected token at all."""

UNCOMPUTED_READING = "a reading this task is judged on was not computed"
"""The flag ground an undecided conformance gate contributes when its reading should exist and does not.

Controlled vocabulary, with each gate and its reason appended."""
CONFORMANCE_UNANSWERED = "answered no conformance question"
UNMEASURED_ASKED = "asked for an operating point nobody has measured"
STORE_ASSERTION_CONTRADICTED = "reported that a stored assertion the same store's measurements contradict"

_SECTION = "verdict"
"""The config section :class:`FoldPolicy` reads. Every key in it is this fold's, not a branch's."""

_ADMIT = "ADMIT"
_PREPROCESS = "PREPROCESS"
_REDACT = "REDACT"
_SPEECH = "SPEECH"
_AIRWAY = "AIRWAY"
_VOICE = "VOICE"
_VERDICT = "VERDICT"
_ROUTING = "routing"

BAD_MAP_VALUES = "routing.hint_branch_map names a branch this graph does not route to"

UNEXPLAINED_CONTENT = (
    "no branch routed and the recording was not measurably empty; the ruleset could not account for what is in it"
)

UNREADABLE_EMPTINESS = (
    "no branch routed and the emptiness bypass could not be read; whether the recording carried anything is unknown"
)

UNREAD_DECLARATION = (
    "a declaration was supplied and no branch decision survived to read it against; "
    "what it claimed is unknown, not empty"
)

CRITICAL_ABSENCE = "a critical measurement is absent, so no gate of at least one branch could be read"
OWNING_BRANCH_INPUT_ABSENT = (
    "the branch owning the declared task lacked an input it needs to look for the task, "
    "so whether the task was performed is unknown"
)
NO_LEXICAL_ITEM_PRODUCED = "SPEECH ran over a task that asks for words and read no lexical item"
"""The flag ground a critical failure contributes, with the branch, gate and recorded absence appended.

Never a discard ground. See ``specs/20260817-triage-workflow-dag/critical-failure.md``.
"""

UNPLACED_FINDING_OPEN = (
    "a detector finding could not be placed on the transcript's words, and the reviewer neither placed it "
    "nor read the original as clean"
)
"""The flag ground an unplaced finding contributes where a reading exists and settles nothing about it.

Controlled vocabulary, with the families of the unplaced findings appended.
"""

REVIEWER_NAMED_NO_WORDS = "the redaction reviewer judged the redaction wrong but named no words"
"""The flag ground a flagged reading with no proposal entries contributes, under
``verdict.llm_contradiction_flags``.

A reading may only move a mask by naming the words it moves, so such a reading moves none: REDACT's
masks stand, and the recording goes to a person."""

INSTRUCTIONS_SPOKEN = "the task's instructions are spoken in the recording, by the examiner or the participant"
"""The flag ground where the reviewer quoted passages in which the task's own instructions are spoken,
verbatim or paraphrased; who spoke them is left to a person. ``verdict.llm_instructions_spoken_flags``."""

REVIEWER_HEARD_SECOND_SPEAKER = "the redaction reviewer read another speaker in the transcript"
"""The flag ground a reading's ``speakers: more_than_one`` contributes, under ``verdict.llm_second_speaker_flags``,
where diarization's own gate has not already flagged another speaker in the task extent. Any other voice
flags, whether or not the task's instructions expect it; the ground names each quote and whether the
instructions expect that voice, or says no words were quoted."""

PERSON_NAME_AWAITS_REVIEW = "a person's name stays masked until a person approves its release"
"""The flag ground a masked person's name contributes, under ``verdict.person_name_review_flags``: the reviewer
may propose releasing a public figure's name, and only a human's approval (``redaction.name_approvals``)
releases it. Controlled vocabulary, with the number of masked name words and any proposed release appended;
the release is unchanged."""

SECOND_OPINION_DISAGREES = "the second-opinion model confidently disagrees with the redaction reviewer"
"""The flag ground a decision model's confident disagreement with the reviewer contributes, under
``verdict.second_opinion_disagreement_flags``. Controlled vocabulary, with each disagreeing question, the
model's probability and the reviewer's answer appended; the release is unchanged."""

SECOND_OPINION_QUESTIONS = ("other_voice", "instructions_spoken", "policy_identifier_present")
"""The questions whose disagreement with the reviewer can flag a recording."""

MODEL_SPEAKER_PERMITTED = "the task's instructions permit a model speaker"
"""Appended to the speaker gate's flag ground for a family in ``verdict.model_speaker_families``: the
instructions let someone say the sentence first, which explains the other voice without excusing it."""

LLM_REDACTION_RESIDUE = "the redaction reviewer flagged residue on the redacted transcript"
"""The flag ground a reviewer reading that proposes hiding more contributes to the triage axis.

The same test as the release axis, :func:`_reviewer_found_residue`: a reading that only releases or
proposes nothing contributes no ground. Controlled vocabulary, with the categories of the proposed
redactions appended; the substrings and the reasoning stay in the store's ``redaction_llm_review``
measurements.
"""

EXTRA_SPEAKER_IN_EXTENT = "another speaker holds part of the task extent"
"""The flag ground the multi-speaker gate contributes to the triage axis.

Controlled vocabulary, with the gate's reading and bound appended. It is
``gates.FLAG_GATES["dominant_speaker_share_min"]``, spelled here because the fold names its own
grounds and imports no gate table.
"""


# Ground keys: one stable, machine-readable key per ground, beside the human-readable ``why``.
KEY_UNMEASURABLE = UNMEASURABLE
KEY_ACOUSTICALLY_EMPTY = ACOUSTICALLY_EMPTY
KEY_TOO_SHORT_FOR_TASK = TOO_SHORT_FOR_TASK
KEY_DECLARED_TASK_ABSENT = DECLARED_TASK_ABSENT
KEY_NO_BREATH_CAPTURED = NO_BREATH_CAPTURED
KEY_TASK_MISMATCH = TASK_MISMATCH
KEY_BREATH_REVIEW_LOW_CONFIDENCE = BREATH_REVIEW_LOW_CONFIDENCE
KEY_NO_COUGH_CAPTURED = NO_COUGH_CAPTURED
KEY_COUGH_REVIEW_LOW_CONFIDENCE = COUGH_REVIEW_LOW_CONFIDENCE
KEY_DISCARD_CONTESTED = DISCARD_CONTESTED
KEY_NO_PHONATION_CAPTURED = NO_PHONATION_CAPTURED
KEY_VOICE_REVIEW_LOW_CONFIDENCE = VOICE_REVIEW_LOW_CONFIDENCE
KEY_NO_SYLLABLE_TRAIN_CAPTURED = NO_SYLLABLE_TRAIN_CAPTURED
KEY_SYLLABLE_TRAIN_NOT_TARGET = SYLLABLE_TRAIN_NOT_TARGET
KEY_DDK_REVIEW_WEAK_EVENTS = DDK_REVIEW_WEAK_EVENTS
KEY_DDK_REVIEW_IDENTITY = DDK_REVIEW_IDENTITY
KEY_STREAMS_DISAGREE = STREAMS_DISAGREE
KEY_SPEECH_OUTSIDE_TASK = SPEECH_OUTSIDE_TASK
KEY_PREPROCESS_ERRORED = "preprocess_errored"
KEY_ROUTING_ERRORED = "routing_errored"
KEY_BAD_HINT_MAP = "config_bad_hint_map"
KEY_DECLARATION_UNREAD = "declaration_unread"
KEY_ROUTE_UNEXPLAINED = "route_unexplained"
KEY_ROUTE_UNREADABLE = "route_unreadable"
KEY_CRITICAL_ABSENCE = "critical_absence"
KEY_TAXONOMY_NO_CLASSIFIER = "taxonomy_no_classifier"
KEY_REDACT_RESCAN_INCOMPLETE = "redact_rescan_incomplete"
KEY_UNCOMPUTED_READING = "uncomputed_reading"
KEY_OWNING_BRANCH_INPUT_ABSENT = "owning_branch_input_absent"
KEY_NODE_OUTCOME_UNREADABLE = "node_outcome_unreadable"
KEY_NO_LEXICAL_ITEM = "speech_no_lexical_item"
KEY_REVIEWER_RESIDUE = "reviewer_residue"
KEY_PERSON_NAME_REVIEW = "person_name_review"
KEY_REVIEWER_SECOND_SPEAKER = "reviewer_second_speaker"
KEY_REVIEWER_NAMED_NO_WORDS = "reviewer_named_no_words"
KEY_INSTRUCTIONS_SPOKEN = "instructions_spoken"
KEY_SECOND_OPINION_DISAGREES = "second_opinion_disagreement"
KEY_UNPLACED_OPEN = "unplaced_finding_open"
KEY_UNPLACED_UNREAD = "unplaced_finding_unread"

GROUND_KEYS = (
    KEY_UNMEASURABLE,
    KEY_ACOUSTICALLY_EMPTY,
    KEY_TOO_SHORT_FOR_TASK,
    KEY_DECLARED_TASK_ABSENT,
    KEY_NO_BREATH_CAPTURED,
    KEY_TASK_MISMATCH,
    KEY_BREATH_REVIEW_LOW_CONFIDENCE,
    KEY_NO_COUGH_CAPTURED,
    KEY_COUGH_REVIEW_LOW_CONFIDENCE,
    KEY_DISCARD_CONTESTED,
    KEY_NO_PHONATION_CAPTURED,
    KEY_VOICE_REVIEW_LOW_CONFIDENCE,
    KEY_NO_SYLLABLE_TRAIN_CAPTURED,
    KEY_SYLLABLE_TRAIN_NOT_TARGET,
    KEY_DDK_REVIEW_WEAK_EVENTS,
    KEY_DDK_REVIEW_IDENTITY,
    KEY_STREAMS_DISAGREE,
    KEY_SPEECH_OUTSIDE_TASK,
    KEY_PREPROCESS_ERRORED,
    KEY_ROUTING_ERRORED,
    KEY_BAD_HINT_MAP,
    KEY_DECLARATION_UNREAD,
    KEY_ROUTE_UNEXPLAINED,
    KEY_ROUTE_UNREADABLE,
    KEY_CRITICAL_ABSENCE,
    KEY_TAXONOMY_NO_CLASSIFIER,
    KEY_REDACT_RESCAN_INCOMPLETE,
    KEY_UNCOMPUTED_READING,
    KEY_OWNING_BRANCH_INPUT_ABSENT,
    KEY_NODE_OUTCOME_UNREADABLE,
    KEY_NO_LEXICAL_ITEM,
    KEY_REVIEWER_RESIDUE,
    KEY_PERSON_NAME_REVIEW,
    KEY_REVIEWER_SECOND_SPEAKER,
    KEY_REVIEWER_NAMED_NO_WORDS,
    KEY_INSTRUCTIONS_SPOKEN,
    KEY_SECOND_OPINION_DISAGREES,
    KEY_UNPLACED_OPEN,
    KEY_UNPLACED_UNREAD,
)
"""Every ground key that names no node, gate or branch of its own."""

PREFIX_GATE = "gate"
PREFIX_CONFORMANCE = "conformance"
PREFIX_NOTHING_READ = "nothing_read"
PREFIX_STORE_ASSERTION = "store_assertion_contradicted"
PREFIX_CONFORMANCE_UNANSWERED = "conformance_unanswered"
PREFIX_DEVIATION = "deviation"
PREFIX_UNMEASURED = "unmeasured_operating_point"
PREFIX_ROUTE_MISMATCH = "route_mismatch"
PREFIX_BRANCH_SILENT = "branch_silent"
PREFIX_HINT_MISMATCH = "hint_mismatch"
PREFIX_NODE = "node"
PREFIX_FAULT_IN_TASK = "fault_in_task"
"""A capture fault (``fault_in_task:<kind>``) touching a task span: a review ground."""
PREFIX_FAULT_OUTSIDE_TASK = "fault_outside_task"
"""A capture fault away from every task span, or one QUALITY's parameters do not let decide: an annotation."""
PREFIX_INTERFERENCE_IN_TASK = "interference_in_task"
"""Another source (``interference_in_task:<kind>``) touching a task span: a review ground."""
PREFIX_INTERFERENCE_OUTSIDE_TASK = "interference_outside_task"
"""Another source away from every task span: an annotation."""

GROUND_KEY_PREFIXES = (
    PREFIX_GATE,
    PREFIX_CONFORMANCE,
    PREFIX_NOTHING_READ,
    PREFIX_STORE_ASSERTION,
    PREFIX_CONFORMANCE_UNANSWERED,
    PREFIX_DEVIATION,
    PREFIX_UNMEASURED,
    PREFIX_ROUTE_MISMATCH,
    PREFIX_BRANCH_SILENT,
    PREFIX_HINT_MISMATCH,
    PREFIX_NODE,
    PREFIX_FAULT_IN_TASK,
    PREFIX_FAULT_OUTSIDE_TASK,
    PREFIX_INTERFERENCE_IN_TASK,
    PREFIX_INTERFERENCE_OUTSIDE_TASK,
)
"""Ground keys written ``<prefix>:<name>``, the name being a gate, a reporting node or a branch."""

OPERATIONAL_GROUND_KEYS = frozenset(
    {
        KEY_PREPROCESS_ERRORED,
        KEY_ROUTING_ERRORED,
        KEY_BAD_HINT_MAP,
        KEY_DECLARATION_UNREAD,
        KEY_ROUTE_UNEXPLAINED,
        KEY_ROUTE_UNREADABLE,
        KEY_CRITICAL_ABSENCE,
        KEY_TAXONOMY_NO_CLASSIFIER,
        KEY_REDACT_RESCAN_INCOMPLETE,
        KEY_UNCOMPUTED_READING,
        KEY_OWNING_BRANCH_INPUT_ABSENT,
        KEY_NODE_OUTCOME_UNREADABLE,
    }
)
"""Grounds that say the pipeline owes the recording something, not that the participant did anything.
A flag on one of these makes the file :attr:`RunStatus.INCOMPLETE`, verdict ``review``."""

OPERATIONAL_GROUND_PREFIXES = frozenset({PREFIX_UNMEASURED, PREFIX_BRANCH_SILENT})
"""Prefixed grounds that are operational in the same sense."""

RELEASE_GROUND_KEYS: dict[str, str] = {
    NO_LEXICAL_WORD: "no_lexical_word",
    NOTHING_BEYOND_STIMULUS: "nothing_beyond_stimulus",
    SCAN_FOUND_NOTHING: "scan_found_nothing",
    NON_LEXICAL_TASK: "non_lexical_task",
    REVIEWER_UNMASKED_ALL: "reviewer_unmasked_all",
    NO_CONTENT_MASKED: "no_content_masked",
    FINDINGS_ARE_TASK_CONTENT: "findings_are_task_content",
    REVIEWER_CLEARED_UNMASKED: "reviewer_cleared_unmasked",
    NO_TRANSCRIPT: "no_transcript",
    SPEECH_UNREAD: "speech_unread",
    REDACTION_OWED: "redaction_owed",
    SCAN_UNRECORDED: "scan_unrecorded",
    REVIEWER_PROPOSED_REDACTION: "reviewer_proposed_redaction",
    UNPLACED_FINDING_UNREAD: "unplaced_finding_unread",
    REDACT_VERIFY_FOUND: "redact_verify_found",
    REDACT_UNRESOLVED: "redact_unresolved",
    REVIEWER_CLEARED_RESCAN: "reviewer_cleared_rescan",
    REVIEWER_UNMASKED_SOME: "reviewer_unmasked_some",
    MASKS_TRIMMED_TO_CONTENT: "masks_trimmed_to_content",
    POLICY_MASKS_ADDED: "policy_masks_added",
    POLICY_MASKS_ONLY: "policy_masks_only",
    DISCARDED: "discarded",
}
"""The stable key of every release ground. A release REDACT itself decided carries
:data:`RELEASE_DECIDED_BY_REDACT`."""

RELEASE_DECIDED_BY_REDACT = "redact_decided"


def is_operational(key: str | None) -> bool:
    """Whether a ground key says the pipeline, not the participant, is what the flag is about.

    Args:
        key: A ground key, or None.

    Returns:
        True for a key in :data:`OPERATIONAL_GROUND_KEYS` or under a prefix in
        :data:`OPERATIONAL_GROUND_PREFIXES`.
    """
    if not key:
        return False
    return key in OPERATIONAL_GROUND_KEYS or key.split(":", 1)[0] in OPERATIONAL_GROUND_PREFIXES


def ground_key(verdict: "NodeVerdict") -> str:
    """The stable key of a verdict's ground, its own where it carries one.

    A deciding node writes no key, and stores written before keys existed carry none, so the key of
    such a verdict is derived from the node and its outcome.

    Args:
        verdict: A contributing verdict.

    Returns:
        The key.
    """
    if verdict.key:
        return verdict.key
    if verdict.node == _ADMIT and verdict.outcome is Outcome.FAIL:
        return KEY_UNMEASURABLE
    if verdict.node == "TAXONOMY" and verdict.outcome is Outcome.FLAG:
        return KEY_TAXONOMY_NO_CLASSIFIER
    if verdict.node == _REDACT and verdict.outcome is Outcome.FLAG:
        return KEY_REDACT_RESCAN_INCOMPLETE
    if "which is not a node outcome" in verdict.why:
        return KEY_NODE_OUTCOME_UNREADABLE
    return f"{PREFIX_NODE}:{verdict.node}:{verdict.outcome.value}"


def release_value(release: Release | None) -> str | None:
    """The stored value of a release, None where there is none.

    Args:
        release: The fold's release, None on a discard or where the graph could not assess it.

    Returns:
        Its string value, or None.
    """
    return release.value if release is not None else None


def release_ground_key(release_ground: str | None) -> str:
    """The stable key of a release ground.

    Args:
        release_ground: The release ground, or None where REDACT itself decided the release.

    Returns:
        Its key; :data:`RELEASE_DECIDED_BY_REDACT` for None, and the text itself for a ground this
        vocabulary does not hold, so an unknown ground stays visible rather than collapsing.
    """
    if release_ground is None:
        return RELEASE_DECIDED_BY_REDACT
    return RELEASE_GROUND_KEYS.get(release_ground, release_ground)


@dataclass(frozen=True)
class NodeVerdict:
    """One conclusion about the recording.

    Written by the deciding nodes — ADMIT, PREPROCESS, TAXONOMY, ``routing``, REDACT — and
    synthesised by :func:`fold_file_verdict` for each ground it flags on; a branch writes a
    :class:`BranchReport` instead.

    Attributes:
        node: The node's name.
        outcome: What it concluded: an ``Outcome`` from a deciding node, a ``Triage`` from the file
            fold.
        kind: The kind the conclusion is about, or None.
        why: The reason, in controlled vocabulary — never transcript text.
        key: The ground's stable key, one of :data:`GROUND_KEYS` or ``<prefix>:<name>`` for a prefix in
            :data:`GROUND_KEY_PREFIXES`; None on a verdict a deciding node wrote, which
            :func:`ground_key` derives.
    """

    node: str
    outcome: Outcome | Triage
    kind: str | None
    why: str
    key: str | None = None


@dataclass(frozen=True)
class BranchReport:
    """What one reporting node returns. It carries no outcome: a branch reports and VERDICT decides.

    The spans are not here: a branch proposes them into the store, in its own family, and VERDICT
    reads them back from there.

    Attributes:
        node: The node's name, which is also the branch name its decision is keyed under.
        kind: The kind it reports on, or None where it reports on no kind.
        conformance: Whether what was asked for happened — True, False, or :data:`UNDETERMINED`.
        conformance_of: What that conformance is about, one of :data:`CONFORMANCE_REFERENTS`.
        deviations: The deviation type names it found, sorted and deduplicated, each one declared
            in ``nodes.branches.DEVIATION_TYPES``. A flag ground only under
            ``verdict.deviation_flags``, which is false for all ten declared types.
        unmeasured: The config paths a body asked for and nobody has measured, in read order; the
            dependent conformance is left :data:`UNDETERMINED`.
        in_family: Whether the branch evaluated a declared task of its own kind — whether
            ``dispatch`` took the align mode.
    """

    node: str
    kind: str | None
    conformance: Conformance
    conformance_of: str
    deviations: tuple[str, ...] = ()
    unmeasured: tuple[str, ...] = ()
    in_family: bool = False


@dataclass(frozen=True)
class BranchDecision:
    """What ``routing`` decided about one branch, as the fold reads it.

    Attributes:
        branch: The branch's name, which is also the name its own verdict is written under.
        will_run: Whether routing selected it.
        route_state: What the ruleset made of this branch, one of :data:`BRANCH_ROUTE_STATES`. The
            content reading alone; a declared route never rewrites it.
        declared: Whether the recording's declaration named this branch — its task family, or a
            hint tag the map resolves.
        forced_by_declaration: Whether the declaration added it: ``declared`` and not
            content-routed.
        hint_tags: The declared tags naming this branch, when a hint supplied any the map resolves.
        bad_map_values: ``routing.hint_branch_map`` entries whose value is not a branch, as
            ``{tag: value}``. A property of the configuration, so every decision carries the same
            one.
        withheld_critical: Whether this branch was not run because the run hit a critical failure,
            rather than because the ruleset and the declaration both left it out.
    """

    branch: str
    will_run: bool
    route_state: str
    forced_by_declaration: bool
    declared: bool = False
    hint_tags: tuple[str, ...] = ()
    bad_map_values: dict[str, str] = field(default_factory=dict)
    withheld_critical: bool = False


@dataclass(frozen=True)
class FoldPolicy:
    """The ``verdict.*`` config section: what this fold does with what the reporting nodes report.

    ``specs/20260817-triage-workflow-dag/config-derivations.md`` § verdict carries each value's
    derivation.

    Attributes:
        conformance_flags: Whether a reported non-conformance about a **task** is a flag ground.
        undetermined_flags: Whether an unanswered conformance is.
        deviation_flags: Whether a reported deviation is. False for all ten declared types.
        unmeasured_points_flag: Whether a reporting node that could not read an operating point it
            wanted is.
        conformance_flags_by_family: Declared task family to whether a non-conformance on it flags,
            overriding ``conformance_flags``. This is what makes the fold task-aware.
        llm_redaction_flags: Whether a REVIEW reading that proposes hiding more is a flag ground on
            the **triage** axis.
        llm_redaction_withholds: Whether a reading that proposes hiding more also withholds a
            recording the evidence would release, with or without REDACT having run. The
            one direction in which a weighting toward the reviewer may move the release axis:
            tightening.
        llm_rescan_clears: Whether a reading of the original as clean, proposing nothing to hide,
            releases the redacted copy of a recording REDACT withheld only because its re-scan still
            read a finding. An incomplete scan or re-scan is never cleared.
        llm_reset_redactions: Whether a reading's ``release`` entries unmask the words they name.
        condition_categories: Upper-cased reviewer categories whose ``proposal`` entries are health
            conditions, as a reading written before conditions had their own part carries them. A
            condition is never masked and never withholds.
        trim_protected_categories: Upper-cased detector categories under which a marked word
            written as a proper noun is never unmasked by the content-word trim.
        cohort_conditions: The packaged cohort profile a listed condition is read against
            (:mod:`senselab.audio.workflows.triage.cohort`), or None where the study declares none,
            in which case every condition is an ``other`` one.
        person_name_review_flags: Whether a masked person's name is a flag ground
            (:data:`PERSON_NAME_AWAITS_REVIEW`); the release is unchanged.
        llm_second_speaker_flags: Whether a reading that heard more than one speaker is a flag ground
            on the triage axis.
        llm_instructions_spoken_flags: Whether a reading that quotes the task's instructions spoken in
            the recording is a flag ground (:data:`INSTRUCTIONS_SPOKEN`); the release is unchanged.
        llm_contradiction_flags: Whether a flagged reading that names no words is a flag ground on
            the triage axis.
        second_opinion_disagreement_flags: Whether a confident disagreement between the second-opinion
            model and the reviewer is a flag ground (:data:`SECOND_OPINION_DISAGREES`).
        second_opinion_confident_yes: The probability at or above which the second opinion is a confident
            "yes"; None leaves the comparison unmeasured.
        second_opinion_confident_no: The probability at or below which it is a confident "no"; None leaves
            the comparison unmeasured.
        uncomputed_reading_flags: Whether a conformance gate left undecided because a reading the
            task is judged on was never computed is a flag ground of its own.
        model_speaker_families: Declared families whose instructions permit someone to say the sentence
            first; the speaker gate still flags them, and its ground says so.
        hint_mismatch_exempt_families: Declared families whose branch not finding the declared sound is
            no flag ground.
    """

    conformance_flags: bool = True
    undetermined_flags: bool = False
    deviation_flags: bool = False
    unmeasured_points_flag: bool = True
    llm_redaction_flags: bool = True
    llm_redaction_withholds: bool = False
    llm_rescan_clears: bool = False
    llm_reset_redactions: bool = False
    condition_categories: tuple[str, ...] = ("CONDITION",)
    trim_protected_categories: tuple[str, ...] = ()
    cohort_conditions: str | None = None
    person_name_review_flags: bool = False
    llm_second_speaker_flags: bool = False
    llm_contradiction_flags: bool = False
    llm_instructions_spoken_flags: bool = False
    uncomputed_reading_flags: bool = False
    second_opinion_disagreement_flags: bool = False
    second_opinion_confident_yes: float | None = None
    second_opinion_confident_no: float | None = None
    hint_mismatch_exempt_families: tuple[str, ...] = ()
    model_speaker_families: tuple[str, ...] = ()
    conformance_flags_by_family: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Any) -> "FoldPolicy":  # noqa: ANN401 — TriageConfig, not imported here
        """The policy this configuration declares.

        Every key is read with ``get``, falling back to the packaged default rather than raising.

        Args:
            config: The resolved triage configuration.

        Returns:
            The policy.
        """
        return cls(
            conformance_flags=bool(config.get(f"{_SECTION}.conformance_flags", True)),
            undetermined_flags=bool(config.get(f"{_SECTION}.undetermined_flags", False)),
            deviation_flags=bool(config.get(f"{_SECTION}.deviation_flags", False)),
            unmeasured_points_flag=bool(config.get(f"{_SECTION}.unmeasured_points_flag", True)),
            llm_redaction_flags=bool(config.get(f"{_SECTION}.llm_redaction_flags", True)),
            llm_redaction_withholds=bool(config.get(f"{_SECTION}.llm_redaction_withholds", False)),
            llm_rescan_clears=bool(config.get(f"{_SECTION}.llm_rescan_clears", False)),
            llm_reset_redactions=bool(config.get(f"{_SECTION}.llm_reset_redactions", False)),
            condition_categories=tuple(
                str(category).upper() for category in (config.get(f"{_SECTION}.condition_categories") or ())
            ),
            person_name_review_flags=bool(config.get(f"{_SECTION}.person_name_review_flags", False)),
            trim_protected_categories=tuple(
                str(category).upper() for category in (config.get(f"{_SECTION}.trim_protected_categories") or ())
            ),
            cohort_conditions=str(config.get(f"{_SECTION}.cohort_conditions") or "") or None,
            llm_second_speaker_flags=bool(config.get(f"{_SECTION}.llm_second_speaker_flags", False)),
            llm_contradiction_flags=bool(config.get(f"{_SECTION}.llm_contradiction_flags", False)),
            llm_instructions_spoken_flags=bool(config.get(f"{_SECTION}.llm_instructions_spoken_flags", False)),
            uncomputed_reading_flags=bool(config.get(f"{_SECTION}.uncomputed_reading_flags", False)),
            second_opinion_disagreement_flags=bool(config.get(f"{_SECTION}.second_opinion_disagreement_flags", False)),
            second_opinion_confident_yes=_optional_float(config.get(f"{_SECTION}.second_opinion_confident_yes")),
            second_opinion_confident_no=_optional_float(config.get(f"{_SECTION}.second_opinion_confident_no")),
            hint_mismatch_exempt_families=tuple(
                str(family) for family in (config.get(f"{_SECTION}.hint_mismatch_exempt_families") or ())
            ),
            model_speaker_families=tuple(
                str(family) for family in (config.get(f"{_SECTION}.model_speaker_families") or ())
            ),
            conformance_flags_by_family={
                str(family): bool(flags)
                for family, flags in (config.get(f"{_SECTION}.conformance_flags_by_family") or {}).items()
            },
        )

    def flags_conformance(self, referent: str, declared_family: str | None) -> bool:
        """Whether a reported non-conformance of this referent, on this task, is a flag ground.

        Args:
            referent: What the conformance was about, one of :data:`CONFORMANCE_REFERENTS`.
            declared_family: The task family the recording declares, or None.

        Returns:
            True for :data:`STORE_ASSERTIONS`, which no task key and no switch governs. For
            :data:`TASK`, the family's own entry where it has one, else ``conformance_flags``. An
            unknown referent never flags.
        """
        if referent == STORE_ASSERTIONS:
            return True
        if referent != TASK:
            return False
        if declared_family is not None and declared_family in self.conformance_flags_by_family:
            return self.conformance_flags_by_family[declared_family]
        return self.conformance_flags


@dataclass(frozen=True)
class FileVerdict:
    """The graph's conclusion about one recording, on both axes.

    Attributes:
        triage: What should happen to the recording.
        release: Which artefact of the recording may be handed on, or None on a discard and wherever the
            graph could not assess it. Never describes the store.
        run_status: Whether the pipeline finished what the decision needs.
        missing: Where ``run_status`` is incomplete, the operational grounds and the release ground
            that left it so. Empty otherwise.
        reason: The decision's primary reason, the first of ``reason_keys``; None on a pass the release
            does not hold.
        reason_keys: Every reason behind the decision, ``data/decision_reasons.yaml``'s precedence first.
        evidence: Every reading the fold weighed, with the decisive ones marked.
        release_ground: Why the release axis reads as it does, in controlled vocabulary, wherever
            REDACT did not decide it — one of :data:`RELEASE_WITHOUT_REDACTION_GROUNDS`,
            :data:`RELEASE_UNKNOWN_GROUNDS`, :data:`RELEASE_WITHHELD_GROUNDS` or
            :data:`RELEASE_WITH_REDACTION_GROUNDS`. None wherever REDACT itself decided.
        discard_ground: ``"unmeasurable"``, ``"too_short_for_task"``, ``"acoustically_empty"``,
            ``"no_breath_captured"``, ``"declared_task_absent"`` or None.
        ground_keys: The stable key of every ground behind the triage state -- the discard ground and
            every flag, sorted and deduplicated. Empty on a pass.
        annotation_keys: The stable key of every annotation, sorted and deduplicated: a reading recorded
            on the recording that moves no triage state (an airway task's ``task_mismatch``).
        annotations: Each annotation, with its node, kind, reason and key.
        findings: What each branch found, as a :class:`KindState` value, read off the spans it
            proposed in its own family. ``uncertain`` where it left no report at all.
        conformance: Each reporting node's conformance, keyed by node — True, False or
            :data:`UNDETERMINED`. QUALITY is in it and is not in ``findings``.
        conformance_of: What each of those conformances is about, one of
            :data:`CONFORMANCE_REFERENTS`.
        deviations: The deviation type names each reporting node found.
        unmeasured: The config paths each reporting node asked for and nobody has measured.
        declared_family: The task family the recording declares, or None.
        routes: What the ruleset made of each branch, one of :data:`BRANCH_ROUTE_STATES`; a
            branch ROUTING wrote no decision for is :data:`UNJUDGED`.
        route_state: What it made of the whole recording, one of :data:`FILE_ROUTE_STATES`, or None
            when ``routing`` wrote no evaluation.
        agreement: ``agree`` | ``mismatch`` | ``resolved`` | ``not_run`` per branch.
        hints: ``claimed_and_found`` | ``claimed_not_found`` | ``found_unclaimed`` | ``no_claim``
            per branch, read against the recording's declaration as ROUTING resolved it.
        reasons: Every contributing verdict, in order — not only the deciding one.
        ran: Whether each node ran.
        branches: The routing decision joined to the branch's reported conformance.
        bad_map_values: ``routing.hint_branch_map`` entries whose value is not a branch.
        llm_redaction: REDACT's LLM re-read annotation — status, iterations, flagged categories,
            model id, resolved commit, failure. Empty when REDACT wrote none.
        second_opinion: SECOND_OPINION's reading as the fold compared it -- status, probabilities,
            the disagreements found, model id and blob digest. Empty when none was written.
        critical_absences: Per branch not one of whose gates could be read, each gate and the
            recorded absence. Non-empty means no branch was run.
        gates: The task group's gates and every one this fold applied — the gate, its reading, the
            bound and the group — so the conformance can be read backwards. Empty where the
            recording declares no task this graph holds a row for.
        breath_pattern: The breathing-pattern measure's reading for a breath family -- pattern, event
            count, durations, intervals and rhythm. Empty for every other family.
        cough_pattern: The cough-onset measure's reading for a cough family -- onsets, events, review
            counts and extent. Empty for every other family.
        quality_join: QUALITY's join of the background against the task spans. Empty where QUALITY
            wrote none.
    """

    triage: Triage
    release: Release | None
    run_status: RunStatus = RunStatus.COMPLETE
    missing: list[str] = field(default_factory=list)
    reason: str | None = None
    reason_keys: list[str] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    discard_ground: str | None = None
    release_ground: str | None = None
    ground_keys: list[str] = field(default_factory=list)
    annotation_keys: list[str] = field(default_factory=list)
    annotations: list[NodeVerdict] = field(default_factory=list)
    findings: dict[str, str] = field(default_factory=dict)
    conformance: dict[str, Conformance] = field(default_factory=dict)
    conformance_of: dict[str, str] = field(default_factory=dict)
    deviations: dict[str, list[str]] = field(default_factory=dict)
    unmeasured: dict[str, list[str]] = field(default_factory=dict)
    declared_family: str | None = None
    routes: dict[str, str] = field(default_factory=dict)
    route_state: str | None = None
    agreement: dict[str, str] = field(default_factory=dict)
    hints: dict[str, str] = field(default_factory=dict)
    reasons: list[NodeVerdict] = field(default_factory=list)
    ran: dict[str, RunState] = field(default_factory=dict)
    branches: dict[str, dict[str, Any]] = field(default_factory=dict)
    bad_map_values: dict[str, str] = field(default_factory=dict)
    llm_redaction: dict[str, Any] = field(default_factory=dict)
    second_opinion: dict[str, Any] = field(default_factory=dict)
    critical_absences: dict[str, dict[str, str]] = field(default_factory=dict)
    gates: dict[str, Any] = field(default_factory=dict)
    breath_pattern: dict[str, Any] = field(default_factory=dict)
    cough_pattern: dict[str, Any] = field(default_factory=dict)
    quality_join: dict[str, Any] = field(default_factory=dict)
    voice_phonation: dict[str, Any] = field(default_factory=dict)
    ddk_task: dict[str, Any] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        """Every decision point of this fold, as JSON-ready values.

        Categorical throughout — outcomes, states, type names and config paths, never transcript
        text or a detected string.

        Returns:
            The decision, keyed as :class:`FileVerdict` names its fields.
        """
        return {
            "triage": self.triage.value,
            "release": release_value(self.release),
            "run_status": self.run_status.value,
            "missing": list(self.missing),
            "reason": self.reason,
            "reason_keys": list(self.reason_keys),
            "evidence": records(self.evidence),
            "discard_ground": self.discard_ground,
            "release_ground": self.release_ground,
            "release_ground_key": release_ground_key(self.release_ground),
            "ground_keys": list(self.ground_keys),
            "annotation_keys": list(self.annotation_keys),
            "annotations": [{"node": a.node, "kind": a.kind, "why": a.why, "key": a.key} for a in self.annotations],
            "declared_family": self.declared_family,
            "findings": dict(self.findings),
            "conformance": dict(self.conformance),
            "conformance_of": dict(self.conformance_of),
            "deviations": {node: list(names) for node, names in self.deviations.items()},
            "unmeasured": {node: list(names) for node, names in self.unmeasured.items()},
            "routes": dict(self.routes),
            "route_state": self.route_state,
            "agreement": dict(self.agreement),
            "hints": dict(self.hints),
            "branches": dict(self.branches),
            "bad_map_values": dict(self.bad_map_values),
            "llm_redaction": dict(self.llm_redaction),
            "second_opinion": dict(self.second_opinion),
            "critical_absences": {branch: dict(gates) for branch, gates in self.critical_absences.items()},
            "gates": dict(self.gates),
            "breath_pattern": dict(self.breath_pattern),
            "cough_pattern": dict(self.cough_pattern),
            "quality_join": dict(self.quality_join),
            "voice_phonation": dict(self.voice_phonation),
            "ddk_task": dict(self.ddk_task),
            "ran": {node: state.value for node, state in self.ran.items()},
            "reasons": [
                {"node": r.node, "outcome": r.outcome.value, "kind": r.kind, "why": r.why, "key": ground_key(r)}
                for r in self.reasons
            ],
        }


def _found(reported: bool, spans_n: int) -> str:
    """What a branch found, read off the spans it proposed rather than off any conclusion of its own.

    Args:
        reported: Whether the branch left a report at all.
        spans_n: How many spans it proposed into its own family.

    Returns:
        ``present`` with a span in hand, ``absent`` where the branch reported and proposed none, and
        ``uncertain`` where it left no report.
    """
    if not reported:
        return KindState.UNCERTAIN.value
    return KindState.PRESENT.value if spans_n > 0 else KindState.ABSENT.value


def _silence(state: RunState | None) -> str:
    """How a branch that was asked to run left no verdict.

    Args:
        state: The branch's run state, if the caller or the store knows one.

    Returns:
        The phrase naming which of the three silences happened.
    """
    if state is RunState.ERRORED:
        return "errored without a verdict"
    if state is RunState.COMPLETED:
        return "completed without a verdict"
    return "never ran"


def _optional_float(value: Any) -> float | None:  # noqa: ANN401 — a config leaf of any type
    """A config value as a float, or None where it is unmeasured."""
    return None if value is None else float(value)


def second_opinion_disagreements(
    opinion: Mapping[str, Any] | None,
    llm_redaction: Mapping[str, Any] | None,
    *,
    confident_yes: float | None,
    confident_no: float | None,
    identifier_masked: bool | None = None,
) -> list[str]:
    """Where the second-opinion model confidently disagrees with the reviewer.

    Compared only where both answered: the opinion's status is ``ok`` and the reviewer read the
    transcript (``clean`` or ``flagged``). ``instructions_spoken`` is compared only for a reading
    that carries the part, so a reading from before the prompt asked it is never a disagreement;
    ``policy_identifier_present`` only where ``identifier_masked`` is given.

    Args:
        opinion: The ``second_opinion_answers`` measurement's attributes, or None.
        llm_redaction: REVIEW's annotation, or None.
        confident_yes: The probability at or above which the opinion is a confident yes.
        confident_no: The probability at or below which it is a confident no.
        identifier_masked: Whether, once the reviewer's reading is folded, any mask stands or the
            reviewer asks to hide more -- the reviewer's answer to ``policy_identifier_present``; None
            where the fold has no plan to say.

    Returns:
        One description per disagreeing question, ``<question> p=<p> reviewer=<yes|no>``, in
        :data:`SECOND_OPINION_QUESTIONS` order; empty where either threshold is unmeasured.
    """
    held = dict(opinion or {})
    annotation = dict(llm_redaction or {})
    if confident_yes is None or confident_no is None:
        return []
    if held.get("status") != "ok" or annotation.get("status") not in ("clean", "flagged"):
        return []
    probabilities = held.get("probabilities") or {}
    others = [entry for entry in annotation.get("other_speakers") or () if isinstance(entry, Mapping)]
    reviewer: dict[str, bool] = {
        "other_voice": annotation.get("speakers") == "more_than_one"
        or (annotation.get("speakers") == "unclear" and bool(others)),
    }
    if identifier_masked is not None:
        reviewer["policy_identifier_present"] = identifier_masked
    if "instructions_spoken" in annotation:
        reviewer["instructions_spoken"] = any(str(text).strip() for text in annotation.get("instructions_spoken") or ())
    found = []
    for question in SECOND_OPINION_QUESTIONS:
        if question not in reviewer or probabilities.get(question) is None:
            continue
        p = float(probabilities[question])
        said = reviewer[question]
        if (p >= confident_yes and not said) or (p <= confident_no and said):
            found.append(f"{question} p={p:.2f} reviewer={'yes' if said else 'no'}")
    return found


def reviewer_named_no_words(llm_redaction: Mapping[str, Any] | None) -> bool:
    """Whether a reading flagged the recording and named no words to move.

    Args:
        llm_redaction: REVIEW's annotation, or None where it wrote none.

    Returns:
        True where the reviewer read the text, reported it ``flagged``, judged the redaction
        ``incomplete``, and its proposal carries no entry at all. A reading that the original carries
        identifying content and the redaction is complete agrees with the masks; it needs no words.
    """
    annotation = dict(llm_redaction or {})
    return (
        annotation.get("status") == "flagged"
        and annotation.get("redaction") == "incomplete"
        and not list(annotation.get("proposal") or ())
    )


def _reviewer_found_residue(llm_redaction: Mapping[str, Any] | None) -> bool:
    """Whether the reading asks for something identifying to be hidden that the detectors left.

    Args:
        llm_redaction: REVIEW's annotation, or None where it wrote none.

    Returns:
        True only where the reviewer read the text, flagged it, and proposed at least one ``redact``
        entry. A proposal that only releases, or proposes nothing, says the detectors hid too much
        or nothing, which is not residue. ``absent``, ``disabled`` and ``nothing_to_read`` are not
        readings and conclude nothing.
    """
    annotation = dict(llm_redaction or {})
    if annotation.get("status") != "flagged":
        return False
    return any(str(entry.get("action")) == "redact" for entry in annotation.get("proposal") or ())


def deciding_reading(
    llm_redaction: Mapping[str, Any] | None,
    agreed: frozenset[int],
    condition_categories: Sequence[str] = ("CONDITION",),
) -> dict[str, Any]:
    """The reading with the ``redact`` entries that agree with the masks, and the listed conditions, set aside.

    Args:
        llm_redaction: REVIEW's annotation, or None where it wrote none.
        agreed: The ``proposal`` positions of ``redact`` entries whose words a mask already hides or
            that place a finding the fold could not place
            (:attr:`~senselab.audio.workflows.triage.nodes.redact.MaskPlan.agreed`).
        condition_categories: Upper-cased categories whose entries are health conditions, which a
            prompt-v6 reading carries as ``redact`` entries; never a proposal to hide.

    Returns:
        The annotation, its ``proposal`` without those entries. An agreeing entry is not the reviewer
        proposing to hide more, and a condition is never hidden, so no test of residue reads either.
    """
    annotation = dict(llm_redaction or {})
    conditions = {category.upper() for category in condition_categories}
    proposal = [
        entry
        for index, entry in enumerate(annotation.get("proposal") or ())
        if str(entry.get("action")) == "release"
        or (index not in agreed and str(entry.get("category") or "").upper() not in conditions)
    ]
    if len(proposal) == len(annotation.get("proposal") or ()):
        return annotation
    return {**annotation, "proposal": proposal}


def _reviewer_cleared(llm_redaction: Mapping[str, Any] | None) -> bool:
    """Whether the reading says the original carries nothing identifying and asks to hide nothing.

    Args:
        llm_redaction: REVIEW's annotation, or None where it wrote none.

    Returns:
        True only where the reviewer read the text (``clean`` or ``flagged``), judged the original
        ``clean``, and proposed no ``redact`` entry.
    """
    annotation = dict(llm_redaction or {})
    if annotation.get("status") not in ("clean", "flagged"):
        return False
    if annotation.get("original") != "clean":
        return False
    return not any(str(entry.get("action")) == "redact" for entry in annotation.get("proposal") or ())


def reviewer_may_unmask(llm_redaction: Mapping[str, Any] | None) -> bool:
    """Whether the reading's ``release`` entries may unmask the words they name.

    Args:
        llm_redaction: REVIEW's annotation, or None where it wrote none.

    Returns:
        True wherever the reviewer read the text (``clean`` or ``flagged``). Its ``redact`` entries
        decide the release on their own (:func:`_reviewer_found_residue`); they do not stop its
        ``release`` entries from saying which masked words are not identifying.
    """
    return dict(llm_redaction or {}).get("status") in ("clean", "flagged")


def _release_from(
    node_verdicts: Sequence[NodeVerdict],
    evidence: RedactionEvidence,
    ran: Mapping[str, RunState],
    reviewer_withholds: str | None = None,
    speech_declined: bool = False,
    reviewer_clears: bool = False,
) -> tuple[Release | None, str | None]:
    """Which artefact may be handed on: the evidence's answer, then the reviewer's moves.

    REDACT runs only where a scan found something, so its absence is the ordinary case and carries
    no implication of its own. The table is in ``specs/20260817-triage-workflow-dag/verdict.md``;
    the vocabulary is in ``specs/20260924-which-artefact-is-releasable/design.md``.

    Args:
        node_verdicts: Every node verdict the fold was given.
        evidence: What the store says about whether anything was redactable, and which masks
            stand once the reviewer's unmasks and the content-word trim are applied.
        ran: Whether each node ran.
        reviewer_withholds: The withholding ground where the reviewer read residue and the policy
            lets that withhold, else None. It may only tighten: it turns either release into a
            withholding, whether or not REDACT ran, and never touches a withholding or a release
            the graph could not assess.
        speech_declined: Whether the ruleset declined SPEECH, in which case a task that carries no
            lexical content by construction is released without redaction.
        reviewer_clears: Whether the reviewer read the original as clean, proposed nothing to hide,
            and the policy lets that release. It moves only a REDACT ``fail`` whose re-scan still read
            a finding, and only to the redacted copy.

    Returns:
        Which artefact may be handed on, never anything about the store, and the ground behind it;
        None where the graph could not assess it, with one of :data:`RELEASE_UNKNOWN_GROUNDS`.
        A REDACT ``pass`` clears the redacted copy and not the original; the ground is None only where
        that pass decided and a planned mask stands, and one of the controlled grounds otherwise. A
        REDACT ``fail`` or ``flag`` that nothing clears withholds on :data:`REDACT_VERIFY_FOUND` or
        :data:`REDACT_UNRESOLVED`.
        A copy that would mask nothing is never released as a redacted copy: the original is.
    """
    release, ground = _release_from_evidence(node_verdicts, evidence, ran, speech_declined)
    if reviewer_withholds is not None and release in (Release.REDACTED, Release.AS_IS):
        return Release.WITHHELD, reviewer_withholds
    redact = next((verdict for verdict in node_verdicts if verdict.node == _REDACT), None)
    if (
        reviewer_clears
        and release is Release.WITHHELD
        and ground is None
        and redact is not None
        and redact.outcome is Outcome.FAIL
        and evidence.rescan_survivors
    ):
        release, ground = Release.REDACTED, REVIEWER_CLEARED_RESCAN
    if (
        release is Release.AS_IS
        and ground == SCAN_FOUND_NOTHING
        and redact is None
        and evidence.masks_final_n > 0
        and evidence.policy_masks_n > 0
    ):
        return Release.REDACTED, POLICY_MASKS_ONLY
    if release is Release.WITHHELD and ground is None:
        verify_found = redact is not None and redact.outcome is Outcome.FAIL and bool(evidence.rescan_survivors)
        ground = REDACT_VERIFY_FOUND if verify_found else REDACT_UNRESOLVED
    if release is not Release.REDACTED:
        return release, ground
    reviewer = evidence.reviewer_unmasked_n > 0
    if evidence.masks_final_n == 0:
        if evidence.masks_changed:
            return Release.AS_IS, REVIEWER_UNMASKED_ALL if reviewer else NO_CONTENT_MASKED
        if ground is None:
            return Release.AS_IS, FINDINGS_ARE_TASK_CONTENT
        if ground == REVIEWER_CLEARED_RESCAN:
            return Release.AS_IS, REVIEWER_CLEARED_UNMASKED
    if evidence.masks_changed:
        if reviewer:
            return Release.REDACTED, REVIEWER_UNMASKED_SOME
        return Release.REDACTED, POLICY_MASKS_ADDED if evidence.policy_masks_n else MASKS_TRIMMED_TO_CONTENT
    return release, ground


def _release_from_evidence(
    node_verdicts: Sequence[NodeVerdict],
    evidence: RedactionEvidence,
    ran: Mapping[str, RunState],
    speech_declined: bool,
) -> tuple[Release | None, str | None]:
    """Which artefact the store's own evidence lets be handed on, before any reviewer reading.

    Args:
        node_verdicts: Every node verdict the fold was given.
        evidence: What the store says about whether anything was redactable.
        ran: Whether each node ran.
        speech_declined: Whether the ruleset declined SPEECH.

    Returns:
        The release and its ground, as :func:`_release_from` describes them.
    """
    redact = next((verdict for verdict in node_verdicts if verdict.node == _REDACT), None)
    if redact is not None:
        return (Release.REDACTED if redact.outcome is Outcome.PASS else Release.WITHHELD), None
    if evidence.findings_n > 0:
        return None, REDACTION_OWED
    speech = ran.get(_SPEECH)
    if speech is RunState.ERRORED:
        return None, SPEECH_UNREAD
    if evidence.lexical_words_n is None:
        if speech is RunState.COMPLETED:
            return None, SPEECH_UNREAD
        if speech_declined:
            return Release.AS_IS, NON_LEXICAL_TASK
        return None, NO_TRANSCRIPT
    if evidence.lexical_words_n == 0:
        return Release.AS_IS, NO_LEXICAL_WORD
    if evidence.scanned is False:
        return Release.AS_IS, NOTHING_BEYOND_STIMULUS
    if evidence.scanned is True:
        return Release.AS_IS, SCAN_FOUND_NOTHING
    return None, SCAN_UNRECORDED


def _agreement(route: str, reported: bool, found_state: str) -> str:
    """One branch's row of verdict.md's agreement table.

    Scores what the branch *found* — the spans it proposed — against what the ruleset routed.

    Args:
        route: What the ruleset made of the branch.
        reported: Whether the branch left a report.
        found_state: What it found, from :func:`_found`.

    Returns:
        ``not_run`` when the branch left no report, ``agree`` or ``mismatch`` against
        :data:`ROUTED` or :data:`DECLINED`, and ``resolved`` for the other three routes —
        :data:`UNAVAILABLE`, :data:`UNGATED` and :data:`UNJUDGED` — none of which made a claim to
        agree or disagree with.
    """
    if not reported:
        return NOT_RUN
    found = found_state == KindState.PRESENT.value
    if route == ROUTED:
        return AGREE if found else MISMATCH
    if route == DECLINED:
        return MISMATCH if found else AGREE
    return RESOLVED


def _hint_reading(claimed: bool, found: bool) -> str:
    """One branch's row of verdict.md's hint table.

    Args:
        claimed: Whether the declaration claimed the kind.
        found: Whether the kind resolved present.

    Returns:
        The reading, one of the four.
    """
    if claimed:
        return CLAIMED_AND_FOUND if found else CLAIMED_NOT_FOUND
    return FOUND_UNCLAIMED if found else NO_CLAIM


def _owner_performed(
    branch: str, by_branch: Mapping[str, BranchReport], findings: Mapping[str, str], evidence: TaskEvidence
) -> bool:
    """Whether one owning branch says the declared task was performed.

    Args:
        branch: An owning branch of the declared family.
        by_branch: The branch reports, keyed by branch.
        findings: What each branch found, from :func:`_found`.
        evidence: The task evidence; its event tokens corroborate an AIRWAY that could not decide.

    Returns:
        True where the branch's task conformance is True, or where it is AIRWAY, it found its kind,
        could not decide its conformance, the family is not one decided on detected events, and the
        transcript carries an event token naming the declared family's own event. A span alone, or
        a conformance of False, is never enough.
    """
    if branch == _AIRWAY and evidence.breath_mode is not None:
        return breath_conforms(evidence)
    if branch == _AIRWAY and evidence.cough_mode is not None:
        return cough_conforms(evidence)
    if branch == _VOICE and evidence.voice_mode is not None:
        return bool(evidence.voice_found) and evidence.voice_mismatch is None
    if branch == _SPEECH and evidence.ddk_mode is not None:
        return ddk_present(evidence)
    report = by_branch.get(branch)
    if report is None or report.conformance_of != TASK:
        return False
    if report.conformance is True:
        return True
    return (
        branch == _AIRWAY
        and report.conformance == UNDETERMINED
        and findings.get(branch) == KindState.PRESENT.value
        and evidence.required_event is None
        and evidence.event_tokens_n > 0
    )


def declared_task_performed(
    by_branch: Mapping[str, BranchReport], findings: Mapping[str, str], evidence: TaskEvidence
) -> bool:
    """Whether the declared task was performed, by the declared task's own branch.

    A branch's span, a speaker turn, a speech run or an activity-envelope extent is no evidence of
    the task; another branch's finding is never evidence of it. Where the recording declares no
    family a branch owns, any branch's finding stands, as before.

    Args:
        by_branch: The branch reports, keyed by branch.
        findings: What each branch found, from :func:`_found`.
        evidence: The task evidence.

    Returns:
        Whether any owning branch says so (:func:`_owner_performed`); with no owning branch named,
        whether any branch found its kind.
    """
    if not evidence.owning_branches:
        return any(state == KindState.PRESENT.value for state in findings.values())
    return any(_owner_performed(branch, by_branch, findings, evidence) for branch in evidence.owning_branches)


def declared_task_absent(
    by_branch: Mapping[str, BranchReport],
    findings: Mapping[str, str],
    evidence: TaskEvidence,
    lexical_words_n: int | None,
) -> bool:
    """Whether the declared task's own branch ran and found none of the task at all.

    Every owning branch must have reported and answered its task conformance without a True. A
    branch found none of the task where it proposed no span into its own family; a SPEECH-owned task
    also found none where the consensus transcript holds no lexical word -- a bracketed token, a
    filler or nothing at all matches no speech task. What another branch found does not count.

    Args:
        by_branch: The branch reports, keyed by branch.
        findings: What each branch found, from :func:`_found`.
        evidence: The task evidence.
        lexical_words_n: How many lexical words SPEECH read off the consensus, or None where it
            left no report to say.

    Returns:
        True where the declared task is absent; False where no owning branch is named, an owner did
        not report, or an owner says the task was performed.
    """
    owners = evidence.owning_branches
    if not owners or any(branch not in by_branch for branch in owners):
        return False
    if declared_task_performed(by_branch, findings, evidence):
        return False

    def found_none(branch: str) -> bool:
        if findings.get(branch) == KindState.ABSENT.value:
            return True
        return branch == _SPEECH and lexical_words_n == 0

    return all(found_none(branch) for branch in owners)


def no_required_event(evidence: TaskEvidence, performed: bool) -> bool:
    """Whether a family decided on detected events had none of its own kind detected.

    Args:
        evidence: The task evidence.
        performed: Whether the declared task's own branch says the task was performed.

    Returns:
        For a breath family read by the breathing-pattern measure, True where it read the inputs and
        found no breath (counted) or no alternating pattern (sustained); for a cough family read by the
        cough-onset measure, True where it read the inputs and found no cough onset. Otherwise True where the
        family names a required event kind and the owning AIRWAY branch reported a count of zero;
        False where it names none, reported no count, or the task was performed.
    """
    if evidence.breath_mode is not None:
        return (
            evidence.breath_decision is not None and not evidence.owner_absent_inputs and not breath_present(evidence)
        )
    if evidence.cough_mode is not None:
        return evidence.cough_onsets_n == 0 and not evidence.owner_absent_inputs
    if evidence.voice_mode is not None:
        return evidence.voice_found is False and not evidence.owner_absent_inputs
    if evidence.ddk_mode is not None:
        return evidence.ddk_decision == "absent" and not evidence.owner_absent_inputs
    return (
        evidence.required_event in EVENT_ABSENT_GROUNDS
        and evidence.events_found_n == 0
        and not evidence.owner_absent_inputs
        and not performed
    )


def ddk_absent_ground(evidence: TaskEvidence) -> str:
    """The discard ground of a syllable task whose reading decided ``absent``.

    Args:
        evidence: The task evidence.

    Returns:
        :data:`SYLLABLE_TRAIN_NOT_TARGET` where a train stood over the floor and its identity read
        another activity; :data:`NO_SYLLABLE_TRAIN_CAPTURED` otherwise.
    """
    standing = (evidence.ddk_reading.get("inputs") or {}).get("standing_n")
    return SYLLABLE_TRAIN_NOT_TARGET if standing else NO_SYLLABLE_TRAIN_CAPTURED


def discard_contested(evidence: TaskEvidence) -> bool:
    """Whether AIRWAY's own event detector contradicts a measure's no-event discard.

    Args:
        evidence: The task evidence.

    Returns:
        True where the family names a contest threshold, the detector found at least that many
        events of the family's own kind, and, where a rise is named, the breath train rose at least
        that far; False otherwise.
    """
    rise_ok = evidence.contest_rise_db_min is None or (
        evidence.breath_train_rise_db is not None and evidence.breath_train_rise_db >= evidence.contest_rise_db_min
    )
    return (
        evidence.contest_events_min is not None
        and evidence.events_found_n is not None
        and evidence.events_found_n >= evidence.contest_events_min
        and rise_ok
    )


JOIN_UNREAD = ("quality_join",)
"""The join's own measurement: a recording whose QUALITY wrote none is not measured. A missing BACKGROUND
reading is the branches' absent input, and a PREPROCESS without a plain stream flags on its own."""


def nothing_captured(quality: Mapping[str, Any]) -> bool:
    """Whether QUALITY's join says the recording captured nothing: no activity, or only the room.

    Args:
        quality: QUALITY's join record.

    Returns:
        True where no stream held activity over the floor, or the recording stands under its session's
        level by the join's bound; False otherwise, and where the join was not read.
    """
    return bool(quality.get("no_activity")) or bool(quality.get("quiet_vs_session"))


def task_event_heard(evidence: TaskEvidence, lexical_words_n: int | None) -> bool:
    """Whether any branch heard an event of the task: a breath, a cough, phonation, syllables or a word.

    Args:
        evidence: The task evidence.
        lexical_words_n: The consensus transcript's lexical word count, or None where SPEECH read none.

    Returns:
        True where a task reading holds an event (a syllable train among them) or the transcript holds
        a lexical word.
    """
    return (
        breath_present(evidence)
        or cough_present(evidence)
        or bool(evidence.voice_found)
        or ddk_present(evidence)
        or bool(lexical_words_n)
    )


def join_unread(quality: Mapping[str, Any]) -> list[str]:
    """The inputs QUALITY's join lacked that leave the recording not measured.

    Args:
        quality: QUALITY's join record.

    Returns:
        The names of :data:`JOIN_UNREAD` the record lists as missing.
    """
    return [name for name in quality.get("missing") or () if name in JOIN_UNREAD]


def _join_reasons(quality: Mapping[str, Any], flag: Callable[..., None], annotate: Callable[..., None]) -> None:
    """Raise QUALITY's join as flags and annotations: what touches a task span reviews, the rest annotates.

    Args:
        quality: QUALITY's join record.
        flag: The fold's flag.
        annotate: The fold's annotate.
    """
    unread = join_unread(quality)
    if unread:
        flag(QUALITY, f"{UNCOMPUTED_READING}: {', '.join(unread)}", KEY_UNCOMPUTED_READING)
        return
    for section, deciding, inside, outside in (
        ("faults", quality.get("faults_in_task") or (), PREFIX_FAULT_IN_TASK, PREFIX_FAULT_OUTSIDE_TASK),
        (
            "interference",
            quality.get("interference_in_task") or (),
            PREFIX_INTERFERENCE_IN_TASK,
            PREFIX_INTERFERENCE_OUTSIDE_TASK,
        ),
    ):
        for kind, split in (quality.get(section) or {}).items():
            held = list(split.get("in") or ())
            elsewhere = list(split.get("out") or ()) + ([] if kind in deciding else held)
            if kind in deciding:
                seconds = round(sum(b - a for a, b in held), 3)
                flag(
                    QUALITY,
                    f"{inside}: {kind}, {len(held)} span(s), {seconds} s, first at {held[0][0]} s",
                    f"{inside}:{kind}",
                )
            if elsewhere:
                annotate(QUALITY, f"{outside}: {kind}, {len(elsewhere)} span(s)", f"{outside}:{kind}")
    if quality.get("streams_disagree"):
        streams = quality.get("streams") or {}
        flag(
            QUALITY,
            f"{STREAMS_DISAGREE}: the enhanced stream loses {streams.get('lost_n')} of the "
            f"{streams.get('standing_n')} {quality.get('event_kind')} events the raw stream stands on",
            KEY_STREAMS_DISAGREE,
        )


def _count_mismatch(applied_gates: Sequence[Mapping[str, Any]], evidence: TaskEvidence) -> str | None:
    """The detected-against-instructed count where an airway task's own events fell short of it.

    Args:
        applied_gates: The declared task's applied gate records.
        evidence: The task evidence, read for the event kind and the instructed count.

    Returns:
        ``"detected N <kind> events where M were instructed"`` where ``events_min`` passed and
        ``instructed_count_min_fraction`` failed; None otherwise.
    """
    by_name = {str(gate.get("gate")): gate for gate in applied_gates}
    found = by_name.get("events_min")
    fraction = by_name.get("instructed_count_min_fraction")
    if (
        evidence.event_kind is None
        or evidence.instructed_count is None
        or found is None
        or fraction is None
        or found.get("passed") is not True
        or fraction.get("passed") is not False
    ):
        return None
    detected = found.get("value")
    return f"detected {detected} {evidence.event_kind} events where {evidence.instructed_count} were instructed"


_COMPARISON = {"at_least": ">=", "at_most": "<="}
_DISCARD_ITEMS = {
    UNMEASURABLE: "admit_outcome",
    TOO_SHORT_FOR_TASK: "task_duration_s",
    ACOUSTICALLY_EMPTY: "capture",
    NO_BREATH_CAPTURED: "breath_event_db_over_floor",
    NO_COUGH_CAPTURED: "cough_onsets",
    NO_PHONATION_CAPTURED: "phonation_found",
    NO_SYLLABLE_TRAIN_CAPTURED: "ddk_event_db_over_floor",
    SYLLABLE_TRAIN_NOT_TARGET: "ddk_identity",
    DECLARED_TASK_ABSENT: "declared_task_found",
}
_INSTRUCTION_ITEMS = frozenset(
    {"breaths_found", "coughs_against_instructed", "glide_travel_declared_st", "ddk_events_against_instructed"}
)
"""Readings compared against the instruction: annotations, never what decides a pass."""


def _flag_item(name: str, effect: str = REVIEW) -> EvidenceItem:
    """A reading that is a flag in its own right.

    Args:
        name: The item's name.
        effect: What it does.

    Returns:
        The item, compared against False.
    """
    return item(name, True, effect, comparison="==", threshold=False)


def _breath_items(task: TaskEvidence) -> list[EvidenceItem]:
    """A breath family's readings, as evidence items.

    Args:
        task: The task evidence the fold read.

    Returns:
        The task layer's comparisons, the breaths against the instruction, and the rhythm.
    """
    evidence = dict(task.breath_reading.get("evidence") or {})
    inputs = dict(evidence.get("inputs") or {})
    floor_db, low = inputs.get("floor_db"), inputs.get("snr_low_db")
    local_db, high = inputs.get("local_db"), inputs.get("snr_high_db")
    over_floor = floor_db is not None and low is not None and floor_db >= low
    items = [
        item(
            "breath_event_db_over_floor",
            floor_db,
            PASS if over_floor else DISCARD,
            unit="dB",
            comparison=">=",
            threshold=low,
        )
    ]
    if over_floor:
        clear = local_db is not None and high is not None and local_db >= high
        items.append(
            item(
                "breath_event_db_over_local",
                local_db,
                PASS if clear else REVIEW,
                unit="dB",
                comparison=">=",
                threshold=high,
            )
        )
        if clear:
            entangled = bool(inputs.get("entangled"))
            items.append(
                item(
                    "breath_events_entangled",
                    entangled,
                    REVIEW if entangled else PASS,
                    comparison="==",
                    threshold=False,
                )
            )
    if task.breath_train_breaths is not None:
        short = task.instructed_count is not None and task.breath_train_breaths < task.instructed_count
        items.append(
            item(
                "breaths_found",
                task.breath_train_breaths,
                ANNOTATION if short else PASS,
                unit="breaths",
                comparison=">=",
                threshold=task.instructed_count,
            )
        )
    rhythm = evidence.get("rhythm")
    if isinstance(rhythm, Mapping) and rhythm.get("hz") is not None:
        items.append(item("breath_rhythm_hz", rhythm.get("hz"), ANNOTATION, unit="Hz"))
    return items


def _cough_items(task: TaskEvidence) -> list[EvidenceItem]:
    """A cough family's readings, as evidence items.

    Args:
        task: The task evidence the fold read.

    Returns:
        The onsets found, against the instruction where counted, and the review reading.
    """
    onsets = task.cough_onsets_n or 0
    items = [item("cough_onsets", onsets, PASS if onsets else DISCARD, unit="coughs", comparison=">=", threshold=1)]
    if onsets and task.instructed_count is not None and task.cough_mode == COUGH_COUNTED:
        items.append(
            item(
                "coughs_against_instructed",
                onsets,
                ANNOTATION if onsets < task.instructed_count else PASS,
                unit="coughs",
                comparison=">=",
                threshold=task.instructed_count,
            )
        )
    if onsets and task.cough_review:
        items.append(_flag_item("cough_review"))
    return items


def _voice_items(task: TaskEvidence) -> list[EvidenceItem]:
    """A voice family's readings, as evidence items.

    Args:
        task: The task evidence the fold read.

    Returns:
        Whether phonation was found, its durations, the glide's travel, the review readings and a cut.
    """
    reading = task.voice_reading
    items = [
        item(
            "phonation_found", task.voice_found, PASS if task.voice_found else DISCARD, comparison="==", threshold=True
        )
    ]
    if not task.voice_found:
        return items
    if reading.get("voiced_s") is not None:
        items.append(item("phonation_voiced_s", reading.get("voiced_s"), ANNOTATION, unit="s"))
    if reading.get("longest_hold_s") is not None:
        items.append(item("longest_hold_s", reading.get("longest_hold_s"), ANNOTATION, unit="s"))
    glide = dict(reading.get("glide") or {})
    if glide.get("travel_declared_semitones") is not None:
        items.append(
            item(
                "glide_travel_declared_st",
                glide.get("travel_declared_semitones"),
                ANNOTATION if task.voice_mismatch else PASS,
                unit="semitones",
            )
        )
    items.extend(_flag_item(f"phonation_review:{review}") for review in task.voice_review)
    return items


_DDK_ANNOTATIONS = (
    ("ddk_syllable_rate_hz", "syllable_rate_hz", "Hz"),
    ("ddk_cycle_rate_hz", "cycle_rate_hz", "Hz"),
    ("ddk_period_cv", "period_cv", None),
    ("ddk_period_trend_s_per_step", "period_trend_s_per_step", "s"),
    ("ddk_train_fraction", "train_fraction", None),
)
"""The syllable task's annotations: evidence item name, the reading's annotation key, the unit."""
DDK_REALISED_MASS = "ddk_realised_mass"
"""The per-position realised mass, one item per template position: ``ddk_realised_mass:<index>.<phone>``."""


def _ddk_items(task: TaskEvidence) -> list[EvidenceItem]:
    """A syllable-repetition family's readings, as evidence items.

    Args:
        task: The task evidence the fold read.

    Returns:
        The task layer's comparisons (an event over the floor, clear events, identity), the events
        against the instruction, and the rates, regularity, train fraction and per-position mass as
        annotations.
    """
    reading = task.ddk_reading
    inputs = dict(reading.get("inputs") or {})
    notes = dict(reading.get("annotations") or {})
    floor_db, low = inputs.get("floor_db"), inputs.get("snr_low_db")
    standing = bool(inputs.get("standing_n"))
    items = [
        item(
            "ddk_event_db_over_floor",
            floor_db,
            PASS if standing else DISCARD,
            unit="dB",
            comparison=">=",
            threshold=low,
        )
    ]
    if not standing:
        return items
    identity = inputs.get("identity")
    absent_max, identity_min = inputs.get("identity_absent_max"), inputs.get("identity_min")
    other = identity is not None and absent_max is not None and identity < absent_max
    ambiguous = identity is not None and identity_min is not None and identity < identity_min
    items.append(
        item(
            "ddk_identity",
            identity,
            DISCARD if other else REVIEW if ambiguous else PASS,
            comparison=">=",
            threshold=identity_min if not other else absent_max,
        )
    )
    if other:
        return items
    clear_needed = inputs.get("events_min")
    clear = inputs.get("clear_free_n")
    items.append(
        item(
            "ddk_clear_events",
            clear,
            PASS if clear is not None and clear_needed is not None and clear >= clear_needed else REVIEW,
            unit=f"{task.ddk_mode}s",
            comparison=">=",
            threshold=clear_needed,
        )
    )
    found, required = notes.get("events_n"), notes.get("required_count")
    if found is not None:
        items.append(
            item(
                "ddk_events_against_instructed" if required is not None else "ddk_events_found",
                found,
                PASS if required is not None and found >= required else ANNOTATION,
                unit=f"{task.ddk_mode}s",
                comparison=">=" if required is not None else None,
                threshold=required,
            )
        )
    for name, key, unit in _DDK_ANNOTATIONS:
        if notes.get(key) is not None:
            items.append(item(name, notes.get(key), ANNOTATION, unit=unit))
    masses, positions = notes.get("realised_mass") or [], notes.get("positions") or []
    for index, mass in enumerate(masses):
        phone = positions[index] if index < len(positions) else "?"
        items.append(item(f"{DDK_REALISED_MASS}:{index}.{phone}", mass, ANNOTATION))
    return items


def _task_items(task: TaskEvidence) -> list[EvidenceItem]:
    """The task family's readings, as evidence items, before decisiveness is marked.

    Args:
        task: The task evidence the fold read.

    Returns:
        One item per reading the family's decision compared, plus the readings it reports beside them.
    """
    items: list[EvidenceItem] = []
    if task.breath_mode is not None and task.breath_decision is not None:
        items.extend(_breath_items(task))
    if task.cough_mode is not None and task.cough_onsets_n is not None:
        items.extend(_cough_items(task))
    if task.voice_mode is not None and task.voice_found is not None:
        items.extend(_voice_items(task))
    if task.ddk_mode is not None and task.ddk_decision is not None:
        items.extend(_ddk_items(task))
    items.extend(_join_items(task.quality))
    return items


def _join_items(quality: Mapping[str, Any]) -> list[EvidenceItem]:
    """QUALITY's join, as evidence items: what touches a task span, the streams' agreement and the level.

    Args:
        quality: QUALITY's join record.

    Returns:
        One item per fault or interference kind touching a task span, the stream agreement where it was
        read, and the level against the session where it was read.
    """
    items: list[EvidenceItem] = []
    minimum = dict(quality.get("fault_min_s") or {})
    for kind in quality.get("faults_in_task") or ():
        held = ((quality.get("faults") or {}).get(kind) or {}).get("in") or ()
        seconds = round(sum(b - a for a, b in held), 3)
        items.append(
            item(
                f"{PREFIX_FAULT_IN_TASK}:{kind}", seconds, REVIEW, unit="s", comparison="<", threshold=minimum.get(kind)
            )
        )
    for kind in quality.get("interference_in_task") or ():
        held = ((quality.get("interference") or {}).get(kind) or {}).get("in") or ()
        items.append(
            item(f"{PREFIX_INTERFERENCE_IN_TASK}:{kind}", len(held), REVIEW, unit="spans", comparison="==", threshold=0)
        )
    streams = dict(quality.get("streams") or {})
    if streams.get("lost_fraction") is not None:
        items.append(
            item(
                STREAMS_DISAGREE,
                streams.get("lost_fraction"),
                REVIEW if quality.get("streams_disagree") else ANNOTATION,
                comparison="<",
                threshold=quality.get("lost_fraction_max"),
            )
        )
    return items


CAPTURE = "capture"
"""The prefix of the capture items, ``capture.<reading>``: what says whether anything was captured."""


def _capture_items(route_state: str | None, quality: Mapping[str, Any], acoustically_empty: bool) -> list[EvidenceItem]:
    """Whether anything was captured, one item per reading: the route state, each stream's activity, the level.

    Args:
        route_state: What the ruleset made of the whole recording.
        quality: QUALITY's join record.
        acoustically_empty: Whether the fold discarded the recording as having captured nothing.

    Returns:
        ``capture.route_state``, ``capture.plain_active_s`` and ``capture.enhanced_active_s`` where the
        route is empty or the join read the activity, and ``capture.level_rel_db`` where the join read the
        level. A reading that fails its comparison discards where the fold discarded the recording as empty
        on it, and is an annotation otherwise (a task event overrode it, or one stream alone was
        silent); a reading that clears its comparison passes.
    """

    def effect(empty: bool, decides: bool = True) -> str:
        if not empty:
            return PASS
        return DISCARD if acoustically_empty and decides else ANNOTATION

    items: list[EvidenceItem] = []
    plain, enhanced = quality.get("plain_active_s"), quality.get("enhanced_active_s")
    if route_state == EMPTY or plain is not None:
        items.append(
            item(
                f"{CAPTURE}.route_state",
                route_state,
                effect(route_state == EMPTY),
                comparison="not in",
                threshold=[EMPTY],
            )
        )
        no_activity = bool(quality.get("no_activity"))
        for stream, active in (("plain", plain), ("enhanced", enhanced)):
            if active is None:
                continue
            items.append(
                item(
                    f"{CAPTURE}.{stream}_active_s",
                    active,
                    effect(not active > 0.0, no_activity),
                    unit="s",
                    comparison=">",
                    threshold=0.0,
                )
            )
    if quality.get("level_rel_db") is not None:
        items.append(
            item(
                f"{CAPTURE}.level_rel_db",
                quality.get("level_rel_db"),
                effect(bool(quality.get("quiet_vs_session"))),
                unit="dB",
                comparison=">",
                threshold=quality.get("level_rel_db_max"),
            )
        )
    return items


def _decision_evidence(
    *,
    triage: Triage,
    ground: str | None,
    release: Release | None,
    release_ground: str | None,
    task: TaskEvidence,
    route_state: str | None,
    unmeasurable: bool,
    performed: bool,
    flag_keys: Sequence[str],
    annotation_keys: Sequence[str],
    gates: Sequence[Mapping[str, Any]],
    lexical_words_n: int | None,
) -> list[EvidenceItem]:
    """Every reading the fold weighed for this recording, with the decisive ones marked.

    Args:
        triage: The verdict.
        ground: The discard ground, or None.
        release: The release, or None.
        release_ground: The release ground.
        task: The task evidence.
        route_state: What the ruleset made of the whole recording.
        unmeasurable: Whether ADMIT failed the recording.
        performed: Whether the branch owning the declared task found it.
        flag_keys: The ground keys of every flag the fold raised.
        annotation_keys: The annotation keys.
        gates: The conformance and flag gates the fold applied.
        lexical_words_n: The consensus transcript's lexical word count, where SPEECH read it.

    Returns:
        The items in the order the fold reads them: acquisition, task, gates, flags, annotations,
        release. On a discard the item behind the ground is decisive; on a review every review item; on
        a pass every task and gate item that passed; a withholding release item is always decisive.
    """
    items: list[EvidenceItem] = []
    if unmeasurable:
        items.append(item("admit_outcome", "fail", DISCARD, comparison="in", threshold=["pass"]))
    if task.duration_s is not None and task.minimum_duration_s is not None:
        short = task.duration_s < task.minimum_duration_s
        items.append(
            item(
                "task_duration_s",
                round(task.duration_s, 3),
                DISCARD if short else PASS,
                unit="s",
                comparison=">=",
                threshold=task.minimum_duration_s,
            )
        )
    items.extend(_capture_items(route_state, task.quality, ground == ACOUSTICALLY_EMPTY))
    if ground == DECLARED_TASK_ABSENT:
        items.append(item("declared_task_found", performed, DISCARD, comparison="==", threshold=True))
    task_items = _task_items(task)
    items.extend(task_items)
    for record in gates:
        passed = record.get("passed")
        if passed is True or passed is False:
            items.append(
                item(
                    f"gate:{record.get('gate')}",
                    record.get("value"),
                    PASS if passed else REVIEW,
                    comparison=_COMPARISON.get(str(record.get("op"))),
                    threshold=record.get("bound") if str(record.get("op")) in _COMPARISON else None,
                )
            )
    if lexical_words_n is not None:
        items.append(item("lexical_words", lexical_words_n, PASS, unit="words"))
    named = {entry.name for entry in items}
    if task.owner_absent_inputs and KEY_OWNING_BRANCH_INPUT_ABSENT in flag_keys:
        items.extend(_flag_item(f"{KEY_OWNING_BRANCH_INPUT_ABSENT}:{absent}") for absent in task.owner_absent_inputs)
        named.add(KEY_OWNING_BRANCH_INPUT_ABSENT)
    items.extend(_flag_item(key) for key in flag_keys if key not in named)
    items.extend(_flag_item(key, ANNOTATION) for key in annotation_keys)
    if triage is not Triage.DISCARD:
        effect = WITHHOLD if release is Release.WITHHELD else REVIEW if release is None else PASS
        items.append(item("release_ground", release_ground_key(release_ground), effect))
    decisive_discard = _DISCARD_ITEMS.get(ground or "")
    task_names = {entry.name for entry in task_items} - _INSTRUCTION_ITEMS

    def decides(entry: EvidenceItem) -> bool:
        if entry.effect == WITHHOLD:
            return True
        if triage is Triage.DISCARD:
            return entry.name == decisive_discard or (
                entry.effect == DISCARD and entry.name.startswith(f"{decisive_discard}.")
            )
        if triage is Triage.REVIEW:
            return entry.effect == REVIEW
        return entry.effect == PASS and (entry.name in task_names or entry.name.startswith("gate:"))

    return [replace(entry, decisive=decides(entry)) for entry in items]


def fold_file_verdict(
    node_verdicts: Sequence[NodeVerdict],
    *,
    branch_reports: Sequence[BranchReport] = (),
    spans_by_node: Mapping[str, int] | None = None,
    branch_decisions: Mapping[str, BranchDecision],
    ran: Mapping[str, RunState],
    hint_claims: Mapping[str, bool] | None,
    route_state: str | None,
    declared_family: str | None = None,
    redaction: RedactionEvidence | None = None,
    llm_redaction: Mapping[str, Any] | None = None,
    critical_absences: Mapping[str, Mapping[str, str]] | None = None,
    gates: Mapping[str, Any] | None = None,
    flag_gates: Sequence[Mapping[str, Any]] | None = None,
    policy: FoldPolicy | None = None,
    agreed_redactions: frozenset[int] = frozenset(),
    unplaced: Sequence[tuple[str, str]] = (),
    second_opinion: Mapping[str, Any] | None = None,
    task: TaskEvidence | None = None,
) -> FileVerdict:
    """Decide the file, from the deciding nodes' verdicts and the reporting nodes' reports.

    Flags on every ground it finds and discards on four: ``unmeasurable``, which is ADMIT's own fail;
    ``too_short_for_task``, a recording shorter than its family's minimum; ``acoustically_empty``,
    the ruleset's :data:`EMPTY` state where the declared task was not performed; and
    ``declared_task_absent``, where the branch owning the declared task ran and found none of it --
    tried after the operational grounds, which make the file ``rerun`` instead.
    Only the declared task's own branch can say the task was performed, and only by its task
    conformance, never by a span alone (:func:`declared_task_performed`). The grounds, the two
    axes and the agreement and hint tables are in
    ``specs/20260817-triage-workflow-dag/verdict.md``.

    Args:
        node_verdicts: Every deciding node's conclusion, in graph order, one per node.
        branch_reports: Every reporting node's report — the three branches and QUALITY — one per
            node. A report joins to its routing decision by node name.
        spans_by_node: How many spans each node proposed into its own family, keyed by node; a
            node absent from it proposed none, and None is the same as empty.
        branch_decisions: What ROUTING decided per branch, keyed by branch name.
        ran: Whether each node ran, keyed by node name.
        hint_claims: Which branches the caller's declaration claimed, keyed by branch, or None
            when nothing in the store can say what a supplied declaration claimed, which empties
            ``hints`` and flags.
        route_state: What the ruleset made of the whole recording, or None when ``routing`` wrote no
            evaluation.
        declared_family: The task family the recording declares, or None. Every conformance
            ground is read against it, under ``policy.conformance_flags_by_family``.
        redaction: What the store says about whether this recording carried anything redactable —
            SPEECH's lexical count, its scan record and the live findings. None is the same as
            :class:`RedactionEvidence` with nothing in it.
        llm_redaction: REVIEW's annotation, or None where it wrote none. A reading that proposes
            hiding more reaches the **triage** axis under ``policy.llm_redaction_flags`` and the
            release axis under ``policy.llm_redaction_withholds``; any other reading reaches neither.
        critical_absences: Per branch not one of whose gates could be read, each gate and the
            recorded absence behind it, as ``routing`` wrote them. Non-empty flags, never discards.
        gates: The task group's gates and the ones this fold's caller applied to the declared
            task's readings, carried onto the verdict so the conformance can be read backwards.
        flag_gates: The gates the caller applied that decide no conformance, each as its own
            ``AppliedGate`` record. One that did not pass is a flag ground, named by its
            ``ground``; one that answered :data:`UNDETERMINED` is never one.
        policy: What to do with what was reported, from the ``verdict.*`` config section. None is
            the packaged policy.
        agreed_redactions: The ``proposal`` positions of reviewer ``redact`` entries that agree with
            the masks; they propose nothing more (:func:`deciding_reading`).
        second_opinion: The ``second_opinion_answers`` measurement's attributes, or None. A confident
            disagreement with the reviewer is a flag ground under ``policy.second_opinion_disagreement_flags``
            (:func:`second_opinion_disagreements`); the release is unchanged.
        task: Whether the declared task was performed at all: its owning branches, the recording's
            duration against the family's minimum, and the ASR event tokens naming its event. None
            is :class:`TaskEvidence` with nothing in it, under which the owning branch is unknown.
        unplaced: ``(family, state)`` for every detector finding SPEECH could not place on words, as
            :class:`~senselab.audio.workflows.triage.nodes.redact.UnplacedFinding` records them. An
            ``open`` one flags; an ``unread`` one flags and withholds.

    Returns:
        The file verdict on both axes, carrying every contributing reason rather than only the
        deciding one.
    """
    rules = policy or FoldPolicy()
    task_evidence = task or TaskEvidence()
    claims = hint_claims or {}
    spans = dict(spans_by_node or {})
    reports = {report.node: report for report in branch_reports}
    by_branch = {name: report for name, report in reports.items() if name in BRANCHES}
    branches_seen = list(dict.fromkeys([*branch_decisions, *by_branch, *claims]))

    routes = {
        branch: branch_decisions[branch].route_state if branch in branch_decisions else UNJUDGED
        for branch in branches_seen
    }
    findings = {branch: _found(branch in by_branch, spans.get(branch, 0)) for branch in branches_seen}
    breath_decides = task_evidence.breath_mode is not None and task_evidence.breath_decision is not None
    if breath_decides and _AIRWAY in findings:
        findings[_AIRWAY] = KindState.PRESENT.value if breath_present(task_evidence) else KindState.ABSENT.value
    cough_decides = task_evidence.cough_mode is not None and task_evidence.cough_onsets_n is not None
    if cough_decides and _AIRWAY in findings:
        findings[_AIRWAY] = KindState.PRESENT.value if cough_present(task_evidence) else KindState.ABSENT.value
    voice_decides = task_evidence.voice_mode is not None and task_evidence.voice_found is not None
    if voice_decides and _VOICE in findings:
        findings[_VOICE] = KindState.PRESENT.value if task_evidence.voice_found else KindState.ABSENT.value
    ddk_decides = task_evidence.ddk_mode is not None and task_evidence.ddk_decision is not None
    if ddk_decides and _SPEECH in findings:
        findings[_SPEECH] = KindState.PRESENT.value if ddk_present(task_evidence) else KindState.ABSENT.value
    agreement = {branch: _agreement(routes[branch], branch in by_branch, findings[branch]) for branch in branches_seen}
    hints = (
        {}
        if hint_claims is None
        else {
            branch: _hint_reading(bool(claims.get(branch, False)), findings[branch] == KindState.PRESENT.value)
            for branch in branches_seen
        }
    )

    bad_map_values: dict[str, str] = {}
    for recorded in branch_decisions.values():
        bad_map_values.update(recorded.bad_map_values)

    reasons = [replace(verdict, key=ground_key(verdict)) for verdict in node_verdicts]

    def flag(node: str, why: str, key: str, kind: str | None = None) -> None:
        reasons.append(NodeVerdict(node, Outcome.FLAG, kind, why, key))

    annotations: list[NodeVerdict] = []

    def annotate(node: str, why: str, key: str, kind: str | None = None) -> None:
        annotations.append(NodeVerdict(node, Outcome.PASS, kind, why, key))

    if ran.get(_PREPROCESS) is RunState.ERRORED:
        flag(
            _PREPROCESS,
            "preprocess failed; no derivative was measured because conditioning itself did not complete",
            KEY_PREPROCESS_ERRORED,
        )
    if ran.get(_ROUTING) is RunState.ERRORED:
        flag(
            _ROUTING,
            "routing failed; branch execution was withheld because no complete routing result was available",
            KEY_ROUTING_ERRORED,
        )
    if bad_map_values:
        named = ", ".join(f"{tag}: {value}" for tag, value in sorted(bad_map_values.items()))
        flag(_ROUTING, f"{BAD_MAP_VALUES}: {named}", KEY_BAD_HINT_MAP)
    if hint_claims is None:
        flag(_VERDICT, UNREAD_DECLARATION, KEY_DECLARATION_UNREAD)
    if route_state == UNEXPLAINED and not (cough_decides and cough_present(task_evidence)):
        flag(_ROUTING, UNEXPLAINED_CONTENT, KEY_ROUTE_UNEXPLAINED)
    if route_state == UNREADABLE:
        flag(_ROUTING, UNREADABLE_EMPTINESS, KEY_ROUTE_UNREADABLE)
    absences = {branch: dict(gates) for branch, gates in (critical_absences or {}).items()}
    if absences:
        named = "; ".join(
            f"{branch}: " + ", ".join(f"{gate} ({why})" for gate, why in sorted(gates.items()))
            for branch, gates in sorted(absences.items())
        )
        flag(_ROUTING, f"{CRITICAL_ABSENCE}: {named}", KEY_CRITICAL_ABSENCE)
    if (
        (redaction or RedactionEvidence()).lexical_words_n == 0
        and ran.get(_SPEECH) is RunState.COMPLETED
        and routes.get(_SPEECH) == ROUTED
    ):
        (annotate if task_evidence.ddk_mode is not None else flag)(
            _SPEECH, NO_LEXICAL_ITEM_PRODUCED, KEY_NO_LEXICAL_ITEM
        )
    annotation = dict(llm_redaction or {})
    deciding = deciding_reading(annotation, agreed_redactions, rules.condition_categories)
    if rules.llm_redaction_flags and _reviewer_found_residue(deciding):
        named = ", ".join(
            sorted(
                {
                    str(entry.get("category"))
                    for entry in deciding.get("proposal") or ()
                    if str(entry.get("action")) == "redact" and entry.get("category")
                }
            )
        )
        flag(
            _VERDICT,
            f"{LLM_REDACTION_RESIDUE}: {named}" if named else LLM_REDACTION_RESIDUE,
            KEY_REVIEWER_RESIDUE,
        )
    evidence = redaction or RedactionEvidence()
    if rules.person_name_review_flags and evidence.person_names_masked_n > 0:
        proposed_n = len(evidence.name_release_proposed)
        why = f"{PERSON_NAME_AWAITS_REVIEW}: {evidence.person_names_masked_n} name word(s) masked" + (
            f"; release proposed for {proposed_n} name(s)" if proposed_n else ""
        )
        flag(_VERDICT, why, KEY_PERSON_NAME_REVIEW)
    diarized_other = any(
        record.get("passed") is False and record.get("gate") == DOMINANT_SPEAKER_GATE for record in flag_gates or ()
    )
    others = [dict(other) for other in annotation.get("other_speakers") or () if isinstance(other, Mapping)]
    heard_other = annotation.get("speakers") == "more_than_one" or (annotation.get("speakers") == "unclear" and others)
    if rules.llm_second_speaker_flags and heard_other and not diarized_other:
        expected_n = sum(1 for other in others if other.get("expected") is True)
        counted = (
            f"{len(others)} passage(s) quoted, {expected_n} expected by the instructions, "
            f"{len(others) - expected_n} not"
            if others
            else "no words quoted"
        )
        flag(_VERDICT, f"{REVIEWER_HEARD_SECOND_SPEAKER}: {counted}", KEY_REVIEWER_SECOND_SPEAKER)
    if rules.llm_contradiction_flags and reviewer_named_no_words(annotation):
        flag(_VERDICT, REVIEWER_NAMED_NO_WORDS, KEY_REVIEWER_NAMED_NO_WORDS)
    spoken = [str(text) for text in annotation.get("instructions_spoken") or () if str(text).strip()]
    if rules.llm_instructions_spoken_flags and spoken:
        flag(_VERDICT, f"{INSTRUCTIONS_SPOKEN}: {len(spoken)} passage(s) quoted", KEY_INSTRUCTIONS_SPOKEN)
    disagreements = second_opinion_disagreements(
        second_opinion,
        annotation,
        confident_yes=rules.second_opinion_confident_yes,
        confident_no=rules.second_opinion_confident_no,
        identifier_masked=evidence.masks_final_n > 0 or evidence.reviewer_requested_n > 0,
    )
    clef_decides = not {_AIRWAY, _VOICE} & set(task_evidence.owning_branches)
    if rules.second_opinion_disagreement_flags and disagreements and clef_decides:
        flag(_VERDICT, f"{SECOND_OPINION_DISAGREES}: {'; '.join(disagreements)}", KEY_SECOND_OPINION_DISAGREES)
    open_families = sorted({family for family, state in unplaced if state in (UNPLACED_OPEN, UNPLACED_UNREAD)})
    if open_families:
        unread_any = any(state == UNPLACED_UNREAD for _, state in unplaced)
        flag(
            _VERDICT,
            f"{UNPLACED_FINDING_UNREAD if unread_any else UNPLACED_FINDING_OPEN}: {', '.join(open_families)}",
            KEY_UNPLACED_UNREAD if unread_any else KEY_UNPLACED_OPEN,
        )
    for record in flag_gates or ():
        if record.get("passed") is not False or task_evidence.voice_mode is not None:
            continue
        flag(
            _VERDICT,
            f"{record.get('ground', record.get('gate'))}: "
            f"{record.get('reading')} read {record.get('value')} against {record.get('bound')}"
            + (
                f"; {MODEL_SPEAKER_PERMITTED}"
                if record.get("gate") == DOMINANT_SPEAKER_GATE and declared_family in rules.model_speaker_families
                else ""
            ),
            f"{PREFIX_GATE}:{record.get('gate')}",
        )
    gate_record = dict(gates or {})
    applied_gates = [g for g in (gate_record.get("applied") or ()) if isinstance(g, Mapping)]
    nothing_read = any(
        g.get("reading") == "expected_tokens_matched" and g.get("value") == 0 and g.get("passed") is False
        for g in applied_gates
    )
    uncomputed = sorted(
        f"{g.get('gate')} ({g.get('reason')})"
        for g in applied_gates
        if g.get("passed") == "UNDETERMINED" and g.get("reason") in ("absent_not_computed", "instrument_absent")
    )
    measure_mode = task_evidence.breath_mode or task_evidence.cough_mode
    measure_decides_gate_node = (
        (measure_mode is not None and gate_record.get("node") == _AIRWAY)
        or (task_evidence.voice_mode is not None and gate_record.get("node") == _VOICE)
        or (task_evidence.ddk_mode is not None and gate_record.get("node") == _SPEECH)
    )
    if uncomputed and rules.uncomputed_reading_flags and not measure_decides_gate_node:
        flag(
            str(gate_record.get("node") or _VERDICT),
            f"{UNCOMPUTED_READING}: {', '.join(uncomputed)}",
            KEY_UNCOMPUTED_READING,
        )
    for name, report in reports.items():
        measure_decides = (
            (measure_mode is not None and name == _AIRWAY)
            or (task_evidence.voice_mode is not None and name == _VOICE)
            or (task_evidence.ddk_mode is not None and name == _SPEECH)
        ) and report.conformance_of == TASK
        if measure_decides:
            if report.deviations and rules.deviation_flags:
                flag(name, f"{name} reported {', '.join(report.deviations)}", f"{PREFIX_DEVIATION}:{name}", report.kind)
            continue
        if report.conformance is False and rules.flags_conformance(report.conformance_of, declared_family):
            why = TASK_NOT_CONFORMED if report.conformance_of == TASK else STORE_ASSERTION_CONTRADICTED
            prefix = PREFIX_CONFORMANCE if report.conformance_of == TASK else PREFIX_STORE_ASSERTION
            if report.conformance_of == TASK and nothing_read and name == gate_record.get("node"):
                why = NOTHING_READ
                prefix = PREFIX_NOTHING_READ
            mismatch = (
                _count_mismatch(applied_gates, task_evidence)
                if report.conformance_of == TASK and name == gate_record.get("node")
                else None
            )
            named = f" on {declared_family}" if report.conformance_of == TASK and declared_family else ""
            if mismatch is not None:
                annotate(name, f"{TASK_MISMATCH}: {mismatch}", KEY_TASK_MISMATCH, report.kind)
            else:
                flag(name, f"{name} {why}{named}", f"{prefix}:{name}", report.kind)
        if report.conformance == UNDETERMINED and rules.undetermined_flags:
            flag(name, f"{name} {CONFORMANCE_UNANSWERED}", f"{PREFIX_CONFORMANCE_UNANSWERED}:{name}", report.kind)
        if report.deviations and rules.deviation_flags:
            flag(name, f"{name} reported {', '.join(report.deviations)}", f"{PREFIX_DEVIATION}:{name}", report.kind)
        if report.unmeasured and rules.unmeasured_points_flag:
            flag(
                name,
                f"{name} {UNMEASURED_ASKED}: {', '.join(report.unmeasured)}",
                f"{PREFIX_UNMEASURED}:{name}",
                report.kind,
            )
    shortfall = breath_shortfall(task_evidence)
    if shortfall is not None:
        annotate(
            _AIRWAY,
            f"{TASK_MISMATCH}: {shortfall}",
            KEY_TASK_MISMATCH,
            by_branch[_AIRWAY].kind if _AIRWAY in by_branch else None,
        )
    if task_evidence.breath_review and breath_present(task_evidence):
        flag(
            _AIRWAY,
            f"{BREATH_REVIEW_LOW_CONFIDENCE}: {breath_review_reading(task_evidence)}",
            KEY_BREATH_REVIEW_LOW_CONFIDENCE,
            by_branch[_AIRWAY].kind if _AIRWAY in by_branch else None,
        )
    cough_short = cough_shortfall(task_evidence)
    if cough_short is not None:
        annotate(
            _AIRWAY,
            f"{TASK_MISMATCH}: {cough_short}",
            KEY_TASK_MISMATCH,
            by_branch[_AIRWAY].kind if _AIRWAY in by_branch else None,
        )
    if task_evidence.cough_review and cough_present(task_evidence):
        flag(
            _AIRWAY,
            f"{COUGH_REVIEW_LOW_CONFIDENCE}: {cough_review_reading(task_evidence)}",
            KEY_COUGH_REVIEW_LOW_CONFIDENCE,
            by_branch[_AIRWAY].kind if _AIRWAY in by_branch else None,
        )
    voice_kind = by_branch[_VOICE].kind if _VOICE in by_branch else None
    if task_evidence.voice_mismatch is not None and task_evidence.voice_found:
        annotate(_VOICE, f"{TASK_MISMATCH}: {task_evidence.voice_mismatch}", KEY_TASK_MISMATCH, voice_kind)
    if task_evidence.voice_review and task_evidence.voice_found:
        flag(
            _VOICE,
            f"{VOICE_REVIEW_LOW_CONFIDENCE}: {', '.join(task_evidence.voice_review)}",
            KEY_VOICE_REVIEW_LOW_CONFIDENCE,
            voice_kind,
        )
    speech_kind = by_branch[_SPEECH].kind if _SPEECH in by_branch else None
    ddk_short = ddk_shortfall(task_evidence)
    if ddk_short is not None:
        annotate(_SPEECH, f"{TASK_MISMATCH}: {ddk_short}", KEY_TASK_MISMATCH, speech_kind)
    if task_evidence.ddk_decision == "review":
        identity = task_evidence.ddk_reading.get("why") == "identity"
        flag(
            _SPEECH,
            f"{DDK_REVIEW_IDENTITY if identity else DDK_REVIEW_WEAK_EVENTS}: {ddk_review_reading(task_evidence)}",
            KEY_DDK_REVIEW_IDENTITY if identity else KEY_DDK_REVIEW_WEAK_EVENTS,
            speech_kind,
        )
    if task_evidence.voice_outside_speech:
        said = [word for run in task_evidence.voice_outside_speech for word in run.get("words") or ()]
        annotate(
            _VOICE,
            f"{SPEECH_OUTSIDE_TASK}: {len(task_evidence.voice_outside_speech)} run(s)"
            + (f", {len(said)} word(s)" if said else ""),
            KEY_SPEECH_OUTSIDE_TASK,
            voice_kind,
        )
    _join_reasons(task_evidence.quality, flag, annotate)
    for branch in branches_seen:
        decision = branch_decisions.get(branch)
        reported = by_branch.get(branch)
        kind = reported.kind if reported is not None else None
        # Only MISMATCH-and-PRESENT is a flag ground, and where the declared task names its owning
        # branches only that branch's task-conformant finding; both directions stay in ``agreement``.
        if (
            agreement[branch] == MISMATCH
            and findings[branch] == KindState.PRESENT.value
            and not (cough_decides and branch == _AIRWAY)
            and (
                not task_evidence.owning_branches
                or (
                    branch in task_evidence.owning_branches
                    and _owner_performed(branch, by_branch, findings, task_evidence)
                )
            )
        ):
            note = annotate if (voice_decides and branch == _VOICE) or (ddk_decides and branch == _SPEECH) else flag
            note(
                branch,
                f"mismatch: routing {routes[branch]} {branch}, it found it",
                f"{PREFIX_ROUTE_MISMATCH}:{branch}",
                kind,
            )
        if decision is not None and decision.will_run and reported is None:
            flag(
                branch,
                f"{branch} was asked to run and {_silence(ran.get(branch))}",
                f"{PREFIX_BRANCH_SILENT}:{branch}",
                kind,
            )
        if hints.get(branch) == CLAIMED_NOT_FOUND and declared_family not in rules.hint_mismatch_exempt_families:
            flag(
                branch,
                f"hint mismatch: {branch} was declared and did not find it",
                f"{PREFIX_HINT_MISMATCH}:{branch}",
                kind,
            )

    branch_view = {
        name: {
            "will_run": decision.will_run,
            "forced_by_declaration": decision.forced_by_declaration,
            "route_state": decision.route_state,
            "withheld_critical": decision.withheld_critical,
            "conformance": by_branch[name].conformance if name in by_branch else None,
        }
        for name, decision in branch_decisions.items()
    }

    performed = declared_task_performed(by_branch, findings, task_evidence)
    owner_unable = bool(task_evidence.owner_absent_inputs) and not performed
    if owner_unable:
        flag(
            task_evidence.owning_branches[0] if task_evidence.owning_branches else _VERDICT,
            f"{OWNING_BRANCH_INPUT_ABSENT}: {', '.join(task_evidence.owner_absent_inputs)}",
            KEY_OWNING_BRANCH_INPUT_ABSENT,
        )
    admit = next((reason for reason in reasons if reason.node == _ADMIT), None)
    unmeasurable = admit is not None and admit.outcome is Outcome.FAIL
    too_short = (
        task_evidence.duration_s is not None
        and task_evidence.minimum_duration_s is not None
        and task_evidence.duration_s < task_evidence.minimum_duration_s
    )
    empty = (
        (route_state == EMPTY or nothing_captured(task_evidence.quality))
        and not performed
        and not owner_unable
        and not task_event_heard(task_evidence, (redaction or RedactionEvidence()).lexical_words_n)
    )
    no_event = no_required_event(task_evidence, performed)
    contested = no_event and not (unmeasurable or too_short or empty) and discard_contested(task_evidence)
    if contested:
        flag(
            _AIRWAY,
            f"{DISCARD_CONTESTED}: the measure found no {task_evidence.required_event}, AIRWAY's detector found "
            f"{task_evidence.events_found_n}",
            KEY_DISCARD_CONTESTED,
            by_branch[_AIRWAY].kind if _AIRWAY in by_branch else None,
        )
    flags = [reason for reason in reasons if reason.outcome is Outcome.FLAG]
    ground: str | None = None
    if unmeasurable and admit is not None:
        triage = Triage.DISCARD
        ground = UNMEASURABLE
        reasons = [admit, *(reason for reason in reasons if reason is not admit)]
    elif too_short:
        triage = Triage.DISCARD
        ground = TOO_SHORT_FOR_TASK
    elif empty:
        triage = Triage.DISCARD
        ground = ACOUSTICALLY_EMPTY
    elif any(is_operational(reason.key) for reason in flags):
        triage = Triage.REVIEW
    elif no_event and not contested:
        triage = Triage.DISCARD
        ground = (
            ddk_absent_ground(task_evidence)
            if task_evidence.ddk_mode is not None
            else EVENT_ABSENT_GROUNDS[str(task_evidence.required_event)]
        )
    elif not contested and declared_task_absent(
        by_branch, findings, task_evidence, (redaction or RedactionEvidence()).lexical_words_n
    ):
        triage = Triage.DISCARD
        ground = DECLARED_TASK_ABSENT
    elif flags:
        triage = Triage.REVIEW
    else:
        triage = Triage.PASS

    withholds = rules.llm_redaction_withholds and _reviewer_found_residue(deciding)
    clears = rules.llm_rescan_clears and _reviewer_cleared(deciding)
    unread = any(state == UNPLACED_UNREAD for _, state in unplaced)
    withholding = REVIEWER_PROPOSED_REDACTION if withholds else UNPLACED_FINDING_UNREAD if unread else None
    release, release_ground = _release_from(
        node_verdicts,
        redaction or RedactionEvidence(),
        ran,
        withholding,
        speech_declined=routes.get(_SPEECH) == DECLINED,
        reviewer_clears=clears,
    )
    missing = sorted({ground_key(reason) for reason in flags if is_operational(reason.key)})
    if triage is Triage.DISCARD:
        release, release_ground = None, DISCARDED
    elif release is None:
        triage = Triage.REVIEW
        missing.append(release_ground_key(release_ground))
    elif missing:
        triage = Triage.REVIEW
    run_status = RunStatus.INCOMPLETE if triage is not Triage.DISCARD and missing else RunStatus.COMPLETE
    ground_keys = sorted({*([ground] if ground else []), *(ground_key(reason) for reason in flags)})
    annotation_keys = sorted({str(annotation.key) for annotation in annotations})
    held = release is Release.WITHHELD or (release is None and triage is not Triage.DISCARD)
    decision_reasons = reasons_of(
        ground_keys,
        release_ground_key(release_ground) if held else None,
    )
    weighed_gates = [
        *(applied_gates if not measure_decides_gate_node else ()),
        *(flag_gates or () if task_evidence.voice_mode is None else ()),
    ]
    evidence_items = _decision_evidence(
        triage=triage,
        ground=ground,
        release=release,
        release_ground=release_ground,
        task=task_evidence,
        route_state=route_state,
        unmeasurable=unmeasurable,
        performed=performed,
        flag_keys=[ground_key(reason) for reason in flags],
        annotation_keys=annotation_keys,
        gates=weighed_gates,
        lexical_words_n=(redaction or RedactionEvidence()).lexical_words_n
        if ran.get(_SPEECH) is RunState.COMPLETED
        else None,
    )

    return FileVerdict(
        triage=triage,
        release=release,
        run_status=run_status,
        missing=missing if run_status is RunStatus.INCOMPLETE else [],
        discard_ground=ground,
        release_ground=release_ground,
        ground_keys=ground_keys,
        annotation_keys=annotation_keys,
        reason=decision_reasons[0] if decision_reasons else None,
        reason_keys=decision_reasons,
        evidence=evidence_items,
        annotations=annotations,
        findings=findings,
        conformance={name: report.conformance for name, report in reports.items()},
        conformance_of={name: report.conformance_of for name, report in reports.items()},
        deviations={name: list(report.deviations) for name, report in reports.items()},
        unmeasured={name: list(report.unmeasured) for name, report in reports.items() if report.unmeasured},
        declared_family=declared_family,
        routes=routes,
        route_state=route_state,
        agreement=agreement,
        hints=hints,
        reasons=reasons,
        ran=dict(ran),
        branches=branch_view,
        bad_map_values=bad_map_values,
        llm_redaction=annotation,
        second_opinion=(
            {
                "status": second_opinion.get("status"),
                "probabilities": dict(second_opinion.get("probabilities") or {}),
                "disagreements": list(disagreements),
                "model_id": second_opinion.get("model_id"),
                "blob_digest": second_opinion.get("blob_digest"),
            }
            if second_opinion
            else {}
        ),
        critical_absences=absences,
        gates=dict(gates or {}),
        breath_pattern=dict(task_evidence.breath_reading),
        cough_pattern=dict(task_evidence.cough_reading),
        quality_join=dict(task_evidence.quality),
        voice_phonation=dict(task_evidence.voice_reading),
        ddk_task=dict(task_evidence.ddk_reading),
    )


TASK_EXTENT_SPAN_ROLE = "task_extent"
SUPERSEDES = "supersedes"


_Span = TypeVar("_Span")


def standing_task_extents(spans: Sequence[_Span]) -> list[_Span]:
    """The task-extent spans that stand: those marked as superseding the rest, where any are.

    Args:
        spans: Live spans of any role.

    Returns:
        The live ``task_extent`` spans carrying ``supersedes`` where any does, else every live
        ``task_extent`` span.
    """
    held: list[Any] = list(spans)
    extents = [s for s in held if s.attributes.get("role") == TASK_EXTENT_SPAN_ROLE and s.extent is not None]
    superseding = [s for s in extents if s.attributes.get(SUPERSEDES) is not None]
    return superseding or extents
