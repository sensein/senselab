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
from typing import Any, Literal, Mapping, Sequence

GRAPH_ORDER = (
    "ADMIT",
    "PREPROCESS",
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
"""The nodes the runner drives, in the order it drives them. VERDICT folds the ten before it."""

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
    """What should happen to this recording. The file axis; a node's ``Outcome`` is not one of these.

    ``rerun`` is a recording the pipeline still owes something -- a missing derivative, a node that did
    not finish, a configuration the fold cannot read -- before anything about the participant can be
    concluded; :data:`OPERATIONAL_GROUND_KEYS` names those grounds.
    """

    PASS = "pass"
    FLAG = "flag"
    RERUN = "rerun"
    DISCARD = "discard"


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

    Four values over one question, total and exclusive: the recording as recorded, only REDACT's
    redacted copy, neither, or the graph cannot say.
    ``specs/20260924-which-artefact-is-releasable/design.md`` holds the vocabulary and what each
    value permits; ``specs/20260817-triage-workflow-dag/verdict.md`` holds the fold.
    """

    WITHOUT_REDACTION = "release_without_redaction"
    WITH_REDACTION = "release_with_redaction"
    WITHHELD = "withheld"
    NOT_ASSESSED = "not_assessed"


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
"""Which reading cleared the recording. One stands behind every :attr:`Release.WITHOUT_REDACTION`."""

NO_TRANSCRIPT = "SPEECH did not run, so nothing read the recording for content a redaction would remove"
SPEECH_UNREAD = "SPEECH left no lexical count, so whether the recording carries redactable content is unknown"
REDACTION_OWED = "the scan found content to redact and REDACT left no verdict over it"
SCAN_UNRECORDED = "SPEECH read lexical words and recorded no scan either way"

RELEASE_UNKNOWN_GROUNDS = (NO_TRANSCRIPT, SPEECH_UNREAD, REDACTION_OWED, SCAN_UNRECORDED)
"""Why the graph could not tell. One of these stands behind every :attr:`Release.NOT_ASSESSED`."""

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
"""Why a recording is withheld. One stands behind every :attr:`Release.WITHHELD`."""

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
    """

    owning_branches: tuple[str, ...] = ()
    duration_s: float | None = None
    minimum_duration_s: float | None = None
    event_tokens_n: int = 0
    owner_absent_inputs: tuple[str, ...] = ()


UNMEASURABLE = "unmeasurable"
ACOUSTICALLY_EMPTY = "acoustically_empty"
TOO_SHORT_FOR_TASK = "too_short_for_task"
"""Discard ground: the recording is far shorter than its declared task can take, a truncated or aborted capture."""
DECLARED_TASK_ABSENT = "declared_task_absent"
"""Discard ground: the branch owning the declared task ran and found none of it, whatever another branch found."""
DISCARD_GROUNDS = (UNMEASURABLE, TOO_SHORT_FOR_TASK, ACOUSTICALLY_EMPTY, DECLARED_TASK_ABSENT)
"""Every ground a file discards on, in the order the fold tries them; an operational flag (``rerun``) is
tried between the third and the fourth, since a missing derivative may be why the task was not found."""

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
A flag on one of these makes the file ``rerun`` (:attr:`Triage.RERUN`)."""

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
        release: Which artefact of the recording may be handed on. Never describes the store.
        release_ground: Why the release axis reads as it does, in controlled vocabulary, wherever
            REDACT did not decide it — one of :data:`RELEASE_WITHOUT_REDACTION_GROUNDS`,
            :data:`RELEASE_UNKNOWN_GROUNDS`, :data:`RELEASE_WITHHELD_GROUNDS` or
            :data:`RELEASE_WITH_REDACTION_GROUNDS`. None wherever REDACT itself decided.
        discard_ground: ``"unmeasurable"``, ``"too_short_for_task"``, ``"acoustically_empty"``,
            ``"declared_task_absent"`` or None.
        ground_keys: The stable key of every ground behind the triage state -- the discard ground and
            every flag, sorted and deduplicated. Empty on a pass.
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
    """

    triage: Triage
    release: Release
    discard_ground: str | None = None
    release_ground: str | None = None
    ground_keys: list[str] = field(default_factory=list)
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

    def record(self) -> dict[str, Any]:
        """Every decision point of this fold, as JSON-ready values.

        Categorical throughout — outcomes, states, type names and config paths, never transcript
        text or a detected string.

        Returns:
            The decision, keyed as :class:`FileVerdict` names its fields.
        """
        return {
            "triage": self.triage.value,
            "release": self.release.value,
            "discard_ground": self.discard_ground,
            "release_ground": self.release_ground,
            "release_ground_key": release_ground_key(self.release_ground),
            "ground_keys": list(self.ground_keys),
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
) -> tuple[Release, str | None]:
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
            withholding, whether or not REDACT ran, and never touches a withholding or a
            ``not_assessed``.
        speech_declined: Whether the ruleset declined SPEECH, in which case a task that carries no
            lexical content by construction is released without redaction.
        reviewer_clears: Whether the reviewer read the original as clean, proposed nothing to hide,
            and the policy lets that release. It moves only a REDACT ``fail`` whose re-scan still read
            a finding, and only to the redacted copy.

    Returns:
        Which artefact may be handed on, never anything about the store, and the ground behind it.
        A REDACT ``pass`` clears the redacted copy and not the original; the ground is None only where
        that pass decided and a planned mask stands, and one of the controlled grounds otherwise. A
        REDACT ``fail`` or ``flag`` that nothing clears withholds on :data:`REDACT_VERIFY_FOUND` or
        :data:`REDACT_UNRESOLVED`.
        A copy that would mask nothing is never released as a redacted copy: the original is.
    """
    release, ground = _release_from_evidence(node_verdicts, evidence, ran, speech_declined)
    if reviewer_withholds is not None and release in (Release.WITH_REDACTION, Release.WITHOUT_REDACTION):
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
        release, ground = Release.WITH_REDACTION, REVIEWER_CLEARED_RESCAN
    if (
        release is Release.WITHOUT_REDACTION
        and ground == SCAN_FOUND_NOTHING
        and redact is None
        and evidence.masks_final_n > 0
        and evidence.policy_masks_n > 0
    ):
        return Release.WITH_REDACTION, POLICY_MASKS_ONLY
    if release is Release.WITHHELD and ground is None:
        verify_found = redact is not None and redact.outcome is Outcome.FAIL and bool(evidence.rescan_survivors)
        ground = REDACT_VERIFY_FOUND if verify_found else REDACT_UNRESOLVED
    if release is not Release.WITH_REDACTION:
        return release, ground
    reviewer = evidence.reviewer_unmasked_n > 0
    if evidence.masks_final_n == 0:
        if evidence.masks_changed:
            return Release.WITHOUT_REDACTION, REVIEWER_UNMASKED_ALL if reviewer else NO_CONTENT_MASKED
        if ground is None:
            return Release.WITHOUT_REDACTION, FINDINGS_ARE_TASK_CONTENT
        if ground == REVIEWER_CLEARED_RESCAN:
            return Release.WITHOUT_REDACTION, REVIEWER_CLEARED_UNMASKED
    if evidence.masks_changed:
        if reviewer:
            return Release.WITH_REDACTION, REVIEWER_UNMASKED_SOME
        return Release.WITH_REDACTION, POLICY_MASKS_ADDED if evidence.policy_masks_n else MASKS_TRIMMED_TO_CONTENT
    return release, ground


def _release_from_evidence(
    node_verdicts: Sequence[NodeVerdict],
    evidence: RedactionEvidence,
    ran: Mapping[str, RunState],
    speech_declined: bool,
) -> tuple[Release, str | None]:
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
        return (Release.WITH_REDACTION if redact.outcome is Outcome.PASS else Release.WITHHELD), None
    if evidence.findings_n > 0:
        return Release.NOT_ASSESSED, REDACTION_OWED
    speech = ran.get(_SPEECH)
    if speech is RunState.ERRORED:
        return Release.NOT_ASSESSED, SPEECH_UNREAD
    if evidence.lexical_words_n is None:
        if speech is RunState.COMPLETED:
            return Release.NOT_ASSESSED, SPEECH_UNREAD
        if speech_declined:
            return Release.WITHOUT_REDACTION, NON_LEXICAL_TASK
        return Release.NOT_ASSESSED, NO_TRANSCRIPT
    if evidence.lexical_words_n == 0:
        return Release.WITHOUT_REDACTION, NO_LEXICAL_WORD
    if evidence.scanned is False:
        return Release.WITHOUT_REDACTION, NOTHING_BEYOND_STIMULUS
    if evidence.scanned is True:
        return Release.WITHOUT_REDACTION, SCAN_FOUND_NOTHING
    return Release.NOT_ASSESSED, SCAN_UNRECORDED


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
        could not decide its conformance, and the transcript carries an event token naming the
        declared family's own event. A span alone, or a conformance of False, is never enough.
    """
    report = by_branch.get(branch)
    if report is None or report.conformance_of != TASK:
        return False
    if report.conformance is True:
        return True
    return (
        branch == _AIRWAY
        and report.conformance == UNDETERMINED
        and findings.get(branch) == KindState.PRESENT.value
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
    if route_state == UNEXPLAINED:
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
        flag(_SPEECH, NO_LEXICAL_ITEM_PRODUCED, KEY_NO_LEXICAL_ITEM)
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
    if rules.second_opinion_disagreement_flags and disagreements:
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
        if record.get("passed") is not False:
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
    if uncomputed and rules.uncomputed_reading_flags:
        flag(
            str(gate_record.get("node") or _VERDICT),
            f"{UNCOMPUTED_READING}: {', '.join(uncomputed)}",
            KEY_UNCOMPUTED_READING,
        )
    for name, report in reports.items():
        if report.conformance is False and rules.flags_conformance(report.conformance_of, declared_family):
            why = TASK_NOT_CONFORMED if report.conformance_of == TASK else STORE_ASSERTION_CONTRADICTED
            prefix = PREFIX_CONFORMANCE if report.conformance_of == TASK else PREFIX_STORE_ASSERTION
            if report.conformance_of == TASK and nothing_read and name == gate_record.get("node"):
                why = NOTHING_READ
                prefix = PREFIX_NOTHING_READ
            named = f" on {declared_family}" if report.conformance_of == TASK and declared_family else ""
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
    for branch in branches_seen:
        decision = branch_decisions.get(branch)
        reported = by_branch.get(branch)
        kind = reported.kind if reported is not None else None
        # Only MISMATCH-and-PRESENT is a flag ground, and where the declared task names its owning
        # branches only that branch's task-conformant finding; both directions stay in ``agreement``.
        if (
            agreement[branch] == MISMATCH
            and findings[branch] == KindState.PRESENT.value
            and (
                not task_evidence.owning_branches
                or (
                    branch in task_evidence.owning_branches
                    and _owner_performed(branch, by_branch, findings, task_evidence)
                )
            )
        ):
            flag(
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
    flags = [reason for reason in reasons if reason.outcome is Outcome.FLAG]
    ground: str | None = None
    if admit is not None and admit.outcome is Outcome.FAIL:
        triage = Triage.DISCARD
        ground = UNMEASURABLE
        reasons = [admit, *(reason for reason in reasons if reason is not admit)]
    elif (
        task_evidence.duration_s is not None
        and task_evidence.minimum_duration_s is not None
        and task_evidence.duration_s < task_evidence.minimum_duration_s
    ):
        triage = Triage.DISCARD
        ground = TOO_SHORT_FOR_TASK
    elif route_state == EMPTY and not performed and not owner_unable:
        triage = Triage.DISCARD
        ground = ACOUSTICALLY_EMPTY
    elif any(is_operational(reason.key) for reason in flags):
        triage = Triage.RERUN
    elif declared_task_absent(by_branch, findings, task_evidence, (redaction or RedactionEvidence()).lexical_words_n):
        triage = Triage.DISCARD
        ground = DECLARED_TASK_ABSENT
    elif flags:
        triage = Triage.FLAG
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

    return FileVerdict(
        triage=triage,
        release=release,
        discard_ground=ground,
        release_ground=release_ground,
        ground_keys=sorted({*([ground] if ground else []), *(ground_key(reason) for reason in flags)}),
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
    )
