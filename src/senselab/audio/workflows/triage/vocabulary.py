"""The triage graph's shared vocabulary and the file-level fold.

The fold's rules — the two axes, the two grounds for a discard, branch authority scoped to the
branch's own kind, and the agreement and hint tables — are in
``specs/20260817-triage-workflow-dag/verdict.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


class Outcome(Enum):
    """What a node concluded."""

    PASS = "pass"
    FLAG = "flag"
    FAIL = "fail"


class Triage(Enum):
    """What should happen to this recording. The file axis; a node's ``Outcome`` is not one of these."""

    PASS = "pass"
    FLAG = "flag"
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
    """Whether a redacted artifact may be handed on."""

    RELEASABLE = "releasable"
    WITHHELD = "withheld"
    NOT_ASSESSED = "not_assessed"


UNMEASURABLE = "unmeasurable"
ACOUSTICALLY_EMPTY = "acoustically_empty"

AGREE = "agree"
MISMATCH = "mismatch"
RESOLVED = "resolved"
NOT_RUN = "not_run"

CLAIMED_AND_FOUND = "claimed_and_found"
CLAIMED_NOT_FOUND = "claimed_not_found"
FOUND_UNCLAIMED = "found_unclaimed"
NO_CLAIM = "no_claim"

_ADMIT = "ADMIT"
_REDACT = "REDACT"
_VERDICT = "VERDICT"

UNREAD_DECLARATION = (
    "a declaration was supplied and no branch decision survived to read it against; "
    "what it claimed is unknown, not empty"
)


@dataclass(frozen=True)
class NodeVerdict:
    """One node's conclusion.

    Attributes:
        node: The node's name.
        outcome: What it concluded. Every node concludes an ``Outcome``; the file fold concludes a
            ``Triage``, and that is the only verdict in the graph carrying the second member.
        kind: The kind the node concludes about, or None.
        why: The reason, in controlled vocabulary — never transcript text.
    """

    node: str
    outcome: Outcome | Triage
    kind: str | None
    why: str


@dataclass(frozen=True)
class BranchDecision:
    """What ``routing`` decided about one branch, as the fold reads it.

    Attributes:
        branch: The branch's name.
        kind: The kind it concludes about.
        will_run: Whether routing selected it.
        kind_state: What TAXONOMY classified, verbatim.
        forced_by_hint: Whether a hint added it.
        hint_tags: The declared tags naming this branch's kind. Non-empty is a claim, whether or not
            it changed the outcome.
    """

    branch: str
    kind: str
    will_run: bool
    kind_state: str
    forced_by_hint: bool
    hint_tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class FileVerdict:
    """The graph's conclusion about one recording, on both axes.

    Attributes:
        triage: What should happen to the recording.
        release: Whether REDACT's artifacts may be handed on. Never describes the store.
        discard_ground: ``"unmeasurable"``, ``"acoustically_empty"`` or None — the two grounds carry
            different reasons and a consumer that cannot tell them apart treats an empty recording as
            a broken one.
        kinds: The resolved state per kind, after branch authority.
        screened: What TAXONOMY classified. Present always, beside ``kinds``.
        agreement: ``agree`` | ``mismatch`` | ``resolved`` | ``not_run`` per kind.
        hints: ``claimed_and_found`` | ``claimed_not_found`` | ``found_unclaimed`` | ``no_claim``.
        reasons: Every contributing verdict, in order — not only the deciding one.
        ran: Whether each node ran.
        branches: The routing decision joined to the branch verdict.
    """

    triage: Triage
    release: Release
    discard_ground: str | None = None
    kinds: dict[str, str] = field(default_factory=dict)
    screened: dict[str, str] = field(default_factory=dict)
    agreement: dict[str, str] = field(default_factory=dict)
    hints: dict[str, str] = field(default_factory=dict)
    reasons: list[NodeVerdict] = field(default_factory=list)
    ran: dict[str, RunState] = field(default_factory=dict)
    branches: dict[str, dict[str, Any]] = field(default_factory=dict)


def _resolved(outcome: Outcome | Triage) -> str:
    """The kind state a branch's own conclusion establishes for the kind it screens.

    Args:
        outcome: The branch's outcome.

    Returns:
        ``absent`` for a branch that found no subject, ``present`` otherwise: a branch that flagged
        still found its kind, and the flag travels beside the resolution.
    """
    return KindState.ABSENT.value if outcome is Outcome.FAIL else KindState.PRESENT.value


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


def _release_from(node_verdicts: Sequence[NodeVerdict]) -> Release:
    """REDACT's outcome as a release state; an absent verdict means unexamined, never releasable.

    Args:
        node_verdicts: Every node verdict the fold was given.

    Returns:
        The release state for REDACT's artifacts only — never for anything in the store. Only
        ``pass`` clears an artifact, so the mapping is total: a flag, or any member added later,
        withholds rather than defaulting to cleared.
    """
    redact = next((verdict for verdict in node_verdicts if verdict.node == _REDACT), None)
    if redact is None:
        return Release.NOT_ASSESSED
    return Release.RELEASABLE if redact.outcome is Outcome.PASS else Release.WITHHELD


def _agreement(classified: str, verdict: NodeVerdict | None, resolved: str) -> str:
    """One kind's row of verdict.md's agreement table.

    Args:
        classified: What TAXONOMY classified for the kind.
        verdict: The branch's conclusion about it, or None when no branch concluded.
        resolved: The kind's resolved state.

    Returns:
        ``not_run`` when no branch concluded, ``agree`` or ``mismatch`` against a settled
        classification, and ``resolved`` where the classification was unsettled — including a state
        no reader can parse, which is unsettled for the same reason.
    """
    if verdict is None:
        return NOT_RUN
    found = resolved == KindState.PRESENT.value
    if classified == KindState.PRESENT.value:
        return AGREE if found else MISMATCH
    if classified == KindState.ABSENT.value:
        return MISMATCH if found else AGREE
    return RESOLVED


def _hint_reading(claimed: bool, found: bool) -> str:
    """One kind's row of verdict.md's hint table.

    Args:
        claimed: Whether the declaration claimed the kind.
        found: Whether the kind resolved present.

    Returns:
        The reading, one of the four.
    """
    if claimed:
        return CLAIMED_AND_FOUND if found else CLAIMED_NOT_FOUND
    return FOUND_UNCLAIMED if found else NO_CLAIM


def fold_file_verdict(
    node_verdicts: Sequence[NodeVerdict],
    *,
    screened: Mapping[str, str],
    branch_decisions: Mapping[str, BranchDecision],
    ran: Mapping[str, RunState],
    hint_claims: Mapping[str, bool] | None,
) -> FileVerdict:
    """Combine every node's verdict, the routing decisions and the classification into one file verdict.

    A branch is the authority on its own kind and on nothing else: its conclusion stands in ``kinds``
    whatever TAXONOMY classified, and the disagreement is recorded in ``agreement`` rather than
    resolved by precedence. A branch ``fail`` is not a file ``discard``.

    Args:
        node_verdicts: Every node's conclusion, in graph order, one per node.
        screened: What TAXONOMY classified per kind, verbatim. A kind it never classified reads
            ``uncertain``, which is the reading ROUTING already applies to a missing element.
        branch_decisions: What ROUTING decided per branch, keyed by branch name.
        ran: Whether each node ran, keyed by node name.
        hint_claims: Which kinds the caller's declaration claimed, keyed by kind, or None when a
            declaration was supplied and nothing in the store can say what it claimed. None
            empties ``hints`` and flags, rather than reporting an unread declaration as no claim.

    Returns:
        The file verdict on both axes, carrying every contributing reason rather than only the
        deciding one.
    """
    by_kind = {verdict.kind: verdict for verdict in node_verdicts if verdict.kind is not None}
    decision_for_kind = {decision.kind: decision for decision in branch_decisions.values()}
    claims = hint_claims or {}
    kinds_seen = list(dict.fromkeys([*decision_for_kind, *screened, *by_kind, *claims]))

    classified: dict[str, str] = {}
    for kind in kinds_seen:
        if kind in screened:
            classified[kind] = screened[kind]
        elif kind in decision_for_kind:
            classified[kind] = decision_for_kind[kind].kind_state
        else:
            classified[kind] = KindState.UNCERTAIN.value
    resolved = {kind: _resolved(by_kind[kind].outcome) if kind in by_kind else classified[kind] for kind in kinds_seen}
    agreement = {kind: _agreement(classified[kind], by_kind.get(kind), resolved[kind]) for kind in kinds_seen}
    hints = (
        {}
        if hint_claims is None
        else {
            kind: _hint_reading(bool(claims.get(kind, False)), resolved[kind] == KindState.PRESENT.value)
            for kind in kinds_seen
        }
    )

    reasons = list(node_verdicts)
    if hint_claims is None:
        reasons.append(NodeVerdict(_VERDICT, Outcome.FLAG, None, UNREAD_DECLARATION))
    for kind in kinds_seen:
        decision = decision_for_kind.get(kind)
        verdict = by_kind.get(kind)
        node = decision.branch if decision is not None else verdict.node if verdict is not None else kind.upper()
        if agreement[kind] == MISMATCH:
            found = "found it" if resolved[kind] == KindState.PRESENT.value else "found no subject"
            reasons.append(
                NodeVerdict(node, Outcome.FLAG, kind, f"mismatch: {kind} classified {classified[kind]}, {node} {found}")
            )
        if decision is not None and decision.will_run and verdict is None:
            reasons.append(
                NodeVerdict(node, Outcome.FLAG, kind, f"{node} was asked to run and {_silence(ran.get(node))}")
            )
        if hints.get(kind) == CLAIMED_NOT_FOUND:
            reasons.append(
                NodeVerdict(node, Outcome.FLAG, kind, f"hint mismatch: {kind} was declared and {node} did not find it")
            )

    branches = {
        name: {
            "will_run": decision.will_run,
            "forced_by_hint": decision.forced_by_hint,
            "kind_state": decision.kind_state,
            "verdict": by_kind[decision.kind].outcome.value if decision.kind in by_kind else None,
        }
        for name, decision in branch_decisions.items()
    }

    admit = next((verdict for verdict in node_verdicts if verdict.node == _ADMIT), None)
    ground: str | None = None
    if admit is not None and admit.outcome is Outcome.FAIL:
        triage = Triage.DISCARD
        ground = UNMEASURABLE
        reasons = [admit, *(reason for reason in reasons if reason is not admit)]
    elif any(reason.outcome is Outcome.FLAG for reason in reasons):
        triage = Triage.FLAG
    elif resolved and all(state == KindState.ABSENT.value for state in resolved.values()):
        triage = Triage.DISCARD
        ground = ACOUSTICALLY_EMPTY
    else:
        triage = Triage.PASS

    return FileVerdict(
        triage=triage,
        release=_release_from(node_verdicts),
        discard_ground=ground,
        kinds=resolved,
        screened=classified,
        agreement=agreement,
        hints=hints,
        reasons=reasons,
        ran=dict(ran),
        branches=branches,
    )
