"""The triage graph's shared vocabulary and the file-level fold.

The fold's rules — the two axes, the two grounds for a discard, branch authority scoped to the
branch's own kind, and the agreement and hint tables — are in
``specs/20260817-triage-workflow-dag/verdict.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

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
    "VERDICT",
)
"""The nodes the runner drives, in the order it drives them. VERDICT folds the nine before it."""

QUALITY = "QUALITY"
"""The terminal node every recording reaches, whatever routed. A graph edge, never a branch."""

BRANCHES = ("AIRWAY", "SPEECH", "VOICE", "DDK")
"""The branches routing selects among; each is the authority on its own kind and no other."""

RULESET_ROUTING = "ruleset_routing"
"""The measurement ``routing`` writes the family taxonomy ruleset's reading of a recording into.

Named here rather than beside the reader so VERDICT can read it back without importing the reader's
dependencies. ``specs/20260912-ruleset-in-pipeline/design.md`` holds its attributes.
"""

ROUTED = "routed"
DECLINED = "declined"
UNAVAILABLE = "unavailable"

BRANCH_ROUTE_STATES = (ROUTED, DECLINED, UNAVAILABLE)
"""What the ruleset made of one branch: a gate fired, every gate was silent, or none could be read."""

EMPTY = "empty"
UNEXPLAINED = "unexplained"

FILE_ROUTE_STATES = (ROUTED, EMPTY, UNEXPLAINED)
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
_PREPROCESS = "PREPROCESS"
_REDACT = "REDACT"
_VERDICT = "VERDICT"
_ROUTING = "routing"

BAD_MAP_VALUES = "routing.hint_branch_map names a branch this graph does not route to"

UNEXPLAINED_CONTENT = (
    "no branch routed and the recording was not measurably empty; the ruleset could not account for what is in it"
)

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
        branch: The branch's name, which is also the name its own verdict is written under.
        will_run: Whether routing selected it.
        route_state: What the ruleset made of this branch, one of :data:`BRANCH_ROUTE_STATES`. The
            content reading alone; a declared route never rewrites it.
        declared: Whether the recording's declaration named this branch — its task family, or a
            hint tag the map resolves. A claim, whether or not it changed the outcome.
        forced_by_declaration: Whether the declaration added it, which is ``declared`` and not
            content-routed. This is the route the declaration created; ``declared`` alone is not.
        hint_tags: The declared tags naming this branch, when a hint supplied any the map resolves.
        bad_map_values: ``routing.hint_branch_map`` entries whose value is not a branch, as
            ``{tag: value}``. A property of the configuration, so every decision carries the same
            one.
    """

    branch: str
    will_run: bool
    route_state: str
    forced_by_declaration: bool
    declared: bool = False
    hint_tags: tuple[str, ...] = ()
    bad_map_values: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FileVerdict:
    """The graph's conclusion about one recording, on both axes.

    Attributes:
        triage: What should happen to the recording.
        release: Whether REDACT's artifacts may be handed on. Never describes the store.
        discard_ground: ``"unmeasurable"``, ``"acoustically_empty"`` or None — the two grounds carry
            different reasons and a consumer that cannot tell them apart treats an empty recording as
            a broken one.
        findings: What each branch found, as a :class:`KindState` value. A branch that reached no
            conclusion reads ``uncertain``.
        routes: What the ruleset made of each branch, one of :data:`BRANCH_ROUTE_STATES`. Present
            always, beside ``findings``.
        route_state: What it made of the whole recording, one of :data:`FILE_ROUTE_STATES`, or None
            when ``routing`` wrote no evaluation.
        agreement: ``agree`` | ``mismatch`` | ``resolved`` | ``not_run`` per branch.
        hints: ``claimed_and_found`` | ``claimed_not_found`` | ``found_unclaimed`` | ``no_claim``
            per branch, read against the recording's declaration as ROUTING resolved it.
        reasons: Every contributing verdict, in order — not only the deciding one.
        ran: Whether each node ran.
        branches: The routing decision joined to the branch verdict.
        bad_map_values: ``routing.hint_branch_map`` entries whose value is not a branch. A
            one-character typo under-claims every file in a run, so it is named here rather than left
            in the decisions for a reader to notice.
    """

    triage: Triage
    release: Release
    discard_ground: str | None = None
    findings: dict[str, str] = field(default_factory=dict)
    routes: dict[str, str] = field(default_factory=dict)
    route_state: str | None = None
    agreement: dict[str, str] = field(default_factory=dict)
    hints: dict[str, str] = field(default_factory=dict)
    reasons: list[NodeVerdict] = field(default_factory=list)
    ran: dict[str, RunState] = field(default_factory=dict)
    branches: dict[str, dict[str, Any]] = field(default_factory=dict)
    bad_map_values: dict[str, str] = field(default_factory=dict)


def _resolved(outcome: Outcome | Triage) -> str:
    """The state a branch's own conclusion establishes for the subject it screens for.

    Args:
        outcome: The branch's outcome.

    Returns:
        ``absent`` for a branch that failed, ``present`` otherwise. The axis is found/not-found: a
        branch flags only with a subject in hand, so its flag travels beside a ``present``; a fail
        is the branch reporting no subject and never resolves one.
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


def _agreement(route: str, verdict: NodeVerdict | None, resolved: str) -> str:
    """One branch's row of verdict.md's agreement table.

    Args:
        route: What the ruleset made of the branch.
        verdict: The branch's conclusion, or None when it did not conclude.
        resolved: What the branch found.

    Returns:
        ``not_run`` when the branch did not conclude, ``agree`` or ``mismatch`` against a route the
        ruleset could read, and ``resolved`` where it could not — a branch whose gates were all
        unreadable made no claim to agree or disagree with.
    """
    if verdict is None:
        return NOT_RUN
    found = resolved == KindState.PRESENT.value
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


def fold_file_verdict(
    node_verdicts: Sequence[NodeVerdict],
    *,
    branch_decisions: Mapping[str, BranchDecision],
    ran: Mapping[str, RunState],
    hint_claims: Mapping[str, bool] | None,
    route_state: str | None,
) -> FileVerdict:
    """Combine every node's verdict and the routing decisions into one file verdict.

    A branch is the authority on its own subject and on nothing else: its conclusion stands in
    ``findings`` whatever the ruleset routed, and the disagreement is recorded in ``agreement``
    rather than resolved by precedence. A branch ``fail`` is not a file ``discard``. PREPROCESS and
    ROUTING are each a gate every later node depends on, so a raise there is folded from ``ran``
    rather than a verdict entity: a node that raised wrote none, so it is otherwise invisible to this
    fold, and a silent, evidence-free ``pass`` would be a worse outcome than the flag reported here.

    The two discard grounds are read off different things. ``unmeasurable`` is ADMIT's own fail;
    ``acoustically_empty`` is the ruleset's :data:`EMPTY` state, which is the emptiness bypass having
    read every tracked stream peak under its floor. A recording nothing routed that was *not* empty
    is :data:`UNEXPLAINED`, which flags rather than discards.

    Args:
        node_verdicts: Every node's conclusion, in graph order, one per node. A branch's own verdict
            is joined to its decision by node name, and only when it names the subject it concluded
            about: a branch verdict carrying no ``kind`` did not conclude about one, which is what a
            reader synthesises for an outcome no reader can act on.
        branch_decisions: What ROUTING decided per branch, keyed by branch name.
        ran: Whether each node ran, keyed by node name.
        hint_claims: Which branches the caller's declaration claimed, keyed by branch, or None when a
            declaration was supplied and nothing in the store can say what it claimed. None
            empties ``hints`` and flags, rather than reporting an unread declaration as no claim.
        route_state: What the ruleset made of the whole recording, or None when ``routing`` wrote no
            evaluation.

    Returns:
        The file verdict on both axes, carrying every contributing reason rather than only the
        deciding one.
    """
    claims = hint_claims or {}
    by_branch = {
        verdict.node: verdict for verdict in node_verdicts if verdict.node in BRANCHES and verdict.kind is not None
    }
    branches_seen = list(dict.fromkeys([*branch_decisions, *by_branch, *claims]))

    routes = {
        branch: branch_decisions[branch].route_state if branch in branch_decisions else UNAVAILABLE
        for branch in branches_seen
    }
    findings = {
        branch: _resolved(by_branch[branch].outcome) if branch in by_branch else KindState.UNCERTAIN.value
        for branch in branches_seen
    }
    agreement = {
        branch: _agreement(routes[branch], by_branch.get(branch), findings[branch]) for branch in branches_seen
    }
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

    reasons = list(node_verdicts)
    if ran.get(_PREPROCESS) is RunState.ERRORED:
        reasons.append(
            NodeVerdict(
                _PREPROCESS,
                Outcome.FLAG,
                None,
                "preprocess failed; no derivative was measured because conditioning itself did not complete",
            )
        )
    if ran.get(_ROUTING) is RunState.ERRORED:
        reasons.append(
            NodeVerdict(
                _ROUTING,
                Outcome.FLAG,
                None,
                "routing failed; branch execution was withheld because no complete routing result was available",
            )
        )
    if bad_map_values:
        named = ", ".join(f"{tag}: {value}" for tag, value in sorted(bad_map_values.items()))
        reasons.append(NodeVerdict(_ROUTING, Outcome.FLAG, None, f"{BAD_MAP_VALUES}: {named}"))
    if hint_claims is None:
        reasons.append(NodeVerdict(_VERDICT, Outcome.FLAG, None, UNREAD_DECLARATION))
    if route_state == UNEXPLAINED:
        reasons.append(NodeVerdict(_ROUTING, Outcome.FLAG, None, UNEXPLAINED_CONTENT))
    for branch in branches_seen:
        decision = branch_decisions.get(branch)
        verdict = by_branch.get(branch)
        kind = verdict.kind if verdict is not None else None
        if agreement[branch] == MISMATCH:
            found = "found it" if findings[branch] == KindState.PRESENT.value else "found no subject"
            reasons.append(
                NodeVerdict(branch, Outcome.FLAG, kind, f"mismatch: routing {routes[branch]} {branch}, it {found}")
            )
        if decision is not None and decision.will_run and verdict is None:
            reasons.append(
                NodeVerdict(branch, Outcome.FLAG, kind, f"{branch} was asked to run and {_silence(ran.get(branch))}")
            )
        if hints.get(branch) == CLAIMED_NOT_FOUND:
            reasons.append(
                NodeVerdict(branch, Outcome.FLAG, kind, f"hint mismatch: {branch} was declared and did not find it")
            )

    branch_view = {
        name: {
            "will_run": decision.will_run,
            "forced_by_declaration": decision.forced_by_declaration,
            "route_state": decision.route_state,
            "verdict": by_branch[name].outcome.value if name in by_branch else None,
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
    elif route_state == EMPTY:
        triage = Triage.DISCARD
        ground = ACOUSTICALLY_EMPTY
    else:
        triage = Triage.PASS

    return FileVerdict(
        triage=triage,
        release=_release_from(node_verdicts),
        discard_ground=ground,
        findings=findings,
        routes=routes,
        route_state=route_state,
        agreement=agreement,
        hints=hints,
        reasons=reasons,
        ran=dict(ran),
        branches=branch_view,
        bad_map_values=bad_map_values,
    )
