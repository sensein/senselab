"""The triage graph's shared vocabulary and the file-level fold.

A branch reports and VERDICT decides. The three branches and QUALITY write a :class:`BranchReport` —
task conformance and typed deviations, no outcome — and the spans they proposed; every decision about
the recording is made here. The fold's rules, the two axes, the two grounds for a discard, the
agreement and hint tables and what each input contributes are in
``specs/20260817-triage-workflow-dag/verdict.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    "VERDICT",
)
"""The nodes the runner drives, in the order it drives them. VERDICT folds the nine before it."""

QUALITY = "QUALITY"
"""The terminal node every recording reaches, whatever routed. A graph edge, never a branch."""

BRANCHES = ("AIRWAY", "SPEECH", "VOICE")
"""The branches routing selects among; each is the authority on its own kind and no other."""

RULESET_ROUTING = "ruleset_routing"
"""The measurement ``routing`` writes the family taxonomy ruleset's reading of a recording into.

``specs/20260912-ruleset-in-pipeline/design.md`` holds its attributes.
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
CONFORMANCE_UNANSWERED = "answered no conformance question"
UNMEASURED_ASKED = "asked for an operating point nobody has measured"
STORE_ASSERTION_CONTRADICTED = "reported that a stored assertion the same store's measurements contradict"

_SECTION = "verdict"
"""The config section :class:`FoldPolicy` reads. Every key in it is this fold's, not a branch's."""

_ADMIT = "ADMIT"
_PREPROCESS = "PREPROCESS"
_REDACT = "REDACT"
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
"""The flag ground a critical failure contributes, with the branch, gate and recorded absence appended.

Never a discard ground. See ``specs/20260817-triage-workflow-dag/critical-failure.md``.
"""

LLM_REDACTION_RESIDUE = "the redaction reviewer flagged residue on the redacted transcript"
"""The flag ground an LLM redaction annotation of ``flagged`` contributes to the triage axis.

Controlled vocabulary, with the reviewer's categories appended; the substrings and the reasoning
stay in the store's ``redaction_llm_review`` measurements.
"""


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
    """

    node: str
    outcome: Outcome | Triage
    kind: str | None
    why: str


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
        llm_redaction_flags: Whether REDACT's LLM re-read flagging residue is a flag ground on the
            **triage** axis. It reaches no other axis.
    """

    conformance_flags: bool = True
    undetermined_flags: bool = False
    deviation_flags: bool = False
    unmeasured_points_flag: bool = True
    llm_redaction_flags: bool = True
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
        release: Whether REDACT's artifacts may be handed on. Never describes the store.
        discard_ground: ``"unmeasurable"``, ``"acoustically_empty"`` or None.
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
        critical_absences: Per branch not one of whose gates could be read, each gate and the
            recorded absence. Non-empty means no branch was run.
        gates: The task group's gates and every one this fold applied — the gate, its reading, the
            bound and the group — so the conformance can be read backwards. Empty where the
            recording declares no task this graph holds a row for.
    """

    triage: Triage
    release: Release
    discard_ground: str | None = None
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
            "critical_absences": {branch: dict(gates) for branch, gates in self.critical_absences.items()},
            "gates": dict(self.gates),
            "ran": {node: state.value for node, state in self.ran.items()},
            "reasons": [
                {"node": r.node, "outcome": r.outcome.value, "kind": r.kind, "why": r.why} for r in self.reasons
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


def _release_from(node_verdicts: Sequence[NodeVerdict]) -> Release:
    """REDACT's outcome as a release state; an absent verdict means unexamined, never releasable.

    Args:
        node_verdicts: Every node verdict the fold was given.

    Returns:
        The release state for REDACT's artifacts only — never for anything in the store. Only
        ``pass`` clears an artifact; every other outcome, and an absent verdict, withholds.
    """
    redact = next((verdict for verdict in node_verdicts if verdict.node == _REDACT), None)
    if redact is None:
        return Release.NOT_ASSESSED
    return Release.RELEASABLE if redact.outcome is Outcome.PASS else Release.WITHHELD


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
    llm_redaction: Mapping[str, Any] | None = None,
    critical_absences: Mapping[str, Mapping[str, str]] | None = None,
    gates: Mapping[str, Any] | None = None,
    policy: FoldPolicy | None = None,
) -> FileVerdict:
    """Decide the file, from the deciding nodes' verdicts and the reporting nodes' reports.

    Flags on every ground it finds and discards on two: ``unmeasurable``, which is ADMIT's own fail,
    and ``acoustically_empty``, which is the ruleset's :data:`EMPTY` state. The grounds, the two
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
        llm_redaction: REDACT's LLM re-read annotation, or None where the node wrote none. Reaches
            the **triage** axis under ``policy.llm_redaction_flags`` and the release axis never.
        critical_absences: Per branch not one of whose gates could be read, each gate and the
            recorded absence behind it, as ``routing`` wrote them. Non-empty flags, never discards.
        gates: The task group's gates and the ones this fold's caller applied to the declared
            task's readings, carried onto the verdict so the conformance can be read backwards.
        policy: What to do with what was reported, from the ``verdict.*`` config section. None is
            the packaged policy.

    Returns:
        The file verdict on both axes, carrying every contributing reason rather than only the
        deciding one.
    """
    rules = policy or FoldPolicy()
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
    if route_state == UNREADABLE:
        reasons.append(NodeVerdict(_ROUTING, Outcome.FLAG, None, UNREADABLE_EMPTINESS))
    absences = {branch: dict(gates) for branch, gates in (critical_absences or {}).items()}
    if absences:
        named = "; ".join(
            f"{branch}: " + ", ".join(f"{gate} ({why})" for gate, why in sorted(gates.items()))
            for branch, gates in sorted(absences.items())
        )
        reasons.append(NodeVerdict(_ROUTING, Outcome.FLAG, None, f"{CRITICAL_ABSENCE}: {named}"))
    annotation = dict(llm_redaction or {})
    if annotation.get("status") == "flagged" and rules.llm_redaction_flags:
        named = ", ".join(str(category) for category in annotation.get("flagged") or ())
        reasons.append(
            NodeVerdict(
                _VERDICT, Outcome.FLAG, None, f"{LLM_REDACTION_RESIDUE}: {named}" if named else LLM_REDACTION_RESIDUE
            )
        )
    for name, report in reports.items():
        if report.conformance is False and rules.flags_conformance(report.conformance_of, declared_family):
            why = TASK_NOT_CONFORMED if report.conformance_of == TASK else STORE_ASSERTION_CONTRADICTED
            named = f" on {declared_family}" if report.conformance_of == TASK and declared_family else ""
            reasons.append(NodeVerdict(name, Outcome.FLAG, report.kind, f"{name} {why}{named}"))
        if report.conformance == UNDETERMINED and rules.undetermined_flags:
            reasons.append(NodeVerdict(name, Outcome.FLAG, report.kind, f"{name} {CONFORMANCE_UNANSWERED}"))
        if report.deviations and rules.deviation_flags:
            named = ", ".join(report.deviations)
            reasons.append(NodeVerdict(name, Outcome.FLAG, report.kind, f"{name} reported {named}"))
        if report.unmeasured and rules.unmeasured_points_flag:
            named = ", ".join(report.unmeasured)
            reasons.append(NodeVerdict(name, Outcome.FLAG, report.kind, f"{name} {UNMEASURED_ASKED}: {named}"))
    for branch in branches_seen:
        decision = branch_decisions.get(branch)
        reported = by_branch.get(branch)
        kind = reported.kind if reported is not None else None
        # Only MISMATCH-and-PRESENT is a flag ground; both directions stay in ``agreement``.
        if agreement[branch] == MISMATCH and findings[branch] == KindState.PRESENT.value:
            reasons.append(
                NodeVerdict(branch, Outcome.FLAG, kind, f"mismatch: routing {routes[branch]} {branch}, it found it")
            )
        if decision is not None and decision.will_run and reported is None:
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
            "withheld_critical": decision.withheld_critical,
            "conformance": by_branch[name].conformance if name in by_branch else None,
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
        critical_absences=absences,
        gates=dict(gates or {}),
    )
