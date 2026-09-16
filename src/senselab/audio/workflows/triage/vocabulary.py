"""The triage graph's shared vocabulary and the file-level fold.

A branch reports and VERDICT decides. The four branches and QUALITY write a :class:`BranchReport` —
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

UNDETERMINED: Literal["UNDETERMINED"] = "UNDETERMINED"
"""A conformance question that was not answered: nothing asked one, or nothing could answer it."""

Conformance = bool | Literal["UNDETERMINED"]
"""Whether what was asked for happened. Mirrors ``branches.Done``, which is where it comes from."""

TASK = "task"
"""A conformance about the instruction the recording declares. What the four branches report."""

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

UNREAD_DECLARATION = (
    "a declaration was supplied and no branch decision survived to read it against; "
    "what it claimed is unknown, not empty"
)


@dataclass(frozen=True)
class NodeVerdict:
    """One conclusion about the recording.

    Written by the nodes that decide — ADMIT, PREPROCESS, TAXONOMY, ``routing``, REDACT — and
    synthesised by :func:`fold_file_verdict` for each ground it flags on. A branch writes none: it
    writes a :class:`BranchReport`.

    Attributes:
        node: The node's name.
        outcome: What it concluded. A deciding node concludes an ``Outcome``; the file fold concludes
            a ``Triage``, and that is the only verdict in the graph carrying the second member.
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

    The spans are not here. A branch proposes them into the store, in its own family, and VERDICT
    reads them back by the activity that generated them — which is the record, where a count copied
    into this report would be a second one able to disagree with it.

    Attributes:
        node: The node's name, which is also the branch name its decision is keyed under.
        kind: The kind it reports on, or None where it reports on no kind.
        conformance: Whether what was asked for happened — True, False, or :data:`UNDETERMINED`.
        conformance_of: What that conformance is about, one of :data:`CONFORMANCE_REFERENTS`.
        deviations: The deviation type names it found, sorted and deduplicated. Recorded, never a
            ground for a flag: ``specs/20260913-branch-contract-and-hints/design.md`` states that
            ``filler`` and ``stimulus_mismatch`` are expected on ordinary read speech, so folding
            them into the flag column would flag the corpus.
        unmeasured: The config paths a body asked for and nobody has measured, in read order. A
            branch never refuses over one — it reports it here and leaves the dependent conformance
            :data:`UNDETERMINED` — so what an unmeasured point means for the file is this fold's,
            under ``verdict.unmeasured_points_flag``.
    """

    node: str
    kind: str | None
    conformance: Conformance
    conformance_of: str
    deviations: tuple[str, ...] = ()
    unmeasured: tuple[str, ...] = ()


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
class FoldPolicy:
    """The ``verdict.*`` config section: what this fold does with what the reporting nodes report.

    Every threshold that turns a reading into a judgement is here rather than in ``branch.*``,
    because a branch reports and VERDICT decides. ``config-derivations.md`` § verdict carries each
    value's derivation; none is fitted against the corpus.

    Attributes:
        conformance_flags: Whether a reported non-conformance about a **task** is a flag ground.
        undetermined_flags: Whether an unanswered conformance is. False, and load-bearing: the
            out-of-family mode evaluates no task and answers :data:`UNDETERMINED` by construction,
            so a True here flags every recording no branch was in-family for.
        deviation_flags: Whether a reported deviation is. False until ground truth exists: two of
            the three deviation types are expected on ordinary read speech.
        unmeasured_points_flag: Whether a reporting node that could not read an operating point it
            wanted is. True: with the packaged section carrying a value for every key, this fires
            only where an override removed one, which is a configuration fault worth seeing.
        conformance_flags_by_family: Declared task family to whether a non-conformance on it flags,
            overriding ``conformance_flags``. **This is what makes the fold task-aware**: what a
            missing conformance means is not the same question on a prolonged vowel as on a story
            recall, and a family whose conformance nobody trusts yet is excepted here by name
            rather than by the branch declining to report one.
    """

    conformance_flags: bool = True
    undetermined_flags: bool = False
    deviation_flags: bool = False
    unmeasured_points_flag: bool = True
    conformance_flags_by_family: dict[str, bool] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Any) -> "FoldPolicy":  # noqa: ANN401 — TriageConfig, not imported here
        """The policy this configuration declares.

        Read with ``get`` rather than ``require`` for one reason: this node is the last in the graph
        and a null here would lose the whole file verdict, where the packaged default it falls back
        to is the same value the packaged file carries.

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
            For :data:`STORE_ASSERTIONS`, always True: a stored assertion the same store's
            measurements contradict is inconsistent whichever task the recording carries and
            whatever a corpus would show, so no task key and no switch governs it. For
            :data:`TASK`, the family's own entry where it has one, else ``conformance_flags``. An
            unknown referent never flags: the fold does not know what the claim was about.
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
        discard_ground: ``"unmeasurable"``, ``"acoustically_empty"`` or None — the two grounds carry
            different reasons and a consumer that cannot tell them apart treats an empty recording as
            a broken one.
        findings: What each branch found, as a :class:`KindState` value, read off the spans the
            branch proposed in its own family: ``present`` with one or more, ``absent`` where it
            reported and proposed none, ``uncertain`` where it left no report at all.
        conformance: Each reporting node's conformance, keyed by node — True, False or
            :data:`UNDETERMINED`. QUALITY is in it and is not in ``findings``: it reports on the
            store's assertions and has no route, no kind and no subject to find.
        conformance_of: What each of those conformances is about, one of
            :data:`CONFORMANCE_REFERENTS`, so ``conformance`` is countable over a corpus without
            knowing which node is which.
        deviations: The deviation type names each reporting node found. Recorded and never folded.
        unmeasured: The config paths each reporting node asked for and nobody has measured.
        declared_family: The task family the recording declares, or None. In the product because the
            fold is task-aware: which flag grounds applied depends on it.
        routes: What the ruleset made of each branch, one of :data:`BRANCH_ROUTE_STATES`. Present
            always, beside ``findings``.
        route_state: What it made of the whole recording, one of :data:`FILE_ROUTE_STATES`, or None
            when ``routing`` wrote no evaluation.
        agreement: ``agree`` | ``mismatch`` | ``resolved`` | ``not_run`` per branch.
        hints: ``claimed_and_found`` | ``claimed_not_found`` | ``found_unclaimed`` | ``no_claim``
            per branch, read against the recording's declaration as ROUTING resolved it.
        reasons: Every contributing verdict, in order — not only the deciding one.
        ran: Whether each node ran.
        branches: The routing decision joined to the branch's reported conformance.
        bad_map_values: ``routing.hint_branch_map`` entries whose value is not a branch. A
            one-character typo under-claims every file in a run, so it is named here rather than left
            in the decisions for a reader to notice.
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


def _found(reported: bool, spans_n: int) -> str:
    """What a branch found, read off the spans it proposed rather than off any conclusion of its own.

    Args:
        reported: Whether the branch left a report at all.
        spans_n: How many spans it proposed into its own family.

    Returns:
        ``present`` with a span in hand, ``absent`` where the branch reported and proposed none, and
        ``uncertain`` where it left no report: not asked is not a measurement of absence.
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
        ``pass`` clears an artifact, so the mapping is total: a flag, or any member added later,
        withholds rather than defaulting to cleared.
    """
    redact = next((verdict for verdict in node_verdicts if verdict.node == _REDACT), None)
    if redact is None:
        return Release.NOT_ASSESSED
    return Release.RELEASABLE if redact.outcome is Outcome.PASS else Release.WITHHELD


def _agreement(route: str, reported: bool, found_state: str) -> str:
    """One branch's row of verdict.md's agreement table.

    The table is unchanged; its branch-side input is. It scored the branch's ``Outcome``, which was
    the branch deciding; it now scores what the branch *found* — the spans it proposed — against what
    the ruleset routed. Neither side is a judgement, so the disagreement between them is this fold's
    to name.

    Args:
        route: What the ruleset made of the branch.
        reported: Whether the branch left a report.
        found_state: What it found, from :func:`_found`.

    Returns:
        ``not_run`` when the branch left no report, ``agree`` or ``mismatch`` against a route the
        ruleset could read, and ``resolved`` where it could not — a branch whose gates were all
        unreadable made no claim to agree or disagree with.
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
    policy: FoldPolicy | None = None,
) -> FileVerdict:
    """Decide the file, from the deciding nodes' verdicts and the reporting nodes' reports.

    **A branch reports; this fold decides.** A branch hands over three things — the spans it proposed
    in its own family, its task conformance, and its typed deviations — and none of them is a verdict.
    What each contributes:

    * **conformance.** ``False`` is the one claim a reporting node makes that this fold turns into a
      flag: the instruction asked for something and it did not happen, located and measured. ``True``
      contributes nothing. :data:`UNDETERMINED` contributes nothing either, and that is load-bearing
      rather than lenient: every numeric ``branch.*`` key ships null and ``detect_*`` evaluates no
      task, so :data:`UNDETERMINED` is the packaged answer, and flagging it would flag the corpus.
    * **the spans.** They are what a branch *found*, so they are the branch side of the agreement
      table — see :func:`_found` and :func:`_agreement`. No span count is ever a flag by itself.
    * **the deviations.** Recorded per node in ``deviations`` and folded into nothing. The reason is
      in :class:`BranchReport`: two of the three deviation types are expected on ordinary read
      speech and no ground truth exists to say when they are not.
    * **the route.** The claim the spans are checked against. A ``mismatch`` flags, and it is the
      only place the route reaches the triage axis; it never overrides a branch on its own subject.
    * **what it could not measure.** A branch never refuses over an unmeasured operating point; it
      names the point and leaves the dependent conformance :data:`UNDETERMINED`. Whether that flags
      is ``policy.unmeasured_points_flag``.
    * **the declared task.** Every conformance ground is read against it, because what a missing
      conformance *means* is not the same question on a prolonged vowel as on a story recall.

    A branch ``absent`` is not a file ``discard``. PREPROCESS and ROUTING are each a gate every later
    node depends on, so a raise there is folded from ``ran`` rather than a verdict entity: a node
    that raised wrote none, so it is otherwise invisible to this fold, and a silent, evidence-free
    ``pass`` would be a worse outcome than the flag reported here.

    The two discard grounds are read off different things. ``unmeasurable`` is ADMIT's own fail;
    ``acoustically_empty`` is the ruleset's :data:`EMPTY` state, which is the emptiness bypass having
    read every tracked stream peak under its floor. A recording nothing routed that was *not* empty
    is :data:`UNEXPLAINED`, which flags rather than discards.

    Args:
        node_verdicts: Every deciding node's conclusion, in graph order, one per node.
        branch_reports: Every reporting node's report — the four branches and QUALITY — one per node.
            A report joins to its routing decision by node name.
        spans_by_node: How many spans each node proposed into its own family, keyed by node. A node
            absent from it proposed none. None is the same as empty.
        branch_decisions: What ROUTING decided per branch, keyed by branch name.
        ran: Whether each node ran, keyed by node name.
        hint_claims: Which branches the caller's declaration claimed, keyed by branch, or None when a
            declaration was supplied and nothing in the store can say what it claimed. None
            empties ``hints`` and flags, rather than reporting an unread declaration as no claim.
        route_state: What the ruleset made of the whole recording, or None when ``routing`` wrote no
            evaluation.
        declared_family: The task family the recording declares, or None. **The fold is task-aware
            through this**: a non-conformance is read against the task that was asked for, and
            ``policy.conformance_flags_by_family`` is where a family is excepted.
        policy: What to do with what was reported, from the ``verdict.*`` config section. None is
            the packaged policy, which is what a caller folding without a configuration gets.

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
        branch: branch_decisions[branch].route_state if branch in branch_decisions else UNAVAILABLE
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
        if agreement[branch] == MISMATCH:
            found = "found it" if findings[branch] == KindState.PRESENT.value else "found no subject"
            reasons.append(
                NodeVerdict(branch, Outcome.FLAG, kind, f"mismatch: routing {routes[branch]} {branch}, it {found}")
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
    )
