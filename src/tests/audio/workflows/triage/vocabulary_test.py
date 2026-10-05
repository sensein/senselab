"""The file-level fold: pass, flag, discard, and each branch authority over its own subject.

A branch reports and this fold decides, so a branch enters through ``branch_reports`` and
``spans_by_node`` rather than through ``node_verdicts``: what it *found* is the spans it proposed and
what it *claims* is its conformance. ``node_verdicts`` now carries only the nodes that decide.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Any, Mapping, Sequence

from senselab.audio.workflows.triage.vocabulary import (
    BAD_MAP_VALUES,
    CRITICAL_ABSENCE,
    DECLARED_TASK_ABSENT,
    DECLINED,
    DOMINANT_SPEAKER_GATE,
    FINDINGS_ARE_TASK_CONTENT,
    GROUND_KEY_PREFIXES,
    GROUND_KEYS,
    INSTRUCTIONS_SPOKEN,
    KEY_OWNING_BRANCH_INPUT_ABSENT,
    KEY_TASK_MISMATCH,
    LLM_REDACTION_RESIDUE,
    MASKS_TRIMMED_TO_CONTENT,
    MODEL_SPEAKER_PERMITTED,
    NO_BREATH_CAPTURED,
    NO_CONTENT_MASKED,
    NO_LEXICAL_ITEM_PRODUCED,
    NO_LEXICAL_WORD,
    NO_TRANSCRIPT,
    NON_LEXICAL_TASK,
    OPERATIONAL_GROUND_KEYS,
    REDACT_UNRESOLVED,
    REDACT_VERIFY_FOUND,
    REDACTION_OWED,
    RELEASE_UNKNOWN_GROUNDS,
    RELEASE_WITH_REDACTION_GROUNDS,
    RELEASE_WITHHELD_GROUNDS,
    RELEASE_WITHOUT_REDACTION_GROUNDS,
    REVIEWER_CLEARED_RESCAN,
    REVIEWER_CLEARED_UNMASKED,
    REVIEWER_HEARD_SECOND_SPEAKER,
    REVIEWER_NAMED_NO_WORDS,
    REVIEWER_PROPOSED_REDACTION,
    REVIEWER_UNMASKED_ALL,
    REVIEWER_UNMASKED_SOME,
    ROUTED,
    SCAN_UNRECORDED,
    SECOND_OPINION_DISAGREES,
    SPEECH_UNREAD,
    TASK,
    TOO_SHORT_FOR_TASK,
    UNAVAILABLE,
    UNDETERMINED,
    UNEXPLAINED_CONTENT,
    UNJUDGED,
    UNPLACED_FINDING_OPEN,
    UNPLACED_FINDING_UNREAD,
    UNPLACED_OPEN,
    UNPLACED_PLACED,
    UNPLACED_UNREAD,
    UNREAD_DECLARATION,
    UNREADABLE_EMPTINESS,
    BranchDecision,
    BranchReport,
    Conformance,
    FileVerdict,
    FoldPolicy,
    NodeVerdict,
    Outcome,
    RedactionEvidence,
    Release,
    RunState,
    TaskEvidence,
    Triage,
    _release_from,
    fold_file_verdict,
    ground_key,
    is_operational,
    release_ground_key,
    reviewer_may_unmask,
)


def _report(node: str, kind: str, *, conformance: Conformance = UNDETERMINED) -> BranchReport:
    """One branch's report, as a branch writes it: a conformance and no outcome.

    Args:
        node: The branch's name.
        kind: The kind it reports on.
        conformance: Whether what the instruction asked for happened.

    Returns:
        The report.
    """
    return BranchReport(node=node, kind=kind, conformance=conformance, conformance_of=TASK)


def _found(*nodes: str) -> dict[str, int]:
    """One proposed span per named node, which is what the fold reads as ``present``.

    Args:
        *nodes: The nodes that proposed a span.

    Returns:
        The span counts, for ``spans_by_node``.
    """
    return {node: 1 for node in nodes}


def _decisions(forced: Sequence[str] = (), **routes: str) -> dict[str, BranchDecision]:
    """One decision per named branch, as ROUTING writes them.

    Args:
        forced: Branches the declaration added although the ruleset did not route them.
        **routes: Branch name to its route state.

    Returns:
        The decisions, keyed by branch name.
    """
    return {
        branch: BranchDecision(
            branch=branch,
            will_run=state == ROUTED or branch in forced,
            route_state=state,
            forced_by_declaration=branch in forced,
            declared=branch in forced,
        )
        for branch, state in routes.items()
    }


def _all_declined(forced: Sequence[str] = ()) -> dict[str, BranchDecision]:
    """The empty execution set: every branch declined.

    Args:
        forced: Branches the declaration added anyway.

    Returns:
        The three decisions.
    """
    return _decisions(forced, AIRWAY=DECLINED, SPEECH=DECLINED, VOICE=DECLINED)


def _with_redact(redact: Outcome, *, speech: bool | None = None, speech_route: str = DECLINED) -> FileVerdict:
    """A fold whose only interesting node is REDACT, optionally with a SPEECH branch beside it.

    Args:
        redact: What REDACT concluded.
        speech: True when the SPEECH branch reported a conforming, span-bearing reading, or None
            when the branch never ran. REDACT is a deciding node and keeps its ``Outcome``; SPEECH
            is a reporting node and has none.
        speech_route: What the ruleset made of SPEECH.

    Returns:
        The folded file verdict.
    """
    node_verdicts = [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")]
    node_verdicts.append(NodeVerdict("REDACT", redact, None, "the scan concluded"))
    return fold_file_verdict(
        node_verdicts,
        branch_reports=[] if speech is None else [_report("SPEECH", "speech", conformance=True)],
        spans_by_node={} if speech is None else _found("SPEECH"),
        branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=speech_route, VOICE=DECLINED),
        ran={},
        hint_claims={},
        route_state=ROUTED,
        redaction=RedactionEvidence(findings_n=1, masks_n=1, masks_final_n=1),
    )


def _without_redact(evidence: RedactionEvidence, *, speech: RunState, speech_route: str = ROUTED) -> FileVerdict:
    """A fold REDACT left no verdict on, over one state of the redaction evidence.

    Args:
        evidence: What the store says about whether anything was redactable.
        speech: Whether SPEECH ran.
        speech_route: What the ruleset made of SPEECH. ``DECLINED`` is a task that asks for no
            words, which the release axis reads as a clearance rather than as an absence.

    Returns:
        The folded file verdict.
    """
    return fold_file_verdict(
        [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
        branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=speech_route, VOICE=DECLINED),
        ran={"SPEECH": speech},
        hint_claims={},
        route_state=ROUTED,
        redaction=evidence,
    )


class TestTheTriageVocabulary:
    """pass, flag, rerun, discard — four values, and fail is not one of them."""

    def test_the_members_are_exactly_four(self) -> None:
        """verdict.md's triage axis; a branch's ``fail`` has no counterpart here."""
        assert {member.value for member in Triage} == {"pass", "flag", "rerun", "discard"}

    def test_a_node_outcome_is_not_a_triage(self) -> None:
        """Outcome stays the node-level vocabulary; the file axis is its own type."""
        assert not isinstance(Outcome.FAIL, Triage)


class TestDiscardIsNarrow:
    """Exactly two grounds, and they carry different reasons."""

    def test_admit_failure_discards_as_unmeasurable(self) -> None:
        """Nothing ran and nothing is claimed about the recording."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.FAIL, None, "decode failure")],
            branch_decisions={},
            ran={},
            hint_claims={},
            route_state=None,
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "unmeasurable"

    def test_the_emptiness_bypass_discards_as_acoustically_empty(self) -> None:
        """Measured: every tracked stream peak fell under the floor, so there is nothing in it."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_all_declined(),
            ran={},
            hint_claims={},
            route_state="empty",
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "acoustically_empty"

    def test_nothing_routed_but_not_empty_reruns_rather_than_discarding(self) -> None:
        """Content no gate could account for is a charge against the ruleset, never against the file."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_all_declined(),
            ran={},
            hint_claims={},
            route_state="unexplained",
        )
        assert folded.triage is Triage.RERUN
        assert folded.ground_keys == ["route_unexplained"]
        assert folded.discard_ground is None
        assert any(reason.why == UNEXPLAINED_CONTENT for reason in folded.reasons)

    def test_an_unreadable_bypass_reruns_under_its_own_ground(self) -> None:
        """Nothing routed and no bypass to read is owed a rerun, and says which of the two it was.

        It is not a discard: neither discard ground is a claim about a measurement that was never
        taken. It is not ``unexplained`` either, which would charge the ruleset for evidence the
        run failed to produce.
        """
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_all_declined(),
            ran={},
            hint_claims={},
            route_state="unreadable",
        )
        assert folded.triage is Triage.RERUN
        assert folded.ground_keys == ["route_unreadable"]
        assert folded.discard_ground is None
        assert any(reason.why == UNREADABLE_EMPTINESS for reason in folded.reasons)
        assert not any(reason.why == UNEXPLAINED_CONTENT for reason in folded.reasons)

    def test_the_two_grounds_are_told_apart_by_their_ground_not_by_their_axis(self) -> None:
        """Both discard; a consumer that cannot tell them apart treats an empty file as a broken one."""
        broken = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.FAIL, None, "decode failure")],
            branch_decisions={},
            ran={},
            hint_claims={},
            route_state=None,
        )
        empty = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_all_declined(),
            ran={},
            hint_claims={},
            route_state="empty",
        )
        assert broken.triage is empty.triage is Triage.DISCARD
        assert broken.discard_ground != empty.discard_ground

    def test_a_pass_carries_no_ground(self) -> None:
        """``discard_ground`` describes a discard and nothing else."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("AIRWAY", "airway", conformance=True)],
            spans_by_node=_found("AIRWAY"),
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.triage is Triage.PASS
        assert folded.discard_ground is None

    def test_no_routing_at_all_is_not_an_empty_recording(self) -> None:
        """A run that routed nothing because ROUTING never concluded has measured nothing."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions={},
            ran={},
            hint_claims={},
            route_state=None,
        )
        assert folded.triage is Triage.PASS
        assert folded.discard_ground is None

    def test_a_branch_fail_is_not_a_discard(self) -> None:
        """A cough recording has no speech; SPEECH failing is the expected outcome."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
                NodeVerdict("SPEECH", Outcome.FAIL, "speech", "no consensus word"),
            ],
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.triage is not Triage.DISCARD

    def test_an_empty_recording_discards_and_keeps_the_hint_mismatch_as_detail(self) -> None:
        """A declared kind no branch found in an empty recording is detail, not a reason to review it."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_all_declined(),
            ran={},
            hint_claims={"SPEECH": True},
            route_state="empty",
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "acoustically_empty"
        assert folded.ground_keys == ["acoustically_empty", "hint_mismatch:SPEECH"]

    def test_an_empty_route_where_a_branch_found_its_kind_is_not_discarded(self) -> None:
        """A forced branch that found what it looks for contradicts the emptiness; that one flags."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("SPEECH", "speech", conformance=True)],
            spans_by_node=_found("SPEECH"),
            branch_decisions=_decisions(SPEECH=DECLINED, AIRWAY=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state="empty",
        )
        assert folded.triage is Triage.FLAG
        assert "route_mismatch:SPEECH" in folded.ground_keys

    def test_the_empty_execution_set_discards_rather_than_flagging_on_its_own(self) -> None:
        """ROUTING records the empty set on its decisions; a ``pass`` verdict beside them does not preempt.

        The alternative recorded in ``benchmarks/open.md`` — routing flagging the empty set — made
        verdict.md's acoustically-empty discard unreachable, since the fold tests any flag first.
        """
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("routing", Outcome.PASS, None, "no branch runs (empty); AIRWAY declined"),
            ],
            branch_decisions=_all_declined(),
            ran={},
            hint_claims={},
            route_state="empty",
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "acoustically_empty"


class TestBranchAuthorityIsScoped:
    """A branch is the authority on its own subject and on nothing else."""

    def test_speech_resolves_speech_and_touches_nothing_else(self) -> None:
        """It refutes neither AIRWAY nor VOICE, and it does not settle them either."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("SPEECH", "speech", conformance=True)],
            spans_by_node=_found("SPEECH"),
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.findings["SPEECH"] == "present"
        assert folded.findings["AIRWAY"] == "uncertain"
        assert folded.findings["VOICE"] == "uncertain"

    def test_a_non_conforming_branch_still_resolves_its_subject(self) -> None:
        """The flag the fold raises travels beside the resolution and does not withhold it."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("VOICE", "voice", conformance=False)],
            spans_by_node=_found("VOICE"),
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=DECLINED, VOICE=ROUTED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.findings["VOICE"] == "present"
        assert folded.conformance["VOICE"] is False
        assert folded.triage is Triage.FLAG

    def test_a_branch_that_proposed_no_span_resolves_its_subject_absent(self) -> None:
        """A branch with no subject is authority for that too, and the spans are what say so."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("SPEECH", "speech"), _report("AIRWAY", "airway", conformance=True)],
            spans_by_node=_found("AIRWAY"),
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.findings["SPEECH"] == "absent"

    def test_an_empty_handed_branch_does_not_carry_its_absence_onto_a_sibling(self) -> None:
        """SPEECH found no subject; that says nothing about the airway AIRWAY found."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("SPEECH", "speech"), _report("AIRWAY", "airway", conformance=True)],
            spans_by_node=_found("AIRWAY"),
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.findings["AIRWAY"] == "present"
        assert folded.findings["SPEECH"] == "absent"


class TestTheRoutingIsReportedBeside:
    """Both maps are always present, and agreement is checkable by a reader."""

    def test_routes_and_findings_are_both_present(self) -> None:
        """Keeping both is what makes agreement checkable rather than asserted."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("SPEECH", "speech", conformance=True)],
            spans_by_node=_found("SPEECH"),
            branch_decisions=_decisions(["SPEECH"], AIRWAY=DECLINED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.routes["SPEECH"] == DECLINED
        assert folded.findings["SPEECH"] == "present"
        assert folded.route_state == ROUTED

    def test_a_declined_branch_that_found_its_subject_is_a_mismatch_and_flags(self) -> None:
        """The ruleset missed it; the mismatch is the product, and it never overrides either side."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("SPEECH", "speech", conformance=True)],
            spans_by_node=_found("SPEECH"),
            branch_decisions=_decisions(["SPEECH"], AIRWAY=DECLINED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.agreement["SPEECH"] == "mismatch"
        assert folded.triage is Triage.FLAG
        assert [reason for reason in folded.reasons if "it found it" in reason.why]

    def test_a_routed_branch_that_found_nothing_is_a_mismatch_and_does_not_flag(self) -> None:
        """Routing is lenient by design, so a branch finding none of its kind is it being right.

        The mismatch stays in the agreement table, which is where a reader checks the ruleset against
        the detectors. It is not a ground: charging the recording for routing's leniency made this
        the largest single flag ground in the corpus, 830 records over a 2,269-recording pilot.
        """
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("SPEECH", "speech")],
            spans_by_node={},
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.agreement["SPEECH"] == "mismatch"
        assert folded.findings["SPEECH"] == "absent"
        assert folded.triage is Triage.PASS
        assert not [reason for reason in folded.reasons if "found no subject" in reason.why]

    def test_a_declared_branch_that_found_nothing_still_flags(self) -> None:
        """The informative case survives: the recording said it held this kind and it does not."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("SPEECH", "speech")],
            spans_by_node={},
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={},
            hint_claims={"SPEECH": True},
            route_state=ROUTED,
        )
        assert folded.triage is Triage.FLAG
        assert [reason for reason in folded.reasons if "was declared and did not find it" in reason.why]

    def test_an_unreadable_route_is_resolved_not_mismatched(self) -> None:
        """A branch whose gates could not be read made no claim to disagree with."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("SPEECH", "speech", conformance=True)],
            spans_by_node=_found("SPEECH"),
            branch_decisions=_decisions(["SPEECH"], AIRWAY=DECLINED, SPEECH=UNAVAILABLE, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.agreement["SPEECH"] == "resolved"
        assert folded.triage is Triage.PASS

    def test_agreeing_branches_are_recorded_as_agreeing(self) -> None:
        """``agree`` is a value a reader can see, not the absence of a mismatch."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("AIRWAY", "airway", conformance=True), _report("SPEECH", "speech")],
            spans_by_node=_found("AIRWAY"),
            branch_decisions=_decisions(["SPEECH"], AIRWAY=ROUTED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.agreement["AIRWAY"] == "agree"
        assert folded.agreement["SPEECH"] == "agree"
        assert folded.triage is Triage.PASS

    def test_the_route_is_never_rewritten_by_the_branch(self) -> None:
        """``routes`` reports what the ruleset made of the branch even where the branch overruled it."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("AIRWAY", "airway")],
            spans_by_node={},
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.routes["AIRWAY"] == ROUTED
        assert folded.findings["AIRWAY"] == "absent"

    def test_a_branch_with_neither_a_route_nor_a_verdict_reads_uncertain(self) -> None:
        """Nothing said anything about VOICE; reading that as absent would invent a measurement."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.FAIL, "speech", "no consensus word"),
            ],
            branch_decisions={},
            ran={},
            hint_claims={"VOICE": True},
            route_state=ROUTED,
        )
        assert folded.routes["VOICE"] == UNJUDGED
        assert folded.findings["VOICE"] == "uncertain"
        assert folded.triage is not Triage.DISCARD

    def test_a_branch_routing_never_judged_is_told_from_one_whose_gates_were_unreadable(self) -> None:
        """Two different facts that shared one token: no decision written, and a decision of unavailable.

        ``unavailable`` is ROUTING's own reading -- it looked, and every gate of the branch was
        unreadable. A branch with no decision entity at all was never looked at. Both resolve the
        agreement table the same way and neither is a claim about the recording, which is exactly
        why one token for both was unreadable rather than harmless.
        """
        judged = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_decisions(VOICE=UNAVAILABLE),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        never = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions={},
            ran={},
            hint_claims={"VOICE": True},
            route_state=ROUTED,
        )
        assert judged.routes["VOICE"] == UNAVAILABLE
        assert never.routes["VOICE"] == UNJUDGED
        assert judged.routes["VOICE"] != never.routes["VOICE"]

    def test_a_routed_branch_that_never_concluded_reads_uncertain(self) -> None:
        """A branch asked to look and silent has not established an absence."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=ROUTED, VOICE=ROUTED),
            ran={"SPEECH": RunState.COMPLETED, "VOICE": RunState.COMPLETED},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.routes["SPEECH"] == ROUTED
        assert folded.findings["SPEECH"] == "uncertain"


class TestABranchThatNeverRanIsNotOneThatFailed:
    """The branch_decision elements are what distinguish the two."""

    def test_declined_and_unforced_is_expected(self) -> None:
        """The graph declined to look, and said why."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("AIRWAY", "airway", conformance=True)],
            spans_by_node=_found("AIRWAY"),
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.agreement["SPEECH"] == "not_run"
        assert folded.triage is Triage.PASS

    def test_asked_but_silent_reruns(self) -> None:
        """will_run true with no verdict is a branch that left no answer: the pipeline owes it one."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={"SPEECH": RunState.ERRORED},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.triage is Triage.RERUN
        assert folded.ground_keys == ["branch_silent:SPEECH"]
        assert any("errored without a verdict" in reason.why for reason in folded.reasons)

    def test_the_three_silent_reasons_are_distinguished(self) -> None:
        """errored, completed-without-a-verdict and never-ran are different findings."""
        for state, phrase in (
            (RunState.ERRORED, "errored without a verdict"),
            (RunState.COMPLETED, "completed without a verdict"),
            (RunState.SKIPPED, "never ran"),
        ):
            folded = fold_file_verdict(
                [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
                branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
                ran={"SPEECH": state},
                hint_claims={},
                route_state=ROUTED,
            )
            assert any(phrase in reason.why for reason in folded.reasons)

    def test_the_silent_reason_names_the_branch(self) -> None:
        """A reason a reader cannot attribute to a branch is one they cannot act on."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert any(reason.node == "SPEECH" for reason in folded.reasons)

    def test_a_routed_branch_with_no_node_is_reported_rather_than_ignored(self) -> None:
        """A routed branch that left no verdict reruns the file, whatever silenced it.

        The state the fold is handed is the general one: the branch was asked to run and concluded
        nothing — skipped, errored, or completed without a verdict. The fold must say so rather
        than pass quietly, and that is what is pinned.
        """
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={"SPEECH": RunState.SKIPPED},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.triage is Triage.RERUN
        assert any(reason.node == "SPEECH" and "never ran" in reason.why for reason in folded.reasons)

    def test_the_branches_map_joins_the_decision_to_the_reported_conformance(self) -> None:
        """A skipped branch carries the reason it was skipped, beside a branch that reported."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("AIRWAY", "airway", conformance=True)],
            spans_by_node=_found("AIRWAY"),
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.branches["AIRWAY"]["conformance"] is True
        assert folded.branches["SPEECH"]["conformance"] is None
        assert folded.branches["SPEECH"]["will_run"] is False
        assert folded.branches["SPEECH"]["route_state"] == DECLINED


class TestHintsForMismatchOnly:
    """A hint names a mismatch and prevents a discard. It has no other power on this axis."""

    def test_a_hinted_branch_that_found_nothing_flags(self) -> None:
        """The branch, the hint that claimed it, and its conclusion, all named."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.FAIL, "airway", "no span carries a label"),
            ],
            branch_decisions=_decisions(["AIRWAY"], AIRWAY=DECLINED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={"AIRWAY": True},
            route_state=ROUTED,
        )
        assert "hint_mismatch:AIRWAY" in folded.ground_keys
        assert folded.triage in (Triage.FLAG, Triage.RERUN)
        assert folded.hints["AIRWAY"] == "claimed_not_found"

    def test_a_hinted_branch_that_found_its_subject_is_an_agreement(self) -> None:
        """The declaration and the measurement said the same thing; nothing is owed a human."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("AIRWAY", "airway", conformance=True)],
            spans_by_node=_found("AIRWAY"),
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={"AIRWAY": True},
            route_state=ROUTED,
        )
        assert folded.hints["AIRWAY"] == "claimed_and_found"
        assert folded.triage is Triage.PASS

    def test_a_subject_found_that_no_hint_claimed_is_recorded_not_flagged(self) -> None:
        """Recorded; not a flag on its own."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("AIRWAY", "airway", conformance=True)],
            spans_by_node=_found("AIRWAY"),
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.hints["AIRWAY"] == "found_unclaimed"
        assert folded.triage is Triage.PASS

    def test_an_unclaimed_branch_that_found_nothing_carries_no_claim(self) -> None:
        """The fourth cell of the table, so a reader never has to infer it from a missing key."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.hints["SPEECH"] == "no_claim"

    def test_a_hint_never_turns_a_flag_into_a_pass(self) -> None:
        """Its one power is to prevent a discard and to name a mismatch."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.FLAG, "speech", "pii in the target's speech"),
            ],
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={},
            hint_claims={"SPEECH": True},
            route_state=ROUTED,
        )
        assert folded.triage is not Triage.PASS

    def test_a_hint_never_resolves_a_subject(self) -> None:
        """A claim is an expectation; only a branch resolves, and here none concluded."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_all_declined(),
            ran={},
            hint_claims={"SPEECH": True},
            route_state="empty",
        )
        assert folded.findings["SPEECH"] == "uncertain"

    def test_a_declaration_nothing_could_read_empties_the_hints_and_reruns(self) -> None:
        """Unknown claims are not no claims: reporting them as ``no_claim`` would clear the file quietly."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions={},
            ran={},
            hint_claims=None,
            route_state=ROUTED,
        )
        assert folded.hints == {}
        assert folded.triage is Triage.RERUN
        assert folded.ground_keys == ["declaration_unread"]
        assert any(reason.why == UNREAD_DECLARATION for reason in folded.reasons)


class TestAConfigTypoIsNamedNotSwallowed:
    """A map value that is not a branch under-claims every file in the run; it is named, never swallowed."""

    def test_a_bad_map_value_is_named_on_an_empty_recording_that_still_discards(self) -> None:
        """The recording is measurably empty whatever the map says; the typo stays visible beside it."""
        decisions = _all_declined()
        decisions["AIRWAY"] = replace(decisions["AIRWAY"], bad_map_values={"cough": "AIRWY"})
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=decisions,
            ran={},
            hint_claims={},
            route_state="empty",
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "acoustically_empty"
        assert "config_bad_hint_map" in folded.ground_keys
        assert folded.bad_map_values == {"cough": "AIRWY"}
        assert any(BAD_MAP_VALUES in reason.why and "AIRWY" in reason.why for reason in folded.reasons)

    def test_a_well_formed_map_flags_nothing(self) -> None:
        """The control: the same recording discards when the map is sound."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_all_declined(),
            ran={},
            hint_claims={},
            route_state="empty",
        )
        assert folded.bad_map_values == {}
        assert folded.triage is Triage.DISCARD


class TestReleaseIsDecidedFromEvidenceNotFromRedactsAbsence:
    """REDACT runs only where there is something to redact, so its silence decides nothing.

    ``specs/20260817-triage-workflow-dag/verdict.md``, the release fold. Measured over 62,548
    recordings, the old rule reported 44,623 as unexamined when 44,622 of them were determined.
    """

    def test_a_scan_that_found_nothing_is_not_unassessed(self) -> None:
        """The largest population the old rule mislabelled: SPEECH scanned and the transcript was clean."""
        folded = _without_redact(
            RedactionEvidence(lexical_words_n=42, scanned=True, findings_n=0), speech=RunState.COMPLETED
        )
        assert folded.release is not Release.NOT_ASSESSED

    def test_a_declined_scan_is_not_unassessed(self) -> None:
        """SPEECH read every lexical word out of the task's own stimulus, so nothing was disclosed."""
        folded = _without_redact(
            RedactionEvidence(lexical_words_n=9, scanned=False, findings_n=0), speech=RunState.COMPLETED
        )
        assert folded.release is not Release.NOT_ASSESSED

    def test_a_transcript_with_no_lexical_word_is_not_unassessed(self) -> None:
        """SPEECH ran and the consensus carried no word; a redaction has nothing to read."""
        folded = _without_redact(
            RedactionEvidence(lexical_words_n=0, scanned=None, findings_n=0), speech=RunState.COMPLETED
        )
        assert folded.release is not Release.NOT_ASSESSED

    def test_a_recording_speech_never_ran_on_is_unassessed(self) -> None:
        """Nothing read its lexical content, so the axis cannot say the original may be handed on.

        ``specs/20260924-which-artefact-is-releasable/design.md`` §3: the 2026-09-22 rule put this
        row beside the three a reading cleared, on the premise that the axis described REDACT's
        artifacts only. The axis now describes the original too, and that premise is void.
        """
        folded = _without_redact(RedactionEvidence(), speech=RunState.SKIPPED)
        assert folded.release is Release.NOT_ASSESSED
        assert folded.release_ground == NO_TRANSCRIPT

    def test_the_four_cleared_grounds_name_the_same_state(self) -> None:
        """One state, four evidence grounds: a reading that cleared the recording is not four answers.

        The rest of the group is reached only through REDACT's masks: the reviewer's, the trim's, and
        a plan that masked nothing.
        """
        cleared = [
            _without_redact(RedactionEvidence(lexical_words_n=42, scanned=True), speech=RunState.COMPLETED),
            _without_redact(RedactionEvidence(lexical_words_n=9, scanned=False), speech=RunState.COMPLETED),
            _without_redact(RedactionEvidence(lexical_words_n=0), speech=RunState.COMPLETED),
            _without_redact(RedactionEvidence(), speech=RunState.SKIPPED, speech_route=DECLINED),
        ]
        assert {folded.release for folded in cleared} == {Release.WITHOUT_REDACTION}
        assert {folded.release_ground for folded in cleared} == set(RELEASE_WITHOUT_REDACTION_GROUNDS) - {
            REVIEWER_UNMASKED_ALL,
            NO_CONTENT_MASKED,
            FINDINGS_ARE_TASK_CONTENT,
            REVIEWER_CLEARED_UNMASKED,
        }

    def test_a_speech_that_errored_is_unassessed(self) -> None:
        """The one genuine unknown: nothing can say whether the recording carried anything."""
        folded = _without_redact(RedactionEvidence(), speech=RunState.ERRORED)
        assert folded.release is Release.NOT_ASSESSED
        assert folded.release_ground == SPEECH_UNREAD

    def test_a_finding_redact_never_answered_is_unassessed(self) -> None:
        """The scan found something and no redaction verdict stands over it."""
        folded = _without_redact(
            RedactionEvidence(lexical_words_n=42, scanned=True, findings_n=3), speech=RunState.COMPLETED
        )
        assert folded.release is Release.NOT_ASSESSED
        assert folded.release_ground == REDACTION_OWED

    def test_lexical_content_with_no_scan_record_is_unassessed(self) -> None:
        """SPEECH reached words and recorded no scan either way; the graph cannot say."""
        folded = _without_redact(
            RedactionEvidence(lexical_words_n=42, scanned=None, findings_n=0), speech=RunState.COMPLETED
        )
        assert folded.release is Release.NOT_ASSESSED
        assert folded.release_ground == SCAN_UNRECORDED

    def test_a_cleared_recording_does_not_read_as_a_redacted_one(self) -> None:
        """Nothing was redacted, so there is no redacted artifact for the axis to be about."""
        folded = _without_redact(RedactionEvidence(lexical_words_n=0), speech=RunState.COMPLETED)
        assert folded.release is not Release.WITH_REDACTION

    def test_a_redact_verdict_still_decides_where_one_stands(self) -> None:
        """Evidence never overrides the node that actually ran."""
        assert _with_redact(Outcome.PASS).release is Release.WITH_REDACTION
        assert _with_redact(Outcome.PASS).release_ground is None


class TestTheReleaseAxisNamesWhichArtefactMayBeHandedOn:
    """Four values over one question: which artefact of this recording may I hand on?

    ``specs/20260924-which-artefact-is-releasable/design.md``.
    """

    def test_the_axis_offers_exactly_the_four_answers(self) -> None:
        """Total and exclusive: the original, only the redacted copy, neither, or unknown."""
        assert {member.value for member in Release} == {
            "release_without_redaction",
            "release_with_redaction",
            "withheld",
            "not_assessed",
        }

    def test_a_redact_flag_withholds(self) -> None:
        """Unresolved is not cleared."""
        folded = _with_redact(Outcome.FLAG)
        assert folded.release is Release.WITHHELD

    def test_a_redact_fail_withholds(self) -> None:
        """A finding survived verification."""
        assert _with_redact(Outcome.FAIL).release is Release.WITHHELD

    def test_a_redact_pass_releases_only_the_redacted_artefact(self) -> None:
        """REDACT passes on having removed findings the original still carries."""
        assert _with_redact(Outcome.PASS).release is Release.WITH_REDACTION

    def test_only_a_reading_that_ran_releases_the_original(self) -> None:
        """Every ground behind the permissive value is one where something read the recording."""
        assert NO_TRANSCRIPT not in RELEASE_WITHOUT_REDACTION_GROUNDS
        assert NO_TRANSCRIPT in RELEASE_UNKNOWN_GROUNDS

    def test_no_ground_stands_behind_two_states(self) -> None:
        """A reader must be able to go from the ground back to the state without the table."""
        groups = [
            RELEASE_WITHOUT_REDACTION_GROUNDS,
            RELEASE_UNKNOWN_GROUNDS,
            RELEASE_WITHHELD_GROUNDS,
            RELEASE_WITH_REDACTION_GROUNDS,
        ]
        for i, group in enumerate(groups):
            for other in groups[i + 1 :]:
                assert not set(group) & set(other)

    def test_the_reviewer_reaches_the_axis_through_two_declared_parameters(self) -> None:
        """``_release_from``'s parameters are the whole input to the axis.

        ``design.md`` §5. The reviewer reaches it through one parameter that tightens and one that
        clears a re-scan fail, both defaulting to reading nothing; which words its ``release``
        entries unmask arrives already decided, on the evidence.
        """
        parameters = inspect.signature(_release_from).parameters
        assert set(parameters) == {
            "node_verdicts",
            "evidence",
            "ran",
            "reviewer_withholds",
            "speech_declined",
            "reviewer_clears",
        }
        assert parameters["reviewer_withholds"].default is None
        assert parameters["reviewer_clears"].default is False
        assert parameters["speech_declined"].default is False
        assert sorted(name for name in parameters if "review" in name) == ["reviewer_clears", "reviewer_withholds"]

    def test_the_reviewer_tightens_a_pass_and_no_withholding_loosens_a_fail(self) -> None:
        """Without a clearing reading, a REDACT fail stays withheld. ``design.md`` §5."""
        passed = [NodeVerdict("REDACT", Outcome.PASS, None, "the scan concluded")]
        failed = [NodeVerdict("REDACT", Outcome.FAIL, None, "the scan concluded")]
        evidence = RedactionEvidence(rescan_survivors=("DATE_TIME",), masks_n=1, masks_final_n=1)
        ran: dict[str, RunState] = {}
        assert _release_from(passed, evidence, ran)[0] is Release.WITH_REDACTION
        assert (
            _release_from(passed, evidence, ran, reviewer_withholds=REVIEWER_PROPOSED_REDACTION)[0] is Release.WITHHELD
        )
        assert _release_from(failed, evidence, ran, reviewer_withholds=None)[0] is Release.WITHHELD
        assert (
            _release_from(failed, evidence, ran, reviewer_withholds=REVIEWER_PROPOSED_REDACTION)[0] is Release.WITHHELD
        )

    def test_every_withholding_carries_its_ground(self) -> None:
        """A REDACT fail or flag that nothing clears names why it withholds."""
        evidence = RedactionEvidence(rescan_survivors=("PERSON",), masks_n=1, masks_final_n=1)
        ran: dict[str, RunState] = {}
        failed = [NodeVerdict("REDACT", Outcome.FAIL, None, "verification found pii")]
        flagged = [NodeVerdict("REDACT", Outcome.FLAG, None, "unresolved")]
        assert _release_from(failed, evidence, ran) == (Release.WITHHELD, REDACT_VERIFY_FOUND)
        assert _release_from(flagged, evidence, ran) == (Release.WITHHELD, REDACT_UNRESOLVED)
        assert _release_from(failed, RedactionEvidence(masks_n=1, masks_final_n=1), ran) == (
            Release.WITHHELD,
            REDACT_UNRESOLVED,
        )
        assert {REDACT_VERIFY_FOUND, REDACT_UNRESOLVED} <= set(RELEASE_WITHHELD_GROUNDS)
        assert _with_redact(Outcome.FAIL).release_ground is not None
        assert _with_redact(Outcome.FLAG).release_ground is not None

    def test_the_reviewer_withholds_where_redact_never_ran(self) -> None:
        """Every path that would release is tightened, not only the one through a REDACT pass."""
        ran = {"SPEECH": RunState.COMPLETED}
        for evidence in (
            RedactionEvidence(lexical_words_n=42, scanned=True),
            RedactionEvidence(lexical_words_n=9, scanned=False),
            RedactionEvidence(lexical_words_n=0),
        ):
            assert _release_from([], evidence, ran)[0] is Release.WITHOUT_REDACTION
            assert _release_from([], evidence, ran, reviewer_withholds=REVIEWER_PROPOSED_REDACTION) == (
                Release.WITHHELD,
                REVIEWER_PROPOSED_REDACTION,
            )
        assert _release_from(
            [], RedactionEvidence(), {}, reviewer_withholds=REVIEWER_PROPOSED_REDACTION, speech_declined=True
        ) == (
            Release.WITHHELD,
            REVIEWER_PROPOSED_REDACTION,
        )

    def test_a_reviewer_withholding_names_its_ground(self) -> None:
        """A withholding REDACT did not make must be told apart from one it did."""
        passed = [NodeVerdict("REDACT", Outcome.PASS, None, "the scan concluded")]
        failed = [NodeVerdict("REDACT", Outcome.FAIL, None, "the scan concluded")]
        assert (
            _release_from(passed, RedactionEvidence(), {}, reviewer_withholds=REVIEWER_PROPOSED_REDACTION)[1]
            == REVIEWER_PROPOSED_REDACTION
        )
        assert (
            _release_from(failed, RedactionEvidence(), {}, reviewer_withholds=REVIEWER_PROPOSED_REDACTION)[1]
            == REDACT_UNRESOLVED
        )

    def test_a_reviewer_withholding_leaves_a_discard_a_discard(self) -> None:
        """The release axis tightens; the triage axis is not the reviewer's to move."""
        proposal = {
            "status": "flagged",
            "redaction": "incomplete",
            "original": "carries_pii",
            "proposal": [{"text": "alice", "action": "redact", "category": "PERSON", "why": "a name"}],
        }
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.FAIL, None, "unmeasurable")],
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={"SPEECH": RunState.COMPLETED},
            hint_claims={},
            route_state=ROUTED,
            redaction=RedactionEvidence(lexical_words_n=42, scanned=True),
            llm_redaction=proposal,
            policy=FoldPolicy(llm_redaction_withholds=True),
        )
        assert folded.triage is Triage.DISCARD
        assert folded.release is Release.WITHHELD

    def test_the_reviewer_never_moves_an_unassessed_recording(self) -> None:
        """``not_assessed`` stays what it is: a gap is not a release for the reviewer to tighten."""
        cases = [
            (RedactionEvidence(), {"SPEECH": RunState.ERRORED}),
            (RedactionEvidence(lexical_words_n=42, scanned=True, findings_n=3), {"SPEECH": RunState.COMPLETED}),
            (RedactionEvidence(), {"SPEECH": RunState.SKIPPED}),
        ]
        for evidence, ran in cases:
            assert _release_from([], evidence, ran, reviewer_withholds=REVIEWER_PROPOSED_REDACTION) == _release_from(
                [], evidence, ran
            )
            assert _release_from([], evidence, ran)[0] is Release.NOT_ASSESSED


class TestAReviewerReadingClearsAReScanFail:
    """A clean reading releases the redacted copy of a REDACT re-scan fail.

    Owner, 2026-09-26: where REDACT withholds only because its re-scan still reads a finding, the
    reviewer, which already read the text, decides.
    """

    _FAILED = [NodeVerdict("REDACT", Outcome.FAIL, None, "verification found pii on the redacted transcript")]
    _CLEAN: Mapping[str, Any] = {"status": "clean", "original": "clean", "redaction": "complete", "proposal": []}

    def test_a_clean_reading_releases_the_redacted_copy_under_its_own_ground(self) -> None:
        """The redacted copy, never the original, and a ground no other path carries."""
        evidence = RedactionEvidence(
            lexical_words_n=40, scanned=True, findings_n=1, rescan_survivors=("DATE_TIME",), masks_n=1, masks_final_n=1
        )
        assert _release_from(self._FAILED, evidence, {}, reviewer_clears=True) == (
            Release.WITH_REDACTION,
            REVIEWER_CLEARED_RESCAN,
        )
        assert _release_from(self._FAILED, evidence, {}) == (Release.WITHHELD, REDACT_VERIFY_FOUND)

    def test_an_incomplete_scan_is_never_cleared(self) -> None:
        """A fail with no re-scan survivor is an unchecked recording, which no reading clears."""
        evidence = RedactionEvidence(lexical_words_n=40, scanned=True, findings_n=1)
        assert _release_from(self._FAILED, evidence, {}, reviewer_clears=True) == (Release.WITHHELD, REDACT_UNRESOLVED)

    def test_clearing_moves_nothing_but_a_redact_fail(self) -> None:
        """A pass, a recording REDACT never read, and an unassessed one are untouched."""
        passed = [NodeVerdict("REDACT", Outcome.PASS, None, "the scan concluded")]
        survivors = RedactionEvidence(lexical_words_n=40, scanned=True, rescan_survivors=("PERSON",))
        assert _release_from(passed, survivors, {}, reviewer_clears=True) == _release_from(passed, survivors, {})
        ran = {"SPEECH": RunState.COMPLETED}
        for evidence in (
            RedactionEvidence(lexical_words_n=42, scanned=True),
            RedactionEvidence(lexical_words_n=42, scanned=True, findings_n=3),
        ):
            assert _release_from([], evidence, ran, reviewer_clears=True) == _release_from([], evidence, ran)

    def _fold(self, annotation: Mapping[str, Any], policy: FoldPolicy) -> FileVerdict:
        """A REDACT re-scan fail folded with this reading under this policy."""
        return fold_file_verdict(
            self._FAILED,
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={"SPEECH": RunState.COMPLETED, "REDACT": RunState.COMPLETED},
            hint_claims={},
            route_state=ROUTED,
            redaction=RedactionEvidence(
                lexical_words_n=42,
                scanned=True,
                findings_n=1,
                rescan_survivors=("DATE_TIME",),
                masks_n=1,
                masks_final_n=1,
            ),
            llm_redaction=annotation,
            policy=policy,
        )

    def test_the_fold_clears_on_a_clean_original_with_nothing_to_hide(self) -> None:
        """``flagged`` with only releases proposed is still a reading that the original is clean."""
        on = FoldPolicy(llm_redaction_withholds=True, llm_rescan_clears=True)
        release_only = {
            "status": "flagged",
            "original": "clean",
            "redaction": "incomplete",
            "proposal": [{"text": "the past couple of weeks", "action": "release", "category": "DATE_TIME"}],
        }
        for annotation in (self._CLEAN, release_only):
            folded = self._fold(annotation, on)
            assert (folded.release, folded.release_ground) == (Release.WITH_REDACTION, REVIEWER_CLEARED_RESCAN)

    def test_a_reading_that_finds_pii_keeps_it_withheld(self) -> None:
        """A redact proposal, or an original read as carrying PII, is not a clearing reading."""
        on = FoldPolicy(llm_redaction_withholds=True, llm_rescan_clears=True)
        redact = {
            "status": "flagged",
            "original": "clean",
            "proposal": [{"text": "brooklyn", "action": "redact", "category": "LOCATION"}],
        }
        carries = {"status": "flagged", "original": "carries_pii", "proposal": []}
        for annotation in (redact, carries):
            assert self._fold(annotation, on).release is Release.WITHHELD

    def test_no_reading_and_the_policy_off_clear_nothing(self) -> None:
        """``absent``, ``disabled`` and ``nothing_to_read`` are not readings; the key is the switch."""
        on = FoldPolicy(llm_rescan_clears=True)
        for status in ("absent", "disabled", "nothing_to_read"):
            assert self._fold({"status": status, "original": "clean", "proposal": []}, on).release is Release.WITHHELD
        assert self._fold(self._CLEAN, FoldPolicy(llm_rescan_clears=False)).release is Release.WITHHELD


class TestTheMasksThatStandDecideTheRelease:
    """The evidence carries which masks stand; the fold names the release and its ground from it.

    Owner, 2026-09-27: the reviewer unmasks exactly the words it names, and no mask ever keeps a
    non-content word. Which words those are is decided before the fold, in ``redact.mask_plan``.
    """

    _PASSED = [NodeVerdict("REDACT", Outcome.PASS, None, "every finding redacted")]
    _FAILED = [NodeVerdict("REDACT", Outcome.FAIL, None, "verification found pii on the redacted transcript")]

    @staticmethod
    def _evidence(
        final_n: int, *, unmasked_n: int = 0, changed: bool = True, survivors: tuple[str, ...] = ()
    ) -> RedactionEvidence:
        """A recording REDACT masked twice, ``final_n`` masks standing after the word-level rule."""
        return RedactionEvidence(
            lexical_words_n=40,
            scanned=True,
            findings_n=2,
            rescan_survivors=survivors,
            masks_n=2,
            masks_final_n=final_n,
            masks_changed=changed,
            reviewer_unmasked_n=unmasked_n,
        )

    def test_unchanged_masks_keep_redacts_own_decision(self) -> None:
        """No mask lost a word: REDACT's copy, with no ground of the fold's own."""
        assert _release_from(self._PASSED, self._evidence(2, changed=False), {}) == (Release.WITH_REDACTION, None)

    def test_the_reviewers_unmasks_thin_the_copy_or_release_the_original(self) -> None:
        """Some masks standing is a partial copy; none standing is the original."""
        assert _release_from(self._PASSED, self._evidence(1, unmasked_n=2), {}) == (
            Release.WITH_REDACTION,
            REVIEWER_UNMASKED_SOME,
        )
        assert _release_from(self._PASSED, self._evidence(0, unmasked_n=3), {}) == (
            Release.WITHOUT_REDACTION,
            REVIEWER_UNMASKED_ALL,
        )

    def test_the_content_word_trim_alone_has_its_own_grounds(self) -> None:
        """A mask trimmed to its content words with no reviewer involved is named as the trim's."""
        assert _release_from(self._PASSED, self._evidence(2), {}) == (Release.WITH_REDACTION, MASKS_TRIMMED_TO_CONTENT)
        assert _release_from(self._PASSED, self._evidence(0), {}) == (Release.WITHOUT_REDACTION, NO_CONTENT_MASKED)

    def test_a_cleared_rescan_fail_composes_with_the_unmasks(self) -> None:
        """The reading clears the fail and then unmasks what it named."""
        evidence = self._evidence(0, unmasked_n=2, survivors=("DATE_TIME",))
        assert _release_from(self._FAILED, evidence, {}, reviewer_clears=True) == (
            Release.WITHOUT_REDACTION,
            REVIEWER_UNMASKED_ALL,
        )
        assert _release_from(self._FAILED, evidence, {}) == (Release.WITHHELD, REDACT_VERIFY_FOUND)

    def test_a_pass_that_planned_no_mask_releases_the_original(self) -> None:
        """REDACT exempted every finding as declared task content: no copy masks anything.

        Measured on r6: 539 recordings released "with redaction" whose copy was byte-for-byte the
        original, story-recall and productive-vocabulary above all.
        """
        exempt_only = RedactionEvidence(lexical_words_n=40, scanned=True, findings_n=1, masks_n=0, masks_final_n=0)
        assert _release_from(self._PASSED, exempt_only, {}) == (Release.WITHOUT_REDACTION, FINDINGS_ARE_TASK_CONTENT)

    def test_a_cleared_rescan_fail_with_no_mask_releases_the_original(self) -> None:
        """The reviewer clears a fail REDACT planned no mask over: the original, under its own ground."""
        unmasked = RedactionEvidence(
            lexical_words_n=40, scanned=True, findings_n=1, rescan_survivors=("PERSON",), masks_n=0, masks_final_n=0
        )
        assert _release_from(self._FAILED, unmasked, {}, reviewer_clears=True) == (
            Release.WITHOUT_REDACTION,
            REVIEWER_CLEARED_UNMASKED,
        )

    def test_a_mask_over_no_word_still_masks_the_audio(self) -> None:
        """A standing mask that placed on no transcript word keeps the copy: its audio is still masked."""
        kept = RedactionEvidence(
            lexical_words_n=40, scanned=True, findings_n=1, rescan_survivors=("PERSON",), masks_n=1, masks_final_n=1
        )
        assert _release_from(self._FAILED, kept, {}, reviewer_clears=True) == (
            Release.WITH_REDACTION,
            REVIEWER_CLEARED_RESCAN,
        )

    def test_standing_masks_never_move_a_withholding_or_an_unassessed_recording(self) -> None:
        """The word-level rule only thins a released copy; it releases nothing the evidence withheld."""
        assert _release_from(self._FAILED, self._evidence(0, unmasked_n=2), {}) == (Release.WITHHELD, REDACT_UNRESOLVED)
        assert _release_from(self._PASSED, self._evidence(0, unmasked_n=2), {}, REVIEWER_PROPOSED_REDACTION) == (
            Release.WITHHELD,
            REVIEWER_PROPOSED_REDACTION,
        )
        ran = {"SPEECH": RunState.COMPLETED}
        owed = RedactionEvidence(lexical_words_n=42, scanned=True, findings_n=3, masks_changed=True)
        assert _release_from([], owed, ran)[0] is Release.NOT_ASSESSED

    def test_any_reading_taken_may_unmask(self) -> None:
        """Owner, 2026-09-27: the reviewer names exactly what to unmask, whatever else it proposes.

        A reading that also proposes a redaction is withheld on that ground; its unmasks still
        change which words the ledger shows masked. No reading, or none taken, unmasks nothing.
        """
        release_only = {"status": "flagged", "proposal": [{"text": "brooklyn", "action": "release"}]}
        hides_more = {
            "status": "flagged",
            "proposal": [{"text": "brooklyn", "action": "release"}, {"text": "alice", "action": "redact"}],
        }
        assert reviewer_may_unmask(release_only) is True
        assert reviewer_may_unmask({"status": "clean", "proposal": []}) is True
        assert reviewer_may_unmask(hides_more) is True
        assert reviewer_may_unmask(None) is False
        assert reviewer_may_unmask({"status": "absent", "failure": "CUDA error", "proposal": []}) is False
        assert reviewer_may_unmask({"status": "nothing_to_read", "proposal": []}) is False


class TestAConditionNeverWithholds:
    """Owner, 2026-09-27: a named condition identifies by rarity and context, so it is flagged for review."""

    _PASSED = [NodeVerdict("REDACT", Outcome.PASS, None, "every finding redacted")]

    @staticmethod
    def _reading(*categories: str) -> dict[str, Any]:
        """A reading proposing one ``redact`` entry per category, and releasing a date."""
        return {
            "status": "flagged",
            "original": "carries_pii",
            "proposal": [
                {"text": "this morning", "action": "release", "category": "DATE_TIME"},
                *(
                    {"text": f"a {category.lower()}", "action": "redact", "category": category}
                    for category in categories
                ),
            ],
        }

    def _fold(self, annotation: Mapping[str, Any], policy: FoldPolicy) -> FileVerdict:
        """A REDACT pass folded with this reading under this policy."""
        return fold_file_verdict(
            self._PASSED,
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={"SPEECH": RunState.COMPLETED, "REDACT": RunState.COMPLETED},
            hint_claims={},
            route_state=ROUTED,
            redaction=RedactionEvidence(lexical_words_n=40, scanned=True, findings_n=1, masks_n=1, masks_final_n=1),
            llm_redaction=annotation,
            policy=policy,
        )

    _POLICY = FoldPolicy(llm_redaction_withholds=True, cohort_conditions="bridge2ai_voice_adult_2026-09-04")

    @staticmethod
    def _conditions(*texts: str) -> dict[str, Any]:
        """A prompt-v6 reading listing these conditions as redact entries, and releasing a date."""
        return {
            "status": "flagged",
            "original": "carries_pii",
            "proposal": [
                {"text": "this morning", "action": "release", "category": "DATE_TIME"},
                *({"text": text, "action": "redact", "category": "CONDITION"} for text in texts),
            ],
        }

    def test_a_condition_alone_releases_and_flags_nothing(self) -> None:
        """Policy v7: a condition is neither withheld nor a residue flag, whatever the cohort."""
        folded = self._fold(self._conditions("Parkinson's", "synovial joint cyst"), self._POLICY)
        assert folded.release is Release.WITH_REDACTION
        assert not any(reason.why.startswith(LLM_REDACTION_RESIDUE) for reason in folded.reasons)

    def test_a_condition_beside_another_category_withholds_for_the_other_one(self) -> None:
        """A name beside the condition is residue the ordinary way, and the ground names only the name."""
        folded = self._fold(self._reading("CONDITION", "PERSON"), self._POLICY)
        assert (folded.release, folded.release_ground) == (Release.WITHHELD, REVIEWER_PROPOSED_REDACTION)
        assert any(reason.why == f"{LLM_REDACTION_RESIDUE}: PERSON" for reason in folded.reasons)

    def test_the_packaged_config_names_condition(self) -> None:
        """The shipped key names CONDITION and nothing else."""
        from senselab.audio.workflows.triage.config import load_triage_config

        assert FoldPolicy.from_config(load_triage_config()).condition_categories == ("CONDITION",)

    def test_the_packaged_config_names_the_study_s_cohort_profile(self) -> None:
        """The shipped key names the Bridge2AI-Voice adult profile, and it loads."""
        from senselab.audio.workflows.triage.cohort import load_cohort_profile
        from senselab.audio.workflows.triage.config import load_triage_config

        name = FoldPolicy.from_config(load_triage_config()).cohort_conditions
        assert name == "bridge2ai_voice_adult_2026-09-04"
        assert load_cohort_profile(name).diagnosis("spasmodic dysphonia") == "laryngeal_dystonia"


class TestANonLexicalTaskIsClearedRatherThanHeld:
    """Owner, 2026-09-25: a task that asks for no words is not a recording the graph cannot judge.

    19,097 recordings -- breath, cough, sustained vowels, glides -- reached ``NO_TRANSCRIPT`` and
    answered ``not_assessed`` because the ruleset had declined SPEECH. The ruleset's own decision is
    the reading: there is nothing a redaction could remove from a task that never asked for a word.
    """

    def test_a_declined_speech_branch_clears_the_recording(self) -> None:
        """The ruleset declining SPEECH is evidence, not the absence of it."""
        folded = _without_redact(RedactionEvidence(), speech=RunState.SKIPPED, speech_route=DECLINED)
        assert folded.release is Release.WITHOUT_REDACTION
        assert folded.release_ground == NON_LEXICAL_TASK

    def test_a_routed_speech_branch_that_never_ran_is_still_unassessed(self) -> None:
        """Routing asked for SPEECH and got nothing back: that is a gap, not a clearance."""
        folded = _without_redact(RedactionEvidence(), speech=RunState.SKIPPED, speech_route=ROUTED)
        assert folded.release is Release.NOT_ASSESSED
        assert folded.release_ground == NO_TRANSCRIPT

    def test_clearing_a_non_lexical_task_raises_no_flag(self) -> None:
        """A cough that reads as a cough is not a finding."""
        folded = _without_redact(RedactionEvidence(), speech=RunState.SKIPPED, speech_route=DECLINED)
        assert folded.triage is Triage.PASS


class TestASpeechTaskThatProducedNoWordIsFlagged:
    """The other half of the same instruction: cleared is not the same as unremarkable.

    A task the ruleset routed to SPEECH is a task that asks for words. SPEECH running over it and
    reading none is the task not having happened, and the release axis calling it releasable is
    true without being the whole of it.
    """

    def test_a_routed_speech_task_with_no_lexical_item_flags(self) -> None:
        """What the owner asked to see: the task asked for words and none came."""
        folded = _without_redact(RedactionEvidence(lexical_words_n=0), speech=RunState.COMPLETED)
        assert "speech_no_lexical_item" in folded.ground_keys and folded.triage is not Triage.PASS
        assert any(NO_LEXICAL_ITEM_PRODUCED in reason.why for reason in folded.reasons)

    def test_it_is_still_releasable(self) -> None:
        """Flagging it does not withhold it: there is nothing in it to redact."""
        folded = _without_redact(RedactionEvidence(lexical_words_n=0), speech=RunState.COMPLETED)
        assert folded.release is Release.WITHOUT_REDACTION
        assert folded.release_ground == NO_LEXICAL_WORD

    def test_a_declined_branch_with_no_words_does_not_flag(self) -> None:
        """A task that asks for no words may produce none without it meaning anything."""
        folded = _without_redact(RedactionEvidence(), speech=RunState.SKIPPED, speech_route=DECLINED)
        assert not any(NO_LEXICAL_ITEM_PRODUCED in reason.why for reason in folded.reasons)

    def test_a_task_that_produced_words_does_not_flag(self) -> None:
        """The obvious control."""
        folded = _without_redact(RedactionEvidence(lexical_words_n=42, scanned=True), speech=RunState.COMPLETED)
        assert not any(NO_LEXICAL_ITEM_PRODUCED in reason.why for reason in folded.reasons)


class TestARedactNonPassIsVisibleWithoutFlippingTriage:
    """Triage asks whether a human must look; release asks whether an artifact may be handed on."""

    def test_a_surviving_finding_does_not_move_triage(self) -> None:
        """A release problem is not a measurement problem."""
        folded = _with_redact(Outcome.FAIL, speech=True, speech_route=ROUTED)
        assert folded.triage is Triage.PASS
        assert folded.release is Release.WITHHELD

    def test_it_appears_in_reasons_regardless(self) -> None:
        """A consumer filtering on triage == pass sees the release axis in the same record."""
        folded = _with_redact(Outcome.FAIL, speech=True, speech_route=ROUTED)
        assert any(reason.node == "REDACT" for reason in folded.reasons)

    def test_an_incomplete_verification_reruns(self) -> None:
        """REDACT's ``flag`` is verification that did not finish: the pipeline owes the recording a re-scan."""
        folded = _with_redact(Outcome.FLAG, speech=True, speech_route=ROUTED)
        assert folded.triage is Triage.RERUN
        assert "redact_rescan_incomplete" in folded.ground_keys
        assert folded.release is Release.WITHHELD


class TestReasonsCarryEveryContribution:
    """A flag naming one cause hides the others."""

    def test_two_non_conforming_branches_both_appear(self) -> None:
        """Not only the first, and not only the deciding one."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[
                _report("AIRWAY", "airway", conformance=False),
                _report("VOICE", "voice", conformance=False),
            ],
            spans_by_node=_found("AIRWAY", "VOICE"),
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=DECLINED, VOICE=ROUTED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.triage is Triage.FLAG
        assert len([reason for reason in folded.reasons if reason.outcome is Outcome.FLAG]) == 2

    def test_an_admit_failure_leads_the_reasons_without_erasing_them(self) -> None:
        """The deciding verdict reads first; what else the graph found is still in the record."""
        folded = fold_file_verdict(
            [
                NodeVerdict("PREPROCESS", Outcome.PASS, None, "conditioned"),
                NodeVerdict("ADMIT", Outcome.FAIL, None, "decode failure"),
                NodeVerdict("AIRWAY", Outcome.FLAG, "airway", "a labelled span is short"),
            ],
            branch_decisions=_decisions(AIRWAY=ROUTED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.triage is Triage.DISCARD
        assert folded.reasons[0].node == "ADMIT"
        assert {"PREPROCESS", "AIRWAY"} <= {reason.node for reason in folded.reasons}


_CONSENSUS_GONE = {"SPEECH": {"speech.lexical": "consensus_transcript: both asr blocks failed"}}


class TestACriticalAbsenceFlagsAndNamesItself:
    """A short-circuited run reaches the fold; it must not reach it silently."""

    def test_the_reason_names_the_branch_the_gate_and_the_recorded_absence(self) -> None:
        """A file the graph refused to route must say which measurement it was refused over."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_all_declined(),
            ran={},
            hint_claims={},
            route_state="unexplained",
            critical_absences=_CONSENSUS_GONE,
        )
        why = next(reason.why for reason in folded.reasons if reason.why.startswith(CRITICAL_ABSENCE))
        assert "SPEECH" in why
        assert "speech.lexical" in why
        assert "consensus_transcript" in why
        assert "both asr blocks failed" in why

    def test_it_reruns_rather_than_discarding(self) -> None:
        """Neither discard ground is a claim about a measurement that failed to be taken."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_all_declined(),
            ran={},
            hint_claims={},
            route_state="unexplained",
            critical_absences=_CONSENSUS_GONE,
        )
        assert folded.triage is Triage.RERUN
        assert "critical_absence" in folded.ground_keys
        assert folded.discard_ground is None
        assert folded.critical_absences == _CONSENSUS_GONE

    def test_an_admit_refusal_still_wins(self) -> None:
        """The existing refusal path is not weakened by a ground that only ever flags."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.FAIL, None, "decode failure")],
            branch_decisions=_all_declined(),
            ran={},
            hint_claims={},
            route_state="unexplained",
            critical_absences=_CONSENSUS_GONE,
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "unmeasurable"

    def test_an_ordinary_run_carries_no_critical_absence(self) -> None:
        """A branch partly unreadable is not a critical failure and must not read as one."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_decisions(AIRWAY=ROUTED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.critical_absences == {}
        assert not any(reason.why.startswith(CRITICAL_ABSENCE) for reason in folded.reasons)


class TestAWithheldBranchIsNotAnUnselectedOne:
    """Four states, and the two that share ``will_run: false`` need a second field to separate."""

    def test_the_branch_view_separates_withheld_from_not_selected(self) -> None:
        """Without this a branch nobody could ask reads exactly like one nobody wanted."""
        withheld = replace(_all_declined()["VOICE"], withheld_critical=True)
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions={**_all_declined(), "VOICE": withheld},
            ran={},
            hint_claims={},
            route_state="unexplained",
            critical_absences=_CONSENSUS_GONE,
        )
        assert folded.branches["VOICE"]["withheld_critical"] is True
        assert folded.branches["AIRWAY"]["withheld_critical"] is False
        assert folded.branches["VOICE"]["will_run"] is False
        assert folded.branches["AIRWAY"]["will_run"] is False

    def test_a_branch_that_ran_and_found_nothing_stays_distinct_from_both(self) -> None:
        """It reported; the two withheld states did not, and no report is what says so."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("VOICE", "voice")],
            branch_decisions=_decisions(VOICE=ROUTED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.findings["VOICE"] == "absent"
        assert folded.branches["VOICE"]["withheld_critical"] is False
        assert folded.branches["VOICE"]["will_run"] is True


class TestAgreementUnplacedFindingsAndASecondSpeaker:
    """Owner, 2026-09-27: agreeing with a mask is not hiding more; an unplaced finding and a second voice flag."""

    POLICY = FoldPolicy(llm_redaction_withholds=True, llm_second_speaker_flags=True)

    def _fold(
        self,
        reading: Mapping[str, Any],
        *,
        agreed: frozenset[int] = frozenset(),
        unplaced: Sequence[tuple[str, str]] = (),
        flag_gates: Sequence[Mapping[str, Any]] = (),
        policy: FoldPolicy | None = None,
        declared_family: str | None = None,
    ) -> FileVerdict:
        """A REDACT pass with one mask standing, and one reading."""
        return fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok"), NodeVerdict("REDACT", Outcome.PASS, None, "ok")],
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
            redaction=RedactionEvidence(findings_n=1, masks_n=1, masks_final_n=1),
            llm_redaction=reading,
            flag_gates=list(flag_gates),
            policy=policy or self.POLICY,
            agreed_redactions=agreed,
            unplaced=list(unplaced),
            declared_family=declared_family,
        )

    READING = {
        "status": "flagged",
        "original": "carries_pii",
        "speakers": "one",
        "proposal": [{"action": "redact", "category": "LOCATION", "text": "Wisconsin"}],
    }

    def test_a_redact_entry_a_mask_already_hides_withholds_nothing(self) -> None:
        """r6's open-response card: "Wisconsin" is masked, so agreeing with it releases the redacted copy."""
        assert self._fold(self.READING).release is Release.WITHHELD
        agreed = self._fold(self.READING, agreed=frozenset({0}))
        assert (agreed.release, agreed.triage) == (Release.WITH_REDACTION, Triage.PASS)

    def test_an_unplaced_finding_no_one_read_withholds_and_flags(self) -> None:
        """Nothing placed the finding and no reviewer read the transcript: nothing may be released."""
        folded = self._fold({}, unplaced=[("PERSON", UNPLACED_UNREAD)])
        assert (folded.release, folded.release_ground) == (Release.WITHHELD, UNPLACED_FINDING_UNREAD)
        assert folded.triage is Triage.FLAG

    def test_an_open_unplaced_finding_flags_and_a_placed_one_does_not(self) -> None:
        """A reading that settles nothing about it sends the recording to review; placing it settles it."""
        reading = {"status": "flagged", "original": "carries_pii", "proposal": []}
        opened = self._fold(reading, unplaced=[("PERSON", UNPLACED_OPEN)])
        assert opened.triage is Triage.FLAG
        assert any(reason.why.startswith(UNPLACED_FINDING_OPEN) for reason in opened.reasons)
        assert opened.release is Release.WITH_REDACTION
        placed = self._fold(reading, unplaced=[("PERSON", UNPLACED_PLACED)])
        assert placed.triage is Triage.PASS

    def test_a_second_speaker_flags_unless_diarization_already_did(self) -> None:
        """The reviewer's ``more_than_one`` flags; off by policy, or already a diarization flag, it adds nothing."""
        reading = {"status": "clean", "original": "clean", "speakers": "more_than_one", "proposal": []}
        heard = self._fold(reading)
        assert heard.triage is Triage.FLAG
        assert [reason.why for reason in heard.reasons if reason.node == "VERDICT"] == [
            f"{REVIEWER_HEARD_SECOND_SPEAKER}: no words quoted"
        ]
        assert heard.release is Release.WITH_REDACTION, "a flag for review, not a release decision"
        assert self._fold(reading, policy=FoldPolicy()).triage is Triage.PASS
        gate = {"gate": DOMINANT_SPEAKER_GATE, "passed": False, "ground": "another speaker", "reading": "share"}
        diarized = self._fold(reading, flag_gates=[gate])
        assert not any(reason.why.startswith(REVIEWER_HEARD_SECOND_SPEAKER) for reason in diarized.reasons)

    def test_a_participant_addressing_the_examiner_is_no_second_speaker(self) -> None:
        """Animal fluency, "Is that enough?": the reviewer reads one speaker, and one speaker flags nothing."""
        one = {"status": "clean", "original": "clean", "speakers": "one", "proposal": [], "other_speakers": []}
        assert self._fold(one).triage is Triage.PASS

    def test_a_second_speaker_the_task_expects_still_flags(self) -> None:
        """Owner, 2026-09-30: an examiner reading the story-recall instructions is another voice, and flags."""
        expected = {"text": "You were given the text.", "expected": True, "why": "the examiner's instruction"}
        reading = {"status": "clean", "original": "clean", "speakers": "more_than_one", "proposal": []}
        heard = self._fold({**reading, "other_speakers": [expected]})
        assert heard.triage is Triage.FLAG
        assert heard.release is Release.WITH_REDACTION, "a flag for review, not a release decision"
        assert [reason.why for reason in heard.reasons if reason.node == "VERDICT"] == [
            f"{REVIEWER_HEARD_SECOND_SPEAKER}: 1 passage(s) quoted, 1 expected by the instructions, 0 not"
        ]
        intruder = {"text": "Who are you talking to?", "expected": False, "why": "nobody the task asks for"}
        both = self._fold({**reading, "other_speakers": [expected, intruder]})
        assert [reason.why for reason in both.reasons if reason.node == "VERDICT"] == [
            f"{REVIEWER_HEARD_SECOND_SPEAKER}: 2 passage(s) quoted, 1 expected by the instructions, 1 not"
        ]
        assert not any("talking to" in reason.why for reason in both.reasons), "no transcript in a verdict"

    def test_an_unclear_reading_that_quotes_another_voice_flags(self) -> None:
        """r9 story recall: examiner instructions the reviewer cannot attribute still flag for a person to check."""
        examiner = {"text": "I said you have up to five minutes", "expected": True, "why": "possibly the examiner"}
        reading = {"status": "clean", "original": "clean", "speakers": "unclear", "proposal": []}
        heard = self._fold({**reading, "other_speakers": [examiner]})
        assert heard.triage is Triage.FLAG
        assert heard.release is Release.WITH_REDACTION, "a flag for review, not a release decision"
        assert [reason.why for reason in heard.reasons if reason.node == "VERDICT"] == [
            f"{REVIEWER_HEARD_SECOND_SPEAKER}: 1 passage(s) quoted, 1 expected by the instructions, 0 not"
        ]
        assert self._fold({**reading, "other_speakers": []}).triage is Triage.PASS, (
            "unclear with no quote flags nothing"
        )

    def test_a_model_speaker_on_harvard_flags_from_both_readers(self) -> None:
        """Harvard permits a model speaker: the reviewer's quote and the diarization gate each flag, and say so."""
        model = {"text": "The birch canoe slid on the smooth planks.", "expected": True, "why": "the model reading"}
        reading = {"status": "clean", "original": "clean", "speakers": "more_than_one", "proposal": []}
        heard = self._fold({**reading, "other_speakers": [model]}, declared_family="harvard-sentences-list")
        assert heard.triage is Triage.FLAG
        gate = {
            "gate": DOMINANT_SPEAKER_GATE,
            "passed": False,
            "ground": "another speaker holds part of the task extent",
            "reading": "extent_dominant_speaker_share",
            "value": 0.88,
            "bound": 0.9,
        }
        policy = replace(self.POLICY, model_speaker_families=("harvard-sentences-list",))
        diarized = self._fold(
            {**reading, "speakers": "one", "other_speakers": []},
            flag_gates=[gate],
            policy=policy,
            declared_family="harvard-sentences-list",
        )
        assert diarized.triage is Triage.FLAG
        assert [reason.why for reason in diarized.reasons if reason.node == "VERDICT"] == [
            "another speaker holds part of the task extent: extent_dominant_speaker_share read 0.88 against 0.9; "
            f"{MODEL_SPEAKER_PERMITTED}"
        ]

    def test_the_packaged_config_exempts_no_family_from_the_speaker_gate(self) -> None:
        """Owner, 2026-09-30: no exemption; the model-speaker families are named for the ground only."""
        from senselab.audio.workflows.triage.config import load_triage_config
        from senselab.audio.workflows.triage.nodes.verdict import flag_gate_exemptions

        config = load_triage_config()
        assert flag_gate_exemptions(config, "harvard-sentences-list") == ()
        assert "harvard-sentences-list" in FoldPolicy.from_config(config).model_speaker_families

    def test_a_flagged_reading_that_names_no_words_flags_for_review(self) -> None:
        """Owner, 2026-09-28: a judgment with no entries moves no mask and goes to a person."""
        reading = {"status": "flagged", "original": "clean", "redaction": "incomplete", "proposal": []}
        on = self._fold(reading, policy=FoldPolicy(llm_redaction_withholds=True, llm_contradiction_flags=True))
        assert on.triage is Triage.FLAG and REVIEWER_NAMED_NO_WORDS in [reason.why for reason in on.reasons]
        assert on.release is Release.WITH_REDACTION, "a flag for review, not a release decision"
        assert self._fold(reading).triage is Triage.PASS
        agreeing = {"status": "flagged", "original": "carries_pii", "redaction": "complete", "proposal": []}
        assert self._fold(agreeing, policy=FoldPolicy(llm_contradiction_flags=True)).triage is Triage.PASS
        named = {**reading, "proposal": [{"action": "release", "text": "alice", "category": "PERSON"}]}
        assert REVIEWER_NAMED_NO_WORDS not in [
            r.why for r in self._fold(named, policy=FoldPolicy(llm_contradiction_flags=True)).reasons
        ]

    def test_the_packaged_config_turns_the_contradiction_flag_on(self) -> None:
        """Owner, 2026-09-28."""
        from senselab.audio.workflows.triage.config import load_triage_config

        assert FoldPolicy.from_config(load_triage_config()).llm_contradiction_flags is True

    def test_the_packaged_config_turns_the_second_speaker_flag_on(self) -> None:
        """Owner, 2026-09-27."""
        from senselab.audio.workflows.triage.config import load_triage_config

        assert FoldPolicy.from_config(load_triage_config()).llm_second_speaker_flags is True


class TestSecondOpinionDisagreementFlagsForReview:
    """Owner, 2026-10-01: a confident second-opinion disagreement with the reviewer flags for review."""

    _PASSED = [NodeVerdict("REDACT", Outcome.PASS, None, "every finding redacted")]
    _ON = FoldPolicy(
        second_opinion_disagreement_flags=True, second_opinion_confident_yes=0.8, second_opinion_confident_no=0.2
    )

    @staticmethod
    def _opinion(**probabilities: float) -> dict[str, Any]:
        held = {"other_voice": 0.02, "instructions_spoken": 0.02, "policy_identifier_present": 0.02}
        held.update(probabilities)
        return {"status": "ok", "probabilities": held, "model_id": "ollama:clef:27b", "blob_digest": "sha256:ab"}

    @staticmethod
    def _reading(**fields: Any) -> dict[str, Any]:  # noqa: ANN401
        reading: dict[str, Any] = {"status": "clean", "original": "clean", "speakers": "one", "proposal": []}
        reading.update(fields)
        return reading

    def _fold(
        self,
        opinion: Mapping[str, Any] | None,
        reading: Mapping[str, Any],
        policy: FoldPolicy,
        *,
        masks_final_n: int = 0,
        reviewer_requested_n: int = 0,
    ) -> FileVerdict:
        evidence = RedactionEvidence(
            lexical_words_n=40,
            scanned=True,
            findings_n=masks_final_n,
            masks_n=masks_final_n,
            masks_final_n=masks_final_n,
            reviewer_requested_n=reviewer_requested_n,
        )
        return fold_file_verdict(
            self._PASSED,
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={"SPEECH": RunState.COMPLETED, "REDACT": RunState.COMPLETED, "REVIEW": RunState.COMPLETED},
            hint_claims={},
            route_state=ROUTED,
            redaction=evidence,
            llm_redaction=dict(reading),
            policy=policy,
            second_opinion=opinion,
        )

    def _grounds(self, folded: FileVerdict) -> list[str]:
        return [str(reason.why) for reason in folded.reasons if str(reason.why).startswith(SECOND_OPINION_DISAGREES)]

    def test_a_confident_yes_against_the_reviewers_no_flags_and_names_it(self) -> None:
        """The story-recall pilot case: p=0.85 another voice, the reviewer heard one."""
        folded = self._fold(self._opinion(other_voice=0.85), self._reading(), self._ON)
        assert "second_opinion_disagreement" in folded.ground_keys and folded.triage is not Triage.PASS
        assert self._grounds(folded) == [f"{SECOND_OPINION_DISAGREES}: other_voice p=0.85 reviewer=no"]
        assert folded.record()["second_opinion"]["disagreements"] == ["other_voice p=0.85 reviewer=no"]

    def test_a_confident_no_against_the_reviewers_yes_flags(self) -> None:
        """The reviewer's reading leaves a mask standing; the model is sure nothing the policy removes is there."""
        folded = self._fold(self._opinion(policy_identifier_present=0.05), self._reading(), self._ON, masks_final_n=1)
        assert self._grounds(folded) == [f"{SECOND_OPINION_DISAGREES}: policy_identifier_present p=0.05 reviewer=yes"]

    def test_an_identifier_the_reviewer_cleared_flags_when_the_model_is_sure(self) -> None:
        """No mask stands and the reviewer asks for none; the model is sure an identifier is there."""
        folded = self._fold(self._opinion(policy_identifier_present=0.93), self._reading(), self._ON)
        assert self._grounds(folded) == [f"{SECOND_OPINION_DISAGREES}: policy_identifier_present p=0.93 reviewer=no"]
        requested = self._fold(
            self._opinion(policy_identifier_present=0.93), self._reading(), self._ON, reviewer_requested_n=1
        )
        assert not self._grounds(requested)

    def test_a_listed_condition_is_never_a_disagreement(self) -> None:
        """r12, sub-00053adb free-speech-2: conditions listed, an old named_diagnosis p=0.96; no flag."""
        reading = self._reading(
            status="flagged",
            conditions=[{"text": "essential tremors", "why": "x"}, {"text": "synovial joint cyst", "why": "y"}],
        )
        opinion = self._opinion()
        opinion["probabilities"]["named_diagnosis"] = 0.96
        assert not self._grounds(self._fold(opinion, reading, self._ON))
        v6 = self._reading(
            status="flagged", proposal=[{"text": "asthma", "action": "redact", "category": "CONDITION", "why": "x"}]
        )
        assert not self._grounds(self._fold(opinion, v6, self._ON))

    def test_the_middle_band_and_agreement_do_not_flag(self) -> None:
        """0.5 is not confident; a confident yes the reviewer shares is agreement."""
        assert not self._grounds(self._fold(self._opinion(other_voice=0.5), self._reading(), self._ON))
        agreed = self._reading(speakers="more_than_one")
        assert not self._grounds(self._fold(self._opinion(other_voice=0.95), agreed, self._ON))

    def test_instructions_are_compared_only_where_the_reading_carries_the_part(self) -> None:
        """A pre-v5 reading never answered the question, so it cannot disagree."""
        opinion = self._opinion(instructions_spoken=0.95)
        assert not self._grounds(self._fold(opinion, self._reading(), self._ON))
        carried = self._reading(instructions_spoken=[])
        assert self._grounds(self._fold(opinion, carried, self._ON))

    def test_nothing_is_compared_without_both_readings(self) -> None:
        """An absent opinion, or a reviewer that read nothing, contributes nothing."""
        absent = {"status": "absent", "probabilities": {}}
        assert not self._grounds(self._fold(absent, self._reading(), self._ON))
        unread = self._reading(status="absent")
        assert not self._grounds(self._fold(self._opinion(other_voice=0.99), unread, self._ON))
        assert self._fold(None, self._reading(), self._ON).record()["second_opinion"] == {}

    def test_the_switch_off_or_unmeasured_thresholds_do_not_flag(self) -> None:
        """Off, or either threshold null, leaves the ground silent."""
        opinion = self._opinion(other_voice=0.99)
        for policy in (
            FoldPolicy(second_opinion_confident_yes=0.8, second_opinion_confident_no=0.2),
            FoldPolicy(
                second_opinion_disagreement_flags=True,
                second_opinion_confident_yes=None,
                second_opinion_confident_no=0.2,
            ),
        ):
            assert not self._grounds(self._fold(opinion, self._reading(), policy))

    def test_the_release_is_unchanged(self) -> None:
        """The flag is triage only."""
        opinion = self._opinion(other_voice=0.99)
        on = self._fold(opinion, self._reading(), self._ON)
        off = self._fold(opinion, self._reading(), FoldPolicy())
        assert on.release is off.release

    def test_the_packaged_policy_flags_at_the_proposed_thresholds(self) -> None:
        """Owner's switch on; thresholds 0.8 and 0.2, unfitted."""
        from senselab.audio.workflows.triage.config import load_triage_config

        policy = FoldPolicy.from_config(load_triage_config())
        assert (
            policy.second_opinion_disagreement_flags,
            policy.second_opinion_confident_yes,
            policy.second_opinion_confident_no,
        ) == (
            True,
            0.8,
            0.2,
        )


class TestSpokenInstructionsFlagForReview:
    """Owner, 2026-10-01: a reading quoting the task's instructions spoken in the recording flags for review."""

    _PASSED = [NodeVerdict("REDACT", Outcome.PASS, None, "every finding redacted")]
    _QUOTE = "I said you have up to five minutes to read it as many times as you want"

    def _fold(self, spoken: list[str], policy: FoldPolicy) -> FileVerdict:
        reading = {
            "status": "clean",
            "original": "clean",
            "speakers": "one",
            "proposal": [],
            "instructions_spoken": spoken,
        }
        return fold_file_verdict(
            self._PASSED,
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={"SPEECH": RunState.COMPLETED, "REDACT": RunState.COMPLETED, "REVIEW": RunState.COMPLETED},
            hint_claims={},
            route_state=ROUTED,
            redaction=RedactionEvidence(lexical_words_n=40, scanned=True, findings_n=0),
            llm_redaction=reading,
            policy=policy,
        )

    def test_a_quoted_passage_flags_and_names_it(self) -> None:
        """The story-recall card's quote is a flag ground carrying the words."""
        folded = self._fold([self._QUOTE], FoldPolicy(llm_instructions_spoken_flags=True))
        grounds = [str(reason.why) for reason in folded.reasons]
        assert "instructions_spoken" in folded.ground_keys and folded.triage is not Triage.PASS
        assert any(why == f"{INSTRUCTIONS_SPOKEN}: 1 passage(s) quoted" for why in grounds)
        assert not any("five minutes" in why for why in grounds), "no transcript in a verdict"

    def test_an_empty_part_or_the_policy_off_does_not_flag(self) -> None:
        """[] never flags, and the key off turns the ground off."""
        for spoken, policy in (([], FoldPolicy(llm_instructions_spoken_flags=True)), ([self._QUOTE], FoldPolicy())):
            grounds = [str(reason.why) for reason in self._fold(spoken, policy).reasons]
            assert not any(why.startswith(INSTRUCTIONS_SPOKEN) for why in grounds)

    def test_the_release_is_unchanged(self) -> None:
        """The flag is triage only."""
        on = self._fold([self._QUOTE], FoldPolicy(llm_instructions_spoken_flags=True))
        off = self._fold([self._QUOTE], FoldPolicy())
        assert on.release is off.release

    def test_the_packaged_policy_flags(self) -> None:
        """The packaged config turns the ground on."""
        from senselab.audio.workflows.triage.config import load_triage_config

        assert FoldPolicy.from_config(load_triage_config()).llm_instructions_spoken_flags is True


_SENTINEL = "Zebulon Quimby of Kalamazoo"
"""A string that stands for transcript text; no verdict ground may carry it."""


def _every_ground_fold(**overrides: Any) -> FileVerdict:  # noqa: ANN401 — fold keyword arguments of every type
    """One fold that raises every participant ground the fold builds, each fed transcript-like text.

    Args:
        **overrides: Keyword arguments replacing the defaults below.

    Returns:
        The folded file verdict.
    """
    policy = FoldPolicy(
        llm_redaction_flags=True,
        person_name_review_flags=True,
        llm_second_speaker_flags=True,
        llm_contradiction_flags=True,
        llm_instructions_spoken_flags=True,
        second_opinion_disagreement_flags=True,
        second_opinion_confident_yes=0.8,
        second_opinion_confident_no=0.2,
        uncomputed_reading_flags=True,
    )
    annotation = {
        "status": "flagged",
        "original": "carries_pii",
        "redaction": "incomplete",
        "speakers": "more_than_one",
        "other_speakers": [{"text": _SENTINEL, "expected": False, "why": _SENTINEL}],
        "instructions_spoken": [_SENTINEL],
        "conditions": [{"text": _SENTINEL, "why": _SENTINEL}],
        "proposal": [{"action": "redact", "category": "PERSON", "text": _SENTINEL, "why": _SENTINEL}],
    }
    kwargs: dict[str, Any] = {
        "branch_reports": [_report("SPEECH", "speech", conformance=False)],
        "spans_by_node": _found("SPEECH"),
        "branch_decisions": _decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
        "ran": {"SPEECH": RunState.COMPLETED},
        "hint_claims": {},
        "route_state": ROUTED,
        "declared_family": "free-speech",
        "redaction": RedactionEvidence(
            lexical_words_n=12,
            scanned=True,
            findings_n=1,
            masks_n=1,
            masks_final_n=1,
            person_names_masked_n=2,
            name_release_proposed=(_SENTINEL,),
            reviewer_requested_n=1,
        ),
        "llm_redaction": annotation,
        "second_opinion": {
            "status": "ok",
            "probabilities": {"other_voice": 0.01, "instructions_spoken": 0.01, "policy_identifier_present": 0.01},
        },
        "flag_gates": [
            {
                "gate": DOMINANT_SPEAKER_GATE,
                "passed": False,
                "ground": "another speaker",
                "reading": "share",
                "value": 0.5,
            }
        ],
        "unplaced": [("PERSON", UNPLACED_OPEN)],
        "policy": policy,
    }
    kwargs.update(overrides)
    return fold_file_verdict([NodeVerdict("ADMIT", Outcome.PASS, None, "ok")], **kwargs)


class TestEveryGroundHasAKeyAndNoTranscript:
    """DAG review proposal 1: a flag is countable from its key, and no transcript text reaches a verdict."""

    def test_every_flag_carries_a_key_from_the_vocabulary(self) -> None:
        """Each key is a named ground or ``<prefix>:<name>`` under a declared prefix."""
        folded = _every_ground_fold()
        flags = [reason for reason in folded.reasons if reason.outcome is Outcome.FLAG]
        assert len(flags) >= 7
        for reason in flags:
            key = ground_key(reason)
            assert key in GROUND_KEYS or key.split(":", 1)[0] in GROUND_KEY_PREFIXES, key
        assert folded.ground_keys == sorted({ground_key(reason) for reason in flags})

    def test_the_sweep_raises_the_participant_grounds_it_is_meant_to(self) -> None:
        """The sweep below is only as good as the grounds it reaches."""
        keys = set(_every_ground_fold().ground_keys)
        assert {
            "reviewer_residue",
            "person_name_review",
            "instructions_spoken",
            "second_opinion_disagreement",
            "unplaced_finding_open",
            "gate:dominant_speaker_share_min",
            "conformance:SPEECH",
        } <= keys

    def test_no_ground_quotes_transcript_text(self) -> None:
        """Invariant 6: quoted words and proposed names stay in REVIEW's annotation, never in a reason."""
        folded = _every_ground_fold()
        assert not any(_SENTINEL in reason.why for reason in folded.reasons)
        heard = _every_ground_fold(flag_gates=[])
        assert "reviewer_second_speaker" in heard.ground_keys
        assert not any(_SENTINEL in reason.why for reason in heard.reasons)
        record = folded.record()
        assert not any(_SENTINEL in str(reason["why"]) for reason in record["reasons"])

    def test_the_record_carries_the_keys(self) -> None:
        """What the parquet reads: the ground keys, a key per reason, and the release ground's key."""
        record = _every_ground_fold().record()
        assert record["ground_keys"] == _every_ground_fold().ground_keys
        assert all(reason["key"] for reason in record["reasons"])
        assert record["release_ground_key"] == release_ground_key(record["release_ground"])

    def test_every_release_ground_has_a_key(self) -> None:
        """The release axis is keyed the same way; REDACT's own decision has its key too."""
        for ground in (
            *RELEASE_WITHOUT_REDACTION_GROUNDS,
            *RELEASE_UNKNOWN_GROUNDS,
            *RELEASE_WITHHELD_GROUNDS,
            *RELEASE_WITH_REDACTION_GROUNDS,
        ):
            assert release_ground_key(ground) != ground
        assert release_ground_key(None) == "redact_decided"


class TestOperationalGroundsRerun:
    """DAG review proposal 4: a missing derivative is the pipeline's to fix, not the participant's."""

    def test_a_participant_ground_alone_flags(self) -> None:
        """The control: nothing operational, so the file goes to a person."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("SPEECH", "speech", conformance=False)],
            spans_by_node=_found("SPEECH"),
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.triage is Triage.FLAG
        assert folded.ground_keys == ["conformance:SPEECH"]

    def test_no_classifier_output_reruns_and_keeps_the_participant_ground(self) -> None:
        """TAXONOMY with nothing to consolidate outranks the review flag; both keys stay."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("TAXONOMY", Outcome.FLAG, None, "no per-span classifier produced scores"),
            ],
            branch_reports=[_report("SPEECH", "speech", conformance=False)],
            spans_by_node=_found("SPEECH"),
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.triage is Triage.RERUN
        assert folded.ground_keys == ["conformance:SPEECH", "taxonomy_no_classifier"]

    def test_every_operational_key_reads_as_operational(self) -> None:
        """The classification is one function, so the parquet and the fold cannot disagree."""
        for key in OPERATIONAL_GROUND_KEYS:
            assert is_operational(key)
        assert is_operational("branch_silent:SPEECH")
        assert is_operational("unmeasured_operating_point:VOICE")
        for participant in ("conformance:SPEECH", "gate:events_min", "person_name_review", "acoustically_empty"):
            assert not is_operational(participant)
        assert not is_operational(None)

    def test_a_rerun_leaves_the_release_axis_alone(self) -> None:
        """The release is read from the evidence as before; a rerun state changes no artefact."""
        common: dict[str, Any] = {
            "branch_decisions": _decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            "ran": {"SPEECH": RunState.COMPLETED},
            "hint_claims": {},
            "route_state": ROUTED,
            "redaction": RedactionEvidence(lexical_words_n=5, scanned=True),
            "branch_reports": [_report("SPEECH", "speech", conformance=True)],
            "spans_by_node": _found("SPEECH"),
        }
        clean = fold_file_verdict([NodeVerdict("ADMIT", Outcome.PASS, None, "ok")], **common)
        owed = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("TAXONOMY", Outcome.FLAG, None, "no per-span classifier produced scores"),
            ],
            **common,
        )
        assert owed.triage is Triage.RERUN
        assert (owed.release, owed.release_ground) == (clean.release, clean.release_ground)


class TestAnEmptyRecordingDiscards:
    """DAG review proposal 3: emptiness is folded before the conformance grounds."""

    def test_a_forced_branch_reporting_non_conformance_does_not_save_an_empty_recording(self) -> None:
        """The r12 shape: a declared branch forced to run on an empty file reports non-conformance."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_reports=[_report("SPEECH", "speech", conformance=False)],
            branch_decisions=_all_declined(forced=("SPEECH",)),
            ran={"SPEECH": RunState.COMPLETED},
            hint_claims={"SPEECH": True},
            route_state="empty",
            declared_family="harvard-sentences",
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "acoustically_empty"
        assert "conformance:SPEECH" in folded.ground_keys


class TestTheDeclaredTaskDecides:
    """Owner, 2026-10-05: a fragment is not the task, and only the declared task's branch can say it was done."""

    _ADMIT_OK = [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")]

    def test_a_short_filler_only_sentence_discards_as_too_short(self) -> None:
        """A 0.3 s "[UM]" Harvard sentence, SPEECH holding a speaker turn: too short for the task."""
        folded = fold_file_verdict(
            self._ADMIT_OK,
            branch_reports=[_report("SPEECH", "speech", conformance=False)],
            spans_by_node=_found("SPEECH"),
            branch_decisions=_all_declined(forced=("SPEECH",)),
            ran={"SPEECH": RunState.COMPLETED},
            hint_claims={"SPEECH": True},
            route_state="empty",
            declared_family="harvard-sentences-list",
            redaction=RedactionEvidence(lexical_words_n=0, scanned=True),
            task=TaskEvidence(owning_branches=("SPEECH",), duration_s=0.3, minimum_duration_s=1.0),
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == TOO_SHORT_FOR_TASK
        assert TOO_SHORT_FOR_TASK in folded.ground_keys

    def test_a_single_breath_extent_does_not_perform_a_five_breath_task(self) -> None:
        """AIRWAY's activity-envelope extent on an empty route is a span, not the task: it discards."""
        folded = fold_file_verdict(
            self._ADMIT_OK,
            branch_reports=[_report("AIRWAY", "airway", conformance=UNDETERMINED)],
            spans_by_node=_found("AIRWAY"),
            branch_decisions=_all_declined(forced=("AIRWAY",)),
            ran={"AIRWAY": RunState.COMPLETED},
            hint_claims={"AIRWAY": True},
            route_state="empty",
            declared_family="respiration-and-cough-fivebreaths",
            task=TaskEvidence(owning_branches=("AIRWAY",), duration_s=1.2, minimum_duration_s=1.0),
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "acoustically_empty"
        assert not any(key.startswith("route_mismatch") for key in folded.ground_keys)

    def test_a_short_single_breath_recording_discards_as_too_short(self) -> None:
        """A 0.16 s single-breath fivebreaths recording: far shorter than the task can take."""
        folded = fold_file_verdict(
            self._ADMIT_OK,
            branch_reports=[_report("AIRWAY", "airway", conformance=UNDETERMINED)],
            spans_by_node=_found("AIRWAY"),
            branch_decisions=_all_declined(forced=("AIRWAY",)),
            ran={"AIRWAY": RunState.COMPLETED},
            hint_claims={"AIRWAY": True},
            route_state="empty",
            declared_family="respiration-and-cough-fivebreaths",
            task=TaskEvidence(owning_branches=("AIRWAY",), duration_s=0.16, minimum_duration_s=1.0),
        )
        assert folded.discard_ground == TOO_SHORT_FOR_TASK

    def test_a_cough_token_corroborates_an_undecided_cough_task(self) -> None:
        """[cough] in a cough task, AIRWAY undecided but finding its kind: the task stands."""
        folded = fold_file_verdict(
            self._ADMIT_OK,
            branch_reports=[_report("AIRWAY", "airway", conformance=UNDETERMINED)],
            spans_by_node=_found("AIRWAY"),
            branch_decisions=_all_declined(forced=("AIRWAY",)),
            ran={"AIRWAY": RunState.COMPLETED},
            hint_claims={"AIRWAY": True},
            route_state="empty",
            declared_family="respiration-and-cough-cough",
            task=TaskEvidence(owning_branches=("AIRWAY",), duration_s=4.0, minimum_duration_s=1.0, event_tokens_n=2),
        )
        assert folded.triage is not Triage.DISCARD
        assert "route_mismatch:AIRWAY" in folded.ground_keys

    def test_a_filler_token_corroborates_no_cough_task(self) -> None:
        """No [cough] token (an [UM] is not one): the same recording discards."""
        folded = fold_file_verdict(
            self._ADMIT_OK,
            branch_reports=[_report("AIRWAY", "airway", conformance=UNDETERMINED)],
            spans_by_node=_found("AIRWAY"),
            branch_decisions=_all_declined(forced=("AIRWAY",)),
            ran={"AIRWAY": RunState.COMPLETED},
            hint_claims={"AIRWAY": True},
            route_state="empty",
            declared_family="respiration-and-cough-cough",
            task=TaskEvidence(owning_branches=("AIRWAY",), duration_s=4.0, minimum_duration_s=1.0, event_tokens_n=0),
        )
        assert folded.discard_ground == "acoustically_empty"

    def test_an_airway_task_with_only_speech_found_discards_for_lack_of_the_task(self) -> None:
        """SPEECH read words, AIRWAY ran and found nothing: the declared airway task is absent."""
        folded = fold_file_verdict(
            self._ADMIT_OK,
            branch_reports=[
                _report("AIRWAY", "airway", conformance=False),
                BranchReport(node="SPEECH", kind="speech", conformance=UNDETERMINED, conformance_of=TASK),
            ],
            spans_by_node=_found("SPEECH"),
            branch_decisions=_decisions(forced=("AIRWAY",), AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={"AIRWAY": RunState.COMPLETED, "SPEECH": RunState.COMPLETED},
            hint_claims={"AIRWAY": True},
            route_state="routed",
            declared_family="respiration-and-cough-cough",
            redaction=RedactionEvidence(lexical_words_n=6, scanned=True),
            task=TaskEvidence(owning_branches=("AIRWAY",), duration_s=6.0, minimum_duration_s=1.0),
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == DECLARED_TASK_ABSENT

    def test_a_speech_task_holding_no_lexical_word_has_no_task_match(self) -> None:
        """Only bracketed tokens on a reading task: the declared speech task is absent."""
        folded = fold_file_verdict(
            self._ADMIT_OK,
            branch_reports=[_report("SPEECH", "speech", conformance=False)],
            spans_by_node=_found("SPEECH"),
            branch_decisions=_decisions(SPEECH=ROUTED, AIRWAY=DECLINED, VOICE=DECLINED),
            ran={"SPEECH": RunState.COMPLETED},
            hint_claims={"SPEECH": True},
            route_state="routed",
            declared_family="harvard-sentences-list",
            redaction=RedactionEvidence(lexical_words_n=0, scanned=True),
            task=TaskEvidence(owning_branches=("SPEECH",), duration_s=3.0, minimum_duration_s=1.0),
        )
        assert folded.discard_ground == DECLARED_TASK_ABSENT

    def test_a_performed_task_is_unaffected(self) -> None:
        """A normal Harvard sentence: conformant, long enough, words read -- it passes."""
        folded = fold_file_verdict(
            self._ADMIT_OK,
            branch_reports=[_report("SPEECH", "speech", conformance=True)],
            spans_by_node=_found("SPEECH"),
            branch_decisions=_decisions(SPEECH=ROUTED, AIRWAY=DECLINED, VOICE=DECLINED),
            ran={"SPEECH": RunState.COMPLETED},
            hint_claims={"SPEECH": True},
            route_state="routed",
            declared_family="harvard-sentences-list",
            redaction=RedactionEvidence(lexical_words_n=8, scanned=True),
            task=TaskEvidence(owning_branches=("SPEECH",), duration_s=3.5, minimum_duration_s=1.0),
        )
        assert folded.triage is Triage.PASS
        assert folded.discard_ground is None

    def test_a_missing_derivative_reruns_before_the_task_is_called_absent(self) -> None:
        """A route the ruleset could not explain is owed a rerun; the empty task waits for it."""
        folded = fold_file_verdict(
            self._ADMIT_OK,
            branch_reports=[_report("SPEECH", "speech", conformance=False)],
            spans_by_node={},
            branch_decisions=_decisions(SPEECH=ROUTED, AIRWAY=DECLINED, VOICE=DECLINED),
            ran={"SPEECH": RunState.COMPLETED},
            hint_claims={"SPEECH": True},
            route_state="unexplained",
            declared_family="diadochokinesis-ka",
            redaction=RedactionEvidence(lexical_words_n=0, scanned=True),
            task=TaskEvidence(owning_branches=("SPEECH",), duration_s=3.9, minimum_duration_s=1.0),
        )
        assert folded.triage is Triage.RERUN
        assert folded.discard_ground is None

    def _breath(self, *, route_state: str, duration_s: float, absent: tuple[str, ...]) -> FileVerdict:
        return fold_file_verdict(
            self._ADMIT_OK,
            branch_reports=[_report("AIRWAY", "airway", conformance=UNDETERMINED)],
            spans_by_node={},
            branch_decisions=(
                _all_declined(forced=("AIRWAY",))
                if route_state == "empty"
                else _decisions(AIRWAY=ROUTED, SPEECH=DECLINED, VOICE=DECLINED)
            ),
            ran={"AIRWAY": RunState.COMPLETED},
            hint_claims={"AIRWAY": True},
            route_state=route_state,
            declared_family="respiration-and-cough-breath",
            task=TaskEvidence(
                owning_branches=("AIRWAY",),
                duration_s=duration_s,
                minimum_duration_s=1.0,
                owner_absent_inputs=absent,
            ),
        )

    def test_an_owner_without_its_instrument_reruns_rather_than_calling_the_task_absent(self) -> None:
        """AIRWAY had no hear_scores to look with and found nothing: owed a rerun, not a discard."""
        folded = self._breath(route_state="routed", duration_s=1.98, absent=("AIRWAY:hear_scores",))
        assert folded.triage is Triage.RERUN
        assert folded.discard_ground is None
        assert KEY_OWNING_BRANCH_INPUT_ABSENT in folded.ground_keys

    def test_an_owner_with_its_instruments_finding_nothing_is_still_task_absent(self) -> None:
        """The same recording with every input present: the declared task is absent."""
        folded = self._breath(route_state="routed", duration_s=1.98, absent=())
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == DECLARED_TASK_ABSENT

    def test_an_empty_route_whose_owner_lacked_an_input_reruns(self) -> None:
        """An empty route is not called empty while the owning branch could not look."""
        folded = self._breath(route_state="empty", duration_s=3.0, absent=("AIRWAY:hear_scores",))
        assert folded.triage is Triage.RERUN
        assert folded.discard_ground is None

    def test_a_too_short_recording_discards_whatever_its_owner_lacked(self) -> None:
        """A duration needs no instrument: too short stays a discard."""
        folded = self._breath(route_state="routed", duration_s=0.3, absent=("AIRWAY:hear_scores",))
        assert folded.discard_ground == TOO_SHORT_FOR_TASK


class TestABreathTaskIsDecidedOnDetectedBreaths:
    """Owner, 2026-10-05: a breath task stands on breath events AIRWAY detected, never on HeAR or [breath]."""

    _ADMIT_OK = [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")]

    def _fold(
        self,
        *,
        family: str,
        conformance: Any,  # noqa: ANN401
        events: int | None,
        instructed: int | None = None,
        gates: Mapping[str, Any] | None = None,
        absent: tuple[str, ...] = (),
        tokens: int = 0,
    ) -> FileVerdict:
        return fold_file_verdict(
            self._ADMIT_OK,
            branch_reports=[_report("AIRWAY", "airway", conformance=conformance)],
            spans_by_node=_found("AIRWAY"),
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={"AIRWAY": RunState.COMPLETED},
            hint_claims={"AIRWAY": True},
            route_state="routed",
            declared_family=family,
            gates=gates,
            task=TaskEvidence(
                owning_branches=("AIRWAY",),
                duration_s=30.0,
                minimum_duration_s=1.0,
                event_tokens_n=tokens,
                owner_absent_inputs=absent,
                required_event="breath",
                events_found_n=events,
                event_kind="breath",
                instructed_count=instructed,
            ),
        )

    def test_a_sustained_breath_recording_with_a_detected_breath_passes(self) -> None:
        """One breath event: events_min holds, the task was performed."""
        folded = self._fold(family="respiration-and-cough-breath", conformance=True, events=1)
        assert folded.triage is Triage.PASS
        assert folded.discard_ground is None

    def test_a_sustained_breath_recording_with_no_detected_breath_discards(self) -> None:
        """Zero breath events with every input present: no breath captured."""
        folded = self._fold(family="respiration-and-cough-breath", conformance=False, events=0)
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == NO_BREATH_CAPTURED
        assert NO_BREATH_CAPTURED in folded.ground_keys

    def test_a_breath_too_quiet_to_detect_is_no_breath_captured(self) -> None:
        """A counted breath task whose breaths the detector could not find discards the same way."""
        folded = self._fold(
            family="respiration-and-cough-v2-threebreathsmouth", conformance=False, events=0, instructed=3
        )
        assert folded.discard_ground == NO_BREATH_CAPTURED

    def test_absent_hear_scores_rerun_rather_than_discard(self) -> None:
        """AIRWAY lacked hear_scores, so it reported no count and could not look: owed a rerun."""
        folded = self._fold(
            family="respiration-and-cough-breath",
            conformance=UNDETERMINED,
            events=None,
            absent=("AIRWAY:hear_scores",),
        )
        assert folded.triage is Triage.RERUN
        assert folded.discard_ground is None

    def test_one_long_breath_for_three_quick_ones_flags_a_task_mismatch(self) -> None:
        """A breath was detected, but one where three were instructed: flagged, never discarded."""
        folded = self._fold(
            family="respiration-and-cough-threequickbreaths",
            conformance=False,
            events=1,
            instructed=3,
            gates={
                "node": "AIRWAY",
                "applied": [
                    {"gate": "events_min", "reading": "airway_events_found", "value": 1, "passed": True},
                    {
                        "gate": "instructed_count_min_fraction",
                        "reading": "instructed_count_fraction",
                        "value": 0.33,
                        "passed": False,
                    },
                ],
            },
        )
        assert folded.triage is Triage.FLAG
        assert folded.discard_ground is None
        assert KEY_TASK_MISMATCH in folded.ground_keys
        assert "conformance:AIRWAY" not in folded.ground_keys
        [reason] = [reason for reason in folded.reasons if reason.key == KEY_TASK_MISMATCH]
        assert "detected 1 breath events where 3 were instructed" in reason.why

    def test_a_breath_token_alone_supports_nothing(self) -> None:
        """[breath] on a recording with no detected breath does not stand in for one."""
        folded = self._fold(family="respiration-and-cough-breath", conformance=UNDETERMINED, events=0, tokens=3)
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == NO_BREATH_CAPTURED
