"""The file-level fold: pass, flag, discard, and each branch authority over its own subject."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from senselab.audio.workflows.triage.vocabulary import (
    BAD_MAP_VALUES,
    DECLINED,
    ROUTED,
    UNAVAILABLE,
    UNEXPLAINED_CONTENT,
    UNREAD_DECLARATION,
    BranchDecision,
    FileVerdict,
    NodeVerdict,
    Outcome,
    Release,
    RunState,
    Triage,
    fold_file_verdict,
)


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


def _with_redact(redact: Outcome, *, speech: Outcome | None = None, speech_route: str = DECLINED) -> FileVerdict:
    """A fold whose only interesting node is REDACT, optionally with a SPEECH branch beside it.

    Args:
        redact: What REDACT concluded.
        speech: What SPEECH concluded, or None when the branch never ran.
        speech_route: What the ruleset made of SPEECH.

    Returns:
        The folded file verdict.
    """
    node_verdicts = [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")]
    if speech is not None:
        node_verdicts.append(NodeVerdict("SPEECH", speech, "speech", "words in the store"))
    node_verdicts.append(NodeVerdict("REDACT", redact, None, "the scan concluded"))
    return fold_file_verdict(
        node_verdicts,
        branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=speech_route, VOICE=DECLINED),
        ran={},
        hint_claims={},
        route_state=ROUTED,
    )


class TestTheTriageVocabulary:
    """pass, flag, discard — three values, and fail is not one of them."""

    def test_the_members_are_exactly_three(self) -> None:
        """verdict.md's triage axis; a branch's ``fail`` has no counterpart here."""
        assert {member.value for member in Triage} == {"pass", "flag", "discard"}

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

    def test_nothing_routed_but_not_empty_flags_rather_than_discarding(self) -> None:
        """Content no gate could account for is a charge against the ruleset, never against the file."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_all_declined(),
            ran={},
            hint_claims={},
            route_state="unexplained",
        )
        assert folded.triage is Triage.FLAG
        assert folded.discard_ground is None
        assert any(reason.why == UNEXPLAINED_CONTENT for reason in folded.reasons)

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
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok"), NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled")],
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

    def test_a_hint_turns_the_empty_ground_into_a_flag(self) -> None:
        """Discarding would delete the evidence that the graph was wrong."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_all_declined(),
            ran={},
            hint_claims={"SPEECH": True},
            route_state="empty",
        )
        assert folded.triage is Triage.FLAG

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
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.PASS, "speech", "words in the store"),
            ],
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.findings["SPEECH"] == "present"
        assert folded.findings["AIRWAY"] == "uncertain"
        assert folded.findings["VOICE"] == "uncertain"

    def test_a_flagged_branch_still_resolves_its_subject(self) -> None:
        """The flag travels beside the resolution and is not a reason to withhold it."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("VOICE", Outcome.FLAG, "voice", "a declared range is not met"),
            ],
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=DECLINED, VOICE=ROUTED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.findings["VOICE"] == "present"
        assert folded.triage is Triage.FLAG

    def test_a_failed_branch_resolves_its_subject_absent(self) -> None:
        """A branch with no subject is authority for that too."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.FAIL, "speech", "no consensus word"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.findings["SPEECH"] == "absent"

    def test_a_failing_branch_does_not_carry_its_absence_onto_a_sibling(self) -> None:
        """SPEECH found no subject; that says nothing about the airway AIRWAY found."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.FAIL, "speech", "no consensus word"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
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
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.PASS, "speech", "words"),
            ],
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
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.PASS, "speech", "words"),
            ],
            branch_decisions=_decisions(["SPEECH"], AIRWAY=DECLINED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.agreement["SPEECH"] == "mismatch"
        assert folded.triage is Triage.FLAG

    def test_a_routed_branch_that_found_nothing_is_a_mismatch(self) -> None:
        """The other direction of the same row: the ruleset over-routed."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.FAIL, "speech", "no consensus word"),
            ],
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.agreement["SPEECH"] == "mismatch"

    def test_an_unreadable_route_is_resolved_not_mismatched(self) -> None:
        """A branch whose gates could not be read made no claim to disagree with."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.PASS, "speech", "words"),
            ],
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
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
                NodeVerdict("SPEECH", Outcome.FAIL, "speech", "no consensus word"),
            ],
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
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.FAIL, "airway", "no span carries a label"),
            ],
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
        assert folded.routes["VOICE"] == UNAVAILABLE
        assert folded.findings["VOICE"] == "uncertain"
        assert folded.triage is not Triage.DISCARD

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
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.agreement["SPEECH"] == "not_run"
        assert folded.triage is Triage.PASS

    def test_asked_but_silent_flags(self) -> None:
        """will_run true with no verdict is a branch that left no answer."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=ROUTED, VOICE=DECLINED),
            ran={"SPEECH": RunState.ERRORED},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.triage is Triage.FLAG
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
        """A routed branch that left no verdict flags the file, whatever silenced it.

        DDK is the name here because it was the only branch with no node when this was written. It
        has one now, so the state the fold is handed is the general one: the branch was asked to
        run and concluded nothing — skipped, errored, or completed without a verdict. The fold must
        say so rather than pass quietly, and that is what is pinned.
        """
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_decisions(AIRWAY=DECLINED, SPEECH=DECLINED, VOICE=DECLINED, DDK=ROUTED),
            ran={"DDK": RunState.SKIPPED},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.triage is Triage.FLAG
        assert any(reason.node == "DDK" and "never ran" in reason.why for reason in folded.reasons)

    def test_the_branches_map_joins_the_decision_to_the_verdict(self) -> None:
        """A skipped branch carries the reason it was skipped, beside a branch that concluded."""
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
        assert folded.branches["AIRWAY"]["verdict"] == "pass"
        assert folded.branches["SPEECH"]["verdict"] is None
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
        assert folded.triage is Triage.FLAG
        assert folded.hints["AIRWAY"] == "claimed_not_found"

    def test_a_hinted_branch_that_found_its_subject_is_an_agreement(self) -> None:
        """The declaration and the measurement said the same thing; nothing is owed a human."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
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
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
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
        assert folded.triage is Triage.FLAG

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

    def test_a_declaration_nothing_could_read_empties_the_hints_and_flags(self) -> None:
        """Unknown claims are not no claims: reporting them as ``no_claim`` would clear the file quietly."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions={},
            ran={},
            hint_claims=None,
            route_state=ROUTED,
        )
        assert folded.hints == {}
        assert folded.triage is Triage.FLAG
        assert any(reason.why == UNREAD_DECLARATION for reason in folded.reasons)


class TestAConfigTypoIsNamedNotSwallowed:
    """A map value that is not a branch under-claims every file in the run; it must not discard one."""

    def test_a_bad_map_value_flags_where_the_file_would_otherwise_discard(self) -> None:
        """One character in the map turns a declared cough into a silent discard of the evidence."""
        decisions = _all_declined()
        decisions["AIRWAY"] = replace(decisions["AIRWAY"], bad_map_values={"cough": "AIRWY"})
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=decisions,
            ran={},
            hint_claims={},
            route_state="empty",
        )
        assert folded.triage is Triage.FLAG
        assert folded.discard_ground is None
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


class TestTheReleaseAxis:
    """Only a REDACT pass clears an artifact, and not_assessed is not releasable."""

    def test_no_redact_verdict_is_not_assessed(self) -> None:
        """No speech branch, no words, or no PII found."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            branch_decisions=_decisions(AIRWAY=ROUTED, SPEECH=DECLINED, VOICE=DECLINED),
            ran={},
            hint_claims={},
            route_state=ROUTED,
        )
        assert folded.release is Release.NOT_ASSESSED

    def test_a_redact_flag_withholds(self) -> None:
        """Unresolved is not cleared."""
        folded = _with_redact(Outcome.FLAG)
        assert folded.release is Release.WITHHELD

    def test_a_redact_fail_withholds(self) -> None:
        """A finding survived verification."""
        assert _with_redact(Outcome.FAIL).release is Release.WITHHELD

    def test_a_redact_pass_is_releasable(self) -> None:
        """For its artifacts only; never for the store."""
        assert _with_redact(Outcome.PASS).release is Release.RELEASABLE


class TestARedactNonPassIsVisibleWithoutFlippingTriage:
    """Triage asks whether a human must look; release asks whether an artifact may be handed on."""

    def test_a_surviving_finding_does_not_move_triage(self) -> None:
        """A release problem is not a measurement problem."""
        folded = _with_redact(Outcome.FAIL, speech=Outcome.PASS, speech_route=ROUTED)
        assert folded.triage is Triage.PASS
        assert folded.release is Release.WITHHELD

    def test_it_appears_in_reasons_regardless(self) -> None:
        """A consumer filtering on triage == pass sees the release axis in the same record."""
        folded = _with_redact(Outcome.FAIL, speech=Outcome.PASS, speech_route=ROUTED)
        assert any(reason.node == "REDACT" for reason in folded.reasons)

    def test_an_incomplete_verification_still_flags(self) -> None:
        """REDACT's ``flag`` is a node flag like any other: verification that did not finish."""
        folded = _with_redact(Outcome.FLAG, speech=Outcome.PASS, speech_route=ROUTED)
        assert folded.triage is Triage.FLAG
        assert folded.release is Release.WITHHELD


class TestReasonsCarryEveryContribution:
    """A flag naming one cause hides the others."""

    def test_two_flagging_branches_both_appear(self) -> None:
        """Not only the first, and not only the deciding one."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.FLAG, "airway", "a labelled span is short"),
                NodeVerdict("VOICE", Outcome.FLAG, "voice", "a declared range is not met"),
            ],
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
