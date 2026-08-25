"""The file-level fold: pass, flag, discard, and each branch authority over its own kind."""

from __future__ import annotations

from senselab.audio.workflows.triage.vocabulary import (
    BranchDecision,
    FileVerdict,
    NodeVerdict,
    Outcome,
    Release,
    RunState,
    Triage,
    fold_file_verdict,
)

_BRANCH_FOR_KIND = {"airway": "AIRWAY", "speech": "SPEECH", "voice": "VOICE"}


def _decisions(*, airway: bool, speech: bool, voice: bool) -> dict[str, BranchDecision]:
    """One decision per branch, as ROUTING writes them: a branch runs unless its kind read absent.

    Args:
        airway: Whether the AIRWAY branch was selected.
        speech: Whether the SPEECH branch was selected.
        voice: Whether the VOICE branch was selected.

    Returns:
        The decisions, keyed by branch name.
    """
    selected = {"airway": airway, "speech": speech, "voice": voice}
    return {
        _BRANCH_FOR_KIND[kind]: BranchDecision(
            branch=_BRANCH_FOR_KIND[kind],
            kind=kind,
            will_run=will_run,
            kind_state="uncertain" if will_run else "absent",
            forced_by_hint=False,
        )
        for kind, will_run in selected.items()
    }


def _all_skipped() -> dict[str, BranchDecision]:
    """The empty execution set: every branch declined, none forced.

    Returns:
        The three decisions, all ``will_run: False``.
    """
    return _decisions(airway=False, speech=False, voice=False)


def _with_redact(redact: Outcome, *, speech: Outcome | None = None, screened_speech: str = "absent") -> FileVerdict:
    """A fold whose only interesting node is REDACT, optionally with a SPEECH branch beside it.

    Args:
        redact: What REDACT concluded.
        speech: What SPEECH concluded, or None when the branch never ran.
        screened_speech: What TAXONOMY classified for ``speech``.

    Returns:
        The folded file verdict.
    """
    node_verdicts = [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")]
    if speech is not None:
        node_verdicts.append(NodeVerdict("SPEECH", speech, "speech", "words in the store"))
    node_verdicts.append(NodeVerdict("REDACT", redact, None, "the scan concluded"))
    return fold_file_verdict(
        node_verdicts,
        screened={"speech": screened_speech, "airway": "absent", "voice": "absent"},
        branch_decisions=_decisions(airway=False, speech=speech is not None, voice=False),
        ran={},
        hint_claims={},
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
            screened={},
            branch_decisions={},
            ran={},
            hint_claims={},
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "unmeasurable"

    def test_all_absent_with_nothing_found_discards_as_acoustically_empty(self) -> None:
        """Measured, and there is nothing of interest in it."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            screened={"speech": "absent", "airway": "absent", "voice": "absent"},
            branch_decisions=_all_skipped(),
            ran={},
            hint_claims={},
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "acoustically_empty"

    def test_the_two_grounds_are_told_apart_by_their_ground_not_by_their_axis(self) -> None:
        """Both discard; a consumer that cannot tell them apart treats an empty file as a broken one."""
        broken = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.FAIL, None, "decode failure")],
            screened={},
            branch_decisions={},
            ran={},
            hint_claims={},
        )
        empty = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            screened={"speech": "absent", "airway": "absent", "voice": "absent"},
            branch_decisions=_all_skipped(),
            ran={},
            hint_claims={},
        )
        assert broken.triage is empty.triage is Triage.DISCARD
        assert broken.discard_ground != empty.discard_ground

    def test_a_pass_carries_no_ground(self) -> None:
        """``discard_ground`` describes a discard and nothing else."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok"), NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled")],
            screened={"speech": "absent", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=False, voice=False),
            ran={},
            hint_claims={},
        )
        assert folded.triage is Triage.PASS
        assert folded.discard_ground is None

    def test_no_kinds_at_all_is_not_every_kind_absent(self) -> None:
        """A run that screened nothing has not measured emptiness; it has measured nothing."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            screened={},
            branch_decisions={},
            ran={},
            hint_claims={},
        )
        assert folded.triage is Triage.PASS

    def test_a_branch_fail_is_not_a_discard(self) -> None:
        """A cough recording has no speech; SPEECH failing is the expected outcome."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
                NodeVerdict("SPEECH", Outcome.FAIL, "speech", "no consensus word"),
            ],
            screened={"speech": "absent", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=True, voice=False),
            ran={},
            hint_claims={},
        )
        assert folded.triage is not Triage.DISCARD

    def test_a_hint_turns_the_empty_ground_into_a_flag(self) -> None:
        """Discarding would delete the evidence that the graph was wrong."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            screened={"speech": "absent", "airway": "absent", "voice": "absent"},
            branch_decisions=_all_skipped(),
            ran={},
            hint_claims={"speech": True},
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
                NodeVerdict("routing", Outcome.PASS, None, "no branch runs; airway absent, speech absent"),
            ],
            screened={"speech": "absent", "airway": "absent", "voice": "absent"},
            branch_decisions=_all_skipped(),
            ran={},
            hint_claims={},
        )
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "acoustically_empty"


class TestBranchAuthorityIsScoped:
    """A branch is the authority on its own kind and on nothing else."""

    def test_speech_resolves_speech_and_touches_nothing_else(self) -> None:
        """It refutes neither airway nor voice."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.PASS, "speech", "words in the store"),
            ],
            screened={"speech": "uncertain", "airway": "present", "voice": "present"},
            branch_decisions=_decisions(airway=False, speech=True, voice=False),
            ran={},
            hint_claims={},
        )
        assert folded.kinds["speech"] == "present"
        assert folded.kinds["airway"] == "present"
        assert folded.kinds["voice"] == "present"

    def test_a_flagged_branch_still_resolves_its_kind(self) -> None:
        """The flag travels beside the resolution and is not a reason to withhold it."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("VOICE", Outcome.FLAG, "voice", "a declared range is not met"),
            ],
            screened={"speech": "absent", "airway": "absent", "voice": "uncertain"},
            branch_decisions=_decisions(airway=False, speech=False, voice=True),
            ran={},
            hint_claims={},
        )
        assert folded.kinds["voice"] == "present"
        assert folded.triage is Triage.FLAG

    def test_a_failed_branch_resolves_its_kind_absent(self) -> None:
        """A branch with no subject is authority for that too."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.FAIL, "speech", "no consensus word"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
            screened={"speech": "present", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=True, voice=False),
            ran={},
            hint_claims={},
        )
        assert folded.kinds["speech"] == "absent"


class TestTaxonomyIsReportedBeside:
    """Both maps are always present, and agreement is checkable by a reader."""

    def test_screened_and_kinds_are_both_present(self) -> None:
        """Keeping both is what makes agreement checkable rather than asserted."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.PASS, "speech", "words"),
            ],
            screened={"speech": "absent", "airway": "absent", "voice": "absent"},
            branch_decisions=_decisions(airway=False, speech=True, voice=False),
            ran={},
            hint_claims={},
        )
        assert folded.screened["speech"] == "absent"
        assert folded.kinds["speech"] == "present"

    def test_absent_classified_but_found_is_a_mismatch_and_flags(self) -> None:
        """It flags; it never overrides, and both stay in the product."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.PASS, "speech", "words"),
            ],
            screened={"speech": "absent", "airway": "absent", "voice": "absent"},
            branch_decisions=_decisions(airway=False, speech=True, voice=False),
            ran={},
            hint_claims={},
        )
        assert folded.agreement["speech"] == "mismatch"
        assert folded.triage is Triage.FLAG

    def test_present_classified_but_not_found_is_a_mismatch(self) -> None:
        """The other direction of the same row."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.FAIL, "speech", "no consensus word"),
            ],
            screened={"speech": "present", "airway": "absent", "voice": "absent"},
            branch_decisions=_decisions(airway=False, speech=True, voice=False),
            ran={},
            hint_claims={},
        )
        assert folded.agreement["speech"] == "mismatch"

    def test_uncertain_classified_is_resolved_not_mismatched(self) -> None:
        """A branch settling an unsettled kind is the design working, not a disagreement."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.PASS, "speech", "words"),
            ],
            screened={"speech": "uncertain", "airway": "absent", "voice": "absent"},
            branch_decisions=_decisions(airway=False, speech=True, voice=False),
            ran={},
            hint_claims={},
        )
        assert folded.agreement["speech"] == "resolved"
        assert folded.triage is Triage.PASS

    def test_agreeing_kinds_are_recorded_as_agreeing(self) -> None:
        """``agree`` is a value a reader can see, not the absence of a mismatch."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
                NodeVerdict("SPEECH", Outcome.FAIL, "speech", "no consensus word"),
            ],
            screened={"speech": "absent", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=True, voice=False),
            ran={},
            hint_claims={},
        )
        assert folded.agreement["airway"] == "agree"
        assert folded.agreement["speech"] == "agree"
        assert folded.triage is Triage.PASS

    def test_the_classification_is_never_rewritten_by_the_branch(self) -> None:
        """``screened`` reports what TAXONOMY said even where the branch overruled it on ``kinds``."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.FAIL, "airway", "no span carries a label"),
            ],
            screened={"speech": "absent", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=False, voice=False),
            ran={},
            hint_claims={},
        )
        assert folded.screened["airway"] == "present"
        assert folded.kinds["airway"] == "absent"

    def test_a_kind_with_neither_a_classification_nor_a_decision_reads_uncertain(self) -> None:
        """Nothing said anything about ``voice``; reading that as absent would invent a measurement.

        It is the difference between a discard and a pass on a file whose other kinds were found.
        """
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.FAIL, "speech", "no consensus word"),
            ],
            screened={},
            branch_decisions={},
            ran={},
            hint_claims={"voice": True},
        )
        assert folded.screened["voice"] == "uncertain"
        assert folded.kinds["voice"] == "uncertain"
        assert folded.triage is not Triage.DISCARD

    def test_a_kind_taxonomy_never_classified_reads_uncertain(self) -> None:
        """No classification is not a classification of absent; it is the unsettled state."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
            screened={},
            branch_decisions=_decisions(airway=True, speech=True, voice=True),
            ran={"SPEECH": RunState.COMPLETED, "VOICE": RunState.COMPLETED},
            hint_claims={},
        )
        assert folded.screened["speech"] == "uncertain"
        assert folded.kinds["speech"] == "uncertain"


class TestABranchThatNeverRanIsNotOneThatFailed:
    """The branch_decision elements are what distinguish the two."""

    def test_declined_and_unforced_is_expected(self) -> None:
        """The graph declined to look, and said why."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
            screened={"speech": "absent", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=False, voice=False),
            ran={},
            hint_claims={},
        )
        assert folded.agreement["speech"] == "not_run"
        assert folded.triage is Triage.PASS

    def test_asked_but_silent_flags(self) -> None:
        """will_run true with no verdict is a branch that left no answer."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            screened={"speech": "present", "airway": "absent", "voice": "absent"},
            branch_decisions=_decisions(airway=False, speech=True, voice=False),
            ran={"SPEECH": RunState.ERRORED},
            hint_claims={},
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
                screened={"speech": "present", "airway": "absent", "voice": "absent"},
                branch_decisions=_decisions(airway=False, speech=True, voice=False),
                ran={"SPEECH": state},
                hint_claims={},
            )
            assert any(phrase in reason.why for reason in folded.reasons)

    def test_the_silent_reason_names_the_branch(self) -> None:
        """A reason a reader cannot attribute to a branch is one they cannot act on."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            screened={"speech": "present", "airway": "absent", "voice": "absent"},
            branch_decisions=_decisions(airway=False, speech=True, voice=False),
            ran={},
            hint_claims={},
        )
        assert any(reason.node == "SPEECH" and reason.kind == "speech" for reason in folded.reasons)

    def test_the_branches_map_joins_the_decision_to_the_verdict(self) -> None:
        """A skipped branch carries the reason it was skipped, beside a branch that concluded."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
            screened={"speech": "absent", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=False, voice=False),
            ran={},
            hint_claims={},
        )
        assert folded.branches["AIRWAY"]["verdict"] == "pass"
        assert folded.branches["SPEECH"]["verdict"] is None
        assert folded.branches["SPEECH"]["will_run"] is False
        assert folded.branches["SPEECH"]["kind_state"] == "absent"


class TestHintsForMismatchOnly:
    """A hint names a mismatch and prevents a discard. It has no other power on this axis."""

    def test_a_hinted_kind_the_branch_did_not_find_flags(self) -> None:
        """The kind, the hint that claimed it, and the branch's conclusion, all named."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.FAIL, "airway", "no span carries a label"),
            ],
            screened={"speech": "absent", "airway": "absent", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=False, voice=False),
            ran={},
            hint_claims={"airway": True},
        )
        assert folded.triage is Triage.FLAG
        assert folded.hints["airway"] == "claimed_not_found"

    def test_a_hinted_kind_the_branch_found_is_an_agreement(self) -> None:
        """The declaration and the measurement said the same thing; nothing is owed a human."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
            screened={"speech": "absent", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=False, voice=False),
            ran={},
            hint_claims={"airway": True},
        )
        assert folded.hints["airway"] == "claimed_and_found"
        assert folded.triage is Triage.PASS

    def test_a_kind_found_that_no_hint_claimed_is_recorded_not_flagged(self) -> None:
        """Recorded; not a flag on its own."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
            screened={"speech": "absent", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=False, voice=False),
            ran={},
            hint_claims={},
        )
        assert folded.hints["airway"] == "found_unclaimed"
        assert folded.triage is Triage.PASS

    def test_an_unclaimed_kind_nobody_found_carries_no_claim(self) -> None:
        """The fourth cell of the table, so a reader never has to infer it from a missing key."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("AIRWAY", Outcome.PASS, "airway", "labelled"),
            ],
            screened={"speech": "absent", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=False, voice=False),
            ran={},
            hint_claims={},
        )
        assert folded.hints["speech"] == "no_claim"

    def test_a_hint_never_turns_a_flag_into_a_pass(self) -> None:
        """Its one power is to prevent a discard and to name a mismatch."""
        folded = fold_file_verdict(
            [
                NodeVerdict("ADMIT", Outcome.PASS, None, "ok"),
                NodeVerdict("SPEECH", Outcome.FLAG, "speech", "pii in the target's speech"),
            ],
            screened={"speech": "present", "airway": "absent", "voice": "absent"},
            branch_decisions=_decisions(airway=False, speech=True, voice=False),
            ran={},
            hint_claims={"speech": True},
        )
        assert folded.triage is Triage.FLAG

    def test_a_hint_never_resolves_a_kind(self) -> None:
        """A claim is an expectation; only a branch resolves, and here none ran."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            screened={"speech": "absent", "airway": "absent", "voice": "absent"},
            branch_decisions=_all_skipped(),
            ran={},
            hint_claims={"speech": True},
        )
        assert folded.kinds["speech"] == "absent"


class TestTheReleaseAxis:
    """Only a REDACT pass clears an artifact, and not_assessed is not releasable."""

    def test_no_redact_verdict_is_not_assessed(self) -> None:
        """No speech branch, no words, or no PII found."""
        folded = fold_file_verdict(
            [NodeVerdict("ADMIT", Outcome.PASS, None, "ok")],
            screened={"speech": "absent", "airway": "present", "voice": "absent"},
            branch_decisions=_decisions(airway=True, speech=False, voice=False),
            ran={},
            hint_claims={},
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
        folded = _with_redact(Outcome.FAIL, speech=Outcome.PASS, screened_speech="present")
        assert folded.triage is Triage.PASS
        assert folded.release is Release.WITHHELD

    def test_it_appears_in_reasons_regardless(self) -> None:
        """A consumer filtering on triage == pass sees the release axis in the same record."""
        folded = _with_redact(Outcome.FAIL, speech=Outcome.PASS, screened_speech="present")
        assert any(reason.node == "REDACT" for reason in folded.reasons)

    def test_an_incomplete_verification_still_flags(self) -> None:
        """REDACT's ``flag`` is a node flag like any other: verification that did not finish."""
        folded = _with_redact(Outcome.FLAG, speech=Outcome.PASS, screened_speech="present")
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
            screened={"speech": "absent", "airway": "present", "voice": "present"},
            branch_decisions=_decisions(airway=True, speech=False, voice=True),
            ran={},
            hint_claims={},
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
            screened={"airway": "present"},
            branch_decisions=_decisions(airway=True, speech=False, voice=False),
            ran={},
            hint_claims={},
        )
        assert folded.triage is Triage.DISCARD
        assert folded.reasons[0].node == "ADMIT"
        assert {"PREPROCESS", "AIRWAY"} <= {reason.node for reason in folded.reasons}
