"""The file-level fold: a branch fail is not a file fail."""

from __future__ import annotations

from senselab.audio.workflows.triage.vocabulary import (
    KindState,
    NodeVerdict,
    Outcome,
    RunState,
    fold_file_verdict,
)


def _v(node: str, outcome: Outcome, kind: str | None = None) -> NodeVerdict:
    """Build a NodeVerdict with a fixed why."""
    return NodeVerdict(node=node, outcome=outcome, kind=kind, why="test")


class TestBranchFailIsNotFileFail:
    """A branch with no subject fails without failing the file."""

    def test_a_branch_failing_on_an_absent_kind_is_expected(self) -> None:
        """SPEECH failing where TAXONOMY predicted no speech leaves the file a pass."""
        out = fold_file_verdict(
            node_verdicts=[
                _v("ADMIT", Outcome.PASS),
                _v("AIRWAY", Outcome.PASS, "airway"),
                _v("SPEECH", Outcome.FAIL, "speech"),
            ],
            kind_predictions={"airway": KindState.PRESENT, "speech": KindState.ABSENT},
            ran={"AIRWAY": RunState.COMPLETED, "SPEECH": RunState.COMPLETED},
        )
        assert out.triage is Outcome.PASS


class TestContradictions:
    """A branch outcome disagreeing with TAXONOMY's prediction is a flag."""

    def test_present_kind_with_a_failing_branch_flags(self) -> None:
        """A kind predicted present whose branch found no subject flags the file."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS), _v("SPEECH", Outcome.FAIL, "speech")],
            kind_predictions={"speech": KindState.PRESENT},
            ran={"SPEECH": RunState.COMPLETED},
        )
        assert out.triage is Outcome.FLAG
        assert any("contradiction" in r.why for r in out.reasons)

    def test_absent_kind_with_a_passing_branch_flags_and_resolves_the_kind(self) -> None:
        """A kind predicted absent whose branch passed flags, and the kind resolves present."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS), _v("AIRWAY", Outcome.PASS, "airway")],
            kind_predictions={"airway": KindState.ABSENT},
            ran={"AIRWAY": RunState.COMPLETED},
        )
        assert out.triage is Outcome.FLAG
        assert out.kinds["airway"] is KindState.PRESENT


class TestNeverRan:
    """A branch that never ran is read against what its kind predicted."""

    def test_a_skipped_branch_on_a_present_kind_flags(self) -> None:
        """Skipping the branch of a kind predicted present flags the file."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS)],
            kind_predictions={"speech": KindState.PRESENT},
            ran={"SPEECH": RunState.SKIPPED},
        )
        assert out.triage is Outcome.FLAG

    def test_a_skipped_branch_on_an_absent_kind_is_expected(self) -> None:
        """Skipping the branch of a kind predicted absent is the graph working as designed."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS), _v("AIRWAY", Outcome.PASS, "airway")],
            kind_predictions={"airway": KindState.PRESENT, "speech": KindState.ABSENT},
            ran={"AIRWAY": RunState.COMPLETED, "SPEECH": RunState.SKIPPED},
        )
        assert out.triage is Outcome.PASS


class TestOrdering:
    """The distinct fail cases stay distinct, and no reason is dropped."""

    def test_admit_failing_wins_over_everything(self) -> None:
        """ADMIT failing is the file verdict regardless of what the branches said."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.FAIL), _v("AIRWAY", Outcome.FLAG, "airway")],
            kind_predictions={},
            ran={"ADMIT": RunState.COMPLETED},
        )
        assert out.triage is Outcome.FAIL
        assert out.reasons[0].node == "ADMIT"

    def test_every_kind_absent_is_a_different_fail_from_admit(self) -> None:
        """No branch having a subject fails the file with a reason that is not ADMIT's."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS)],
            kind_predictions={"airway": KindState.ABSENT, "speech": KindState.ABSENT},
            ran={},
        )
        assert out.triage is Outcome.FAIL
        assert out.reasons[-1].node != "ADMIT"

    def test_reasons_carry_every_contribution_not_only_the_deciding_one(self) -> None:
        """Two flagging branches both appear in the reasons, not just the first."""
        out = fold_file_verdict(
            node_verdicts=[
                _v("ADMIT", Outcome.PASS),
                _v("AIRWAY", Outcome.FLAG, "airway"),
                _v("VOICE", Outcome.FLAG, "voice_no_words"),
            ],
            kind_predictions={"airway": KindState.PRESENT, "voice_no_words": KindState.PRESENT},
            ran={"AIRWAY": RunState.COMPLETED, "VOICE": RunState.COMPLETED},
        )
        assert out.triage is Outcome.FLAG
        assert len([r for r in out.reasons if r.outcome is Outcome.FLAG]) == 2


class TestSilentFallThrough:
    """A kind the graph was asked about must never end up with no answer and no reason."""

    def test_a_node_absent_from_ran_on_a_present_kind_flags(self) -> None:
        """A kind predicted present whose node the graph never even recorded trying flags the file."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS)],
            kind_predictions={"speech": KindState.PRESENT},
            ran={},
        )
        assert out.triage is Outcome.FLAG
        assert any(r.kind == "speech" and "never ran" in r.why for r in out.reasons)

    def test_a_completed_node_that_wrote_no_verdict_flags_with_its_own_reason(self) -> None:
        """A node that ran to completion and said nothing flags, and the why is not the never-ran one."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS)],
            kind_predictions={"speech": KindState.PRESENT},
            ran={"SPEECH": RunState.COMPLETED},
        )
        assert out.triage is Outcome.FLAG
        silent = [r for r in out.reasons if r.kind == "speech" and r.outcome is Outcome.FLAG]
        assert silent
        assert all("never ran" not in r.why for r in silent)


class TestUndecided:
    """An undecided kind resolves only on evidence, and its branch is still owed an answer."""

    def test_an_undecided_kind_with_a_passing_branch_resolves_present(self) -> None:
        """A branch pass on an undecided kind resolves it to present without flagging."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS), _v("SPEECH", Outcome.PASS, "speech")],
            kind_predictions={"speech": KindState.UNDECIDED},
            ran={"SPEECH": RunState.COMPLETED},
        )
        assert out.triage is Outcome.PASS
        assert out.kinds["speech"] is KindState.PRESENT

    def test_an_undecided_kind_with_a_failing_branch_resolves_absent(self) -> None:
        """A branch fail on an undecided kind resolves it to absent without flagging."""
        out = fold_file_verdict(
            node_verdicts=[
                _v("ADMIT", Outcome.PASS),
                _v("AIRWAY", Outcome.PASS, "airway"),
                _v("SPEECH", Outcome.FAIL, "speech"),
            ],
            kind_predictions={"airway": KindState.PRESENT, "speech": KindState.UNDECIDED},
            ran={"AIRWAY": RunState.COMPLETED, "SPEECH": RunState.COMPLETED},
        )
        assert out.triage is Outcome.PASS
        assert out.kinds["speech"] is KindState.ABSENT

    def test_an_undecided_kind_whose_branch_never_ran_flags_and_stays_undecided(self) -> None:
        """An undecided kind with no branch answer at all flags and does not resolve."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS)],
            kind_predictions={"speech": KindState.UNDECIDED},
            ran={},
        )
        assert out.triage is Outcome.FLAG
        assert out.kinds["speech"] is KindState.UNDECIDED

    def test_an_undecided_kind_with_a_flagging_branch_stays_undecided(self) -> None:
        """A branch flag is neither pass nor fail, so the kind must not resolve to absent."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.PASS), _v("SPEECH", Outcome.FLAG, "speech")],
            kind_predictions={"speech": KindState.UNDECIDED},
            ran={"SPEECH": RunState.COMPLETED},
        )
        assert out.triage is Outcome.FLAG
        assert out.kinds["speech"] is KindState.UNDECIDED


class TestAdmitFailKeepsEveryReason:
    """An ADMIT fail decides the verdict without erasing what else the graph found."""

    def test_a_co_occurring_verdict_survives_an_admit_fail(self) -> None:
        """The ADMIT verdict leads the reasons, and an AIRWAY flag beside it is not dropped."""
        out = fold_file_verdict(
            node_verdicts=[_v("ADMIT", Outcome.FAIL), _v("AIRWAY", Outcome.FLAG, "airway")],
            kind_predictions={"airway": KindState.PRESENT},
            ran={"ADMIT": RunState.COMPLETED, "AIRWAY": RunState.COMPLETED},
        )
        assert out.triage is Outcome.FAIL
        assert out.reasons[0].node == "ADMIT"
        assert any(r.node == "AIRWAY" and r.outcome is Outcome.FLAG for r in out.reasons)
