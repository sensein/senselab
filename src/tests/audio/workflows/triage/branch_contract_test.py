"""The owner decision, pinned: a branch reports, VERDICT decides, and a branch never refuses.

These are the contract-level tests, kept apart from each node's own file because each one pins a
property of the *boundary* rather than of a body: that no branch writes an outcome, that the fold
decides from what a branch reports and from nothing else, that the packaged state is sane, that a
deviation is recorded and not folded, that the REDACT interlock and the three operational states
came through untouched, and that DDK is folded asymmetrically for the reason the owner gave.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any, Sequence

import pytest

from senselab.audio.workflows.triage.config import (
    UnknownConfigKey,
    UnmeasuredConfigKey,
    load_triage_config,
)
from senselab.audio.workflows.triage.nodes import airway as airway_module
from senselab.audio.workflows.triage.nodes import ddk as ddk_module
from senselab.audio.workflows.triage.nodes import quality as quality_module
from senselab.audio.workflows.triage.nodes import speech as speech_module
from senselab.audio.workflows.triage.nodes import voice as voice_module
from senselab.audio.workflows.triage.nodes.branches import (
    PARAM_KEYS,
    PARAM_SECTION,
    POINT_TYPES,
    UNMEASURED_POINTS,
    branch_params,
)
from senselab.audio.workflows.triage.nodes.common import RESERVED_REPORT_KEYS, write_report
from senselab.audio.workflows.triage.vocabulary import (
    BRANCHES,
    DECLINED,
    ROUTED,
    STORE_ASSERTIONS,
    TASK,
    UNDETERMINED,
    BranchDecision,
    BranchReport,
    Conformance,
    FoldPolicy,
    NodeVerdict,
    Outcome,
    Release,
    Triage,
    fold_file_verdict,
)

_BRANCH_MODULES = (airway_module, speech_module, voice_module, ddk_module, quality_module)
_REPORTING_SOURCES = tuple(Path(module.__file__ or "") for module in _BRANCH_MODULES)


def _report(
    node: str,
    kind: str | None = "airway",
    *,
    conformance: Conformance = UNDETERMINED,
    referent: str = TASK,
    deviations: Sequence[str] = (),
    unmeasured: Sequence[str] = (),
    in_family: bool = True,
) -> BranchReport:
    """One reporting node's report, spelled out.

    Args:
        node: The node's name.
        kind: The kind it reports on.
        conformance: Whether what was asked for happened.
        referent: What that conformance is about.
        deviations: The deviation type names found.
        unmeasured: The config paths asked for and unmeasured.
        in_family: Whether it evaluated a declared task of its own kind.

    Returns:
        The report.
    """
    return BranchReport(
        node=node,
        kind=kind,
        conformance=conformance,
        conformance_of=referent,
        deviations=tuple(deviations),
        unmeasured=tuple(unmeasured),
        in_family=in_family,
    )


def _decisions(**routes: str) -> dict[str, BranchDecision]:
    """One routing decision per named branch.

    Args:
        **routes: Branch name to its route state.

    Returns:
        The decisions.
    """
    return {
        branch: BranchDecision(branch=branch, will_run=state == ROUTED, route_state=state, forced_by_declaration=False)
        for branch, state in routes.items()
    }


def _fold(
    reports: Sequence[BranchReport] = (),
    *,
    spans: dict[str, int] | None = None,
    routes: dict[str, str] | None = None,
    declared_family: str | None = None,
    policy: FoldPolicy | None = None,
    node_verdicts: Sequence[NodeVerdict] = (),
) -> Any:  # noqa: ANN401 — FileVerdict, kept off the signature for brevity
    """The fold over an admitted recording, with only what a test names.

    Args:
        reports: The reporting nodes' reports.
        spans: How many spans each proposed in its own family.
        routes: What the ruleset made of each branch.
        declared_family: The task family the recording declares.
        policy: The fold policy; None is the packaged one.
        node_verdicts: Extra deciding-node verdicts beside ADMIT's pass.

    Returns:
        The file verdict.
    """
    return fold_file_verdict(
        [NodeVerdict("ADMIT", Outcome.PASS, None, "ok"), *node_verdicts],
        branch_reports=reports,
        spans_by_node=spans or {},
        branch_decisions=_decisions(**(routes or {})),
        ran={},
        hint_claims={},
        route_state=ROUTED,
        declared_family=declared_family,
        policy=policy,
    )


class TestNoBranchWritesAnOutcome:
    """The decision the owner took, pinned at the writer rather than at a reader."""

    @pytest.mark.parametrize("source", _REPORTING_SOURCES, ids=lambda path: path.stem)
    def test_no_reporting_node_names_the_outcome_vocabulary(self, source: Path) -> None:
        """An AST sweep, so an ``Outcome`` reintroduced anywhere in a reporting node fails here.

        A grep would match a docstring; this matches a name the module actually loads or evaluates,
        which is what would let a branch decide again.
        """
        tree = ast.parse(source.read_text(encoding="utf-8"))
        named: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                named.update(alias.name for alias in node.names)
            elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                if node.value.id == "Outcome":
                    named.add("Outcome")
        assert "Outcome" not in named, f"{source.stem} names Outcome; a reporting node has no outcome"

    def test_the_report_writer_refuses_an_outcome_in_its_detail(self) -> None:
        """``outcome`` is reserved on a ``branch_report``, so it cannot arrive through ``detail``."""
        assert "outcome" in RESERVED_REPORT_KEYS

    def test_a_report_carries_conformance_and_never_an_outcome(self, tmp_path: Path) -> None:
        """The written entity's own attributes, read back."""
        from senselab.audio.workflows.triage.nodes.common import software_agent
        from senselab.utils.prov_store import ProvStore

        del tmp_path
        store = ProvStore(run_id="contract")
        agent = software_agent(store)
        activity = store.activity(node="AIRWAY", step="branch", parameters={})
        entity_id, report = write_report(
            store,
            activity,
            agent,
            node="AIRWAY",
            kind="airway",
            conformance=True,
            conformance_of=TASK,
            deviations=("off_task_extent",),
            detail={"labelled_n": 2},
        )
        attributes = store.get_entity(entity_id).attributes
        assert store.get_entity(entity_id).prov_type == "branch_report"
        assert "outcome" not in attributes
        assert attributes["conformance"] is True
        assert attributes["conformance_of"] == TASK
        assert report.deviations == ("off_task_extent",)

    def test_the_writer_refuses_an_unknown_conformance_referent(self) -> None:
        """A referent outside the two is a claim the fold cannot read; it is refused at the write."""
        from senselab.audio.workflows.triage.nodes.common import software_agent
        from senselab.utils.prov_store import ProvStore

        store = ProvStore(run_id="contract")
        agent = software_agent(store)
        activity = store.activity(node="AIRWAY", step="branch", parameters={})
        with pytest.raises(ValueError, match="conformance_of must be one of"):
            write_report(
                store,
                activity,
                agent,
                node="AIRWAY",
                kind="airway",
                conformance=True,
                conformance_of="vibes",
                deviations=(),
                detail={},
            )


class TestTheFoldDecidesFromWhatIsReported:
    """Conformance, deviations, spans, route, declared task — and nothing that was decided upstream."""

    def test_a_non_conformance_flags(self) -> None:
        """The one claim a reporting node makes that reaches the triage axis."""
        folded = _fold([_report("AIRWAY", conformance=False)], spans={"AIRWAY": 1}, routes={"AIRWAY": ROUTED})
        assert folded.triage is Triage.FLAG
        assert folded.conformance["AIRWAY"] is False

    def test_a_conformance_does_not_flag(self) -> None:
        """The instruction was met; there is nothing for a human to look at."""
        folded = _fold([_report("AIRWAY", conformance=True)], spans={"AIRWAY": 1}, routes={"AIRWAY": ROUTED})
        assert folded.triage is Triage.PASS

    def test_the_spans_are_what_say_a_branch_found_its_subject(self) -> None:
        """``findings`` is read off the spans, not off any conclusion of the branch's own."""
        present = _fold([_report("AIRWAY", conformance=True)], spans={"AIRWAY": 1}, routes={"AIRWAY": ROUTED})
        absent = _fold([_report("AIRWAY", conformance=True)], spans={}, routes={"AIRWAY": DECLINED})
        silent = _fold([], spans={}, routes={"AIRWAY": DECLINED})
        assert present.findings["AIRWAY"] == "present"
        assert absent.findings["AIRWAY"] == "absent"
        assert silent.findings["AIRWAY"] == "uncertain"

    def test_the_route_reaches_triage_only_through_a_mismatch(self) -> None:
        """Routed and found nothing is over-routing; declined and found it is a miss. Both flag."""
        over = _fold([_report("AIRWAY", conformance=True)], spans={}, routes={"AIRWAY": ROUTED})
        missed = _fold([_report("AIRWAY", conformance=True)], spans={"AIRWAY": 1}, routes={"AIRWAY": DECLINED})
        assert over.agreement["AIRWAY"] == "mismatch"
        assert missed.agreement["AIRWAY"] == "mismatch"
        assert over.triage is Triage.FLAG
        assert missed.triage is Triage.FLAG

    def test_the_fold_is_task_aware(self) -> None:
        """The same non-conformance flags on one declared family and is excepted on another."""
        policy = FoldPolicy(conformance_flags_by_family={"story-recall": False})
        flagged = _fold(
            [_report("SPEECH", "speech", conformance=False)],
            spans={"SPEECH": 1},
            routes={"SPEECH": ROUTED},
            declared_family="prolonged-vowel",
            policy=policy,
        )
        excepted = _fold(
            [_report("SPEECH", "speech", conformance=False)],
            spans={"SPEECH": 1},
            routes={"SPEECH": ROUTED},
            declared_family="story-recall",
            policy=policy,
        )
        assert flagged.triage is Triage.FLAG
        assert excepted.triage is Triage.PASS
        assert excepted.conformance["SPEECH"] is False
        assert excepted.declared_family == "story-recall"

    def test_the_reason_names_the_task_it_was_read_against(self) -> None:
        """A flag that does not say which instruction was not met is not actionable."""
        folded = _fold(
            [_report("VOICE", "voice", conformance=False)],
            spans={"VOICE": 1},
            routes={"VOICE": ROUTED},
            declared_family="prolonged-vowel",
        )
        assert any("prolonged-vowel" in reason.why for reason in folded.reasons)


class TestADeviationAloneDoesNotFlag:
    """The ground-truth constraint, which survives this change and bounds what the fold may do."""

    def test_a_reported_deviation_is_recorded_and_the_file_still_passes(self) -> None:
        """``filler`` and ``stimulus_mismatch`` are expected on ordinary read speech."""
        folded = _fold(
            [_report("SPEECH", "speech", conformance=True, deviations=("filler", "stimulus_mismatch"))],
            spans={"SPEECH": 1},
            routes={"SPEECH": ROUTED},
        )
        assert folded.triage is Triage.PASS
        assert folded.deviations["SPEECH"] == ["filler", "stimulus_mismatch"]

    def test_the_packaged_policy_does_not_fold_deviations(self) -> None:
        """The switch exists for when ground truth does, and ships off."""
        assert FoldPolicy.from_config(load_triage_config()).deviation_flags is False

    def test_turning_the_switch_on_is_what_would_flag_them(self) -> None:
        """The discriminating half: the deviation is genuinely the only thing in this fold."""
        folded = _fold(
            [_report("SPEECH", "speech", conformance=True, deviations=("filler",))],
            spans={"SPEECH": 1},
            routes={"SPEECH": ROUTED},
            policy=FoldPolicy(deviation_flags=True),
        )
        assert folded.triage is Triage.FLAG


class TestAllUndeterminedIsSane:
    """Every branch answering UNDETERMINED is a state the graph must handle, not an edge case."""

    def test_no_branch_answering_a_conformance_question_does_not_flag(self) -> None:
        """``detect_*`` evaluates no task, so UNDETERMINED is the answer for every out-of-family run."""
        folded = _fold(
            [_report(branch, "x", conformance=UNDETERMINED, in_family=False) for branch in BRANCHES],
            spans={branch: 1 for branch in BRANCHES},
            routes={branch: ROUTED for branch in BRANCHES},
        )
        assert folded.triage is Triage.PASS
        assert set(folded.conformance.values()) == {UNDETERMINED}

    def test_the_packaged_policy_does_not_flag_an_unanswered_conformance(self) -> None:
        """A True here would flag every recording no branch was in-family for."""
        assert FoldPolicy.from_config(load_triage_config()).undetermined_flags is False

    def test_turning_that_switch_on_is_what_would_flag_them(self) -> None:
        """The discriminating half."""
        folded = _fold(
            [_report("AIRWAY", conformance=UNDETERMINED)],
            spans={"AIRWAY": 1},
            routes={"AIRWAY": ROUTED},
            policy=FoldPolicy(undetermined_flags=True),
        )
        assert folded.triage is Triage.FLAG

    def test_an_all_undetermined_fold_still_reports_findings_and_agreement(self) -> None:
        """Answering no conformance question is not the same as measuring nothing."""
        folded = _fold(
            [_report("AIRWAY", conformance=UNDETERMINED)],
            spans={"AIRWAY": 2},
            routes={"AIRWAY": ROUTED},
        )
        assert folded.findings["AIRWAY"] == "present"
        assert folded.agreement["AIRWAY"] == "agree"


class TestDdkIsFoldedAsymmetrically:
    """Owner, 2026-09-16: for DDK, finding the subject *is* evaluating the task."""

    def test_an_out_of_family_ddk_result_is_a_detector_covariate_not_a_reading(self) -> None:
        """A train on a declared story-recall says more about the detector than about the audio."""
        folded = _fold(
            [_report("DDK", "ddk", conformance=False, in_family=False)],
            spans={"DDK": 1},
            routes={"DDK": ROUTED},
            declared_family="rainbow-passage",
        )
        assert folded.triage is Triage.PASS
        assert folded.detector_covariates["DDK"]["spans_n"] == 1
        assert folded.detector_covariates["DDK"]["conformance"] is False

    def test_an_out_of_family_ddk_mismatch_does_not_flag_either(self) -> None:
        """The over-routing is one gate's doing; a mismatch on it charges the ruleset, not the file."""
        folded = _fold(
            [_report("DDK", "ddk", conformance=UNDETERMINED, in_family=False)],
            spans={"DDK": 1},
            routes={"DDK": DECLINED},
            declared_family="free-speech",
        )
        assert folded.agreement["DDK"] == "mismatch"
        assert folded.triage is Triage.PASS
        assert "DDK" in folded.detector_covariates

    def test_in_family_ddk_folds_like_any_other_branch(self) -> None:
        """The asymmetry is about the mode, not about the branch being exempt."""
        folded = _fold(
            [_report("DDK", "ddk", conformance=False, in_family=True)],
            spans={"DDK": 1},
            routes={"DDK": ROUTED},
            declared_family="ddk-pataka",
        )
        assert folded.triage is Triage.FLAG
        assert "DDK" not in folded.detector_covariates

    def test_the_other_three_are_not_in_the_asymmetric_set(self) -> None:
        """Breath, phonation and lexical content all occur incidentally; a train does not."""
        policy = FoldPolicy.from_config(load_triage_config())
        assert policy.detection_is_evaluation == ("DDK",)
        for branch in ("AIRWAY", "SPEECH", "VOICE"):
            folded = _fold(
                [_report(branch, "x", conformance=False, in_family=False)],
                spans={branch: 1},
                routes={branch: ROUTED},
            )
            assert folded.triage is Triage.FLAG, branch


class TestQualityEntersThroughConformanceAboutTheStore:
    """QUALITY has no route and no declared task, so its conformance is about something else."""

    def test_a_contradicted_store_assertion_flags_whatever_the_task(self) -> None:
        """No task key governs it: a store contradicting itself is inconsistent on every recording."""
        policy = FoldPolicy(conformance_flags=False, conformance_flags_by_family={"anything": False})
        folded = _fold(
            [_report("QUALITY", None, conformance=False, referent=STORE_ASSERTIONS, in_family=False)],
            declared_family="anything",
            policy=policy,
        )
        assert folded.triage is Triage.FLAG
        assert folded.conformance_of["QUALITY"] == STORE_ASSERTIONS

    def test_quality_gets_no_route_no_finding_and_no_hint_row(self) -> None:
        """It is not in BRANCHES, so it has no subject to find and nothing to agree with."""
        folded = _fold([_report("QUALITY", None, conformance=True, referent=STORE_ASSERTIONS, in_family=False)])
        assert "QUALITY" not in folded.findings
        assert "QUALITY" not in folded.routes
        assert "QUALITY" not in folded.agreement
        assert folded.conformance["QUALITY"] is True

    def test_a_task_conformance_switched_off_still_leaves_quality_flagging(self) -> None:
        """The discriminating half: the referent, not the switch, is what carries QUALITY."""
        policy = FoldPolicy(conformance_flags=False)
        task = _fold([_report("AIRWAY", conformance=False)], spans={"AIRWAY": 1}, policy=policy)
        store_side = _fold(
            [_report("QUALITY", None, conformance=False, referent=STORE_ASSERTIONS, in_family=False)], policy=policy
        )
        assert task.triage is Triage.PASS
        assert store_side.triage is Triage.FLAG


class TestTheRedactInterlockIsUntouched:
    """``release`` depends on REDACT's outcome, and REDACT is a deciding node."""

    def test_release_still_comes_from_redacts_own_outcome(self) -> None:
        """REDACT keeps its ``Outcome``; the split reached the branches and not it."""
        releasable = _fold(node_verdicts=[NodeVerdict("REDACT", Outcome.PASS, None, "scanned")])
        withheld = _fold(node_verdicts=[NodeVerdict("REDACT", Outcome.FAIL, None, "a finding survived")])
        unassessed = _fold()
        assert releasable.release is Release.RELEASABLE
        assert withheld.release is Release.WITHHELD
        assert unassessed.release is Release.NOT_ASSESSED

    def test_no_branch_report_can_move_the_release_axis(self) -> None:
        """A branch has no say in it, which is what keeps the interlock a REDACT question."""
        folded = _fold(
            [_report("SPEECH", "speech", conformance=False)],
            spans={"SPEECH": 1},
            node_verdicts=[NodeVerdict("REDACT", Outcome.PASS, None, "scanned")],
        )
        assert folded.release is Release.RELEASABLE

    def test_redact_is_still_gated_on_pii_entities_not_on_speechs_report(self) -> None:
        """``run._speech_found_pii`` reads ``pii`` entities; nothing in this change touched it."""
        import inspect

        from senselab.audio.workflows.triage import run as run_module

        gate = inspect.getsource(run_module._drive_branches)
        assert '"SPEECH" in selected and _speech_found_pii(store)' in gate
        assert 'store.entities("pii")' in inspect.getsource(run_module._speech_found_pii)


class TestTheOperationalStatesAreUnchanged:
    """COMPLETED / ERRORED / SKIPPED are what the runner did, not what the graph concluded."""

    def test_there_are_three_and_they_are_the_runners_own(self) -> None:
        """The brief called them six; there are three, and none of them is a judgement."""
        from senselab.audio.workflows.triage.vocabulary import RunState

        assert {member.value for member in RunState} == {"completed", "skipped", "errored"}

    def test_a_node_that_reported_is_completed_exactly_as_one_that_decided(self) -> None:
        """The split must not push every branch into the errored column."""
        import inspect

        from senselab.audio.workflows.triage.nodes import verdict as verdict_module

        derived = inspect.getsource(verdict_module._derived_ran)
        assert "{v.node for v in verdicts} | {r.node for r in reports}" in derived

    def test_the_runner_still_records_the_three_states_at_the_same_sites(self) -> None:
        """``_attempt`` records two and ``_drive_branches`` the third; that is the whole mechanism."""
        import inspect

        from senselab.audio.workflows.triage import run as run_module
        from senselab.audio.workflows.triage.vocabulary import RunState

        attempt = inspect.getsource(run_module._attempt)
        assert "RunState.ERRORED" in attempt and "RunState.COMPLETED" in attempt
        assert "RunState.SKIPPED" in inspect.getsource(run_module._drive_branches)
        assert run_module.NodeOutcome(node="AIRWAY", state=RunState.SKIPPED).report is None


class TestABranchNeverRefuses:
    """A refusal is a decision, so an unmeasured operating point is a fact to report."""

    def test_require_tells_a_typo_from_an_unmeasured_key(self) -> None:
        """One ``ValueError`` for both is what would let a typo read as an unmeasured point."""
        config = load_triage_config()
        with pytest.raises(UnknownConfigKey):
            config.require("branch.smooting_window_s")
        with pytest.raises(UnmeasuredConfigKey):
            config.require("verdict.tilt_max_db_per_octave")
        assert issubclass(UnknownConfigKey, ValueError)
        assert issubclass(UnmeasuredConfigKey, ValueError)

    def test_an_unmeasured_point_returns_none_and_is_recorded_in_read_order(self, tmp_path: Path) -> None:
        """What was asked for first is what says where the evaluation stopped being possible."""
        override = tmp_path / "nulled.yaml"
        override.write_text("branch:\n  train_min_s: null\n  score_min: null\n")
        params = branch_params(load_triage_config(override))
        assert params.point("train_min_s") is None
        assert params.point("score_min") is None
        assert params.missing == ["train_min_s", "score_min"]
        finding = params.record()[0]
        assert finding.name == UNMEASURED_POINTS
        assert finding.evidence["value"] == ["train_min_s", "score_min"]

    def test_a_misspelled_point_still_raises_from_inside_a_branch(self) -> None:
        """The distinction the whole split exists for: this must not read as one more null."""
        params = branch_params(load_triage_config())
        with pytest.raises(KeyError, match="smooting_window_s"):
            params.point("smooting_window_s")

    def test_a_setting_outside_the_branch_section_is_read_the_same_way(self) -> None:
        """A branch reading another section's key must not refuse over it either."""
        params = branch_params(load_triage_config())
        assert params.setting("verdict.tilt_max_db_per_octave") is None
        assert params.missing == ["verdict.tilt_max_db_per_octave"]
        with pytest.raises(UnknownConfigKey):
            params.setting("verdict.not_a_key")

    def test_an_unanswerable_conformance_is_undetermined_and_never_false(self) -> None:
        """The fix the migration surfaced, pinned where the distinction matters.

        A qualifier whose own boundary is unmeasured could neither admit nor reject, so an empty
        result has two causes: the recording carried nothing, or the configuration could not say. A
        ``False`` on the second reads as "the instruction was not met" and reaches the flag column
        indistinguishably from a genuine non-conformance, on the strength of a number nobody chose.
        ``qualifying_phonation``, ``ddk_carrier`` and the airway label search all returned the empty
        set either way, and all three now separate the two.
        """
        import inspect

        for module, marker in (
            (voice_module, "unmeasured_gate"),
            (ddk_module, "unmeasured_gate"),
            (airway_module, "_events_reading"),
        ):
            source = inspect.getsource(module)
            assert marker in source, f"{module.__name__} no longer separates the two causes"

    def test_an_unmeasured_point_flags_through_the_fold_rather_than_through_a_raise(self) -> None:
        """Where the refusal went: the branch names the key, and this fold decides about it."""
        folded = _fold([_report("AIRWAY", conformance=UNDETERMINED, unmeasured=["score_min"])], spans={})
        assert folded.unmeasured["AIRWAY"] == ["score_min"]
        assert folded.triage is Triage.FLAG
        quiet = _fold(
            [_report("AIRWAY", conformance=UNDETERMINED, unmeasured=["score_min"])],
            spans={},
            policy=FoldPolicy(unmeasured_points_flag=False),
        )
        assert quiet.triage is Triage.PASS


class TestTheBranchSectionShipsValues:
    """The owner reversed the all-null state; the branches run on the packaged config."""

    def test_every_branch_key_ships_a_value(self) -> None:
        """A key nobody could reason a default for was removed, not defaulted to a guess."""
        values = load_triage_config().values[PARAM_SECTION]
        unset = sorted(key for key, value in values.items() if value is None)
        assert unset == [], f"{unset} ship null; either reason a default or remove the key"

    def test_the_three_declarations_agree(self) -> None:
        """``POINT_TYPES``, ``PARAM_KEYS`` and the packaged section are one vocabulary in three places."""
        packaged = set(load_triage_config().values[PARAM_SECTION])
        assert set(POINT_TYPES) == set(PARAM_KEYS) == packaged

    def test_the_removed_key_is_gone_from_all_three(self) -> None:
        """``omission_score_max`` cut an acoustic score no derivative in the graph produces."""
        assert "omission_score_max" not in POINT_TYPES
        assert "omission_score_max" not in PARAM_KEYS
        assert "omission_score_max" not in load_triage_config().values[PARAM_SECTION]

    def test_the_decision_keys_moved_to_the_verdict_section(self) -> None:
        """A threshold that judges the recording is this fold's, not a detector's."""
        values = load_triage_config().values
        for key in ("min_contrast_db", "tilt_max_db_per_octave", "level_min_dbfs"):
            assert key not in values[PARAM_SECTION], f"branch.{key} judges the recording"
            assert key in values["verdict"], f"verdict.{key} is where it belongs"


class TestTheReportSurvivesTheStore:
    """Every other test here builds a ``BranchReport``; the pipeline reads one back off an entity.

    A field the writer stores and the reader drops is invisible to a constructed report, so the two
    fields the fold reads that no other test round-trips are pinned here.
    """

    @staticmethod
    def _round_trip(**written: Any) -> BranchReport:  # noqa: ANN401 — the writer's own kwargs
        """One report through ``write_report`` and back out through VERDICT's own reader."""
        from senselab.audio.workflows.triage.nodes.common import software_agent
        from senselab.audio.workflows.triage.nodes.verdict import _branch_reports
        from senselab.utils.prov_store import ProvStore

        store = ProvStore(run_id="round-trip")
        agent = software_agent(store)
        activity = store.activity(node=written["node"], step="branch", parameters={})
        write_report(store, activity, agent, detail={}, **written)
        pairs = _branch_reports(store)
        assert len(pairs) == 1
        return pairs[0][1]

    def test_unmeasured_reaches_the_fold(self) -> None:
        """Every branch passes ``params.missing``; dropping it here silences the flag entirely."""
        report = self._round_trip(
            node="SPEECH",
            kind="speech",
            conformance=UNDETERMINED,
            conformance_of=TASK,
            deviations=(),
            unmeasured=("branch.target_match_cosine",),
        )
        assert report.unmeasured == ("branch.target_match_cosine",)
        folded = _fold([report], spans={"SPEECH": 0}, routes={"SPEECH": ROUTED})
        assert folded.triage is Triage.FLAG
        assert folded.unmeasured["SPEECH"] == ["branch.target_match_cosine"]

    def test_in_family_reaches_the_fold(self) -> None:
        """The DDK asymmetry reads this off the entity, so the store is where it has to survive."""
        report = self._round_trip(
            node="DDK",
            kind="ddk",
            conformance=False,
            conformance_of=TASK,
            deviations=(),
            in_family=False,
        )
        assert report.in_family is False
        folded = _fold(
            [report],
            spans={"DDK": 1},
            routes={"DDK": ROUTED},
            declared_family="rainbow-passage",
        )
        assert "DDK" in folded.detector_covariates
        assert folded.triage is Triage.PASS

    def test_in_family_is_not_hardcoded_at_the_writer(self) -> None:
        """The align mode has to survive too, or the asymmetry exempts every DDK run."""
        report = self._round_trip(
            node="DDK",
            kind="ddk",
            conformance=False,
            conformance_of=TASK,
            deviations=(),
            in_family=True,
        )
        assert report.in_family is True
        folded = _fold(
            [report],
            spans={"DDK": 1},
            routes={"DDK": ROUTED},
            declared_family="ddk-pataka",
        )
        assert "DDK" not in folded.detector_covariates
        assert folded.triage is Triage.FLAG
