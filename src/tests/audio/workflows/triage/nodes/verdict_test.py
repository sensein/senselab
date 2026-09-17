"""VERDICT reads the store into the fold and records the result. Nothing here loads a model."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import pytest

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import routing as routing_module
from senselab.audio.workflows.triage.nodes import verdict as verdict_module
from senselab.audio.workflows.triage.nodes.branches import BRANCH_FAMILY
from senselab.audio.workflows.triage.nodes.common import software_agent, write_report, write_verdict
from senselab.audio.workflows.triage.nodes.routing import routing
from senselab.audio.workflows.triage.routing_analysis.ruleset import GateOutcome, RouteEvaluation, RouteState
from senselab.audio.workflows.triage.run import GRAPH_ORDER
from senselab.audio.workflows.triage.vocabulary import (
    LLM_REDACTION_RESIDUE,
    REDACTION_LLM_ANNOTATION,
    TASK,
    UNDETERMINED,
    UNREAD_DECLARATION,
    Conformance,
    FoldPolicy,
    Outcome,
    Release,
    RunState,
    Triage,
)
from senselab.utils.prov_store import Entity, ProvStore

BASE: tuple[tuple[str, Outcome, str | None], ...] = (
    ("ADMIT", Outcome.PASS, None),
    ("TAXONOMY", Outcome.PASS, None),
    ("AIRWAY", Outcome.PASS, "airway"),
    ("SPEECH", Outcome.PASS, "speech"),
)
ROUTED_PAIR = ("AIRWAY", "SPEECH")

BRANCH_READING: dict[Outcome, tuple[Conformance, int]] = {
    Outcome.PASS: (True, 1),
    Outcome.FLAG: (False, 1),
    Outcome.FAIL: (UNDETERMINED, 0),
}
"""How this fixture seeds a branch, keyed by the recording it is standing in for.

A branch writes no ``Outcome`` any more, so the ``Outcome`` in a ``concluded`` row is the *shape of
recording* the row describes and this table is the one place it is translated into what the branch
would actually write: a conformance and a span count. ``PASS`` was "found its subject and the
instruction was met"; ``FLAG`` was "found its subject and the instruction was not"; ``FAIL`` was
"found no subject", which under the new contract is no span and no conformance question answered.
"""


def _hint_config(tmp_path: Path) -> TriageConfig:
    """The packaged config with a hint map supplied, the same one ROUTING is tested against.

    Args:
        tmp_path: The test's temporary directory, holding the override file.

    Returns:
        The merged configuration.
    """
    path = tmp_path / "hints.yaml"
    path.write_text("routing:\n  hint_branch_map:\n    cough: AIRWAY\n    read-speech: SPEECH\n")
    return load_triage_config(path)


def _evaluation(
    routed: tuple[str, ...],
    state: RouteState,
    unavailable: Mapping[str, tuple[str, ...]],
    declared: tuple[str, ...] = (),
    family: str = "",
) -> RouteEvaluation:
    """One ruleset reading, standing in for the reduction the real ROUTING would run."""
    return RouteEvaluation(
        stem="sub-01",
        family=family,
        routed=routed,
        declared=declared,
        agreed=tuple(branch for branch in routed if branch in declared),
        missed=tuple(branch for branch in declared if branch not in routed),
        extra=tuple(branch for branch in routed if branch not in declared),
        unavailable=dict(unavailable),
        flags={},
        state=state,
        gate_outcomes={"airway.cough": GateOutcome.SILENT},
    )


@pytest.fixture
def make_verdict_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., ProvStore]:
    """A builder running the real ROUTING over a constructed reading, then seeding node verdicts."""

    def _make(
        *,
        concluded: Sequence[tuple[str, Outcome, str | None]] = (),
        routed: Sequence[str] = (),
        route_state: RouteState = RouteState.ROUTED,
        unavailable: Mapping[str, tuple[str, ...]] | None = None,
        route: bool = True,
        config: TriageConfig | None = None,
        hint: AudioHints | None = None,
        declared: Sequence[str] = (),
        family: str = "",
        recording_path: str | None = None,
    ) -> ProvStore:
        store = ProvStore(run_id="verdict-test")
        agent = software_agent(store)
        if recording_path is not None:
            store.entity(
                prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", "path": recording_path}
            )
        if route:
            reading = _evaluation(tuple(routed), route_state, unavailable or {}, tuple(declared), family)
            monkeypatch.setattr(routing_module, "evaluate_live_routes", lambda *a, **k: reading)
            routing(store, None, config or load_triage_config(), hint, run_dir=tmp_path)
        for node, outcome, kind in concluded:
            activity = store.activity(node=node, step="seed", parameters={})
            store.was_associated_with(activity, agent)
            if node in BRANCH_FAMILY:
                conformance, spans_n = BRANCH_READING[outcome]
                for index in range(spans_n):
                    span_id = store.entity(
                        prov_type="span",
                        extent=(float(index), float(index) + 1.0),
                        attributes={"family": BRANCH_FAMILY[node], "role": "task_extent"},
                    )
                    store.was_generated_by(span_id, activity)
                    store.was_attributed_to(span_id, agent)
                write_report(
                    store,
                    activity,
                    agent,
                    node=node,
                    kind=kind,
                    conformance=conformance,
                    conformance_of=TASK,
                    deviations=(),
                    detail={},
                )
                continue
            write_verdict(
                store,
                activity,
                agent,
                node=node,
                outcome=outcome,
                kind=kind,
                why=f"{node} concluded {outcome.value}",
                detail={},
            )
        return store

    return _make


def _file_verdict_entity(store: ProvStore) -> Entity:
    """The verdict entity whose node attribute is VERDICT — the file verdict, not a node's.

    Args:
        store: The provenance store.

    Returns:
        The single file verdict entity.
    """
    found = [e for e in store.entities("verdict") if e.attributes["node"] == "VERDICT"]
    assert len(found) == 1, f"expected exactly one file verdict, found {len(found)}"
    return found[0]


class TestTheTriageAxisIsWired:
    """The three values reach the store, and each carries the ground the fold gave it."""

    def test_a_branch_that_never_ran_leaves_a_file_pass(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A cough recording: AIRWAY routed and found its subject, SPEECH declined and never looked."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            routed=("AIRWAY",),
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.PASS
        assert result.file_verdict.findings["AIRWAY"] == "present"
        assert result.file_verdict.findings["SPEECH"] == "uncertain"

    def test_an_admit_failure_discards_as_unmeasurable(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Nothing ran, so nothing is claimed about the recording."""
        store = make_verdict_store(concluded=[("ADMIT", Outcome.FAIL, None)], route=False)
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.DISCARD
        assert result.file_verdict.discard_ground == "unmeasurable"

    def test_an_empty_recording_discards_as_acoustically_empty(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The emptiness bypass read every tracked stream peak under its floor; the fold reads that."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("TAXONOMY", Outcome.FAIL, None)],
            routed=(),
            route_state=RouteState.EMPTY,
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.DISCARD
        assert result.file_verdict.discard_ground == "acoustically_empty"

    def test_the_outcome_attribute_carries_the_triage_value(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """``discard`` is not an ``Outcome``, and the entity must still record it."""
        store = make_verdict_store(concluded=[("ADMIT", Outcome.FAIL, None)], route=False)
        verdict_module.verdict(store, None, config, run_dir=tmp_path)
        entity = _file_verdict_entity(store)
        assert entity.attributes["outcome"] == "discard"
        assert entity.attributes["triage"] == "discard"
        assert entity.attributes["discard_ground"] == "unmeasurable"


class TestTheRouteIsReadVerbatim:
    """A route state is read off ROUTING's own decision and never re-derived here."""

    def test_an_unreadable_route_is_resolved_by_the_branch(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A branch whose gates could not be read made no claim, so its branch settles it alone."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("SPEECH", Outcome.PASS, "speech")],
            routed=(),
            unavailable={"SPEECH": ("speech.words",)},
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.routes["SPEECH"] == "unavailable"
        assert result.file_verdict.findings["SPEECH"] == "present"
        assert result.file_verdict.agreement["SPEECH"] == "resolved"

    def test_the_route_is_never_rewritten_in_the_store(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A branch resolving its subject leaves ROUTING's decision exactly as it was."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("SPEECH", Outcome.PASS, "speech")],
            routed=(),
            route_state=RouteState.EMPTY,
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.agreement["SPEECH"] == "mismatch"
        assert result.file_verdict.triage is Triage.FLAG
        assert [
            e.attributes["route_state"] for e in store.entities("branch_decision") if e.attributes["branch"] == "SPEECH"
        ] == ["declined"]


class TestAnUnreadableNodeVerdictDoesNotKillTheFold:
    """One node writing something no reader can act on must not cost the whole file verdict."""

    def test_an_alien_outcome_flags_and_names_the_node_and_the_value(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The union on ``write_verdict``'s outcome means a node can write a triage value by mistake."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            routed=ROUTED_PAIR,
        )
        activity = store.activity(node="SPEECH", step="seed", parameters={})
        agent = software_agent(store)
        store.was_associated_with(activity, agent)
        alien = store.entity(
            prov_type="verdict",
            extent=None,
            attributes={"node": "SPEECH", "outcome": "discard", "kind": "speech", "why": "a node erred"},
        )
        store.was_generated_by(alien, activity)

        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.FLAG
        assert any("SPEECH" in reason.why and "'discard'" in reason.why for reason in result.file_verdict.reasons), (
            "the offending node and the value it wrote are both named"
        )

    def test_the_unreadable_verdict_resolves_nothing_and_the_fold_still_completes(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Every other node's conclusion survives, and the subject that node screened stays unanswered."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            routed=ROUTED_PAIR,
        )
        activity = store.activity(node="SPEECH", step="seed", parameters={})
        store.was_associated_with(activity, software_agent(store))
        alien = store.entity(
            prov_type="verdict",
            extent=None,
            attributes={"node": "SPEECH", "outcome": "banana", "kind": "speech", "why": "a node erred"},
        )
        store.was_generated_by(alien, activity)

        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.findings["AIRWAY"] == "present"
        assert result.file_verdict.findings["SPEECH"] == "uncertain", "no branch answered for it"
        assert result.file_verdict.agreement["SPEECH"] == "not_run"
        assert _file_verdict_entity(store).attributes["triage"] == "flag"


class TestTheBranchDecisionsAreRead:
    """Which branch was asked is a store fact now, not a guess from the classification."""

    def test_a_declined_branch_is_expected_and_does_not_flag(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """ROUTING declined SPEECH and said why; a missing verdict there is the design working."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            routed=("AIRWAY",),
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.agreement["SPEECH"] == "not_run"
        assert result.file_verdict.triage is Triage.PASS

    def test_an_asked_branch_that_left_no_verdict_flags(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """ROUTING selected SPEECH and nothing came back; the reason names which silence it was."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            routed=ROUTED_PAIR,
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path, ran={"SPEECH": RunState.ERRORED})
        assert result.file_verdict.triage is Triage.FLAG
        assert any("errored without a verdict" in reason.why for reason in result.file_verdict.reasons)

    def test_the_branches_map_joins_the_decision_to_the_reported_conformance(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A skipped branch carries the reason it was skipped, in the same record as the one that ran."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            routed=("AIRWAY",),
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        branches = result.file_verdict.branches
        assert branches["AIRWAY"] == {
            "will_run": True,
            "forced_by_declaration": False,
            "withheld_by_gate": False,
            "route_state": "routed",
            "conformance": True,
        }
        assert branches["SPEECH"]["will_run"] is False
        assert branches["SPEECH"]["conformance"] is None

    def test_a_run_with_no_routing_element_reads_no_branch_as_asked(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """ROUTING itself never ran, so no branch was asked and none is owed an answer."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None)],
            routed=ROUTED_PAIR,
            route=False,
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.branches == {}
        assert result.file_verdict.triage is Triage.PASS

    def test_a_routing_error_without_decisions_flags_instead_of_discarding(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """An execution failure cannot be mistaken for ROUTING deliberately declining every branch."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("TAXONOMY", Outcome.PASS, None)],
            routed=(),
            route_state=RouteState.EMPTY,
            route=False,
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path, ran={"routing": RunState.ERRORED})
        assert result.file_verdict.triage is Triage.FLAG
        assert result.file_verdict.discard_ground is None
        assert any(
            "routing failed; branch execution was withheld" in reason.why for reason in result.file_verdict.reasons
        )


class TestHintsAreReadThroughRoutingsMap:
    """The tag that forces a branch is the tag that can name a mismatch; one map, not two."""

    def test_a_declared_kind_no_branch_found_flags(
        self, make_verdict_store: Callable[..., ProvStore], tmp_path: Path
    ) -> None:
        """The declaration claimed a cough and AIRWAY found no labelled span."""
        hint = AudioHints(may_contain=["cough"])
        hint_config = _hint_config(tmp_path)
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.FAIL, "airway")],
            routed=(),
            route_state=RouteState.EMPTY,
            config=hint_config,
            hint=hint,
        )
        result = verdict_module.verdict(store, None, hint_config, hint, run_dir=tmp_path)
        assert result.file_verdict.hints["AIRWAY"] == "claimed_not_found"
        assert result.file_verdict.triage is Triage.FLAG

    def test_a_speech_type_value_is_a_claim_like_any_tag(
        self, make_verdict_store: Callable[..., ProvStore], tmp_path: Path
    ) -> None:
        """``speech_type`` goes through the same map, so the two cannot disagree about a tag."""
        hint = AudioHints(metadata={"speech_type": "read-speech"})
        hint_config = _hint_config(tmp_path)
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("SPEECH", Outcome.FAIL, "speech")],
            routed=("SPEECH",),
            config=hint_config,
            hint=hint,
        )
        result = verdict_module.verdict(store, None, hint_config, hint, run_dir=tmp_path)
        assert result.file_verdict.hints["SPEECH"] == "claimed_not_found"

    def test_a_tag_the_map_does_not_cover_claims_nothing(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The packaged map is null, so no tag reaches a kind and nothing is claimed."""
        hint = AudioHints(may_contain=["cough"])
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            routed=("AIRWAY",),
            hint=hint,
        )
        result = verdict_module.verdict(store, None, config, hint, run_dir=tmp_path)
        assert result.file_verdict.hints["AIRWAY"] == "found_unclaimed"
        assert result.file_verdict.triage is Triage.PASS

    def test_the_claim_is_read_off_the_decision_not_re_derived_from_the_config(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """ROUTING read the declaration with a map; VERDICT is handed one without it and must agree.

        The two nodes resolving the same tag independently is the divergence this reading removes:
        the claim is ROUTING's record of what it made of the hint, not a second opinion about it.
        """
        hint = AudioHints(may_contain=["cough"])
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.FAIL, "airway")],
            routed=(),
            route_state=RouteState.EMPTY,
            config=_hint_config(tmp_path),
            hint=hint,
        )
        result = verdict_module.verdict(store, None, config, hint, run_dir=tmp_path)
        assert result.file_verdict.hints["AIRWAY"] == "claimed_not_found"
        assert result.file_verdict.triage is Triage.FLAG

    def test_a_declaration_no_decision_survived_to_read_is_named_not_dropped(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """ROUTING errored, so what the declaration claimed is unknown; reading it as no claim is silent."""
        hint = AudioHints(may_contain=["cough"])
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None)],
            routed=(),
            route_state=RouteState.EMPTY,
            route=False,
            hint=hint,
        )
        result = verdict_module.verdict(store, None, config, hint, run_dir=tmp_path)
        assert result.file_verdict.hints == {}
        assert result.file_verdict.triage is Triage.FLAG
        assert any(reason.why == UNREAD_DECLARATION for reason in result.file_verdict.reasons)
        assert _file_verdict_entity(store).attributes["hints"] == {}

    def test_no_declaration_and_no_decision_claims_nothing_without_flagging(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """With no hint there is nothing to have lost, so the empty claim map is the honest one."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            routed=("AIRWAY",),
            route=False,
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.hints["AIRWAY"] == "found_unclaimed"
        assert result.file_verdict.triage is Triage.PASS

    def test_the_declared_task_is_a_claim_with_no_hint_and_the_null_map(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The falsehood the null map produced: a cough task read ``found_unclaimed`` before this."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("AIRWAY", Outcome.PASS, "airway")],
            routed=("AIRWAY",),
            declared=("AIRWAY",),
            family="voluntary-cough",
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.hints["AIRWAY"] == "claimed_and_found"
        assert result.file_verdict.hints["SPEECH"] == "no_claim"

    def test_a_declared_task_the_branch_did_not_find_flags(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The other half of the same table, reachable for the first time under the null map."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("VOICE", Outcome.FAIL, "voice")],
            routed=(),
            declared=("VOICE",),
            family="prolonged-vowel",
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.hints["VOICE"] == "claimed_not_found"
        assert result.file_verdict.triage is Triage.FLAG

    def test_a_declared_task_no_decision_survived_to_read_is_named_without_any_hint(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """ROUTING errored on a BIDS-named recording: the declaration exists and went unread."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None)],
            route=False,
            recording_path="sub-01_ses-1_task-voluntary-cough.wav",
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.hints == {}
        assert any(reason.why == UNREAD_DECLARATION for reason in result.file_verdict.reasons)

    def test_a_stem_declaring_no_task_and_no_hint_is_not_an_unread_declaration(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The control: without a task token on the path there was no declaration to lose."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None)], route=False, recording_path="some-recording.wav"
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert not any(reason.why == UNREAD_DECLARATION for reason in result.file_verdict.reasons)

    def test_a_map_typo_flags_the_file_it_would_otherwise_have_discarded(
        self, make_verdict_store: Callable[..., ProvStore], tmp_path: Path
    ) -> None:
        """ROUTING recorded the typo on every decision; the fold must not discard over it."""
        path = tmp_path / "typo.yaml"
        path.write_text("routing:\n  hint_branch_map:\n    cough: AIRWY\n")
        typo_config = load_triage_config(path)
        hint = AudioHints(may_contain=["cough"])
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None)],
            routed=(),
            route_state=RouteState.EMPTY,
            config=typo_config,
            hint=hint,
        )
        result = verdict_module.verdict(store, None, typo_config, hint, run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.FLAG
        assert result.file_verdict.discard_ground is None
        assert result.file_verdict.bad_map_values == {"cough": "AIRWY"}
        assert any("AIRWY" in reason.why for reason in result.file_verdict.reasons)
        assert _file_verdict_entity(store).attributes["bad_map_values"] == {"cough": "AIRWY"}

    def test_a_declaration_prevents_the_empty_discard(
        self, make_verdict_store: Callable[..., ProvStore], tmp_path: Path
    ) -> None:
        """Discarding a file the declaration says had a cough would delete the graph's own error."""
        hint = AudioHints(may_contain=["cough"])
        hint_config = _hint_config(tmp_path)
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None)],
            routed=(),
            route_state=RouteState.EMPTY,
            config=hint_config,
            hint=hint,
        )
        result = verdict_module.verdict(store, None, hint_config, hint, run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.FLAG
        assert result.file_verdict.discard_ground is None


class TestTheReleaseAxis:
    """REDACT's verdict, and nothing else, decides whether an artifact may be handed on."""

    def test_no_redact_verdict_is_not_assessed(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A recording with no scan is unexamined, which must not read as cleared."""
        store = make_verdict_store(concluded=BASE, routed=ROUTED_PAIR)
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.release is Release.NOT_ASSESSED

    def test_a_fail_withholds_and_a_pass_releases(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The two ends of the mapping, and the attribute the store records them in."""
        withheld = make_verdict_store(concluded=[*BASE, ("REDACT", Outcome.FAIL, None)], routed=ROUTED_PAIR)
        released = make_verdict_store(concluded=[*BASE, ("REDACT", Outcome.PASS, None)], routed=ROUTED_PAIR)
        assert verdict_module.verdict(withheld, None, config, run_dir=tmp_path).file_verdict.release is Release.WITHHELD
        result = verdict_module.verdict(released, None, config, run_dir=tmp_path)
        assert result.file_verdict.release is Release.RELEASABLE
        assert _file_verdict_entity(released).attributes["release"] == "releasable"

    def test_a_surviving_finding_does_not_move_the_triage_axis(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A release problem is not a measurement problem, and it is in the same record regardless."""
        store = make_verdict_store(concluded=[*BASE, ("REDACT", Outcome.FAIL, None)], routed=ROUTED_PAIR)
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.PASS
        assert result.file_verdict.release is Release.WITHHELD
        assert any(reason.node == "REDACT" for reason in result.file_verdict.reasons)

    def test_the_later_redact_verdict_governs(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A repaired REDACT run wrote fail then pass; the release axis must read the repair."""
        store = make_verdict_store(
            concluded=[*BASE, ("REDACT", Outcome.FAIL, None), ("REDACT", Outcome.PASS, None)], routed=ROUTED_PAIR
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.release is Release.RELEASABLE
        assert [r.outcome for r in result.file_verdict.reasons if r.node == "REDACT"] == [Outcome.PASS]


def _annotate(store: ProvStore, **attributes: Any) -> str:  # noqa: ANN401 — the re-read's own fields
    """Seed the annotation REDACT's LLM re-read writes, exactly as REDACT writes it.

    Args:
        store: The provenance store.
        **attributes: The re-read's own fields — status, iterations, flagged, model_id, revision,
            failure.

    Returns:
        The measurement's id.
    """
    agent = software_agent(store)
    activity = store.activity(node="REDACT", step="llm_check", parameters={})
    store.was_associated_with(activity, agent)
    entity = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={"name": REDACTION_LLM_ANNOTATION, "signal": "redacted_transcript", **attributes},
    )
    store.was_generated_by(entity, activity)
    store.was_attributed_to(entity, agent)
    return entity


def _llm_off(tmp_path: Path) -> TriageConfig:
    """The packaged config with the redaction re-read's triage ground switched off."""
    path = tmp_path / "llm-off.yaml"
    path.write_text("verdict:\n  llm_redaction_flags: false\n")
    return load_triage_config(path)


class TestTheRedactionReviewerAnnotatesAndThisNodeDecides:
    """Owner, 2026-09-17: "the llm is part of a branch, so it can only annotate (with provenance)"."""

    def test_the_packaged_config_ships_the_key_and_the_policy_reads_it(self) -> None:
        """A ground with no key is an unmeasured decision with no way to see or turn it off."""
        assert load_triage_config().require("verdict.llm_redaction_flags") is True
        assert FoldPolicy.from_config(load_triage_config()).llm_redaction_flags is True

    def test_a_flagged_re_read_does_not_withhold_the_release(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The defect. Release is REDACT's detector verdict; an unmeasured model gates no artifact."""
        store = make_verdict_store(concluded=[*BASE, ("REDACT", Outcome.PASS, None)], routed=ROUTED_PAIR)
        _annotate(store, status="flagged", iterations=2, flagged=["LOCATION"], model_id="stub/model", revision="a" * 40)
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.release is Release.RELEASABLE
        assert _file_verdict_entity(store).attributes["release"] == "releasable"

    def test_a_flagged_re_read_raises_the_triage_axis_under_the_shipped_key(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The safety signal survives the split: a human should look, and the ground names why."""
        store = make_verdict_store(concluded=[*BASE, ("REDACT", Outcome.PASS, None)], routed=ROUTED_PAIR)
        _annotate(store, status="flagged", iterations=2, flagged=["LOCATION"], model_id="stub/model", revision="a" * 40)
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.FLAG
        ground = next(reason for reason in result.file_verdict.reasons if LLM_REDACTION_RESIDUE in reason.why)
        assert ground.node == "VERDICT" and ground.why.endswith("LOCATION")

    def test_the_key_flipped_off_leaves_the_triage_axis_alone(
        self, make_verdict_store: Callable[..., ProvStore], tmp_path: Path
    ) -> None:
        """The switch is what makes turning the ground off a visible decision rather than a silence."""
        store = make_verdict_store(concluded=[*BASE, ("REDACT", Outcome.PASS, None)], routed=ROUTED_PAIR)
        _annotate(store, status="flagged", iterations=2, flagged=["LOCATION"], model_id="stub/model", revision="a" * 40)
        result = verdict_module.verdict(store, None, _llm_off(tmp_path), run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.PASS
        assert not [reason for reason in result.file_verdict.reasons if LLM_REDACTION_RESIDUE in reason.why]
        assert result.file_verdict.llm_redaction["status"] == "flagged", "off is not unrecorded"

    def test_a_detector_fail_still_withholds_whatever_the_reviewer_said(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The real redaction path is untouched: a surviving finding withholds, re-read or none."""
        store = make_verdict_store(concluded=[*BASE, ("REDACT", Outcome.FAIL, None)], routed=ROUTED_PAIR)
        _annotate(store, status="clean", iterations=1, flagged=[], model_id="stub/model", revision="a" * 40)
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.release is Release.WITHHELD

    def test_an_absent_re_read_grounds_nothing_and_is_never_silent(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A GPU queue must not decide a release; the absence is in the product instead."""
        store = make_verdict_store(concluded=[*BASE, ("REDACT", Outcome.PASS, None)], routed=ROUTED_PAIR)
        _annotate(
            store, status="absent", iterations=1, flagged=[], model_id="stub/model", revision=None, failure="timeout"
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.triage is Triage.PASS
        assert result.file_verdict.release is Release.RELEASABLE
        assert result.file_verdict.llm_redaction == {
            "status": "absent",
            "iterations": 1,
            "flagged": [],
            "model_id": "stub/model",
            "revision": None,
            "failure": "timeout",
        }

    def test_the_annotation_and_its_provenance_reach_the_written_verdict(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The model at its resolved commit is what makes the annotation evidence rather than an opinion."""
        store = make_verdict_store(concluded=[*BASE, ("REDACT", Outcome.PASS, None)], routed=ROUTED_PAIR)
        annotation_id = _annotate(
            store, status="flagged", iterations=2, flagged=["LOCATION"], model_id="stub/model", revision="a" * 40
        )
        verdict_module.verdict(store, None, config, run_dir=tmp_path)
        recorded = _file_verdict_entity(store).attributes["llm_redaction"]
        assert recorded["model_id"] == "stub/model" and recorded["revision"] == "a" * 40
        activity = next(a for a in store.activities("VERDICT"))
        assert annotation_id in store.uses_of(activity.id), "the fold cites the annotation it read"


class TestWhatTheStoreRecords:
    """The written detail is verdict.md's product, and every id it folded is used."""

    def test_the_detail_is_the_product(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """T9 reads this entity and nothing else; a missing key there is a re-derivation."""
        store = make_verdict_store(concluded=BASE, routed=ROUTED_PAIR)
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        attributes = _file_verdict_entity(store).attributes
        assert {
            "triage",
            "release",
            "discard_ground",
            "reasons",
            "ran",
            "branches",
            "findings",
            "routes",
            "route_state",
            "agreement",
            "hints",
            "bad_map_values",
        } <= attributes.keys()
        assert attributes["findings"] == result.file_verdict.findings
        assert attributes["routes"] == result.file_verdict.routes
        assert attributes["agreement"] == result.file_verdict.agreement
        assert attributes["hints"] == result.file_verdict.hints
        assert attributes["branches"] == result.file_verdict.branches
        assert attributes["ran"] == {node: state.value for node, state in result.file_verdict.ran.items()}

    def test_reasons_carry_every_contribution(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A flag naming one cause hides the others; every node's verdict appears in reasons."""
        store = make_verdict_store(
            concluded=[
                ("ADMIT", Outcome.PASS, None),
                ("PREPROCESS", Outcome.PASS, None),
                ("TAXONOMY", Outcome.PASS, None),
                ("AIRWAY", Outcome.FAIL, "airway"),
                ("SPEECH", Outcome.PASS, "speech"),
            ],
            routed=ROUTED_PAIR,
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        # Only the deciding nodes contribute a reason of their own now; a branch contributes one
        # only where this fold raises a ground against it, which is what "every contribution" means
        # after the split. Both branches are still in the record — under `conformance`.
        assert {"ADMIT", "PREPROCESS", "TAXONOMY"} <= {r.node for r in result.file_verdict.reasons}
        assert {"AIRWAY", "SPEECH"} <= result.file_verdict.conformance.keys()
        stored: list[dict[str, Any]] = _file_verdict_entity(store).attributes["reasons"]
        assert [r["node"] for r in stored] == [r.node for r in result.file_verdict.reasons]
        assert [r["outcome"] for r in stored] == [r.outcome.value for r in result.file_verdict.reasons]
        assert [r["why"] for r in stored] == [r.why for r in result.file_verdict.reasons]

    def test_every_folded_id_is_used_and_the_view_leads_with_the_file_verdict(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Used edges to every node verdict, the reading and every decision; view = [file id, *folded ids]."""
        store = make_verdict_store(concluded=BASE, routed=ROUTED_PAIR)
        folded_ids = (
            {e.id for e in store.entities("verdict")}
            | {e.id for e in store.entities("measurement")}
            | {e.id for e in store.entities("branch_decision")}
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)

        file_entity = _file_verdict_entity(store)
        assert result.verdict_entity_id == file_entity.id
        assert result.view[0] == file_entity.id
        # Both record types are folded, so both are in the view: a `verdict` per deciding node and a
        # `branch_report` per reporting one.
        reports = {e.id for e in store.entities("branch_report")}
        assert set(result.view[1:]) == folded_ids | reports
        assert len(result.view) == len(set(result.view))

        activity_id = store.generated_by(file_entity.id)
        assert activity_id is not None
        activity = store.get_activity(activity_id)
        assert activity.node == "VERDICT"
        assert activity.parameters["config_hash"] == config.config_hash
        assert folded_ids <= set(store.uses_of(activity_id))
        agent_ids = store.associated_with(activity_id)
        assert agent_ids, "the software agent runs the fold"
        assert store.get_agent(agent_ids[0]).agent_type == "software"

    def test_the_file_verdict_is_not_folded_back_into_itself(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Running twice over one store must not read the first file verdict as a node's."""
        store = make_verdict_store(concluded=BASE, routed=ROUTED_PAIR)
        first = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        second = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert second.file_verdict == first.file_verdict
        assert "VERDICT" not in {r.node for r in second.file_verdict.reasons}

    def test_an_invalidated_branch_report_does_not_vote(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """An invalidated report is not a report; the branch that wrote it is owed an answer again."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("SPEECH", Outcome.PASS, "speech")],
            routed=("SPEECH",),
        )
        speech = next(e for e in store.entities("branch_report") if e.attributes["node"] == "SPEECH")
        store.was_invalidated_by(speech.id, store.activity(node="SPEECH", step="withdraw", parameters={}))
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert "SPEECH" not in result.file_verdict.conformance
        assert result.file_verdict.findings["SPEECH"] == "uncertain"
        assert result.file_verdict.triage is Triage.FLAG
        assert speech.id not in result.view

    def test_a_superseded_report_is_replaced_not_added(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Two reports from one branch are one contribution, the latest."""
        store = make_verdict_store(
            concluded=[
                ("ADMIT", Outcome.PASS, None),
                ("SPEECH", Outcome.FAIL, "speech"),
                ("SPEECH", Outcome.PASS, "speech"),
            ],
            routed=("SPEECH",),
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.conformance["SPEECH"] is True
        assert result.file_verdict.findings["SPEECH"] == "present"
        assert len([r for r in result.file_verdict.reasons if r.node == "SPEECH"]) == 0


class TestGraphOrderAndRan:
    """The node order is the runner's, and ``ran`` is merged, the runner's over the store's."""

    def test_routing_is_in_the_graph_order_the_runner_uses(self) -> None:
        """The casing is ``run.GRAPH_ORDER``'s: a name that sorts as unknown reports no run state."""
        assert "routing" in verdict_module._GRAPH_ORDER
        assert set(verdict_module._GRAPH_ORDER) <= set(GRAPH_ORDER)

    def test_node_verdicts_are_folded_in_graph_order(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Seeded out of order, the reasons still read in graph order; unknown nodes come last."""
        store = make_verdict_store(
            concluded=[
                ("REDACT", Outcome.PASS, None),
                ("SOMETHING_ELSE", Outcome.PASS, None),
                ("ADMIT", Outcome.PASS, None),
                ("TAXONOMY", Outcome.PASS, None),
            ],
            routed=(),
            route_state=RouteState.EMPTY,
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        seeded = [r.node for r in result.file_verdict.reasons if r.node != "VERDICT"]
        assert seeded == ["ADMIT", "TAXONOMY", "routing", "REDACT", "SOMETHING_ELSE"]

    def test_ran_is_derived_when_omitted_and_the_runners_wins_where_it_speaks(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A verdict is completed, an activity without one is errored, neither is skipped."""
        store = make_verdict_store(concluded=BASE, routed=ROUTED_PAIR)
        derived = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert derived.file_verdict.ran["ADMIT"] is RunState.COMPLETED
        assert derived.file_verdict.ran["routing"] is RunState.COMPLETED
        assert derived.file_verdict.ran["REDACT"] is RunState.SKIPPED

        supplied = verdict_module.verdict(
            make_verdict_store(concluded=BASE, routed=ROUTED_PAIR),
            None,
            config,
            run_dir=tmp_path,
            ran={"REDACT": RunState.ERRORED},
        )
        assert supplied.file_verdict.ran["REDACT"] is RunState.ERRORED
        assert supplied.file_verdict.ran["ADMIT"] is RunState.COMPLETED

    def test_a_node_that_ran_and_left_no_verdict_is_errored_not_skipped(
        self, make_verdict_store: Callable[..., ProvStore], config: TriageConfig, tmp_path: Path
    ) -> None:
        """An activity with no verdict is the raising node's signature; neither is never having run."""
        store = make_verdict_store(
            concluded=[("ADMIT", Outcome.PASS, None), ("TAXONOMY", Outcome.PASS, None)],
            routed=("SPEECH",),
        )
        store.was_associated_with(
            store.activity(node="SPEECH", step="transcript", parameters={}), software_agent(store)
        )
        result = verdict_module.verdict(store, None, config, run_dir=tmp_path)
        assert result.file_verdict.ran["SPEECH"] is RunState.ERRORED
        assert result.file_verdict.ran["VOICE"] is RunState.SKIPPED
        assert any("errored without a verdict" in r.why for r in result.file_verdict.reasons)
        assert _file_verdict_entity(store).attributes["ran"]["SPEECH"] == "errored"


class TestTheFoldIsWiredNotReimplemented:
    """The node maps store facts onto the fold's inputs and does not decide anything itself."""

    def test_the_fold_is_called_and_no_table_lives_here(self) -> None:
        """The kind-to-branch table is ROUTING's, and the gate question is answered by its elements."""
        source = inspect.getsource(verdict_module)
        assert "fold_file_verdict(" in source
        assert "_BRANCH_FOR_KIND" not in source
        assert "_is_gated" not in source

    def test_no_sibling_node_is_imported(self) -> None:
        """A node calling into another node's module couples two nodes outside the store."""
        source = inspect.getsource(verdict_module)
        assert "workflows.triage.nodes.routing" not in source
