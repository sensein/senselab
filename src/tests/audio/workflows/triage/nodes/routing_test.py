"""ROUTING: which branches run, why, and the record that lets VERDICT tell 'nothing' from 'never looked'."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import pytest

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import routing as routing_module
from senselab.audio.workflows.triage.nodes.common import find_measurement, live_entities
from senselab.audio.workflows.triage.nodes.routing import routing
from senselab.audio.workflows.triage.routing_analysis.ruleset import GateOutcome, RouteEvaluation, RouteState
from senselab.audio.workflows.triage.vocabulary import BRANCHES, RULESET_ROUTING, Outcome
from senselab.utils.prov_store import ProvStore


def _config(tmp_path: Path, entries: str) -> TriageConfig:
    """The packaged config with ``routing.hint_branch_map`` supplied from the given YAML entries."""
    path = tmp_path / "routing.yaml"
    path.write_text("routing:\n  hint_branch_map:\n" + entries)
    return load_triage_config(path)


def _gated(tmp_path: Path, required: str, entries: str = "    ddk: DDK\n") -> TriageConfig:
    """The packaged config with a hint map and ``routing.declaration_required`` both supplied."""
    path = tmp_path / "gated.yaml"
    path.write_text(f"routing:\n  declaration_required: {required}\n  hint_branch_map:\n{entries}")
    return load_triage_config(path)


def _map(tmp_path: Path) -> TriageConfig:
    """The packaged config with a hint map supplied, covering tags and one speech_type value."""
    return _config(
        tmp_path,
        "    speech: SPEECH\n"
        "    read-speech: SPEECH\n"
        "    cough: AIRWAY\n"
        "    phonation: VOICE\n"
        "    prolonged-vowel: VOICE\n",
    )


def _evaluation(
    routed: Sequence[str] = (),
    *,
    state: RouteState = RouteState.ROUTED,
    unavailable: Mapping[str, tuple[str, ...]] | None = None,
    flags: Mapping[str, tuple[str, ...]] | None = None,
    gate_outcomes: Mapping[str, GateOutcome] | None = None,
    declared: Sequence[str] = (),
    family: str = "",
) -> RouteEvaluation:
    """One ruleset reading, as ``evaluate_live_routes`` would return it."""
    return RouteEvaluation(
        stem="sub-01_task-x",
        family=family,
        routed=tuple(routed),
        declared=tuple(declared),
        agreed=tuple(branch for branch in routed if branch in declared),
        missed=tuple(branch for branch in declared if branch not in routed),
        extra=tuple(branch for branch in routed if branch not in declared),
        unavailable=dict(unavailable or {}),
        flags=dict(flags or {}),
        state=state,
        gate_outcomes=dict(gate_outcomes or {"airway.cough": GateOutcome.SILENT}),
    )


@pytest.fixture
def reads(monkeypatch: pytest.MonkeyPatch) -> Callable[[RouteEvaluation], None]:
    """Make ROUTING's ruleset reading return a constructed evaluation rather than reduce the store."""

    def _install(evaluation: RouteEvaluation) -> None:
        monkeypatch.setattr(routing_module, "evaluate_live_routes", lambda *a, **k: evaluation)

    return _install


class TestTheRulesetDecides:
    """The routed set is the execution set. Nothing else selects a branch."""

    def test_a_routed_branch_runs(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """A gate fired for SPEECH, so SPEECH runs and nothing else does."""
        reads(_evaluation(["SPEECH"]))
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert result.runs == ("SPEECH",)
        assert set(result.skipped) == {"AIRWAY", "VOICE", "DDK"}
        assert result.route_state == "routed"

    def test_a_branch_nothing_routed_is_declined(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """Every gate was evaluated and none fired, which is a reading and not an absence."""
        reads(_evaluation(["SPEECH"]))
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        decisions = {e.attributes["branch"]: e.attributes for e in live_entities(store, "branch_decision")}
        assert decisions["AIRWAY"]["route_state"] == "declined"
        assert decisions["AIRWAY"]["why"] == "route_declined"
        assert decisions["SPEECH"]["route_state"] == "routed"

    def test_a_branch_whose_gates_could_not_be_read_is_unavailable_and_still_withheld(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """A gate that was never measured did not decline; that is recorded, not turned into a run."""
        reads(_evaluation(["SPEECH"], unavailable={"VOICE": ("voice.glide",)}))
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert "VOICE" not in result.runs
        voice = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "VOICE")
        assert voice.attributes["route_state"] == "unavailable"
        assert voice.attributes["unavailable_gates"] == ["voice.glide"]

    def test_a_fired_flag_is_recorded_on_the_branch_it_annotates(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """A flag gate never routes; it must still reach the decision a reader inspects."""
        reads(_evaluation(["SPEECH"], flags={"SPEECH": ("speech.short",)}))
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        speech = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "SPEECH")
        assert speech.attributes["flag_gates"] == ["speech.short"]

    def test_the_reading_is_recorded_as_a_measurement(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """ROUTING owns the measurement now, under its own node, with no authority flag on it."""
        reads(_evaluation(["SPEECH"]))
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        recorded = find_measurement(store, RULESET_ROUTING)
        assert recorded is not None
        assert recorded.attributes["state"] == "routed"
        assert "authoritative" not in recorded.attributes
        assert "error" not in recorded.attributes
        activity_id = store.generated_by(recorded.id)
        assert activity_id is not None
        activity = store.get_activity(activity_id)
        assert (activity.node, activity.step) == ("routing", "ruleset_routing")

    def test_a_failed_evaluation_fails_the_node(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The ruleset decides execution, so a reading that cannot be made is a failure of the graph."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=["one", "two"])
        path = tmp_path / "unmeasured.yaml"
        path.write_text("windows:\n  yamnet: {default_threshold: null}\n")
        with pytest.raises(ValueError):
            routing(store, None, load_triage_config(path), run_dir=tmp_path)


class TestDDKRoutesOnTheDeclarationOnly:
    """``routing.declaration_required`` ships ``[DDK]``: the declaration decides, in both directions."""

    def test_a_ruleset_route_alone_does_not_run_a_gated_branch(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """Content analysis routed DDK and nothing declared it, so the branch does not run."""
        reads(_evaluation(["DDK"]))
        result = routing(store, None, _gated(tmp_path, "[DDK]"), run_dir=tmp_path)
        assert result.runs == ()
        assert "DDK" in result.skipped

    def test_the_withheld_route_records_what_the_ruleset_thought(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """``route_state`` is not overwritten, so the gate's decision stays separable from the reading."""
        reads(_evaluation(["DDK"]))
        routing(store, None, _gated(tmp_path, "[DDK]"), run_dir=tmp_path)
        ddk = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "DDK")
        assert ddk.attributes["route_state"] == "routed"
        assert ddk.attributes["withheld_by_gate"] is True
        assert ddk.attributes["will_run"] is False
        assert ddk.attributes["why"] == "route_routed_withheld_pending_declaration"

    def test_a_declared_family_runs_the_gated_branch_whatever_the_ruleset_thought(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """The other direction: the declaration is ground truth, so a declined reading does not stop it."""
        reads(_evaluation([], declared=["DDK"], family="diadochokinesis"))
        result = routing(store, None, _gated(tmp_path, "[DDK]"), run_dir=tmp_path)
        assert result.runs == ("DDK",)
        ddk = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "DDK")
        assert ddk.attributes["route_state"] == "declined"
        assert ddk.attributes["withheld_by_gate"] is False
        assert ddk.attributes["forced_by_declaration"] is True

    def test_a_hint_tag_satisfies_the_gate_as_the_family_does(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """The gate reads ``by_declaration``, which is the family or a mapped tag, not the family alone."""
        reads(_evaluation(["DDK"]))
        hint = AudioHints(may_contain=["ddk"])
        result = routing(store, None, _gated(tmp_path, "[DDK]"), hint, run_dir=tmp_path)
        assert result.runs == ("DDK",)
        ddk = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "DDK")
        assert ddk.attributes["withheld_by_gate"] is False

    def test_a_branch_the_key_does_not_name_keeps_the_additive_behaviour(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """Only the named branches change; AIRWAY still runs on the ruleset's route alone."""
        reads(_evaluation(["AIRWAY", "DDK"]))
        result = routing(store, None, _gated(tmp_path, "[DDK]"), run_dir=tmp_path)
        assert result.runs == ("AIRWAY",)
        airway = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "AIRWAY")
        assert airway.attributes["withheld_by_gate"] is False
        assert airway.attributes["declaration_required"] is False

    def test_an_empty_key_restores_the_ungated_behaviour_everywhere(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """The gate is a membership list, so an empty one gates nothing."""
        reads(_evaluation(["DDK"]))
        result = routing(store, None, _gated(tmp_path, "[]"), run_dir=tmp_path)
        assert result.runs == ("DDK",)

    def test_a_configured_name_that_is_not_a_branch_is_recorded_on_every_decision(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """A typo gates nothing and is otherwise silent, so it is carried the way a bad map value is."""
        reads(_evaluation(["DDK"]))
        result = routing(store, None, _gated(tmp_path, "[DDKK]"), run_dir=tmp_path)
        assert result.runs == ("DDK",)
        decisions = live_entities(store, "branch_decision")
        assert all(e.attributes["bad_declaration_required"] == ["DDKK"] for e in decisions)

    def test_every_branch_gets_a_decision(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """One decision per branch in the vocabulary, DDK included, and none for REDACT."""
        reads(_evaluation(["SPEECH"]))
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        branches = {e.attributes["branch"] for e in live_entities(store, "branch_decision")}
        assert branches == set(BRANCHES)


class TestTheDeclaredTaskAlwaysAddsItsBranch:
    """Owner decision: a declared task routes to its own branch whatever the content gates read."""

    def test_a_declared_branch_runs_although_every_gate_was_silent(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """The whole decision: content said nothing about VOICE and the declaration routes it anyway."""
        reads(_evaluation([], declared=["VOICE"], family="prolonged-vowel"))
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert result.runs == ("VOICE",)
        assert result.forced == ("VOICE",)
        assert result.declared == ("VOICE",)

    def test_the_added_route_needs_no_hint_at_all(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """The declaration is the task on the recording's own stem, not something the caller passes."""
        reads(_evaluation([], declared=["AIRWAY"], family="voluntary-cough"))
        assert routing(store, None, _map(tmp_path), None, run_dir=tmp_path).runs == ("AIRWAY",)

    def test_the_declared_route_does_not_rewrite_the_content_reading(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """A branch added by declaration still records that its gates declined; that is the mismatch."""
        reads(_evaluation([], declared=["VOICE"], family="prolonged-vowel"))
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        voice = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "VOICE")
        assert voice.attributes["route_state"] == "declined"
        assert voice.attributes["will_run"] is True
        assert voice.attributes["why"] == "route_declined_forced_by_declaration"

    def test_a_declaration_never_removes_a_content_route(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """A family declaring only VOICE leaves a content-routed SPEECH running."""
        reads(_evaluation(["SPEECH"], declared=["VOICE"], family="prolonged-vowel"))
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert set(result.runs) == {"SPEECH", "VOICE"}
        speech = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "SPEECH")
        assert speech.attributes["route_state"] == "routed"
        assert speech.attributes["forced_by_declaration"] is False

    def test_a_declared_branch_the_content_already_routed_is_counted_once(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """Both sources naming the same branch is one route, and the added-route record stays empty."""
        reads(_evaluation(["SPEECH"], declared=["SPEECH"], family="rainbow-passage"))
        result = routing(store, None, _map(tmp_path), AudioHints(may_contain=["speech"]), run_dir=tmp_path)
        assert result.runs == ("SPEECH",)
        assert result.runs.count("SPEECH") == 1
        assert result.forced == ()
        assert result.declared == ("SPEECH",)
        speech = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "SPEECH")
        assert speech.attributes["declared"] is True
        assert speech.attributes["forced_by_declaration"] is False
        assert speech.attributes["why"] == "route_routed"

    def test_the_decision_tells_a_declared_route_from_a_content_route(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """A reader must be able to say which source put each branch in the execution set."""
        reads(_evaluation(["SPEECH"], declared=["DDK"], family="diadochokinesis-pa"))
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        decisions = {e.attributes["branch"]: e.attributes for e in live_entities(store, "branch_decision")}
        assert (decisions["SPEECH"]["route_state"], decisions["SPEECH"]["forced_by_declaration"]) == ("routed", False)
        assert (decisions["DDK"]["route_state"], decisions["DDK"]["forced_by_declaration"]) == ("declined", True)
        assert decisions["DDK"]["declared_by_family"] is True
        assert decisions["VOICE"]["will_run"] is False
        assert decisions["VOICE"]["declared"] is False

    def test_the_declared_family_is_recorded_on_every_decision(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """What declared the route must be nameable from the store, not only that something did."""
        reads(_evaluation([], declared=["AIRWAY"], family="voluntary-cough"))
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        families = {e.attributes["declared_family"] for e in live_entities(store, "branch_decision")}
        assert families == {"voluntary-cough"}

    def test_the_reading_records_what_the_declaration_named(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """The measurement carries the declaration, so the added route is auditable from the run."""
        reads(_evaluation(["SPEECH"], declared=["SPEECH", "DDK"], family="diadochokinesis-pa"))
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        recorded = find_measurement(store, RULESET_ROUTING)
        assert recorded is not None
        assert recorded.attributes["declared"] == ["SPEECH", "DDK"]
        assert recorded.attributes["family"] == "diadochokinesis-pa"
        assert recorded.attributes["routed"] == ["SPEECH"]

    def test_no_declaration_and_no_hint_routes_by_content_alone(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """A recording whose stem declares nothing behaves exactly as it did before this decision."""
        reads(_evaluation(["SPEECH"]))
        result = routing(store, None, config, None, run_dir=tmp_path)
        assert result.runs == ("SPEECH",)
        assert result.forced == ()
        assert result.declared == ()
        assert all(
            (d.attributes["declared"], d.attributes["forced_by_declaration"], d.attributes["declared_family"])
            == (False, False, "")
            for d in live_entities(store, "branch_decision")
        )

    def test_a_declaration_nothing_routed_still_leaves_the_file_state_alone(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """The added route is a branch decision; what the ruleset made of the recording is unchanged."""
        reads(_evaluation([], state=RouteState.EMPTY, declared=["VOICE"], family="prolonged-vowel"))
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert result.route_state == "empty"
        assert result.runs == ("VOICE",)
        assert result.empty_set is False


class TestTheDeclaredTaskIsReadOffTheStem:
    """The route comes from the recording's own path, through the ruleset's family sets."""

    def test_a_bids_stem_declares_its_branch_without_any_mocking(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The whole chain: ADMIT's stream path -> task id -> family -> reference set -> the route."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=["one", "two"])
        store.entity(
            prov_type="stream",
            extent=(0.0, 1.0),
            attributes={"name": "recording", "path": "sub-01_ses-1_task-prolonged-vowel-1.wav"},
        )
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert "VOICE" in result.declared
        assert "VOICE" in result.runs

    def test_a_stem_with_no_task_entity_declares_nothing(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The control for the test above: without the task token the same store routes by content."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=["one", "two"])
        store.entity(
            prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", "path": "some-recording.wav"}
        )
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert result.declared == ()
        assert result.forced == ()


class TestHintsForceAndNothingElse:
    """A hint adds a branch. It never rewrites a reading and never removes a branch."""

    def test_a_hint_forces_a_declined_branch(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """The branch runs against a declined route, which is the mismatch VERDICT detects."""
        reads(_evaluation([]))
        result = routing(store, None, _map(tmp_path), AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert result.runs == ("AIRWAY",)
        assert result.forced == ("AIRWAY",)

    def test_speech_type_metadata_forces_too(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """Both ``may_contain`` and the task's ``speech_type`` are forcing inputs."""
        reads(_evaluation([]))
        hint = AudioHints(metadata={"speech_type": "read-speech"})
        assert routing(store, None, _map(tmp_path), hint, run_dir=tmp_path).runs == ("SPEECH",)

    def test_forcing_does_not_rewrite_the_route_state(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """The disagreement between decision and reading is the product, not a thing to erase."""
        reads(_evaluation([]))
        routing(store, None, _map(tmp_path), AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        airway = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "AIRWAY")
        assert airway.attributes["route_state"] == "declined"
        assert airway.attributes["forced_by_declaration"] is True
        assert airway.attributes["declared_by_family"] is False
        assert airway.attributes["why"] == "route_declined_forced_by_declaration"

    def test_a_hint_naming_a_routed_branch_forces_nothing_and_is_still_recorded(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """Forcing means the hint changed the outcome, not merely that it named the branch."""
        reads(_evaluation(["SPEECH"]))
        result = routing(store, None, _map(tmp_path), AudioHints(may_contain=["speech"]), run_dir=tmp_path)
        assert result.runs == ("SPEECH",)
        assert result.forced == ()
        speech = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "SPEECH")
        assert speech.attributes["forced_by_declaration"] is False
        assert speech.attributes["declared"] is True
        assert speech.attributes["hint_tags"] == ["speech"]
        assert speech.attributes["why"] == "route_routed"

    def test_forcing_never_removes_a_branch(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """A hint naming only cough leaves a routed SPEECH running."""
        reads(_evaluation(["SPEECH"]))
        result = routing(store, None, _map(tmp_path), AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert set(result.runs) == {"SPEECH", "AIRWAY"}

    def test_an_unmapped_tag_forces_nothing_and_is_recorded(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """A tag with no entry is data about the hint, not a silent no-op."""
        reads(_evaluation([]))
        result = routing(store, None, _map(tmp_path), AudioHints(may_contain=["birdsong"]), run_dir=tmp_path)
        assert result.runs == ()
        assert live_entities(store, "branch_decision")[0].attributes["unmapped_tags"] == ["birdsong"]

    def test_a_map_value_that_is_not_a_branch_forces_nothing_and_names_the_typo(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """A config typo silently under-routing every file in a run is its own thing to chase.

        The tag reached no branch, so it is unmapped like any other tag that reached none — which
        keeps the accounting total, every declared tag landing in exactly one of ``hint_tags`` and
        ``unmapped_tags``. ``bad_map_values`` then says *why* it reached none.
        """
        reads(_evaluation([]))
        config = _config(tmp_path, "    cough: AIRWY\n")
        result = routing(store, None, config, AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert result.runs == ()
        assert result.forced == ()
        decisions = live_entities(store, "branch_decision")
        assert [d.attributes["hint_tags"] for d in decisions] == [[]] * len(BRANCHES)
        assert decisions[0].attributes["unmapped_tags"] == ["cough"]
        assert decisions[0].attributes["bad_map_values"] == {"cough": "AIRWY"}

    def test_a_good_map_records_no_bad_values(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """The control: the typo record must stay empty when the map is well formed."""
        reads(_evaluation([]))
        routing(store, None, _map(tmp_path), AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert all(d.attributes["bad_map_values"] == {} for d in live_entities(store, "branch_decision"))

    def test_a_null_map_forces_nothing(
        self, store: ProvStore, config: TriageConfig, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """While the vocabulary is unmeasured, every tag is unmapped and nothing is forced."""
        reads(_evaluation([]))
        result = routing(store, None, config, AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert result.runs == ()
        assert result.empty_set is True


class TestTheEmptyExecutionSet:
    """A file that enters no branch is recorded, not judged: the fold decides what it means."""

    def test_no_branch_is_recorded_without_a_flag(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """A flag here preempts VERDICT's acoustically-empty discard, which would be unreachable."""
        reads(_evaluation([], state=RouteState.EMPTY))
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert result.empty_set is True
        assert result.route_state == "empty"
        assert "empty" in result.verdict.why
        assert all(d.attributes["will_run"] is False for d in live_entities(store, "branch_decision"))

    def test_an_unexplained_recording_is_recorded_as_such(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """Nothing routed and the recording was not empty: a charge against the ruleset, kept apart."""
        reads(_evaluation([], state=RouteState.UNEXPLAINED))
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert result.route_state == "unexplained"
        assert result.empty_set is True

    def test_any_branch_running_passes(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """A non-empty execution set is a pass; nothing here is a judgement about the recording."""
        reads(_evaluation(["SPEECH"]))
        assert routing(store, None, _map(tmp_path), run_dir=tmp_path).verdict.outcome is Outcome.PASS


class TestTheStoreContract:
    """One decision per branch, before any branch runs, tied to the reading it rests on."""

    def test_each_decision_names_the_stream_it_was_taken_over(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """A second pass over another stream stays tellable apart."""
        reads(_evaluation(["SPEECH"]))
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert {e.attributes["stream"] for e in live_entities(store, "branch_decision")} == {"plain"}

    def test_a_second_pass_over_another_stream_records_that_streams_name(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """The unit is encapsulated over one input stream, and the decision says which one."""
        reads(_evaluation(["SPEECH"]))
        routing(store, "suppressed_foreground", _map(tmp_path), run_dir=tmp_path)
        assert {e.attributes["stream"] for e in live_entities(store, "branch_decision")} == {"suppressed_foreground"}

    def test_each_decision_is_derived_from_the_reading(
        self, store: ProvStore, tmp_path: Path, reads: Callable[[RouteEvaluation], None]
    ) -> None:
        """``wasDerivedFrom`` ties the decision to the evaluation, and ``used`` records the read."""
        reads(_evaluation(["SPEECH"]))
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        recorded = find_measurement(store, RULESET_ROUTING)
        assert recorded is not None
        decision = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "SPEECH")
        assert recorded.id in store.derived_from(decision.id)
        activity_id = store.generated_by(decision.id)
        assert activity_id is not None
        assert recorded.id in store.uses_of(activity_id)


class TestItReadsTheLiveStore:
    """The reading runs over the store the graph has written, not over a file on disk."""

    def test_a_seeded_store_routes_without_a_serialised_run(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A reading that always failed would pass every mocked test above and say nothing."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=["one", "two"])
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert result.route_state in ("routed", "empty", "unexplained")
        recorded = find_measurement(store, RULESET_ROUTING)
        assert recorded is not None
        assert recorded.attributes["gate_outcomes"]
