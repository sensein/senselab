"""ROUTING: which branches run, why, and the record that lets VERDICT tell 'nothing' from 'never looked'."""

from __future__ import annotations

from pathlib import Path

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.common import live_entities
from senselab.audio.workflows.triage.nodes.routing import BRANCH_FOR_KIND, routing
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore


def _map(tmp_path: Path) -> TriageConfig:
    """The packaged config with a hint map supplied, covering tags and one speech_type value."""
    path = tmp_path / "routing.yaml"
    path.write_text(
        "routing:\n"
        "  hint_kind_map:\n"
        "    speech: speech\n"
        "    read-speech: speech\n"
        "    cough: airway\n"
        "    phonation: voice\n"
        "    prolonged-vowel: voice\n"
    )
    return load_triage_config(path)


def _kinds(store: ProvStore, **states: str) -> None:
    """Write one kind element per named kind, as TAXONOMY would."""
    activity = store.activity(node="TAXONOMY", step="fold", parameters={})
    for kind, state in states.items():
        entity_id = store.entity(
            prov_type="kind", extent=None, attributes={"kind": kind, "state": state, "lines": {}, "stream": "plain"}
        )
        store.was_generated_by(entity_id, activity)


class TestTheRule:
    """present runs, uncertain runs, absent does not."""

    def test_present_runs(self, store: ProvStore, tmp_path: Path) -> None:
        """A kind the classification found runs its branch."""
        _kinds(store, speech="present", airway="absent", voice="absent")
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert result.runs == ("SPEECH",)

    def test_uncertain_runs(self, store: ProvStore, tmp_path: Path) -> None:
        """A kind the classification could not settle is exactly what a branch exists to settle."""
        _kinds(store, speech="uncertain", airway="absent", voice="absent")
        assert routing(store, None, _map(tmp_path), run_dir=tmp_path).runs == ("SPEECH",)

    def test_absent_does_not_run(self, store: ProvStore, tmp_path: Path) -> None:
        """With no hint, an absent kind's branch is skipped and the decision says why."""
        _kinds(store, speech="absent", airway="present", voice="absent")
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert result.runs == ("AIRWAY",)
        assert set(result.skipped) == {"SPEECH", "VOICE"}


class TestHintsForceAndNothingElse:
    """A hint adds a branch. It never rewrites a classification and never removes a branch."""

    def test_a_hint_forces_an_absent_kinds_branch(self, store: ProvStore, tmp_path: Path) -> None:
        """The branch runs against an absent classification, which is the mismatch VERDICT detects."""
        _kinds(store, speech="absent", airway="absent", voice="absent")
        result = routing(store, None, _map(tmp_path), AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert result.runs == ("AIRWAY",)
        assert result.forced == ("AIRWAY",)

    def test_speech_type_metadata_forces_too(self, store: ProvStore, tmp_path: Path) -> None:
        """routing.md names both may_contain and the task's speech_type as forcing inputs."""
        _kinds(store, speech="absent", airway="absent", voice="absent")
        hint = AudioHints(metadata={"speech_type": "read-speech"})
        assert routing(store, None, _map(tmp_path), hint, run_dir=tmp_path).runs == ("SPEECH",)

    def test_forcing_does_not_rewrite_the_kind_element(self, store: ProvStore, tmp_path: Path) -> None:
        """The disagreement between decision and classification is the product, not a thing to erase."""
        _kinds(store, speech="absent", airway="absent", voice="absent")
        routing(store, None, _map(tmp_path), AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        airway = next(e for e in live_entities(store, "kind") if e.attributes["kind"] == "airway")
        assert airway.attributes["state"] == "absent"
        decision = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "AIRWAY")
        assert decision.attributes["kind_state"] == "absent"
        assert decision.attributes["forced_by_hint"] is True

    def test_forcing_never_removes_a_branch(self, store: ProvStore, tmp_path: Path) -> None:
        """A hint naming only cough leaves a present speech kind's branch running."""
        _kinds(store, speech="present", airway="absent", voice="absent")
        result = routing(store, None, _map(tmp_path), AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert set(result.runs) == {"SPEECH", "AIRWAY"}

    def test_an_unmapped_tag_forces_nothing_and_is_recorded(self, store: ProvStore, tmp_path: Path) -> None:
        """A tag with no entry is data about the hint, not a silent no-op."""
        _kinds(store, speech="absent", airway="absent", voice="absent")
        result = routing(store, None, _map(tmp_path), AudioHints(may_contain=["birdsong"]), run_dir=tmp_path)
        assert result.runs == ()
        decision = live_entities(store, "branch_decision")[0]
        assert decision.attributes["unmapped_tags"] == ["birdsong"]

    def test_a_null_map_forces_nothing(self, store: ProvStore, config: TriageConfig, tmp_path: Path) -> None:
        """While the vocabulary is unmeasured, every tag is unmapped and nothing is forced."""
        _kinds(store, speech="absent", airway="absent", voice="absent")
        result = routing(store, None, config, AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert result.runs == ()
        assert result.verdict.outcome is Outcome.FLAG


class TestTheEmptyExecutionSet:
    """A file that enters no branch is flagged, not failed and not discarded."""

    def test_no_branch_flags(self, store: ProvStore, tmp_path: Path) -> None:
        """ROUTING has no fail; whether an empty set discards the file is the fold's decision."""
        _kinds(store, speech="absent", airway="absent", voice="absent")
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FLAG
        assert result.empty_set is True
        assert "absent" in result.verdict.why

    def test_any_branch_running_passes(self, store: ProvStore, tmp_path: Path) -> None:
        """A non-empty execution set is a pass; nothing here is a judgement about the recording."""
        _kinds(store, speech="present", airway="absent", voice="absent")
        assert routing(store, None, _map(tmp_path), run_dir=tmp_path).verdict.outcome is Outcome.PASS


class TestTheStoreContract:
    """One decision per branch, before any branch runs, tied to the classification it rests on."""

    def test_three_decisions_and_none_for_redact(self, store: ProvStore, tmp_path: Path) -> None:
        """REDACT is a step of SPEECH, not a branch beside it."""
        _kinds(store, speech="present", airway="present", voice="present")
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        branches = {e.attributes["branch"] for e in live_entities(store, "branch_decision")}
        assert branches == set(BRANCH_FOR_KIND.values()) == {"AIRWAY", "SPEECH", "VOICE"}

    def test_each_decision_is_derived_from_its_kind_element(self, store: ProvStore, tmp_path: Path) -> None:
        """``wasDerivedFrom`` ties the decision to the classification, and used records the read."""
        _kinds(store, speech="present", airway="absent", voice="absent")
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        speech_kind = next(e for e in live_entities(store, "kind") if e.attributes["kind"] == "speech")
        decision = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "SPEECH")
        assert speech_kind.id in store.derived_from(decision.id)
        activity = store.get_activity(store.generated_by(decision.id))
        assert speech_kind.id in store.uses_of(activity.id)

    def test_a_kind_taxonomy_never_wrote_is_uncertain_and_runs(self, store: ProvStore, tmp_path: Path) -> None:
        """A classification that is not in the store is not an absence; the branch is asked."""
        _kinds(store, speech="present")
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert set(result.runs) == {"SPEECH", "AIRWAY", "VOICE"}
        airway = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "AIRWAY")
        assert airway.attributes["kind_state"] == "uncertain"
