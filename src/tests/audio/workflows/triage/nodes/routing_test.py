"""ROUTING: which branches run, why, and the record that lets VERDICT tell 'nothing' from 'never looked'."""

from __future__ import annotations

from pathlib import Path

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.common import live_entities
from senselab.audio.workflows.triage.nodes.routing import BRANCH_FOR_KIND, routing
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore


def _config(tmp_path: Path, entries: str) -> TriageConfig:
    """The packaged config with ``routing.hint_kind_map`` supplied from the given YAML entries."""
    path = tmp_path / "routing.yaml"
    path.write_text("routing:\n  hint_kind_map:\n" + entries)
    return load_triage_config(path)


def _map(tmp_path: Path) -> TriageConfig:
    """The packaged config with a hint map supplied, covering tags and one speech_type value."""
    return _config(
        tmp_path,
        "    speech: speech\n"
        "    read-speech: speech\n"
        "    cough: airway\n"
        "    phonation: voice\n"
        "    prolonged-vowel: voice\n",
    )


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

    def test_a_state_this_node_cannot_read_runs_the_branch(self, store: ProvStore, tmp_path: Path) -> None:
        """A state nobody can read is not evidence of absence, so only ``absent`` withholds a branch.

        The same rule TAXONOMY applies to a missing derivative. Reading the rule the other way round
        — run only on the states this node knows — would make an unreadable classification silently
        skip the instrument that would have settled it.

        The state does not travel verbatim, either: ``kind_state`` and ``why`` are closed
        vocabularies a downstream reader switches on, so an unrecognised state folds to one token
        and the string TAXONOMY actually wrote is kept beside it.
        """
        _kinds(store, speech="wobbly", airway="absent", voice="absent")
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert result.runs == ("SPEECH",)
        decision = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "SPEECH")
        assert decision.attributes["kind_state"] == "unreadable"
        assert decision.attributes["why"] == "kind_unreadable"
        assert decision.attributes["raw_state"] == "wobbly"

    def test_a_readable_state_keeps_its_own_word(self, store: ProvStore, tmp_path: Path) -> None:
        """The control: folding the unreadable case must not flatten the three states that are read."""
        _kinds(store, speech="present", airway="absent", voice="uncertain")
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        states = {e.attributes["branch"]: e.attributes["kind_state"] for e in live_entities(store, "branch_decision")}
        assert states == {"SPEECH": "present", "AIRWAY": "absent", "VOICE": "uncertain"}
        raw = {e.attributes["branch"]: e.attributes["raw_state"] for e in live_entities(store, "branch_decision")}
        assert raw == states


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
        """The disagreement between decision and classification is the product, not a thing to erase.

        Read the way every store reader reads: the *latest* live kind element per kind wins, so a
        node that appended a rewritten classification beside the original would be caught here and
        not by a first-match read. The second assertion closes the same hole from the other side —
        ROUTING generates no kind element at all, whatever its state.
        """
        _kinds(store, speech="absent", airway="absent", voice="absent")
        routing(store, None, _map(tmp_path), AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        airway = [e for e in live_entities(store, "kind") if e.attributes["kind"] == "airway"][-1]
        assert airway.attributes["state"] == "absent"
        by_routing = {activity.id for activity in store.activities(node="routing")}
        assert [e for e in live_entities(store, "kind") if store.generated_by(e.id) in by_routing] == []
        decision = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "AIRWAY")
        assert decision.attributes["kind_state"] == "absent"
        assert decision.attributes["forced_by_hint"] is True

    def test_a_hint_naming_a_present_kind_forces_nothing_and_is_still_recorded(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """Forcing means the hint changed the outcome, not merely that it named the kind.

        A branch the classification was already running is not forced, so ``forced_by_hint`` stays
        equivalent to "this branch runs against an absent classification" — the mismatch verdict.md
        detects. The tags are recorded against the branch all the same: a hint that agreed with a
        running branch is a fact about the hint, not silence.
        """
        _kinds(store, speech="present", airway="absent", voice="absent")
        result = routing(store, None, _map(tmp_path), AudioHints(may_contain=["speech"]), run_dir=tmp_path)
        assert result.runs == ("SPEECH",)
        assert result.forced == ()
        decision = next(e for e in live_entities(store, "branch_decision") if e.attributes["branch"] == "SPEECH")
        assert decision.attributes["forced_by_hint"] is False
        assert decision.attributes["hint_tags"] == ["speech"]
        assert decision.attributes["why"] == "kind_present"

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

    def test_a_map_value_that_is_not_a_kind_forces_nothing_and_names_the_typo(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """A typo'd map value must not make a declared tag vanish from both records.

        The tag reached no kind this graph screens, so it is unmapped like any other tag that
        reached none — that keeps the accounting total, every declared tag landing in exactly one of
        ``hint_tags`` and ``unmapped_tags``. ``bad_map_values`` then says *why* it reached none,
        because a config typo silently under-routing every file in a run is a different thing to
        chase than a tag the vocabulary does not cover.
        """
        _kinds(store, speech="absent", airway="absent", voice="absent")
        config = _config(tmp_path, "    cough: airwy\n")
        result = routing(store, None, config, AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert result.runs == ()
        assert result.forced == ()
        decisions = live_entities(store, "branch_decision")
        assert [d.attributes["hint_tags"] for d in decisions] == [[], [], []]
        assert decisions[0].attributes["unmapped_tags"] == ["cough"]
        assert decisions[0].attributes["bad_map_values"] == {"cough": "airwy"}

    def test_a_good_map_records_no_bad_values(self, store: ProvStore, tmp_path: Path) -> None:
        """The control: the typo record must stay empty when the map is well formed."""
        _kinds(store, speech="absent", airway="absent", voice="absent")
        routing(store, None, _map(tmp_path), AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert all(d.attributes["bad_map_values"] == {} for d in live_entities(store, "branch_decision"))

    def test_a_null_map_forces_nothing(self, store: ProvStore, config: TriageConfig, tmp_path: Path) -> None:
        """While the vocabulary is unmeasured, every tag is unmapped and nothing is forced."""
        _kinds(store, speech="absent", airway="absent", voice="absent")
        result = routing(store, None, config, AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert result.runs == ()
        assert result.empty_set is True


class TestTheEmptyExecutionSet:
    """A file that enters no branch is recorded, not judged: the fold decides what it means."""

    def test_no_branch_is_recorded_without_a_flag(self, store: ProvStore, tmp_path: Path) -> None:
        """A flag here preempts VERDICT's acoustically-empty discard, which would be unreachable."""
        _kinds(store, speech="absent", airway="absent", voice="absent")
        result = routing(store, None, _map(tmp_path), run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert result.empty_set is True
        assert "absent" in result.verdict.why
        assert all(d.attributes["will_run"] is False for d in live_entities(store, "branch_decision"))

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

    def test_each_decision_names_its_kind_and_the_stream_it_was_taken_over(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """``kind`` is the key T8 joins branch verdicts to decisions on; ``stream`` is V14."""
        _kinds(store, speech="present", airway="present", voice="present")
        routing(store, None, _map(tmp_path), run_dir=tmp_path)
        by_branch = {e.attributes["branch"]: e for e in live_entities(store, "branch_decision")}
        assert {branch: e.attributes["kind"] for branch, e in by_branch.items()} == {
            branch: kind for kind, branch in BRANCH_FOR_KIND.items()
        }
        assert {e.attributes["stream"] for e in by_branch.values()} == {"plain"}

    def test_a_second_pass_over_another_stream_records_that_streams_name(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """The unit is encapsulated over one input stream, and the decision says which one."""
        _kinds(store, speech="present", airway="absent", voice="absent")
        routing(store, "suppressed_foreground", _map(tmp_path), run_dir=tmp_path)
        assert {e.attributes["stream"] for e in live_entities(store, "branch_decision")} == {"suppressed_foreground"}

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
