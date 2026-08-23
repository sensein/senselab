"""The provenance store: PROV entities, activities, agents; append-only and order-independent."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from senselab.utils.prov_store import Activity, Agent, Entity, ProvStore


def _store() -> ProvStore:
    return ProvStore(run_id="run-1")


def _rewrite_records(path: Path, kind: str, mutate: Callable[[dict[str, object]], object]) -> None:
    """Rewrite every record of one kind in a written JSONL file through the given mutation."""
    lines = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("record") == kind:
            mutate(rec)
            line = json.dumps(rec, sort_keys=True)
        lines.append(line)
    path.write_text("\n".join(lines) + "\n")


def _written_model_agent(tmp_path: Path) -> Path:
    """Write a store holding one valid model agent and return the file path."""
    s = _store()
    s.agent(agent_type="model", model_id="google/hear", commit_sha="9b2eb2853c426676255cc6ac5804b7f1fe8e563f")
    path = tmp_path / "prov.jsonl"
    s.write_jsonl(path)
    return path


class TestPurpose:
    """A store records what was produced, by what, using what."""

    def test_an_entity_records_the_activity_that_generated_it(self) -> None:
        """The wasGeneratedBy relation replaces an author field."""
        s = _store()
        act = s.activity(node="PREPROCESS", step="spans", parameters={"k_db": 18.0})
        ent = s.entity(prov_type="span", extent=(1.0, 2.0), attributes={"peak_over_floor_db": 31.4})
        s.was_generated_by(ent, act)
        assert s.generated_by(ent) == act
        assert s.get_entity(ent).extent == (1.0, 2.0)

    def test_used_records_what_a_node_read(self) -> None:
        """The relation that makes dependency order inspectable rather than inferred."""
        s = _store()
        upstream = s.entity(prov_type="span", extent=(1.0, 2.0), attributes={})
        act = s.activity(node="AIRWAY", step="classify", parameters={})
        s.used(act, upstream)
        assert s.uses_of(act) == [upstream]

    def test_an_assertion_is_an_entity_derived_from_what_it_is_about(self) -> None:
        """label/confirm/contest are entities, so a confirm can name the assertion it answers."""
        s = _store()
        span = s.entity(prov_type="span", extent=(7.9, 8.5), attributes={})
        act = s.activity(node="AIRWAY", step="classify", parameters={})
        label = s.entity(prov_type="assertion", extent=None, attributes={"verb": "label", "value": "Cough"})
        s.was_generated_by(label, act)
        s.was_derived_from(label, span)
        confirm = s.entity(prov_type="assertion", extent=None, attributes={"verb": "confirm"})
        s.was_derived_from(confirm, label)
        assert s.derived_from(confirm) == [label]
        assert s.derived_from(label) == [span]


class TestAgents:
    """An agent may be a model, and its commit may be unknown."""

    def test_a_resolved_commit_is_accepted(self) -> None:
        """A 40-hex commit_sha is stored as given."""
        s = _store()
        sha = "9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
        a = s.agent(agent_type="model", model_id="google/hear", commit_sha=sha)
        assert s.get_agent(a).commit_sha == sha

    def test_a_ref_masquerading_as_a_commit_is_refused(self) -> None:
        """A ref recorded as a commit is rejected outright."""
        s = _store()
        with pytest.raises(ValueError, match="40-hex"):
            s.agent(agent_type="model", model_id="google/hear", commit_sha="main")

    def test_a_sha_with_a_trailing_newline_is_refused(self) -> None:
        """Unstripped subprocess output must not become a recorded commit."""
        s = _store()
        sha_with_newline = "9b2eb2853c426676255cc6ac5804b7f1fe8e563f\n"
        with pytest.raises(ValueError, match="40-hex"):
            s.agent(agent_type="model", model_id="google/hear", commit_sha=sha_with_newline)

    def test_a_model_agent_needs_a_model_id(self) -> None:
        """The docstring's 'required for a model agent' is enforced."""
        s = _store()
        with pytest.raises(ValueError, match="model_id"):
            s.agent(agent_type="model", commit_sha="9b2eb2853c426676255cc6ac5804b7f1fe8e563f")

    def test_an_empty_unresolved_reason_is_refused(self) -> None:
        """An empty reason is not a reason."""
        s = _store()
        with pytest.raises(ValueError, match="empty"):
            s.agent(agent_type="model", model_id="google/hear", unresolved_reason="")

    def test_a_commit_and_an_unresolved_reason_together_are_refused(self) -> None:
        """A resolved commit with an unresolved reason is a self-contradictory record."""
        s = _store()
        with pytest.raises(ValueError, match="exactly one"):
            s.agent(
                agent_type="model",
                model_id="google/hear",
                commit_sha="9b2eb2853c426676255cc6ac5804b7f1fe8e563f",
                unresolved_reason="hub 503",
            )

    def test_an_unresolved_commit_is_representable_rather_than_fatal(self) -> None:
        """A Hub outage must degrade, not block every write."""
        s = _store()
        a = s.agent(agent_type="model", model_id="google/hear", unresolved_reason="hub 503")
        got = s.get_agent(a)
        assert got.commit_sha is None and got.unresolved_reason == "hub 503"

    def test_a_model_agent_needs_one_of_the_two(self) -> None:
        """A model agent must carry a commit or the reason it is missing."""
        s = _store()
        with pytest.raises(ValueError, match="commit_sha or unresolved_reason"):
            s.agent(agent_type="model", model_id="google/hear")

    def test_an_activity_records_which_agent_ran_it(self) -> None:
        """The wasAssociatedWith relation links an activity to its agent."""
        s = _store()
        a = s.agent(agent_type="software", version="0.1.0")
        act = s.activity(node="PREPROCESS", step=None, parameters={})
        s.was_associated_with(act, a)
        assert s.associated_with(act) == [a]


class TestInvalidation:
    """Withdrawal keeps the entity."""

    def test_an_invalidated_entity_is_still_readable(self) -> None:
        """Invalidation marks an entity unusable without removing it."""
        s = _store()
        seg = s.entity(prov_type="speaker", extent=(7.9, 9.0), attributes={"speaker": "SPEAKER_00"})
        act = s.activity(node="SPEECH", step="withdraw", parameters={"reason": "airway span"})
        s.was_invalidated_by(seg, act)
        assert s.is_invalidated(seg)
        assert s.get_entity(seg).attributes["speaker"] == "SPEAKER_00"

    def test_nothing_can_be_deleted(self) -> None:
        """The store exposes no deletion surface."""
        s = _store()
        assert not hasattr(s, "delete_entity")
        assert not hasattr(s, "remove_relation")


class TestOrderIndependence:
    """Append-only makes a merge a set union."""

    def test_merging_in_either_order_gives_the_same_fingerprint(self) -> None:
        """Merge order does not change the merged content hash."""
        a, b = _store(), _store()
        a.entity(prov_type="span", extent=(1.0, 2.0), attributes={})
        b.entity(prov_type="word", extent=(1.1, 1.4), attributes={"word": "hello"})
        assert ProvStore.merge([a, b]).fingerprint() == ProvStore.merge([b, a]).fingerprint()

    def test_repeating_a_relation_does_not_change_the_fingerprint(self) -> None:
        """A re-run node recording the same relation twice is a no-op."""
        s = _store()
        act = s.activity(node="PREPROCESS", step=None, parameters={})
        ent = s.entity(prov_type="span", extent=(1.0, 2.0), attributes={})
        s.was_generated_by(ent, act)
        before = s.fingerprint()
        s.was_generated_by(ent, act)
        assert s.fingerprint() == before

    def test_merging_a_store_with_itself_alone_is_idempotent(self) -> None:
        """merge([s]) carries the same content hash as s."""
        s = _store()
        act = s.activity(node="PREPROCESS", step=None, parameters={})
        ent = s.entity(prov_type="span", extent=(1.0, 2.0), attributes={})
        s.was_generated_by(ent, act)
        assert ProvStore.merge([s]).fingerprint() == s.fingerprint()


class TestRoundTrip:
    """PROV-JSON-shaped JSONL survives a round trip."""

    def test_entities_activities_agents_and_relations_all_return(self, tmp_path: Path) -> None:
        """Every field of every record type returns by equality, not only the fields the ids digest."""
        s = _store()
        resolved = s.agent(
            agent_type="model", model_id="google/hear", commit_sha="9b2eb2853c426676255cc6ac5804b7f1fe8e563f"
        )
        unresolved = s.agent(agent_type="model", model_id="openai/whisper-large-v3", unresolved_reason="hub 503")
        s.agent(agent_type="software", version="0.1.0")
        act = s.activity(
            node="AIRWAY",
            step="classify",
            parameters={"labels": ["Cough"]},
            started="2026-08-21T10:00:00+00:00",
            ended="2026-08-21T10:00:07+00:00",
        )
        span = s.entity(prov_type="span", extent=(7.9, 8.5), attributes={"peak_over_floor_db": 31.4})
        verdict = s.entity(prov_type="verdict", extent=None, attributes={"value": "keep"})
        s.was_generated_by(verdict, act)
        s.used(act, span)
        s.was_associated_with(act, resolved)
        s.was_attributed_to(verdict, unresolved)
        s.was_derived_from(verdict, span)
        s.was_invalidated_by(span, act)
        path = tmp_path / "prov.jsonl"
        s.write_jsonl(path)
        back = ProvStore.read_jsonl(path)
        assert back.fingerprint() == s.fingerprint()
        assert back._entities == s._entities
        assert back._activities == s._activities
        assert back._agents == s._agents
        assert back._relations == s._relations

    def test_an_unrecognised_record_kind_is_a_legible_error(self, tmp_path: Path) -> None:
        """A corrupt line names its kind instead of dying on a raw KeyError."""
        path = tmp_path / "prov.jsonl"
        path.write_text('{"record": "banana"}\n')
        with pytest.raises(ValueError, match="banana"):
            ProvStore.read_jsonl(path)


class TestReadBackValidation:
    """What is enforced at write time also holds after a read."""

    def test_a_duplicated_relation_line_reads_back_as_one_triple(self, tmp_path: Path) -> None:
        """A file carrying the same relation twice must not resurrect the duplicate-triple divergence."""
        s = _store()
        act = s.activity(node="PREPROCESS", step=None, parameters={})
        ent = s.entity(prov_type="span", extent=(1.0, 2.0), attributes={})
        s.was_generated_by(ent, act)
        path = tmp_path / "prov.jsonl"
        s.write_jsonl(path)
        relation_line = next(ln for ln in path.read_text().splitlines() if json.loads(ln)["record"] == "relation")
        path.write_text(path.read_text() + relation_line + "\n")
        back = ProvStore.read_jsonl(path)
        assert back._relations == s._relations
        assert ProvStore.merge([back]).fingerprint() == back.fingerprint()

    def test_a_ref_in_commit_sha_is_refused_on_read(self, tmp_path: Path) -> None:
        """A hand-edited ref in commit_sha fails the read, naming the record."""
        path = _written_model_agent(tmp_path)
        _rewrite_records(path, "agent", lambda rec: rec.update(commit_sha="main"))
        with pytest.raises(ValueError, match="agent-.*40-hex"):
            ProvStore.read_jsonl(path)

    def test_a_newline_suffixed_sha_is_refused_on_read(self, tmp_path: Path) -> None:
        """A newline-suffixed SHA fails the read exactly as it fails the write."""
        path = _written_model_agent(tmp_path)
        _rewrite_records(path, "agent", lambda rec: rec.update(commit_sha="9b2eb2853c426676255cc6ac5804b7f1fe8e563f\n"))
        with pytest.raises(ValueError, match="40-hex"):
            ProvStore.read_jsonl(path)

    def test_a_model_agent_with_neither_commit_nor_reason_is_refused_on_read(self, tmp_path: Path) -> None:
        """Silence about the commit is refused on read, as agent() refuses it on write."""
        path = _written_model_agent(tmp_path)
        _rewrite_records(path, "agent", lambda rec: rec.update(commit_sha=None))
        with pytest.raises(ValueError, match="commit_sha or unresolved_reason"):
            ProvStore.read_jsonl(path)

    def test_a_record_missing_required_keys_names_them(self, tmp_path: Path) -> None:
        """A record missing a key fails naming the key and the line, not with a bare TypeError."""
        s = _store()
        s.activity(node="PREPROCESS", step=None, parameters={})
        path = tmp_path / "prov.jsonl"
        s.write_jsonl(path)
        _rewrite_records(path, "activity", lambda rec: rec.pop("node"))
        with pytest.raises(ValueError, match=r"missing keys \['node'\]"):
            ProvStore.read_jsonl(path)

    def test_a_line_without_a_record_key_is_a_legible_error(self, tmp_path: Path) -> None:
        """A line that is not a record object fails legibly, not with a KeyError."""
        path = tmp_path / "prov.jsonl"
        path.write_text('{"id": "span-abc"}\n')
        with pytest.raises(ValueError, match="'record' key"):
            ProvStore.read_jsonl(path)

    def test_an_unknown_relation_is_refused_on_read(self, tmp_path: Path) -> None:
        """A relation outside PROV's own six fails the read instead of loading quietly."""
        s = _store()
        act = s.activity(node="PREPROCESS", step=None, parameters={})
        ent = s.entity(prov_type="span", extent=(1.0, 2.0), attributes={})
        s.was_generated_by(ent, act)
        path = tmp_path / "prov.jsonl"
        s.write_jsonl(path)
        _rewrite_records(path, "relation", lambda rec: rec.update(relation="causedBy"))
        with pytest.raises(ValueError, match="causedBy"):
            ProvStore.read_jsonl(path)


class TestActivityTimestamps:
    """An activity records when it ran, without the timestamps entering its identity."""

    def test_timestamps_survive_a_round_trip(self, tmp_path: Path) -> None:
        """``started`` and ``ended`` come back exactly as written."""
        s = _store()
        act = s.activity(
            node="AIRWAY",
            step="classify",
            parameters={"labels": ["Cough"]},
            started="2026-08-21T10:00:00+00:00",
            ended="2026-08-21T10:00:07+00:00",
        )
        path = tmp_path / "prov.jsonl"
        s.write_jsonl(path)
        back = ProvStore.read_jsonl(path)
        assert back._activities[act].started == "2026-08-21T10:00:00+00:00"
        assert back._activities[act].ended == "2026-08-21T10:00:07+00:00"

    def test_two_runs_differing_only_in_timestamps_share_an_id(self) -> None:
        """Re-running a node with identical inputs mints the same activity id."""
        s = _store()
        first = s.activity(node="AIRWAY", step="classify", parameters={"k": 1}, started="2026-08-21T10:00:00+00:00")
        second = s.activity(node="AIRWAY", step="classify", parameters={"k": 1}, started="2026-08-21T11:30:00+00:00")
        assert first == second

    def test_timestamps_are_optional(self) -> None:
        """An activity without timestamps is still representable, with both fields None."""
        s = _store()
        act = s.activity(node="PREPROCESS", step=None, parameters={})
        assert s._activities[act].started is None
        assert s._activities[act].ended is None


class TestSpeechStoreAdditions:
    """The two one-token additions SPEECH forces on the store."""

    def test_target_match_is_a_prov_type(self, tmp_path: Path) -> None:
        """branch-speech.md's product table names target_match as an element kind."""
        store = ProvStore(run_id="t")
        eid = store.entity(prov_type="target_match", extent=None, attributes={"speaker": "SPEAKER_00"})
        assert store.get_entity(eid).prov_type == "target_match"
        path = tmp_path / "prov.jsonl"
        store.write_jsonl(path)
        assert ProvStore.read_jsonl(path).get_entity(eid).prov_type == "target_match"

    def test_get_activity_returns_what_activity_recorded(self) -> None:
        """An entity's author node is reachable: generated_by -> get_activity -> .node."""
        store = ProvStore(run_id="t")
        act = store.activity(node="SPEECH", step="diarize", parameters={})
        eid = store.entity(prov_type="speaker", extent=(1.0, 2.0), attributes={})
        store.was_generated_by(eid, act)
        generated = store.generated_by(eid)
        assert generated is not None
        assert store.get_activity(generated).node == "SPEECH"


class TestActivityReads:
    """Activities are readable as a set, the way entities are."""

    def test_activities_return_all_or_one_node(self) -> None:
        """A reader asking which nodes ran needs the activities without touching a private field."""
        s = _store()
        first = s.activity(node="AIRWAY", step="classify", parameters={})
        second = s.activity(node="AIRWAY", step="confirm", parameters={})
        third = s.activity(node="SPEECH", step="transcript", parameters={})
        assert {a.id for a in s.activities()} == {first, second, third}
        assert {a.id for a in s.activities("AIRWAY")} == {first, second}
        assert s.activities("VOICE") == []
