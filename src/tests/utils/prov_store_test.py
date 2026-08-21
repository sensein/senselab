"""The provenance store: PROV entities, activities, agents; append-only and order-independent."""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.utils.prov_store import Activity, Agent, Entity, ProvStore


def _store() -> ProvStore:
    return ProvStore(run_id="run-1")


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


class TestRoundTrip:
    """PROV-JSON-shaped JSONL survives a round trip."""

    def test_entities_activities_agents_and_relations_all_return(self, tmp_path: Path) -> None:
        """Writing then reading a store preserves records, relations and the fingerprint."""
        s = _store()
        ag = s.agent(agent_type="model", model_id="google/hear", commit_sha="9b2eb2853c426676255cc6ac5804b7f1fe8e563f")
        act = s.activity(node="AIRWAY", step="classify", parameters={"labels": ["Cough"]})
        ent = s.entity(prov_type="span", extent=(7.9, 8.5), attributes={})
        s.was_generated_by(ent, act)
        s.was_associated_with(act, ag)
        path = tmp_path / "prov.jsonl"
        s.write_jsonl(path)
        back = ProvStore.read_jsonl(path)
        assert back.fingerprint() == s.fingerprint()
        assert back.associated_with(act) == [ag]
