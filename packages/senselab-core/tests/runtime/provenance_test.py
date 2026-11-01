"""The provenance store: PROV entities, activities, agents; append-only and order-independent."""

from __future__ import annotations

import collections
import copy
import enum
import json
import math
import os
import unicodedata
from collections.abc import Callable
from decimal import Decimal
from fractions import Fraction
from pathlib import Path, PurePath, PurePosixPath
from typing import cast, get_args

import numpy
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from senselab_core.runtime import provenance
from senselab_core.runtime.provenance import (
    PROV_TYPE,
    Activity,
    Agent,
    Entity,
    ProvStore,
    UnrecordableValue,
    normalised,
    recorded,
    revised,
)
from senselab_worker.canonical import jcs_dumps

pytestmark = pytest.mark.logic

FORMAT_LINE = '{"format": "provenance@2.0", "record": "format"}'


def _store() -> ProvStore:
    return ProvStore(run_id="run-1")


def _rewrite_records(path: Path, kind: str, mutate: Callable[[dict[str, object]], object]) -> None:
    """Rewrite every record of one kind in a written JSONL file through the given mutation."""
    lines = []
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("record") == kind:
            mutate(rec)
            line = jcs_dumps(rec).decode("utf-8")
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
        ent = s.entity(prov_type="artifact", extent=(1.0, 2.0), attributes={"peak_over_floor_db": 31.4})
        s.was_generated_by(ent, act)
        assert s.generated_by(ent) == act
        assert s.get_entity(ent).extent == (1.0, 2.0)

    def test_used_records_what_a_node_read(self) -> None:
        """The relation that makes dependency order inspectable rather than inferred."""
        s = _store()
        upstream = s.entity(prov_type="artifact", extent=(1.0, 2.0), attributes={})
        act = s.activity(node="AIRWAY", step="classify", parameters={})
        s.used(act, upstream)
        assert s.uses_of(act) == [upstream]

    def test_an_assertion_is_an_entity_derived_from_what_it_is_about(self) -> None:
        """label/confirm/contest are entities, so a confirm can name the assertion it answers."""
        s = _store()
        span = s.entity(prov_type="artifact", extent=(7.9, 8.5), attributes={})
        act = s.activity(node="AIRWAY", step="classify", parameters={})
        label = s.entity(prov_type="artifact", extent=None, attributes={"verb": "label", "value": "Cough"})
        s.was_generated_by(label, act)
        s.was_derived_from(label, span)
        confirm = s.entity(prov_type="artifact", extent=None, attributes={"verb": "confirm"})
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

    def test_every_agent_is_enumerable_by_type(self) -> None:
        """A reader collecting a run's model provenance must not have to know the ids in advance."""
        s = _store()
        model = s.agent(agent_type="model", model_id="google/hear", unresolved_reason="hub 503")
        software = s.agent(agent_type="software", version="0.1.0")
        assert [g.id for g in s.agents("model")] == [model]
        assert [g.id for g in s.agents("software")] == [software]
        assert {g.id for g in s.agents()} == {model, software}

    def test_an_agent_names_the_activities_it_ran(self) -> None:
        """The inverse of associated_with: what one agent did, rather than who ran one activity."""
        s = _store()
        a = s.agent(agent_type="model", model_id="google/hear", unresolved_reason="hub 503")
        first = s.activity(node="PREPROCESS", step="hear", parameters={})
        second = s.activity(node="PREPROCESS", step="hear_windows", parameters={})
        s.was_associated_with(first, a)
        s.was_associated_with(second, a)
        assert s.associations_of(a) == [first, second]

    def test_an_agent_that_ran_nothing_names_no_activity(self) -> None:
        """An agent recorded but never associated reports no work, rather than every activity."""
        s = _store()
        a = s.agent(agent_type="software", version="0.1.0")
        s.activity(node="PREPROCESS", step=None, parameters={})
        assert s.associations_of(a) == []


class TestEnvironments:
    """A host environment and a per-venv environment are their own record kind."""

    def test_a_host_environment_round_trips_by_id(self) -> None:
        """environment() returns an id get_environment() resolves back to the same record."""
        s = _store()
        eid = s.environment(
            kind="host",
            label="host",
            python_version="3.12.11",
            operating_system="macOS-15.0-arm64-arm-64bit",
            dependencies={"torch": "2.8.0"},
            senselab_version="1.3.1",
        )
        got = s.get_environment(eid)
        assert got.kind == "host"
        assert got.dependencies == {"torch": "2.8.0"}
        assert got.senselab_version == "1.3.1"

    def test_a_venv_environment_carries_a_digest_and_no_senselab_version(self) -> None:
        """A venv environment names a declared subset plus a digest, not the host's own version."""
        s = _store()
        eid = s.environment(
            kind="venv",
            label="crisperwhisper-cpu",
            python_version="3.12.11",
            dependencies={"torch": "2.14.0", "transformers": "5.16.1"},
            dependencies_digest="a" * 64,
        )
        got = s.get_environment(eid)
        assert got.kind == "venv"
        assert got.operating_system is None
        assert got.senselab_version is None
        assert got.dependencies_digest == "a" * 64

    def test_environments_are_enumerable_by_kind(self) -> None:
        """A reader collecting a run's venv environments must not have to know the ids in advance."""
        s = _store()
        host = s.environment(kind="host", label="host", python_version="3.12.11")
        venv = s.environment(kind="venv", label="hear", python_version="3.11.15")
        assert [e.id for e in s.environments("host")] == [host]
        assert [e.id for e in s.environments("venv")] == [venv]
        assert {e.id for e in s.environments()} == {host, venv}

    def test_two_venvs_of_the_same_backend_at_different_device_keys_are_distinct(self) -> None:
        """The resolved directory name -- which carries the device key -- is part of the id."""
        s = _store()
        cpu = s.environment(kind="venv", label="crisperwhisper-cpu", python_version="3.12.11")
        gpu = s.environment(kind="venv", label="crisperwhisper-cu128", python_version="3.12.11")
        assert cpu != gpu

    def test_round_trips_through_jsonl(self, tmp_path: Path) -> None:
        """write_jsonl / read_jsonl preserve every environment field."""
        s = _store()
        s.environment(
            kind="host",
            label="host",
            python_version="3.12.11",
            operating_system="macOS-15.0-arm64-arm-64bit",
            dependencies={"torch": "2.8.0", "numpy": "2.4.4"},
            senselab_version="1.3.1",
        )
        s.environment(
            kind="venv",
            label="hear",
            python_version="3.11.15",
            dependencies={"tensorflow": "2.21.0", "keras": "3.15.1"},
            dependencies_digest="b" * 64,
        )
        path = tmp_path / "prov.jsonl"
        s.write_jsonl(path)
        back = ProvStore.read_jsonl(path)
        assert back._environments == s._environments
        assert back.fingerprint() == s.fingerprint()

    def test_a_store_with_no_environment_lines_still_reads(self, tmp_path: Path) -> None:
        """A store that recorded no environment reads back with none, not an error."""
        s = _store()
        s.agent(agent_type="software", version="senselab 1.3.1")
        path = tmp_path / "prov.jsonl"
        s.write_jsonl(path)
        assert "environment" not in {json.loads(ln)["record"] for ln in path.read_text().splitlines()}
        back = ProvStore.read_jsonl(path)
        assert back.environments() == []

    def test_an_unknown_kind_is_refused_on_read(self, tmp_path: Path) -> None:
        """A hand-edited kind outside host/venv fails the read instead of loading quietly."""
        s = _store()
        s.environment(kind="host", label="host", python_version="3.12.11")
        path = tmp_path / "prov.jsonl"
        s.write_jsonl(path)
        _rewrite_records(path, "environment", lambda rec: rec.update(kind="container"))
        with pytest.raises(ValueError, match="container"):
            ProvStore.read_jsonl(path)

    def test_merge_unions_environments(self) -> None:
        """merge() carries environments across, the same as entities/activities/agents."""
        a = _store()
        a.environment(kind="host", label="host", python_version="3.12.11")
        b = _store()
        b.environment(kind="venv", label="yamnet", python_version="3.12.0")
        merged = ProvStore.merge([a, b])
        assert len(merged.environments()) == 2


class TestInvalidation:
    """Withdrawal keeps the entity."""

    def test_an_invalidated_entity_is_still_readable(self) -> None:
        """Invalidation marks an entity unusable without removing it."""
        s = _store()
        seg = s.entity(prov_type="artifact", extent=(7.9, 9.0), attributes={"speaker": "SPEAKER_00"})
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
        a.entity(prov_type="artifact", extent=(1.0, 2.0), attributes={})
        b.entity(prov_type="artifact", extent=(1.1, 1.4), attributes={"word": "hello"})
        assert ProvStore.merge([a, b]).fingerprint() == ProvStore.merge([b, a]).fingerprint()

    def test_repeating_a_relation_does_not_change_the_fingerprint(self) -> None:
        """A re-run node recording the same relation twice is a no-op."""
        s = _store()
        act = s.activity(node="PREPROCESS", step=None, parameters={})
        ent = s.entity(prov_type="artifact", extent=(1.0, 2.0), attributes={})
        s.was_generated_by(ent, act)
        before = s.fingerprint()
        s.was_generated_by(ent, act)
        assert s.fingerprint() == before

    def test_merging_a_store_with_itself_alone_is_idempotent(self) -> None:
        """merge([s]) carries the same content hash as s."""
        s = _store()
        act = s.activity(node="PREPROCESS", step=None, parameters={})
        ent = s.entity(prov_type="artifact", extent=(1.0, 2.0), attributes={})
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
        span = s.entity(prov_type="artifact", extent=(7.9, 8.5), attributes={"peak_over_floor_db": 31.4})
        verdict = s.entity(prov_type="artifact", extent=None, attributes={"value": "keep"})
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
        path.write_text(f'{FORMAT_LINE}\n{{"record": "banana"}}\n')
        with pytest.raises(ValueError, match="banana"):
            ProvStore.read_jsonl(path)

    @pytest.mark.parametrize("prov_type", sorted(get_args(PROV_TYPE)))
    def test_every_entity_type_round_trips(self, prov_type: str, tmp_path: Path) -> None:
        """Every type in :data:`PROV_TYPE` that the writer writes, the reader reads back."""
        s = _store()
        entity = s.entity(prov_type=cast(PROV_TYPE, prov_type), extent=None, attributes={"state": "uncertain"})
        path = tmp_path / "prov.jsonl"
        s.write_jsonl(path)
        back = ProvStore.read_jsonl(path)
        assert back.get_entity(entity).prov_type == prov_type
        assert back._entities == s._entities


class TestReadBackValidation:
    """What is enforced at write time also holds after a read."""

    def test_a_duplicated_relation_line_reads_back_as_one_triple(self, tmp_path: Path) -> None:
        """A file carrying the same relation twice must not resurrect the duplicate-triple divergence."""
        s = _store()
        act = s.activity(node="PREPROCESS", step=None, parameters={})
        ent = s.entity(prov_type="artifact", extent=(1.0, 2.0), attributes={})
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
        path.write_text(f'{FORMAT_LINE}\n{{"id": "span-abc"}}\n')
        with pytest.raises(ValueError, match="'record' key"):
            ProvStore.read_jsonl(path)

    def test_an_unknown_relation_is_refused_on_read(self, tmp_path: Path) -> None:
        """A relation outside PROV's own six fails the read instead of loading quietly."""
        s = _store()
        act = s.activity(node="PREPROCESS", step=None, parameters={})
        ent = s.entity(prov_type="artifact", extent=(1.0, 2.0), attributes={})
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

    def test_get_activity_returns_what_activity_recorded(self) -> None:
        """An entity's author node is reachable: generated_by -> get_activity -> .node."""
        store = ProvStore(run_id="t")
        act = store.activity(node="SPEECH", step="diarize", parameters={})
        eid = store.entity(prov_type="artifact", extent=(1.0, 2.0), attributes={})
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


class TestAtomicWrite:
    """``write_jsonl`` leaves the previous store or the new one whole, never a truncated file."""

    def _two_versions(self, tmp_path: Path) -> tuple[ProvStore, Path, bytes]:
        s = _store()
        s.entity(prov_type="artifact", extent=None, attributes={"name": "first"})
        target = tmp_path / "store.jsonl"
        s.write_jsonl(target)
        s.entity(prov_type="artifact", extent=None, attributes={"name": "second"})
        return s, target, target.read_bytes()

    def test_a_write_interrupted_before_the_rename_keeps_the_previous_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An fsync that dies (a preempted job) leaves the old file byte-identical and no temporary behind."""
        s, target, before = self._two_versions(tmp_path)

        def _killed(descriptor: int) -> None:
            raise KeyboardInterrupt("SIGTERM during the write")

        monkeypatch.setattr(os, "fsync", _killed)
        with pytest.raises(KeyboardInterrupt):
            s.write_jsonl(target)
        assert target.read_bytes() == before
        assert [p.name for p in tmp_path.iterdir()] == ["store.jsonl"]

    def test_a_completed_write_replaces_the_store_and_leaves_nothing_else(self, tmp_path: Path) -> None:
        """The new store reads back whole; no temporary file stays in the directory."""
        s, target, _ = self._two_versions(tmp_path)
        s.write_jsonl(target)
        names = {e.attributes["name"] for e in ProvStore.read_jsonl(target).entities("artifact")}
        assert names == {"first", "second"}
        assert [p.name for p in tmp_path.iterdir()] == ["store.jsonl"]

    def test_an_existing_store_keeps_its_permission_bits(self, tmp_path: Path) -> None:
        """A group-readable store stays group-readable after it is rewritten."""
        s, target, _ = self._two_versions(tmp_path)
        target.chmod(0o640)
        s.write_jsonl(target)
        assert target.stat().st_mode & 0o777 == 0o640


class Colour(enum.Enum):
    """An enum family member with a plain value."""

    RED = "red"
    ONE = 1


class Level(enum.IntEnum):
    """An enum family member that is also an int."""

    ONE = 1


_KEYS = st.text(alphabet="abfé\u0301$i", max_size=4).map(lambda k: unicodedata.normalize("NFC", k))
_FLOATS = st.floats() | st.sampled_from([float("inf"), float("-inf"), float("nan"), -0.0, 2.0**60, 1e300])
_JSON_SCALARS = (
    st.none() | st.booleans() | st.integers(-(2**70), 2**70) | _FLOATS | st.text(alphabet="aé\u0301$fNI", max_size=5)
)
_NUMPY_SCALARS = st.one_of(
    _FLOATS.map(numpy.float64),
    st.floats(width=32).map(numpy.float32),
    st.integers(-(2**63), 2**63 - 1).map(numpy.int64),
    st.booleans().map(numpy.bool_),
)
_FAMILIES = st.one_of(
    st.text(alphabet="ab/.", max_size=6).map(PurePosixPath),
    st.sampled_from(list(Colour) + list(Level)),
    st.decimals(allow_nan=True, allow_infinity=True, places=None).filter(lambda d: not d.is_snan()),
    st.binary(max_size=4),
    st.binary(max_size=4).map(bytearray),
    st.lists(st.integers(-5, 5), max_size=4).map(lambda v: numpy.array(v, dtype="<i8")),
    st.lists(_FLOATS, max_size=3).map(lambda v: numpy.array(v, dtype=">f8")),
    st.frozensets(st.integers(-3, 3) | st.text(alphabet="ab", max_size=2), max_size=3),
    st.sets(st.integers(-3, 3), max_size=3),
)
_PAYLOADS = st.recursive(
    _JSON_SCALARS | _NUMPY_SCALARS | _FAMILIES,
    lambda inner: (
        st.lists(inner, max_size=3) | st.lists(inner, max_size=3).map(tuple) | st.dictionaries(_KEYS, inner, max_size=3)
    ),
    max_leaves=10,
)


def _number(value: float | int) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return ("nonfinite", "nan" if math.isnan(value) else repr(value))
    if isinstance(value, int) and abs(value) > 2**53 - 1:
        return ("bigint", value)
    return ("n", Fraction(value))


def _oracle(value: object) -> object:
    """An independent identity, modulo ADR-006's equivalences only.

    Finite numbers compare by value (1 = 1.0 = numpy 1, -0.0 = 0), every NaN is one, text compares after NFC,
    a tuple is a list, bytes equal bytearray of the same content, a set equals a frozenset, an array compares
    by dtype name, shape and values whatever its byte order. Every family stays apart from JSON and from the
    others.
    """
    if isinstance(value, enum.Enum):
        return ("enum", f"{type(value).__module__}.{type(value).__qualname__}", _oracle(value.value))
    if isinstance(value, numpy.ndarray):
        return ("nd", value.dtype.name, value.shape, tuple(_oracle(v) for v in value.reshape(-1).tolist()))
    if isinstance(value, numpy.generic):
        return _oracle(value.item())
    if value is None or isinstance(value, bool):
        return ("c", value)
    if isinstance(value, (int, float)):
        return _number(value)
    if isinstance(value, str):
        return ("s", unicodedata.normalize("NFC", value))
    if isinstance(value, PurePath):
        return ("path", value.as_posix())
    if isinstance(value, Decimal):
        return ("dec", str(value))
    if isinstance(value, (bytes, bytearray)):
        return ("bytes", bytes(value))
    if isinstance(value, (set, frozenset)):
        return ("set", tuple(sorted(repr(_oracle(v)) for v in value)))
    if isinstance(value, (list, tuple)):
        return ("l", tuple(_oracle(v) for v in value))
    assert isinstance(value, dict)
    return ("d", frozenset((k, _oracle(v)) for k, v in value.items()))


def _id(value: object) -> str:
    return ProvStore(run_id="r").entity(prov_type="artifact", extent=None, attributes={"value": value})


@settings(max_examples=500, deadline=None)
@given(a=_PAYLOADS, b=_PAYLOADS)
def test_provenance_digest_is_total_deterministic_and_injective_over_the_declared_type(a: object, b: object) -> None:
    """Every value of the declared provenance type gets an id, the same id again, and its own id.

    Two values share an id exactly when an independent oracle calls them equal modulo ADR-006's equivalences.
    """
    for value in (a, b):
        assert _id(value) == _id(copy.deepcopy(value))
    assert (_id(a) == _id(b)) is (_oracle(a) == _oracle(b))


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (float("nan"), {"$f": "NaN"}),
        (float("-inf"), {"$f": "-Infinity"}),
        (float("-inf"), "-Infinity"),
        (2**60, {"$int": str(2**60)}),
        (2**60, str(2**60)),
        (2**60, float(2**60)),
        ({"$f": "NaN"}, {"$$f": "NaN"}),
        ([float("inf")], [float("-inf")]),
        (PurePosixPath("a/b"), "a/b"),
        (PurePosixPath("a/b"), {"$path": "a/b"}),
        (Colour.RED, "red"),
        (Level.ONE, 1),
        (Colour.ONE, Level.ONE),
        (Decimal("1.0"), Decimal("1.00")),
        (Decimal("1"), 1),
        (b"\x01", "01"),
        (b"\x01", [1]),
        (frozenset({1, 2}), [1, 2]),
        (numpy.array([1, 2]), [1, 2]),
        (numpy.array([1, 2], dtype="<i8"), numpy.array([1, 2], dtype="<i4")),
        (numpy.array([[1, 2]]), numpy.array([1, 2])),
    ],
)
def test_tokens_never_collide_with_recorded_values_or_each_other(a: object, b: object) -> None:
    """A token is never the id of an ordinary object or string, of another family, or of another shape or dtype."""
    assert _id(a) != _id(b)


class _Opaque:
    """A type outside the declared provenance value type."""


@pytest.mark.parametrize(
    ("attributes", "path", "type_name"),
    [
        ({"evidence": {"window": _Opaque()}}, "attributes.evidence.window", "_Opaque"),
        ({"values": [1, {"x": complex(1, 2)}]}, "attributes.values[1].x", "builtins.complex"),
        ({"when": numpy.datetime64("2026-10-31")}, "attributes.when", "numpy.datetime64"),
        ({"rows": numpy.array([(1, 2.0)], dtype="i4,f8")}, "attributes.rows", "numpy.ndarray"),
        ({"ordered": collections.OrderedDict(a=1)}, "attributes.ordered", "collections.OrderedDict"),
        ({1: "x"}, "attributes[1]", "builtins.int"),
        ({"\u00e9": 1, "e\u0301": 2}, "attributes.e\u0301", "builtins.str"),
    ],
)
def test_an_undeclared_value_raises_a_typed_error_at_the_boundary(attributes: dict, path: str, type_name: str) -> None:
    """`ProvStore.entity` refuses a value outside the declared type with `UnrecordableValue` naming path and type."""
    with pytest.raises(provenance.UnrecordableValue) as caught:
        ProvStore(run_id="r").entity(prov_type="artifact", extent=None, attributes=attributes)
    assert caught.value.path == path and caught.value.type_name.endswith(type_name)
    assert path in str(caught.value) and type_name in str(caught.value)


def test_cycles_and_depth_raise_the_typed_error_through_the_shared_guard() -> None:
    """A container holding itself, recorded nesting past `MAX_DEPTH`, or tokens pushing past it, are typed errors."""
    loop: dict[str, object] = {}
    loop["self"] = loop
    deep: object = 1.0
    for _ in range(200):
        deep = [deep]
    edge: object = float("-inf")
    for _ in range(62):
        edge = [edge]
    for attributes in ({"loop": loop}, {"deep": deep}, {"edge": edge}):
        with pytest.raises(provenance.UnrecordableValue, match="contains itself|deeper"):
            ProvStore(run_id="r").entity(prov_type="artifact", extent=None, attributes=attributes)


def test_a_level_measurement_of_silence_gets_an_id() -> None:
    """The regression: a measurement entity carrying `-inf` dB (digital silence) is recorded with an id."""
    store = ProvStore(run_id="r")
    first = store.entity(prov_type="artifact", extent=(0.0, 1.0), attributes={"level_dbfs": float("-inf")})
    again = ProvStore(run_id="r").entity(prov_type="artifact", extent=(0.0, 1.0), attributes={"level_dbfs": -math.inf})
    other = store.entity(prov_type="artifact", extent=(0.0, 1.0), attributes={"level_dbfs": -120.0})
    assert first == again != other


def test_a_real_file_entity_gets_an_id(tmp_path: Path) -> None:
    """`mtime_ns` stays an integer stamp beyond 2**53; it is tokenised for the digest, not rewritten."""
    target = tmp_path / "a.wav"
    target.write_bytes(b"x")
    attributes = {"path": str(target), **provenance.file_attributes(target)}
    assert attributes[provenance.MTIME_KEY] == target.stat().st_mtime_ns > 2**53
    ProvStore(run_id="r").entity(prov_type="artifact", extent=None, attributes=attributes)


@settings(max_examples=300, deadline=None)
@given(
    attributes=st.dictionaries(_KEYS, _PAYLOADS, max_size=3), parameters=st.dictionaries(_KEYS, _PAYLOADS, max_size=3)
)
def test_write_and_read_are_inverse_and_ids_recompute_from_the_file(
    tmp_path_factory: pytest.TempPathFactory, attributes: dict[str, object], parameters: dict[str, object]
) -> None:
    """Whatever the store records, the file reads back to equal records whose ids recompute, and writes back the same.

    Every record the writer accepts the reader accepts: they build records through one validator.
    """
    store = ProvStore(run_id="run")
    eid = store.entity(prov_type="artifact", extent=(0.0, 1.5), attributes=attributes)
    aid = store.activity(node="N", step=None, parameters=parameters, started="t0")
    store.agent(agent_type="software", version="1")
    store.environment(kind="host", label="h", python_version="3.12", dependencies={"numpy": "2"})
    store.was_generated_by(eid, aid)
    path = tmp_path_factory.mktemp("prov") / "prov.jsonl"
    store.write_jsonl(path)
    back = ProvStore.read_jsonl(path)
    assert back.records() == store.records() and back.relations() == store.relations()
    assert all(record.minted_id() == record.id for record in back.records())
    again = path.with_name("again.jsonl")
    back.write_jsonl(again)
    assert again.read_bytes() == path.read_bytes()


_PROVENANCE_1_LINES = [
    '{"record": "entity", "id": "span-0123456789abcdef", "prov_type": "span", "extent": [0, 1.5], "attributes": {}}',
    '{"record": "relation", "relation": "wasGeneratedBy", "source": "span-0123456789abcdef", '
    '"target": "act-0123456789abcdef"}',
]


@pytest.mark.parametrize(
    ("first", "found"),
    [
        (None, None),
        ('{"format": "provenance@1.1", "record": "format"}', "provenance@1.1"),
        ('{"format": "provenance@2.1", "record": "format"}', "provenance@2.1"),
        ('{"format": "provenance@2.0", "record": "format", "extra": 1}', "provenance@2.0"),
    ],
)
def test_a_file_without_the_provenance_2_format_record_is_refused_typed(
    tmp_path: Path, first: str | None, found: str | None
) -> None:
    """A `provenance@1` file, a `provenance@1.1` file, or any other first line is refused, naming the migration."""
    path = tmp_path / "old.jsonl"
    path.write_text("\n".join([*([first] if first else []), *_PROVENANCE_1_LINES]) + "\n")
    with pytest.raises(provenance.ProvenanceFormatRefused) as refused:
        ProvStore.read_jsonl(path)
    assert refused.value.found == found and refused.value.path == str(path)
    assert "migration-ledger.md" in str(refused.value) and "re-run" in str(refused.value)
    if first is None:
        assert "1.x provenance store (provenance@1)" in str(refused.value)


def test_an_empty_file_is_refused_typed(tmp_path: Path) -> None:
    """An empty file carries no format record and is not read as an empty store."""
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(provenance.ProvenanceFormatRefused):
        ProvStore.read_jsonl(path)


def test_a_kind_entity_is_refused_by_the_writer_and_the_reader(tmp_path: Path) -> None:
    """`kind` is not in `PROV_TYPE`: `entity()` refuses it, and so does `read_jsonl` on a 2.0 line carrying it."""
    assert "kind" not in get_args(PROV_TYPE)
    with pytest.raises(ValueError, match="unknown prov_type 'kind'"):
        _store().entity(prov_type=cast(PROV_TYPE, "kind"), extent=None, attributes={"state": "uncertain"})
    store = _store()
    store.entity(prov_type="artifact", extent=None, attributes={"state": "uncertain"})
    path = tmp_path / "prov.jsonl"
    store.write_jsonl(path)
    _rewrite_records(path, "entity", lambda rec: rec.update(prov_type="kind"))
    with pytest.raises(ValueError, match="unknown prov_type 'kind'"):
        ProvStore.read_jsonl(path)


def test_every_store_the_writer_writes_is_provenance_2_and_every_record_carries_its_run_id(
    tmp_path: Path,
) -> None:
    """The writer has one format, and every record it writes carries the run it was minted in."""
    store = _store()
    store.entity(prov_type="artifact", extent=(0.0, 1.0), attributes={"a": 1})
    path = tmp_path / "prov.jsonl"
    store.write_jsonl(path)
    assert path.read_text().splitlines()[0] == FORMAT_LINE
    assert all(record.run_id == "run-1" for record in ProvStore.read_jsonl(path).records())


_RECORD_FIELDS: dict[type, dict[str, object]] = {
    Entity: {"prov_type": "artifact", "extent": None},
    Activity: {"node": "N", "step": None},
    Agent: {"agent_type": "software", "version": "1"},
    provenance.Environment: {"kind": "host", "label": "h", "python_version": "3.12"},
}


@pytest.mark.parametrize("record_type", list(_RECORD_FIELDS), ids=lambda t: t.__name__)
def test_no_record_can_be_built_without_a_run_id(record_type: type) -> None:
    """`run_id` is a required field of every record type: omitting it or passing None is refused."""
    fields = _RECORD_FIELDS[record_type]
    with pytest.raises(TypeError, match="run_id"):
        record_type(id="", **fields)
    with pytest.raises(ValueError, match="run_id must be text"):
        record_type(id="", run_id=None, **fields)
    record = record_type(id="", run_id="r", **fields)
    assert record.run_id == "r" and record.id == record.minted_id()


@pytest.mark.parametrize("record_type", list(_RECORD_FIELDS), ids=lambda t: t.__name__)
def test_revised_keeps_the_run_and_mints_the_id_of_the_new_content(record_type: type) -> None:
    """`revised` returns a record of the same run whose id is its own content's; `id` and `run_id` cannot change."""
    record = record_type(id="", run_id="r", **_RECORD_FIELDS[record_type])
    name = {Entity: "extent", Activity: "step", Agent: "version", provenance.Environment: "label"}[record_type]
    value = {"extent": (0.0, 1.0), "step": "s", "version": "2", "label": "other"}[name]
    changed = provenance.revised(record, **{name: value})
    assert changed.run_id == "r" and changed.id == changed.minted_id() and changed.id != record.id
    ProvStore.merge([])._hold(changed)
    for frozen in ("id", "run_id"):
        with pytest.raises(ValueError, match="cannot revise"):
            provenance.revised(record, **{frozen: "x"})


@settings(max_examples=300, deadline=None)
@given(attributes=st.dictionaries(_KEYS, _PAYLOADS, max_size=3), extra=_PAYLOADS)
def test_the_store_records_again_every_form_it_returns(
    tmp_path_factory: pytest.TempPathFactory, attributes: dict[str, object], extra: object
) -> None:
    """A read-back's attributes, whole, nested in a new mapping, or revised, record the same values; files agree.

    Recording `get_entity(x).attributes` mints `x` again; a new mapping holding read-back objects records them as
    they are; `revised` adds a value as the store would record it; write, read and write give identical lines.
    """
    store = ProvStore(run_id="run")
    eid = store.entity(prov_type="artifact", extent=(0.0, 1.0), attributes=attributes)
    path = tmp_path_factory.mktemp("prov") / "prov.jsonl"
    store.write_jsonl(path)
    back = ProvStore.read_jsonl(path, run_id="run")
    held = cast(dict[str, object], back.get_entity(eid).attributes)
    assert back.entity(prov_type="artifact", extent=(0.0, 1.0), attributes=held) == eid
    nested = back.entity(prov_type="artifact", extent=None, attributes={"inner": held, "list": [held]})
    tokens = provenance.normalised(back.get_entity(nested).attributes)
    assert tokens == provenance.normalised({"inner": attributes, "list": [attributes]})
    with_extra = provenance.revised(held, added=extra)
    direct = back.entity(prov_type="artifact", extent=None, attributes={**dict(attributes), "added": extra})
    assert back.entity(prov_type="artifact", extent=None, attributes=with_extra) == direct
    again = path.with_name("again.jsonl")
    back.write_jsonl(again)
    third = path.with_name("third.jsonl")
    ProvStore.read_jsonl(again, run_id="run").write_jsonl(third)
    assert again.read_bytes() == third.read_bytes()
    assert all(record.minted_id() == record.id for record in ProvStore.read_jsonl(third).records())


def test_with_entity_attributes_keeps_order_and_relations_and_remints_the_edited_entity() -> None:
    """`testing.provenance.with_entity_attributes` changes one entity's attributes; the copy is a 2.0 store."""
    from senselab_core.testing.provenance import with_entity_attributes

    store = _store()
    first = store.entity(prov_type="artifact", extent=(0.0, 1.0), attributes={"signal": "enhanced", "n": [1]})
    second = store.entity(prov_type="artifact", extent=(1.0, 2.0), attributes={})
    act = store.activity(node="N", step=None, parameters={})
    store.was_generated_by(first, act)
    edited = with_entity_attributes(store, first, signal="plain")
    new_first, held_second = edited.entities()
    assert held_second == store.get_entity(second) and new_first.id != first
    assert new_first.attributes == {"signal": "plain", "n": [1]} and new_first.minted_id() == new_first.id
    assert edited.relations() == [("wasGeneratedBy", new_first.id, act)]
    assert store.get_entity(first).attributes["signal"] == "enhanced"


def _decoded_type(value: object) -> type | None:
    """The type a caller reads back for a value of a token family, or None for a JSON value."""
    if isinstance(value, enum.Enum):
        return provenance.EnumMember
    if isinstance(value, PurePath):
        return Path
    if isinstance(value, numpy.ndarray):
        return numpy.ndarray
    if isinstance(value, (set, frozenset)):
        return frozenset
    if isinstance(value, (bytes, bytearray)):
        return bytes
    if isinstance(value, Decimal):
        return Decimal
    if isinstance(value, float) and not math.isfinite(value):
        return float
    if isinstance(value, int) and not isinstance(value, bool) and abs(value) > 2**53 - 1:
        return int
    return None


def _read_as_decoded(read: object, value: object) -> bool:
    """Whether `read` is `value` decoded: the same containers, and each family as its decoded type, never a token."""
    family = _decoded_type(value)
    if family is not None:
        return isinstance(read, family)
    if isinstance(value, dict):
        return (
            isinstance(read, dict)
            and set(read) == set(value)
            and all(_read_as_decoded(read[k], v) for k, v in value.items())
        )
    if isinstance(value, (list, tuple)):
        return (
            isinstance(read, list)
            and len(read) == len(value)
            and all(_read_as_decoded(r, v) for r, v in zip(read, value))
        )
    return not isinstance(read, (dict, list))


@settings(max_examples=300, deadline=None)
@given(value=_PAYLOADS, key=_KEYS)
def test_every_read_path_hands_back_the_decoded_value_never_a_token(
    tmp_path_factory: pytest.TempPathFactory, value: object, key: str
) -> None:
    """On every read path a caller reads each family decoded, as its decoded type, and records it again losslessly."""
    store = ProvStore(run_id="run")
    eid = store.entity(prov_type="artifact", extent=None, attributes={key: value})
    aid = store.activity(node="N", step=None, parameters={key: value})
    path = tmp_path_factory.mktemp("prov") / "prov.jsonl"
    store.write_jsonl(path)
    back = ProvStore.read_jsonl(path, run_id="run")
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    (line,) = [r for r in raw if r.get("id") == eid]
    reads = {
        "get_entity": store.get_entity(eid).attributes[key],
        "entities": store.entities("artifact")[0].attributes[key],
        "get_activity": store.get_activity(aid).parameters[key],
        "read_jsonl": back.get_entity(eid).attributes[key],
        "read_jsonl activities": back.activities()[0].parameters[key],
        "recorded": provenance.recorded({key: value})[key],
        "plain_values": provenance.plain_values(line["attributes"])[key],
        "revised": provenance.revised(store.get_entity(eid).attributes)[key],
    }
    expected = provenance.normalised(value)
    for how, read in reads.items():
        assert _read_as_decoded(read, value), f"{how}: {read!r} for {value!r}"
        assert provenance.normalised(read) == expected, how


def _compares_by_value(value: object) -> bool:
    """Whether Python equality is reflexive on `value`'s read-back: no NaN and no array anywhere in it."""
    if isinstance(value, numpy.ndarray):
        return False
    if isinstance(value, (float, numpy.floating)):
        return not math.isnan(value)
    if isinstance(value, Decimal):
        return not value.is_nan()
    if isinstance(value, dict):
        return all(_compares_by_value(v) for v in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return all(_compares_by_value(v) for v in value)
    if isinstance(value, enum.Enum):
        return _compares_by_value(value.value)
    return True


def test_an_enum_read_back_equals_another_read_back_of_the_same_member() -> None:
    """`recorded(x) == recorded(x)` for an enum member, alone and nested; another type or value is unequal."""
    assert recorded(Colour.RED) == recorded(Colour.RED)
    assert recorded({"e": [Colour.ONE]}) == recorded({"e": [Colour.ONE]})
    assert hash(recorded(Colour.RED)) == hash(recorded(Colour.RED))
    assert recorded(Colour.ONE) != recorded(Level.ONE) and recorded(Colour.ONE) != recorded(Colour.RED)


@settings(max_examples=300, deadline=None)
@given(value=_PAYLOADS)
def test_every_read_back_equals_another_read_back_of_the_same_value(value: object) -> None:
    """Two read-backs of one value are equal for every family; arrays and NaN compare as numpy and IEEE do."""
    assume(_compares_by_value(value))
    assert recorded(value) == recorded(copy.deepcopy(value))


@pytest.mark.logic
def test_recorded_content_cannot_drift_from_its_id(tmp_path: Path) -> None:
    """Mutating what was passed in, or what `get_entity` returns, cannot change a recorded entity."""
    store = ProvStore(run_id="r")
    attributes: dict[str, object] = {"ev": {"n": [1, 2, 3]}}
    extent = [0.0, 1.0]
    eid = store.entity(prov_type="artifact", extent=extent, attributes=attributes)  # type: ignore[arg-type]
    attributes["ev"]["n"].append(4)  # type: ignore[index]
    extent[1] = 99.0
    entity = store.get_entity(eid)
    assert entity.attributes == {"ev": {"n": [1, 2, 3]}} and entity.extent == (0.0, 1.0)
    for write in (
        lambda: entity.attributes.__setitem__("x", 1),  # type: ignore[index]
        lambda: entity.attributes["ev"].__setitem__("x", 1),
        lambda: entity.attributes["ev"]["n"].append(4),
    ):
        with pytest.raises((TypeError, AttributeError)):
            write()
    assert entity.minted_id() == eid
    act = store.get_activity(store.activity(node="N", step=None, parameters={"p": [1]}))
    with pytest.raises(TypeError):
        act.parameters["p"] = 2  # type: ignore[index]
    timed = ProvStore(run_id="r")
    timed.activity(node="N", step=None, parameters={"p": [1]}, started="2026-01-01T00:00:00Z")
    one, other = ProvStore.merge([store, timed]), ProvStore.merge([timed, store])
    assert one.records() == other.records() and one.get_activity(act.id).started == "2026-01-01T00:00:00Z"


@pytest.mark.logic
def test_jsonl_is_lossless_and_ids_recompute(tmp_path: Path) -> None:
    """Arrays, paths, non-finite floats, enums, sets and bytes survive a write and read; ids recompute."""
    import enum

    class E(enum.Enum):
        A = 1

    one, two = numpy.arange(2000), numpy.arange(2000)
    two[1000] = -1
    store = ProvStore(run_id="r")
    for array in (one, two):
        store.entity(
            prov_type="artifact",
            extent=None,
            attributes={
                "a": array,
                "p": PurePosixPath("x/y"),
                "t": (1, 2),
                "f": [float("nan"), float("-inf")],
                "e": E.A,
                "s": {1, 2},
                "b": b"ab",
            },
        )
    path = tmp_path / "prov.jsonl"
    store.write_jsonl(path)
    lines = path.read_bytes().splitlines()
    assert all(json.loads(line) is not None for line in lines) and b"NaN" not in b"".join(lines).replace(b'"NaN"', b"")
    back = ProvStore.read_jsonl(path)
    assert back.records() == store.records() and len(back.entities()) == 2
    assert all(record.minted_id() == record.id for record in back.records())
    again = tmp_path / "again.jsonl"
    back.write_jsonl(again)
    assert again.read_bytes() == path.read_bytes()


@pytest.mark.logic
def test_a_line_without_a_run_id_is_refused(tmp_path: Path) -> None:
    """A 2.0 line whose `run_id` is null is refused; `revised` keeps the run and re-mints the id."""
    store = ProvStore(run_id="r")
    eid = store.entity(prov_type="artifact", extent=(0.0, 1.0), attributes={"a": 1})
    edited = revised(store.get_entity(eid), attributes={"a": 2})
    assert edited.run_id == "r" and edited.id != eid and edited.id == edited.minted_id()
    path = tmp_path / "edited.jsonl"
    line = {**store.get_entity(eid).as_record(), "run_id": None}
    path.write_bytes(
        b"".join(jcs_dumps(row) + b"\n" for row in [{"format": "provenance@2.0", "record": "format"}, line])
    )
    with pytest.raises(ValueError, match="run_id must be text"):
        ProvStore.read_jsonl(path)
    store.write_jsonl(path)
    written = path.read_text()
    for bad, refusal in (("[NaN, 1.0]", "non-finite"), ("[0, 1e400]", "finite")):
        path.write_text(written.replace('"extent": [0.0, 1.0]', f'"extent": {bad}'))
        with pytest.raises(ValueError, match=refusal):
            ProvStore.read_jsonl(path)


@pytest.mark.logic
def test_write_refuses_what_read_refuses() -> None:
    """Unknown types and kinds, non-mapping attributes and malformed extents are refused when written."""
    store = ProvStore(run_id="r")
    with pytest.raises(ValueError):
        store.entity(prov_type="bogus", extent=None, attributes={})  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        store.agent(agent_type="bogus", version="1")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        store.environment(kind="bogus", label="l", python_version="3")  # type: ignore[arg-type]
    with pytest.raises(UnrecordableValue):
        store.entity(prov_type="artifact", extent=None, attributes=[1])  # type: ignore[arg-type]
    for extent in (("a", 1.0), (0.0, float("nan")), (2.0, 1.0), (0.0,)):
        with pytest.raises(ValueError):
            store.entity(prov_type="artifact", extent=extent, attributes={})  # type: ignore[arg-type]


@pytest.mark.logic
def test_a_ten_million_element_array_normalises_as_bytes_not_elements() -> None:
    """Arrays normalise as canonical `.npy` bytes, so memory stays a small multiple of the data, not of its elements."""
    import tracemalloc

    a = numpy.arange(10_000_000, dtype=numpy.float64)
    tracemalloc.start()
    try:
        token = normalised(a)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert isinstance(token, dict) and list(token) == ["$nd"]
    assert peak < 4 * a.nbytes


@pytest.mark.logic
def test_normalisation_is_total_and_typed() -> None:
    """An integer too long to write, and empty arrays of refused dtypes, raise `UnrecordableValue`."""
    with pytest.raises(UnrecordableValue):
        normalised(10**5000)
    for dtype in ("c16", "M8[ns]", "S3", "O", [("a", "i4"), ("b", "i4")]):
        with pytest.raises(UnrecordableValue):
            normalised(numpy.zeros(0, dtype=dtype))
