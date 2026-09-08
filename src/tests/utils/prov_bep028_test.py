"""The BEP028 serializer: identity round-trips, relations resolve, and nothing is invented."""

import json
from pathlib import Path

import pytest

from senselab.utils.prov_bep028 import (
    BEP028_CONTEXT,
    SHA256_ALGORITHM,
    bids_uri,
    check_graph,
    convert_store_file,
    store_id,
    to_bep028_graph,
    write_bep028_files,
)
from senselab.utils.prov_store import CHECKSUM_KEY, CHECKSUM_UNRESOLVED_KEY, ProvStore

DIGEST = "b" * 64


@pytest.fixture
def store() -> ProvStore:
    """A store exercising every record kind, both checksum outcomes and every relation."""
    store = ProvStore(run_id="run-1")
    software = store.agent(agent_type="software", version="senselab 1.3.1a45.dev542")
    model = store.agent(agent_type="model", model_id="MIT/ast", commit_sha="f" * 40)
    unpinned = store.agent(
        agent_type="model", model_id="https://tfhub.dev/google/yamnet/1", unresolved_reason="TF-Hub URL pin"
    )
    admit = store.activity(node="ADMIT", step=None, parameters={"audio_file": "a.wav"})
    step = store.activity(
        node="PREPROCESS", step="gammatone", parameters={"hop_s": 0.01}, started="2026-09-08T00:00:00"
    )
    recording = store.entity(
        prov_type="stream", extent=(0.0, 1.5), attributes={"path": "a.wav", CHECKSUM_KEY: DIGEST, "size_bytes": 9}
    )
    lost = store.entity(
        prov_type="measurement",
        extent=None,
        attributes={"path": "derivatives/gone.npz", CHECKSUM_UNRESOLVED_KEY: "file not found"},
    )
    span = store.entity(prov_type="span", extent=(0.2, 0.4), attributes={"kind": "speech"})
    store.was_generated_by(recording, admit)
    store.was_attributed_to(recording, software)
    store.was_associated_with(admit, software)
    store.was_associated_with(step, model)
    store.was_associated_with(step, unpinned)
    store.used(step, recording)
    store.was_generated_by(lost, step)
    store.was_derived_from(span, recording)
    store.was_invalidated_by(span, step)
    return store


def test_identity_round_trips(store: ProvStore) -> None:
    """Every id in the graph reverses to the store id it came from."""
    graph = to_bep028_graph(store)
    seen = {store_id(node["Id"]) for nodes in graph["Records"].values() for node in nodes}
    expected = {e.id for e in store.entities()} | {a.id for a in store.activities()} | {g.id for g in store.agents()}
    assert seen == expected


@pytest.mark.parametrize("identifier", ["stream-abc", "measurement-abc", "act-abc", "agent-abc"])
@pytest.mark.parametrize("dataset", ["", "triage"])
def test_bids_uri_is_reversible(identifier: str, dataset: str) -> None:
    """The mapping is total and mechanical in both directions."""
    uri = bids_uri(identifier, dataset=dataset)
    assert uri.startswith(f"bids:{dataset}:prov#")
    assert store_id(uri) == identifier


def test_store_id_refuses_a_foreign_uri() -> None:
    """A URI the serializer did not produce is an error, not a silent pass-through."""
    with pytest.raises(ValueError, match="not a BIDS provenance URI"):
        store_id("https://example.org/thing")


def test_graph_passes_its_own_checks(store: ProvStore) -> None:
    """The serializer's output is JSON-LD whose context resolves every term it uses."""
    assert check_graph(to_bep028_graph(store)) == []


def test_graph_survives_a_json_round_trip(store: ProvStore) -> None:
    """The document is plain JSON; nothing in it needs a custom encoder."""
    graph = json.loads(json.dumps(to_bep028_graph(store)))
    assert graph["@context"][0] == BEP028_CONTEXT
    assert check_graph(graph) == []


def test_files_and_other_entities_are_separated(store: ProvStore) -> None:
    """An entity with a path is a BEP028 ``Files`` object; one without is a ``prov:Entity``."""
    records = to_bep028_graph(store)["Records"]
    assert {node["AtLocation"] for node in records["Files"]} == {"a.wav", "derivatives/gone.npz"}
    assert all("AtLocation" not in node for node in records["prov:Entity"])


def test_a_digest_becomes_an_spdx_checksum(store: ProvStore) -> None:
    """The SHA-256 is emitted as BEP028 requires it."""
    records = to_bep028_graph(store)["Records"]
    [recording] = [node for node in records["Files"] if node["AtLocation"] == "a.wav"]
    assert recording["Checksum"] == [{"ChecksumAlgorithm": SHA256_ALGORITHM, "ChecksumValue": DIGEST}]


def test_a_missing_digest_states_the_reason_and_emits_no_checksum(store: ProvStore) -> None:
    """BEP028's Checksum cannot say "unknown", so nothing is put there and the reason is carried."""
    records = to_bep028_graph(store)["Records"]
    [lost] = [node for node in records["Files"] if node["AtLocation"] == "derivatives/gone.npz"]
    assert "Checksum" not in lost
    assert lost["ChecksumUnresolvedReason"] == "file not found"


def test_an_unresolved_model_commit_omits_the_version(store: ProvStore) -> None:
    """A required BEP028 field is left out rather than filled with something untrue."""
    software = to_bep028_graph(store)["Records"]["Software"]
    [unpinned] = [node for node in software if node["Label"].startswith("https://tfhub.dev")]
    assert "Version" not in unpinned
    assert unpinned["VersionUnresolvedReason"] == "TF-Hub URL pin"


def test_a_resolved_model_commit_is_the_version(store: ProvStore) -> None:
    """A model agent's version is its resolved commit, never a ref."""
    software = to_bep028_graph(store)["Records"]["Software"]
    [model] = [node for node in software if node["Label"] == "MIT/ast"]
    assert model["Version"] == "f" * 40


def test_the_software_agent_splits_into_a_name_and_a_version(store: ProvStore) -> None:
    """``senselab <version>`` becomes BEP028's separate Label and Version."""
    software = to_bep028_graph(store)["Records"]["Software"]
    [senselab] = [node for node in software if node["Label"] == "senselab"]
    assert senselab["Version"] == "1.3.1a45.dev542"


def test_every_relation_is_emitted(store: ProvStore) -> None:
    """All six store relations reach the graph, ``wasInvalidatedBy`` included."""
    records = to_bep028_graph(store)["Records"]
    terms = {term for nodes in records.values() for node in nodes for term in node}
    assert {"GeneratedBy", "Used", "AssociatedWith", "AttributedTo", "DerivedFrom", "InvalidatedBy"} <= terms


def test_an_activity_is_not_claimed_to_be_a_command(store: ProvStore) -> None:
    """``Command`` is REQUIRED and we have none, so it is null and a Description says why."""
    [step] = [node for node in to_bep028_graph(store)["Records"]["Activities"] if node["Label"].endswith("gammatone")]
    assert step["Command"] is None
    assert "not a shell command" in step["Description"]
    assert step["Parameters"] == {"hop_s": 0.01}
    assert step["StartedAtTime"] == "2026-09-08T00:00:00"


def test_no_environment_is_invented(store: ProvStore) -> None:
    """The store records no environment, so the graph claims none."""
    assert "Environments" not in to_bep028_graph(store)["Records"]


def test_check_graph_catches_a_dangling_reference(store: ProvStore) -> None:
    """A relation naming an id no node declares is a problem, not a silent broken edge."""
    graph = to_bep028_graph(store)
    graph["Records"]["Files"][0]["GeneratedBy"] = ["bids::prov#act-nothere"]
    assert any("is not a node in the graph" in problem for problem in check_graph(graph))


def test_check_graph_catches_a_term_that_would_be_dropped(store: ProvStore) -> None:
    """An undeclared prefix expands to nothing, so JSON-LD would drop the key silently."""
    graph = to_bep028_graph(store)
    graph["Records"]["Files"][0]["nope:Thing"] = 1
    assert any("would be dropped" in problem for problem in check_graph(graph))


def test_check_graph_requires_a_vocabulary(store: ProvStore) -> None:
    """Without @vocab, every store attribute would vanish from the graph."""
    graph = to_bep028_graph(store)
    graph["@context"] = [BEP028_CONTEXT, {}]
    assert any("@vocab" in problem for problem in check_graph(graph))


def test_check_graph_catches_a_duplicate_id(store: ProvStore) -> None:
    """Two nodes sharing an Id would merge into one on load."""
    graph = to_bep028_graph(store)
    graph["Records"]["Files"].append(dict(graph["Records"]["Files"][0]))
    assert any("declared twice" in problem for problem in check_graph(graph))


def test_check_graph_catches_a_malformed_checksum(store: ProvStore) -> None:
    """A digest that is not 64 lowercase hex characters is not a SHA-256."""
    graph = to_bep028_graph(store)
    graph["Records"]["Files"][0]["Checksum"] = [{"ChecksumAlgorithm": SHA256_ALGORITHM, "ChecksumValue": "nope"}]
    assert any("64 lowercase hex" in problem for problem in check_graph(graph))


def test_a_written_store_converts_without_the_pipeline(store: ProvStore, tmp_path: Path) -> None:
    """An existing run directory converts from its store.jsonl alone."""
    path = tmp_path / "store.jsonl"
    store.write_jsonl(path)
    assert convert_store_file(path) == to_bep028_graph(store)


def test_the_split_files_partition_the_graph(store: ProvStore, tmp_path: Path) -> None:
    """The BEP028-named files hold each record kind once, under the same context."""
    graph = to_bep028_graph(store, label="triage")
    written = write_bep028_files(graph, tmp_path / "prov", label="triage")
    assert [path.name for path in written] == ["prov-triage_io.json", "prov-triage_act.json", "prov-triage_soft.json"]
    bodies = [json.loads(path.read_text()) for path in written]
    assert all(body["@context"] == graph["@context"] for body in bodies)
    keys = [key for body in bodies for key in body if key != "@context"]
    assert sorted(keys) == ["Activities", "Files", "Software", "prov:Entity"]
