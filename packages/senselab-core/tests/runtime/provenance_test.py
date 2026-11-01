"""`ProvStore`: one writer's PROV-O JSON-LD document per writer, and a run's provenance as their union by `@id`."""

import itertools
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy
import pytest
from senselab_core.kernel import prov
from senselab_core.kernel.prov import CONTEXT_URL, Association, Generation, Usage, Vocabulary
from senselab_core.runtime.provenance import (
    PROV_DIR,
    ConflictingNode,
    DanglingReference,
    MalformedDocument,
    ProvStore,
)
from senselab_core.runtime.vocabulary import core_vocabulary
from senselab_worker.canonical import jcs_dumps, jcs_loads

pytestmark = pytest.mark.logic

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)
INPUT = "content:" + "1" * 64
OUTPUT = "output:" + "2" * 64
KEY = {"task": "asr", "version": "1", "inputs": {"audio": INPUT}}


def _prop(term: str, domain: str, rng: str) -> dict[str, Any]:
    return {"@id": term, "@type": "rdf:Property", "rdfs:domain": {"@id": domain}, "rdfs:range": {"@id": rng}}


VOCAB = Vocabulary.from_document(
    {
        "@context": CONTEXT_URL,
        "@graph": [
            {"@id": "senselab:", "@type": "owl:Ontology", "owl:versionInfo": "senselab-vocab@1.0"},
            _prop("senselab:context", "prov:Activity", "rdf:JSON"),
            _prop("senselab:label", "prov:Entity", "xsd:string"),
            _prop("senselab:waveform", "prov:Entity", "senselab:File"),
            _prop("senselab:loudness", "prov:Entity", "xsd:double"),
        ],
    },
    base=core_vocabulary(),
)


def _agent(source: str = "hf") -> prov.Record:
    return prov.agent(VOCAB, types=["senselab:SoftwareAgent"], attributes={"senselab:modelSource": source})


def _execution(context: dict[str, object], *, started: datetime = T0) -> prov.Record:
    return prov.activity(
        VOCAB,
        attributes={"senselab:logicalKey": KEY, "senselab:context": context},
        used=[Usage(INPUT, port="audio", time=started)],
        generated=[Generation(OUTPUT, port="transcript", time=started + timedelta(seconds=1))],
        associations=[Association(_agent().id, node_id="asr")],
        started=started,
        ended=started + timedelta(seconds=1),
    )


def _entities() -> list[prov.Record]:
    return [prov.entity(INPUT, VOCAB), prov.entity(OUTPUT, VOCAB, attributes={"senselab:label": "transcript"})]


def _run(tmp_path: Path, *stores: ProvStore) -> Path:
    run = tmp_path / "run"
    run.mkdir(exist_ok=True)
    for store in stores:
        store.write(run)
    return run


def _document_path(run: Path, store: ProvStore) -> Path:
    return run / PROV_DIR / f"{store.writer}.jsonld"


def _rewrite(path: Path, edit: Any) -> None:  # noqa: ANN401 - a callable over the parsed document
    doc = jcs_loads(path.read_bytes())
    edit(doc)
    path.write_bytes(jcs_dumps(doc))


def test_one_key_two_contexts_two_activities(tmp_path: Path) -> None:
    """One logical key run in two execution contexts is two activities, each carrying the key as an attribute."""
    cpu, gpu = _execution({"device": "cpu"}), _execution({"device": "cuda"})
    store = ProvStore([cpu, gpu, *_entities(), _agent()])
    assert cpu.id != gpu.id
    assert all(prov.literal_value(r.node()["senselab:logicalKey"]) == KEY for r in (cpu, gpu))
    read = ProvStore.read(_run(tmp_path, store), VOCAB)
    assert [r.id for r in read.activities(KEY)] == sorted([cpu.id, gpu.id])


def test_reuse_is_an_activity(tmp_path: Path) -> None:
    """A reuse of a stored result is its own activity under the same key, using what the execution generated."""
    execution = _execution({"device": "cpu"})
    reuse = prov.activity(
        VOCAB,
        attributes={"senselab:logicalKey": KEY},
        used=[Usage(OUTPUT, port="transcript", time=T0 + timedelta(hours=1))],
        started=T0 + timedelta(hours=1),
    )
    read = ProvStore.read(_run(tmp_path, ProvStore([execution, reuse, *_entities(), _agent()])), VOCAB)
    assert [r.id for r in read.activities(KEY)] == sorted([execution.id, reuse.id])
    assert read.get(reuse.id).node()["prov:used"] == [{"@id": OUTPUT}]


def test_writers_union_by_id(tmp_path: Path) -> None:
    """The run's provenance is the union of its writers' documents; a node two writers hold appears once."""
    execution = _execution({"device": "cpu"})
    executor = ProvStore([execution, _agent(), *_entities()])
    store = ProvStore(_entities())
    read = ProvStore.read(_run(tmp_path, executor, store), VOCAB)
    assert sorted(p.name for p in (tmp_path / "run" / PROV_DIR).iterdir()) == sorted(
        f"{s.writer}.jsonld" for s in (executor, store)
    )
    assert read.records() == ProvStore.merge([executor, store]).records()
    assert [r.id for r in read.entities()] == [INPUT, OUTPUT]
    assert [r.id for r in read.records()] == sorted(r.id for r in read.records())


def test_copied_activity_is_one_node(tmp_path: Path) -> None:
    """An activity read back from the executing writer and added by another writer stays one node."""
    executor = ProvStore([_execution({"device": "cpu"}), _agent(), *_entities()])
    run = _run(tmp_path, executor)
    copier = ProvStore(ProvStore.read(run, VOCAB).activities(KEY))
    assert [r.id for r in ProvStore.read(_run(tmp_path, copier), VOCAB).activities()] == [
        r.id for r in executor.activities()
    ]


def test_conflicting_node_refused(tmp_path: Path) -> None:
    """Two writers holding different nodes under one `@id` are refused, in memory and on read."""
    one = ProvStore([prov.entity(OUTPUT, VOCAB, attributes={"senselab:label": "a"})])
    other = ProvStore([prov.entity(OUTPUT, VOCAB, attributes={"senselab:label": "b"})])
    with pytest.raises(ConflictingNode) as caught:
        ProvStore.merge([one, other])
    assert caught.value.node_id == OUTPUT
    with pytest.raises(ConflictingNode):
        one.add(*other.records())
    with pytest.raises(ConflictingNode):
        ProvStore.read(_run(tmp_path, one, other), VOCAB)


def test_merge_is_order_independent() -> None:
    """`merge` gives the same records, in `@id` order, for every order of the stores and of their records."""
    records = [_execution({"device": "cpu"}), _execution({"device": "cuda"}), _agent(), _agent("local"), *_entities()]
    stores = [ProvStore(records[:3]), ProvStore(records[2:5]), ProvStore(reversed(records[4:]))]
    merged = {tuple(r.jcs for r in ProvStore.merge(order).records()) for order in itertools.permutations(stores)}
    assert len(merged) == 1
    ids = [r.id for r in ProvStore.merge(stores).records()]
    assert ids == sorted({r.id for r in records})
    for accessor in ("entities", "activities", "agents", "environments"):
        listed = [r.id for r in getattr(ProvStore.merge(stores), accessor)()]
        assert listed == sorted(listed)


def test_write_replaces_the_writers_document(tmp_path: Path) -> None:
    """A second write replaces the writer's one document with every record it now holds."""
    store = ProvStore([prov.entity(INPUT, VOCAB)])
    run = _run(tmp_path, store)
    store.add(prov.entity(OUTPUT, VOCAB))
    store.write(run)
    assert [p.name for p in (run / PROV_DIR).iterdir()] == [f"{store.writer}.jsonld"]
    assert jcs_loads(_document_path(run, store).read_bytes()) == store.document()


def test_read_round_trips_every_record_kind(tmp_path: Path) -> None:
    """Entities, files named by attributes, agents, environments and qualified activities read back identical."""
    waveform = prov.entity(
        "content:" + "3" * 64,
        VOCAB,
        attributes={"senselab:waveform": numpy.arange(4, dtype=numpy.float32), "senselab:loudness": float("-inf")},
    )
    env = prov.environment(VOCAB, {})
    store = ProvStore([_execution({"device": "cpu", "batch": [1, 2.5, None]}), _agent(), env, waveform, *_entities()])
    read = ProvStore.read(_run(tmp_path, store), VOCAB)
    assert [r.jcs for r in read.records()] == [r.jcs for r in store.records()]
    assert [r.id for r in read.environments()] == [env.id]
    assert len(waveform.files) == 1
    assert read.get(waveform.files[0].id).types() == ("prov:Entity", "senselab:File")


def test_read_of_a_run_without_provenance_is_empty(tmp_path: Path) -> None:
    """A run with no `prov/` directory has no records."""
    assert ProvStore.read(tmp_path).records() == []


def test_dangling_endpoint_refused(tmp_path: Path) -> None:
    """A used, generated or associated `@id` no writer holds is refused; one another writer holds is accepted."""
    execution = _execution({"device": "cpu"})
    executor = ProvStore([execution, _agent()])
    with pytest.raises(DanglingReference) as caught:
        ProvStore.read(_run(tmp_path, executor), VOCAB)
    assert (caught.value.node_id, caught.value.target) == (execution.id, INPUT)
    assert ProvStore.read(_run(tmp_path, ProvStore(_entities())), VOCAB).get(execution.id) == execution


def test_dangling_attribute_reference_refused(tmp_path: Path) -> None:
    """An attribute naming a file no writer holds is refused."""
    entity = prov.entity(INPUT, VOCAB, attributes={"senselab:waveform": numpy.zeros(2)})
    run = _run(tmp_path, store := ProvStore([entity]))
    _rewrite(_document_path(run, store), lambda doc: doc["@graph"].pop(0 if doc["@graph"][0]["@id"] != INPUT else 1))
    with pytest.raises(DanglingReference):
        ProvStore.read(run, VOCAB)


def _drop_qualified_time(doc: dict[str, Any]) -> None:
    for node in doc["@graph"]:
        if node["@id"].startswith("activity:"):
            del node["prov:qualifiedUsage"][0]["prov:atTime"]


def _out_of_range(doc: dict[str, Any]) -> None:
    doc["@graph"][-1]["senselab:loudness"] = "/home/someone/sub-01_task-read.wav"


def _path_id(doc: dict[str, Any]) -> None:
    doc["@graph"][-1]["@id"] = "/home/someone/sub-01_task-read.wav"


def _unknown_term(doc: dict[str, Any]) -> None:
    doc["@graph"][-1]["senselab:notATerm"] = 1


def _unqualified_agent(doc: dict[str, Any]) -> None:
    doc["@graph"][0]["prov:wasAssociatedWith"].append({"@id": "/home/someone/sub-01_task-read.wav"})


def _extra_key(doc: dict[str, Any]) -> None:
    doc["format"] = "x"


@pytest.mark.parametrize(
    "edit",
    [_drop_qualified_time, _out_of_range, _path_id, _unknown_term, _unqualified_agent, _extra_key],
    ids=[
        "activity-not-its-address",
        "attribute-out-of-range",
        "path-as-id",
        "unknown-term",
        "path-as-agent",
        "extra-key",
    ],
)
def test_a_node_the_builders_would_not_write_is_refused(tmp_path: Path, edit: Any) -> None:  # noqa: ANN401
    """Every node is rebuilt and compared byte for byte; the error names the writer, never the path or value."""
    store = ProvStore([_execution({"device": "cpu"}), _agent(), prov.entity(INPUT, VOCAB), prov.entity(OUTPUT, VOCAB)])
    run = _run(tmp_path, store)
    _rewrite(_document_path(run, store), edit)
    with pytest.raises(MalformedDocument) as caught:
        ProvStore.read(run, VOCAB)
    assert caught.value.writer == store.writer
    assert "sub-01" not in str(caught.value)
    assert str(tmp_path) not in str(caught.value)


def test_a_reference_cycle_is_refused(tmp_path: Path) -> None:
    """Two entities naming each other through attributes, which no builder can write, are refused."""
    file_b = prov.entity(OUTPUT, VOCAB, types=["senselab:File"])
    file_a = prov.entity(INPUT, VOCAB, types=["senselab:File"], attributes={"senselab:waveform": file_b})
    run = _run(tmp_path, store := ProvStore([file_a, file_b]))
    _rewrite(_document_path(run, store), lambda doc: doc["@graph"][1].update({"senselab:waveform": {"@id": INPUT}}))
    with pytest.raises(MalformedDocument, match="names itself"):
        ProvStore.read(run, VOCAB)


def test_a_term_outside_the_readers_vocabulary_is_refused(tmp_path: Path) -> None:
    """A node is read over the reader's vocabulary: a term only the writer's vocabulary holds is refused."""
    run = _run(tmp_path, ProvStore([prov.entity(INPUT, VOCAB, attributes={"senselab:label": "x"})]))
    with pytest.raises(MalformedDocument, match="UnknownTerm"):
        ProvStore.read(run)


def test_files_that_are_not_writer_documents(tmp_path: Path) -> None:
    """A temporary of an interrupted write is skipped; any other foreign file in `prov/` is refused unnamed."""
    store = ProvStore(_entities())
    run = _run(tmp_path, store)
    (run / PROV_DIR / f".{store.writer}.jsonld.1.2.{'a' * 32}.tmp").write_bytes(b"{partial")
    assert ProvStore.read(run, VOCAB).records() == store.records()
    (run / PROV_DIR / "sub-01_notes.txt").write_text("x")
    with pytest.raises(MalformedDocument) as caught:
        ProvStore.read(run, VOCAB)
    assert caught.value.writer is None
    assert "sub-01" not in str(caught.value)


def test_a_document_that_is_not_canonical_is_refused(tmp_path: Path) -> None:
    """A writer's document is the JCS text of `{"@context": CONTEXT_URL, "@graph": [...]}`."""
    store = ProvStore(_entities())
    run = _run(tmp_path, store)
    path = _document_path(run, store)
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(MalformedDocument, match="canonical"):
        ProvStore.read(run, VOCAB)
    path.write_bytes(jcs_dumps({"@context": "https://example.org/other", "@graph": []}))
    with pytest.raises(MalformedDocument, match="@context"):
        ProvStore.read(run, VOCAB)
