"""`ProvStore`: one writer's PROV-O JSON-LD document per writer, and a run's provenance as their union by `@id`."""

import itertools
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy
import pytest
from senselab_core.kernel import prov
from senselab_core.kernel.formats import PROVENANCE_FORMAT
from senselab_core.kernel.prov import CONTEXT_URL, Association, Generation, Usage, Vocabulary
from senselab_core.runtime.provenance import (
    PROV_DIR,
    ConflictingNode,
    DanglingReference,
    MalformedDocument,
    ProvenanceFormatRefused,
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
        types=["senselab:Reuse"],
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


def test_a_plan_an_association_names_is_read_back_and_must_be_held(tmp_path: Path) -> None:
    """`prov:hadPlan` names a plan node of the union: read back with it, refused without it."""
    plan = prov.plan("3" * 64, VOCAB)
    linked = prov.activity(VOCAB, associations=[Association(_agent().id, node_id="asr", plan=plan.id)])
    with pytest.raises(DanglingReference) as caught:
        ProvStore.read(_run(tmp_path, ProvStore([linked, _agent()])), VOCAB)
    assert (caught.value.node_id, caught.value.target) == (linked.id, plan.id)
    alone = prov.activity(VOCAB, associations=[Association(_agent().id)], started=T0)
    read = ProvStore.read(_run(tmp_path, ProvStore([plan, alone])), VOCAB)
    assert read.get(linked.id) == linked and read.get(alone.id) == alone and read.plans() == [plan]


def test_an_association_without_an_agent_is_read_back_with_its_node_and_plan(tmp_path: Path) -> None:
    """An agentless association round-trips through write and read; its plan must still be held."""
    plan = prov.plan("3" * 64, VOCAB)
    settled = prov.activity(
        VOCAB, associations=[Association(None, node_id="asr", plan=plan.id)], used=[Usage(INPUT)], started=T0
    )
    with pytest.raises(DanglingReference) as caught:
        ProvStore.read(_run(tmp_path, ProvStore([settled, *_entities()])), VOCAB)
    assert (caught.value.node_id, caught.value.target) == (settled.id, plan.id)
    read = ProvStore.read(_run(tmp_path, ProvStore([plan, settled])), VOCAB)
    assert read.get(settled.id) == settled and "prov:wasAssociatedWith" not in settled.node()


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
    doc["@graph"][-1]["senselab:loudness"] = "/home/user/sub-01_task-read.wav"


def _path_id(doc: dict[str, Any]) -> None:
    doc["@graph"][-1]["@id"] = "/home/user/sub-01_task-read.wav"


def _unknown_term(doc: dict[str, Any]) -> None:
    doc["@graph"][-1]["senselab:notATerm"] = 1


def _unqualified_agent(doc: dict[str, Any]) -> None:
    doc["@graph"][0]["prov:wasAssociatedWith"].append({"@id": "/home/user/sub-01_task-read.wav"})


def _decimal_not_a_number(doc: dict[str, Any]) -> None:
    doc["@graph"][-1]["senselab:notATerm"] = {"@value": "/home/user/sub-01_task-read.wav", "@type": "xsd:decimal"}


def _extra_key(doc: dict[str, Any]) -> None:
    doc["format"] = "x"


@pytest.mark.parametrize(
    "edit",
    [
        _drop_qualified_time,
        _out_of_range,
        _path_id,
        _unknown_term,
        _unqualified_agent,
        _decimal_not_a_number,
        _extra_key,
    ],
    ids=[
        "activity-not-its-address",
        "attribute-out-of-range",
        "path-as-id",
        "unknown-term",
        "path-as-agent",
        "decimal-not-a-number",
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
    assert caught.value.__context__ is None
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
    """A writer's document is the JCS text of `{"@context": CONTEXT_URL, "format": <provenance@3>, "@graph": [...]}`."""
    store = ProvStore(_entities())
    run = _run(tmp_path, store)
    path = _document_path(run, store)
    assert jcs_loads(path.read_bytes())["format"] == PROVENANCE_FORMAT == "provenance@3.0"  # type: ignore[index]
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(MalformedDocument, match="canonical") as caught:
        ProvStore.read(run, VOCAB)
    assert caught.value.__context__ is None
    for document, refusal in [
        ({"@context": "https://example.org/other", "format": PROVENANCE_FORMAT, "@graph": []}, "@context"),
        ({"@context": CONTEXT_URL, "@graph": []}, "format"),
        ({"@context": CONTEXT_URL, "format": ["provenance@3.0"], "@graph": []}, "format is"),
    ]:
        path.write_bytes(jcs_dumps(document))
        with pytest.raises(MalformedDocument, match=refusal):
            ProvStore.read(run, VOCAB)


def test_a_later_minor_is_read(tmp_path: Path) -> None:
    """A document of a later minor of `provenance@3` is read."""
    store = ProvStore(_entities())
    run = _run(tmp_path, store)
    _rewrite(_document_path(run, store), lambda doc: doc.update(format="provenance@3.7"))
    assert ProvStore.read(run, VOCAB).records() == store.records()


@pytest.mark.parametrize(
    ("name", "lines", "found"),
    [
        ("store.jsonl", [{"format": "provenance@2.0", "record": "format"}, {"record": "entity"}], "provenance@2.0"),
        (f"{'a' * 32}.jsonld", [{"format": "provenance@2.0", "record": "format"}], "provenance@2.0"),
        (f"{'b' * 32}.jsonld", [{"@context": CONTEXT_URL, "format": "provenance@4.0", "@graph": []}], "provenance@4.0"),
    ],
)
def test_provenance2_refused(tmp_path: Path, name: str, lines: list[dict[str, object]], found: str) -> None:
    """A `provenance@2.0` store, or any provenance file but `provenance@3`, is refused naming its format."""
    run = _run(tmp_path, ProvStore(_entities()))
    (run / PROV_DIR / name).write_bytes(b"".join(jcs_dumps(line) + b"\n" for line in lines))
    with pytest.raises(ProvenanceFormatRefused) as caught:
        ProvStore.read(run, VOCAB)
    assert caught.value.found == found
    assert found in str(caught.value) and "migration-ledger.md#retired-provenance-formats" in str(caught.value)
    assert str(tmp_path) not in str(caught.value) and name not in str(caught.value)


def test_indented_retired_provenance_is_refused_not_malformed(tmp_path: Path) -> None:
    """A retired document written indented over many lines is a format refusal, not a malformed document."""
    run = _run(tmp_path, ProvStore(_entities()))
    name = f"{'c' * 32}.jsonld"
    (run / PROV_DIR / name).write_text('{\n  "format": "provenance@2.0",\n  "record": "format"\n}\n', "utf-8")
    with pytest.raises(ProvenanceFormatRefused) as caught:
        ProvStore.read(run, VOCAB)
    assert caught.value.found == "provenance@2.0"


def test_add_is_all_or_nothing() -> None:
    """An `add` that meets a conflict adds none of its records."""
    store = ProvStore([prov.entity(OUTPUT, VOCAB, attributes={"senselab:label": "a"})])
    with pytest.raises(ConflictingNode):
        store.add(prov.entity(INPUT, VOCAB), prov.entity(OUTPUT, VOCAB, attributes={"senselab:label": "b"}))
    assert [r.id for r in store.records()] == [OUTPUT]
    fresh = ProvStore()
    with pytest.raises(ConflictingNode):
        fresh.add(
            prov.entity(INPUT, VOCAB),
            prov.entity(OUTPUT, VOCAB, attributes={"senselab:label": "a"}),
            prov.entity(OUTPUT, VOCAB, attributes={"senselab:label": "b"}),
        )
    assert fresh.records() == []


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-0 file")
def test_an_unreadable_document_is_refused_without_its_path(tmp_path: Path) -> None:
    """A writer's document the reader cannot open is refused by writer, never by path."""
    store = ProvStore(_entities())
    run = _run(tmp_path, store)
    _document_path(run, store).chmod(0)
    try:
        with pytest.raises(MalformedDocument, match="cannot be read") as caught:
            ProvStore.read(run, VOCAB)
    finally:
        _document_path(run, store).chmod(0o600)
    assert caught.value.writer == store.writer
    assert caught.value.__context__ is None
    assert str(tmp_path) not in str(caught.value)


def test_attribute_references_nest_no_deeper_than_the_canonical_depth(tmp_path: Path) -> None:
    """A chain of attribute references deeper than `MAX_DEPTH` is refused as malformed, not by the interpreter."""
    chain = [prov.entity("content:" + f"{70:064x}", VOCAB, types=["senselab:File"])]
    for i in range(69, -1, -1):
        address = "content:" + f"{i:064x}"
        chain.append(prov.entity(address, VOCAB, types=["senselab:File"], attributes={"senselab:waveform": chain[-1]}))
    run = _run(tmp_path, ProvStore(chain))
    with pytest.raises(MalformedDocument, match="deeper than 64") as caught:
        ProvStore.read(run, VOCAB)
    assert caught.value.__context__ is None


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_each_forked_process_writes_its_own_document(tmp_path: Path) -> None:
    """A store shared across `fork` is a new writer in each child, so no child's document replaces another's."""
    run = _run(tmp_path, store := ProvStore())
    children = []
    for i in range(4):
        pid = os.fork()
        if pid == 0:  # pragma: no cover - runs in the child
            code = 1
            try:
                store.add(prov.entity("content:" + f"{i:064x}", VOCAB))
                store.write(run)
                code = 0
            finally:
                os._exit(code)
        children.append(pid)
    assert [os.waitpid(pid, 0)[1] for pid in children] == [0] * 4
    assert [r.id for r in ProvStore.read(run, VOCAB).records()] == ["content:" + f"{i:064x}" for i in range(4)]


@pytest.mark.logic
def test_the_format_sniff_reads_to_the_depth_bound_and_no_further() -> None:
    """A document 64 deep declares its format; one 65 deep declares none, whatever the parser could parse."""
    from senselab_core.runtime.provenance import _declared_format

    def document(depth: int) -> bytes:
        return b'{"format": "provenance@3.0", "a": ' + b"[" * (depth - 1) + b"]" * (depth - 1) + b"}"

    assert _declared_format(document(64)) == "provenance@3.0"
    assert _declared_format(document(65)) is None
