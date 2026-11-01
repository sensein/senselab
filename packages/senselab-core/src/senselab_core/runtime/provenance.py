"""A run's provenance in PROV-O JSON-LD (ADR-010): one document per writer, read back as their union by `@id`.

Each writer holds the records it adds (`senselab_core.kernel.prov`) and replaces its own document,
`<run>/prov/<writer uuid4 hex>.jsonld`, atomically. A reader takes the union of every writer's document; two
different nodes under one `@id` are refused, and every node read is rebuilt through the record builders.
"""

from __future__ import annotations

import os
import re
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from senselab_worker.canonical import jcs_dumps, jcs_loads, require_depth

from senselab_core.kernel import prov as kprov
from senselab_core.runtime.storage import TEMPORARY_NAME, write_atomically
from senselab_core.runtime.vocabulary import core_vocabulary

WRITER_DOCUMENT = re.compile(r"[0-9a-f]{32}\.jsonld")
"""A writer's document in `<run>/prov/`: its writer uuid4 hex, `.jsonld`."""
PROV_DIR = "prov"
"""The run subdirectory that holds the writers' documents."""
_ACTIVITY_KEYS = frozenset(
    {
        "@id",
        "@type",
        "@reverse",
        "prov:startedAtTime",
        "prov:endedAtTime",
        "prov:used",
        "prov:qualifiedUsage",
        "prov:generated",
        "prov:wasAssociatedWith",
        "prov:qualifiedAssociation",
    }
)
_NODE_KEYS = frozenset({"@id", "@type"})
_ENDPOINT_KEYS = ("prov:used", "prov:generated", "prov:wasAssociatedWith")
_REBUILD_ERRORS = (ValueError, TypeError, KeyError, AttributeError)


class ConflictingNode(ValueError):
    """Two different nodes under one `@id`.

    Attributes:
        node_id: The `@id` the two nodes share.
    """

    def __init__(self, node_id: str) -> None:
        """Name the shared `@id`."""
        super().__init__(f"two different nodes have the @id {node_id}")
        self.node_id = node_id


class DanglingReference(ValueError):
    """A relation or attribute naming an `@id` no writer's document holds.

    Attributes:
        node_id: The node that names it.
        target: The `@id` it names.
    """

    def __init__(self, node_id: str, target: str) -> None:
        """Name the node and the missing target."""
        super().__init__(f"{node_id} names {target}, which no writer's document holds")
        self.node_id = node_id
        self.target = target


class MalformedDocument(ValueError):
    """A writer's document, or a node in it, that the record builders would not write.

    Attributes:
        writer: The writer's uuid4 hex, or None for a file in `prov/` that is not a writer's document.
    """

    def __init__(self, writer: str | None, message: str) -> None:
        """Name the writer; the message never echoes a refused id or value."""
        super().__init__(f"writer {writer}: {message}" if writer else message)
        self.writer = writer


class ProvStore:
    """A set of PROV records keyed by `@id`: one writer's records, or the union of a run's writers.

    `add` holds a record and the `senselab:File` entities it names; `write` replaces this writer's document,
    `<run>/prov/<writer>.jsonld`, atomically; `read` returns the union of every writer's document in a run.
    Accessors return records in `@id` order.

    A store used after a `fork` is a new writer in the child, so each process writes its own document.
    """

    def __init__(self, records: Iterable[kprov.Record] = ()) -> None:
        """Hold `records`.

        Raises:
            ConflictingNode: if two of them are different nodes under one `@id`.
        """
        self._writer = (os.getpid(), uuid.uuid4().hex)
        self._records: dict[str, kprov.Record] = {}
        self.add(*records)

    @property
    def writer(self) -> str:
        """This process's writer uuid4 hex: the name of the document `write` replaces."""
        if self._writer[0] != os.getpid():
            self._writer = (os.getpid(), uuid.uuid4().hex)
        return self._writer[1]

    def add(self, *records: kprov.Record) -> None:
        """Hold each record and the files it names; a record already held under its `@id` is kept once.

        Nothing is added unless every record is.

        Raises:
            ConflictingNode: if a record differs from the one held, or another one given, under its `@id`.
        """
        added: dict[str, kprov.Record] = {}
        for r in (r for record in records for r in (record, *record.files)):
            held = self._records.get(r.id) or added.setdefault(r.id, r)
            if held.jcs != r.jcs:
                raise ConflictingNode(r.id)
        self._records.update(added)

    @classmethod
    def merge(cls, stores: Iterable[ProvStore]) -> ProvStore:
        """Return the union of `stores` by `@id`; the result does not depend on their order.

        Raises:
            ConflictingNode: if two stores hold different nodes under one `@id`.
        """
        return cls(r for store in stores for r in store.records())

    def records(self) -> list[kprov.Record]:
        """Return every record, in `@id` order."""
        return [self._records[k] for k in sorted(self._records)]

    def get(self, node_id: str) -> kprov.Record:
        """Return the record under `node_id`.

        Raises:
            KeyError: if none is held.
        """
        return self._records[node_id]

    def _of(self, *prefixes: str) -> list[kprov.Record]:
        return [r for r in self.records() if r.id.partition(":")[0] in prefixes]

    def entities(self) -> list[kprov.Record]:
        """Return the content-addressed entities (`content:`, `output:`), in `@id` order."""
        return self._of("content", "output")

    def activities(self, logical_key: object = None) -> list[kprov.Record]:
        """Return the activities, or those whose `senselab:logicalKey` is `logical_key`, in `@id` order."""
        out = self._of("activity")
        if logical_key is None:
            return out
        wanted = jcs_dumps(logical_key)
        return [
            r
            for r in out
            if "senselab:logicalKey" in (node := r.node())
            and jcs_dumps(kprov.literal_value(node["senselab:logicalKey"])) == wanted
        ]

    def agents(self) -> list[kprov.Record]:
        """Return the agents, in `@id` order."""
        return self._of("agent")

    def environments(self) -> list[kprov.Record]:
        """Return the environments, in `@id` order."""
        return self._of("environment")

    def document(self) -> dict[str, object]:
        """Return the JSON-LD document of every record, nodes in `@id` order."""
        return kprov.document(self.records())

    def write(self, run: str | Path) -> None:
        """Replace this writer's document, `<run>/prov/<writer>.jsonld`, atomically with `document()` as JCS."""
        directory = Path(run) / PROV_DIR
        directory.mkdir(exist_ok=True)
        write_atomically(directory / f"{self.writer}.jsonld", jcs_dumps(self.document()))

    @classmethod
    def read(cls, run: str | Path, vocabulary: kprov.Vocabulary | None = None) -> ProvStore:
        """Return the union of the writers' documents in `<run>/prov/`, every node rebuilt over `vocabulary`.

        Each node is rebuilt through the record builders over `vocabulary` (the core vocabulary when None) and
        must be byte-identical to its rebuild, so its `@id`, terms and literals are all checked. Every relation
        endpoint and every record an attribute names must be a node of the union. Temporaries of an interrupted
        write are skipped.

        Raises:
            MalformedDocument: on a file in `prov/` that is not a writer's document or cannot be read, a document
                of another shape, a node that differs from its rebuild, or attribute references nested deeper than
                `MAX_DEPTH`.
            ConflictingNode: if two writers hold different nodes under one `@id`.
            DanglingReference: if a node names an `@id` the union does not hold.
        """
        nodes: dict[str, tuple[bytes, str]] = {}
        directory = Path(run) / PROV_DIR
        try:
            listing = sorted(directory.iterdir()) if directory.is_dir() else []
        except OSError as err:
            raise MalformedDocument(None, f"{PROV_DIR}/ cannot be listed ({type(err).__name__})") from None
        for path in listing:
            if TEMPORARY_NAME.fullmatch(path.name):
                continue
            if not (WRITER_DOCUMENT.fullmatch(path.name) and path.is_file() and not path.is_symlink()):
                raise MalformedDocument(None, f"{PROV_DIR}/ holds a file that is not a writer's document")
            writer = path.name.removesuffix(".jsonld")
            try:
                data = path.read_bytes()
            except OSError as err:
                raise MalformedDocument(writer, f"the document cannot be read ({type(err).__name__})") from None
            for node_id, raw in _writer_nodes(writer, data):
                if nodes.setdefault(node_id, (raw, writer))[0] != raw:
                    raise ConflictingNode(node_id)
        return cls(_Rebuild(nodes, vocabulary or core_vocabulary()).all())


def _writer_nodes(writer: str, data: bytes) -> list[tuple[str, bytes]]:
    try:
        doc = jcs_loads(data)
    except ValueError:
        raise MalformedDocument(writer, "the document is not canonical JSON") from None
    graph = doc.get("@graph") if isinstance(doc, dict) else None
    if not (isinstance(doc, dict) and doc.keys() == {"@context", "@graph"} and isinstance(graph, list)):
        raise MalformedDocument(writer, f"a document is {{'@context': {kprov.CONTEXT_URL!r}, '@graph': [...]}}")
    if doc["@context"] != kprov.CONTEXT_URL:
        raise MalformedDocument(writer, f"a document's @context is {kprov.CONTEXT_URL}")
    out = []
    for node in graph:
        node_id = node.get("@id") if isinstance(node, dict) else None
        if not (isinstance(node_id, str) and kprov.NODE_ID.fullmatch(node_id)):
            raise MalformedDocument(writer, f"a node's @id is not of the form {kprov.NODE_ID.pattern}")
        out.append((node_id, jcs_dumps(node)))
    return out


class _Rebuild:
    """Rebuild a union of raw nodes through the record builders, checking each against its bytes."""

    def __init__(self, nodes: Mapping[str, tuple[bytes, str]], vocabulary: kprov.Vocabulary) -> None:
        self.nodes = nodes
        self.vocabulary = vocabulary
        self.done: dict[str, kprov.Record] = {}
        self.pending: set[str] = set()

    def all(self) -> list[kprov.Record]:
        return [self.record(node_id, node_id) for node_id in sorted(self.nodes)]

    def record(self, node_id: str, named_by: str) -> kprov.Record:
        if node_id in self.done:
            return self.done[node_id]
        if node_id not in self.nodes:
            raise DanglingReference(named_by, node_id)
        raw, writer = self.nodes[node_id]
        if node_id in self.pending:
            raise MalformedDocument(writer, f"{node_id} names itself through its attributes")
        try:
            require_depth(len(self.pending) + 1)
        except ValueError as err:
            raise MalformedDocument(writer, f"{node_id}: attribute references: {err}") from None
        self.pending.add(node_id)
        node = cast("dict[str, Any]", jcs_loads(raw))
        try:
            rebuilt = self.build(node_id, node)
        except (DanglingReference, MalformedDocument):
            raise
        except _REBUILD_ERRORS as err:
            raise MalformedDocument(
                writer, f"{node_id} is not what the builders write ({type(err).__name__})"
            ) from None
        if rebuilt.jcs != raw:
            raise MalformedDocument(writer, f"{node_id} differs from its rebuild")
        for target in sorted({ref["@id"] for key in _ENDPOINT_KEYS for ref in node.get(key, [])}):
            if target not in self.nodes:
                raise DanglingReference(node_id, target)
        self.pending.discard(node_id)
        self.done[node_id] = rebuilt
        return rebuilt

    def build(self, node_id: str, node: dict[str, Any]) -> kprov.Record:
        kind = node_id.partition(":")[0]
        types = node["@type"]
        if kind == "activity":
            return self.activity(node_id, node)
        attributes = self.attributes(node_id, node, _NODE_KEYS)
        if kind == "agent":
            return kprov.agent(self.vocabulary, types=types, attributes=attributes)
        if kind == "environment":
            return kprov.environment(self.vocabulary, attributes)
        return kprov.entity(node_id, self.vocabulary, types=types, attributes=attributes)

    def activity(self, node_id: str, node: dict[str, Any]) -> kprov.Record:
        usages = [
            kprov.Usage(q["prov:entity"]["@id"], port=_role(q, "senselab:port"), time=_time(q.get("prov:atTime")))
            for q in node.get("prov:qualifiedUsage", [])
        ]
        generations = [
            kprov.Generation(
                q["@reverse"]["prov:qualifiedGeneration"]["@id"],
                port=_role(q, "senselab:port"),
                time=_time(q.get("prov:atTime")),
            )
            for q in node.get("@reverse", {}).get("prov:activity", [])
        ]
        usages += [kprov.Usage(e) for e in _ids(node, "prov:used") - {u.entity for u in usages}]
        generations += [kprov.Generation(e) for e in _ids(node, "prov:generated") - {g.entity for g in generations}]
        associations = [
            kprov.Association(q["prov:agent"]["@id"], q["prov:hadRole"]["senselab:nodeId"])
            for q in node.get("prov:qualifiedAssociation", [])
        ]
        return kprov.activity(
            self.vocabulary,
            types=node["@type"],
            attributes=self.attributes(node_id, node, _ACTIVITY_KEYS),
            used=usages,
            generated=generations,
            associations=associations,
            started=_time(node.get("prov:startedAtTime")),
            ended=_time(node.get("prov:endedAtTime")),
        )

    def attributes(self, node_id: str, node: dict[str, Any], structural: frozenset[str]) -> dict[str, object]:
        out: dict[str, object] = {}
        for term, value in node.items():
            if term in structural:
                continue
            prop = self.vocabulary.property(term)
            if prop is not None and prop.range not in kprov.LITERAL_RANGES:
                out[term] = self.reference(node_id, value)
            else:
                out[term] = kprov.literal_value(value)
        return out

    def reference(self, node_id: str, value: Mapping[str, Any]) -> object:
        if value.keys() == {"@set"}:
            return frozenset(self.reference(node_id, v) for v in value["@set"])
        if value.keys() == {"@list"}:
            return tuple(self.reference(node_id, v) for v in value["@list"])
        target = value["@id"]
        if not (isinstance(target, str) and kprov.NODE_ID.fullmatch(target)):
            raise ValueError("a reference is an @id")
        return self.record(target, node_id)


def _ids(node: Mapping[str, Any], key: str) -> set[str]:
    return {ref["@id"] for ref in node.get(key, [])}


def _role(qualified: Mapping[str, Any], term: str) -> str | None:
    role = qualified.get("prov:hadRole")
    return None if role is None else role[term]


def _time(literal: object) -> datetime | None:
    if literal is None:
        return None
    value = kprov.literal_value(literal)
    if not isinstance(value, datetime):
        raise TypeError("a time is an xsd:dateTime")
    return value
