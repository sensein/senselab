"""Export redaction and the BEP028 profile (ADR-007, ADR-010).

A file is named in provenance by its content address and a per-run label, `file-<n>`. The label-to-path map is
the run's private `<run>/private/labels@1.json` (mode 0600); it is never copied or exported. `export` copies a run's
documents after `check_profile` finds the BEP028 profile met and after no path of the map is found in them.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

from senselab_core.kernel import prov as kprov
from senselab_core.kernel.formats import LABELS_FORMAT, PROVENANCE_FORMAT, readable
from senselab_core.runtime.locks import Lease
from senselab_core.runtime.provenance import PROV_DIR, WRITER_DOCUMENT, ProvStore, pathless
from senselab_core.runtime.storage import PRIVATE_DIR_MODE, StorageDir, write_atomically
from senselab_core.runtime.vocabulary import core_vocabulary

PRIVATE_DIR = "private"
"""The run subdirectory that holds the label map; nothing in it is exported."""
LABELS_NAME = "labels@1.json"
"""The label map's file name in `<run>/private/`."""
LABELS_MODE = 0o600
"""The mode of the label map and of its lease."""
FILE_LABEL = re.compile(r"file-[1-9][0-9]*")
"""A file's label: `file-` and its number in the run."""
_LABELS_FILE = re.compile(re.escape(LABELS_NAME))
_ENTITY_ID = re.compile(r"(content|output):[0-9a-f]{64}")
_KINDS = {"prov:Entity", "prov:Activity", "prov:Agent"}


class ProfileViolation(ValueError):
    """A run whose provenance does not meet the BEP028 profile, or whose documents hold a recorded path.

    Attributes:
        violations: One message per violation; none echoes a value of the document.
    """

    def __init__(self, violations: list[str]) -> None:
        """Hold `violations`."""
        super().__init__("; ".join(violations))
        self.violations = violations


def _read_labels(directory: StorageDir) -> dict[str, tuple[str, list[str]]]:
    data = directory.read_regular(LABELS_NAME)
    if data is None:
        return {}
    try:
        record = json.loads(data)
        labels = record["labels"]
        valid = (
            readable(LABELS_FORMAT, record["format"])
            and isinstance(labels, dict)
            and all(
                FILE_LABEL.fullmatch(k)
                and isinstance(v, dict)
                and v.keys() == {"address", "paths"}
                and isinstance(v["address"], str)
                and _ENTITY_ID.fullmatch(v["address"])
                and isinstance(v["paths"], list)
                and all(isinstance(p, str) for p in v["paths"])
                for k, v in labels.items()
            )
        )
    except (ValueError, KeyError, TypeError):
        valid = False
    if not valid:
        raise ValueError(f"the label map is not {LABELS_FORMAT}")
    return {k: (v["address"], list(v["paths"])) for k, v in labels.items()}


class LabelMap:
    """A run's private map from `file-<n>` labels to the content address and paths each names.

    One label per content address; a second path of the same bytes is added to that label's paths.

    Args:
        run: The run directory; `private/` is made under it, mode 0700, when a label is first assigned.
    """

    def __init__(self, run: str | Path) -> None:
        """Name the run; nothing is read or made until the first call."""
        self._run = Path(run)

    def assign(self, files: Iterable[tuple[str, str]]) -> list[str]:
        """Return the label of each `(content address, path)`, recording the path under it, under one lease.

        Raises:
            ValueError: an address is not a `content:` or `output:` address, or the map on disk is not `labels@1`.
            RunIOError: the map cannot be read or written; the error names no path.
        """
        items = list(files)
        for address, _ in items:
            if not _ENTITY_ID.fullmatch(address):
                raise ValueError("a file's address is a content: or output: address")
        return pathless(lambda: self._assign(items))

    def _assign(self, items: list[tuple[str, str]]) -> list[str]:
        with (
            StorageDir.open_root(self._run, create=True) as run,
            run.walk(PRIVATE_DIR, create=True, mode=PRIVATE_DIR_MODE, aside=None) as private,
            Lease(private.path / LABELS_NAME, file_mode=LABELS_MODE, directory=private),
        ):
            labels = _read_labels(private)
            by_address = {address: label for label, (address, _) in labels.items()}
            assigned = []
            for address, path in items:
                label = by_address.get(address)
                if label is None:
                    label = by_address[address] = f"file-{len(labels) + 1}"
                    labels[label] = (address, [])
                if path not in labels[label][1]:
                    labels[label][1].append(path)
                assigned.append(label)
            record = {k: {"address": a, "paths": p} for k, (a, p) in labels.items()}
            body = json.dumps({"format": LABELS_FORMAT, "labels": record}, sort_keys=True, indent=2)
            private.write_atomically(LABELS_NAME, body.encode(), grammar=_LABELS_FILE, mode=LABELS_MODE)
            return assigned

    def paths(self) -> dict[str, list[str]]:
        """Return `{label: paths}` of the map, empty when the run has none.

        Raises:
            ValueError: the map on disk is not `labels@1`.
            RunIOError: the map cannot be read; the error names no path.
        """

        def read() -> dict[str, list[str]]:
            try:
                with StorageDir.open_root(self._run) as run, run.walk(PRIVATE_DIR) as private:
                    return {k: p for k, (_, p) in _read_labels(private).items()}
            except FileNotFoundError:
                return {}

        return pathless(read)

    def entities(
        self, files: Iterable[tuple[str, str]], vocabulary: kprov.Vocabulary | None = None
    ) -> list[kprov.Record]:
        """Return a `senselab:File` entity, carrying its label and no path, for each `(content address, path)`."""
        items = list(files)
        return [
            file_entity(address, label, vocabulary)
            for (address, _), label in zip(items, self.assign(items), strict=True)
        ]


def file_entity(address: str, label: str, vocabulary: kprov.Vocabulary | None = None) -> kprov.Record:
    """Return the `senselab:File` entity of content `address` carrying `label`.

    Raises:
        ValueError: `label` is not `file-<n>`, or `address` is not a content address.
    """
    if not FILE_LABEL.fullmatch(label):
        raise ValueError("a file's label is file-<n>")
    return kprov.entity(
        address, vocabulary or core_vocabulary(), types=["senselab:File"], attributes={"senselab:fileLabel": label}
    )


def check_profile(doc: Mapping[str, object]) -> list[str]:
    """Return the ways `doc`, a `provenance@3` document, does not meet the BEP028 profile; none when it does.

    The profile: the published `@context` and a readable `format`; every graph node has a unique `@id` of the
    address grammar and a PROV class as `@type`; an activity's end has its start and is not before it; every
    `prov:used`, `prov:generated` and `prov:wasAssociatedWith` target is a node of the graph; a file label is
    `file-<n>` and names one node. Messages name node ids (addresses) and terms, never a value.
    """
    problems: list[str] = []
    if doc.get("@context") != kprov.CONTEXT_URL:
        problems.append(f"@context is not {kprov.CONTEXT_URL}")
    if not readable(PROVENANCE_FORMAT, doc.get("format")):
        problems.append(f"format is not {PROVENANCE_FORMAT}")
    graph = doc.get("@graph")
    nodes = [n for n in graph if isinstance(n, dict)] if isinstance(graph, list) else []
    if not isinstance(graph, list) or len(nodes) != len(graph):
        return [*problems, "@graph is not a list of nodes"]
    ids = [n.get("@id") for n in nodes]
    known = {i for i in ids if isinstance(i, str) and kprov.NODE_ID.fullmatch(i)}
    labels: dict[str, str] = {}
    for index, node in enumerate(nodes):
        node_id = node.get("@id")
        name = node_id if node_id in known else f"node {index}"
        if node_id not in known:
            problems.append(f"{name} has no @id of the address grammar")
        elif ids.count(node_id) > 1:
            problems.append(f"{name} is not the only node with its @id")
        problems += _node_problems(name, node, known, labels)
    return problems


def _node_problems(name: str, node: Mapping[str, object], known: set[str], labels: dict[str, str]) -> list[str]:
    problems = []
    kinds = node.get("@type")
    if not isinstance(kinds, list) or not _KINDS.intersection(kinds):
        problems.append(f"{name} is not a prov:Entity, prov:Activity or prov:Agent")
    elif "prov:Activity" in kinds:
        problems += _activity_problems(name, node, known)
    label = node.get("senselab:fileLabel")
    if label is not None:
        if not (isinstance(label, str) and FILE_LABEL.fullmatch(label)):
            problems.append(f"{name} has a senselab:fileLabel that is not file-<n>")
        elif labels.setdefault(label, name) != name:
            problems.append(f"{name} has a senselab:fileLabel another node has")
    return problems


def _activity_problems(name: str, node: Mapping[str, object], known: set[str]) -> list[str]:
    problems = []
    start, end = node.get("prov:startedAtTime"), node.get("prov:endedAtTime")
    if end is not None and start is None:
        problems.append(f"{name} ends and never starts")
    elif start is not None and end is not None and (_time(end) or datetime.max) < (_time(start) or datetime.min):
        problems.append(f"{name} ends before it starts")
    for term in ("prov:used", "prov:generated", "prov:wasAssociatedWith"):
        refs = node.get(term, [])
        targets = [r.get("@id") for r in refs if isinstance(r, dict)] if isinstance(refs, list) else [None]
        if any(t not in known for t in targets):
            problems.append(f"{name} has a {term} that is not a node of the document")
    return problems


def _time(literal: object) -> datetime | None:
    try:
        value = kprov.literal_value(literal)
    except ValueError:
        return None
    return value if isinstance(value, datetime) else None


def export(run: str | Path, dest: str | Path, vocabulary: kprov.Vocabulary | None = None) -> list[Path]:
    """Copy a run's provenance documents to `<dest>/prov/` and return the copies; the private directory is never read.

    The run's provenance is read over `vocabulary` (the core vocabulary when None) and must meet
    `check_profile`; no string of any document may contain a path of the run's label map.

    Raises:
        ProfileViolation: a profile violation, or a document that holds a path of the label map; nothing is copied.
        ProvenanceFormatRefused, MalformedDocument, ConflictingNode, DanglingReference: as `ProvStore.read`.
        RunIOError: a document cannot be read or copied; the error names no path.
    """
    source = Path(run) / PROV_DIR
    store = ProvStore.read(run, vocabulary)
    problems = check_profile(store.document())
    documents = pathless(
        lambda: (
            {p.name: p.read_bytes() for p in sorted(source.iterdir()) if WRITER_DOCUMENT.fullmatch(p.name)}
            if source.is_dir()
            else {}
        )
    )
    paths = {path for group in LabelMap(run).paths().values() for path in group if path}
    for data in documents.values():
        if any(path in text for text in _strings(json.loads(data)) for path in paths):
            problems.append("a document holds a path of the run's label map")
            break
    if problems:
        raise ProfileViolation(problems)
    target = Path(dest) / PROV_DIR

    def copy() -> list[Path]:
        target.mkdir(parents=True, exist_ok=True)
        for name, data in documents.items():
            write_atomically(target / name, data, mode=0o644)
        return [target / name for name in documents]

    return pathless(copy)


def _strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
