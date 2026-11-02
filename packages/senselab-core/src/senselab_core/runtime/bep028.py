"""Export redaction and the BEP028 profile (ADR-007, ADR-010).

A file is named in provenance by its content address and a per-run label, `file-<n>`. The label-to-path map is
the run's private `<run>/private/labels@1.json` (mode 0600); it is never copied or exported. `export` copies a run's
documents after `check_profile` finds the BEP028 profile met and after no path of the map is found in them.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path

from senselab_worker.canonical import jcs_dumps

from senselab_core.kernel import prov as kprov
from senselab_core.kernel.formats import LABELS_FORMAT, FormatVerdict, check
from senselab_core.runtime.locks import Lease
from senselab_core.runtime.provenance import (
    FILE_LABEL,
    PROV_DIR,
    ConflictingNode,
    DanglingReference,
    MalformedDocument,
    ProvenanceFormatRefused,
    ProvStore,
    pathless,
)
from senselab_core.runtime.storage import PRIVATE_DIR_MODE, StorageDir, write_atomically
from senselab_core.runtime.vocabulary import core_vocabulary

PRIVATE_DIR = "private"
"""The run subdirectory that holds the label map; nothing in it is exported."""
LABELS_NAME = "labels@1.json"
"""The label map's file name in `<run>/private/`."""
LABELS_MODE = 0o600
"""The mode of the label map and of its lease."""
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
            check(LABELS_FORMAT, record["format"]) is FormatVerdict.ACCEPTED
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
        return pathless("assigning file labels", lambda: self._assign(items))

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

        return pathless("reading the label map", read)

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


_VALIDATION_ERRORS = (MalformedDocument, ConflictingNode, DanglingReference, ProvenanceFormatRefused)
_WHOLE_DOCUMENT = "0" * 32 + ".jsonld"
_TOKENS = re.compile(r"[/\\\s\"',;=]+")
_SUBJECT = re.compile(r"(?<![a-z0-9])(?:sub|ses)-[a-z0-9]+")
_LAYOUT_NAME = "prov-{writer}_all.jsonld"


def layout_name(writer_document: str) -> str:
    """Return the BEP028 name of a run's document `<writer>.jsonld`: `prov-<writer>_all.jsonld`, in `prov/`."""
    return _LAYOUT_NAME.format(writer=writer_document.removesuffix(".jsonld"))


def check_profile(doc: Mapping[str, object]) -> list[str]:
    """Return the ways `doc`, a `provenance@3` document, leaves the profile; none when it conforms.

    `doc` is validated as `ProvStore.from_documents` validates a writer's document. On top of that: an activity's
    end has a start and is not before it, and a `senselab:fileLabel` names one node.
    """
    try:
        store = ProvStore.from_documents({_WHOLE_DOCUMENT: jcs_dumps(dict(doc))})
    except _VALIDATION_ERRORS as err:
        return [str(err)]
    except ValueError:
        return ["the document has no canonical JSON form"]
    problems: list[str] = []
    labels: dict[str, str] = {}
    for record in store.records():
        node = record.node()
        if "prov:Activity" in record.types():
            problems += _time_problems(record.id, node)
        label = node.get("senselab:fileLabel")
        if isinstance(label, str) and labels.setdefault(label, record.id) != record.id:
            problems.append(f"{record.id} has a senselab:fileLabel another node has")
    return problems


def _time_problems(name: str, node: Mapping[str, object]) -> list[str]:
    start, end = node.get("prov:startedAtTime"), node.get("prov:endedAtTime")
    if end is None:
        return []
    if start is None:
        return [f"{name} ends and never starts"]
    began, ended = kprov.literal_value(start), kprov.literal_value(end)
    if isinstance(began, datetime) and isinstance(ended, datetime) and ended < began:
        return [f"{name} ends before it starts"]
    return []


def _strings(value: object) -> Iterable[str]:
    """Yield every string of a parsed JSON value: its values and its object keys."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)


def _spelling(text: str) -> str:
    """Return `text` with backslashes as `/` and empty and `.` segments dropped."""
    return "/".join(t for t in text.replace("\\", "/").split("/") if t not in ("", "."))


def _fold(text: str) -> str:
    """Return `text` in NFC and casefolded, so the spelling of a letter does not decide a match."""
    return unicodedata.normalize("NFC", text).casefold()


def _leaks(documents: Iterable[bytes], paths: set[str]) -> bool:
    """Whether a string of a document holds a recorded path, its basename, or a BIDS subject label.

    Strings and paths are compared in NFC, casefolded and with backslashes as `/`. A path is held whole as a substring,
    however spelt; its basename is held as a substring, or as the stem of a token (`clip.json` for `clip.wav`). A
    directory segment alone is not: a port `audio` is not the path `audio/clip.wav`.
    """
    whole = {_spelling(_fold(p)) for p in paths} - {""}
    names = {Path(w).name for w in whole}
    stems = {Path(n).stem + "." for n in names}
    for data in documents:
        for text in _strings(json.loads(data)):
            folded = _fold(text)
            spelled = _spelling(folded)
            tokens = [t for t in _TOKENS.split(folded) if t]
            if (
                _SUBJECT.search(folded)
                or any(w in spelled for w in whole)
                or any(n in spelled for n in names)
                or any(t.startswith(stem) for t in tokens for stem in stems)
            ):
                return True
    return False


def export(run: str | Path, dest: str | Path, vocabulary: kprov.Vocabulary | None = None) -> list[Path]:
    """Copy a run's provenance documents to `<dest>/prov/prov-<writer>_all.jsonld` and return the copies.

    The documents are read once; the bytes validated, profile-checked and scanned are the bytes copied. The run's
    private directory is never copied. A document may not hold a path of the run's label map or its basename, however
    spelt (`./`, `//`, backslashes), nor a BIDS `sub-`/`ses-` label.

    Raises:
        ProfileViolation: a profile violation or a leak; nothing is copied.
        ProvenanceFormatRefused, MalformedDocument, ConflictingNode, DanglingReference: as `ProvStore.from_documents`.
        RunIOError: the label map or a copy cannot be read or written; the error names no path.
    """
    documents = ProvStore.read_documents(run)
    problems = check_profile(ProvStore.from_documents(documents, vocabulary).document())
    if _leaks(documents.values(), {p for group in LabelMap(run).paths().values() for p in group}):
        problems.append("a document names a path of the run's label map, or a BIDS subject")
    if problems:
        raise ProfileViolation(problems)
    target = Path(dest) / PROV_DIR

    def copy() -> list[Path]:
        target.mkdir(parents=True, exist_ok=True)
        copies = []
        for name, data in documents.items():
            write_atomically(target / layout_name(name), data, mode=0o644)
            copies.append(target / layout_name(name))
        return copies

    return pathless("exporting the provenance documents", copy)
