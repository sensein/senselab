"""A minimal BIDS adapter: audio files named `key-value_..._suffix.ext` read as a dataset."""

import csv
import json
import os
import re
import unicodedata
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from senselab_core.kernel.dataset import Dataset, FileInput, Record
from senselab_core.runtime.datasets import RecordSet

_PAIR = re.compile(r"^([a-z]+)-([A-Za-z0-9]+)$")
_SUFFIX = re.compile(r"^[A-Za-z0-9]+$")
_NO_SESSION = 3
_RESERVED = frozenset({"datatype", "suffix", "extension"})
_NOT_RAW = frozenset({"derivatives", "sourcedata", "code"})


class BidsError(ValueError):
    """A tree the adapter refuses to read."""


class BidsCollision(BidsError):
    """Two paths of a tree differ only in letter case."""


class BidsDataset(RecordSet):
    """A dataset read from a BIDS tree; `ignored` counts the files of a requested extension outside the grammar."""

    def __init__(self, records: list[Record], root: Path, ignored: int) -> None:
        """Hold `records` read under `root`."""
        super().__init__(records, root)
        self.ignored = ignored


def parse_name(stem: str) -> tuple[dict[str, str], str] | None:
    """Return the entities and the suffix of a file stem `key-value_..._suffix`, or None outside that grammar."""
    *pairs, suffix = stem.split("_")
    entities: dict[str, str] = {}
    for part in pairs:
        match = _PAIR.match(part)
        if match is None or match.group(1) in entities or match.group(1) in _RESERVED:
            return None
        entities[match.group(1)] = match.group(2)
    if not pairs or not _SUFFIX.match(suffix):
        return None
    return entities, suffix


def _split(name: str) -> tuple[str, str]:
    stem, dot, extension = name.partition(".")
    return stem, dot + extension


def _checked(root: Path, relative: str, path: Path) -> None:
    try:
        target = path.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise BidsError(f"{relative} is a broken or looping link") from exc
    if not target.is_relative_to(root):
        raise BidsError(f"{relative} links outside {root}")


def _participants(root: Path) -> dict[str, dict[str, str]]:
    table = root / "participants.tsv"
    if not table.is_file():
        return {}
    with table.open(newline="", encoding="utf-8") as f:
        return {
            row["participant_id"]: dict(row) for row in csv.DictReader(f, delimiter="\t") if "participant_id" in row
        }


def _read_json(path: Path, root: Path) -> dict[str, Any]:
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise BidsError(f"{path.relative_to(root).as_posix()} is not JSON") from exc
    if not isinstance(content, dict):
        raise BidsError(f"{path.relative_to(root).as_posix()} is not a JSON object")
    return content


class _Sidecars:
    """The JSON files of a tree, read once per directory."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._levels: dict[Path, list[tuple[dict[str, str], str, int, Path]]] = {}

    def _level(self, directory: Path) -> list[tuple[dict[str, str], str, int, Path]]:
        if directory not in self._levels:
            found = []
            for candidate in sorted(directory.glob("*.json")):
                parsed = parse_name(candidate.stem)
                if parsed is not None:
                    found.append((parsed[0], parsed[1], len(parsed[0]), candidate))
            self._levels[directory] = found
        return self._levels[directory]

    def of(self, directory: Path, entities: Mapping[str, str], suffix: str) -> dict[str, Any]:
        """Merge the JSON files that apply to a file, root first, within a directory fewer entities first."""
        merged: dict[str, Any] = {}
        relative = directory.relative_to(self._root)
        for level in [self._root, *(self._root / Path(*relative.parts[: i + 1]) for i in range(len(relative.parts)))]:
            applicable = [
                (count, path)
                for named, kind, count, path in self._level(level)
                if kind == suffix and all(entities.get(k) == v for k, v in named.items())
            ]
            for _, path in sorted(applicable, key=lambda item: (item[0], item[1].name)):
                merged.update(_read_json(path, self._root))
        return merged


def _placed(parts: tuple[str, ...], entities: Mapping[str, str]) -> bool:
    """Whether `parts` is `sub-<label>/[ses-<label>/]<datatype>/<file>` with labels equal to the name's."""
    if len(parts) not in (3, 4) or parts[0] != f"sub-{entities.get('sub')}":
        return False
    return len(parts) == _NO_SESSION or parts[1] == f"ses-{entities.get('ses')}"


def _scan(root: Path, extensions: tuple[str, ...]) -> dict[str, Path]:
    """Return the files of `extensions` of the raw tree by their normalised name, after checking the links read."""
    found: dict[str, Path] = {}
    links: list[Path] = [root / "participants.tsv"] if (root / "participants.tsv").exists() else []
    for directory, dirs, files in os.walk(root):
        here = Path(directory)
        top = here == root
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and not (top and d in _NOT_RAW))
        links.extend(here / d for d in dirs)
        for name in sorted(files):
            if _split(name)[1].lower() not in extensions:
                continue
            path = here / name
            relative = path.relative_to(root).as_posix()
            key = unicodedata.normalize("NFC", relative).casefold()
            if key in found:
                raise BidsCollision(f"{found[key].relative_to(root).as_posix()} and {relative}")
            found[key] = path
            links.append(path)
    for link in links:
        if link.is_symlink() or not link.exists():
            _checked(root, link.relative_to(root).as_posix(), link)
    return found


def bids_dataset(
    root: Path,
    *,
    datatype: str | None = None,
    suffix: str | None = None,
    extensions: tuple[str, ...] = (".wav", ".flac"),
) -> Dataset:
    """Return the files of the BIDS tree at `root` with one of `extensions`, one record per file.

    A file is read only at `sub-<label>/[ses-<label>/]<datatype>/<name>`, with a name `key-value_..._suffix.ext`
    whose `sub` and `ses` equal the directories'; any other file of those extensions is counted in `ignored`.
    Dot directories and the top-level `derivatives`, `sourcedata` and `code` are not read. A record's id is the
    file's root-relative POSIX path (NFC); its entities are the name's `key-value` pairs plus `datatype`,
    `suffix` and `extension`; its one input is named by the suffix. Its sidecar is the `participants.tsv` row of its
    `sub` and the JSON files that apply to it, nearer ones overriding. `datatype` and `suffix` keep only matching
    files. `filter` and `shard` return plain datasets, without `ignored`.

    Raises:
        BidsCollision: if two paths differ only in letter case or Unicode form.
        BidsError: if a file or directory read is a link leaving `root` or a broken or looping link, a sidecar is
            not a JSON object, or two files have the same entities.
    """
    root = root.resolve()
    found = _scan(root, extensions)
    participants = _participants(root)
    sidecars = _Sidecars(root)
    records: dict[frozenset[tuple[str, str]], Record] = {}
    ignored = 0
    for path in sorted(found.values()):
        stem, extension = _split(path.name)
        parsed = parse_name(stem)
        parts = path.relative_to(root).parts
        if parsed is None or not _placed(parts, parsed[0]):
            ignored += 1
            continue
        entities, kind = parsed
        place = parts[-2]
        if (datatype is not None and place != datatype) or (suffix is not None and kind != suffix):
            continue
        relative = path.relative_to(root).as_posix()
        sub = participants.get(f"sub-{entities['sub']}", {})
        sidecar = {k: v for k, v in sub.items() if k != "participant_id"}
        sidecar.update(sidecars.of(path.parent, entities, kind))
        full = {**entities, "datatype": place, "suffix": kind, "extension": extension.lower()}
        if frozenset(full.items()) in records:
            raise BidsError(f"{relative} has the same entities as {records[frozenset(full.items())].id}")
        records[frozenset(full.items())] = Record(
            id=unicodedata.normalize("NFC", relative),
            entities=full,
            inputs={kind: FileInput(path=relative)},
            sidecar=sidecar,
        )
    return BidsDataset(list(records.values()), root, ignored)
