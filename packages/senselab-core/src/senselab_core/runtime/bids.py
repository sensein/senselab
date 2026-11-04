"""A minimal BIDS adapter: audio files named `key-value_..._suffix.ext` read as a dataset."""

import csv
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from senselab_core.kernel.dataset import Dataset, FileInput, Record
from senselab_core.runtime.datasets import RecordSet

_PAIR = re.compile(r"^([a-z]+)-([A-Za-z0-9]+)$")
_SUFFIX = re.compile(r"^[A-Za-z0-9]+$")


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
        if match is None or match.group(1) in entities:
            return None
        entities[match.group(1)] = match.group(2)
    if not pairs or not _SUFFIX.match(suffix):
        return None
    return entities, suffix


def _split(name: str) -> tuple[str, str]:
    stem, dot, extension = name.partition(".")
    return stem, dot + extension


def _inside(root: Path, path: Path) -> bool:
    return path.resolve().is_relative_to(root)


def _participants(root: Path) -> dict[str, dict[str, str]]:
    table = root / "participants.tsv"
    if not table.is_file():
        return {}
    with table.open(newline="", encoding="utf-8") as f:
        return {
            row["participant_id"]: dict(row) for row in csv.DictReader(f, delimiter="\t") if "participant_id" in row
        }


def _sidecars(root: Path, directory: Path, entities: Mapping[str, str], suffix: str) -> dict[str, Any]:
    """Merge the JSON files that apply to a file, root first, within a directory fewer entities first."""
    merged: dict[str, Any] = {}
    chain = [root, *(root / p for p in reversed(directory.relative_to(root).parents) if p != Path("."))]
    if directory != root:
        chain.append(directory)
    for level in dict.fromkeys(chain):
        found = []
        for candidate in level.glob("*.json"):
            parsed = parse_name(candidate.stem)
            if parsed is None or parsed[1] != suffix:
                continue
            if all(entities.get(k) == v for k, v in parsed[0].items()):
                found.append((len(parsed[0]), candidate))
        for _, candidate in sorted(found, key=lambda item: (item[0], item[1].name)):
            merged.update(json.loads(candidate.read_text(encoding="utf-8")))
    return merged


def bids_dataset(
    root: Path,
    *,
    datatype: str | None = None,
    suffix: str | None = None,
    extensions: tuple[str, ...] = (".wav", ".flac"),
) -> Dataset:
    """Return the files of the BIDS tree at `root` with one of `extensions`, one record per file.

    A record's id is the file's root-relative POSIX path; its entities are the `key-value` pairs of the name plus
    `datatype` (the directory holding the file) and `suffix`; its one input is named by the suffix. Its sidecar
    is the `participants.tsv` row of its `sub` and the JSON files that apply to it, nearer ones overriding. `datatype`
    and `suffix` keep only matching files.

    Raises:
        BidsCollision: if two paths differ only in letter case.
        BidsError: if a file or directory is a link leaving `root`.
    """
    root = root.resolve()
    participants = _participants(root)
    found: dict[str, Path] = {}
    ignored = 0
    for directory, dirs, files in os.walk(root):
        dirs.sort()
        here = Path(directory)
        for name in [*dirs, *files]:
            if not _inside(root, here / name):
                raise BidsError(f"{(here / name).relative_to(root).as_posix()} links outside {root}")
        for name in sorted(files):
            stem, extension = _split(name)
            if extension not in extensions:
                continue
            path = here / name
            key = path.relative_to(root).as_posix().casefold()
            if key in found:
                raise BidsCollision(
                    f"{found[key].relative_to(root).as_posix()} and {path.relative_to(root).as_posix()}"
                )
            found[key] = path
    records = []
    for path in sorted(found.values()):
        stem, _ = _split(path.name)
        parsed = parse_name(stem)
        if parsed is None:
            ignored += 1
            continue
        entities, kind = parsed
        place = path.parent.name
        if (datatype is not None and place != datatype) or (suffix is not None and kind != suffix):
            continue
        relative = path.relative_to(root).as_posix()
        sub = participants.get(f"sub-{entities['sub']}", {}) if "sub" in entities else {}
        sidecar = {k: v for k, v in sub.items() if k != "participant_id"}
        sidecar.update(_sidecars(root, path.parent, entities, kind))
        records.append(
            Record(
                id=relative,
                entities={**entities, "datatype": place, "suffix": kind},
                inputs={kind: FileInput(path=relative)},
                sidecar=sidecar,
            )
        )
    return BidsDataset(records, root, ignored)
