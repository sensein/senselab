"""Dataset constructors: in-memory records, and input manifests built from a dataset's files."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path, PurePath
from typing import Any

from senselab_core.kernel._base import value_digest
from senselab_core.kernel.dataset import (
    Dataset,
    FileInput,
    InputManifest,
    InputRef,
    ManifestRow,
    Record,
    ValueInput,
)
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.audio_io import open_audio


class RecordSet:
    """A `Dataset` over records held in memory, with the `root` its file inputs are relative to."""

    def __init__(self, records: Iterable[Record], root: Path | None = None) -> None:
        """Hold `records` sorted by id.

        Raises:
            ValueError: if two records share an id.
        """
        self._records = tuple(sorted(records, key=lambda r: r.id))
        ids = [r.id for r in self._records]
        if len(set(ids)) != len(ids):
            raise ValueError(f"record ids repeat: {sorted({i for i in ids if ids.count(i) > 1})}")
        self._root = root

    @property
    def root(self) -> Path | None:
        """The directory file inputs are relative to."""
        return self._root

    def records(self) -> Iterator[Record]:
        """Yield the records sorted by id."""
        return iter(self._records)

    def entities(self) -> Mapping[str, tuple[str, ...]]:
        """Return each entity name with the sorted values it takes."""
        found: dict[str, set[str]] = {}
        for record in self._records:
            for name, value in record.entities.items():
                found.setdefault(name, set()).add(value)
        return {name: tuple(sorted(values)) for name, values in sorted(found.items())}

    def filter(self, **entities: str | Sequence[str]) -> Dataset:
        """Return the records whose entity `name` is the given value, or one of the given values."""
        wanted = {n: {v} if isinstance(v, str) else set(v) for n, v in entities.items()}
        kept = [r for r in self._records if all(r.entities.get(n) in values for n, values in wanted.items())]
        return RecordSet(kept, self._root)

    def shard(self, by: str, index: int, count: int) -> Dataset:
        """Return shard `index` of `count`: the sorted values of entity `by` dealt round-robin, a value never split.

        Raises:
            ValueError: if `index` is outside `[0, count)` or a record has no entity `by`.
        """
        if count < 1 or not 0 <= index < count:
            raise ValueError(f"shard {index} of {count} is not a shard")
        missing = [r.id for r in self._records if by not in r.entities]
        if missing:
            raise ValueError(f"records {missing[:3]} have no entity {by!r}; a shard is closed under {by!r}")
        values = sorted({r.entities[by] for r in self._records})
        mine = {v for i, v in enumerate(values) if i % count == index}
        return RecordSet([r for r in self._records if r.entities[by] in mine], self._root)


def _ref(value: Any) -> InputRef:  # noqa: ANN401 - a row's input is any JSON value or a path
    if isinstance(value, PurePath):
        return FileInput(path=value.as_posix())
    return ValueInput(value=value)


def from_records(rows: Iterable[Mapping[str, Any]], *, id_key: str = "id") -> Dataset:
    """Return a dataset of `rows`: mappings with the id at `id_key` and optional `entities`, `inputs` and `sidecar`.

    An input given as a `Path` is a file, used as given; any other input is a JSON value.

    Raises:
        ValueError: if a row has no `id_key` or two rows share an id.
    """
    records = []
    for row in rows:
        if id_key not in row:
            raise ValueError(f"a row has no {id_key!r}: {sorted(row)}")
        records.append(
            Record(
                id=str(row[id_key]),
                entities={k: str(v) for k, v in row.get("entities", {}).items()},
                inputs={k: _ref(v) for k, v in row.get("inputs", {}).items()},
                sidecar=row.get("sidecar", {}),
            )
        )
    return RecordSet(records)


def _identity(root: Path | None, ref: InputRef) -> InputIdentity:
    if isinstance(ref, ValueInput):
        return "content:" + value_digest("json", {"value": ref.value})
    path = Path(ref.path) if root is None else root / ref.path
    return open_audio(path).identity


def build_manifest(dataset: Dataset, ports: Mapping[str, str]) -> InputManifest:
    """Return the manifest of `dataset`: each record's identity of the input `ports[port]` it names, per port.

    A file input is stamped (its bytes hashed in chunks) and its identity is the recording's; a value input's is
    the digest of its JSON.

    Raises:
        ValueError: if a record has no input named for a port.
    """
    rows = []
    for record in dataset.records():
        missing = sorted(key for key in ports.values() if key not in record.inputs)
        if missing:
            raise ValueError(f"record {record.id!r} has no input {missing}")
        rows.append(
            ManifestRow(
                id=record.id,
                entities=record.entities,
                inputs={port: _identity(dataset.root, record.inputs[key]) for port, key in ports.items()},
            )
        )
    return InputManifest(rows=tuple(rows))
