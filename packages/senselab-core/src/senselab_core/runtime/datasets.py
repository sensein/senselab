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
from senselab_core.kernel.declare import kernel_types
from senselab_core.kernel.identity import InputIdentity
from senselab_core.runtime.call import bind_input


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

    def locate(self, path: str) -> str:
        """Return the file system path of a `FileInput.path`: under the root, or as given without one."""
        return path if self._root is None else str(self._root / path)

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
        """Return shard `index` of `count`: the records whose entity `by` hashes to `index`, a value never split.

        Membership depends on a record's own `by` value and `count` only, so it commutes with `filter` and a
        record added to the dataset moves no other.

        Raises:
            ValueError: if `index` is outside `[0, count)` or a record has no entity `by`.
        """
        if count < 1 or not 0 <= index < count:
            raise ValueError(f"shard {index} of {count} is not a shard")
        missing = [r.id for r in self._records if by not in r.entities]
        if missing:
            raise ValueError(f"records {missing[:3]} have no entity {by!r}; a shard is closed under {by!r}")
        kept = [
            r
            for r in self._records
            if int(value_digest("shard", {"by": by, "value": r.entities[by]}), 16) % count == index
        ]
        return RecordSet(kept, self._root)


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


def build_manifest(
    dataset: Dataset, ports: Mapping[str, str], *, bind: Mapping[str, str] | None = None
) -> InputManifest:
    """Return the manifest of `dataset` for a graph whose inputs are `ports`, as port name to port type.

    See `sourced_manifest`, which also returns what each row's inputs were read from.
    """
    return sourced_manifest(dataset, ports, bind=bind)[0]


def sourced_manifest(
    dataset: Dataset, ports: Mapping[str, str], *, bind: Mapping[str, str] | None = None
) -> tuple[InputManifest, list[dict[str, object]]]:
    """Return the manifest of `dataset` for a graph whose inputs are `ports`, and each row's input sources by port.

    A source is what `bind_input` reads the input from: the file's path as text, or the JSON value.

    Each port binds to the record input of the same name; `bind[port]` names another input instead (for BIDS, a
    suffix). A file input is read as `bind_input` reads a path for the port's type, so its identity is the one a
    run keys on and comes from the file's content.

    Raises:
        ValueError: if a record has no input for a port, or a file input is bound to a port type not read from a path.
    """
    chosen = bind or {}
    rows = []
    sources: list[dict[str, object]] = []
    for record in dataset.records():
        identities: dict[str, InputIdentity] = {}
        read: dict[str, object] = {}
        for port, port_type in ports.items():
            ref = record.inputs.get(chosen.get(port, port))
            if ref is None:
                raise ValueError(
                    f"record {record.id!r} has no input for port {port!r} (looked for {chosen.get(port, port)!r}); "
                    f"it has {sorted(record.inputs)}"
                )
            if isinstance(ref, FileInput):
                if port_type not in kernel_types() or port_type == "Text":
                    raise ValueError(f"port {port!r} is a {port_type}, which is not read from a file")
                value: object = dataset.locate(ref.path)
            else:
                value = ref.value
            identities[port] = bind_input(port, port_type, value)[0]
            read[port] = value
        rows.append(ManifestRow(id=record.id, entities=record.entities, inputs=identities))
        sources.append(read)
    return InputManifest(rows=tuple(rows)), sources
