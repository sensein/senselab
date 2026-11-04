"""Datasets: records of named inputs, the `Dataset` protocol, and the input manifest `input-manifest@1.0`.

A `Record` names its inputs, either files under a dataset's root or values. A `Dataset` is read through `records`,
`entities`, `filter` and `shard`. `InputManifest` lists each record's id, entities and the content identity of
every input a graph binds; `InputManifest.digest` identifies it and does not depend on where the root is.
"""

from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, ClassVar, Protocol

from pydantic import AfterValidator, PlainSerializer, StringConstraints, model_validator
from senselab_worker.digest import identity_digest

from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr
from senselab_core.kernel.formats import INPUT_MANIFEST_FORMAT, FormatTag, Formatted, later_minor
from senselab_core.kernel.identity import InputIdentity, Params

MANIFEST_DIGEST_TAG = "input-manifest@1"
"""The identity tag of an input manifest's canonical JSON."""

RecordId = Annotated[str, StringConstraints(min_length=1)]
"""A record's id, stable across runs: the root-relative POSIX path of a BIDS file, the given id in memory."""


def _frozen(value: Mapping[str, Any]) -> FrozenMapping[str, Any]:
    return FrozenMapping(value)


class FileInput(Frozen):
    """An input read from a file: `path` is POSIX text relative to the dataset root, or as given without a root."""

    path: NonEmptyStr


class ValueInput(Frozen):
    """An input given as a JSON value."""

    value: Any


InputRef = FileInput | ValueInput
"""What a record's input names: a file or a value."""


class Record(Frozen):
    """One item of a dataset: its id, its entities, its named inputs and its sidecar metadata.

    `sidecar` is read from files beside the data and is not part of a manifest's identity.
    """

    id: RecordId
    entities: Annotated[Mapping[str, str], AfterValidator(_frozen), PlainSerializer(dict)] = FrozenMapping({})
    inputs: Annotated[Mapping[str, InputRef], AfterValidator(_frozen), PlainSerializer(dict)] = FrozenMapping({})
    sidecar: Params = FrozenMapping({})


class Dataset(Protocol):
    """Records a graph runs over. `root` is the directory its `FileInput` paths are relative to, or None."""

    @property
    def root(self) -> Path | None:
        """The directory file inputs are relative to."""
        ...

    def records(self) -> Iterator[Record]:
        """Yield the records sorted by id."""
        ...

    def entities(self) -> Mapping[str, tuple[str, ...]]:
        """Return each entity name with the sorted values it takes."""
        ...

    def filter(self, **entities: str | Sequence[str]) -> "Dataset":
        """Return the records whose entity `name` is the given value, or one of the given values."""
        ...

    def shard(self, by: str, index: int, count: int) -> "Dataset":
        """Return shard `index` of `count`: the sorted values of entity `by` dealt round-robin, a value never split.

        Raises:
            ValueError: if `index` is outside `[0, count)` or a record has no entity `by`.
        """
        ...


class ManifestRow(Frozen):
    """One record of a manifest: its id, its entities and the identity of each bound input."""

    id: RecordId
    entities: Annotated[Mapping[str, str], AfterValidator(_frozen), PlainSerializer(dict)]
    inputs: Annotated[Mapping[str, InputIdentity], AfterValidator(_frozen), PlainSerializer(dict)]


class InputManifest(Formatted):
    """An `input-manifest@1.0` document: rows sorted by id, one per record, identified by `digest`."""

    format: FormatTag = INPUT_MANIFEST_FORMAT
    port_value: ClassVar[bool] = False
    rows: tuple[ManifestRow, ...] = ()

    @model_validator(mode="after")
    def _sorted_unique(self) -> "InputManifest":
        ids = [r.id for r in self.rows]
        if ids != sorted(set(ids)):
            raise ValueError("manifest rows are sorted by id, each id once")
        return self

    def digest(self) -> str:
        """Return `identity_digest("input-manifest@1", {"rows": ...})`.

        Raises:
            ValueError: if the manifest declares a later minor than `input-manifest@1.0`.
        """
        if later_minor(INPUT_MANIFEST_FORMAT, self.format):
            raise ValueError(f"a {self.format} manifest has fields a {INPUT_MANIFEST_FORMAT} reader does not hold")
        return identity_digest(MANIFEST_DIGEST_TAG, {"rows": self.model_dump(mode="json")["rows"]})
