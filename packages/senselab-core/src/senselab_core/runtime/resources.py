"""Resource tables (`resource-table@1.0`): what one task and backend measured per model and resource class.

A table is a generated file `<root>/<task>/<backend>.yaml`, never hand-edited: packaged under `data/resources` by
`scripts/generate_resources.py` from scale-ladder results, or written under the cache by a calibration round
(`Measures.put`). Each row names the model by `(id, revision)` (both None for a backend without a model), its download
size, and per resource class the seconds to load, the memory resident after it and at peak, the seconds per input
second and the input seconds per item, with the run and jobs that measured them.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, ClassVar, Protocol

from pydantic import AfterValidator, Field, PlainSerializer, StringConstraints, model_validator
from senselab_worker.digest import COMMIT_HEX, SHA256_HEX

from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, thawed
from senselab_core.kernel.formats import RESOURCE_TABLE_FORMAT, FormatTag, Formatted
from senselab_core.kernel.placement import ResourceClassName
from senselab_core.runtime.config import dump_yaml, load_yaml
from senselab_core.runtime.storage import write_atomically

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class Batched(Frozen):
    """Seconds of work per second of input of items sent `batch_size` per request, at the batch measured."""

    batch_size: Annotated[int, Field(ge=2)]
    s_per_input_s: Seconds


class ResourceMeasure(Frozen):
    """One model on one resource class at one item per request.

    Attributes:
        load_s: Seconds from a worker's start to its model loaded.
        claim_s: Seconds a worker takes to claim one item.
        resident_bytes: Host memory held once loaded, between requests; None where it could not be read.
        peak_memory_bytes: The host high-water memory over the measured requests; None with `resident_bytes`.
        device_resident_bytes: Device memory held once loaded, between requests; None off a device.
        device_peak_bytes: The highest device memory read between requests; None off a device.
        s_per_input_s: Seconds of work per second of recording input; per item when `input_s_per_item` is None.
        input_s_per_item: The median recording seconds of a measured item; None for items without a recording.
        batched: The seconds per input second at the largest batch measured; None where none was measured.
    """

    load_s: Seconds
    claim_s: Seconds
    resident_bytes: Annotated[int, Field(ge=0)] | None
    peak_memory_bytes: Annotated[int, Field(gt=0)] | None
    device_resident_bytes: Annotated[int, Field(ge=0)] | None = None
    device_peak_bytes: Annotated[int, Field(gt=0)] | None = None
    s_per_input_s: Seconds
    input_s_per_item: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None
    batched: Batched | None = None

    @model_validator(mode="after")
    def _check_memory(self) -> "ResourceMeasure":
        pairs = [(self.resident_bytes, self.peak_memory_bytes), (self.device_resident_bytes, self.device_peak_bytes)]
        if (self.device_resident_bytes is None) != (self.device_peak_bytes is None):
            raise ValueError("device resident and peak memory are both measured, or neither")
        if (self.resident_bytes is None) != (self.peak_memory_bytes is None):
            raise ValueError("host resident and peak memory are both measured, or neither")
        for resident, peak in pairs:
            if resident is not None and peak is not None and resident > peak:
                raise ValueError(f"resident {resident} bytes exceed the peak {peak}")
        return self


class Measured(Frozen):
    """Where a class was measured: the 40-hex commit run (None for a calibration round), the run and the job ids."""

    sha: Annotated[str, StringConstraints(pattern=rf"^{COMMIT_HEX.pattern}$")] | None
    run: NonEmptyStr | None = None
    jobs: tuple[NonEmptyStr, ...] = Field(min_length=1)


class ClassMeasure(ResourceMeasure):
    """A resource class's measure in a table, with where it was measured."""

    measured: Measured


def _frozen(v: Mapping[str, object]) -> Mapping[str, object]:
    return FrozenMapping(v)


class ResourceRow(Frozen):
    """One model's measures: its `(id, revision)`, download bytes (None unmeasured) and a measure per resource class."""

    id: NonEmptyStr | None
    revision: NonEmptyStr | None
    download_bytes: Annotated[int, Field(ge=0)] | None
    classes: Annotated[Mapping[ResourceClassName, ClassMeasure], AfterValidator(_frozen), PlainSerializer(thawed)]

    @model_validator(mode="after")
    def _check_model(self) -> "ResourceRow":
        if (self.id is None) != (self.revision is None):
            raise ValueError("a row names both id and revision, or neither for a backend without a model")
        if self.revision is not None and not (
            COMMIT_HEX.fullmatch(self.revision) or SHA256_HEX.fullmatch(self.revision)
        ):
            raise ValueError(f"revision must be a 40-hex commit or a 64-hex weights digest, got {self.revision!r}")
        if not self.classes:
            raise ValueError("a row measures at least one resource class")
        return self


class ResourceTable(Formatted):
    """A `resource-table@1.0` document: the rows of one task and backend, one per model."""

    format: FormatTag = RESOURCE_TABLE_FORMAT
    port_value: ClassVar[bool] = False
    task: NonEmptyStr
    backend: NonEmptyStr
    rows: tuple[ResourceRow, ...]

    @model_validator(mode="after")
    def _check_rows(self) -> "ResourceTable":
        models = [(r.id, r.revision) for r in self.rows]
        if len(set(models)) != len(models):
            raise ValueError(f"{self.task}/{self.backend}: a model has two rows")
        return self

    def row(self, model_id: str | None, revision: str | None) -> ResourceRow | None:
        """Return the row of `(model_id, revision)`, or None."""
        return next((r for r in self.rows if (r.id, r.revision) == (model_id, revision)), None)


def load_resource_tables(root: Traversable | Path) -> tuple[ResourceTable, ...]:
    """Return every table under `root`, `<task>/<backend>.yaml`, sorted by task and backend; none when it is absent.

    Raises:
        ConfigError: a file cannot be read as YAML.
        ValueError: a file's table names another task or backend than its path, or does not validate.
    """
    if not root.is_dir():
        return ()
    tables = []
    for task_dir in sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name):
        for file in sorted((f for f in task_dir.iterdir() if f.name.endswith(".yaml")), key=lambda f: f.name):
            table = ResourceTable.model_validate(load_yaml(file, "resource table"))
            if (table.task, table.backend) != (task_dir.name, file.name.removesuffix(".yaml")):
                raise ValueError(f"{task_dir.name}/{file.name} holds the table of {table.task}/{table.backend}")
            tables.append(table)
    return tuple(tables)


@dataclass(frozen=True)
class Unit:
    """What one measure is of: a task's backend, its model as `(id, revision)` (None without one), a resource class."""

    task: str
    backend: str
    model_id: str | None
    revision: str | None
    resource_class: str


class MeasureSource(Protocol):
    """Where a unit's measure is read, and a calibration round's written."""

    def get(self, unit: Unit) -> ResourceMeasure | None:
        """Return the measure of `unit`, None when unmeasured."""

    def put(self, unit: Unit, measure: ResourceMeasure, measured: Measured) -> None:
        """Record `measure` of `unit`, measured as `measured`."""


def write_row(root: Path, unit: Unit, measure: ResourceMeasure, measured: Measured) -> Path:
    """Write `measure` of `unit` into the table `<root>/<task>/<backend>.yaml`; return the file.

    Only the unit's class of its model's row is replaced, with `measured` as its derivation; the row keeps its other
    classes and its download bytes. The file is replaced whole.
    """
    path = Path(root) / unit.task / f"{unit.backend}.yaml"
    table = ResourceTable.model_validate(load_yaml(path, "resource table")) if path.is_file() else None
    rows = list(table.rows) if table is not None else []
    old = next((r for r in rows if (r.id, r.revision) == (unit.model_id, unit.revision)), None)
    entry = ClassMeasure(**ResourceMeasure.model_validate(measure.model_dump()).model_dump(), measured=measured)
    classes = {**(dict(old.classes) if old is not None else {}), unit.resource_class: entry}
    row = ResourceRow(
        id=unit.model_id,
        revision=unit.revision,
        download_bytes=None if old is None else old.download_bytes,
        classes=classes,
    )
    rows = [r for r in rows if r is not old] + [row]
    doc = ResourceTable(task=unit.task, backend=unit.backend, rows=tuple(sorted(rows, key=_row_order)))
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, dump_yaml(doc.model_dump(mode="json")).encode())
    return path


def _row_order(row: ResourceRow) -> tuple[str, str]:
    return (row.id or "", row.revision or "")
