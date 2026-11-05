"""Resource tables (`resource-table@1.0`): what one task and backend measured per model and resource class.

A table is a generated file `data/resources/<task>/<backend>.yaml`; it is never hand-edited. Each row names the
model by `(id, revision)` (both None for a backend without a model), its download size, and per resource class the
seconds to load, the peak memory and the seconds per input second, with the commit and jobs that measured them.
"""

from collections.abc import Mapping
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import AfterValidator, Field, PlainSerializer, StringConstraints, model_validator
from senselab_worker.digest import COMMIT_HEX, SHA256_HEX

from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, thawed
from senselab_core.kernel.formats import RESOURCE_TABLE_FORMAT, FormatTag, Formatted
from senselab_core.kernel.placement import ResourceClassName
from senselab_core.runtime.config import load_yaml

Seconds = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ResourceMeasure(Frozen):
    """One model on one resource class: seconds to load, peak memory in bytes, seconds per input second."""

    load_s: Seconds
    peak_memory_bytes: Annotated[int, Field(gt=0)]
    s_per_input_s: Seconds


class Measured(Frozen):
    """Where a row was measured: the 40-hex commit run and the scheduler job ids."""

    sha: Annotated[str, StringConstraints(pattern=rf"^{COMMIT_HEX.pattern}$")]
    jobs: tuple[NonEmptyStr, ...]


def _frozen(v: Mapping[str, object]) -> Mapping[str, object]:
    return FrozenMapping(v)


class ResourceRow(Frozen):
    """One model's measures: its `(id, revision)`, download bytes, and a measure per resource class."""

    id: NonEmptyStr | None
    revision: NonEmptyStr | None
    download_bytes: Annotated[int, Field(ge=0)]
    classes: Annotated[Mapping[ResourceClassName, ResourceMeasure], AfterValidator(_frozen), PlainSerializer(thawed)]
    measured: Measured

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
