"""Packaged measures, `resource-table@2.0`: what one task and backend measured, row by exact measure unit.

A table is a generated file `<root>/<task>/<backend>.yaml`, never hand-edited: packaged under `data/resources` by
`scripts/generate_resources.py` from measures roots. A row is a `MeasureUnit` with its `Estimate` and where it came
from (`Derivation`); `PackagedTable` reads a row on the unit's whole identity, so it answers for its own env build,
backend params and hardware only.
"""

from collections.abc import Sequence
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Annotated, ClassVar

from pydantic import Field, StringConstraints, model_validator
from senselab_worker.digest import COMMIT_HEX

from senselab_core.kernel._base import Frozen, NonEmptyStr
from senselab_core.kernel.formats import RESOURCE_TABLE_FORMAT, FormatTag, Formatted
from senselab_core.runtime.calibration import Estimate
from senselab_core.runtime.config import load_yaml
from senselab_core.runtime.measures import MeasureUnit


class Derivation(Frozen):
    """Where a packaged row was measured: the 40-hex commit, the scheduler jobs and the probes."""

    commit: Annotated[str, StringConstraints(pattern=rf"^{COMMIT_HEX.pattern}$")]
    jobs: tuple[NonEmptyStr, ...] = Field(min_length=1)
    probe_ids: tuple[NonEmptyStr, ...] = Field(min_length=1)


class MeasureRow(Frozen):
    """A measure unit with its estimate and derivation; the estimate's source is the derivation's commit and probes."""

    unit: MeasureUnit
    estimate: Estimate
    derivation: Derivation

    @model_validator(mode="after")
    def _check_derivation(self) -> "MeasureRow":
        source = self.estimate.source
        if (source.commit, source.probe_ids) != (self.derivation.commit, self.derivation.probe_ids):
            raise ValueError("a row's estimate names the commit and probes its derivation names")
        return self


class MeasureTable(Formatted):
    """A `resource-table@2.0` document: the rows of one task and backend."""

    format: FormatTag = RESOURCE_TABLE_FORMAT
    port_value: ClassVar[bool] = False
    task: NonEmptyStr
    backend: NonEmptyStr
    rows: tuple[MeasureRow, ...]

    @model_validator(mode="after")
    def _check_rows(self) -> "MeasureTable":
        for row in self.rows:
            if (row.unit.task, row.unit.backend) != (self.task, self.backend):
                raise ValueError(f"{self.task}/{self.backend} holds a row of {row.unit.task}/{row.unit.backend}")
        digests = [r.unit.digest() for r in self.rows]
        if len(set(digests)) != len(digests):
            raise ValueError(f"{self.task}/{self.backend}: a unit has two rows")
        return self


def load_measure_tables(root: Traversable | Path) -> tuple[MeasureTable, ...]:
    """Return every table under `root`, `<task>/<backend>.yaml`, sorted by task and backend; none when it is absent.

    Raises:
        ConfigError: a file cannot be read as YAML.
        ValueError: a file does not validate, or holds the table of another task or backend than its path.
    """
    if not root.is_dir():
        return ()
    tables = []
    for task_dir in sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name):
        for file in sorted((f for f in task_dir.iterdir() if f.name.endswith(".yaml")), key=lambda f: f.name):
            table = MeasureTable.model_validate(load_yaml(file, "resource table"))
            if (table.task, table.backend) != (task_dir.name, file.name.removesuffix(".yaml")):
                raise ValueError(f"{task_dir.name}/{file.name} holds the table of {table.task}/{table.backend}")
            tables.append(table)
    return tuple(tables)


class PackagedTable:
    """The `resource-table@2.0` rows under a root, read on exact unit identity."""

    def __init__(self, tables: Sequence[MeasureTable]) -> None:
        """Index the rows of `tables` by unit digest."""
        self._rows = {r.unit.digest(): r.estimate for t in tables for r in t.rows}

    @classmethod
    def load(cls, root: Traversable | Path) -> "PackagedTable":
        """Read every `<root>/<task>/<backend>.yaml` (`load_measure_tables`); an absent root holds none."""
        return cls(load_measure_tables(root))

    def get(self, unit: MeasureUnit) -> Estimate | None:
        """Return the estimate of the row whose unit is `unit`, None when there is none."""
        return self._rows.get(unit.digest())
