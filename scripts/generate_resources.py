"""Write the packaged resource tables (`resource-table@2.0`) from measures roots; never hand-edit them.

Usage::

    uv run python scripts/generate_resources.py ROOT... [--suite SUITE] [--execution PROFILE] [--out DIR]

Each `ROOT` is a measures root (a cluster's), or a ladder run directory, whose `<label>@<point>/measures` roots are
read. Per unit a root holds, `Probe.estimate` runs over the pooled observations of every root under the execution
profile's calibration settings (`--execution`, `orcd` by default; a ladder run's roots are read with `--suite`, the
suite whose table profile they were probed under). Per (task, backend) a table of `DIR/<task>/<backend>.yaml`
(`packages/senselab/src/senselab/data/resources` by default) is written: one row per unit, the unit's full identity
with its estimate and its derivation (commit, scheduler jobs, probe ids); a table that exists keeps the rows of the
units this run does not regenerate. A unit of a task and backend the packaged registry does not declare (the fake
stack) is listed and skipped. The run is refused, and nothing written, when a declared unit's estimate is unmeasured, a
probe behind a row carries no commit or no scheduler job, the rows (new or kept) are of more than one commit, or a
ladder run directory's points were probed under calibration settings other than the ones read (give `--suite`).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from collections.abc import Collection, Iterable, Sequence
from pathlib import Path

from senselab_core.kernel.formats import MEASURES_LAYOUT_TAG
from senselab_core.runtime.call import packaged
from senselab_core.runtime.config import dump_yaml, load_yaml
from senselab_core.runtime.measures import MeasureRoots, MeasureUnit, ProbeFile
from senselab_core.runtime.probe import Probe
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.resources import Derivation, MeasureRow, MeasureTable
from senselab_core.runtime.storage import StorePolicy, write_atomically

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "packages" / "senselab" / "src" / "senselab" / "data" / "resources"
LADDER = ROOT / "benchmarks" / "scale" / "ladder.py"
ESTIMATE_FREE = frozenset({"measures_roots", "trust"})
"""The calibration settings that do not shape an estimate: where measures are read and whose files are."""


class Refused(ValueError):
    """The measures cannot be written as packaged rows; each reason is a line, and nothing was written."""


def expand(roots: Iterable[Path]) -> list[Path]:
    """Return the measures roots `roots` name: each one that holds `measures@1`, else its `<label>@<point>/measures`.

    Raises:
        Refused: a path holds no measures root.
    """
    found: list[Path] = []
    for path in roots:
        held = [path] if (path / MEASURES_LAYOUT_TAG).is_dir() else sorted(path.glob("*/measures"))
        if not held:
            raise Refused(f"{path} is no measures root and holds no ladder points' roots")
        found += held
    return found


def _commit(probe: ProbeFile) -> str | None:
    build = probe.provenance.senselab
    return None if build is None else build.commit


def _row(
    unit: MeasureUnit, found: MeasureRoots, profile: ExecutionProfile
) -> tuple[MeasureRow | None, str | None, list[str]]:
    """Return the row of `unit` and its commit, else why there is none: unmeasured, no commit or job, several commits."""
    estimate = Probe.estimator(
        profile.calibration, records_of=found.regime_records, memory_headroom=profile.sizing.memory_headroom
    )(unit, found.observations(unit))
    name = f"{unit.task}~{unit.backend} unit {unit.label()}"
    if estimate.status != "measured":
        return None, None, [f"{name}: unmeasured ({', '.join(estimate.reasons)})"]
    behind = [p for p in found.probes(unit) if p.probe in estimate.source.probe_ids]
    why = [f"{name}: probe {p.probe} carries no commit" for p in behind if _commit(p) is None]
    why += [f"{name}: probe {p.probe} carries no scheduler job" for p in behind if p.provenance.job is None]
    commits = sorted({c for p in behind if (c := _commit(p)) is not None})
    if len(commits) > 1:
        why.append(f"{name}: probes of several commits {commits}")
    if not behind:
        why.append(f"{name}: no probe file behind its estimate")
    if why:
        return None, None, why
    jobs = sorted({p.provenance.job for p in behind if p.provenance.job is not None})
    source = estimate.source.model_copy(update={"commit": commits[0]})
    derivation = Derivation(commit=commits[0], jobs=tuple(jobs), probe_ids=estimate.source.probe_ids)
    row = MeasureRow(unit=unit, estimate=estimate.model_copy(update={"source": source}), derivation=derivation)
    return row, commits[0], []


def generate(
    roots: Sequence[Path], out: Path, declared: Collection[tuple[str, str]], profile: ExecutionProfile
) -> list[str]:
    """Write the tables of the units `roots` hold under `out`; return a line per unit skipped and per file written.

    Raises:
        Refused: a declared unit is unmeasured, a probe behind a row carries no commit or no scheduler job, or the
            rows are of several commits; nothing is written.
    """
    found = MeasureRoots(
        roots, StorePolicy.load_default(), "group", os.geteuid(), stale_after_s=profile.lease.stale_after_s
    )
    skipped: list[str] = []
    refused: list[str] = []
    rows: list[MeasureRow] = []
    commits: set[str] = set()
    for unit in found.units():
        if (unit.task, unit.backend) not in declared:
            skipped.append(f"{unit.task}~{unit.backend} unit {unit.label()}: not a packaged backend; skipped")
            continue
        row, commit, why = _row(unit, found, profile)
        refused += why
        if row is not None and commit is not None:
            rows.append(row)
            commits.add(commit)
    if len(commits) > 1:
        refused.append(f"rows of several commits {sorted(commits)}; generate from one commit's measures")
    if refused:
        raise Refused("\n".join(refused))
    tables = _merged(rows, out, commits, refused)
    if refused:
        raise Refused("\n".join(refused))
    written = []
    for (task, backend), merged in sorted(tables.items()):
        path = out / task / f"{backend}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        table = MeasureTable(task=task, backend=backend, rows=tuple(merged))
        write_atomically(path, dump_yaml(table.model_dump(mode="json")).encode())
        written.append(str(path))
    return sorted(skipped) + written


def _merged(
    rows: Sequence[MeasureRow], out: Path, commits: set[str], refused: list[str]
) -> dict[tuple[str, str], list[MeasureRow]]:
    """Return each (task, backend)'s table rows: the regenerated `rows` over the rows `out` already holds.

    A row already written for a unit this run did not regenerate is kept; one of another commit than this run's is
    refused, so a table holds rows of one commit.
    """
    tables: dict[tuple[str, str], dict[str, MeasureRow]] = {}
    for row in rows:
        tables.setdefault((row.unit.task, row.unit.backend), {})[row.unit.digest()] = row
    for (task, backend), mine in tables.items():
        path = out / task / f"{backend}.yaml"
        if not path.is_file():
            continue
        for old in MeasureTable.model_validate(load_yaml(path, "resource table")).rows:
            if old.unit.digest() in mine:
                continue
            if {old.derivation.commit} != commits:
                refused.append(
                    f"{path.relative_to(out)} holds unit {old.unit.label()} of commit {old.derivation.commit}, which this "
                    f"run (commit {sorted(commits)}) does not regenerate; regenerate it too, or remove the table"
                )
            mine[old.unit.digest()] = old
    return {key: [mine[d] for d in sorted(mine)] for key, mine in tables.items()}


def ladder_mismatches(paths: Iterable[Path], profile: ExecutionProfile) -> list[str]:
    """Return a line per ladder point under `paths` whose result names calibration settings other than `profile`'s.

    The settings the estimate is made under must be those the points were probed under; a path that is a measures
    root rather than a ladder run directory has no result to compare and is not read.
    """
    now = profile.calibration.model_dump(mode="json")
    lines = []
    for path in paths:
        if (path / MEASURES_LAYOUT_TAG).is_dir():
            continue
        for root in sorted(path.glob("*/measures")):
            result = path / f"{root.parent.name}.json"
            ran = json.loads(result.read_text(encoding="utf-8")).get("calibration") if result.is_file() else None
            if not isinstance(ran, dict):
                lines.append(f"{root.parent.name}: no result naming the calibration it ran under")
                continue
            differ = sorted(k for k in now.keys() - ESTIMATE_FREE if now[k] != ran.get(k))
            if differ:
                lines.append(f"{root.parent.name}: probed under other calibration settings ({', '.join(differ)}); give --suite")
    return lines


def _profile(name: str, suite: Path | None) -> ExecutionProfile:
    profile = ExecutionProfile.load(name)
    if suite is None:
        return profile
    spec = importlib.util.spec_from_file_location("scale_ladder", LADDER)
    assert spec is not None and spec.loader is not None
    ladder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ladder
    spec.loader.exec_module(ladder)
    return ladder.table_profile(profile, ladder.Suite.load(suite))


def main(argv: Sequence[str] | None = None) -> int:
    """Write the tables of the measures roots named in `argv` for the packaged registry's backends; print each line."""
    parser = argparse.ArgumentParser(prog="generate_resources.py")
    parser.add_argument("roots", nargs="+", type=Path, help="measures roots, or ladder run directories")
    parser.add_argument("--suite", type=Path, help="the scale suite whose table profile the roots were probed under")
    parser.add_argument("--execution", default="orcd", help="the packaged execution profile whose calibration is read")
    parser.add_argument("--out", type=Path, default=OUT, help="the resource-table root")
    args = parser.parse_args(argv)
    registry = packaged().registry
    declared = {(t.name, b.name) for t in registry.tasks() for b in registry.backends(t.name)}
    try:
        profile = _profile(args.execution, args.suite)
        if mismatched := ladder_mismatches(args.roots, profile):
            raise Refused("\n".join(mismatched))
        lines = generate(expand(args.roots), args.out, declared, profile)
    except Refused as err:
        print(f"refused, nothing written:\n{err}", file=sys.stderr)
        return 1
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
