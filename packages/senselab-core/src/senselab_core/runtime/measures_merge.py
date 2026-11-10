"""Merge measures roots into one: how a group builds the measures of its own resource, shared as a root.

`merge_measures` copies, unit by unit, the probe files that still pool and the `regime` and `oom` records of the
source roots into a destination root. The destination is an ordinary `measures@1` root: a group lists it in its
profile's `calibration.measures_roots`, and `MeasureSource` reads it on a unit's whole identity like any other.
"""

import os
from collections.abc import Sequence
from pathlib import Path

from senselab_core.kernel.formats import MEASURES_LAYOUT_TAG
from senselab_core.runtime.measures import ESTIMATE_RECORDS, MeasureRoots, MeasureUnit
from senselab_core.runtime.storage import StorePolicy


class NoMeasuresRoot(ValueError):
    """A path holds no measures root and no ladder point's root."""


def expand_roots(paths: Sequence[Path]) -> list[Path]:
    """Return the measures roots `paths` name: each one that holds `measures@1`, else its `<label>@<point>/measures`.

    Raises:
        NoMeasuresRoot: a path holds neither.
    """
    found: list[Path] = []
    for path in paths:
        held = [path] if (path / MEASURES_LAYOUT_TAG).is_dir() else sorted(path.glob("*/measures"))
        if not held:
            raise NoMeasuresRoot(f"{path} is no measures root and holds no ladder points' roots")
        found += held
    return found


def merge_measures(sources: Sequence[Path], dest: Path, *, stale_after_s: float) -> list[str]:
    """Copy the measures of `sources` into the root `dest`; return a line per unit with what was copied.

    A probe file keeps its id and provenance and is written once, so a second merge copies nothing new; a record
    already in `dest` is not copied again. Nothing is estimated and no unit is judged measured or not: the reader
    does that under its own calibration settings.
    """
    policy = StorePolicy.load_default()
    read = MeasureRoots(sources, policy, "group", os.geteuid(), stale_after_s=stale_after_s)
    write = MeasureRoots([dest], policy, "own", os.geteuid(), stale_after_s=stale_after_s)
    lines = []
    for unit in read.units():
        have = {p.probe for p in write.probes(unit)}
        held = {(r.kind, repr(sorted(r.record.items()))) for r in write.records(unit)}
        probes = [p for p in read.probes(unit) if p.probe not in have]
        records = [
            r
            for r in read.records(unit)
            if r.kind in ESTIMATE_RECORDS and (r.kind, repr(sorted(r.record.items()))) not in held
        ]
        for probe in probes:
            _persisted(write.put_observations(unit, probe.probe, probe.observations, probe.provenance), dest, unit)
        for record in records:
            _persisted(write.put_record(unit, record.kind, record.record, record.provenance), dest, unit)
        lines.append(f"{unit.task}~{unit.backend} unit {unit.label()}: {len(probes)} probes, {len(records)} records")
    return lines


def _persisted(path: Path | None, dest: Path, unit: MeasureUnit) -> None:
    if path is None:
        raise OSError(f"{dest} is not writable: unit {unit.label()} was not copied")
