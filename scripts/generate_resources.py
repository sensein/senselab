"""Write the packaged resource tables (`resource-table@1.0`) from scale-ladder results; never hand-edit them.

Usage::

    uv run python scripts/generate_resources.py RESULT... [--out DIR]

Each `RESULT` is a `scale-result@1.0` file or a directory of them. Per (task, backend, model, resource class) the
measure of the result with the most items (the most input seconds on a tie) is written to
`DIR/<task>/<backend>.yaml` (`packages/senselab/src/senselab/data/resources` by default), with the commit and the
job ids of every result of that unit as its derivation. A result without a measure (no pool worker reported a
sample) is listed and skipped. Results of different commits are refused.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from senselab_core.runtime.resources import Measured, ResourceMeasure, Unit, write_row

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "packages" / "senselab" / "src" / "senselab" / "data" / "resources"
RESULT_FORMAT = "scale-result@1.0"


def results(paths: Iterable[Path]) -> list[dict[str, Any]]:
    """Return every `scale-result@1.0` document of `paths`: files, or the `*.json` files of directories."""
    files = [f for p in paths for f in (sorted(p.glob("*.json")) if p.is_dir() else [p])]
    loaded = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    return [r for r in loaded if isinstance(r, dict) and r.get("format") == RESULT_FORMAT]


def unit_of(result: Mapping[str, Any]) -> Unit:
    """Return the unit a result measured."""
    model = result["model"] or {"id": None, "revision": None}
    return Unit(result["task"], result["backend"], model["id"], model["revision"], result["resource_class"])


def rows(found: Sequence[Mapping[str, Any]]) -> tuple[dict[Unit, tuple[ResourceMeasure, Measured]], list[str]]:
    """Return each unit's chosen measure and derivation, and a line per result without a measure.

    Raises:
        ValueError: the results are of more than one commit.
    """
    shas = sorted({r["sha"] for r in found})
    if len(shas) > 1:
        raise ValueError(f"results of several commits {shas}; generate from one commit's results")
    chosen: dict[Unit, tuple[ResourceMeasure, Measured]] = {}
    skipped = [f"{r['task']}@{r['point']}: no measure; skipped" for r in found if r["measure"] is None]
    measured = [r for r in found if r["measure"] is not None]
    for unit in sorted({unit_of(r) for r in measured}, key=repr):
        of_unit = [r for r in measured if unit_of(r) == unit]
        best = max(of_unit, key=lambda r: (r["items"], r["input_s"] or 0.0))
        jobs = tuple(sorted({j for r in of_unit for j in r["jobs"]}))
        chosen[unit] = (ResourceMeasure.model_validate(best["measure"]), Measured(sha=shas[0], jobs=jobs))
    return chosen, skipped


def main(argv: Sequence[str] | None = None) -> int:
    """Write the tables of the results named in `argv`; print each file written and each result skipped."""
    parser = argparse.ArgumentParser(prog="generate_resources.py")
    parser.add_argument("results", nargs="+", type=Path, help="scale-result@1.0 files or directories of them")
    parser.add_argument("--out", type=Path, default=OUT, help="the resource-table root")
    args = parser.parse_args(argv)
    chosen, skipped = rows(results(args.results))
    for line in skipped:
        print(line)
    for unit, (measure, measured) in chosen.items():
        print(write_row(args.out, unit, measure, measured))
    return 0


if __name__ == "__main__":
    sys.exit(main())
