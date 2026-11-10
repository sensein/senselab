"""Merge measures roots, or scale-ladder run directories, into one measures root a group shares.

Usage::

    uv run python scripts/merge_measures.py SOURCE... --out DEST [--execution PROFILE]

Each `SOURCE` is a measures root, or a ladder run directory whose `<label>@<point>/measures` roots are read. `DEST`
is a `measures@1` root; list it in the profile's `calibration.measures_roots` and every run that reads it sizes a
unit measured there. A unit absent from it is probed.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from senselab_core.runtime.measures_merge import NoMeasuresRoot, expand_roots, merge_measures
from senselab_core.runtime.profiles import ExecutionProfile


def main(argv: Sequence[str] | None = None) -> int:
    """Merge the roots named in `argv` into `--out`; print a line per unit; return 1 when a source holds no root."""
    parser = argparse.ArgumentParser(prog="merge_measures.py")
    parser.add_argument("sources", nargs="+", type=Path, help="measures roots, or ladder run directories")
    parser.add_argument("--out", required=True, type=Path, help="the measures root to write")
    parser.add_argument("--execution", default="orcd", help="the packaged execution profile whose lease timing is used")
    args = parser.parse_args(argv)
    try:
        sources = expand_roots(args.sources)
    except NoMeasuresRoot as err:
        print(err, file=sys.stderr)
        return 1
    stale = ExecutionProfile.load(args.execution).lease.stale_after_s
    for line in merge_measures(sources, args.out, stale_after_s=stale):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
