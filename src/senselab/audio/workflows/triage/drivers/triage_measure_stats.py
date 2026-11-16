"""Report how every measurement the triage graph writes is distributed over a corpus.

    uv run python scripts/triage_measure_stats.py RUN_ROOT [--out DIR] [--slice N --slices M]

A threshold is only derivable against the distribution of the reading it cuts. This reads finished
stores and reports, per measurement and per declared family, the quantiles of that reading — so a
bound can be argued against measured values, and a count nobody asked for can be replaced by a
measured central tendency.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from senselab.audio.workflows.triage.measure_stats import collect, readings, render_markdown


def main(argv: list[str] | None = None) -> int:
    """Write the distributions for one run tree.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when at least one reading was found, 1 when the tree held none.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_root", type=Path, help="A tree of finished run directories.")
    parser.add_argument("--out", type=Path, default=None, help="Where the report goes; default RUN_ROOT.")
    parser.add_argument("--slice", type=int, default=0, help="This worker's index, for a sharded scan.")
    parser.add_argument("--slices", type=int, default=1, help="How many workers share the tree.")
    args = parser.parse_args(argv)

    stats = collect(
        (family, name, value)
        for index, (family, name, value) in enumerate(readings(args.run_root))
        if args.slices == 1 or index % args.slices == args.slice
    )
    out = args.out or args.run_root
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.slices == 1 else f".{args.slice:03d}"
    payload = {name: {fam: asdict(d) for fam, d in fams.items()} for name, fams in stats.items()}
    (out / f"measure_stats{suffix}.json").write_text(json.dumps(payload, indent=1, sort_keys=True, default=str) + "\n")
    if args.slices == 1:
        (out / "measure_stats.md").write_text(render_markdown(stats, args.run_root))
    print(f"{len(stats)} distinct measurements -> {out}")
    return 0 if stats else 1


if __name__ == "__main__":
    sys.exit(main())
