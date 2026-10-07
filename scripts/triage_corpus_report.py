"""Fold a tree of triage runs into one corpus decision report.

Reads every ``*.row.json`` a corpus driver wrote and every ``run.json`` the runner wrote beside a
store, counts what the graph decided, and writes the report beside the tree.

    uv run python scripts/triage_corpus_report.py RUN_DIR [--out DIR]

Every value it reports is categorical or a count, as the decisions it reads are: no transcript text
and no detected string passes through it.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from senselab.audio.workflows.triage.corpus_report import aggregate, decisions, render_markdown


def main(argv: list[str] | None = None) -> int:
    """Write the corpus fold for one run tree.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when at least one decision was read, 1 when the tree held none.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="A tree of driver rows, runner logs, or both.")
    parser.add_argument("--out", type=Path, default=None, help="Where the report goes; default RUN_DIR.")
    args = parser.parse_args(argv)

    report = aggregate(decisions(args.run_dir))
    out = args.out or args.run_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "corpus_decisions.json").write_text(json.dumps(asdict(report), indent=2, sort_keys=True) + "\n")
    (out / "corpus_decisions.md").write_text(render_markdown(report, args.run_dir))
    print(f"{report.files} recordings read, {report.unread} without a decision -> {out}")
    for outcome, count in report.triage.items():
        print(f"  {outcome}: {count}")
    return 0 if report.files else 1


if __name__ == "__main__":
    sys.exit(main())
