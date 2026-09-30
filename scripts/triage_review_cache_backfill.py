#!/usr/bin/env python3
r"""Seed the reviewer's result cache from finished runs whose readings were made with the current prompt.

    uv run python scripts/triage_review_cache_backfill.py RUN_ROOT [RUN_ROOT ...] [--manifest FILE] \
        [--config OVERRIDE.yaml] [--log FILE]

Each run root's store is read, and a reading made with the current ``PROMPT_VERSION`` by the configured
model is stored under the key REVIEW would compute for it (``nodes/review.py:review_cache_key``), so the
next run reading the same text, context and settings takes it from the cache instead of the GPU. A
reading made with another prompt version is skipped: its key could never be asked for. ``--manifest``
names a file of run roots, one per line. ``SENSELAB_RESULT_CACHE`` / ``SENSELAB_CACHE`` choose the cache,
as for every other process. One JSON line per root is printed (or written to ``--log``), and a summary.

The design is in ``specs/20260929-task-check-alignment/design.md`` (section E).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.extend import read_store
from senselab.audio.workflows.triage.nodes.review import BACKFILL_SKIPPED, backfill_from_store


def main(argv: list[str] | None = None) -> int:
    """Seed the cache from each named run root.

    Args:
        argv: The command line, without the program name.

    Returns:
        0 when every root was read, 1 when any could not be.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="*", type=Path, help="Run roots")
    parser.add_argument("--manifest", type=Path, default=None, help="File of run roots, one per line")
    parser.add_argument("--config", type=Path, default=None, help="The run's config override")
    parser.add_argument("--log", type=Path, default=None, help="Where the per-root JSON lines go")
    args = parser.parse_args(argv)
    roots = list(args.roots)
    if args.manifest is not None:
        roots += [Path(line.strip()) for line in args.manifest.read_text().splitlines() if line.strip()]
    config = load_triage_config(args.config) if args.config is not None else load_triage_config()
    states: Counter[str] = Counter()
    failed = 0
    out = args.log.open("w", encoding="utf-8") if args.log is not None else sys.stdout
    try:
        for root in roots:
            try:
                state, why = backfill_from_store(read_store(root), config)
            except Exception as exc:  # noqa: BLE001 — one unreadable store is a row, not the end of the pass
                state, why, failed = "error", f"{type(exc).__name__}: {exc}", failed + 1
            states[state if state != BACKFILL_SKIPPED else f"{state}: {why.split(' is not ')[0]}"] += 1
            out.write(json.dumps({"root": str(root), "state": state, "detail": why}) + "\n")
    finally:
        if out is not sys.stdout:
            out.close()
    print(json.dumps({"roots": len(roots), "states": dict(states)}), file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
