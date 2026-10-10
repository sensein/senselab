#!/usr/bin/env python3
"""Select the r18c rows the prompt-12 REVIEW and question-set-4 SECOND_OPINION re-run reads.

    python select_rows.py MANIFEST OUT.jsonl [--workers N]

A row is kept when its store holds a live ``pii`` finding, or a live ``task_speech_reading`` with at
least one word outside the task (``words_n`` > 0). Prints counts only.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from senselab.audio.workflows.triage.extend import read_manifest, read_store, run_root_of
from senselab.audio.workflows.triage.nodes.common import find_measurement, live_entities
from senselab.audio.workflows.triage.task_speech import TASK_SPEECH_READING


def reason(row: dict[str, Any]) -> str:
    """Why a row is read again: ``pii``, ``task_speech``, ``none`` or ``error``."""
    try:
        store = read_store(run_root_of(Path(row["enhanced"])))
    except (OSError, ValueError, KeyError):
        return "error"
    if live_entities(store, "pii"):
        return "pii"
    reading = find_measurement(store, TASK_SPEECH_READING)
    if reading is not None and int(reading.attributes.get("words_n") or 0) > 0:
        return "task_speech"
    return "none"


def main() -> int:
    """Write the selected rows and print the counts.

    Returns:
        0.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    rows = read_manifest(args.manifest, required=("stem", "enhanced"))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        reasons = list(pool.map(reason, rows))
    counts: dict[str, int] = {}
    for why in reasons:
        counts[why] = counts.get(why, 0) + 1
    with args.out.open("w", encoding="utf-8") as handle:
        for row, why in zip(rows, reasons):
            if why in ("pii", "task_speech"):
                handle.write(json.dumps(row) + "\n")
    print(json.dumps({"rows": len(rows), **counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
