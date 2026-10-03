"""Count release x release_ground over a finished run tree, read-only.

Reads each recording's ``summary/summary.json`` only. Writes nothing under the tree.
"""

from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

root = Path(os.environ["SCAN_ROOT"])
out = Path(os.environ["SCAN_OUT"])
out.mkdir(parents=True, exist_ok=True)
index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))

paths = sorted(root.glob("*/*/*/summary/summary.json"))
mine = paths[index::count]
counts: collections.Counter = collections.Counter()
bad = 0
for path in mine:
    try:
        record = json.loads(path.read_text())
    except Exception:
        bad += 1
        continue
    verdict = record.get("verdict") or {}
    llm = verdict.get("llm_redaction") or {}
    counts[
        (
            str(verdict.get("release")),
            str(verdict.get("release_ground")),
            str(verdict.get("triage")),
            str(llm.get("status") if isinstance(llm, dict) else None),
        )
    ] += 1

(out / f"{index:04d}.json").write_text(
    json.dumps({"n": len(mine), "bad": bad, "counts": [[list(k), v] for k, v in counts.items()]})
)
print(f"slice {index}/{count}: {len(mine)} summaries, {bad} unreadable", file=sys.stderr)
