"""Count the two decision axes and the task conformance over a finished run tree, read-only.

Reads each recording's ``summary/summary.json`` only, and writes nothing under the tree, whose
``streams/`` are symlinks into another tree. One output file per array task.

    SCAN_ROOT=<run>/out SCAN_OUT=<dir> python3 axes_census.py
"""

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
release = collections.Counter()
triage = collections.Counter()
task_conformance = collections.Counter()
bad = 0
for path in mine:
    try:
        record = json.loads(path.read_text())
    except Exception:
        bad += 1
        continue
    verdict = record.get("verdict") or {}
    family = str(verdict.get("declared_family"))
    release[(str(verdict.get("release")), str(verdict.get("release_ground")))] += 1
    triage[(family, str(verdict.get("triage")))] += 1
    about = verdict.get("conformance_of") or {}
    for node, answer in (verdict.get("conformance") or {}).items():
        if about.get(node) != "task":
            continue
        task_conformance[(family, str(node), str(answer))] += 1

(out / f"{index:04d}.json").write_text(
    json.dumps(
        {
            "n": len(mine),
            "bad": bad,
            "release": [[list(k), v] for k, v in release.items()],
            "triage": [[list(k), v] for k, v in triage.items()],
            "task_conformance": [[list(k), v] for k, v in task_conformance.items()],
        }
    )
)
print(f"slice {index}/{count}: {len(mine)} summaries, {bad} unreadable", file=sys.stderr)
