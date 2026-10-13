"""Pass 1 of the r37 words/speaker measurement: each recording's level and clipping, read-only.

    python condition.py MANIFEST K N OUT_PREFIX

Writes one JSON line per run: stem, participant, and COHORT's ``condition_readings`` (level, the clip detector's live spans).
No transcript text. Nothing is written into any store.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from senselab.audio.workflows.triage.extend import read_store, run_root_of
from senselab.audio.workflows.triage.cohort_stage import condition_readings


def one(row: dict) -> dict:
    """One run's condition readings."""
    try:
        run_root = run_root_of(Path(row["enhanced"]))
        store = read_store(run_root)
    except Exception as error:  # noqa: BLE001
        return {"stem": row.get("stem"), "error": repr(error)[:200]}
    return {"stem": run_root.name, "participant": run_root.name.split("_")[0], **condition_readings(store, None)}


if __name__ == "__main__":
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    k, n = int(sys.argv[2]), int(sys.argv[3])
    mine = rows[k::n]
    with open(f"{sys.argv[4]}.{k:03d}.jsonl", "w") as out, ProcessPoolExecutor(8) as pool:
        for record in pool.map(one, mine, chunksize=8):
            out.write(json.dumps(record) + "\n")
