"""B2 measurement: COHORT's r36 speaker pools against the same pools restricted to acoustic activity, read-only.

    python pools.py MANIFEST K N OUT_PREFIX

Per run with a measured pool: each pool's r36 seconds and match, and its seconds once restricted to PREPROCESS's
amplitude spans. Re-embedding needs the enrollment vectors (not stored), so the match itself is re-read by the
r37 COHORT run; this measures how much of each pool was silence.
"""

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from senselab.audio.workflows.triage.cohort_fold import COHORT_READING, pool_block
from senselab.audio.workflows.triage.cohort_stage import _within, activity_of
from senselab.audio.workflows.triage.extend import read_store, run_root_of
from senselab.audio.workflows.triage.nodes.common import find_measurement


def one(row: dict) -> dict | None:
    """One run's pools, before and after the restriction."""
    try:
        run_root = run_root_of(Path(row["enhanced"]))
        store = read_store(run_root)
    except Exception as error:  # noqa: BLE001
        return {"stem": row.get("stem"), "error": repr(error)[:200]}
    cohort = find_measurement(store, COHORT_READING)
    if cohort is None:
        return None
    activity = activity_of(store)
    out = {"stem": run_root.name, "activity": None if activity is None else round(sum(b - a for a, b in activity), 2)}
    for pool in ("target", "non_task"):
        block = pool_block(cohort.attributes, pool)
        spans = [(float(a), float(b)) for a, b in block.get("spans") or ()]
        if not spans:
            continue
        kept = spans if activity is None else _within(spans, activity)
        out[pool] = {
            "status": block.get("status"),
            "match": block.get("match"),
            "pooled_s": round(sum(b - a for a, b in spans), 3),
            "restricted_s": round(sum(b - a for a, b in kept), 3),
        }
    return out if len(out) > 2 else None


if __name__ == "__main__":
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    k, n = int(sys.argv[2]), int(sys.argv[3])
    with open(f"{sys.argv[4]}.{k:03d}.jsonl", "w") as out, ProcessPoolExecutor(8) as pool:
        for record in pool.map(one, rows[k::n], chunksize=8):
            if record is not None:
                out.write(json.dumps(record) + "\n")
