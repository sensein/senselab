"""Summarise the PREPROCESS step probe: where the time goes, and what YAMNet costs."""

import json
import statistics
import sys
from collections import defaultdict


def main() -> None:
    """Print the corrected step table and the YAMNet invocation census."""
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    print(f"recordings={len(rows)}  failures={[r['failure'] for r in rows if r['failure']]}")

    totals: dict = defaultdict(float)
    counts: dict = defaultdict(int)
    grand = 0.0
    for r in rows:
        for s in r["steps"]:
            key = f"{s['node']} {s['step']}"
            totals[key] += s["seconds"]
            counts[key] += 1
            grand += s["seconds"]

    print(f"\n=== steps, interval attributed to the step that does the work (grand total {grand:.1f} s)")
    print(f"{'step':<42} {'n':>3} {'mean s':>8} {'total s':>9} {'share':>7}")
    for key, total in sorted(totals.items(), key=lambda kv: -kv[1])[:20]:
        print(f"{key:<42} {counts[key]:3d} {total / counts[key]:8.2f} {total:9.1f} {100 * total / grand:6.1f}%")

    # The wrapper sits on the shared ``subprocess.run``, so every venv backend's call is recorded.
    # YAMNet's four call sites are the only ones inside a step whose name is or ends in "yamnet".
    def is_yamnet(call: dict) -> bool:
        step = call.get("after_step") or ""
        return step == "yamnet" or step.endswith("_yamnet")

    print("\n=== YAMNet invocations")
    print(f"{'duration_s':>10} {'total_s':>9} {'n_invoc':>8} {'yamnet_s':>9} {'share':>7}  (step, wall_s)")
    inv, ys, ts = [], [], []
    for r in rows:
        calls = [c for c in r["yamnet_calls"] if is_yamnet(c)]
        seconds = sum(c["wall_s"] for c in calls)
        share = 100 * seconds / r["total_s"] if r["total_s"] else 0.0
        inv.append(len(calls))
        ys.append(seconds)
        ts.append(r["total_s"])
        sizes = [(c["after_step"], round(c["wall_s"], 2)) for c in calls]
        print(f"{str(r['duration_s']):>10} {r['total_s']:9.1f} {len(calls):8d} {seconds:9.1f} {share:6.1f}%  {sizes}")
    print(
        f"\ninvocations per recording: median {statistics.median(inv)}, min {min(inv)}, max {max(inv)}"
        f"\nyamnet seconds per recording: median {statistics.median(ys):.1f}, mean {statistics.mean(ys):.1f}"
        f"\nyamnet share of the whole graph: {100 * sum(ys) / sum(ts):.1f}%"
    )


if __name__ == "__main__":
    main()
