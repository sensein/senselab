"""List one triage run's span roles and extents. Timings, roles and counts only."""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    """Print prov-type counts and every speech-structure span of one run directory."""
    d = sys.argv[1]
    with open(os.path.join(d, "run", "store.jsonl")) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    types: dict = {}
    for r in recs:
        t = r.get("prov_type") or r.get("record")
        types[t] = types.get(t, 0) + 1
    print("prov types:", types)
    roles: dict = {}
    for r in recs:
        if r.get("prov_type") == "span":
            ro = (r.get("attributes") or {}).get("role")
            roles[ro] = roles.get(ro, 0) + 1
    print("span roles:", roles)
    for r in recs:
        if r.get("prov_type") != "span":
            continue
        a = r.get("attributes") or {}
        ro = str(a.get("role") or "")
        if ro.startswith(("speech_run", "speaker_turn", "phrase_run")) or ro == "task_extent":
            keep = {k: v for k, v in a.items() if k in ("words_n", "speaker", "attributed_to", "nontarget", "note")}
            print(ro, r.get("extent"), keep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
