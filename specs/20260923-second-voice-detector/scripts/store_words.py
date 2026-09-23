"""Dump one run's word entities with their timings, for locating structure by hand.

Operator tool. Its output carries transcript text and must never be pasted into a commit,
spec, report or log — it exists so a person can find *where* something happens and then
record only the seconds.
"""

from __future__ import annotations

import json
import os
import sys


def main() -> int:
    """Print every word entity of one run directory as ``start end text``."""
    d = sys.argv[1]
    with open(os.path.join(d, "run", "store.jsonl")) as fh:
        recs = [json.loads(line) for line in fh if line.strip()]
    for r in recs:
        if r.get("prov_type") != "word":
            continue
        a = r.get("attributes") or {}
        ext = r.get("extent") or [None, None]
        txt = a.get("text") or a.get("token") or a.get("word") or ""
        print(f"{ext[0]}\t{ext[1]}\t{txt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
