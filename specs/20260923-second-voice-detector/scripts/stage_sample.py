"""Copy each sampled recording's plain stream into one flat directory named by stem and arm.

Reads through the corpus's symlinks and writes only into the destination, never back through
one — the corpus tree is read-only and an earlier driver overwrote it by writing through a
``streams/`` symlink.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> int:
    """Copy every resolvable sample audio into ``--dest``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sample", help="sample.jsonl from pick_sample.py")
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(args.sample) as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            src = row.get("audio")
            if not src:
                continue
            out = dest / f"{row['arm']}__{row['stem']}.flac"
            shutil.copyfile(Path(src).resolve(), out)
            n += 1
    print(f"copied {n} files into {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
