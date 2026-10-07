#!/usr/bin/env python3
r"""Give finished triage runs PREPROCESS's own-floor measurement, the first half of the session floor.

    uv run python scripts/extend_background_floor.py MANIFEST --slice-index I --slice-count N [--log-dir DIR]

``MANIFEST`` is a JSONL, one object per line, each carrying ``stem`` and ``enhanced`` (the absolute
path of that recording's ``run/streams/enhanced.flac``), as every extend driver takes it. Each store
gains one ``background_floor`` measurement, read off its ``plain`` and ``residual`` streams; a store
already carrying a live one is skipped. Run it over the whole corpus, then
``scripts/extend_session_floor.py``, then ``scripts/extend_replay_decisions.py``.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from senselab.audio.workflows.triage.extend import (
    ERROR,
    OK,
    RUN_SUBDIR,
    SKIPPED,
    SLICES_SUBDIR,
    attempt_derivation,
    export_prov,
    read_manifest,
    read_store,
    run_root_of,
    take_slice,
    write_store,
)
from senselab.audio.workflows.triage.nodes.background import OWN_FLOOR, write_own_floor
from senselab.audio.workflows.triage.nodes.common import capture_environments, describe_exception, find_measurement


def build_parser() -> argparse.ArgumentParser:
    """The CLI: a manifest, and which shard of it this task takes.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("manifest", type=Path, help="Manifest JSONL: one object per recording")
    parser.add_argument("--slice-index", type=int, required=True, help="This task's index, 0-based")
    parser.add_argument("--slice-count", type=int, required=True, help="How many tasks the array has")
    parser.add_argument(
        "--log-dir", type=Path, default=None, help=f"Where {SLICES_SUBDIR}/ goes (default: beside the manifest)"
    )
    return parser


def extend_one(run_root: Path) -> tuple[str, str]:
    """Give one finished run its own-floor measurement, or say why it gets none.

    Args:
        run_root: The run root.

    Returns:
        ``(status, detail)``.
    """
    try:
        store = read_store(run_root)
    except (OSError, ValueError) as error:
        return ERROR, describe_exception(error)
    if find_measurement(store, OWN_FLOOR) is not None:
        return SKIPPED, f"the store already carries a live {OWN_FLOOR} measurement"
    outcome = attempt_derivation(lambda: write_own_floor(store, run_root / RUN_SUBDIR))
    if outcome.failed:
        return ERROR, outcome.detail
    capture_environments(store, {})
    write_store(store, run_root)
    export_prov(store, run_root)
    return OK, outcome.detail


def process(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extend every run named by these rows, one at a time.

    Args:
        rows: The manifest rows this task owns.

    Returns:
        One outcome record per input row, in order.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            run_root = run_root_of(Path(row["enhanced"]))
        except ValueError as error:
            out.append({**row, "status": ERROR, OWN_FLOOR: describe_exception(error)})
            continue
        status, detail = extend_one(run_root)
        out.append({**row, "status": status, OWN_FLOOR: detail})
    return out


def run_slice(manifest: Path, *, slice_index: int, slice_count: int, log_dir: Path) -> dict[str, Any]:
    """Extend every run in one array task's stride of the manifest.

    Args:
        manifest: The manifest JSONL.
        slice_index: This task's 0-based index.
        slice_count: How many tasks the array has.
        log_dir: Where this task's ``slices/`` log goes.

    Returns:
        The task's summary.
    """
    started = time.time()
    mine = take_slice(read_manifest(manifest, required=("stem", "enhanced")), slice_index, slice_count)
    print(f"[slice {slice_index}/{slice_count}] {len(mine)} rows", flush=True)
    log = process(mine)
    counts: dict[str, int] = {}
    for record in log:
        counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1
    slices_dir = log_dir / SLICES_SUBDIR
    slices_dir.mkdir(parents=True, exist_ok=True)
    label = f"background-floor-slice-{slice_index}-of-{slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    log_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in log), encoding="utf-8")
    summary = {
        "manifest": str(manifest),
        "slice_index": slice_index,
        "slice_count": slice_count,
        "rows": len(mine),
        "counts": counts,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Extend every run in one shard and print what happened.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when every row reached a determinate outcome, 1 when any is ``error``, 2 on bad arguments.
    """
    args = build_parser().parse_args(argv)
    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    try:
        summary = run_slice(
            args.manifest,
            slice_index=args.slice_index,
            slice_count=args.slice_count,
            log_dir=args.log_dir if args.log_dir is not None else args.manifest.parent,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Rows:    {summary['rows']}")
    print(f"Log:     {summary['log']}")
    for status, number in sorted(summary["counts"].items()):
        print(f"  {status:<9} {number}")
    return 1 if summary["counts"].get(ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
