#!/usr/bin/env python3
r"""Cut every finished run's plain, enhanced and released redacted streams to its task extent.

    uv run python scripts/extend_task_audio.py MANIFEST --slice-index I --slice-count N \
        [--log-dir DIR] [--config OVERRIDE.yaml]

``MANIFEST`` is the JSONL every ``extend_*`` driver takes: ``stem`` and ``enhanced``, the path of
the run's ``run/streams/enhanced.flac``, from which the run root is derived.

Each row runs :func:`~senselab.audio.workflows.triage.task_audio.cut_task_audio` over the run's
store: no model is loaded and only the streams being cut are read. The cuts land at
``run/streams/task_{plain,enhanced,redacted}.flac`` beside ``run/derivatives/task_audio.json``,
and the store and ``prov/`` are rewritten only where a cut was written or retired. A resubmitted
slice therefore redoes nothing: a run whose cuts match their inputs comes back ``present``.

See ``specs/20261002-task-extent-audio/design.md``.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.extend import (
    ABSENT,
    ERROR,
    OK,
    PRESENT,
    RUN_SUBDIR,
    SLICES_SUBDIR,
    SliceLog,
    export_prov,
    logged,
    read_manifest,
    read_store,
    run_root_of,
    take_slice,
    write_store,
)
from senselab.audio.workflows.triage.nodes.common import describe_exception
from senselab.audio.workflows.triage.task_audio import NODE, SOURCES, cut_task_audio


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
        "--log-dir",
        type=Path,
        default=None,
        help=f"Where this task's {SLICES_SUBDIR}/ log goes (default: beside the manifest)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Partial YAML deep-merged over the packaged config")
    return parser


def cut_one(run_root: Path, config: TriageConfig) -> dict[str, Any]:
    """Cut one finished run and write its store only if a cut moved.

    Args:
        run_root: The run root.
        config: The triage configuration.

    Returns:
        ``status`` (``ok``, ``present``, ``absent`` or ``error``), each source's outcome under
        ``task_<source>``, and the extent's bounds and the recording's duration where one stands.
    """
    try:
        store = read_store(run_root)
    except (OSError, ValueError) as error:
        return {"status": ERROR, NODE: describe_exception(error)}
    try:
        outcome = cut_task_audio(store, config, run_dir=run_root / RUN_SUBDIR)
    except (OSError, ValueError, LookupError, RuntimeError) as error:
        return {"status": ERROR, NODE: describe_exception(error)}
    if outcome.changed:
        write_store(store, run_root)
        export_prov(store, run_root)
    record: dict[str, Any] = {f"task_{name}": outcome.cuts[name] for name in SOURCES}
    if outcome.extent is None:
        return {"status": ABSENT, **record}
    extent = outcome.extent
    record.update(
        {
            "start_s": extent.start_s,
            "end_s": extent.end_s,
            "recording_s": extent.recording_s,
            "clamped_start": extent.clamped_start,
            "clamped_end": extent.clamped_end,
        }
    )
    return {"status": OK if outcome.changed else PRESENT, **record}


def process(
    rows: Sequence[dict[str, Any]], config: TriageConfig, *, log: SliceLog | None = None
) -> list[dict[str, Any]]:
    """Cut every run named by these rows, one at a time.

    Args:
        rows: The manifest rows this task owns.
        config: The triage configuration.
        log: Where each row's record is appended as it lands, or None.

    Returns:
        One outcome record per input row, in order.
    """

    def one(row: dict[str, Any]) -> dict[str, Any]:
        try:
            run_root = run_root_of(Path(row["enhanced"]))
        except ValueError as error:
            return {**row, "status": ERROR, NODE: describe_exception(error)}
        return {**row, **cut_one(run_root, config)}

    return logged(rows, one, log)


def run_slice(
    manifest: Path, *, slice_index: int, slice_count: int, config: TriageConfig, log_dir: Path
) -> dict[str, Any]:
    """Cut every run in one array task's stride of the manifest.

    Args:
        manifest: The manifest JSONL.
        slice_index: This task's 0-based index.
        slice_count: How many tasks the array has.
        config: The triage configuration.
        log_dir: Where this task's ``slices/`` log goes.

    Returns:
        The task's summary: its counts, its parameters, and where its log went.
    """
    started = time.time()
    mine = take_slice(read_manifest(manifest, required=("stem", "enhanced")), slice_index, slice_count)
    print(f"[slice {slice_index}/{slice_count}] {len(mine)} rows", flush=True)

    slices_dir = log_dir / SLICES_SUBDIR
    label = f"task-audio-slice-{slice_index}-of-{slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    rows_log = SliceLog(log_path, slice_index=slice_index, slice_count=slice_count, total=len(mine))
    try:
        log = process(mine, config, log=rows_log)
    finally:
        rows_log.close()

    counts: dict[str, int] = {}
    per_source: dict[str, dict[str, int]] = {name: {} for name in SOURCES}
    for record in log:
        counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1
        for name in SOURCES:
            value = str(record.get(f"task_{name}") or "-").split(":", maxsplit=1)[0]
            per_source[name][value] = per_source[name].get(value, 0) + 1

    summary = {
        "manifest": str(manifest),
        "slice_index": slice_index,
        "slice_count": slice_count,
        "config_hash": config.config_hash,
        "rows": len(mine),
        "counts": counts,
        "cuts": per_source,
        "host": os.uname().nodename,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Cut one array task's stride of a finished corpus.

    Args:
        argv: The command line, or None for ``sys.argv``.

    Returns:
        0 where every row reached a determinate outcome, 1 where any row is ``error``, 2 where the
        arguments could not be resolved.
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
            config=load_triage_config(args.config) if args.config else load_triage_config(),
            log_dir=args.log_dir if args.log_dir is not None else args.manifest.parent,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Rows:    {summary['rows']}")
    print(f"Log:     {summary['log']}")
    for status, number in sorted(summary["counts"].items()):
        print(f"  {status:<10} {number}")
    for name, values in summary["cuts"].items():
        print(f"  task_{name:<10} " + ", ".join(f"{k} {v}" for k, v in sorted(values.items())))
    return 1 if summary["counts"].get(ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
