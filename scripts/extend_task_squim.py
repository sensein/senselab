#!/usr/bin/env python3
r"""Read SQUIM over fixed windows of every finished run's task extent, on the plain stream.

    uv run python scripts/extend_task_squim.py MANIFEST --slice-index I --slice-count N \
        [--log-dir DIR] [--out-root DIR] [--config OVERRIDE.yaml]

``MANIFEST`` is the JSONL every ``extend_*`` driver takes: ``stem`` and ``enhanced``, the path of
the run's ``run/streams/enhanced.flac``, from which the run root is derived.

Each row runs :func:`~senselab.audio.workflows.triage.task_squim.measure_task_squim` over the run's
store: the plain stream is read inside the task extent, in the windows ``data/task_squim.yaml``
declares, by senselab's objective-SQUIM reader with its pinned weights. The store gains one
``task_squim`` measurement and ``prov/`` is re-exported; a run whose live measurement already
matches is not rewritten, so a resubmitted slice redoes nothing.

``--out-root DIR`` leaves the finished tree untouched: each run's store is copied into
``DIR/<sub>/<ses>/<run>/run/store.jsonl`` and written there, with every other file of the run
reachable through symlinks, so the recording vectors and ``triage.tsv`` can be built over ``DIR``.

See ``specs/20261027-task-extent-squim/design.md``.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.extend import (
    ERROR,
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
from senselab.audio.workflows.triage.task_squim import NODE, measure_task_squim

STORE_FILE = "store.jsonl"
PROV_SUBDIR = "prov"


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
    parser.add_argument(
        "--out-root", type=Path, default=None, help="Write each run's store under this tree instead of in place"
    )
    parser.add_argument("--config", type=Path, default=None, help="Partial YAML deep-merged over the packaged config")
    return parser


def mirror_run_root(run_root: Path, out_root: Path) -> Path:
    """A run root under ``out_root`` holding a copy of the store and links to everything else.

    Args:
        run_root: The finished run root, ``<tree>/<sub>/<ses>/<run>``.
        out_root: The tree the mirrored roots are created under.

    Returns:
        ``out_root/<sub>/<ses>/<run>``, created. Its ``run/store.jsonl`` is a copy made once and
        kept thereafter; ``prov/`` is its own directory; every other entry is a symlink.
    """
    dest = out_root / run_root.parent.parent.name / run_root.parent.name / run_root.name
    run_dir = dest / RUN_SUBDIR
    run_dir.mkdir(parents=True, exist_ok=True)
    (dest / PROV_SUBDIR).mkdir(exist_ok=True)
    for entry in sorted(run_root.iterdir()):
        if entry.name in (RUN_SUBDIR, PROV_SUBDIR):
            continue
        link = dest / entry.name
        if not link.is_symlink() and not link.exists():
            link.symlink_to(entry)
    for entry in sorted((run_root / RUN_SUBDIR).iterdir()):
        target = run_dir / entry.name
        if entry.name == STORE_FILE:
            if not target.exists():
                shutil.copy2(entry, target)
        elif not target.is_symlink() and not target.exists():
            target.symlink_to(entry)
    return dest


def measure_one(run_root: Path, config: TriageConfig, *, stem: str, out_root: Path | None) -> dict[str, Any]:
    """Measure one finished run and write its store only if the measurement moved.

    Args:
        run_root: The finished run root.
        config: The triage configuration.
        stem: The recording's BIDS stem.
        out_root: Where to write the store instead of in place, or None.

    Returns:
        ``status`` (``ok``, ``present``, ``absent`` or ``error``), the reason for ``absent``, and the
        medians, ``windows_n`` and ``covered_s`` where measured.
    """
    try:
        target = mirror_run_root(run_root, out_root) if out_root is not None else run_root
        store = read_store(target)
    except (OSError, ValueError) as error:
        return {"status": ERROR, NODE: describe_exception(error)}
    try:
        outcome = measure_task_squim(store, config, run_dir=target / RUN_SUBDIR, stem=stem)
    except (OSError, ValueError, LookupError, RuntimeError) as error:
        return {"status": ERROR, NODE: describe_exception(error)}
    if outcome.changed:
        write_store(store, target)
        export_prov(store, target)
    record: dict[str, Any] = {"status": outcome.status}
    if outcome.reason is not None:
        record["reason"] = outcome.reason
    if outcome.summary is not None:
        record.update(outcome.summary)
    return record


def process(
    rows: Sequence[dict[str, Any]],
    config: TriageConfig,
    *,
    out_root: Path | None = None,
    log: SliceLog | None = None,
) -> list[dict[str, Any]]:
    """Measure every run named by these rows, one at a time.

    Args:
        rows: The manifest rows this task owns.
        config: The triage configuration.
        out_root: Where to write stores instead of in place, or None.
        log: Where each row's record is appended as it lands, or None.

    Returns:
        One outcome record per input row, in order.
    """

    def one(row: dict[str, Any]) -> dict[str, Any]:
        started = time.time()
        try:
            run_root = run_root_of(Path(row["enhanced"]))
        except ValueError as error:
            return {**row, "status": ERROR, NODE: describe_exception(error)}
        record = measure_one(run_root, config, stem=str(row["stem"]), out_root=out_root)
        return {**row, **record, "elapsed_s": round(time.time() - started, 3)}

    return logged(rows, one, log)


def run_slice(
    manifest: Path,
    *,
    slice_index: int,
    slice_count: int,
    config: TriageConfig,
    log_dir: Path,
    out_root: Path | None = None,
) -> dict[str, Any]:
    """Measure every run in one array task's stride of the manifest.

    Args:
        manifest: The manifest JSONL.
        slice_index: This task's 0-based index.
        slice_count: How many tasks the array has.
        config: The triage configuration.
        log_dir: Where this task's ``slices/`` log goes.
        out_root: Where to write stores instead of in place, or None.

    Returns:
        The task's summary: its counts, its parameters, and where its log went.
    """
    started = time.time()
    mine = take_slice(read_manifest(manifest, required=("stem", "enhanced")), slice_index, slice_count)
    print(f"[slice {slice_index}/{slice_count}] {len(mine)} rows", flush=True)

    slices_dir = log_dir / SLICES_SUBDIR
    label = f"task-squim-slice-{slice_index}-of-{slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    rows_log = SliceLog(log_path, slice_index=slice_index, slice_count=slice_count, total=len(mine))
    try:
        log = process(mine, config, out_root=out_root, log=rows_log)
    finally:
        rows_log.close()

    counts: dict[str, int] = {}
    for record in log:
        counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1
    summary = {
        "manifest": str(manifest),
        "slice_index": slice_index,
        "slice_count": slice_count,
        "config_hash": config.config_hash,
        "out_root": str(out_root) if out_root is not None else None,
        "rows": len(mine),
        "counts": counts,
        "windows_n": sum(int(r.get("windows_n") or 0) for r in log),
        "host": os.uname().nodename,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Measure one array task's stride of a finished corpus.

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
            out_root=args.out_root,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Rows:    {summary['rows']}")
    print(f"Log:     {summary['log']}")
    print(f"Windows: {summary['windows_n']}")
    for status, number in sorted(summary["counts"].items()):
        print(f"  {status:<10} {number}")
    return 1 if summary["counts"].get(ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
