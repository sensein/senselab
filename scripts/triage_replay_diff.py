#!/usr/bin/env python3
r"""Say what a replay over a finished triage corpus changed, store by store and then corpus-wide.

Two phases, because the reading is per recording and the fold is over all of them:

    uv run python scripts/triage_replay_diff.py rows MANIFEST --slice-index I --slice-count N \
        [--out-root DIR] [--log-dir DIR]
    uv run python scripts/triage_replay_diff.py report ROWS_DIR [--out DIR]

``rows`` reads one array task's stride of the manifest, opens each replayed store and writes one
differential row per recording into ``<log-dir>/slices/replay-diff-slice-I-of-N.jsonl``. ``report``
folds every row under a tree into ``replay_diff.json`` and ``replay_diff.md``.

``MANIFEST`` is the JSONL the replay itself took: one object per line carrying ``stem`` and
``enhanced`` (the absolute path of that recording's ``run/streams/enhanced.flac`` in the finished
corpus). ``--out-root`` is the tree the replay wrote its mirrored run roots under, and must be the
one it was given; without it the rows are read from the finished tree, where the replay ran in
place.

Nothing is written under either the corpus tree or the replay tree: the rows go wherever
``--log-dir`` names, beside the manifest by default.

A replayed store carries both decisions and the invalidation edges between them, so the
differential needs no second tree to compare against. Every value it emits is categorical or a
count: no transcript text and no detected string passes through it.

The design is in ``specs/20260922-replay-decisions-over-a-finished-corpus/differential.md``.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from senselab.audio.workflows.triage.extend import (
    ERROR,
    SLICES_SUBDIR,
    read_manifest,
    run_root_of,
    take_slice,
)
from senselab.audio.workflows.triage.replay_diff import (
    NO_STORE,
    aggregate,
    diff_run,
    read_rows,
    write_report,
)
from senselab.audio.workflows.triage.run import entity_subdir

LABEL = "replay-diff"
"""What a slice log is named after."""


def build_parser() -> argparse.ArgumentParser:
    """The CLI: one subcommand that reads stores, one that folds what it wrote.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rows = subparsers.add_parser("rows", help="Read one array task's stride of replayed stores")
    rows.add_argument("manifest", type=Path, help="Manifest JSONL: one object per recording")
    rows.add_argument("--slice-index", type=int, required=True, help="This task's index, 0-based")
    rows.add_argument("--slice-count", type=int, required=True, help="How many tasks the array has")
    rows.add_argument("--out-root", type=Path, default=None, help="Where the replay wrote its mirrored roots")
    rows.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=f"Where this task's {SLICES_SUBDIR}/ log goes (default: beside the manifest)",
    )

    report = subparsers.add_parser("report", help="Fold every row under a tree into the corpus differential")
    report.add_argument("rows_dir", type=Path, help="A tree of differential slice logs")
    report.add_argument("--out", type=Path, default=None, help="Where the report goes; default ROWS_DIR")
    return parser


def replayed_root(row: dict[str, Any], out_root: Path | None) -> Path:
    """Where the replay wrote one recording's run, from the manifest row the replay itself took.

    Args:
        row: The manifest row, carrying ``stem`` and ``enhanced``.
        out_root: The tree the replay mirrored into, or None when it replayed in place.

    Returns:
        The replayed run root.

    Raises:
        ValueError: If ``enhanced`` is not a stream path inside a run root.
    """
    finished = run_root_of(Path(str(row["enhanced"])))
    if out_root is None:
        return finished
    return out_root / entity_subdir(str(row["stem"])) / finished.name


def process(rows: Sequence[dict[str, Any]], out_root: Path | None) -> list[dict[str, Any]]:
    """Read every replayed store these rows name, one at a time.

    Args:
        rows: The manifest rows this task owns.
        out_root: The tree the replay mirrored into, or None.

    Returns:
        One differential row per input row, in order. A recording whose run root cannot be derived
        gets an error row; its neighbours are unaffected.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        stem = str(row.get("stem") or "")
        try:
            root = replayed_root(row, out_root)
        except (KeyError, ValueError) as error:
            out.append({"stem": stem, "status": ERROR, "detail": f"{type(error).__name__}: {error}"})
            continue
        out.append(diff_run(root, stem=stem))
    return out


def run_slice(
    manifest: Path,
    *,
    slice_index: int,
    slice_count: int,
    out_root: Path | None,
    log_dir: Path,
) -> dict[str, Any]:
    """Read the replayed stores in one array task's stride and write its rows.

    Args:
        manifest: The manifest JSONL.
        slice_index: This task's 0-based index.
        slice_count: How many tasks the array has.
        out_root: The tree the replay mirrored into, or None.
        log_dir: Where this task's ``slices/`` log goes.

    Returns:
        The task's summary: its counts, its parameters, and where its log went.
    """
    started = time.time()
    mine = take_slice(read_manifest(manifest, required=("stem", "enhanced")), slice_index, slice_count)
    print(f"[slice {slice_index}/{slice_count}] {len(mine)} rows", flush=True)

    log = process(mine, out_root)
    counts: Counter[str] = Counter(str(row.get("status")) for row in log)
    identical = sum(1 for row in log if row.get("identical"))

    slices_dir = log_dir / SLICES_SUBDIR
    slices_dir.mkdir(parents=True, exist_ok=True)
    label = f"{LABEL}-slice-{slice_index}-of-{slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    log_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in log), encoding="utf-8")

    summary = {
        "manifest": str(manifest),
        "slice_index": slice_index,
        "slice_count": slice_count,
        "out_root": str(out_root) if out_root is not None else None,
        "rows": len(mine),
        "counts": dict(counts.most_common()),
        "identical": identical,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Read one shard of replayed stores, or fold a tree of rows into the corpus differential.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when the phase did its work, 1 when a row errored or the fold found nothing to read, and
        2 when the arguments could not be resolved and nothing was read.
    """
    args = build_parser().parse_args(argv)

    if args.command == "report":
        report = aggregate(read_rows(args.rows_dir))
        json_path, markdown_path = write_report(report, args.out or args.rows_dir, args.rows_dir)
        print(f"{report.rows} rows read, {report.compared} compared, {report.identical} identical")
        for status, count in report.statuses.items():
            print(f"  {status:<16} {count}")
        for blocker, count in (report.not_replayable.get("by_blocker") or {}).items():
            print(f"  not replayable: {blocker} — {count}")
        print(f"JSON:     {json_path}")
        print(f"Markdown: {markdown_path}")
        return 0 if report.rows else 1

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    try:
        summary = run_slice(
            args.manifest,
            slice_index=args.slice_index,
            slice_count=args.slice_count,
            out_root=args.out_root,
            log_dir=args.log_dir if args.log_dir is not None else args.manifest.parent,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(f"Rows:      {summary['rows']}")
    print(f"Log:       {summary['log']}")
    print(f"Identical: {summary['identical']}")
    for status, count in summary["counts"].items():
        print(f"  {status:<16} {count}")
    unread = int(summary["counts"].get(NO_STORE, 0))
    if unread:
        print(f"  {unread} of them not yet replayed or holding no store", file=sys.stderr)
    return 1 if summary["counts"].get(ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
