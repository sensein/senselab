"""Write one compact row per recording from a tree of finished triage stores.

    uv run python scripts/triage_recording_vectors.py RUN_ROOT --out DIR [--slice N --slices M]
    uv run python scripts/triage_recording_vectors.py --merge DIR --out DIR

The first form writes one shard, ``recording_vectors.NNN.parquet``, plus its scan report, and the
same recordings' decision tables, ``triage_decisions.NNN.parquet`` and ``triage_evidence.NNN.parquet``.
The second concatenates every shard in a directory into ``recording_vectors.parquet``,
``triage_decisions.parquet`` (with ``triage_decisions.tsv``) and ``triage_evidence.parquet``.

The output carries transcript text and marked PII extents. It is written mode 600 and must not be
committed, published or copied to a shared location.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from senselab.audio.workflows.triage import decision_tables
from senselab.audio.workflows.triage.recording_vectors import (
    DICTIONARY_KEY,
    SCHEMA_VERSION,
    ScanReport,
    latest_recordings,
    scan,
    schema,
    to_table,
)
from senselab.utils import fastio

SHARD_GLOB = "recording_vectors.*.parquet"
MERGED_NAME = "recording_vectors.parquet"
PRIVATE_FILE_MODE = 0o600
COMPRESSION = "zstd"
ROW_GROUP_SIZE = 1024


def write_private(table: pa.Table, path: Path) -> int:
    """Write a parquet only its owner can read.

    Args:
        table: The rows.
        path: Where the file goes.

    Returns:
        The file's size in bytes.
    """
    pq.write_table(table, path, compression=COMPRESSION, row_group_size=ROW_GROUP_SIZE, write_page_index=True)
    os.chmod(path, PRIVATE_FILE_MODE)
    return path.stat().st_size


def merge(shard_dir: Path, out: Path) -> int:
    """Concatenate every shard into one parquet.

    Args:
        shard_dir: Where the shards are.
        out: Where the merged file goes.

    Returns:
        How many rows were written.

    Raises:
        SystemExit: When there are no shards, or the shards carry different data dictionaries.
    """
    shards = sorted(shard_dir.glob(SHARD_GLOB))
    if not shards:
        raise SystemExit(f"no shards matching {SHARD_GLOB} under {shard_dir}")
    tables = [pq.read_table(shard) for shard in shards]
    dictionaries = {(table.schema.metadata or {}).get(DICTIONARY_KEY) for table in tables}
    if len(dictionaries) != 1:
        raise SystemExit(f"the shards under {shard_dir} carry {len(dictionaries)} different data dictionaries")
    table = pa.concat_tables(tables).combine_chunks()
    table = table.replace_schema_metadata(tables[0].schema.metadata)
    out.parent.mkdir(parents=True, exist_ok=True)
    size = write_private(table, out)
    reports = [json.loads(p.read_text()) for p in sorted(shard_dir.glob("recording_vectors.*.report.json"))]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "shards": len(shards),
        "rows": table.num_rows,
        "bytes": size,
        "bytes_per_row": round(size / table.num_rows, 1) if table.num_rows else None,
        "considered": sum(r["considered"] for r in reports),
        "incomplete": sum(len(r["incomplete"]) for r in reports),
        "unreadable": sum(len(r["unreadable"]) for r in reports),
        "superseded": sum(len(r["superseded"]) for r in reports),
        "anomalies": {
            name: sum(r["anomalies"].get(name, 0) for r in reports)
            for name in sorted({name for r in reports for name in r["anomalies"]})
        },
        "null_counts": {field.name: table.column(field.name).null_count for field in table.schema},
    }
    (out.parent / "recording_vectors.summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(f"{table.num_rows} rows -> {out} ({size / 1e6:.1f} MB, {summary['bytes_per_row']} B/row)")
    merge_decision_tables(shard_dir, out.parent)
    return table.num_rows


def merge_decision_tables(shard_dir: Path, out_dir: Path) -> None:
    """Concatenate the decision-table shards and write the merged tables, private.

    Args:
        shard_dir: Where the shards are.
        out_dir: Where the merged tables go.
    """
    merged = {}
    for name, empty in (
        (decision_tables.DECISIONS_NAME, decision_tables.decisions_schema()),
        (decision_tables.EVIDENCE_NAME, decision_tables.evidence_schema()),
    ):
        shards = sorted(shard_dir.glob(f"{name}.*.parquet"))
        tables = [pq.read_table(shard) for shard in shards]
        merged[name] = pa.concat_tables(tables).combine_chunks() if tables else empty.empty_table()
    paths = decision_tables.write_tables(
        merged[decision_tables.DECISIONS_NAME], merged[decision_tables.EVIDENCE_NAME], out_dir
    )
    for path in paths.values():
        os.chmod(path, PRIVATE_FILE_MODE)
    decided, items = merged[decision_tables.DECISIONS_NAME].num_rows, merged[decision_tables.EVIDENCE_NAME].num_rows
    print(f"{decided} decisions, {items} evidence items -> {out_dir}")


def main(argv: list[str] | None = None) -> int:
    """Write one shard, or merge a directory of them.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when rows were written, 1 when none were.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_root", type=Path, nargs="?", help="A tree of finished run directories.")
    parser.add_argument("--out", type=Path, required=True, help="Where the parquet and report go.")
    parser.add_argument("--slice", type=int, default=0, help="This worker's index.")
    parser.add_argument("--slices", type=int, default=1, help="How many workers share the tree.")
    parser.add_argument("--merge", type=Path, default=None, help="Concatenate the shards in this directory.")
    parser.add_argument(
        "--walk-threads", type=int, default=fastio.DEFAULT_THREADS, help="Directory listings in flight while scanning."
    )
    args = parser.parse_args(argv)

    if args.merge is not None:
        return 0 if merge(args.merge, args.out / MERGED_NAME) else 1
    if args.run_root is None:
        parser.error("run_root is required unless --merge is given")

    report = ScanReport()
    recordings = latest_recordings(args.run_root, args.slice, args.slices, report, walk_threads=args.walk_threads)
    rows, report = scan(args.run_root, recordings=recordings, report=report)
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f".{args.slice:03d}"
    table = to_table(rows) if rows else schema().empty_table()
    size = write_private(table, args.out / f"recording_vectors{suffix}.parquet")
    (args.out / f"recording_vectors{suffix}.report.json").write_text(
        json.dumps(asdict(report), indent=1, sort_keys=True) + "\n"
    )
    decisions, evidence = decision_tables.tables(recordings, args.run_root)
    write_private(decisions, args.out / f"{decision_tables.DECISIONS_NAME}{suffix}.parquet")
    write_private(evidence, args.out / f"{decision_tables.EVIDENCE_NAME}{suffix}.parquet")
    per_row = round(size / table.num_rows, 1) if table.num_rows else None
    print(f"shard {args.slice}/{args.slices}: {report.written} rows, {size} bytes, {per_row} B/row")
    return 0 if report.written else 1


if __name__ == "__main__":
    sys.exit(main())
