"""Build one speaker embedding per subject from a tree of finished triage runs.

    uv run python scripts/triage_speaker_vectors.py <corpus_root> --out <dir>
    uv run python scripts/triage_speaker_vectors.py <corpus_root> --out <dir> --slice 3 --slices 64
    uv run python scripts/triage_speaker_vectors.py --merge <dir>

The parquet is a biometric derived from human-subject audio: it is written mode 600, is gitignored,
and must never be copied to a shared location.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senselab.audio.workflows.triage.speaker_vectors import (  # noqa: E402
    SCHEMA_VERSION,
    scan,
    schema,
    to_table,
)

SHARD_GLOB = "speaker_vectors.*.parquet"
MERGED_NAME = "speaker_vectors.parquet"
SUMMARY_NAME = "speaker_vectors.summary.json"
PRIVATE_FILE_MODE = 0o600
COMPRESSION = "zstd"
ROW_GROUP_SIZE = 1024


def write_private(table: pa.Table, path: Path) -> int:
    """Write a parquet only its owner can read.

    Args:
        table: The table to write.
        path: Destination.

    Returns:
        The file size in bytes.
    """
    pq.write_table(table, path, compression=COMPRESSION, row_group_size=ROW_GROUP_SIZE, write_page_index=True)
    os.chmod(path, PRIVATE_FILE_MODE)
    return path.stat().st_size


def merge(shard_dir: Path) -> int:
    """Concatenate every shard in a directory into one parquet plus a summary.

    Args:
        shard_dir: Where the shards were written.

    Returns:
        A process exit code.

    Raises:
        SystemExit: If the directory holds no shard.
    """
    shards = sorted(shard_dir.glob(SHARD_GLOB))
    if not shards:
        raise SystemExit(f"no shards matching {SHARD_GLOB} under {shard_dir}")
    table = pa.concat_tables([pq.read_table(shard) for shard in shards]).combine_chunks()
    out = shard_dir / MERGED_NAME
    size = write_private(table, out)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "shards": len(shards),
        "rows": table.num_rows,
        "bytes": size,
        "bytes_per_row": round(size / table.num_rows, 1) if table.num_rows else None,
        "null_counts": {field.name: table.column(field.name).null_count for field in table.schema},
    }
    summary_path = shard_dir / SUMMARY_NAME
    summary_path.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    os.chmod(summary_path, PRIVATE_FILE_MODE)
    print(f"{table.num_rows} rows -> {out} ({size / 1e6:.1f} MB, {summary['bytes_per_row']} B/row)")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one shard, or merge a directory of shards.

    Args:
        argv: Command line, or ``None`` for ``sys.argv[1:]``.

    Returns:
        A process exit code: 1 when a shard wrote no row.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("corpus_root", type=Path, nargs="?", help="A tree of finished run directories.")
    parser.add_argument("--out", type=Path, help="Where the parquet and report go.")
    parser.add_argument("--slice", type=int, default=0, help="This worker's index.")
    parser.add_argument("--slices", type=int, default=1, help="How many workers share the tree.")
    parser.add_argument("--merge", type=Path, default=None, help="Concatenate the shards in this directory.")
    parser.add_argument("--created-at", default=None, help="ISO-8601 timestamp stamped into provenance.")
    parser.add_argument("--device", default=None, choices=("cpu", "cuda"), help="Force a device.")
    args = parser.parse_args(argv)

    if args.merge is not None:
        return merge(args.merge)
    if args.corpus_root is None or args.out is None:
        parser.error("corpus_root and --out are required unless --merge is given")

    device = None
    if args.device is not None:
        from senselab.utils.data_structures import DeviceType  # noqa: PLC0415

        device = DeviceType.CUDA if args.device == "cuda" else DeviceType.CPU

    rows, report = scan(
        args.corpus_root,
        args.slice,
        args.slices,
        device=device,
        created_at=args.created_at,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f".{args.slice:03d}"
    table = to_table(rows) if rows else schema().empty_table()
    size = write_private(table, args.out / f"speaker_vectors{suffix}.parquet")
    report_path = args.out / f"speaker_vectors{suffix}.report.json"
    report_path.write_text(json.dumps(asdict(report), indent=1, sort_keys=True) + "\n")
    os.chmod(report_path, PRIVATE_FILE_MODE)
    per_row = round(size / table.num_rows, 1) if table.num_rows else None
    print(
        f"shard {args.slice}/{args.slices}: {report.subjects_written} speakers, "
        f"{report.extents_admitted} extents, {report.extents_refused_short} refused short, "
        f"{sum(report.extents_refused_family.values())} refused by family, "
        f"{size} bytes, {per_row} B/row"
    )
    return 0 if report.subjects_written else 1


if __name__ == "__main__":
    sys.exit(main())
