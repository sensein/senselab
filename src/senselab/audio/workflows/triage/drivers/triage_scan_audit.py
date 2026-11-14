r"""Count what the triage stores under a corpus root record of their PII scans, as the release decision reads them.

    python -m senselab.audio.workflows.triage.drivers.triage_scan_audit ROOT --out-dir DIR [--workers N]

Every run root under ``ROOT`` (a directory holding ``run/store.jsonl``) is read for its live ``pii_scan``
measurements, and put in one state by :func:`~senselab.audio.workflows.triage.nodes.verdict.scan_state`, the
reading the release decision applies:

- ``no_pii_scan_record``: no live ``pii_scan`` measurement;
- ``declined``: a record says ``scanned: false`` (nothing beyond the stimulus to scan);
- ``complete``: a record names a detector in ``scanned_by`` and none in ``failed`` or ``missing``;
- ``incomplete``: a record, but none of the above, so the scan is unrecorded;
- ``error``: the store would not open.

Each store with a record is also counted per criterion: ``scanned_by`` empty in every record, a ``failed`` detector,
a ``missing`` detector; overall, among the ``declined`` stores and among the rest. Where SPEECH read lexical words,
a ``complete`` store can be released as ``scan_found_nothing`` and an ``incomplete`` one is ``scan_unrecorded``.
``affected`` counts the stores the earlier rule read as scanned (any record, unless one says ``scanned: false``) and
:func:`scan_state` does not: those whose release that change moves from found-nothing to unrecorded.

It writes ``DIR/summary.json`` and ``DIR/not_complete.tsv``, one line per store not ``complete`` (its store path,
state, the three criteria and whether it is affected), and prints the summary. Only paths, detector names and counts
are written; nothing is read from the transcript.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from senselab.audio.workflows.triage.extend import RUN_SUBDIR, STORE_FILE, read_store
from senselab.audio.workflows.triage.nodes.common import find_measurements
from senselab.audio.workflows.triage.nodes.verdict import scan_state
from senselab.audio.workflows.triage.vocabulary import PII_SCAN, SCANNED

NO_RECORD = "no_pii_scan_record"
DECLINED = "declined"
COMPLETE = "complete"
INCOMPLETE = "incomplete"
ERROR = "error"
STATES = (NO_RECORD, DECLINED, COMPLETE, INCOMPLETE, ERROR)

CRITERIA = ("scanned_by_empty", "failed_nonempty", "missing_nonempty")
"""Per store with a record: no record names a detector in ``scanned_by``; one names a ``failed`` detector; one names a
``missing`` detector."""

TSV_NAME = "not_complete.tsv"
SUMMARY_NAME = "summary.json"
TSV_COLUMNS = ("store", "state", "scanned_by_empty", "failed", "missing", "affected")


def build_parser() -> argparse.ArgumentParser:
    """The CLI.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("root", type=Path, help="The corpus root the run roots are under")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where summary.json and not_complete.tsv go")
    parser.add_argument("--workers", type=int, default=None, help="Processes (default: SLURM_CPUS_PER_TASK, else 4)")
    return parser


def run_roots(root: Path) -> Iterator[Path]:
    """Every run root under ``root``, in sorted order; nothing inside a run root is walked.

    Args:
        root: The corpus root.

    Yields:
        Each directory holding ``run/store.jsonl``.
    """
    if (root / RUN_SUBDIR / STORE_FILE).is_file():
        yield root
        return
    try:
        children = sorted(entry.path for entry in os.scandir(root) if entry.is_dir(follow_symlinks=False))
    except OSError:
        return
    for child in children:
        yield from run_roots(Path(child))


def earlier_scan_state(scans: Sequence[Mapping[str, Any]]) -> bool | None:
    """The reading :func:`scan_state` replaced: scanned wherever a record stands and none says ``scanned: false``.

    Args:
        scans: The attributes of every live ``pii_scan`` measurement.

    Returns:
        None with no record, else whether no record says ``scanned: false``.
    """
    return None if not scans else not any(scan.get(SCANNED) is False for scan in scans)


def audit_store(run_root: Path) -> dict[str, Any]:
    """One run root's scan state and criteria.

    Args:
        run_root: The run root.

    Returns:
        ``store``, ``state`` (one of :data:`STATES`), and for a store with a record ``records``, the three
        :data:`CRITERIA` (``failed`` and ``missing`` as the detectors named), ``current`` and ``earlier`` (the two
        readings) and ``affected``; ``error`` (the exception's type) for a store that would not open.
    """
    row: dict[str, Any] = {"store": str(run_root / RUN_SUBDIR / STORE_FILE)}
    try:
        scans = [measurement.attributes for measurement in find_measurements(read_store(run_root), PII_SCAN)]
    except Exception as error:  # noqa: BLE001 -- a store that will not open is counted, never fatal
        return {**row, "state": ERROR, "error": type(error).__name__}
    current = scan_state(scans)
    if not scans:
        return {**row, "state": NO_RECORD, "records": 0, "current": None, "earlier": None, "affected": False}
    earlier = earlier_scan_state(scans)
    return {
        **row,
        "state": DECLINED if current is False else COMPLETE if current else INCOMPLETE,
        "records": len(scans),
        "scanned_by_empty": not any(scan.get("scanned_by") for scan in scans),
        "failed": sorted({str(name) for scan in scans for name in scan.get("failed") or []}),
        "missing": sorted({str(name) for scan in scans for name in scan.get("missing") or []}),
        "current": current,
        "earlier": earlier,
        "affected": earlier is True and current is None,
    }


def _criteria(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """How many of these stores meet each of :data:`CRITERIA`."""
    return {
        "stores": len(rows),
        "scanned_by_empty": sum(1 for row in rows if row["scanned_by_empty"]),
        "failed_nonempty": sum(1 for row in rows if row["failed"]),
        "missing_nonempty": sum(1 for row in rows if row["missing"]),
    }


def summarise(root: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The counts over every store.

    Args:
        root: The corpus root.
        rows: :func:`audit_store` per run root.

    Returns:
        The summary :func:`audit` writes.
    """
    states = Counter(str(row["state"]) for row in rows)
    recorded = [row for row in rows if row["state"] in (DECLINED, COMPLETE, INCOMPLETE)]
    return {
        "root": str(root),
        "stores": len(rows),
        "states": {state: states.get(state, 0) for state in STATES},
        "criteria": _criteria(recorded),
        "criteria_declined": _criteria([row for row in recorded if row["state"] == DECLINED]),
        "criteria_not_declined": _criteria([row for row in recorded if row["state"] != DECLINED]),
        "release_reading": {
            "scan_found_nothing_open": states.get(COMPLETE, 0),
            "scan_unrecorded_with_record": states.get(INCOMPLETE, 0),
            "scan_unrecorded_no_record": states.get(NO_RECORD, 0),
            "nothing_beyond_stimulus": states.get(DECLINED, 0),
        },
        "earlier_rule_scanned": sum(1 for row in recorded if row["earlier"] is True),
        "affected": sum(1 for row in rows if row.get("affected")),
        "errors": dict(Counter(str(row["error"]) for row in rows if row["state"] == ERROR)),
    }


def audit(root: Path, out_dir: Path, *, workers: int = 1) -> dict[str, Any]:
    """Read every store under ``root`` and write the summary and the stores that are not complete.

    Args:
        root: The corpus root.
        out_dir: Where :data:`SUMMARY_NAME` and :data:`TSV_NAME` go.
        workers: Processes to read stores with.

    Returns:
        The summary written.
    """
    roots = list(run_roots(root))
    if workers > 1 and len(roots) > 1:
        with ProcessPoolExecutor(workers) as pool:
            rows = list(pool.map(audit_store, roots, chunksize=64))
    else:
        rows = [audit_store(run_root) for run_root in roots]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / TSV_NAME).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(TSV_COLUMNS)
        for row in rows:
            if row["state"] != COMPLETE:
                writer.writerow(
                    [
                        row["store"],
                        row["state"] if row["state"] != ERROR else f"{ERROR}:{row['error']}",
                        "" if "scanned_by_empty" not in row else str(row["scanned_by_empty"]).lower(),
                        ",".join(row.get("failed") or []),
                        ",".join(row.get("missing") or []),
                        str(bool(row.get("affected"))).lower(),
                    ]
                )
    summary = summarise(root, rows)
    (out_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Audit the stores under a corpus root.

    Args:
        argv: The command line, or None for ``sys.argv``.

    Returns:
        0; 1 where a store would not open or a store is affected; 2 where the root is not a directory.
    """
    args = build_parser().parse_args(argv)
    if not args.root.is_dir():
        print(f"ERROR: not a directory: {args.root}", file=sys.stderr)
        return 2
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
    summary = audit(args.root, args.out_dir, workers=workers)
    print(json.dumps(summary, indent=1))
    return 1 if summary["states"][ERROR] or summary["affected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
