#!/usr/bin/env python3
r"""Write SESSION's floor into finished triage runs, one BIDS session at a time.

    uv run python scripts/extend_session_floor.py MANIFEST --slice-index I --slice-count N [--log-dir DIR]

``MANIFEST`` is the JSONL every extend driver takes (``stem``, ``enhanced``). Rows are grouped by
their BIDS session (``sub-<label>_ses-<label>`` off the stem); ``--slice-index`` / ``--slice-count``
shard the sessions, never a session's members, so each task sees every member of the sessions it
owns. Every member's store must already carry PREPROCESS's ``background_floor``
(``scripts/extend_background_floor.py``); a member without one is left out of its session's floor,
and its own store records the missing floor. Each store gains one ``session_floor`` measurement.
Run ``scripts/extend_replay_decisions.py`` afterwards: BACKGROUND reads this floor.

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
    SLICES_SUBDIR,
    export_prov,
    read_manifest,
    read_store,
    run_root_of,
    take_slice,
    write_store,
)
from senselab.audio.workflows.triage.nodes.background import (
    OWN_FLOOR,
    SESSION_FLOOR,
    session_key,
    write_session_floor,
)
from senselab.audio.workflows.triage.nodes.common import capture_environments, describe_exception, find_measurement
from senselab.utils.prov_store import ProvStore


def build_parser() -> argparse.ArgumentParser:
    """The CLI: a manifest, and which shard of its sessions this task takes.

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


def sessions_of(rows: Sequence[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """The manifest's rows grouped by BIDS session, sessions in sorted order.

    Args:
        rows: Every manifest row.

    Returns:
        ``(session, rows)`` pairs; a stem naming no session is its own group, keyed by the stem.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(session_key(str(row["stem"])) or str(row["stem"]), []).append(row)
    return sorted(groups.items())


def process_session(session: str, rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Write the session floor into every member store of one session.

    Args:
        session: The session key.
        rows: Its members' manifest rows.

    Returns:
        One outcome record per member.
    """
    opened: list[tuple[dict[str, Any], Path | None, ProvStore | None, str]] = []
    for row in rows:
        try:
            run_root = run_root_of(Path(row["enhanced"]))
            opened.append((row, run_root, read_store(run_root), ""))
        except (OSError, ValueError) as error:
            opened.append((row, None, None, describe_exception(error)))
    members = []
    for _, _, store, _ in opened:
        own = find_measurement(store, OWN_FLOOR) if store is not None else None
        if own is not None and not own.attributes.get("missing"):
            members.append(dict(own.attributes))
    out: list[dict[str, Any]] = []
    for row, run_root, store, error in opened:
        if store is None or run_root is None:
            out.append({**row, "status": ERROR, SESSION_FLOOR: error})
            continue
        write_session_floor(store, members, session=session)
        capture_environments(store, {})
        write_store(store, run_root)
        export_prov(store, run_root)
        written = find_measurement(store, SESSION_FLOOR)
        source = written.attributes.get("floor_source") if written is not None else None
        out.append(
            {
                **row,
                "status": OK,
                "session": session,
                "members_n": len(rows),
                "members_read_n": len(members),
                SESSION_FLOOR: source or "missing",
            }
        )
    return out


def run_slice(manifest: Path, *, slice_index: int, slice_count: int, log_dir: Path) -> dict[str, Any]:
    """Write the session floor for every session in one array task's stride.

    Args:
        manifest: The manifest JSONL.
        slice_index: This task's 0-based index.
        slice_count: How many tasks the array has.
        log_dir: Where this task's ``slices/`` log goes.

    Returns:
        The task's summary.
    """
    started = time.time()
    sessions = sessions_of(read_manifest(manifest, required=("stem", "enhanced")))
    mine = take_slice([{"session": s, "rows": rows} for s, rows in sessions], slice_index, slice_count)
    print(f"[slice {slice_index}/{slice_count}] {len(mine)} sessions", flush=True)
    log: list[dict[str, Any]] = []
    for group in mine:
        log.extend(process_session(group["session"], group["rows"]))
    counts: dict[str, int] = {}
    for record in log:
        key = f"{record['status']}:{record.get(SESSION_FLOOR)}"
        counts[key] = counts.get(key, 0) + 1
    slices_dir = log_dir / SLICES_SUBDIR
    slices_dir.mkdir(parents=True, exist_ok=True)
    label = f"session-floor-slice-{slice_index}-of-{slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    log_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in log), encoding="utf-8")
    summary = {
        "manifest": str(manifest),
        "slice_index": slice_index,
        "slice_count": slice_count,
        "sessions": len(mine),
        "rows": len(log),
        "counts": counts,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Write the session floor for one shard of sessions and print what happened.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when every member store was written, 1 when any is ``error``, 2 on bad arguments.
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
    print(f"Sessions: {summary['sessions']}  Rows: {summary['rows']}")
    print(f"Log:      {summary['log']}")
    for status, number in sorted(summary["counts"].items()):
        print(f"  {status:<24} {number}")
    return 1 if any(key.startswith(ERROR) for key in summary["counts"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
