#!/usr/bin/env python3
r"""Write SESSION's floor into finished triage runs, one session at a time.

    uv run python scripts/extend_session_floor.py MANIFEST --slice-index I --slice-count N --profile PROFILE \
        [--config OVERRIDE.yaml] [--log-dir DIR]

``MANIFEST`` is the JSONL every extend driver takes (``stem``, ``enhanced``, and the corpus table's identity
columns). Rows are grouped by the session their recorded identity names (``<participant>/<session>``; a recording
naming no participant and session is its own group, keyed by its recording id); ``--slice-index`` /
``--slice-count`` shard the sessions, never a session's members, so each task sees every member of the sessions it
owns. A session any of whose
members' runs is refused (a stale manifest row, a recording or profile that differs from the run's record) is
written into nowhere. ``--profile`` names the dataset profile the runs were made with. Every member's store must
already carry PREPROCESS's ``background_floor`` (``scripts/extend_background_floor.py``); a member without one
is left out of its session's floor, and its own store records the missing floor. Each store gains one
``speech_residual`` measurement, read off its plain, enhanced and residual streams, and one ``session_floor``
measurement carrying the session floor and the session background over the session's speech tasks' residuals. Run
``scripts/extend_replay_decisions.py`` afterwards: BACKGROUND reads this floor.

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

from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.extend import (
    ERROR,
    OK,
    RUN_SUBDIR,
    SLICES_SUBDIR,
    ReplayRefused,
    StaleManifestRow,
    add_config_arguments,
    driver_config,
    export_prov,
    read_checked_store,
    read_manifest,
    read_store,
    run_root_of,
    take_slice,
    write_store,
)
from senselab.audio.workflows.triage.identity import row_groups
from senselab.audio.workflows.triage.nodes.background import (
    OWN_FLOOR,
    SESSION_FLOOR,
    write_session_floor,
    write_speech_residual,
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
    add_config_arguments(parser)
    return parser


def sessions_of(rows: Sequence[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """The manifest's rows grouped by the session their identity columns name, sessions in sorted order.

    Args:
        rows: Every manifest row, carrying the corpus table's identity columns.

    Returns:
        ``(session, rows)`` pairs; a recording naming no participant and session, or whose run records no
        identity, is its own group, keyed by its stem.

    Raises:
        ManifestWithoutIdentity: If a row carries no identity columns.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row_groups(row)[1], []).append(row)
    return sorted(groups.items())


def process_session(
    session: str, rows: Sequence[dict[str, Any]], config: TriageConfig | None = None
) -> list[dict[str, Any]]:
    """Write the session floor into every member store of one session.

    Args:
        session: The session key.
        rows: Its members' manifest rows.
        config: The configuration each store is checked against (:func:`read_checked_store`), or None for the
            recording alone.

    Returns:
        One outcome record per member. Where any member's run is refused (:class:`ReplayRefused`: a stale row, or
        a recording or profile that differs from the run's record), every member is an error and nothing is
        written: the floor of the rest would be read without it.
    """
    opened: list[tuple[dict[str, Any], Path | None, ProvStore | None, str]] = []
    stale: list[str] = []
    refused: list[str] = []
    for row in rows:
        try:
            run_root = run_root_of(Path(row["enhanced"]))
            opened.append((row, run_root, read_checked_store(run_root, config, row=row), ""))
        except (OSError, ValueError) as error:
            if isinstance(error, StaleManifestRow):
                stale.append(str(row["stem"]))
            elif isinstance(error, ReplayRefused):
                refused.append(f"{row['stem']} ({describe_exception(error)})")
            opened.append((row, None, None, describe_exception(error)))
    if stale or refused:
        why = "; ".join(
            [f"the manifest is stale for {', '.join(sorted(stale))}"] * bool(stale)
            + [f"the run of {member} is refused" for member in sorted(refused)]
        )
        return [{**row, "status": ERROR, SESSION_FLOOR: f"session refused: {why}; nothing was written"} for row in rows]
    members = []
    readings = []
    for _, member_root, store, _ in opened:
        own = find_measurement(store, OWN_FLOOR) if store is not None else None
        if own is not None and not own.attributes.get("missing"):
            members.append(dict(own.attributes))
        if store is not None and member_root is not None:
            readings.append(write_speech_residual(store, member_root / RUN_SUBDIR))
    out: list[dict[str, Any]] = []
    for row, member_root, store, failure in opened:
        if store is None or member_root is None:
            out.append({**row, "status": ERROR, SESSION_FLOOR: failure})
            continue
        write_session_floor(store, members, session=session, background_members=readings)
        capture_environments(store, {})
        write_store(store, member_root)
        export_prov(store, member_root)
        written = find_measurement(store, SESSION_FLOOR)
        source = written.attributes.get("floor_source") if written is not None else None
        background = (written.attributes.get("session_background") or {}) if written is not None else {}
        out.append(
            {
                **row,
                "status": OK,
                "session": session,
                "members_n": len(rows),
                "members_read_n": len(members),
                SESSION_FLOOR: source or "missing",
                "session_background": background.get("source") or background.get("fallback") or "missing",
                "session_background_members_n": background.get("members_n", 0),
            }
        )
    return out


def run_slice(
    manifest: Path, *, slice_index: int, slice_count: int, log_dir: Path, config: TriageConfig | None = None
) -> dict[str, Any]:
    """Write the session floor for every session in one array task's stride.

    Args:
        manifest: The manifest JSONL.
        slice_index: This task's 0-based index.
        slice_count: How many tasks the array has.
        log_dir: Where this task's ``slices/`` log goes.
        config: As :func:`process_session`.

    Returns:
        The task's summary.
    """
    started = time.time()
    sessions = sessions_of(read_manifest(manifest, required=("stem", "enhanced")))
    mine = take_slice([{"session": s, "rows": rows} for s, rows in sessions], slice_index, slice_count)
    print(f"[slice {slice_index}/{slice_count}] {len(mine)} sessions", flush=True)
    log: list[dict[str, Any]] = []
    for group in mine:
        log.extend(process_session(group["session"], group["rows"], config))
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
        config = driver_config(args)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    try:
        summary = run_slice(
            args.manifest,
            slice_index=args.slice_index,
            slice_count=args.slice_count,
            log_dir=args.log_dir if args.log_dir is not None else args.manifest.parent,
            config=config,
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
