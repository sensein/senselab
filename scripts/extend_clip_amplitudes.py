#!/usr/bin/env python3
r"""Extend finished triage runs with the clip-amplitude measurement QUALITY reads, in place.

    uv run python scripts/extend_clip_amplitudes.py MANIFEST --slice-index I --slice-count N \
        [--log-dir DIR] [--config OVERRIDE.yaml]

``MANIFEST`` is a JSONL, one object per line, each carrying ``stem`` and ``enhanced`` (the absolute
path of that recording's ``run/streams/enhanced.flac``) -- the same manifest
``scripts/extend_ppg_praat.py`` takes, and the run root is derived from ``enhanced`` the same way:
``<run_root>/run/store.jsonl`` and ``<run_root>/prov/``.

**The source audio is not the enhanced stream and is not in the run tree.** It is the original
recording, whose absolute path and SHA-256 ADMIT recorded on the ``recording`` stream entity, so
every run names its own source and this driver holds no corpus path of its own. A recording whose
bytes no longer match that digest is refused rather than measured.

``--slice-index`` / ``--slice-count`` shard the manifest for a Slurm array: task *i* of *n* takes
``rows[i::n]``.

Nothing is written outside the recording's own run: the store gains one activity and one
measurement, ``run/store.jsonl`` is replaced atomically and ``prov/`` is re-exported. No clip span
is touched, and no other PREPROCESS block is recomputed. A recording whose store already holds a
live ``clip_amplitude`` measurement is skipped and its store is not rewritten at all.

There is no venv, no model and no batching, which is why this is its own driver rather than a mode
of ``extend_ppg_praat.py``; what the two share -- the run layout, the store read/write, the BEP028
re-export, the manifest slicing -- is ``senselab.audio.workflows.triage.extend``.

The design is in ``specs/20260912-quality-clip-consistency/design.md``.

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

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
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
from senselab.audio.workflows.triage.nodes.common import capture_environments, describe_exception, find_measurement
from senselab.audio.workflows.triage.nodes.preprocess import extend_clip_amplitudes
from senselab.audio.workflows.triage.nodes.quality import CLIP_AMPLITUDE_MEASUREMENT
from senselab.utils.prov_store import ProvStore


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


NO_SPAN = "no clip span over the recording; QUALITY needs no measurement to read"
"""What a store with nothing to measure records. Nothing was asserted, so nothing is read."""


def measure(store: ProvStore, config: TriageConfig, *, run_dir: Path) -> str:
    """Append one finished run's clip amplitudes, or say there are none to append.

    Args:
        store: The run's store.
        config: The triage configuration.
        run_dir: The run directory stream paths are relative to.

    Returns:
        The measurement's id, or :data:`NO_SPAN` when the store carries no clip span.
    """
    measurement_id = extend_clip_amplitudes(store, config, run_dir=run_dir)
    return NO_SPAN if measurement_id is None else measurement_id


def extend_one(run_root: Path, config: TriageConfig) -> tuple[str, str]:
    """Give one finished run the measurement it is missing, or say why it gets none.

    Args:
        run_root: The run root.
        config: The triage configuration.

    Returns:
        ``(status, detail)`` — ``ok`` and the measurement id, ``skipped`` and why nothing was
        needed, or ``error`` and the reason.
    """
    try:
        store = read_store(run_root)
    except (OSError, ValueError) as error:
        return ERROR, describe_exception(error)
    if find_measurement(store, CLIP_AMPLITUDE_MEASUREMENT) is not None:
        return SKIPPED, "the store already carries a live clip_amplitude measurement"
    before = store.fingerprint()
    outcome = attempt_derivation(lambda: measure(store, config, run_dir=run_root / RUN_SUBDIR))
    if outcome.failed:
        return ERROR, outcome.detail
    if store.fingerprint() == before:
        return SKIPPED, outcome.detail
    capture_environments(store, {})
    write_store(store, run_root)
    export_prov(store, run_root)
    return OK, outcome.detail


def process(rows: Sequence[dict[str, Any]], config: TriageConfig) -> list[dict[str, Any]]:
    """Extend every run named by these rows, one at a time.

    A recording whose store will not open, whose source recording has moved or whose bytes changed
    gets an outcome record; its neighbours are unaffected.

    Args:
        rows: The manifest rows this task owns.
        config: The triage configuration.

    Returns:
        One outcome record per input row, in order.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        try:
            run_root = run_root_of(Path(row["enhanced"]))
        except ValueError as error:
            out.append({**row, "status": ERROR, "clip_amplitude": describe_exception(error)})
            continue
        status, detail = extend_one(run_root, config)
        out.append({**row, "status": status, "clip_amplitude": detail})
    return out


def run_slice(
    manifest: Path,
    *,
    slice_index: int,
    slice_count: int,
    config: TriageConfig,
    log_dir: Path,
) -> dict[str, Any]:
    """Extend every run in one array task's stride of the manifest.

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

    log = process(mine, config)

    counts: dict[str, int] = {}
    for record in log:
        counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1

    slices_dir = log_dir / SLICES_SUBDIR
    slices_dir.mkdir(parents=True, exist_ok=True)
    label = f"clip-amplitudes-slice-{slice_index}-of-{slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    log_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in log), encoding="utf-8")

    summary = {
        "manifest": str(manifest),
        "slice_index": slice_index,
        "slice_count": slice_count,
        "config_hash": config.config_hash,
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
        0 when every recording in the shard reached a determinate outcome, 1 when any row is
        ``error`` — the other stores are written either way — and 2 when the arguments could not be
        resolved and nothing was measured.
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
        print(f"  {status:<9} {number}")
    return 1 if summary["counts"].get(ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
