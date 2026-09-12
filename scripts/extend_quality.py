#!/usr/bin/env python3
r"""Give a finished triage run the QUALITY verdict its graph pass never reached, in place.

    uv run python scripts/extend_quality.py MANIFEST --slice-index I --slice-count N \
        [--log-dir DIR] [--config OVERRIDE.yaml]

``MANIFEST`` is a JSONL, one object per line, each carrying ``stem`` and ``enhanced`` (the absolute
path of that recording's ``run/streams/enhanced.flac``) -- the same manifest
``scripts/extend_clip_amplitudes.py`` and ``scripts/extend_reprocessed_outputs.py`` take, and the
run root is derived from ``enhanced`` the same way: ``<run_root>/run/store.jsonl`` and
``<run_root>/prov/``.

``--slice-index`` / ``--slice-count`` shard the manifest for a Slurm array: task *i* of *n* takes
``rows[i::n]``.

QUALITY reads stored outputs only -- no audio, no sidecar, no stream decode -- so a finished run
holds every input it takes, and this pass is the node itself over the store it would have read.
**Run it after** ``scripts/extend_clip_amplitudes.py``: a store carrying clip spans with no
``clip_amplitude`` measurement beside them is a refusal, which is that recording's error.

A run with no clip span at all is not a refusal. QUALITY passes it -- nothing was asserted, so
nothing can be contradicted -- and the verdict saying so is written like any other.

Nothing is written outside the recording's own run: the store gains one activity, one assertion per
contested clip span and one verdict, ``run/store.jsonl`` is replaced atomically and ``prov/`` is
re-exported. A store already carrying a live QUALITY verdict is skipped and not rewritten at all.

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
    PRESENT,
    RUN_SUBDIR,
    SKIPPED,
    SLICES_SUBDIR,
    attempt_derivation,
    export_prov,
    extend_quality,
    read_manifest,
    read_store,
    run_root_of,
    take_slice,
    write_store,
)
from senselab.audio.workflows.triage.nodes.common import capture_environments, describe_exception
from senselab.audio.workflows.triage.vocabulary import QUALITY
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


def run_quality(store: ProvStore, config: TriageConfig, *, run_dir: Path) -> str:
    """Read one finished run for internal contradiction, or say why it is not read again.

    Args:
        store: The run's store.
        config: The triage configuration.
        run_dir: The run directory, for the shared node shape.

    Returns:
        The verdict's outcome — ``pass`` or ``flag`` — or ``present`` when the store already carries
        a live QUALITY verdict.
    """
    result = extend_quality(store, config, run_dir=run_dir)
    return PRESENT if result is None else result.verdict.outcome.value


def extend_one(run_root: Path, config: TriageConfig) -> dict[str, str]:
    """Run QUALITY over one finished run and write the store only if it changed.

    The fingerprint before and after is the convergence argument the other extend drivers use: a
    second pass recomputes the activity, the assertions and the verdict the store already holds,
    which is a set-union no-op, and a store whose fingerprint did not move has nothing to write
    back. A refusal leaves the store unwritten, so a recording QUALITY cannot read keeps the store
    it had.

    Args:
        run_root: The run root.
        config: The triage configuration.

    Returns:
        ``{status, QUALITY}`` — ``ok`` when the verdict landed, ``skipped`` when the store was
        already as this pass would leave it, ``error`` when QUALITY refused the store.
    """
    try:
        store = read_store(run_root)
    except (OSError, ValueError) as error:
        return {"status": ERROR, QUALITY: describe_exception(error)}
    before = store.fingerprint()
    outcome = attempt_derivation(lambda: run_quality(store, config, run_dir=run_root / RUN_SUBDIR))
    if outcome.failed:
        return {"status": ERROR, QUALITY: outcome.detail}
    if store.fingerprint() == before:
        return {"status": SKIPPED, QUALITY: outcome.detail}
    capture_environments(store, {})
    write_store(store, run_root)
    export_prov(store, run_root)
    return {"status": OK, QUALITY: outcome.detail}


def process(rows: Sequence[dict[str, Any]], config: TriageConfig) -> list[dict[str, Any]]:
    """Run QUALITY over every run named by these rows, one at a time.

    A recording whose store will not open, or whose clip spans carry no amplitudes, gets an outcome
    record; its neighbours are unaffected.

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
            out.append({**row, "status": ERROR, QUALITY: describe_exception(error)})
            continue
        out.append({**row, **extend_one(run_root, config)})
    return out


def run_slice(
    manifest: Path,
    *,
    slice_index: int,
    slice_count: int,
    config: TriageConfig,
    log_dir: Path,
) -> dict[str, Any]:
    """Read every run in one array task's stride of the manifest.

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
    verdicts: dict[str, int] = {}
    for record in log:
        verdicts[str(record[QUALITY])] = verdicts.get(str(record[QUALITY]), 0) + 1

    slices_dir = log_dir / SLICES_SUBDIR
    slices_dir.mkdir(parents=True, exist_ok=True)
    label = f"quality-slice-{slice_index}-of-{slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    log_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in log), encoding="utf-8")

    summary = {
        "manifest": str(manifest),
        "slice_index": slice_index,
        "slice_count": slice_count,
        "config_hash": config.config_hash,
        "rows": len(mine),
        "counts": counts,
        "verdicts": verdicts,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run QUALITY over every run in one shard and print what happened.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when every recording in the shard reached a verdict or already had one, 1 when any row is
        ``error`` — the other stores are written either way — and 2 when the arguments could not be
        resolved and nothing was read.
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
    for outcome, number in sorted(summary["verdicts"].items()):
        print(f"  {QUALITY} {outcome:<9} {number}")
    return 1 if summary["counts"].get(ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
