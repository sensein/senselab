#!/usr/bin/env python3
r"""Withdraw the clip spans a finished triage run's own amplitudes contradict, in place.

    uv run python scripts/extend_withdraw_clips.py MANIFEST --slice-index I --slice-count N \
        [--log-dir DIR] [--config OVERRIDE.yaml]

``MANIFEST`` is a JSONL, one object per line, each carrying ``stem`` and ``enhanced`` (the absolute
path of that recording's ``run/streams/enhanced.flac``) -- the same manifest
``scripts/extend_clip_amplitudes.py`` and ``scripts/extend_quality.py`` take, and the run root is
derived from ``enhanced`` the same way: ``<run_root>/run/store.jsonl`` and ``<run_root>/prov/``.

``--slice-index`` / ``--slice-count`` shard the manifest for a Slurm array: task *i* of *n* takes
``rows[i::n]``.

Every number this pass reads is already in the store, so it opens no audio, loads no model and needs
no venv. **Run it after** ``scripts/extend_clip_amplitudes.py``: a store carrying clip spans with no
``clip_amplitude`` measurement beside them is a refusal, which is that recording's error.

Per withdrawn span the store gains one assertion, in the vocabulary a fresh run's ``_clip_spans``
writes, and the span is superseded; the ``clip_amplitude`` measurement is rewritten over the
survivors and the old one superseded. A store with no contradicted clip span is left byte-identical
and reported ``skipped``; only a store something was withdrawn from has ``run/store.jsonl`` replaced
and ``prov/`` re-exported.

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
    ABSENT,
    CURRENT,
    ERROR,
    OK,
    REWRITTEN,
    RUN_SUBDIR,
    SKIPPED,
    SLICES_SUBDIR,
    SOURCE_STREAM,
    WITHDRAW_CLIPS,
    attempt_derivation,
    export_prov,
    read_manifest,
    read_store,
    run_root_of,
    take_slice,
    withdraw_contradicted_clips,
    write_store,
)
from senselab.audio.workflows.triage.nodes.common import capture_environments, describe_exception
from senselab.audio.workflows.triage.nodes.quality import clip_spans
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


def run_withdrawal(store: ProvStore, config: TriageConfig) -> str:
    """Withdraw one finished run's contradicted clip spans, or say why none was.

    Args:
        store: The run's store.
        config: The triage configuration.

    Returns:
        ``rewritten`` when a span was withdrawn, ``current`` when the store's clip spans stand
        against its own amplitudes, ``absent`` when it carries no clip span at all.
    """
    if not clip_spans(store, SOURCE_STREAM):
        return ABSENT
    return REWRITTEN if withdraw_contradicted_clips(store, config) is not None else CURRENT


def extend_one(run_root: Path, config: TriageConfig) -> dict[str, str]:
    """Withdraw over one finished run and write the store only if it changed.

    Args:
        run_root: The run root.
        config: The triage configuration.

    Returns:
        ``{status, withdraw_contradicted_clips}`` — ``ok`` when a span was withdrawn, ``skipped``
        when the store was already as this pass would leave it, ``error`` when the store could not
        be read or carries clip spans with no amplitudes.
    """
    try:
        store = read_store(run_root)
    except (OSError, ValueError) as error:
        return {"status": ERROR, WITHDRAW_CLIPS: describe_exception(error)}
    before = store.fingerprint()
    outcome = attempt_derivation(lambda: run_withdrawal(store, config))
    if outcome.failed:
        return {"status": ERROR, WITHDRAW_CLIPS: outcome.detail}
    if store.fingerprint() == before:
        return {"status": SKIPPED, WITHDRAW_CLIPS: outcome.detail}
    capture_environments(store, {})
    write_store(store, run_root)
    export_prov(store, run_root)
    return {"status": OK, WITHDRAW_CLIPS: outcome.detail}


def process(rows: Sequence[dict[str, Any]], config: TriageConfig) -> list[dict[str, Any]]:
    """Withdraw over every run named by these rows, one at a time.

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
            out.append({**row, "status": ERROR, WITHDRAW_CLIPS: describe_exception(error)})
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
    outcomes: dict[str, int] = {}
    for record in log:
        outcomes[str(record[WITHDRAW_CLIPS])] = outcomes.get(str(record[WITHDRAW_CLIPS]), 0) + 1

    slices_dir = log_dir / SLICES_SUBDIR
    slices_dir.mkdir(parents=True, exist_ok=True)
    label = f"withdraw-clips-slice-{slice_index}-of-{slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    log_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in log), encoding="utf-8")

    summary = {
        "manifest": str(manifest),
        "slice_index": slice_index,
        "slice_count": slice_count,
        "config_hash": config.config_hash,
        "rows": len(mine),
        "counts": counts,
        "outcomes": outcomes,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Withdraw contradicted clip spans over every run in one shard and print what happened.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when every recording in the shard was read, 1 when any row is ``error`` — the other stores
        are written either way — and 2 when the arguments could not be resolved and nothing was read.
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
    for outcome, number in sorted(summary["outcomes"].items()):
        print(f"  {WITHDRAW_CLIPS} {outcome:<9} {number}")
    return 1 if summary["counts"].get(ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
