#!/usr/bin/env python3
r"""Extend finished triage runs with the posteriorgram and Praat blocks, in place.

    uv run python scripts/extend_ppg_praat.py MANIFEST --slice-index I --slice-count N \
        [--batch-size 500] [--device cpu]

``MANIFEST`` is a JSONL, one object per line, each carrying ``stem``, ``enhanced`` (the absolute
path of that recording's ``run/streams/enhanced.flac``), ``family``, ``duration_s`` and ``lexical``.
The run root is the enhanced stream's third parent, so ``store.jsonl`` is
``<run_root>/run/store.jsonl`` and the BEP028 tree is ``<run_root>/prov/``; the manifest carries no
run root of its own and this layout is what the driver relies on.

``--slice-index`` / ``--slice-count`` shard the manifest for a Slurm array: task *i* of *n* takes
``rows[i::n]``, so every task sees the same stride of long and short recordings.

Nothing is written outside the recording's own run. Each recording's store is read with
``ProvStore.read_jsonl``, the two PREPROCESS blocks add their entities to it, its environments are
captured, the merged store replaces ``run/store.jsonl`` and ``prov/`` is re-exported from it.

The ppgs venv must already exist. This driver refuses rather than building one, because a cold
build outlasts the venv lock's patience and every task racing it waits:

    uv run python -c \
        "from senselab.audio.tasks.features_extraction import ensure_ppgs_venv; ensure_ppgs_venv()"

A recording whose store already holds both measurements is skipped, so a task that dies mid-way
restarts where it stopped and a completed slice re-run changes nothing.

The reasoning -- the batch-size benchmark, the venv-lock hazard, the convergence argument -- is in
``specs/20260911-ppg-praat-batch/design.md``.

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

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction import (
    PpgsPosteriorgramUnavailable,
    extract_ppgs_from_audios,
    ppgs_venv_is_provisioned,
)
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.extend import (
    RUN_SUBDIR,
    SLICES_SUBDIR,
    batches,
    export_prov,
    read_manifest,
    read_store,
    run_root_of,
    take_slice,
    write_store,
)
from senselab.audio.workflows.triage.nodes.common import capture_environments, describe_exception, find_measurement
from senselab.audio.workflows.triage.nodes.preprocess import (
    PPG_MEASUREMENT,
    PRAAT_MEASUREMENT,
    ppg_input,
    praat_features,
    write_ppg_posteriorgram,
)
from senselab.utils.data_structures import DeviceType
from senselab.utils.prov_store import ProvStore
from senselab.utils.subprocess_venv import record_venv_use

DEFAULT_BATCH_SIZE = 500
"""Recordings per ``extract_ppgs_from_audios`` call. Its derivation is in the spec."""

_OK = "ok"
_ABSENT = "absent"
_ERROR = "error"
_SKIPPED = "skipped"


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
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Recordings per ppgs call (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=[DeviceType.CPU.value, DeviceType.CUDA.value],
        help="Device ppgs runs on (default: auto)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=f"Where this task's {SLICES_SUBDIR}/ log goes (default: beside the manifest)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Partial YAML deep-merged over the packaged config")
    return parser


def pending(store: ProvStore) -> tuple[bool, bool]:
    """Which of the two measurements this store is still missing.

    Args:
        store: The run's store.

    Returns:
        ``(ppg_pending, praat_pending)``.
    """
    return find_measurement(store, PPG_MEASUREMENT) is None, find_measurement(store, PRAAT_MEASUREMENT) is None


def _posteriorgrams(audios: list[Audio], device: DeviceType | None) -> tuple[list[Any] | None, str | None]:
    """Run one batch through ppgs, turning a whole-batch failure into a message rather than a raise.

    Args:
        audios: The batch's conditioned audios.
        device: Where ppgs runs.

    Returns:
        ``(results, None)`` when the call returned one entry per audio, else ``(None, message)``.
    """
    try:
        out = extract_ppgs_from_audios(audios, device=device)
    except Exception as error:  # noqa: BLE001 — every recording in the batch records it and survives
        return None, describe_exception(error)
    if len(out) != len(audios):
        return None, f"ppgs returned {len(out)} results for {len(audios)} audios"
    return list(out), None


def process_batch(
    rows: Sequence[dict[str, Any]],
    config: TriageConfig,
    *,
    device: DeviceType | None,
    used_venvs: dict[str, Path],
) -> list[dict[str, Any]]:
    """Extend every run in one batch, one ppgs call across all of them.

    A recording whose store will not open, whose posteriorgram the model refused, or whose Praat
    call raised still gets an outcome record; the rest of the batch is unaffected and every
    recording that produced something still has its store and its ``prov/`` rewritten.

    Args:
        rows: The batch's manifest rows.
        config: The triage configuration.
        device: Where ppgs runs.
        used_venvs: The venv-use record the slice opened, read when each store is written.

    Returns:
        One outcome record per input row, in order.
    """
    outcomes: dict[int, dict[str, Any]] = {}
    opened: dict[int, tuple[Path, ProvStore]] = {}
    inputs: list[tuple[int, str, Audio]] = []

    for position, row in enumerate(rows):
        try:
            run_root = run_root_of(Path(row["enhanced"]))
            store = read_store(run_root)
        except (OSError, ValueError) as error:
            outcomes[position] = {
                "status": _ERROR,
                "ppg": describe_exception(error),
                "praat": describe_exception(error),
            }
            continue
        ppg_pending, praat_pending = pending(store)
        if not ppg_pending and not praat_pending:
            outcomes[position] = {"status": _SKIPPED, "ppg": _SKIPPED, "praat": _SKIPPED}
            continue
        opened[position] = (run_root, store)
        if not ppg_pending:
            continue
        try:
            enhanced_id, audio = ppg_input(store, run_root / RUN_SUBDIR)
        except (OSError, ValueError, LookupError) as error:
            # No enhanced stream to read is both blocks' absence, not one's: Praat reads it too.
            reason = describe_exception(error)
            outcomes[position] = {"status": _ABSENT, "ppg": reason, "praat": reason}
            continue
        inputs.append((position, enhanced_id, audio))

    results, batch_error = _posteriorgrams([audio for _, _, audio in inputs], device) if inputs else ([], None)

    ppg_status: dict[int, str] = {}
    for offset, (position, enhanced_id, audio) in enumerate(inputs):
        run_root, store = opened[position]
        if results is None:
            ppg_status[position] = batch_error or "ppgs produced no result"
            continue
        result = results[offset]
        if isinstance(result, PpgsPosteriorgramUnavailable):
            ppg_status[position] = describe_exception(result)
            continue
        try:
            write_ppg_posteriorgram(
                store,
                run_dir=run_root / RUN_SUBDIR,
                enhanced_id=enhanced_id,
                audio=audio,
                posteriorgram=result,
            )
        except Exception as error:  # noqa: BLE001 — one recording's write is not the batch's
            ppg_status[position] = describe_exception(error)
            continue
        ppg_status[position] = _OK

    written: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        if position in outcomes:
            written.append({**row, **outcomes[position]})
            continue
        run_root, store = opened[position]
        ppg = ppg_status.get(position, _SKIPPED)
        _, praat_pending = pending(store)
        praat: str
        if not praat_pending:
            praat = _SKIPPED
        else:
            try:
                praat_features(store, config, run_dir=run_root / RUN_SUBDIR)
                praat = _OK
            except Exception as error:  # noqa: BLE001 — a recording Praat refuses is an outcome
                praat = describe_exception(error)
        capture_environments(store, used_venvs)
        write_store(store, run_root)
        export_prov(store, run_root)
        landed = [ppg in (_OK, _SKIPPED), praat in (_OK, _SKIPPED)]
        status = _OK if all(landed) else (_ABSENT if any(landed) else _ERROR)
        written.append({**row, "status": status, "ppg": ppg, "praat": praat})
    return written


def run_slice(
    manifest: Path,
    *,
    slice_index: int,
    slice_count: int,
    batch_size: int,
    device: DeviceType | None,
    config: TriageConfig,
    log_dir: Path,
) -> dict[str, Any]:
    """Extend every run in one array task's stride of the manifest.

    Args:
        manifest: The manifest JSONL.
        slice_index: This task's 0-based index.
        slice_count: How many tasks the array has.
        batch_size: Recordings per ppgs call.
        device: Where ppgs runs.
        config: The triage configuration.
        log_dir: Where this task's ``slices/`` log goes.

    Returns:
        The task's summary: its counts, its parameters, and where its log went.
    """
    started = time.time()
    mine = take_slice(read_manifest(manifest, required=("stem", "enhanced")), slice_index, slice_count)
    print(
        f"[slice {slice_index}/{slice_count}] {len(mine)} rows, in batches of {batch_size}",
        flush=True,
    )

    log: list[dict[str, Any]] = []
    with record_venv_use() as used_venvs:
        for number, batch in enumerate(batches(mine, batch_size), start=1):
            batch_started = time.time()
            log.extend(process_batch(batch, config, device=device, used_venvs=used_venvs))
            print(
                f"[slice {slice_index}/{slice_count}] batch {number}: {len(batch)} recordings in "
                f"{time.time() - batch_started:.0f}s",
                flush=True,
            )

    counts: dict[str, int] = {}
    for record in log:
        counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1

    slices_dir = log_dir / SLICES_SUBDIR
    slices_dir.mkdir(parents=True, exist_ok=True)
    label = f"slice-{slice_index}-of-{slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    log_path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in log), encoding="utf-8")

    summary = {
        "manifest": str(manifest),
        "slice_index": slice_index,
        "slice_count": slice_count,
        "batch_size": batch_size,
        "device": device.value if device is not None else None,
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
        ``error`` — the stores are written either way — and 2 when the arguments could not be
        resolved, or the ppgs venv is absent, and nothing was measured.
    """
    args = build_parser().parse_args(argv)

    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    if not ppgs_venv_is_provisioned():
        print(
            "ERROR: no ppgs venv on this host. Build it once, on its own, before submitting the array:\n"
            '  uv run python -c "from senselab.audio.tasks.features_extraction import ensure_ppgs_venv; '
            'ensure_ppgs_venv()"',
            file=sys.stderr,
        )
        return 2
    try:
        summary = run_slice(
            args.manifest,
            slice_index=args.slice_index,
            slice_count=args.slice_count,
            batch_size=args.batch_size,
            device=DeviceType(args.device) if args.device else None,
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
    return 1 if summary["counts"].get(_ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
