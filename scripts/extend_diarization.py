#!/usr/bin/env python3
r"""Extend finished triage runs with the PREPROCESS diarization block, in place.

    uv run python scripts/extend_diarization.py MANIFEST --slice-index I --slice-count N \
        [--batch-size 200] [--device cpu] [--config variant.yaml] [--force]

``MANIFEST`` is a JSONL, one object per line, each carrying ``stem`` and ``enhanced`` — the same
manifest ``extend_ppg_praat.py`` takes. The run root is the named stream's third parent, so
``store.jsonl`` is ``<run_root>/run/store.jsonl`` and the BEP028 tree is ``<run_root>/prov/``.

**The manifest path locates the run; it does not choose the streams.** Which streams are diarized
is ``diarization.streams``'s decision — ``[enhanced, residual]`` as packaged — and the block reads
each of them back out of the store. One measurement per stream, ``<stream>_diarization``, and the
two counts are reported side by side in the slice log and **never summed**: the enhanced half says
how many voices survived enhancement, the residual half says whether one was removed.

``--slice-index`` / ``--slice-count`` shard the manifest for a Slurm array: task *i* of *n* takes
``rows[i::n]``, so every task sees the same stride of long and short recordings.

Nothing is written outside the recording's own run. Each recording's store is read with
``ProvStore.read_jsonl``, the block adds its entities to it, its environments are captured, the
merged store replaces ``run/store.jsonl`` and ``prov/`` is re-exported from it.

Skipping is **per stream**, so a pass that adds ``residual`` to runs that already hold ``enhanced``
diarizes only the residual; a task that dies mid-way restarts where it stopped, and a completed
slice re-run writes nothing at all. ``--force`` re-derives every configured stream and, where a new
reading differs from the stored one, supersedes the stored one.

A residual holding no voice is the **ordinary** outcome and is recorded as ``n_speakers: 0`` — a
value, not an absence and not a failure. An absence here means the stream could not be read or the
model could not be obtained, which is a different fact.

This driver measures. It writes no multi-voice verdict and no gate: what a speaker count above one
*means* for a recording is QUALITY's judgement, taken against each branch's refined spans.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.extend import (
    ABSENT,
    DIARIZATION_MEASUREMENT_SUPERSEDED,
    ERROR,
    OK,
    RUN_SUBDIR,
    SKIPPED,
    SLICES_SUBDIR,
    attempt_derivation,
    batches,
    export_prov,
    read_manifest,
    read_store,
    run_root_of,
    supersede,
    take_slice,
    write_store,
)
from senselab.audio.workflows.triage.nodes.common import (
    capture_environments,
    describe_exception,
    find_measurement,
    software_agent,
)
from senselab.audio.workflows.triage.nodes.preprocess import NODE as PREPROCESS_NODE
from senselab.audio.workflows.triage.nodes.preprocess import (
    diarization,
    diarization_measurement,
    diarization_streams,
)
from senselab.utils.data_structures import DeviceType
from senselab.utils.prov_store import ProvStore
from senselab.utils.subprocess_venv import record_venv_use

DEFAULT_BATCH_SIZE = 200
"""Recordings per progress report. The diarizer runs one recording at a time; see the spec."""

_SUPERSEDED_REASON = "it was measured before this pass; the diarizer read the same stream again"


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
        help=f"Recordings per progress report (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--device",
        default=None,
        choices=[DeviceType.CPU.value, DeviceType.CUDA.value],
        help="Device the diarizer runs on (default: auto)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help=f"Where this task's {SLICES_SUBDIR}/ log goes (default: beside the manifest)",
    )
    parser.add_argument("--config", type=Path, default=None, help="Partial YAML deep-merged over the packaged config")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-derive the diarization where the store already holds one, superseding the old",
    )
    return parser


def pending(store: ProvStore, streams: Sequence[str], *, force: bool) -> list[str]:
    """Which of the configured streams this store is still owed a diarization of.

    Per stream rather than per store: a pass that adds ``residual`` to runs that already hold
    ``enhanced`` must diarize only the residual, and must still skip the half that is there.

    Args:
        store: The run's store.
        streams: The configured streams, in order.
        force: Re-derive every stream even where one is held.

    Returns:
        The streams to measure, in the configured order.
    """
    if force:
        return list(streams)
    return [name for name in streams if find_measurement(store, diarization_measurement(name)) is None]


def _derive_one(
    store: ProvStore,
    config: TriageConfig,
    run_dir: Path,
    stream: str,
    held: Any,  # noqa: ANN401 — the stored Entity, or None
    device: DeviceType | None,
) -> str:
    """Diarize one stream of one run, retiring a stored reading the new one replaces.

    Args:
        store: The run's store.
        config: The triage configuration.
        run_dir: The run's ``run/`` directory.
        stream: Which stream to diarize.
        held: The measurement a forced pass is replacing, or None.
        device: Where the diarizer runs.

    Returns:
        :data:`OK`.
    """
    written_id = diarization(store, config, run_dir=run_dir, stream=stream, device=device)
    if held is not None and held.id != written_id:
        supersede(
            store,
            held.id,
            node=PREPROCESS_NODE,
            step=DIARIZATION_MEASUREMENT_SUPERSEDED,
            reason=_SUPERSEDED_REASON,
            software=software_agent(store),
        )
    return OK


def _counts_of(store: ProvStore, streams: Sequence[str]) -> dict[str, Any]:
    """Per stream, what the store's live diarization of it says; for the slice log.

    The two counts are reported side by side and never summed: ``enhanced`` says how many voices
    survived enhancement, ``residual`` says whether one was removed, and a reader needs to know
    which stream saw what.

    Args:
        store: The run's store.
        streams: The configured streams.

    Returns:
        ``<stream>_n_speakers`` and friends per stream the store holds a reading of.
    """
    out: dict[str, Any] = {}
    for stream in streams:
        measurement = find_measurement(store, diarization_measurement(stream))
        if measurement is None:
            continue
        for key in ("n_speakers", "n_segments", "speech_s", "overlap_s"):
            if key in measurement.attributes:
                out[f"{stream}_{key}"] = measurement.attributes[key]
    return out


def process_batch(
    rows: Sequence[dict[str, Any]],
    config: TriageConfig,
    *,
    device: DeviceType | None,
    used_venvs: dict[str, Path],
    force: bool = False,
) -> list[dict[str, Any]]:
    """Extend every run in one batch with one diarization per configured stream.

    A recording whose store will not open, or a stream whose diarizer could not run, still gets an
    outcome record; the rest of the batch and the row's other streams are unaffected. A store with
    no live stream of a configured name is that stream's ``error``, not an absence — the shared
    rule in ``extend.py`` reads a ``LookupError`` as a failure, and a manifest naming a run whose
    stream the configuration cannot find is a manifest that does not match the configuration.

    The row's status is the worst of its streams': ``error`` if any stream failed, else ``absent``
    if any was unavailable, else ``skipped`` if every stream was already held, else ``ok``.

    Args:
        rows: The batch's manifest rows.
        config: The triage configuration.
        device: Where the diarizer runs.
        used_venvs: The venv-use record the slice opened, read when each store is written.
        force: Re-derive every stream on stores that already hold one, superseding the stored
            reading when the new one differs.

    Returns:
        One outcome record per input row, in order.
    """
    streams = diarization_streams(config)
    written: list[dict[str, Any]] = []
    for row in rows:
        try:
            run_root = run_root_of(Path(row["enhanced"]))
            store = read_store(run_root)
        except (OSError, ValueError) as error:
            reason = describe_exception(error)
            written.append({**row, "status": ERROR, **{f"{name}_diarization": reason for name in streams}})
            continue
        owed = pending(store, streams, force=force)
        outcomes = {f"{name}_diarization": SKIPPED for name in streams if name not in owed}
        failed: set[str] = set()
        for stream in owed:
            held = find_measurement(store, diarization_measurement(stream)) if force else None
            outcome = attempt_derivation(
                functools.partial(_derive_one, store, config, run_root / RUN_SUBDIR, stream, held, device)
            )
            outcomes[f"{stream}_diarization"] = outcome.detail
            if outcome.failed:
                failed.add(stream)
        if owed:
            capture_environments(store, used_venvs)
            write_store(store, run_root)
            export_prov(store, run_root)
        written.append(
            {
                **row,
                "status": _row_status(outcomes.values(), any_failed=bool(failed)),
                "device": _device_name(device),
                **outcomes,
                **_counts_of(store, streams),
            }
        )
    return written


def _row_status(details: Any, *, any_failed: bool) -> str:  # noqa: ANN401 — an iterable of outcome details
    """One recording's status over its streams': the worst any of them reached.

    The per-stream detail keeps the failure's own message — a status word alone would throw away
    the only record of *why* a stream produced nothing.

    Args:
        details: The per-stream outcome details.
        any_failed: Whether any stream's derivation failed.

    Returns:
        :data:`ERROR`, :data:`ABSENT`, :data:`SKIPPED` or :data:`OK`.
    """
    words = list(details)
    if any_failed:
        return ERROR
    if any(str(word).startswith(ABSENT) for word in words):
        return ABSENT
    return SKIPPED if words and all(word == SKIPPED for word in words) else OK


def _device_name(device: DeviceType | None) -> str | None:
    """The device the operator asked for, as the log records it."""
    return device.value if device is not None else None


def run_slice(
    manifest: Path,
    *,
    slice_index: int,
    slice_count: int,
    batch_size: int,
    device: DeviceType | None,
    config: TriageConfig,
    log_dir: Path,
    force: bool = False,
) -> dict[str, Any]:
    """Diarize every run in one array task's stride of the manifest.

    Args:
        manifest: The manifest JSONL.
        slice_index: This task's 0-based index.
        slice_count: How many tasks the array has.
        batch_size: Recordings per progress report.
        device: Where the diarizer runs.
        config: The triage configuration.
        log_dir: Where this task's ``slices/`` log goes.
        force: Re-derive on stores that already hold a diarization.

    Returns:
        The task's summary: its counts, its parameters, and where its log went.
    """
    started = time.time()
    mine = take_slice(read_manifest(manifest, required=("stem", "enhanced")), slice_index, slice_count)
    print(f"[slice {slice_index}/{slice_count}] {len(mine)} rows, in batches of {batch_size}", flush=True)

    log: list[dict[str, Any]] = []
    with record_venv_use() as used_venvs:
        for number, batch in enumerate(batches(mine, batch_size), start=1):
            batch_started = time.time()
            log.extend(process_batch(batch, config, device=device, used_venvs=used_venvs, force=force))
            print(
                f"[slice {slice_index}/{slice_count}] batch {number}: {len(batch)} recordings in "
                f"{time.time() - batch_started:.0f}s",
                flush=True,
            )

    streams = diarization_streams(config)
    counts: dict[str, int] = {}
    speakers: dict[str, dict[str, int]] = {name: {} for name in streams}
    for record in log:
        counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1
        for name in streams:
            found = record.get(f"{name}_n_speakers")
            if found is not None:
                speakers[name][str(found)] = speakers[name].get(str(found), 0) + 1

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
        "device": _device_name(device),
        "config_hash": config.config_hash,
        "streams": list(streams),
        "force": force,
        "rows": len(mine),
        "counts": counts,
        "speaker_counts": speakers,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Diarize every run in one shard and print what happened.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when every recording in the shard reached a determinate outcome, 1 when any row is
        ``error`` — the stores are written either way — and 2 when the arguments could not be
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
            batch_size=args.batch_size,
            device=DeviceType(args.device) if args.device else None,
            config=load_triage_config(args.config) if args.config else load_triage_config(),
            log_dir=args.log_dir if args.log_dir is not None else args.manifest.parent,
            force=args.force,
        )
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    print(f"Rows:    {summary['rows']}")
    print(f"Streams: {', '.join(summary['streams'])}")
    print(f"Log:     {summary['log']}")
    for status, number in sorted(summary["counts"].items()):
        print(f"  {status:<12} {number}")
    for stream, histogram in summary["speaker_counts"].items():
        for found, number in sorted(histogram.items()):
            print(f"  {stream}: n_speakers={found:<3} {number}")
    return 1 if summary["counts"].get(ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
