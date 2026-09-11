#!/usr/bin/env python3
r"""Extract phonetic posteriorgrams and Praat features over a triage run's enhanced tracks.

    uv run python scripts/extract_ppg_praat.py MANIFEST OUT --slice-index I --slice-count N \
        [--batch-size 500] [--device cpu]

``MANIFEST`` is a JSONL, one object per line, each carrying ``stem``, ``enhanced`` (the absolute
path of that recording's ``run/streams/enhanced.flac``), ``family``, ``duration_s`` and ``lexical``.
``--slice-index`` / ``--slice-count`` shard it for a Slurm array: task *i* of *n* takes
``rows[i::n]``, so every task sees the same stride of long and short recordings.

The ppgs venv must already exist. This driver refuses rather than building one, because a cold
build outlasts the venv lock's patience and every task racing it waits:

    uv run python -c \
        "from senselab.audio.tasks.features_extraction import ensure_ppgs_venv; ensure_ppgs_venv()"

Layout under ``OUT``, mirroring the BIDS tree the run already uses, the entity path taken from the
stem by :func:`senselab.audio.workflows.triage.run.entity_subdir`:

    <OUT>/sub-<label>/ses-<label>/<stem>_ppg.npz         the posteriorgram and its phoneme order
    <OUT>/sub-<label>/ses-<label>/<stem>_features.json   the outcome row, with the Praat features
    <OUT>/slices/slice-<i>-of-<n>.jsonl                  every row this task handled
    <OUT>/slices/slice-<i>-of-<n>.summary.json           the task's counts and its provenance

``<stem>_features.json`` is written last and is the completion marker: a recording that has one is
skipped on a re-run. Delete it to recompute that recording.

The reasoning — the batch-size benchmark, the venv-lock hazard, the storage arithmetic — is in
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
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction import (
    PHONEME_LABELS,
    PPGS_SAMPLE_RATE,
    extract_ppgs_from_audios,
    ppgs_venv_is_provisioned,
    to_frame_major_posteriorgram,
)
from senselab.audio.tasks.features_extraction.praat_parselmouth import (
    extract_praat_parselmouth_features_from_audios,
)
from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios
from senselab.audio.workflows.triage.run import entity_subdir
from senselab.utils.data_structures import DeviceType

DEFAULT_BATCH_SIZE = 500
"""Recordings per ``extract_ppgs_from_audios`` call. Its derivation is in the spec."""

PPG_SUFFIX = "_ppg.npz"
ROW_SUFFIX = "_features.json"
SLICES_SUBDIR = "slices"

_OK = "ok"
_NAN = "nan"
_ERROR = "error"
_PARTIAL = "partial"
_SKIPPED = "skipped"


def build_parser() -> argparse.ArgumentParser:
    """The CLI: a manifest, an output directory, and which shard of the manifest this task takes.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("manifest", type=Path, help="Manifest JSONL: one object per recording")
    parser.add_argument("out", type=Path, help="Directory the BIDS-shaped output tree is created in")
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
    return parser


def read_manifest(path: Path) -> list[dict[str, Any]]:
    """Read the manifest into rows.

    Args:
        path: The JSONL file.

    Returns:
        One dict per non-blank line, in file order.

    Raises:
        ValueError: If a line is not a JSON object, or does not carry ``stem`` and ``enhanced``.
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            missing = [key for key in ("stem", "enhanced") if not payload.get(key)]
            if missing:
                raise ValueError(f"{path}:{number} carries no {' or '.join(missing)}")
            rows.append(payload)
    return rows


def take_slice(rows: Sequence[dict[str, Any]], index: int, count: int) -> list[dict[str, Any]]:
    """The stride of the manifest this array task owns.

    Args:
        rows: Every manifest row.
        index: This task's 0-based index.
        count: How many tasks the array has.

    Returns:
        ``rows[index::count]``.

    Raises:
        ValueError: If ``count`` is not positive, or ``index`` is outside it.
    """
    if count < 1:
        raise ValueError(f"--slice-count must be at least 1, got {count}")
    if not 0 <= index < count:
        raise ValueError(f"--slice-index must be in [0, {count}), got {index}")
    return list(rows[index::count])


def row_paths(out_dir: Path, stem: str) -> tuple[Path, Path]:
    """Where one recording's two products go.

    Args:
        out_dir: The output tree's root.
        stem: The recording's file stem.

    Returns:
        ``(posteriorgram_path, row_path)``, both under the stem's entity subdirectory.
    """
    directory = out_dir / entity_subdir(stem)
    return directory / f"{stem}{PPG_SUFFIX}", directory / f"{stem}{ROW_SUFFIX}"


def batches(rows: Sequence[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    """Split rows into fixed-size batches, the last one as short as it needs to be.

    Args:
        rows: The rows to split.
        size: Rows per batch.

    Yields:
        Each batch, in order.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size < 1:
        raise ValueError(f"--batch-size must be at least 1, got {size}")
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def load_for_ppgs(path: Path) -> Audio:
    """Read one recording as ppgs needs it: mono, at :data:`PPGS_SAMPLE_RATE`.

    Args:
        path: The recording.

    Returns:
        The conditioned audio.
    """
    audio = Audio(filepath=str(path))
    if audio.waveform.shape[0] != 1:
        audio = downmix_audios_to_mono([audio])[0]
    if audio.sampling_rate != PPGS_SAMPLE_RATE:
        audio = resample_audios([audio], PPGS_SAMPLE_RATE)[0]
    return audio


def write_posteriorgram(path: Path, audio: Audio, posteriorgram: torch.Tensor) -> dict[str, Any]:
    """Persist one posteriorgram in float16, with the phoneme order beside it.

    Args:
        path: Where the npz goes.
        audio: The recording it was measured on, for the frame rate.
        posteriorgram: The tensor ppgs returned, in any of its layouts.

    Returns:
        The outcome row's ``ppg`` block.
    """
    frame_major = to_frame_major_posteriorgram(posteriorgram)
    frames, phonemes = int(frame_major.shape[0]), int(frame_major.shape[1])
    duration_s = audio.waveform.shape[1] / audio.sampling_rate
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as handle:
        np.savez(
            handle,
            posteriorgram=frame_major.numpy().astype(np.float16),
            phonemes=np.asarray(PHONEME_LABELS, dtype=np.str_),
            seconds_per_frame=np.float64(duration_s / frames if frames else np.nan),
            duration_s=np.float64(duration_s),
            sampling_rate=np.int64(audio.sampling_rate),
        )
    partial.replace(path)
    return {"status": _OK, "path": path.name, "frames": frames, "phonemes": phonemes, "error": None}


def write_row(path: Path, row: dict[str, Any]) -> None:
    """Persist one outcome row, last of the two products and the completion marker.

    Args:
        path: Where the JSON goes.
        row: The row.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def _posteriorgrams(audios: list[Audio], device: DeviceType | None) -> tuple[list[torch.Tensor] | None, str | None]:
    """Run one batch through ppgs, turning a whole-batch failure into a message rather than a raise.

    Args:
        audios: The batch's conditioned audios.
        device: Where ppgs runs.

    Returns:
        ``(posteriorgrams, None)`` when the call returned one tensor per audio, else
        ``(None, message)``.
    """
    try:
        out = extract_ppgs_from_audios(audios, device=device)
    except Exception as error:  # noqa: BLE001 — every recording in the batch records it and survives
        return None, f"{type(error).__name__}: {error}"
    if len(out) != len(audios):
        return None, f"ppgs returned {len(out)} posteriorgrams for {len(audios)} audios"
    return out, None


def _praat(audio: Audio) -> dict[str, Any]:
    """Run one recording through Praat, turning a failure into a message rather than a raise.

    Args:
        audio: The conditioned audio.

    Returns:
        The outcome row's ``praat`` block, carrying ``features`` when the call returned them.
    """
    try:
        features = extract_praat_parselmouth_features_from_audios([audio])[0]
    except Exception as error:  # noqa: BLE001 — a recording Praat refuses is an outcome, not a stop
        return {"status": _ERROR, "features": None, "error": f"{type(error).__name__}: {error}"}
    return {"status": _OK, "features": features, "error": None}


def _outcome(ppg: dict[str, Any], praat: dict[str, Any]) -> str:
    """The row's top-level status, folded from its two products.

    Args:
        ppg: The ``ppg`` block.
        praat: The ``praat`` block.

    Returns:
        ``ok`` when both landed, ``error`` when neither did, ``partial`` otherwise.
    """
    landed = [ppg["status"] == _OK, praat["status"] == _OK]
    if all(landed):
        return _OK
    return _PARTIAL if any(landed) else _ERROR


def process_batch(
    rows: Sequence[dict[str, Any]],
    out_dir: Path,
    device: DeviceType | None,
) -> list[dict[str, Any]]:
    """Extract both feature sets over one batch and write every recording's products.

    A recording whose audio will not load, whose posteriorgram comes back NaN, or whose Praat call
    refuses still gets a row saying so; the rest of the batch is unaffected.

    Args:
        rows: The batch's manifest rows.
        out_dir: The output tree's root.
        device: Where ppgs runs.

    Returns:
        One outcome row per input row, in order.
    """
    loaded: list[tuple[int, Audio]] = []
    outcomes: dict[int, dict[str, Any]] = {}
    for position, row in enumerate(rows):
        try:
            loaded.append((position, load_for_ppgs(Path(row["enhanced"]))))
        except Exception as error:  # noqa: BLE001 — an unreadable recording is an outcome
            message = f"{type(error).__name__}: {error}"
            outcomes[position] = {
                "status": _ERROR,
                "ppg": {"status": _ERROR, "path": None, "frames": None, "phonemes": None, "error": message},
                "praat": {"status": _ERROR, "features": None, "error": message},
            }

    audios = [audio for _, audio in loaded]
    posteriorgrams, batch_error = _posteriorgrams(audios, device) if audios else ([], None)

    for offset, (position, audio) in enumerate(loaded):
        ppg_path, _ = row_paths(out_dir, str(rows[position]["stem"]))
        if posteriorgrams is None:
            ppg = {"status": _ERROR, "path": None, "frames": None, "phonemes": None, "error": batch_error}
        else:
            tensor = posteriorgrams[offset]
            if tensor.numel() == 0 or bool(torch.isnan(tensor).any()):
                ppg = {
                    "status": _NAN,
                    "path": None,
                    "frames": None,
                    "phonemes": None,
                    "error": "ppgs returned a NaN posteriorgram for this recording",
                }
            else:
                ppg = write_posteriorgram(ppg_path, audio, tensor)
        praat = _praat(audio)
        outcomes[position] = {"status": _outcome(ppg, praat), "ppg": ppg, "praat": praat}

    written: list[dict[str, Any]] = []
    for position, row in enumerate(rows):
        _, row_path = row_paths(out_dir, str(row["stem"]))
        record = {**row, **outcomes[position]}
        write_row(row_path, record)
        written.append(record)
    return written


def run_slice(
    manifest: Path,
    out_dir: Path,
    *,
    slice_index: int,
    slice_count: int,
    batch_size: int,
    device: DeviceType | None,
) -> dict[str, Any]:
    """Extract both feature sets over one array task's stride of the manifest.

    Args:
        manifest: The manifest JSONL.
        out_dir: The output tree's root.
        slice_index: This task's 0-based index.
        slice_count: How many tasks the array has.
        batch_size: Recordings per ppgs call.
        device: Where ppgs runs.

    Returns:
        The task's summary: its counts, its provenance, and where its log went.
    """
    started = time.time()
    mine = take_slice(read_manifest(manifest), slice_index, slice_count)

    pending: list[dict[str, Any]] = []
    log: list[dict[str, Any]] = []
    for row in mine:
        _, row_path = row_paths(out_dir, str(row["stem"]))
        if row_path.exists():
            log.append({"stem": row["stem"], "status": _SKIPPED})
        else:
            pending.append(row)

    print(
        f"[slice {slice_index}/{slice_count}] {len(mine)} rows, {len(mine) - len(pending)} already done, "
        f"{len(pending)} to extract in batches of {batch_size}",
        flush=True,
    )

    for number, batch in enumerate(batches(pending, batch_size), start=1):
        batch_started = time.time()
        log.extend(process_batch(batch, out_dir, device))
        print(
            f"[slice {slice_index}/{slice_count}] batch {number}: {len(batch)} recordings in "
            f"{time.time() - batch_started:.0f}s",
            flush=True,
        )

    counts: dict[str, int] = {}
    for record in log:
        counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1

    slices_dir = out_dir / SLICES_SUBDIR
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
        "rows": len(mine),
        "counts": counts,
        "elapsed_s": time.time() - started,
        "ppgs_sample_rate": PPGS_SAMPLE_RATE,
        "phonemes": list(PHONEME_LABELS),
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Extract both feature sets over one shard and print what happened.

    Args:
        argv: The command line, or None to read ``sys.argv``.

    Returns:
        0 when every recording in the shard reached a determinate outcome, 1 when any row is
        ``error`` — the rows are written either way — and 2 when the arguments could not be
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
            args.out,
            slice_index=args.slice_index,
            slice_count=args.slice_count,
            batch_size=args.batch_size,
            device=DeviceType(args.device) if args.device else None,
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
