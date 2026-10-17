"""Time one :func:`extract_ppgs_from_audios` call on a recording tiled to a given length.

Each configuration runs in its own process: the worker's peak resident memory is read from
``RUSAGE_CHILDREN``, which keeps the largest child this process has waited on.

One configuration::

    uv run python scripts/benchmark_ppgs.py tutorial_audio_files/english_conversation_higgs_audio_v2.wav
        --minutes 5 --device cpu --window-s 60 --threads 3

The spec's whole table, serially, one process per row::

    uv run python scripts/benchmark_ppgs.py tutorial_audio_files/english_conversation_higgs_audio_v2.wav --matrix

Prints one JSON line per configuration: wall seconds, real-time factor, worker peak RSS, frame
count, thread cap and the 1-minute load average before and after, since a loaded host inflates time.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction.ppg import (
    PPGS_SAMPLE_RATE,
    extract_ppgs_from_audios,
    require_posteriorgram,
)
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.utils.data_structures import DeviceType


def _tiled(path: Path, minutes: float) -> Audio:
    """The recording, downmixed, at the ppgs rate, repeated to ``minutes`` long."""
    audio = Audio(filepath=str(path))
    mono = Audio(waveform=audio.waveform.mean(dim=0, keepdim=True), sampling_rate=audio.sampling_rate)
    [mono] = resample_audios([mono], PPGS_SAMPLE_RATE)
    samples = int(minutes * 60 * PPGS_SAMPLE_RATE)
    repeats = samples // mono.waveform.shape[-1] + 1
    return Audio(waveform=mono.waveform.repeat(1, repeats)[:, :samples], sampling_rate=PPGS_SAMPLE_RATE)


MATRIX = [
    ("5", "cpu", "60"),
    ("5", "mps", "60"),
    ("30", "cpu", "60"),
    ("30", "mps", "60"),
    ("30", "cpu", "none"),
    ("90", "cpu", "60"),
    ("90", "mps", "60"),
]
"""``(minutes, device, window_s)`` rows of the spec's throughput table."""


def _run_matrix(audio: Path, threads: int) -> int:
    for minutes, device, window in MATRIX:
        command = [sys.executable, __file__, str(audio), "--minutes", minutes, "--device", device]
        command += ["--window-s", window, "--threads", str(threads)]
        result = subprocess.run(command, capture_output=True, text=True)
        lines = result.stdout.strip().splitlines()
        print(lines[-1] if lines else json.dumps({"row": [minutes, device, window], "error": result.stderr[-500:]}))
        sys.stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one timed extraction and print its measurements.

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("audio", type=Path)
    parser.add_argument("--minutes", type=float, default=5.0)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--window-s", default="60", help="seconds per model call, or 'none'")
    parser.add_argument("--save", type=Path, default=None, help="write the (phonemes, frames) array here")
    parser.add_argument("--threads", type=int, default=3, help="OMP/torch thread cap for parent and worker")
    parser.add_argument("--matrix", action="store_true", help="run every MATRIX row serially, one process each")
    args = parser.parse_args(argv)
    if args.matrix:
        return _run_matrix(args.audio, args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    load_before = os.getloadavg()[0]

    audio = _tiled(args.audio, args.minutes)
    window_s = None if args.window_s.lower() == "none" else float(args.window_s)
    started = time.perf_counter()
    [result] = extract_ppgs_from_audios([audio], device=DeviceType(args.device), window_s=window_s)
    wall_s = time.perf_counter() - started
    posteriorgram: torch.Tensor = require_posteriorgram(result)
    if args.save is not None:
        np.save(args.save, posteriorgram[0].numpy())
    peak_bytes = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    if sys.platform != "darwin":
        peak_bytes *= 1024
    print(
        json.dumps(
            {
                "minutes": args.minutes,
                "device": args.device,
                "window_s": window_s,
                "wall_s": round(wall_s, 2),
                "real_time_factor": round(wall_s / (args.minutes * 60), 5),
                "worker_peak_rss_gb": round(peak_bytes / 1e9, 2),
                "frames": int(posteriorgram.shape[-1]),
                "threads": args.threads,
                "load_1min_before": round(load_before, 1),
                "load_1min_after": round(os.getloadavg()[0], 1),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
