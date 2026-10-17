"""Throughput of each component on a long recording: a synthetic dialogue tiled to MINUTES.

Times masking, MossFormer2 separation (and pyannote when its gated terms are accepted), and the
three MaAI runs (VAP + one backchannel run per listener), reporting x real time for each and the
1-minute load average beside every figure. Timings are upper bounds unless the machine is idle.

Usage (one process, three threads)::

    OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 uv run python bench_speed.py DATA_DIR/dialogue_seed0 OUT.json \
        [--minutes 10] [--methods masking mossformer2]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.turn_taking_projection import project_voice_activity, separate_dyad
from senselab.utils.data_structures import ScriptLine
from senselab.utils.venv_worker import shutdown_venv_workers


def tiled(directory: Path, minutes: float) -> tuple:
    """The dialogue's mix and IPUs repeated end to end until ``minutes`` long."""
    truth = json.loads((directory / "truth.json").read_text())
    mix = Audio(filepath=str(directory / "mix.wav")).waveform[0].numpy()
    reps = int(np.ceil(minutes * 60 * 16000 / mix.size))
    period = mix.size / 16000
    segments = [
        ScriptLine(speaker=r["speaker"], start=r["start"] + k * period, end=r["end"] + k * period)
        for k in range(reps)
        for r in truth["ipus"]
    ]
    n = int(minutes * 60 * 16000)
    audio = Audio(waveform=torch.from_numpy(np.tile(mix, reps)[:n]).unsqueeze(0), sampling_rate=16000)
    return audio, [s for s in segments if s.end <= n / 16000]


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dialogue")
    parser.add_argument("out")
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--methods", nargs="+", default=["masking", "mossformer2"])
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "3")))
    audio, segments = tiled(Path(args.dialogue), args.minutes)
    seconds = audio.waveform.shape[-1] / 16000
    report: dict = {"audio_s": seconds, "threads": torch.get_num_threads(), "components": {}}
    pair = None
    for method in args.methods:
        began = time.monotonic()
        try:
            pair = separate_dyad(audio, segments, method=method, speakers=("A", "B"))
        except Exception as exc:  # noqa: BLE001 -- a gated model reports why instead of stopping the bench
            report["components"][method] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
            continue
        wall = time.monotonic() - began
        shutdown_venv_workers()
        report["components"][method] = {"wall_s": wall, "x_real_time": seconds / wall, "load_1m": os.getloadavg()[0]}
        print(method, report["components"][method], flush=True)
    assert pair is not None
    for workers in (1, 3):
        vap = project_voice_activity(pair, speakers=("A", "B"), num_workers=workers)
        report["components"][f"maai_workers{workers}"] = {**vap.timing, "load_1m": os.getloadavg()[0]}
        print(workers, vap.timing, flush=True)
    Path(args.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
