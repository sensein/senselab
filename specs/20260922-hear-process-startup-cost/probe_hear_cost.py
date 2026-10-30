"""Separate HeAR's fixed per-invocation cost from its marginal per-second-of-audio cost.

Writes one JSON line per trial to stdout. Run inside a Slurm job, never on a login node.

Three things are measured, because HeAR's one-shot path has parent-side work YAMNet's does not:
the shipped ``run_hear`` end to end, the worker's own phases, and the parent-side staging
(``ensure_venv``, ``stage_hear_snapshot``) that runs before every invocation.

Usage: python probe_hear_cost.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.health_acoustics.hear import (
    ENCODER_SUBDIR,
    EVENT_DETECTOR_SUBDIRS,
    HEAR_SAMPLING_RATE,
    HEAR_WINDOW_SAMPLES,
    build_worker_payload,
    ensure_venv,
    plan_scan_windows,
    run_hear,
    stage_hear_snapshot,
    venv_python,
)
from senselab.utils.data_structures import DeviceType
from senselab.utils.subprocess_venv import _clean_subprocess_env

DURATIONS = [2.0, 3.0, 5.0, 10.0, 15.0, 21.0, 30.0, 60.0, 120.0, 240.0]
REPEATS = 3
SR = HEAR_SAMPLING_RATE
HOP_SAMPLES = HEAR_WINDOW_SAMPLES  # the config's 2.0 s hop: non-overlapping

# The shipped worker with per-phase timers added and nothing else changed.
WORKER = r"""
import json
import sys
import time

T0 = time.perf_counter()
try:
    import numpy as np
    import soundfile as sf
    import tensorflow as tf

    T_IMPORT = time.perf_counter()
    args = json.loads(sys.stdin.read())
    win = int(args["window_samples"])

    saved_model = tf.saved_model.load(args["saved_model_dir"])
    T_LOAD = time.perf_counter()
    fn = saved_model.signatures["serving_default"]
    spec = fn.structured_input_signature[1]
    input_name = list(spec.keys())[0]
    static_batch = spec[input_name].shape[0]
    batch = 1 if static_batch is not None else max(1, int(args.get("batch_size", 1)))
    T_SIG = time.perf_counter()

    per_job = []
    results = []
    for job in args["jobs"]:
        t_a = time.perf_counter()
        x, sr = sf.read(job["wav"], dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        starts = [int(s) for s in job["starts"]]
        for s in starts:
            if s < 0 or s + win > x.shape[0]:
                raise ValueError("window outside recording")
        windows = np.stack([x[s:s + win] for s in starts]).astype("float32")
        t_read = time.perf_counter()

        outs = []
        first_call_s = None
        for i in range(0, windows.shape[0], batch):
            t_c = time.perf_counter()
            block = windows[i:i + batch]
            out = fn(**{input_name: tf.constant(block, dtype=tf.float32)})
            outs.append(np.asarray(list(out.values())[0]))
            if first_call_s is None:
                first_call_s = time.perf_counter() - t_c
        array = np.concatenate(outs, axis=0).astype("float32")
        t_infer = time.perf_counter()
        np.save(job["out"], array)
        t_post = time.perf_counter()
        per_job.append({
            "seconds": x.shape[0] / sr,
            "n_windows": int(array.shape[0]),
            "read_s": t_read - t_a,
            "infer_s": t_infer - t_read,
            "first_window_s": first_call_s,
            "save_s": t_post - t_infer,
        })
        results.append({"out": job["out"], "shape": [int(v) for v in array.shape]})

    print(json.dumps({
        "results": results,
        "batch": batch,
        "timing": {
            "import_s": T_IMPORT - T0,
            "saved_model_load_s": T_LOAD - T_IMPORT,
            "signature_s": T_SIG - T_LOAD,
            "per_job": per_job,
            "worker_total_s": time.perf_counter() - T0,
        },
    }))
except Exception as exc:
    print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
    sys.exit(1)
"""


def write_noise_wav(path: str, seconds: float) -> None:
    """Deterministic band-limited noise with a slow envelope, float32 as the shipped writer uses."""
    rng = np.random.default_rng(int(seconds * 1000))
    n = int(seconds * SR)
    x = rng.standard_normal(n).astype("float32") * 0.1
    env = (0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n) / SR * 0.7)).astype("float32")
    sf.write(path, (x * env).astype("float32"), SR, subtype="FLOAT")


def run_instrumented(python: str, saved_model_dir: str, jobs: list, env: dict) -> tuple:
    """One subprocess invocation over ``jobs``; returns parent-side wall clock and worker timing."""
    payload = json.dumps(build_worker_payload(saved_model_dir, jobs, 1))
    t0 = time.perf_counter()
    result = subprocess.run(  # noqa: S603
        [python, "-c", WORKER], input=payload, capture_output=True, text=True, timeout=3600, env=env
    )
    wall = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(f"worker failed: {result.stdout[-2000:]} {result.stderr[-2000:]}")
    out = json.loads(result.stdout.strip().splitlines()[-1])
    if "error" in out:
        raise RuntimeError(out["error"])
    return wall, out["timing"]


def main() -> None:
    """Parent-side staging, then a duration sweep, then a batch sweep, then the shipped path."""
    env = _clean_subprocess_env()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env.pop("PYTHONPATH", None)

    # Parent-side per-invocation cost that ``run_hear`` pays before the subprocess starts.
    for rep in range(REPEATS):
        t0 = time.perf_counter()
        venv_dir = ensure_venv("hear", ["tensorflow>=2.16,<3", "numpy", "soundfile"], python_version="3.11")
        t1 = time.perf_counter()
        sha, snapshot = stage_hear_snapshot()
        t2 = time.perf_counter()
        print(
            json.dumps(
                {"trial": "parent_staging", "rep": rep, "ensure_venv_s": t1 - t0, "stage_snapshot_s": t2 - t1}
            ),
            flush=True,
        )
    python = venv_python(venv_dir)
    detector_dir = str(snapshot / EVENT_DETECTOR_SUBDIRS["large"])
    encoder_dir = str(snapshot / ENCODER_SUBDIR) if ENCODER_SUBDIR else str(snapshot)

    with tempfile.TemporaryDirectory(prefix="hperf-") as tmpdir:
        tmp = Path(tmpdir)
        paths = {}
        for d in DURATIONS:
            p = str(tmp / f"noise_{d:g}.wav")
            write_noise_wav(p, d)
            paths[d] = p

        def job(d: float, index: int) -> dict:
            n = int(d * SR)
            return {
                "wav": paths[d],
                "starts": plan_scan_windows(n, HOP_SAMPLES),
                "out": str(tmp / f"out_{index}.npy"),
            }

        # Duration sweep, the detector (what all four PREPROCESS passes run), one audio per call.
        for rep in range(REPEATS):
            for d in DURATIONS:
                wall, timing = run_instrumented(python, detector_dir, [job(d, 0)], env)
                print(
                    json.dumps({"trial": "single", "rep": rep, "seconds": d, "wall_s": wall, "timing": timing}),
                    flush=True,
                )

        # The encoder too: a larger SavedModel, so its load may not split like the detector's.
        for rep in range(REPEATS):
            for d in (2.0, 21.0, 120.0):
                wall, timing = run_instrumented(python, encoder_dir, [job(d, 0)], env)
                print(
                    json.dumps(
                        {"trial": "encoder", "rep": rep, "seconds": d, "wall_s": wall, "timing": timing}
                    ),
                    flush=True,
                )

        # Batch sweep: what a second audio costs inside a process that has already paid the fixed cost.
        for rep in range(REPEATS):
            for n in (1, 2, 4, 8, 16):
                wall, timing = run_instrumented(python, detector_dir, [job(21.0, i) for i in range(n)], env)
                print(
                    json.dumps(
                        {"trial": "batch", "rep": rep, "n": n, "seconds": 21.0, "wall_s": wall, "timing": timing}
                    ),
                    flush=True,
                )

        # The shipped path end to end, so the phase split above is anchored to a real invocation.
        for rep in range(REPEATS):
            for d in (2.0, 7.0, 21.0, 73.0):
                audio = Audio(filepath=paths[min(DURATIONS, key=lambda x: abs(x - d))])
                starts = plan_scan_windows(audio.waveform.shape[-1], HOP_SAMPLES)
                t0 = time.perf_counter()
                run_hear(
                    [audio],
                    [starts],
                    subdir=EVENT_DETECTOR_SUBDIRS["large"],
                    batch_size=1,
                    device=DeviceType.CPU,
                )
                print(
                    json.dumps(
                        {
                            "trial": "shipped_run_hear",
                            "rep": rep,
                            "seconds": audio.waveform.shape[-1] / SR,
                            "n_windows": len(starts),
                            "wall_s": time.perf_counter() - t0,
                        }
                    ),
                    flush=True,
                )

    print(json.dumps({"trial": "done", "sha": sha, "nproc": os.cpu_count()}), flush=True)


if __name__ == "__main__":
    main()
