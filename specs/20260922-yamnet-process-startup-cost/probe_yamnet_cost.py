"""Separate YAMNet's fixed per-invocation cost from its marginal per-second-of-audio cost.

Writes one JSON line per trial to stdout. Run inside a Slurm job, never on a login node.

Usage: python probe_yamnet_cost.py <path-to-yamnet-venv-python>
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

DURATIONS = [1.0, 2.0, 5.0, 10.0, 15.0, 21.0, 30.0, 60.0, 120.0, 240.0]
REPEATS = 3
SR = 16000

# The worker the shipped code runs, with per-phase timing added and nothing else changed.
WORKER = r"""
import json
import sys
import time

T0 = time.perf_counter()
try:
    import os
    import pathlib
    import shutil

    import numpy as np
    import soundfile as sf
    import tensorflow_hub as hub

    T_IMPORT = time.perf_counter()
    args = json.loads(sys.stdin.read())
    audio_paths = args["audio_paths"]
    top_k = args.get("top_k", 5)

    _HUB_URL = "https://tfhub.dev/google/yamnet/1"
    cache_root = pathlib.Path(
        os.environ.get("SENSELAB_TFHUB_CACHE")
        or (pathlib.Path.home() / ".cache" / "senselab" / "tfhub")
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["TFHUB_CACHE_DIR"] = str(cache_root)
    model = hub.load(_HUB_URL)
    T_LOAD = time.perf_counter()

    import csv
    class_map_path = model.class_map_path().numpy().decode("utf-8")
    with open(class_map_path) as f:
        reader = csv.DictReader(f)
        class_names = [row["display_name"] for row in reader]
    T_CLASSMAP = time.perf_counter()

    per_audio = []
    for audio_path in audio_paths:
        t_a = time.perf_counter()
        data, sr = sf.read(audio_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        t_read = time.perf_counter()
        scores, embeddings, spectrogram = model(data)
        scores_np = scores.numpy()
        t_infer = time.perf_counter()
        windows = []
        for i, frame_scores in enumerate(scores_np):
            top_indices = frame_scores.argsort()[::-1][:top_k]
            windows.append({
                "label_scores": [{class_names[idx]: float(frame_scores[idx])} for idx in top_indices],
            })
        t_post = time.perf_counter()
        per_audio.append({
            "seconds": len(data) / sr,
            "n_windows": len(windows),
            "read_s": t_read - t_a,
            "infer_s": t_infer - t_read,
            "post_s": t_post - t_infer,
        })

    print(json.dumps({
        "timing": {
            "import_s": T_IMPORT - T0,
            "hub_load_s": T_LOAD - T_IMPORT,
            "classmap_s": T_CLASSMAP - T_LOAD,
            "per_audio": per_audio,
            "worker_total_s": time.perf_counter() - T0,
        },
    }))
except Exception as exc:
    print(json.dumps({"error": {"type": type(exc).__name__, "message": str(exc)}}))
    sys.exit(1)
"""


def write_noise_wav(path: str, seconds: float) -> None:
    """Deterministic band-limited noise with a slow envelope, at the probe's sample rate."""
    import numpy as np

    rng = np.random.default_rng(int(seconds * 1000))
    n = int(seconds * SR)
    x = rng.standard_normal(n).astype("float32") * 0.1
    env = (0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n) / SR * 0.7)).astype("float32")
    x = (x * env).astype("float32")
    pcm = (np.clip(x, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(SR)
        fh.writeframes(pcm.tobytes())


def run_worker(python: str, paths: list, env: dict) -> tuple:
    """One subprocess invocation over ``paths``; returns parent-side wall clock and worker timing."""
    payload = json.dumps({"audio_paths": paths, "top_k": 5})
    t0 = time.perf_counter()
    result = subprocess.run(
        [python, "-c", WORKER],
        input=payload,
        capture_output=True,
        text=True,
        timeout=1800,
        env=env,
    )
    wall = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(f"worker failed: {result.stdout[-2000:]} {result.stderr[-2000:]}")
    out = json.loads(result.stdout.strip().splitlines()[-1])
    if "error" in out:
        raise RuntimeError(out["error"])
    return wall, out["timing"]


def main() -> None:
    """Sweep duration at one audio per invocation, then batch size at one duration."""
    python = sys.argv[1]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="yperf-") as tmpdir:
        tmp = Path(tmpdir)
        paths = {}
        for d in DURATIONS:
            p = str(tmp / f"noise_{d:g}.wav")
            write_noise_wav(p, d)
            paths[d] = p

        for rep in range(REPEATS):
            for d in DURATIONS:
                wall, timing = run_worker(python, [paths[d]], env)
                print(
                    json.dumps({"trial": "single", "rep": rep, "seconds": d, "wall_s": wall, "timing": timing}),
                    flush=True,
                )

        for rep in range(REPEATS):
            for n in (1, 2, 4, 8, 16):
                wall, timing = run_worker(python, [paths[21.0]] * n, env)
                print(
                    json.dumps({"trial": "batch", "rep": rep, "n": n, "seconds": 21.0, "wall_s": wall, "timing": timing}),
                    flush=True,
                )


if __name__ == "__main__":
    main()
