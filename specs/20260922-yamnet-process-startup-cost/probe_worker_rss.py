"""What a resident YAMNet worker holds between classifications.

Spawns the worker inside the yamnet venv, classifies, and reads VmRSS/VmHWM from /proc while it is
idle and loaded — which is what a second process has to live beside once the worker is resident.

Usage: python probe_worker_rss.py <path-to-yamnet-venv-python>
"""

import json
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

WORKER = r"""
import json, os, sys, time
MARKER = "@@SENSELAB_YAMNET@@"
_replies = sys.stdout
sys.stdout = sys.stderr

def emit(p):
    _replies.write(MARKER + json.dumps(p) + "\n"); _replies.flush()

import csv, pathlib
import numpy as np
import soundfile as sf
import tensorflow_hub as hub

cache_root = pathlib.Path(os.environ.get("SENSELAB_TFHUB_CACHE")
                          or (pathlib.Path.home() / ".cache" / "senselab" / "tfhub"))
cache_root.mkdir(parents=True, exist_ok=True)
os.environ["TFHUB_CACHE_DIR"] = str(cache_root)
model = hub.load("https://tfhub.dev/google/yamnet/1")
cmp = model.class_map_path().numpy().decode("utf-8")
with open(cmp) as f:
    class_names = [r["display_name"] for r in csv.DictReader(f)]
emit({"ready": True})

while True:
    line = sys.stdin.readline()
    if not line:
        break
    req = json.loads(line)
    if req.get("stop"):
        break
    out = []
    for p in req["audio_paths"]:
        data, sr = sf.read(p, dtype="float32")
        scores, _, _ = model(data)
        out.append(int(scores.numpy().shape[0]))
    emit({"n_windows": out})
"""


def rss(pid: int) -> dict:
    """VmRSS and VmHWM of one process, in MiB."""
    out = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(("VmRSS:", "VmHWM:")):
            out[line.split(":")[0]] = int(line.split()[1]) // 1024
    return out


def main() -> None:
    """Start the worker, classify at three durations, and report what it holds."""
    python = sys.argv[1]
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for seconds in (5.0, 30.0, 130.0):
            p = str(Path(tmpdir) / f"n{seconds:g}.wav")
            with wave.open(p, "wb") as fh:
                fh.setnchannels(1)
                fh.setsampwidth(2)
                fh.setframerate(16000)
                fh.writeframes(b"\x00\x01" * int(seconds * 16000))
            paths.append(p)

        proc = subprocess.Popen(  # noqa: S603
            [python, "-c", WORKER],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None and proc.stdin is not None
        for line in proc.stdout:
            if line.startswith("@@SENSELAB_YAMNET@@"):
                break
        print(json.dumps({"phase": "loaded_idle", **rss(proc.pid)}))

        for p in paths:
            proc.stdin.write(json.dumps({"audio_paths": [p]}) + "\n")
            proc.stdin.flush()
            for line in proc.stdout:
                if line.startswith("@@SENSELAB_YAMNET@@"):
                    break
            time.sleep(0.2)
            print(json.dumps({"phase": f"after_{Path(p).stem}", **rss(proc.pid)}))

        print(json.dumps({"phase": "steady_state", **rss(proc.pid)}))
        proc.stdin.write(json.dumps({"stop": True}) + "\n")
        proc.stdin.flush()
        proc.wait(timeout=30)


if __name__ == "__main__":
    main()
