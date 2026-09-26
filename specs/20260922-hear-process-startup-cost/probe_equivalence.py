"""Old one-shot HeAR against the new resident worker: same arrays, and what each costs.

The old path is not reimplemented here. It is loaded from git at the merge-base commit, as its own
module, so what is compared is the shipped code rather than a description of it.

Usage: python probe_equivalence.py <manifest.jsonl> <baseline-git-ref>
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
import types
from pathlib import Path

import numpy as np

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.health_acoustics import hear
from senselab.utils.data_structures import DeviceType

_MODULE = "src/senselab/audio/tasks/health_acoustics/hear.py"
_HOP = hear.HEAR_WINDOW_SAMPLES  # the config's 2.0 s hop
_DETECTOR = hear.EVENT_DETECTOR_SUBDIRS["large"]


def load_baseline(ref: str) -> types.ModuleType:
    """Import the pre-change module from git under a name of its own.

    Args:
        ref: The git ref to read the module out of.

    Returns:
        The imported module.
    """
    source = subprocess.run(  # noqa: S603
        ["git", "show", f"{ref}:{_MODULE}"], capture_output=True, text=True, check=True
    ).stdout
    path = Path("/tmp") / "hear_baseline.py"  # noqa: S108
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("hear_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["hear_baseline"] = module
    spec.loader.exec_module(module)
    return module


def worker_rss_mib() -> int:
    """Resident set size of the live HeAR worker, in MiB, or 0 when none is running."""
    worker = hear._WORKER  # noqa: SLF001 — the probe is measuring the worker itself
    if worker is None or worker._process is None:  # noqa: SLF001
        return 0
    try:
        status = Path(f"/proc/{worker._process.pid}/status").read_text()  # noqa: SLF001
    except OSError:
        return 0
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) // 1024
    return 0


def compare(old: list, new: list) -> dict:
    """Bitwise agreement between two lists of arrays, and the worst numeric difference if not."""
    identical = len(old) == len(new) and all(
        a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes() for a, b in zip(old, new)
    )
    worst = 0.0
    if not identical and len(old) == len(new):
        for a, b in zip(old, new):
            if a.shape == b.shape:
                worst = max(worst, float(np.max(np.abs(a.astype("float64") - b.astype("float64")))))
    return {"identical": identical, "max_abs_diff": worst}


def prepared_and_plan(wav: str) -> tuple:
    """One recording at HeAR's rate, with the scan plan the graph's passes would use."""
    prepared = hear.prepare_audio_for_hear(Audio(filepath=wav))
    return prepared, hear.plan_scan_windows(prepared.waveform.shape[-1], _HOP)


def main() -> None:
    """Run every recording both ways, on both SavedModels, and report agreement and cost."""
    manifest = Path(sys.argv[1])
    baseline = load_baseline(sys.argv[2])
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]

    mismatches = 0
    old_total = 0.0
    new_total = 0.0
    for row in rows:
        prepared, starts = prepared_and_plan(row["wav"])
        t0 = time.perf_counter()
        old = baseline.run_hear([prepared], [starts], subdir=_DETECTOR, batch_size=1, device=DeviceType.CPU)
        t1 = time.perf_counter()
        new = hear.run_hear([prepared], [starts], subdir=_DETECTOR, batch_size=1, device=DeviceType.CPU)
        t2 = time.perf_counter()
        old_total += t1 - t0
        new_total += t2 - t1
        verdict = compare(old, new)
        mismatches += 0 if verdict["identical"] else 1
        print(
            json.dumps(
                {
                    "model": "detector",
                    "duration_s": row.get("duration_s"),
                    "windows": len(starts),
                    **verdict,
                    "old_s": round(t1 - t0, 3),
                    "new_s": round(t2 - t1, 3),
                    "worker_rss_mib": worker_rss_mib(),
                }
            ),
            flush=True,
        )

    # The encoder: a second SavedModel in the same resident process, which the one-shot path
    # never had to hold alongside the detector.
    for row in rows[:3]:
        prepared, starts = prepared_and_plan(row["wav"])
        t0 = time.perf_counter()
        old = baseline.run_hear(
            [prepared], [starts], subdir=hear.ENCODER_SUBDIR, batch_size=8, device=DeviceType.CPU
        )
        t1 = time.perf_counter()
        new = hear.run_hear([prepared], [starts], subdir=hear.ENCODER_SUBDIR, batch_size=8, device=DeviceType.CPU)
        t2 = time.perf_counter()
        verdict = compare(old, new)
        mismatches += 0 if verdict["identical"] else 1
        print(
            json.dumps(
                {
                    "model": "encoder",
                    "duration_s": row.get("duration_s"),
                    "windows": len(starts),
                    **verdict,
                    "old_s": round(t1 - t0, 3),
                    "new_s": round(t2 - t1, 3),
                    "worker_rss_mib": worker_rss_mib(),
                }
            ),
            flush=True,
        )

    # Alternating the two models through one resident worker: the model cache must not swap
    # one graph's answer for the other's.
    prepared, starts = prepared_and_plan(rows[0]["wav"])
    again = hear.run_hear([prepared], [starts], subdir=_DETECTOR, batch_size=1, device=DeviceType.CPU)
    first = baseline.run_hear([prepared], [starts], subdir=_DETECTOR, batch_size=1, device=DeviceType.CPU)
    alternated = compare(first, again)
    mismatches += 0 if alternated["identical"] else 1

    # The four-invocations-per-recording shape, both ways: the resident worker pays the load once.
    prepared, starts = prepared_and_plan(rows[0]["wav"])
    hear.shutdown_hear_worker()
    t0 = time.perf_counter()
    for _ in range(4):
        baseline.run_hear([prepared], [starts], subdir=_DETECTOR, batch_size=1, device=DeviceType.CPU)
    t1 = time.perf_counter()
    for _ in range(4):
        hear.run_hear([prepared], [starts], subdir=_DETECTOR, batch_size=1, device=DeviceType.CPU)
    t2 = time.perf_counter()

    print(
        json.dumps(
            {
                "summary": True,
                "recordings": len(rows),
                "mismatches": mismatches,
                "detector_after_encoder_identical": alternated["identical"],
                "old_total_s": round(old_total, 2),
                "new_total_s": round(new_total, 2),
                "four_calls_old_s": round(t1 - t0, 2),
                "four_calls_new_cold_s": round(t2 - t1, 2),
                "worker_rss_mib": worker_rss_mib(),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
