"""PREPROCESS step timing with the interval attributed to the step that does the work.

Every ``_step``/``_activity`` registration happens *before* the step's own work, so an interval
stamped onto the activity being registered names the step that just ended, not the one starting.
This probe records registration instants and attributes ``[t_i, t_{i+1}]`` to step *i*.

It also counts every YAMNet subprocess invocation, with its batch size and wall clock, by wrapping
``subprocess.run`` on the yamnet module.

Usage: python probe_preprocess_steps.py <manifest.jsonl> <out_dir> [n_recordings]
The manifest is the corpus manifest: one JSON object per line with a ``wav`` path.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import senselab.audio.tasks.classification.yamnet as yamnet_mod
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.run import run_triage
from senselab.utils.prov_store import ProvStore

REGISTRATIONS: list[tuple[str, str, float]] = []
YAMNET_CALLS: list[dict] = []

_real_activity = ProvStore.activity
_real_run = subprocess.run


def _timed_activity(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
    node = kwargs.get("node") or (args[0] if args else "?")
    step = kwargs.get("step") or (args[1] if len(args) > 1 else "?")
    REGISTRATIONS.append((str(node), str(step), time.perf_counter()))
    return _real_activity(self, *args, **kwargs)


def _counted_run(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
    payload = kwargs.get("input") or ""
    n_audio = None
    try:
        n_audio = len(json.loads(payload)["audio_paths"])
    except Exception:  # noqa: BLE001 — a non-YAMNet payload is simply not counted
        pass
    t0 = time.perf_counter()
    try:
        return _real_run(*args, **kwargs)
    finally:
        YAMNET_CALLS.append(
            {
                "n_audio": n_audio,
                "wall_s": time.perf_counter() - t0,
                "step_index": len(REGISTRATIONS),
                "after_step": REGISTRATIONS[-1][1] if REGISTRATIONS else None,
            }
        )


ProvStore.activity = _timed_activity  # type: ignore[method-assign]
yamnet_mod.subprocess.run = _counted_run  # type: ignore[attr-defined]


def main() -> None:
    """Run the whole graph over N recordings, emitting one JSON record per recording."""
    manifest = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_triage_config()

    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()][:limit]
    for row in rows:
        source = Path(row["wav"])
        REGISTRATIONS.clear()
        YAMNET_CALLS.clear()
        t_start = time.perf_counter()
        try:
            run_triage(source, out_dir, config)
            failure = None
        except Exception as exc:  # noqa: BLE001 — a failed recording is still a timing record
            failure = f"{type(exc).__name__}: {exc}"
        t_end = time.perf_counter()

        steps = []
        for i, (node, step, t) in enumerate(REGISTRATIONS):
            end = REGISTRATIONS[i + 1][2] if i + 1 < len(REGISTRATIONS) else t_end
            steps.append({"node": node, "step": step, "seconds": end - t})
        print(
            json.dumps(
                {
                    "duration_s": row.get("duration_s"),
                    "total_s": t_end - t_start,
                    "failure": failure,
                    "yamnet_calls": list(YAMNET_CALLS),
                    "yamnet_invocations": len(YAMNET_CALLS),
                    "yamnet_total_s": sum(call["wall_s"] for call in YAMNET_CALLS),
                    "steps": steps,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
