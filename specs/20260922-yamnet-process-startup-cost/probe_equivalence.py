"""Old one-shot YAMNet against the new resident worker: same scores, and what each costs.

The old path is not reimplemented here. It is loaded from git at the commit before the change, as
its own module, so what is compared is the shipped code rather than a description of it.

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

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.classification.yamnet import YAMNetClassifier, shutdown_yamnet_worker

_MODULE = "src/senselab/audio/tasks/classification/yamnet.py"


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
    path = Path("/tmp") / "yamnet_baseline.py"  # noqa: S108
    path.write_text(source)
    spec = importlib.util.spec_from_file_location("yamnet_baseline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["yamnet_baseline"] = module
    spec.loader.exec_module(module)
    return module


def worker_rss_mib() -> int:
    """Resident set size of the live YAMNet worker, in MiB, or 0 when none is running."""
    from senselab.audio.tasks.classification import yamnet as mod

    worker = mod._WORKER  # noqa: SLF001 — the probe is measuring the worker itself
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


def main() -> None:
    """Classify every recording both ways and report agreement and cost."""
    manifest = Path(sys.argv[1])
    baseline_ref = sys.argv[2]
    baseline = load_baseline(baseline_ref)
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]

    mismatches = 0
    old_total = 0.0
    new_total = 0.0
    for row in rows:
        audio = Audio(filepath=row["wav"])
        t0 = time.perf_counter()
        old = baseline.YAMNetClassifier.classify_with_yamnet([audio], top_k=521)
        t1 = time.perf_counter()
        new = YAMNetClassifier.classify_with_yamnet([audio], top_k=521)
        t2 = time.perf_counter()
        old_total += t1 - t0
        new_total += t2 - t1
        same = json.dumps(old, sort_keys=True) == json.dumps(new, sort_keys=True)
        mismatches += 0 if same else 1
        print(
            json.dumps(
                {
                    "duration_s": row.get("duration_s"),
                    "identical": same,
                    "n_windows": len(new[0]) if new else 0,
                    "old_s": round(t1 - t0, 3),
                    "new_s": round(t2 - t1, 3),
                    "worker_rss_mib": worker_rss_mib(),
                }
            ),
            flush=True,
        )

    # The four-invocations-per-recording shape, both ways: the resident worker pays the load once.
    audios = [Audio(filepath=rows[0]["wav"])]
    shutdown_yamnet_worker()
    t0 = time.perf_counter()
    for _ in range(4):
        baseline.YAMNetClassifier.classify_with_yamnet(audios, top_k=521)
    t1 = time.perf_counter()
    for _ in range(4):
        YAMNetClassifier.classify_with_yamnet(audios, top_k=521)
    t2 = time.perf_counter()

    print(
        json.dumps(
            {
                "summary": True,
                "recordings": len(rows),
                "mismatches": mismatches,
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
