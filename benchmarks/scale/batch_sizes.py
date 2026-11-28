"""Per-item time of one backend's worker at explicit request sizes and under pydra's batch probe.

`uv run python benchmarks/scale/batch_sizes.py --profile PROFILE --items N --sizes 1,2,4,... --seconds 1,2,3,4
[--task speaker_embeddings --backend speechbrain] [--out FILE]`

The worker is started as the job engine starts it (`RealWorkers`: its env, its catalogue model, `cuda` when the host
has a GPU). The records are N synthetic tones whose durations cycle through `--seconds`. One warm-up request loads the
model. Explicit: the N records are sent in requests of each size, and each record's output is compared with its
size-1 output. Probe: the N records are sent call by call at the size pydra's `probe_batch` asks for, from the times
measured so far, under the profile's pool settings (`probe_samples`, `probe_gain`) with the largest size as its bound.
One JSON document is printed (and written to FILE): per size the seconds per item and per audio second, whether every
output equals its size-1 output, and the probe's calls and chosen size. The exit status is 0 when every output is
equal.
"""

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy
from pydra.workers.pool.calibration import TimeRecord
from pydra.workers.pool.fit import probe_batch
from senselab_core.kernel.signals import Audio
from senselab_core.kernel.wire import wire_form
from senselab_core.runtime.config import load_yaml
from senselab_core.runtime.execution.warm import settings
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.testing.workers import RealWorkers

RATE = 16000
"""The sampling rate of the tones: the rate the speech backends take."""


def _records(items: int, seconds: Sequence[float]) -> list[tuple[dict[str, Any], float]]:
    """Return `items` tone records, each with its duration in seconds, the durations cycling through `seconds`."""
    found = []
    for i in range(items):
        s = seconds[i % len(seconds)]
        t = numpy.arange(round(s * RATE)) / RATE
        x = (0.1 * numpy.sin(2 * numpy.pi * (110 + 7 * i) * t)).astype(numpy.float32)[None, :]
        found.append(({"id": f"r{i}", "inputs": {"audio": wire_form(Audio.from_array(x, RATE))}}, s))
    return found


def _send(worker: Any, chunk: Sequence[tuple[dict[str, Any], float]]) -> tuple[float, list[dict[str, Any]]]:  # noqa: ANN401
    """Send one request; return its wall seconds and its results."""
    started = time.perf_counter()
    results = worker.run([item for item, _ in chunk], {"backend_params": {}})
    return time.perf_counter() - started, results


def _json(result: dict[str, Any]) -> str:
    return json.dumps(result.get("outputs", result.get("error")), sort_keys=True, default=str)


def explicit(worker: Any, records: Sequence[tuple[dict[str, Any], float]], sizes: Sequence[int]) -> dict[str, Any]:  # noqa: ANN401
    """Run the records at each size; per size the time per item and audio second, and equality with size 1."""
    alone: dict[str, str] = {}
    found: dict[str, Any] = {}
    for size in sorted(set(sizes) | {1}):
        walls, outputs = [], {}
        for i in range(0, len(records), size):
            chunk = records[i : i + size]
            wall, results = _send(worker, chunk)
            walls.append((wall, len(chunk), sum(s for _, s in chunk)))
            outputs.update({item["id"]: _json(r) for (item, _), r in zip(chunk, results, strict=True)})
        if size == 1:
            alone = outputs
        found[str(size)] = {
            "calls": len(walls),
            "per_item_s": sum(w for w, _, _ in walls) / len(records),
            "per_audio_s": sum(w for w, _, _ in walls) / sum(a for _, _, a in walls),
            "equal_to_alone": outputs == alone,
        }
    return found


def probed(
    worker: Any,  # noqa: ANN401
    records: Sequence[tuple[dict[str, Any], float]],
    bound: int,
    samples: int,
    gain: float,
) -> dict[str, Any]:
    """Run the records call by call at the size `probe_batch` asks for; its calls and the size it settles on."""
    per_unit: dict[int, list[float]] = {}
    calls: list[int] = []
    i = 0
    while i < len(records):
        times = [TimeRecord(b, statistics.median(v), len(v)) for b, v in per_unit.items()]
        size = probe_batch(times, bound, samples, gain)
        chunk = records[i : i + size]
        wall, _ = _send(worker, chunk)
        if len(chunk) == size:
            per_unit.setdefault(size, []).append(wall / sum(s for _, s in chunk))
        calls.append(len(chunk))
        i += len(chunk)
    times = [TimeRecord(b, statistics.median(v), len(v)) for b, v in per_unit.items()]
    return {
        "calls": calls,
        "chosen": probe_batch(times, bound, samples, gain),
        "per_audio_s": {str(b): statistics.median(v) for b, v in sorted(per_unit.items())},
        "samples": samples,
        "gain": gain,
        "bound": bound,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Start the worker, run the explicit sizes and the probe, print the document."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--profile", type=Path, required=True, help="a whole execution profile file")
    parser.add_argument("--items", type=int, required=True, help="records per size")
    parser.add_argument("--sizes", required=True, help="comma-separated request sizes; the largest bounds the probe")
    parser.add_argument("--seconds", required=True, help="comma-separated record durations, cycled")
    parser.add_argument("--task", default="speaker_embeddings", help="the task")
    parser.add_argument("--backend", default="speechbrain", help="the task's backend")
    parser.add_argument("--out", type=Path, default=None, help="where to write the document")
    args = parser.parse_args(argv)
    sizes = [int(s) for s in args.sizes.split(",")]
    if min(sizes) < 1 or args.items < max(sizes):
        parser.error("every size is at least 1 and at most --items")
    profile = ExecutionProfile.model_validate(load_yaml(args.profile, "execution profile"))
    pool = settings(profile)
    records = _records(args.items, [float(s) for s in args.seconds.split(",")])
    workers = RealWorkers(profile)
    try:
        worker = workers(args.task, args.backend)
        load_s, _ = _send(worker, records[:1])
        document = {
            "task": args.task,
            "backend": args.backend,
            "device": workers.device,
            "items": args.items,
            "load_s": load_s,
            "explicit": explicit(worker, records, sizes),
            "probe": probed(worker, records, max(sizes), pool.probe_samples, pool.probe_gain),
        }
    finally:
        workers.close()
    document["pass"] = all(v["equal_to_alone"] for v in document["explicit"].values())
    text = json.dumps(document, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if document["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
