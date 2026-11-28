"""Per-item time and device memory of one host backend at explicit request sizes, and pydra's batch probe over them.

`uv run python benchmarks/scale/batch_sizes.py --profile PROFILE --items N --sizes 1,2,4,... --seconds 1,2,3,4
--rounds R --seed S [--task speech_to_text --backend transformers] [--out FILE]`

The backend's worker module runs in this process on `cuda`, loaded from its catalogue model by commit. It must be a
host backend whose `handle` encodes a request's items together; the default, `transformers` of `speech_to_text`,
runs one pipeline call per request. The records are `tone_set(N, seconds)` sorted by duration, so a request holds
neighbours in length as a pool's call does. Before timing, every size is sent once per distinct duration. Then for R
rounds the sizes run in a seeded shuffled order, recorded in the document. Per size it reports the medians over rounds
of seconds per item and per audio second, the peak device bytes above the loaded model, the errors, and how many
outputs agree with their size-1 output under the packaged run config's thresholds (`compare`).

Probe: pydra's `probe_batch` over this run's medians (not a pool's calibration), with the pool settings senselab gives
a pool (`warm.settings`). Its bound is `MeasuredFit.memory_bound` of the memory fit measured here, on a class of this
device's memory with the profile's `quantum_items` and `memory_headroom`.

PASS: every item of every request has outputs, and for a batch-invariant backend every output agrees with its size-1
output. A batch-variant backend's agreement is reported, not judged.
"""

import argparse
import importlib
import json
import random
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy
from senselab_core.runtime.call import RunConfig, packaged
from senselab_core.runtime.compare import compare
from senselab_core.runtime.config import load_yaml
from senselab_core.runtime.execution.warm import settings
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.testing.pools import DeviceFit, memory_bound, probe_size
from senselab_core.testing.synth import tone_set

AMPLITUDE = 0.1
"""The tones' amplitude: well inside full scale, as recorded speech is."""

LINE_POINTS = 2
"""Distinct units a memory fit needs: two points fix a line."""

Record = tuple[dict[str, Any], float]
Handle = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]


class Device(Protocol):
    """The device's memory counters: bytes allocated now, the peak since `reset`, and the device's total."""

    def allocated(self) -> int:
        """Return the bytes allocated now."""
        ...

    def reset(self) -> None:
        """Start a new peak."""
        ...

    def peak(self) -> int:
        """Return the most bytes allocated since `reset`."""
        ...

    def total(self) -> int:
        """Return the device's memory in bytes."""
        ...


class Cuda:
    """`Device` over `torch.cuda` on device 0, synchronised before each read."""

    def __init__(self) -> None:
        """Import torch."""
        self.torch = importlib.import_module("torch")

    def allocated(self) -> int:
        """Return the bytes allocated now."""
        self.torch.cuda.synchronize()
        return int(self.torch.cuda.memory_allocated())

    def reset(self) -> None:
        """Start a new peak."""
        self.torch.cuda.synchronize()
        self.torch.cuda.reset_peak_memory_stats()

    def peak(self) -> int:
        """Return the most bytes allocated since `reset`."""
        self.torch.cuda.synchronize()
        return int(self.torch.cuda.max_memory_allocated())

    def total(self) -> int:
        """Return the device's memory in bytes."""
        return int(self.torch.cuda.get_device_properties(0).total_memory)


@dataclass
class Call:
    """One request: its size, wall seconds, audio seconds, longest record, peak bytes above the model, and results."""

    size: int
    wall_s: float
    audio_s: float
    longest_s: float
    extra_bytes: int
    results: list[dict[str, Any]] = field(repr=False)


def records(items: int, seconds: Sequence[float]) -> list[Record]:
    """Return `items` tone records with their durations, sorted by duration (stable)."""
    made = [
        ({"id": f"r{i}", "inputs": {"audio": audio}}, seconds[i % len(seconds)])
        for i, audio in enumerate(tone_set(items, seconds, amplitude=AMPLITUDE))
    ]
    return sorted(made, key=lambda r: r[1])


def chunks(recs: Sequence[Record], size: int) -> list[Sequence[Record]]:
    """Return `recs` in consecutive requests of `size`."""
    return [recs[i : i + size] for i in range(0, len(recs), size)]


@dataclass(frozen=True)
class Plan:
    """The request sizes, the timed rounds over them, and the seed of each round's order."""

    sizes: Sequence[int]
    rounds: int
    seed: int


@dataclass(frozen=True)
class Loaded:
    """A loaded backend's handle and whether it declares itself batch-invariant."""

    handle: Handle
    batch_invariant: bool


def agreement(alone: dict[str, Any], calls: Sequence[Call], recs: Sequence[Record]) -> int:
    """Return how many results of `calls` (over `recs`, in order) have outputs agreeing with their size-1 outputs."""
    thresholds = RunConfig().thresholds
    results = [r for c in calls for r in c.results]
    return sum(
        "outputs" in r and item["id"] in alone and compare(alone[item["id"]], r["outputs"], thresholds).agree
        for (item, _), r in zip(recs, results, strict=True)
    )


@dataclass
class Bench:
    """Requests to one loaded backend, each measured on `device` above the model's `resident` bytes."""

    handle: Handle
    device: Device
    resident: int

    def send(self, chunk: Sequence[Record]) -> Call:
        """Send one request and return what it cost."""
        self.device.reset()
        started = time.perf_counter()
        results = self.handle([item for item, _ in chunk])
        wall = time.perf_counter() - started
        lengths = [s for _, s in chunk]
        return Call(len(chunk), wall, sum(lengths), max(lengths), self.device.peak() - self.resident, results)

    def warm_up(self, recs: Sequence[Record], sizes: Sequence[int]) -> int:
        """Send every size once per distinct duration; return the number of requests sent."""
        sent = 0
        for duration in sorted({s for _, s in recs}):
            same = [r for r in recs if r[1] == duration]
            for size in sizes:
                self.send(same[:size])
                sent += 1
        return sent

    def explicit(self, recs: Sequence[Record], plan: Plan) -> tuple[dict[str, Any], list[list[int]], list[Call]]:
        """Run every size over `recs` in seeded shuffled rounds; return per size, the orders, and every call."""
        rng = random.Random(plan.seed)
        orders: list[list[int]] = []
        runs: dict[int, list[list[Call]]] = {s: [] for s in plan.sizes}
        for _ in range(plan.rounds):
            order = list(plan.sizes)
            rng.shuffle(order)
            orders.append(order)
            for size in order:
                runs[size].append([self.send(c) for c in chunks(recs, size)])
        alone_results = [r for c in runs[1][0] for r in c.results]
        alone = {item["id"]: r["outputs"] for (item, _), r in zip(recs, alone_results, strict=True) if "outputs" in r}
        found = {}
        for size in plan.sizes:
            walls = [sum(c.wall_s for c in calls) for calls in runs[size]]
            audio = sum(c.audio_s for c in runs[size][0])
            found[str(size)] = {
                "per_item_s": statistics.median(walls) / len(recs),
                "per_audio_s": statistics.median(walls) / audio,
                "peak_extra_bytes": max(c.extra_bytes for calls in runs[size] for c in calls),
                "errors": sum("outputs" not in r for calls in runs[size] for c in calls for r in c.results),
                "agree_with_alone": agreement(alone, runs[size][0], recs),
            }
        return found, orders, [c for calls in runs.values() for run in calls for c in run]

    def probed(self, recs: Sequence[Record], bound: int, samples: int, gain: float) -> dict[str, Any]:
        """Run `recs` call by call at the size pydra's probe asks for; its calls and the size it settles on."""
        per_unit: dict[int, list[float]] = {}
        calls: list[int] = []
        i = 0
        while i < len(recs):
            size = probe_size(per_unit, bound, samples, gain)
            call = self.send(recs[i : i + size])
            if call.size == size:
                per_unit.setdefault(size, []).append(call.wall_s / call.audio_s)
            calls.append(call.size)
            i += call.size
        return {
            "calls": calls,
            "chosen": probe_size(per_unit, bound, samples, gain),
            "per_audio_s": {str(b): statistics.median(v) for b, v in sorted(per_unit.items())},
            "samples": samples,
            "gain": gain,
            "bound": bound,
        }


def memory_fit(resident: int, calls: Sequence[Call]) -> DeviceFit:
    """Fit each call's peak above the model as `m0 + m1 × (size × longest)`; `m1` None under two distinct units."""
    units = numpy.array([c.size * c.longest_s for c in calls])
    extra = numpy.array([c.extra_bytes for c in calls], dtype=float)
    if len(set(units.tolist())) < LINE_POINTS:
        return DeviceFit(resident, float(extra.max()), None, len(calls))
    m1, m0 = numpy.polyfit(units, extra, 1)
    return DeviceFit(resident, float(m0), float(m1), len(calls))


def loaded(task: str, backend_name: str) -> Loaded:
    """Load the host backend's catalogue model on `cuda`; return its handle over the task's default params."""
    registry = packaged().registry
    found = [b for b in registry.backends(task) if b.name == backend_name]
    if not found or found[0].env.kind != "host":
        raise SystemExit(f"{task!r} has no host backend {backend_name!r}")
    backend = found[0]
    entry = packaged().catalogue.backend_entry(backend)
    if entry is None:
        raise SystemExit(f"backend {backend_name!r} has no catalogue model")
    module = importlib.import_module(backend.worker)
    state = module.load({"model": {"source": "hf", "id": entry.id, "identity": entry.revision}, "device": "cuda"})
    params = {**registry.spec(task).params().model_dump(mode="json"), "backend_params": {}}

    def handle(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return list(module.handle(state, items, params))

    return Loaded(handle, backend.batch_invariant)


def run(
    loaded: Loaded, device: Device, profile: ExecutionProfile, recs: Sequence[Record], plan: Plan
) -> dict[str, Any]:
    """Warm up, run the explicit sizes and the probe over a loaded backend; return the document with its verdict."""
    bench = Bench(loaded.handle, device, device.allocated())
    warmed = bench.warm_up(recs, plan.sizes)
    per_size, orders, calls = bench.explicit(recs, plan)
    fit = memory_fit(bench.resident, calls)
    bound = memory_bound(fit, device.total(), profile, max(s for _, s in recs))
    pool = settings(profile)
    probe = bench.probed(recs, bound, pool.probe_samples, pool.probe_gain)
    errors = sum(v["errors"] for v in per_size.values())
    agree = all(v["agree_with_alone"] == len(recs) for v in per_size.values())
    return {
        "batch_invariant": loaded.batch_invariant,
        "items": len(recs),
        "warm_up_requests": warmed,
        "orders": orders,
        "seed": plan.seed,
        "explicit": per_size,
        "memory_fit": {"resident": fit.resident, "m0": fit.m0, "m1": fit.m1, "n": fit.n},
        "probe": probe,
        "pass": errors == 0 and (agree or not loaded.batch_invariant),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Load the backend, run the measurement and print the document."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--profile", type=Path, required=True, help="a whole execution profile file")
    parser.add_argument("--items", type=int, required=True, help="records per size")
    parser.add_argument("--sizes", required=True, help="comma-separated request sizes; 1 is always added")
    parser.add_argument("--seconds", required=True, help="comma-separated record durations, cycled")
    parser.add_argument("--rounds", type=int, required=True, help="timed rounds over every size")
    parser.add_argument("--seed", type=int, required=True, help="the seed of each round's size order")
    parser.add_argument("--task", default="speech_to_text", help="the task")
    parser.add_argument("--backend", default="transformers", help="the task's host backend")
    parser.add_argument("--out", type=Path, default=None, help="where to write the document")
    args = parser.parse_args(argv)
    sizes = sorted({int(s) for s in args.sizes.split(",")} | {1})
    if args.items < max(sizes) or args.rounds < 1:
        parser.error("--items is at least the largest size and --rounds at least 1")
    profile = ExecutionProfile.model_validate(load_yaml(args.profile, "execution profile"))
    backend = loaded(args.task, args.backend)
    recs = records(args.items, [float(s) for s in args.seconds.split(",")])
    document: dict[str, Any] = {"task": args.task, "backend": args.backend}
    document |= run(backend, Cuda(), profile, recs, Plan(sizes, args.rounds, args.seed))
    text = json.dumps(document, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0 if document["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
