"""Cost of CrisperWhisper's confidence pass: decode wall time with confidence on and off, per window, on one engine.

`uv run python benchmarks/crisperwhisper_confidence_cost.py --engine ct2|transformers --device cpu|cuda --out FILE`
decodes `--items` LibriSpeech test-clean utterances (the `--items` smallest sha256 of the member name, fetched through
the data cache, `$SENSELAB_DATA_CACHE`) in the `crisperwhisper` worker of `--execution` (a profile, default `laptop`),
each with `confidence` true and false after one warm-up decode. The times are the worker's own (`_diagnostics`):
`decode_s` is the library decode, `confidence_s` the confidence pass, `engine_build_s` the one-off engine load. Writes
`confidence-cost@1`.

`--transcript --engine E --device D --out FILE` writes the transcript and diagnostics of one long-form item
(`senselab_core.testing.librispeech.long_form`); `--compare A B` judges two such files
by the packaged `transcript@2` comparison and run-config thresholds.
"""

import argparse
import hashlib
import heapq
import importlib
import io
import json
import platform
import statistics
import subprocess
import tarfile
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

FORMAT = "confidence-cost@1"
MANIFEST = Path(__file__).resolve().parent / "manifests" / "librispeech-test-clean.yaml"
RATE = 16_000
ENGINES = ("ct2", "transformers")
DEVICES = ("cpu", "cuda")


def _percentile(values: Sequence[float], q: float) -> float:
    """Return the nearest-rank `q`-quantile (0 < q <= 1) of `values`."""
    ordered = sorted(values)
    return ordered[max(0, -(-len(ordered) * int(round(q * 100)) // 100) - 1)]


def _stats(values: Sequence[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "p90": _percentile(values, 0.9), "mean": statistics.fmean(values)}


def _shares(items: Sequence[dict[str, Any]]) -> dict[str, float]:
    decode = sum(i["on"]["decode_s"] for i in items)
    confidence = sum(i["on"]["confidence_s"] for i in items)
    off = sum(i["off"]["decode_s"] for i in items)
    total = decode + confidence
    return {
        "share_of_decode": confidence / decode,
        "share_of_total": confidence / total,
        "share_by_difference": (total - off) / total,
    }


def summarize(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return per-window statistics and the confidence shares from per-item records.

    Each item holds `duration_s` and `on` and `off`, each `{"windows", "decode_s", "confidence_s"}`. Per window values
    divide an item's seconds by its windows. Over the `on` runs, `share_of_decode` is confidence seconds over
    decode seconds and `share_of_total` is confidence seconds over decode plus confidence seconds;
    `share_by_difference` is the extra total time of `on` over `off`, over the total time of `on`. `by_duration`
    gives the same shares for the terciles of `duration_s` (shortest, middle, longest). Items without a window are
    left out of the per-window values.
    """
    timed = [i for i in items if i["on"]["windows"] > 0 and i["off"]["windows"] > 0]
    ordered = sorted(items, key=lambda i: i["duration_s"])
    cut = [len(ordered) * k // 3 for k in range(4)]
    terciles = {
        name: {"items": len(part), "duration_s": [part[0]["duration_s"], part[-1]["duration_s"]], **_shares(part)}
        for name, part in zip(("short", "middle", "long"), (ordered[cut[k] : cut[k + 1]] for k in range(3)))
        if part
    }
    return {
        "items": len(items),
        "windows": sum(i["on"]["windows"] for i in items),
        "per_window_s": {
            "decode_on": _stats([i["on"]["decode_s"] / i["on"]["windows"] for i in timed]),
            "confidence_on": _stats([i["on"]["confidence_s"] / i["on"]["windows"] for i in timed]),
            "decode_off": _stats([i["off"]["decode_s"] / i["off"]["windows"] for i in timed]),
            "total_on": _stats([(i["on"]["decode_s"] + i["on"]["confidence_s"]) / i["on"]["windows"] for i in timed]),
        },
        "total_s": {
            "decode_on": sum(i["on"]["decode_s"] for i in items),
            "confidence_on": sum(i["on"]["confidence_s"] for i in items),
            "decode_off": sum(i["off"]["decode_s"] for i in items),
        },
        **_shares(items),
        "by_duration": terciles,
    }


def select_items(archive: Path, count: int) -> list[tuple[str, Any]]:
    """Return the `count` flac members of the test-clean `archive` with the smallest name sha256, as pairs."""
    soundfile = importlib.import_module("soundfile")
    heap: list[tuple[int, str, bytes]] = []
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            if not member.name.endswith(".flac"):
                continue
            handle = tar.extractfile(member)
            assert handle is not None  # noqa: S101 - a flac member is a file
            rank = int(hashlib.sha256(member.name.encode()).hexdigest(), 16)
            heapq.heappush(heap, (-rank, member.name, handle.read()))
            if len(heap) > count:
                heapq.heappop(heap)
    chosen = sorted(heap, key=lambda h: (-h[0], h[1]))
    items = []
    for _, name, data in chosen:
        samples, rate = soundfile.read(io.BytesIO(data), dtype="float32")
        assert rate == RATE, (name, rate)  # noqa: S101 - LibriSpeech is 16 kHz
        items.append((name, samples))
    return items


def _decode(worker: Any, name: str, samples: Any, params: dict[str, Any], confidence: bool) -> dict[str, Any]:  # noqa: ANN401
    audio_module = importlib.import_module("senselab_core.kernel.signals")
    wire = importlib.import_module("senselab_core.kernel.wire").wire_form
    item = {"id": name, "inputs": {"audio": wire(audio_module.Audio.from_array(samples[None, :], RATE))}}
    request = {**params, "backend_params": {**params["backend_params"], "confidence": confidence}}
    started = time.perf_counter()
    [result] = worker.run([item], request)
    wall = time.perf_counter() - started
    if "error" in result:
        raise RuntimeError(f"{name} failed with confidence={confidence}: {result['error']}")
    diagnostics = result["outputs"]["diagnostics"]
    return {
        "windows": diagnostics["windows"],
        "decode_s": diagnostics["decode_s"],
        "confidence_s": diagnostics["confidence_s"],
        "engine_build_s": diagnostics["engine_build_s"],
        "wall_s": wall,
    }


def _git_head() -> str | None:
    """Return the commit of the checkout the imported `senselab` package is in, which is the code measured."""
    package = Path(importlib.import_module("senselab").__file__ or "").parent
    try:
        return subprocess.run(
            ["git", "-C", str(package), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _archive() -> Path:
    fetch = importlib.import_module("senselab_core.runtime.fetch")
    return Path(
        fetch.fetch(fetch.DataManifest.load(MANIFEST), cache_root=fetch.default_cache_root())["test-clean.tar.gz"]
    )


def long_form_transcript(engine: str, device: str, execution: str) -> dict[str, Any]:
    """Return the transcript and diagnostics of the long-form item, confidence on."""
    profiles = importlib.import_module("senselab_core.runtime.profiles")
    workers = importlib.import_module("senselab_core.testing.workers")
    audio_module = importlib.import_module("senselab_core.kernel.signals")
    wire = importlib.import_module("senselab_core.kernel.wire").wire_form
    samples = importlib.import_module("senselab_core.testing.librispeech").long_form(_archive())
    item = {"id": "long", "inputs": {"audio": wire(audio_module.Audio.from_array(samples[None, :], RATE))}}
    params = {"language": None, "timestamps": "word", "backend_params": {"engine": engine}, "_diagnostics": True}
    real = workers.RealWorkers(profiles.ExecutionProfile.load(execution), device)
    try:
        [result] = real("speech_to_text", "crisperwhisper").run([item], params)
    finally:
        real.close()
    if "error" in result:
        raise RuntimeError(f"the long-form item failed: {result['error']}")
    return {
        "engine": engine,
        "device": device,
        "senselab_head": _git_head(),
        "transcript": result["outputs"]["transcript"],
        "diagnostics": result["outputs"]["diagnostics"],
    }


def compare_transcripts(expected: Path, actual: Path) -> dict[str, Any]:
    """Return whether two `long_form_transcript` files agree under the packaged comparison, and how they differ."""
    compare = importlib.import_module("senselab_core.runtime.compare")
    a, b = (json.loads(path.read_text()) for path in (expected, actual))
    result = compare.compare(
        {"transcript": a["transcript"]}, {"transcript": b["transcript"]}, compare.RunConfigFile.load().thresholds
    )
    gaps = [d.absolute for d in result.differences if d.absolute is not None]
    return {
        "agree": result.agree,
        "heads": [a.get("senselab_head"), b.get("senselab_head")],
        "differences": [d.model_dump(mode="json") for d in result.differences],
        "largest_absolute": max(gaps, default=None),
    }


def measure(engine: str, device: str, count: int, execution: str) -> dict[str, Any]:
    """Run the measurement on `engine` and `device` and return the `confidence-cost@1` document."""
    hardware = importlib.import_module("senselab_core.runtime.hardware")
    profiles = importlib.import_module("senselab_core.runtime.profiles")
    workers = importlib.import_module("senselab_core.testing.workers")
    call = importlib.import_module("senselab_core.runtime.call")
    host = hardware.probe_hardware()
    if device == "cuda" and not host.gpus:
        raise SystemExit("--device cuda on a host with no GPU")
    chosen = select_items(_archive(), count + 1)
    warmup, measured = chosen[0], chosen[1:]
    real = workers.RealWorkers(profiles.ExecutionProfile.load(execution), device)
    backend = next(b for b in call.packaged().registry.backends("speech_to_text") if b.name == "crisperwhisper")
    entry = call.packaged().catalogue.backend_entry(backend)
    params: dict[str, Any] = {
        "language": None,
        "timestamps": "word",
        "backend_params": {"engine": engine},
        "_diagnostics": True,
    }
    try:
        started = time.perf_counter()
        worker = real("speech_to_text", "crisperwhisper")
        first = _decode(worker, *warmup, params, True)
        load_s = time.perf_counter() - started
        _decode(worker, *warmup, params, False)
        records = []
        for index, (name, samples) in enumerate(measured):
            order = (True, False) if index % 2 == 0 else (False, True)
            runs = {on: _decode(worker, name, samples, params, on) for on in order}
            records.append(
                {
                    "id": name,
                    "duration_s": len(samples) / RATE,
                    "on": runs[True],
                    "off": runs[False],
                    "on_first": order[0],
                }
            )
    finally:
        real.close()
    return {
        "format": FORMAT,
        "engine": engine,
        "device": device,
        "model": {"id": entry.id, "revision": entry.revision},
        "host": {**asdict(host), "node": platform.node()},
        "senselab_head": _git_head(),
        "load": {"first_decode_wall_s": load_s, "engine_build_s": first["engine_build_s"], "warmup": warmup[0]},
        "items": records,
        "summary": summarize(records),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the command line, measure, and write the document to `--out`."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--engine", choices=ENGINES)
    parser.add_argument("--device", choices=DEVICES)
    parser.add_argument("--items", type=int, default=50)
    parser.add_argument("--execution", default="laptop", help="a packaged execution profile, or a profile file")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--summarize", type=Path, help="print the summary of an existing result file and exit")
    parser.add_argument("--transcript", action="store_true", help="write the long-form item's transcript to --out")
    parser.add_argument("--compare", type=Path, nargs=2, help="compare two --transcript files and exit")
    args = parser.parse_args(argv)
    if args.summarize:
        print(json.dumps(summarize(json.loads(args.summarize.read_text())["items"]), indent=2))
        return
    if args.compare:
        print(json.dumps(compare_transcripts(*args.compare), indent=2))
        return
    if not (args.engine and args.device and args.out):
        parser.error("--engine, --device and --out are required unless --summarize")
    if args.transcript:
        document = long_form_transcript(args.engine, args.device, args.execution)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(document, indent=2) + "\n")
        print(json.dumps(document["diagnostics"], indent=2))
        return
    document = measure(args.engine, args.device, args.items, args.execution)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps(document["summary"], indent=2))


if __name__ == "__main__":
    main()
