#!/usr/bin/env python
"""Score candidate TAXONOMY routing detectors over a completed triage run.

    uv run python scripts/analyze_routing_evidence.py <run_dir> <out_dir> [--workers N]

``run_dir`` is the triage out dir: each recording's ``<stem>.summary.json`` sits beside the run
root it names, under that stem's BIDS entity path. ``out_dir`` receives the extracted features,
the threshold sweeps, the per-family prevalence, the enumerated disagreements and a readable
summary. Idempotent: the manifest and the feature shards are reused when they are already
complete, so a killed run resumes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

from senselab.audio.workflows.triage.routing_analysis.families import task_family, task_id_of
from senselab.audio.workflows.triage.routing_analysis.features import RecordingFeatures, extract_features
from senselab.audio.workflows.triage.routing_analysis.report import load_features, write_report

MANIFEST = "manifest.jsonl"
SHARD_DIR = "features"


def build_manifest(run_dir: Path, out_dir: Path) -> Path:
    """Resolve every recording through its summary's ``run_root`` and record it once.

    Each run root is located beside its own summary, so the entity path the summary sits under is
    carried over rather than reconstructed from the recorded ``run_root``.

    Args:
        run_dir: The triage out dir, searched recursively for per-recording summaries.
        out_dir: Where the manifest is written.

    Returns:
        The manifest path. Reused unchanged when it already exists.

    Raises:
        FileNotFoundError: When ``run_dir`` holds no summaries.
    """
    manifest = out_dir / MANIFEST
    if manifest.exists():
        print(f"[manifest] reusing {manifest} ({sum(1 for _ in manifest.open())} rows)", flush=True)
        return manifest
    summaries = sorted(run_dir.rglob("*.summary.json"))
    if not summaries:
        raise FileNotFoundError(f"no *.summary.json under {run_dir}")
    started = time.time()
    temporary = manifest.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8") as handle:
        for index, summary_path in enumerate(summaries, start=1):
            summary = json.loads(summary_path.read_text())
            run_root = str(summary.get("run_root") or "")
            stem = str(summary.get("stem") or summary_path.name.removesuffix(".summary.json"))
            local = summary_path.parent / Path(run_root).name if run_root else None
            handle.write(
                json.dumps(
                    {
                        "stem": stem,
                        "run_root": run_root,
                        "store": str(local / "run" / "store.jsonl") if local else "",
                        "task_id": task_id_of(stem),
                        "family": task_family(task_id_of(stem)),
                    }
                )
            )
            handle.write("\n")
            if index % 2000 == 0:
                rate = index / max(time.time() - started, 1e-9)
                print(f"[manifest] {index}/{len(summaries)} at {rate:.0f}/s", flush=True)
    temporary.replace(manifest)
    print(f"[manifest] {len(summaries)} rows in {time.time() - started:.0f}s", flush=True)
    return manifest


def _extract_one(row: dict[str, str]) -> dict[str, object] | None:
    """Extract one recording's features, or record why it could not be read.

    Args:
        row: One manifest row.

    Returns:
        The features as a mapping, or None when the store is missing.
    """
    store = Path(row["store"])
    if not store.is_file():
        return None
    try:
        features = extract_features(store, row["stem"], row["run_root"], row["task_id"], row["family"])
    except (OSError, ValueError) as error:
        return {"stem": row["stem"], "error": f"{type(error).__name__}: {error}"}
    return asdict(features)


def extract_all(manifest: Path, out_dir: Path, workers: int) -> Path:
    """Extract features for every manifest row, resuming from what is already on disk.

    Args:
        manifest: The manifest.
        out_dir: Where the shard directory lives.
        workers: How many processes read stores in parallel.

    Returns:
        The features JSONL path.
    """
    shard_dir = out_dir / SHARD_DIR
    shard_dir.mkdir(parents=True, exist_ok=True)
    features_path = shard_dir / "features.jsonl"
    missing_path = shard_dir / "missing.jsonl"
    done: set[str] = set()
    if features_path.exists():
        with features_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    done.add(str(json.loads(line)["stem"]))
        print(f"[extract] resuming with {len(done)} already extracted", flush=True)
    rows = [json.loads(line) for line in manifest.open("r", encoding="utf-8") if line.strip()]
    pending = [row for row in rows if row["stem"] not in done]
    print(f"[extract] {len(pending)} of {len(rows)} to read with {workers} workers", flush=True)
    if not pending:
        return features_path
    started = time.time()
    written = 0
    with (
        features_path.open("a", encoding="utf-8") as sink,
        missing_path.open("a", encoding="utf-8") as misses,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        for index, (row, result) in enumerate(zip(pending, pool.map(_extract_one, pending, chunksize=8)), start=1):
            if result is None or "error" in result:
                reason = "missing" if result is None else str(result["error"])
                misses.write(json.dumps({"stem": row["stem"], "store": row["store"], "why": reason}) + "\n")
            else:
                sink.write(json.dumps(result, sort_keys=True) + "\n")
                written += 1
            if index % 500 == 0:
                elapsed = time.time() - started
                rate = index / max(elapsed, 1e-9)
                remaining = (len(pending) - index) / max(rate, 1e-9)
                print(
                    f"[extract] {index}/{len(pending)} at {rate:.1f}/s, ~{remaining / 60:.1f} min left",
                    flush=True,
                )
                sink.flush()
                misses.flush()
    print(f"[extract] wrote {written} in {time.time() - started:.0f}s", flush=True)
    return features_path


def main(argv: list[str] | None = None) -> int:
    """Run the analysis end to end.

    Args:
        argv: Command-line arguments, or None for ``sys.argv``.

    Returns:
        0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="a triage out dir, searched recursively for per-recording summaries")
    parser.add_argument("out_dir", type=Path, help="where the analysis is written")
    parser.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4)))
    parser.add_argument("--expect", type=int, default=None, help="assert this many recordings resolved")
    arguments = parser.parse_args(argv)

    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(arguments.run_dir, arguments.out_dir)
    n_rows = sum(1 for line in manifest.open() if line.strip())
    if arguments.expect is not None and n_rows != arguments.expect:
        raise SystemExit(f"resolved {n_rows} recordings, expected {arguments.expect}")
    features_path = extract_all(manifest, arguments.out_dir, arguments.workers)
    records: list[RecordingFeatures] = load_features(features_path)
    print(f"[report] scoring {len(records)} recordings", flush=True)
    index = write_report(records, arguments.out_dir)
    print(json.dumps({key: index[key] for key in ("n_recordings", "n_families")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
