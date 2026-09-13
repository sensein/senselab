#!/usr/bin/env python
"""Score candidate TAXONOMY routing detectors over a completed triage run.

    uv run python scripts/analyze_routing_evidence.py <run_dir> <out_dir> [--workers N] [--config FILE]

``run_dir`` is the triage out dir: each recording's ``<stem>.summary.json`` sits beside the run
root it names, under that stem's BIDS entity path. ``out_dir`` receives the extracted features,
the threshold sweeps, the per-family prevalence, the enumerated disagreements and a readable
summary. Idempotent: the manifest and the feature shards are reused when they are already
complete, so a killed run resumes.

A reused manifest is checked against the tree it names: none of its sampled store paths resolving
means it resolves some other tree, and it is refused rather than extracted from. An extract that
writes no features from a manifest with rows in it fails too. Both are the same defect -- a run that
reads nothing and reports success -- caught at the two places it can appear.

Which labels a span carries is read from ``windows.<classifier>`` in the triage configuration, the
same pair PREPROCESS stamps its own windows with, so ``--config`` changes both together.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.label_membership import LabelMembership
from senselab.audio.workflows.triage.routing_analysis.families import task_family, task_id_of
from senselab.audio.workflows.triage.routing_analysis.features import (
    RecordingFeatures,
    extract_features,
    onomatopoeic_vocabulary,
    span_label_memberships,
)
from senselab.audio.workflows.triage.routing_analysis.report import (
    SHARD_SUFFIX,
    dump_features,
    load_feature_column,
    load_features,
    shard_files,
    write_report,
)

MANIFEST = "manifest.jsonl"
SHARD_DIR = "features"
PART_NAME = "part-{part:05d}" + SHARD_SUFFIX
PART_ROWS = 4000
"""How many recordings fill one shard part before it is written and the next one begins."""
MANIFEST_PROBE = 64
"""How many rows of a reused manifest are stat'd before it is trusted to name this tree."""
SUMMARY_DEPTHS = ("*.summary.json", "*/*.summary.json", "*/*/*.summary.json")
"""Every depth ``entity_subdir`` can place a summary at: no entity, subject only, subject-session."""


def check_manifest_resolves(manifest: Path, rows: Sequence[Mapping[str, str]]) -> None:
    """Refuse a manifest whose stores are not where it says they are.

    Args:
        manifest: The manifest being reused, named in the refusal.
        rows: Its rows, each carrying the ``store`` it resolved a recording to.

    Raises:
        FileNotFoundError: When not one of the probed rows names a store that exists.
    """
    if not rows:
        return
    step = max(1, len(rows) // MANIFEST_PROBE)
    probed = [str(row.get("store") or "") for row in rows[::step][:MANIFEST_PROBE]]
    if any(path and Path(path).is_file() for path in probed):
        return
    raise FileNotFoundError(
        f"{manifest} names {len(rows)} recordings and none of the {len(probed)} probed stores "
        f"exists (e.g. {probed[0] or '<no store>'}); it resolves a tree that is not there. Delete it "
        "to rebuild against the tree this run was pointed at."
    )


def build_manifest(run_dir: Path, out_dir: Path) -> Path:
    """Resolve every recording through its summary's ``run_root`` and record it once.

    Each run root is located beside its own summary, so the entity path the summary sits under is
    carried over rather than reconstructed from the recorded ``run_root``.

    Args:
        run_dir: The triage out dir, searched at each depth an entity path can place a summary at.
        out_dir: Where the manifest is written.

    Returns:
        The manifest path. Reused unchanged when it already exists and still resolves.

    Raises:
        FileNotFoundError: When ``run_dir`` holds no summaries, or when a reused manifest names
            stores that no longer exist.
    """
    manifest = out_dir / MANIFEST
    if manifest.exists():
        rows = [json.loads(line) for line in manifest.open("r", encoding="utf-8") if line.strip()]
        check_manifest_resolves(manifest, rows)
        print(f"[manifest] reusing {manifest} ({len(rows)} rows)", flush=True)
        return manifest
    summaries = sorted({path for depth in SUMMARY_DEPTHS for path in run_dir.glob(depth)})
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


def _extract_one(
    row: dict[str, str],
    memberships: Mapping[str, LabelMembership],
    onomatopoeic: frozenset[str],
) -> RecordingFeatures | str | None:
    """Extract one recording's features, or say why it could not be read.

    Args:
        row: One manifest row.
        memberships: Which labels a span carries, per classifier.
        onomatopoeic: The vocabulary an unbracketed consensus word is counted against.

    Returns:
        The features, the reason the store could not be read, or None when it is missing.
    """
    store = Path(row["store"])
    if not store.is_file():
        return None
    try:
        return extract_features(
            store,
            row["stem"],
            row["run_root"],
            row["task_id"],
            row["family"],
            memberships,
            onomatopoeic=onomatopoeic,
        )
    except (OSError, ValueError) as error:
        return f"{type(error).__name__}: {error}"


def _extracted_stems(shard_dir: Path) -> set[str]:
    """Which recordings the shard directory already holds, read off the key column alone.

    Args:
        shard_dir: The shard directory.

    Returns:
        Every stem already written.
    """
    return {str(stem) for shard in shard_files(shard_dir) for stem in load_feature_column(shard, "stem")}


def extract_all(
    manifest: Path,
    out_dir: Path,
    workers: int,
    memberships: Mapping[str, LabelMembership],
    onomatopoeic: frozenset[str],
) -> Path:
    """Extract features for every manifest row, resuming from what is already on disk.

    Each completed part is a shard file of its own, so a killed run loses at most the part it was
    filling and resumes from the parts that landed.

    Args:
        manifest: The manifest.
        out_dir: Where the shard directory lives.
        workers: How many processes read stores in parallel.
        memberships: Which labels a span carries, per classifier.
        onomatopoeic: The vocabulary an unbracketed consensus word is counted against.

    Returns:
        The shard directory.
    """
    shard_dir = out_dir / SHARD_DIR
    shard_dir.mkdir(parents=True, exist_ok=True)
    missing_path = shard_dir / "missing.jsonl"
    done = _extracted_stems(shard_dir)
    if done:
        print(f"[extract] resuming with {len(done)} already extracted", flush=True)
    rows = [json.loads(line) for line in manifest.open("r", encoding="utf-8") if line.strip()]
    pending = [row for row in rows if row["stem"] not in done]
    print(f"[extract] {len(pending)} of {len(rows)} to read with {workers} workers", flush=True)
    if not pending:
        return shard_dir
    started = time.time()
    written = 0
    part = len(shard_files(shard_dir))
    buffered: list[RecordingFeatures] = []
    with (
        missing_path.open("a", encoding="utf-8") as misses,
        ProcessPoolExecutor(max_workers=workers) as pool,
    ):
        extract = partial(_extract_one, memberships=memberships, onomatopoeic=onomatopoeic)
        for index, (row, result) in enumerate(zip(pending, pool.map(extract, pending, chunksize=8)), start=1):
            if isinstance(result, RecordingFeatures):
                buffered.append(result)
                written += 1
            else:
                reason = "missing" if result is None else result
                misses.write(json.dumps({"stem": row["stem"], "store": row["store"], "why": reason}) + "\n")
            if len(buffered) >= PART_ROWS:
                dump_features(buffered, shard_dir / PART_NAME.format(part=part))
                buffered.clear()
                part += 1
            if index % 500 == 0:
                elapsed = time.time() - started
                rate = index / max(elapsed, 1e-9)
                remaining = (len(pending) - index) / max(rate, 1e-9)
                print(
                    f"[extract] {index}/{len(pending)} at {rate:.1f}/s, ~{remaining / 60:.1f} min left",
                    flush=True,
                )
                misses.flush()
        if buffered:
            dump_features(buffered, shard_dir / PART_NAME.format(part=part))
    print(f"[extract] wrote {written} in {time.time() - started:.0f}s", flush=True)
    return shard_dir


def main(argv: list[str] | None = None) -> int:
    """Run the analysis end to end.

    Args:
        argv: Command-line arguments, or None for ``sys.argv``.

    Returns:
        0 on success.

    Raises:
        SystemExit: When the manifest resolves a different number of recordings than ``--expect``,
            or when the extract wrote no features from a manifest that has rows.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="a triage out dir, searched recursively for per-recording summaries")
    parser.add_argument("out_dir", type=Path, help="where the analysis is written")
    parser.add_argument("--workers", type=int, default=min(16, (os.cpu_count() or 4)))
    parser.add_argument("--expect", type=int, default=None, help="assert this many recordings resolved")
    parser.add_argument("--config", type=Path, default=None, help="a partial triage config override")
    arguments = parser.parse_args(argv)

    arguments.out_dir.mkdir(parents=True, exist_ok=True)
    config = load_triage_config(arguments.config)
    memberships = span_label_memberships(config)
    onomatopoeic = onomatopoeic_vocabulary(config)
    print(f"[extract] span labels: config hash {config.config_hash}, {memberships}", flush=True)
    print(f"[extract] onomatopoeic vocabulary: {sorted(onomatopoeic)}", flush=True)
    manifest = build_manifest(arguments.run_dir, arguments.out_dir)
    n_rows = sum(1 for line in manifest.open() if line.strip())
    if arguments.expect is not None and n_rows != arguments.expect:
        raise SystemExit(f"resolved {n_rows} recordings, expected {arguments.expect}")
    shard_dir = extract_all(manifest, arguments.out_dir, arguments.workers, memberships, onomatopoeic)
    records: list[RecordingFeatures] = load_features(shard_dir)
    if n_rows and not records:
        raise SystemExit(
            f"extracted 0 features from {n_rows} manifest rows; every store was unreadable or "
            f"missing. See {shard_dir / 'missing.jsonl'}"
        )
    print(f"[report] scoring {len(records)} recordings", flush=True)
    index = write_report(records, arguments.out_dir)
    print(json.dumps({key: index[key] for key in ("n_recordings", "n_families")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
