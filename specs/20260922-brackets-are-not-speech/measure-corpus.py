"""Measure the bracket-token PII artefact over the replayed corpus. Read-only."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from senselab.audio.workflows.triage.nodes.common import live_entities
from senselab.utils.prov_store import ProvStore

CORPUS = Path("/orcd/scratch/bcs/002/satra/triage_replay_20260922/out")


def _overlaps(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _family(store: ProvStore) -> str | None:
    for activity in store.activities():
        if activity.node == "SPEECH" and activity.step == "expect":
            value = activity.parameters.get("task_family")
            return None if value is None else str(value)
    return None


def measure(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "run" / "store.jsonl"
    if not path.exists():
        return None
    store = ProvStore.read_jsonl(path)
    nodes = {activity.node for activity in store.activities()}
    if "REDACT" not in nodes:
        return None

    words = sorted(live_entities(store, "word"), key=lambda w: int(w.attributes["index"]))
    extents = [(int(w.attributes["index"]), w.extent, bool(w.attributes["bracketed"])) for w in words if w.extent]
    task_extents = [
        e.extent for e in live_entities(store, "span") if e.attributes.get("role") == "task_extent" and e.extent
    ]

    findings = []
    for finding in live_entities(store, "pii"):
        if finding.extent is None:
            continue
        start, end = float(finding.extent[0]), float(finding.extent[1])
        covered = [(i, ex, br) for i, ex, br in extents if _overlaps((start, end), (float(ex[0]), float(ex[1])))]
        hull = (
            (min(float(ex[0]) for _, ex, _ in covered), max(float(ex[1]) for _, ex, _ in covered)) if covered else None
        )
        findings.append(
            {
                "category": str(finding.attributes.get("category")),
                "source": str(finding.attributes.get("source")),
                "haystack": str(finding.attributes.get("haystack")),
                "in_stimulus": finding.attributes.get("in_stimulus"),
                "duration_s": round(end - start, 4),
                "n_covered": len(covered),
                "n_covered_bracketed": sum(1 for _, _, br in covered if br),
                "all_covered_bracketed": bool(covered) and all(br for _, _, br in covered),
                "any_covered_bracketed": any(br for _, _, br in covered),
                "covered_hull_s": None if hull is None else round(hull[1] - hull[0], 4),
                "hits_task_extent": any(_overlaps((start, end), (float(t[0]), float(t[1]))) for t in task_extents),
            }
        )

    return {
        "stem": run_dir.name,
        "family": _family(store),
        "n_words": len(extents),
        "n_bracketed_words": sum(1 for _, _, br in extents if br),
        "n_task_extents": len(task_extents),
        "findings": findings,
    }


def _safe(run_dir: Path) -> dict[str, Any] | None:
    try:
        return measure(run_dir)
    except Exception as error:  # noqa: BLE001
        return {"stem": run_dir.name, "error": f"{type(error).__name__}: {error}"}


def main() -> None:
    out = Path(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    run_dirs = sorted(p.parent.parent for p in CORPUS.glob("sub-*/ses-*/*/run/store.jsonl"))
    print(f"{len(run_dirs)} run directories", flush=True)
    written = 0
    with out.open("w") as handle, ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(_safe, run_dirs, chunksize=16):
            if row is None:
                continue
            handle.write(json.dumps(row) + "\n")
            written += 1
            if written % 2000 == 0:
                print(f"{written} written", flush=True)
    print(f"done: {written} rows", flush=True)


if __name__ == "__main__":
    main()
