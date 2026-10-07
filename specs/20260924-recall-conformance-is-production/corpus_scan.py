r"""Replay the conformance fold over a finished run tree, on both sides of the recall change.

Read-only over the run tree: it reads each recording's ``summary/summary.json`` for the gate
bounds and the decision the run recorded, and its ``run/store.jsonl`` for the readings, and writes
one row per recording to its own output file. It loads no model, decodes no audio, imports no
senselab, and writes nothing under the run tree — ``streams/`` there are symlinks into another
tree and writing through one has overwritten a corpus before.

The pre-change and post-change conformance gate names per family come from
``family_gates.py``'s JSON, which is built from the live expectation rows, so both sides of the
A/B are one table rather than two hand-written lists.

Usage, as a Slurm array task::

    SCAN_ROOT=<run>/out SCAN_TABLE=<family_gates.json> SCAN_OUT=<dir> python corpus_scan.py

``SLURM_ARRAY_TASK_ID`` and ``SLURM_ARRAY_TASK_COUNT`` select this task's slice.
"""

from __future__ import annotations

import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

UNDETERMINED = "UNDETERMINED"
"""What a gate answers when its reading or its bound is absent."""

CONFORMANCE_WHY = "reported that what the instruction asked for did not happen"
"""The substring identifying the flag a failed task conformance raises."""



def store_readings(store: Path) -> dict[str, Any]:
    """The live value of each reading this scan wants, latest write winning.

    Args:
        store: The recording's ``store.jsonl``.

    Returns:
        Reading name to value, absent where the store holds none.
    """
    found: dict[str, Any] = {}
    with store.open() as handle:
        for line in handle:
            if not _READING_LINE.search(line):
                continue
            record = json.loads(line)
            if record.get("record") != "entity" or record.get("prov_type") != "measurement":
                continue
            name = record["attributes"].get("name")
            if name in READINGS:
                found[str(name)] = record["attributes"].get("value")
    return found


def passes(value: Any, bound: Any, op: str) -> Any:  # noqa: ANN401 — a gate's own types
    """Whether one reading clears one bound.

    Args:
        value: The reading, or None when nobody took it.
        bound: The bound, or None when no layer measured one.
        op: ``at_least`` or ``at_most``.

    Returns:
        True, False, or :data:`UNDETERMINED`.
    """
    if value is None or bound is None:
        return UNDETERMINED
    return float(value) >= float(bound) if op == "at_least" else float(value) <= float(bound)


READS = {
    "production_min_s": ("carrier_duration_s", "at_least"),
    "voiced_fraction_min": ("carrier_voiced_fraction", "at_least"),
    "f0_spread_max_semitones": ("carrier_f0_spread_semitones", "at_most"),
    "continuity_min": ("carrier_continuity", "at_least"),
    "dominant_segment_min_fraction": ("sweep_dominant_fraction", "at_least"),
    "monotone_tolerance_semitones": ("sweep_monotone_reversal_semitones", "at_most"),
    "expected_tokens_matched_min": ("expected_tokens_matched", "at_least"),
    "omissions_max": ("expected_tokens_omitted", "at_most"),
    "response_min_s": ("response_duration_s", "at_least"),
    "coverage_min": ("source_content_coverage", "at_least"),
    "items_min": ("items_produced", "at_least"),
    "events_min": ("airway_events_found", "at_least"),
    "repetitions_min": ("ddk_repetitions_found", "at_least"),
}
"""What each gateable conformance gate reads and which way it compares."""

READINGS = (*sorted({reading for reading, _ in READS.values()}), "verbatim_overlap_fraction")
"""Every reading a conformance gate is bound to, plus the recall's own anti-pattern reading."""

_READING_LINE = re.compile("|".join(f'"{name}"' for name in READINGS))


def fold(names: list[str], bounds: dict[str, Any], readings: dict[str, Any]) -> tuple[Any, list[str]]:
    """Apply a group's conformance gates, under ``nodes.gates.apply_gates``'s own rule.

    Args:
        names: The gates the group declares, in order.
        bounds: The bounds the run resolved for this recording.
        readings: The readings the reporting node wrote.

    Returns:
        The conformance and the gate names actually applied.
    """
    applied: list[str] = []
    outcomes: list[Any] = []
    for name in names:
        if name not in bounds:
            continue
        reading, op = READS[name]
        applied.append(name)
        outcomes.append(passes(readings.get(reading), bounds[name], op))
    if not applied or UNDETERMINED in outcomes:
        return UNDETERMINED, applied
    return all(outcomes), applied


def row(run_root: Path, table: dict[str, Any]) -> dict[str, Any]:
    """One recording's before-and-after row.

    Args:
        run_root: The recording's own output directory.
        table: The family table from ``family_gates.py``.

    Returns:
        What to write for this recording.
    """
    out: dict[str, Any] = {"stem": run_root.name, "error": None}
    try:
        summary = json.loads((run_root / "summary" / "summary.json").read_text())
        verdict = summary["verdict"]
        decisions = summary["decisions"]
        family = verdict.get("declared_family")
        gates = verdict.get("gates") or {}
        bounds = gates.get("bounds") or {}
        entry = table.get(family or "")
        readings = store_readings(run_root / "run" / "store.jsonl")
        recorded = [gate["gate"] for gate in gates.get("applied") or []]
        before_conformance = (verdict.get("conformance") or {}).get(gates.get("node"))
        flags = decisions.get("flags") or []
        out.update(
            {
                "family": family,
                "node": gates.get("node"),
                "recorded_applied": recorded,
                "recorded_conformance": before_conformance,
                "file_triage": decisions.get("file_triage"),
                "flags_n": len(flags),
                "conformance_flag": any(CONFORMANCE_WHY in str(flag.get("why")) for flag in flags),
                "readings": {name: readings.get(name) for name in READINGS},
                "bounds": {name: bounds.get(name) for name in ("response_min_s", "coverage_min")},
            }
        )
        if entry is None:
            out["replayed"] = None
            return out
        before, before_applied = fold(entry["old"], bounds, readings)
        after, after_applied = fold(entry["new"], bounds, readings)
        out.update(
            {
                "replayed": True,
                "before_applied": before_applied,
                "after_applied": after_applied,
                "before": before,
                "after": after,
                "replay_agrees": before_applied == recorded and before == before_conformance,
            }
        )
    except Exception:  # noqa: BLE001 — a recording that cannot be read is data, not a crash
        out["error"] = traceback.format_exc(limit=2)
    return out


def main() -> None:
    """Scan this array task's slice of the run tree."""
    root = Path(os.environ["SCAN_ROOT"])
    table = json.loads(Path(os.environ["SCAN_TABLE"]).read_text())
    out_dir = Path(os.environ["SCAN_OUT"])
    index = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    count = int(os.environ.get("SLURM_ARRAY_TASK_COUNT", "1"))
    runs = sorted(path for path in root.glob("sub-*/ses-*/*") if path.is_dir())
    mine = runs[index::count]
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"scan-{index:04d}.jsonl"
    with target.open("w") as handle:
        for position, run_root in enumerate(mine):
            handle.write(json.dumps(row(run_root, table)) + "\n")
            if position % 200 == 0:
                handle.flush()
                print(f"{position}/{len(mine)}", flush=True)
    print(f"wrote {target} ({len(mine)} rows)", flush=True)


if __name__ == "__main__":
    main()
