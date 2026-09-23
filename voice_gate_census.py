"""Census of voice-branch task extents and carrier rejections over a replayed triage corpus.

Read-only over the corpus. Walks <corpus>/sub-*/ses-*/<stem>/run/store.jsonl, counts, and writes
a JSON payload and a text summary into its own output directory. Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections import Counter
from pathlib import Path

TASKS = {
    "maximum-phonation-time": "task-maximum-phonation-time",
    "glides-low-to-high": "task-glides-low-to-high",
    "glides-high-to-low": "task-glides-high-to-low",
}
CARRIER_MIN_S = 0.5
FOCUS_GATES = ("f0_spread_max_semitones", "dominant_segment_min_fraction")


def scan_store(path: str) -> dict:
    """Parse one store.jsonl and return what this census needs from it.

    Args:
        path: Absolute path to a run's ``store.jsonl``.

    Returns:
        A dict with ``ok``, ``error``, ``has_task_extent``, ``n_task_extents`` and ``rejections``.
    """
    invalidated: set[str] = set()
    extents: list[tuple[str, dict]] = []
    rejects: list[tuple[str, dict]] = []
    try:
        with open(path, "r", errors="replace") as handle:
            for line in handle:
                if "task_extent" not in line and "carrier_rejected" not in line and "wasInvalidatedBy" not in line:
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                kind = record.get("record")
                if kind == "relation":
                    if record.get("relation") == "wasInvalidatedBy":
                        invalidated.add(str(record.get("source")))
                    continue
                if kind != "entity":
                    continue
                attributes = record.get("attributes") or {}
                eid = str(record.get("id"))
                ptype = record.get("prov_type")
                if ptype == "span" and attributes.get("family") == "voice" and attributes.get("role") == "task_extent":
                    extents.append((eid, attributes))
                elif ptype == "measurement" and attributes.get("name") == "carrier_rejected":
                    rejects.append((eid, attributes))
    except OSError as err:
        return {
            "ok": False,
            "error": f"{type(err).__name__}: {err}",
            "has_task_extent": False,
            "n_task_extents": 0,
            "rejections": [],
        }

    live_extents = [a for eid, a in extents if eid not in invalidated]
    live_rejects = []
    for eid, a in rejects:
        if eid in invalidated:
            continue
        carrier = a.get("carrier_s")
        carrier_f = float(carrier) if isinstance(carrier, (int, float)) else None
        value_read = a.get("value_read")
        value_f = float(value_read) if isinstance(value_read, (int, float)) else None
        bound = a.get("bound")
        live_rejects.append(
            {
                "gate": a.get("value"),
                "carrier_s": carrier_f,
                "value_read": value_f,
                "bound": float(bound) if isinstance(bound, (int, float)) else None,
                "signal": a.get("signal"),
            }
        )
    return {
        "ok": True,
        "error": None,
        "has_task_extent": bool(live_extents),
        "n_task_extents": len(live_extents),
        "rejections": live_rejects,
    }


def worker(item: tuple[str, str]) -> tuple[str, str, dict]:
    """Scan one store for one task family.

    Args:
        item: ``(task family key, store path)``.

    Returns:
        ``(task family key, store path, scan result)``.
    """
    family, path = item
    return family, path, scan_store(path)


def quantiles(values: list[float]) -> dict:
    """Median, p90, min and max of a list of numbers.

    Args:
        values: The numbers.

    Returns:
        A dict with ``n``, ``min``, ``median``, ``p90``, ``max``; Nones when empty.
    """
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return {"n": 0, "min": None, "median": None, "p90": None, "max": None}

    def pick(q: float) -> float:
        idx = min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1)))))
        return vals[idx]

    return {"n": len(vals), "min": vals[0], "median": pick(0.5), "p90": pick(0.9), "max": vals[-1]}


def discover(corpus: Path) -> list[tuple[str, str]]:
    """Find every target run's store file.

    Args:
        corpus: The ``out/`` tree holding ``sub-*/ses-*/<stem>/``.

    Returns:
        ``(task family key, store path)`` pairs.
    """
    found: list[tuple[str, str]] = []
    with os.scandir(corpus) as subs:
        sub_dirs = [e.path for e in subs if e.is_dir() and e.name.startswith("sub-")]
    for sub in sorted(sub_dirs):
        try:
            with os.scandir(sub) as sessions:
                ses_dirs = [e.path for e in sessions if e.is_dir()]
        except OSError:
            continue
        for ses in sorted(ses_dirs):
            try:
                with os.scandir(ses) as runs:
                    run_dirs = [(e.name, e.path) for e in runs if e.is_dir()]
            except OSError:
                continue
            for name, path in sorted(run_dirs):
                for key, token in TASKS.items():
                    if token in name:
                        found.append((key, os.path.join(path, "run", "store.jsonl")))
                        break
    return found


def main() -> None:
    """Run the census and write results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    corpus = Path(args.corpus)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = discover(corpus)
    print(f"discovered {len(targets)} target runs under {corpus}", flush=True)

    totals: dict[str, dict] = {
        key: {
            "n_recordings": 0,
            "n_store_missing": 0,
            "n_store_error": 0,
            "n_with_task_extent": 0,
            "n_no_task_extent": 0,
            "no_extent_with_long_rejection": 0,
            "no_extent_without_long_rejection": 0,
            "gate_recording_counts": Counter(),
            "gate_event_counts": Counter(),
            "gate_recording_counts_all_carriers": Counter(),
            "signal_event_counts": Counter(),
            "errors": [],
        }
        for key in TASKS
    }
    focus: dict[str, dict] = {key: {g: {"carrier_s": [], "value_read": []} for g in FOCUS_GATES} for key in TASKS}
    focus_all: dict[str, dict] = {key: {g: {"carrier_s": [], "value_read": []} for g in FOCUS_GATES} for key in TASKS}

    done = 0
    with mp.Pool(processes=args.workers) as pool:
        for family, path, result in pool.imap_unordered(worker, targets, chunksize=8):
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(targets)}", flush=True)
            bucket = totals[family]
            bucket["n_recordings"] += 1
            if not result["ok"]:
                if "FileNotFoundError" in (result["error"] or ""):
                    bucket["n_store_missing"] += 1
                else:
                    bucket["n_store_error"] += 1
                if len(bucket["errors"]) < 20:
                    bucket["errors"].append({"path": path, "error": result["error"]})
                continue
            rejections = result["rejections"]
            long_rejections = [r for r in rejections if r["carrier_s"] is not None and r["carrier_s"] >= CARRIER_MIN_S]
            if result["has_task_extent"]:
                bucket["n_with_task_extent"] += 1
                continue
            bucket["n_no_task_extent"] += 1
            if long_rejections:
                bucket["no_extent_with_long_rejection"] += 1
            else:
                bucket["no_extent_without_long_rejection"] += 1
            for gate in {r["gate"] for r in long_rejections}:
                bucket["gate_recording_counts"][str(gate)] += 1
            for gate in {r["gate"] for r in rejections}:
                bucket["gate_recording_counts_all_carriers"][str(gate)] += 1
            for r in long_rejections:
                bucket["gate_event_counts"][str(r["gate"])] += 1
                bucket["signal_event_counts"][str(r["signal"])] += 1
                if r["gate"] in FOCUS_GATES:
                    focus[family][r["gate"]]["carrier_s"].append(r["carrier_s"])
                    if r["value_read"] is not None:
                        focus[family][r["gate"]]["value_read"].append(r["value_read"])
            for r in rejections:
                if r["gate"] in FOCUS_GATES:
                    if r["carrier_s"] is not None:
                        focus_all[family][r["gate"]]["carrier_s"].append(r["carrier_s"])
                    if r["value_read"] is not None:
                        focus_all[family][r["gate"]]["value_read"].append(r["value_read"])

    payload: dict = {
        "corpus": str(corpus),
        "carrier_min_s": CARRIER_MIN_S,
        "definition": {
            "live": "entity carrying no wasInvalidatedBy relation with it as source",
            "task_extent": "live span entity with attributes.family == 'voice' and attributes.role == 'task_extent'",
            "carrier_rejected": "live measurement entity with attributes.name == 'carrier_rejected'",
            "gate": "the carrier_rejected measurement's attributes.value",
        },
        "families": {},
    }
    for key, bucket in totals.items():
        n_no = bucket["n_no_task_extent"]
        n_rec = bucket["n_recordings"]
        payload["families"][key] = {
            "n_recordings": n_rec,
            "n_store_missing": bucket["n_store_missing"],
            "n_store_error": bucket["n_store_error"],
            "n_with_task_extent": bucket["n_with_task_extent"],
            "n_no_task_extent": n_no,
            "pct_no_task_extent": round(100.0 * n_no / n_rec, 2) if n_rec else None,
            "no_extent_with_long_rejection": bucket["no_extent_with_long_rejection"],
            "no_extent_without_long_rejection": bucket["no_extent_without_long_rejection"],
            "pct_of_no_extent_found_and_refused": (
                round(100.0 * bucket["no_extent_with_long_rejection"] / n_no, 2) if n_no else None
            ),
            "pct_of_no_extent_never_found": (
                round(100.0 * bucket["no_extent_without_long_rejection"] / n_no, 2) if n_no else None
            ),
            "gate_recording_counts_carrier_ge_0_5": dict(bucket["gate_recording_counts"].most_common()),
            "gate_event_counts_carrier_ge_0_5": dict(bucket["gate_event_counts"].most_common()),
            "gate_recording_counts_any_carrier": dict(bucket["gate_recording_counts_all_carriers"].most_common()),
            "signal_event_counts_carrier_ge_0_5": dict(bucket["signal_event_counts"].most_common()),
            "focus_gates_no_extent_carrier_ge_0_5": {
                g: {
                    "carrier_s": quantiles(focus[key][g]["carrier_s"]),
                    "value_read": quantiles(focus[key][g]["value_read"]),
                }
                for g in FOCUS_GATES
            },
            "focus_gates_no_extent_any_carrier": {
                g: {
                    "carrier_s": quantiles(focus_all[key][g]["carrier_s"]),
                    "value_read": quantiles(focus_all[key][g]["value_read"]),
                }
                for g in FOCUS_GATES
            },
            "errors_sample": bucket["errors"],
        }

    json_path = out_dir / "census.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    lines: list[str] = []
    lines.append(f"voice gate census over {corpus}")
    lines.append(f"carrier_s threshold: >= {CARRIER_MIN_S} s")
    lines.append("")
    for key in TASKS:
        f = payload["families"][key]
        lines.append(f"== {key} ==")
        lines.append(f"  recordings                      {f['n_recordings']}")
        lines.append(f"  store missing / unreadable      {f['n_store_missing']} / {f['n_store_error']}")
        lines.append(f"  minted a voice task_extent      {f['n_with_task_extent']}")
        lines.append(f"  no voice task_extent            {f['n_no_task_extent']} ({f['pct_no_task_extent']}%)")
        lines.append(
            f"    found and refused (>=0.5 s)   {f['no_extent_with_long_rejection']} "
            f"({f['pct_of_no_extent_found_and_refused']}% of no-extent)"
        )
        lines.append(
            f"    never found                   {f['no_extent_without_long_rejection']} "
            f"({f['pct_of_no_extent_never_found']}% of no-extent)"
        )
        lines.append("  gate breakdown (no-extent recordings with >=1 rejection at carrier_s >= 0.5):")
        for gate, count in f["gate_recording_counts_carrier_ge_0_5"].items():
            pct = round(100.0 * count / f["n_no_task_extent"], 2) if f["n_no_task_extent"] else None
            lines.append(f"    {gate:<36} {count:>7}  ({pct}% of no-extent)")
        lines.append("  gate breakdown (rejection events, carrier_s >= 0.5):")
        for gate, count in f["gate_event_counts_carrier_ge_0_5"].items():
            lines.append(f"    {gate:<36} {count:>7}")
        for gate in FOCUS_GATES:
            dist = f["focus_gates_no_extent_carrier_ge_0_5"][gate]
            lines.append(f"  {gate} (no-extent recordings, carrier_s >= 0.5):")
            lines.append(f"    carrier_s  {dist['carrier_s']}")
            lines.append(f"    value_read {dist['value_read']}")
        lines.append("")
    text_path = out_dir / "census.txt"
    text_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {text_path}", flush=True)


if __name__ == "__main__":
    main()
