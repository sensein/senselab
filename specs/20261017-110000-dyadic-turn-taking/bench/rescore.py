"""Recompute every measure from the frames ``evaluate_inputs.py`` saved, with the current code.

Reads ``OUT_DIR/seed*_<input>.csv.gz`` and ``OUT_DIR/report.json`` (for timings), re-runs
``summarize_projection`` and the clean-vs-input comparison, and writes ``OUT_DIR/rescored.json``
plus a per-input mean table on stdout. No model runs.

Usage: uv run python rescore.py DATA_DIR OUT_DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from evaluate_inputs import compare, lines

from senselab.audio.tasks.turn_taking_projection import summarize_projection
from senselab.audio.tasks.turn_taking_projection.result import VAPResult


def load(path: Path) -> VAPResult:
    """A saved frame table as a VAPResult."""
    return VAPResult(frame_rate=10.0, speakers=("A", "B"), frames=pd.read_csv(path))


def main() -> None:
    """CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("out")
    args = parser.parse_args()
    out = Path(args.out)
    timings = json.loads((out / "report.json").read_text()) if (out / "report.json").is_file() else {}
    rescored: Dict[str, Dict[str, dict]] = {}
    for clean_csv in sorted(out.glob("seed*_clean.csv.gz")):
        seed = clean_csv.name.split("_")[0]
        truth = json.loads((Path(args.data) / f"dialogue_{seed}" / "truth.json").read_text())
        turns, ipus, bcs = lines(truth["turns"]), lines(truth["ipus"]), lines(truth["backchannels"])
        clean = load(clean_csv)
        rows: Dict[str, dict] = {}
        for csv in sorted(out.glob(f"{seed}_*.csv.gz")):
            name = csv.name[len(seed) + 1 : -len(".csv.gz")]
            vap = load(csv)
            summary = summarize_projection(vap, turns, ipus, bcs)
            row = {
                "holdshift": summary["overall"]["pauses"],
                "shifts": {k: summary["overall"][k] for k in ("n_shifts", "mean_p_now_next", "projected_rate")},
                "backchannel": {s: summary["speakers"][s]["backchannel"] for s in ("A", "B")},
                "timing": timings.get(seed, {}).get("inputs", {}).get(name.replace("_", "/", 1), {}).get("timing"),
            }
            if name != "clean":
                row["vs_clean"] = compare(clean, vap)
            rows[name] = row
        rescored[seed] = rows
    (out / "rescored.json").write_text(json.dumps(rescored, indent=1, default=float))

    table: Dict[str, Dict[str, List[float]]] = {}
    for rows in rescored.values():
        for name, row in rows.items():
            t = table.setdefault(name, {})
            vs = row.get("vs_clean", {})
            for key, value in [
                ("p_now_corr", vs.get("p_now_corr")),
                ("p_now_mae", vs.get("p_now_mae")),
                ("vad_corr", np.mean([vs["vad_a_corr"], vs["vad_b_corr"]]) if vs else None),
                ("p_bc_corr", np.mean([vs["p_bc_a_corr"], vs["p_bc_b_corr"]]) if vs else None),
                ("shift_event_f1", vs.get("shift_event_f1")),
                ("hs_bal_acc", row["holdshift"]["balanced_accuracy"]),
                ("hs_auc", row["holdshift"]["auc"]),
                ("shift_p_next", row["shifts"]["mean_p_now_next"]),
                ("bc_resp_A", row["backchannel"]["A"]["responsiveness"]),
                ("bc_resp_B", row["backchannel"]["B"]["responsiveness"]),
            ]:
                if value is not None:
                    t.setdefault(key, []).append(float(value))
    frame = pd.DataFrame({name: {k: np.mean(v) for k, v in cols.items()} for name, cols in table.items()}).T
    print(frame.round(3).to_string())


if __name__ == "__main__":
    main()
