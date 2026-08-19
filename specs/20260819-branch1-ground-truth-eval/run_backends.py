"""Run YAMNet and HeAR's event detector over every variant and store the full score matrices.

Both backends get every variant in one call, so one subprocess venv start covers all of them.
Nothing is thresholded here: the full posterior matrix is stored so the sweep in score.py can
choose operating points without re-running a model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import io_util
from senselab.audio.data_structures import Audio

RAW = Path(__file__).parent / "raw"
HEAR_HOP = 0.25


def _variants() -> List[Dict[str, Any]]:
    meta = json.loads((RAW / "degradations.json").read_text())
    return list(meta["variants"])


def _matrix(windows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Turn a backend's per-window label_scores into a dense [n_windows, n_labels] matrix."""
    labels: List[str] = []
    seen = set()
    for win in windows:
        for entry in win["label_scores"]:
            for name in entry:
                if name not in seen:
                    seen.add(name)
                    labels.append(name)
    index = {name: i for i, name in enumerate(labels)}
    scores = np.zeros((len(windows), len(labels)), dtype=np.float32)
    for row, win in enumerate(windows):
        for entry in win["label_scores"]:
            for name, value in entry.items():
                scores[row, index[name]] = value
    return {
        "labels": labels,
        "scores": scores.tolist(),
        "starts": [float(w["start"]) for w in windows],
        "ends": [float(w["end"]) for w in windows],
        "win_length": float(windows[0]["win_length"]),
        "hop_length": float(windows[0]["hop_length"]),
    }


def run_yamnet() -> None:
    """Every variant through YAMNet, all 521 AudioSet posteriors kept."""
    from senselab.audio.tasks.classification.yamnet import YAMNetClassifier

    variants = _variants()
    audios = [Audio(filepath=v["path"]) for v in variants]
    results = YAMNetClassifier.classify_with_yamnet(audios=audios, top_k=521)
    out = {v["name"]: _matrix(windows) for v, windows in zip(variants, results)}
    io_util.dump(out, RAW / "yamnet.json.gz")
    first = out[variants[0]["name"]]
    print(f"yamnet: {len(out)} variants, {len(first['starts'])} windows, {len(first['labels'])} labels")


def run_hear() -> None:
    """Every variant through HeAR's 8-label event detector at a 0.25 s hop."""
    from senselab.audio.tasks.health_acoustics import detect_health_acoustic_events

    variants = _variants()
    audios = [Audio(filepath=v["path"]) for v in variants]
    results = detect_health_acoustic_events(audios=audios, hop_length=HEAR_HOP, top_k=None)
    out = {v["name"]: _matrix(windows) for v, windows in zip(variants, results)}
    io_util.dump(out, RAW / "hear.json.gz")
    first = out[variants[0]["name"]]
    print(f"hear: {len(out)} variants, {len(first['starts'])} windows, {len(first['labels'])} labels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["yamnet", "hear", "both"])
    args = parser.parse_args()
    if args.backend in ("yamnet", "both"):
        run_yamnet()
    if args.backend in ("hear", "both"):
        run_hear()
