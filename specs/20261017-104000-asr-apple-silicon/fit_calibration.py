"""Fit the accuracy-estimate calibration on AMI headset-mix recordings with references.

Usage: uv run python specs/20261017-104000-asr-apple-silicon/fit_calibration.py CALIB_DIR REFS_DIR OUT_JSON

CALIB_DIR holds ``<stem>.crisperwhisper.json`` / ``<stem>.qwen3asr.json`` from run_pair.py; REFS_DIR holds
``<meeting>_<mix>_refs.json`` (utterance start/end/text) from build_mix.py. Each recording is cut into windows
of about WINDOW_S seconds at points outside every reference utterance; for each window the script
measures what ``estimate_transcript_accuracy`` measures plus the primary's true WER, fits the linear
maps by word-weighted least squares, and reports leave-one-recording-out error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import jiwer
import numpy as np

from senselab.audio.tasks.speech_to_text_evaluation.accuracy_estimate import _features, _normalise, _words
from senselab.utils.data_structures import ScriptLine

WINDOW_S = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0
COVERAGE = 0.8


def boundaries(refs: list[dict], end: float) -> list[float]:
    """Window cut points about WINDOW_S apart, each moved past any utterance it would split."""
    cuts = [0.0]
    t = WINDOW_S
    while t < end:
        inside = [r["end"] for r in refs if r["start"] < t < r["end"]]
        while inside:
            t = max(inside) + 1e-3
            inside = [r["end"] for r in refs if r["start"] < t < r["end"]]
        cuts.append(t)
        t += WINDOW_S
    cuts.append(end + 1.0)
    return cuts


def in_span(s: float | None, e: float | None, a: float, b: float) -> bool:
    """Whether a word's midpoint lies in [a, b)."""
    return s is not None and e is not None and a <= (s + e) / 2 < b


def word_labels(prim_words: list, ref_tokens: list[str]) -> list[tuple[float, bool]]:
    """(confidence, correct) per normalised primary token, via the WER alignment."""
    toks, confs = [], []
    for text, _, _, score in prim_words:
        for tok in _normalise(text, True):
            toks.append(tok)
            confs.append(score)
    if not toks or not ref_tokens:
        return []
    out = jiwer.process_words(" ".join(ref_tokens), " ".join(toks))
    correct = [False] * len(toks)
    for chunk in out.alignments[0]:
        if chunk.type == "equal":
            for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                correct[j] = True
    return [(c, ok) for c, ok in zip(confs, correct) if c is not None]


def wls(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:  # noqa: N803
    """Weighted least squares coefficients."""
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    return coef


def main() -> None:
    """Fit and report."""
    calib, refs_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    rows, labels = [], []
    for prim_path in sorted(calib.glob("*.crisperwhisper.json")):
        stem = prim_path.name.split(".")[0]
        sec_path = calib / f"{stem}.qwen3asr.json"
        if not sec_path.exists():
            continue
        refs = json.loads((refs_dir / f"{'_'.join(stem.split('_')[:2])}_refs.json").read_text())
        prim = _words(ScriptLine.model_validate_json(prim_path.read_text()))
        sec = _words(ScriptLine.model_validate_json(sec_path.read_text()))
        end = max(r["end"] for r in refs)
        cuts = boundaries(refs, end)
        for a, b in zip(cuts[:-1], cuts[1:]):
            ref_tokens = [t for r in refs if a <= (r["start"] + r["end"]) / 2 < b for t in _normalise(r["text"], True)]
            p = [w for w in prim if in_span(w[1], w[2], a, b)]
            s = [w for w in sec if in_span(w[1], w[2], a, b)]
            feats = _features(p, s, True, 0.5)
            prim_tokens = [t for w in p for t in _normalise(w[0], True)]
            if len(ref_tokens) < 20 or feats["inter_system_disagreement"] is None:
                continue
            true_wer = jiwer.wer(" ".join(ref_tokens), " ".join(prim_tokens)) if prim_tokens else 1.0
            sec_tokens = [t for w in s for t in _normalise(w[0], True)]
            sec_wer = jiwer.wer(" ".join(ref_tokens), " ".join(sec_tokens)) if sec_tokens else 1.0
            rows.append(
                {
                    "recording": stem,
                    "start": a,
                    "end": b,
                    "n_ref": len(ref_tokens),
                    "true_wer": true_wer,
                    "secondary_true_wer": sec_wer,
                    **feats,
                }
            )
            labels.extend(word_labels(p, ref_tokens))

    n = np.array([r["n_primary_words"] for r in rows], float)
    nref = np.array([r["n_ref"] for r in rows], float)
    y = np.array([r["true_wer"] for r in rows])
    isw = np.array([r["inter_system_disagreement"] for r in rows])
    conf = np.array([r["mean_confidence"] if r["mean_confidence"] is not None else np.nan for r in rows])
    recs = np.array([r["recording"] for r in rows])
    keep = ~np.isnan(conf)
    rows = [r for r, k in zip(rows, keep) if k]
    n, nref, y, isw, conf, recs = n[keep], nref[keep], y[keep], isw[keep], conf[keep], recs[keep]

    # Low-confidence threshold: the highest confidence below which a primary word is wrong at least half the time.
    lab_c = np.array([c for c, _ in labels])
    lab_ok = np.array([ok for _, ok in labels])
    grid = np.round(np.arange(0.05, 1.0, 0.05), 2)
    err_below = {float(t): float(1 - lab_ok[lab_c < t].mean()) if (lab_c < t).sum() >= 30 else None for t in grid}
    threshold = max((t for t, e in err_below.items() if e is not None and e >= 0.5), default=0.5)
    bins = np.array([0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0001])
    word_bins = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (lab_c >= lo) & (lab_c < hi)
        word_bins.append(
            {
                "lo": float(lo),
                "hi": float(min(hi, 1.0)),
                "n": int(m.sum()),
                "error_rate": float(1 - lab_ok[m].mean()) if m.sum() else None,
            }
        )

    designs = {
        "with_confidence": np.column_stack([np.ones_like(isw), isw, 1 - conf]),
        "confidence_free": np.column_stack([np.ones_like(isw), isw]),
    }
    report: dict = {
        "window_s": WINDOW_S,
        "n_windows": len(rows),
        "recordings": sorted(set(recs.tolist())),
        "n_scored_words": int(len(labels)),
        "low_confidence_threshold": threshold,
        "error_rate_below_threshold": err_below,
        "word_error_by_confidence": word_bins,
        "models": {},
    }
    for name, X in designs.items():
        coef = wls(X, y, nref)
        pred = X @ coef
        z = np.abs(y - pred) * np.sqrt(n)
        q = float(np.quantile(z, COVERAGE))
        # Leave one recording out.
        loo_pred = np.empty_like(y)
        for rec in set(recs.tolist()):
            m = recs == rec
            c = wls(X[~m], y[~m], nref[~m])
            loo_pred[m] = X[m] @ c
        loo_z = np.abs(y - loo_pred) * np.sqrt(n)
        covered = float(np.mean(np.abs(y - np.maximum(loo_pred, 0)) <= q / np.sqrt(n)))
        # Whole-recording estimates: word-weighted mean of window predictions vs pooled truth.
        per_rec = {}
        for rec in sorted(set(recs.tolist())):
            m = recs == rec
            per_rec[rec] = {
                "true_wer_pooled": float(np.sum(y[m] * nref[m]) / np.sum(nref[m])),
                "loo_estimate": float(np.sum(np.maximum(loo_pred[m], 0) * nref[m]) / np.sum(nref[m])),
                "disagreement_pooled": float(np.sum(isw[m] * n[m]) / np.sum(n[m])),
                "secondary_true_wer_pooled": float(
                    np.sum(np.array([r["secondary_true_wer"] for r in rows])[m] * nref[m]) / np.sum(nref[m])
                ),
            }
        report["models"][name] = {
            "coef": [float(c) for c in coef],
            "residual_quantile": q,
            "fit_mae": float(np.average(np.abs(y - pred), weights=nref)),
            "fit_pearson_r": float(np.corrcoef(pred, y)[0, 1]),
            "loo_mae": float(np.average(np.abs(y - np.maximum(loo_pred, 0)), weights=nref)),
            "loo_pearson_r": float(np.corrcoef(loo_pred, y)[0, 1]),
            "loo_interval_coverage": covered,
            "loo_residual_quantile": float(np.quantile(loo_z, COVERAGE)),
            "per_recording": per_rec,
        }
    report["single_feature_pearson_r"] = {
        "inter_system_disagreement": float(np.corrcoef(isw, y)[0, 1]),
        "inter_system_wer": float(np.corrcoef([min(r["inter_system_wer"] or 0, 5) for r in rows], y)[0, 1]),
        "one_minus_mean_confidence": float(np.corrcoef(1 - conf, y)[0, 1]),
        "low_confidence_fraction": float(np.corrcoef([r["low_confidence_fraction"] or 0 for r in rows], y)[0, 1]),
    }
    report["true_wer_range"] = [float(y.min()), float(np.median(y)), float(y.max())]
    report["windows"] = rows
    out_path.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "windows"}, indent=1))


if __name__ == "__main__":
    main()
