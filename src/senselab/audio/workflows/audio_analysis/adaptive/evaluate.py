"""Ground-truth evaluation against a Label Studio export (tasks.md T034).

Consumes the LS JSON export format (list of tasks, ``annotations[].result``
with paired ``labels``/``textarea`` items sharing region ids) and scores:

- **speech_presence**: bucket-level accuracy/precision/recall of ``speech_presence_confidence ≥ 0.5``
  against labeled speech spans; mean uncertainty inside vs outside speech.
- **transcript**: WER of the fused consensus (and each contributing model)
  against the concatenated GT texts, computed only over words whose midpoint
  falls inside a *transcribed* GT segment (untranscribed GT spans are excluded
  from both sides — the annotator's own uncertainty is not a reference).
- **diarization**: greedy cluster↔GT-speaker mapping by time overlap +
  speaker-attribution accuracy over fused words; speaker-count comparison.
- **boundary/uncertainty checks**: speaker uncertainty at GT speaker
  boundaries vs within segments; asr uncertainty + fused word confidence
  inside the untranscribed GT span vs elsewhere (the region a human could not
  transcribe should be where the pipeline is least certain).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from senselab.audio.workflows.audio_analysis.aggregate import _normalize_transcript_for_wer
from senselab.audio.workflows.audio_analysis.harvesters import _levenshtein

# ``final_dir`` alone. Every other layout helper this module imported named a place in the belief
# tree, and EVAL consumes the deliverable: the shortest statement of that rule is an import list
# with nowhere else in it.
from senselab.audio.workflows.audio_analysis.layout import final_dir

_TOKEN_EQUIV = {"u": "you"}  # annotator shorthand normalization, reported separately


# LS export *parsing* lives next to the export builders (architecture-review
# T049); re-exported here for existing callers.
from senselab.audio.workflows.audio_analysis.labelstudio import (  # noqa: E402, F401
    load_ls_ground_truth,
)


def _tokens(text: str, *, equiv: bool) -> list[str]:
    toks = _normalize_transcript_for_wer(text or "").split()
    return [_TOKEN_EQUIV.get(t, t) for t in toks] if equiv else toks


def _wer(ref: list[str], hyp: list[str]) -> float | None:
    """WER over token lists — jiwer-backed when available (T049), Levenshtein fallback."""
    if not ref:
        return None
    try:
        from senselab.audio.tasks.speech_to_text_evaluation import calculate_wer  # noqa: PLC0415

        return round(float(calculate_wer(" ".join(ref), " ".join(hyp))), 4)
    except (ImportError, ModuleNotFoundError):
        return round(_levenshtein(ref, hyp) / len(ref), 4)


def _in_any(mid: float, spans: list[tuple[float, float]]) -> bool:
    return any(s <= mid < e for s, e in spans)


def evaluate_against_ground_truth(
    *,
    out_dir: Path,
    gt_path: Path,
    word_streams: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Score final outputs in ``out_dir`` against the LS ground truth; write eval.json."""
    gt = load_ls_ground_truth(gt_path)
    # The evaluator scores the deliverable and nothing else. Every read here is of ``final/``,
    # which is what makes it a consumer of the answer rather than a stage that builds it — it
    # used to reach into ``L2/`` for the presence track, the baseline round's uncertainty mass
    # and the last round's speaker axis, and each of those was a scorer scoring an intermediate.
    final = final_dir(out_dir)
    transcript = json.loads((final / "transcript.json").read_text())
    diarization = json.loads((final / "diarization.json").read_text())
    import pandas as pd

    estimates = final / "estimates"
    speech_presence = pd.read_parquet(estimates / "speech_presence.parquet")

    speech_spans = [(s["start"], s["end"]) for s in gt["segments"]]
    transcribed = [(s["start"], s["end"]) for s in gt["segments"] if s["text"]]
    untranscribed = [(s["start"], s["end"]) for s in gt["segments"] if not s["text"]]

    # ── speech_presence ─────────────────────────────────────────────────────────
    tp = fp = fn = tn = 0
    unc_speech: list[float] = []
    unc_sil: list[float] = []
    for _, row in speech_presence.iterrows():
        mid = (float(row["start"]) + float(row["end"])) / 2.0
        pv = row.get("speech_presence_confidence")
        if pv is None or pv != pv:
            continue
        gt_speech = _in_any(mid, speech_spans)
        pred = float(pv) >= 0.5
        tp += gt_speech and pred
        fp += (not gt_speech) and pred
        fn += gt_speech and (not pred)
        tn += (not gt_speech) and (not pred)
        u = row.get("uncertainty")
        if u is not None and u == u:
            (unc_speech if gt_speech else unc_sil).append(float(u))
    speech_presence_eval = {
        "buckets": tp + fp + fn + tn,
        "accuracy": round((tp + tn) / max(1, tp + fp + fn + tn), 4),
        "precision": round(tp / max(1, tp + fp), 4),
        "recall": round(tp / max(1, tp + fn), 4),
        "mean_uncertainty_in_gt_speech": round(sum(unc_speech) / len(unc_speech), 4) if unc_speech else None,
        "mean_uncertainty_in_gt_silence": round(sum(unc_sil) / len(unc_sil), 4) if unc_sil else None,
    }

    # ── transcript (fused + per model) ──────────────────────────────────
    ref_tokens = [t for s in gt["segments"] if s["text"] for t in _tokens(s["text"], equiv=True)]

    def _score_words(words: list[dict[str, Any]]) -> dict[str, Any]:
        hyp_words = [w for w in words if _in_any((w["start"] + w["end"]) / 2.0, transcribed)]
        hyp_plain = [t for w in hyp_words for t in _tokens(w["text"], equiv=False)]
        hyp_equiv = [t for w in hyp_words for t in _tokens(w["text"], equiv=True)]
        ref_plain = [t for s in gt["segments"] if s["text"] for t in _tokens(s["text"], equiv=False)]
        return {
            "n_words_scored": len(hyp_equiv),
            "wer": _wer(ref_plain, hyp_plain),
            "wer_normalized": _wer(ref_tokens, hyp_equiv),
        }

    transcript_eval: dict[str, Any] = {"fused": _score_words(transcript["words"])}
    if word_streams:
        transcript_eval["per_model"] = {m: _score_words(ws) for m, ws in sorted(word_streams.items())}

    # ── diarization ──────────────────────────────────────────────────────
    overlap: dict[tuple[str, str], float] = {}
    for seg in diarization.get("segments") or []:
        for g in gt["segments"]:
            ov = min(seg["end"], g["end"]) - max(seg["start"], g["start"])
            if ov > 0 and g["speaker"]:
                key = (seg["cluster_id"], g["speaker"])
                overlap[key] = overlap.get(key, 0.0) + ov
    mapping: dict[str, str] = {}
    used_gt: set[str] = set()
    for (cid, spk), _ov in sorted(overlap.items(), key=lambda kv: -kv[1]):
        if cid not in mapping and spk not in used_gt:
            mapping[cid] = spk
            used_gt.add(spk)
    attributed = correct = 0
    for w in transcript["words"]:
        mid = (w["start"] + w["end"]) / 2.0
        gt_spk = next((s["speaker"] for s in gt["segments"] if s["start"] <= mid < s["end"]), None)
        if gt_spk is None or w.get("speaker") is None:
            continue
        attributed += 1
        correct += mapping.get(w["speaker"]) == gt_spk
    # Boundary F1: predicted segment starts vs GT starts (±tol), excluding t=0.
    tol = 0.25
    gt_bounds = [g["start"] for g in gt["segments"][1:]]
    pred_bounds = sorted({round(s["start"], 3) for s in (diarization.get("segments") or [])[1:]})
    b_tp = sum(1 for b in gt_bounds if any(abs(b - p) <= tol for p in pred_bounds))
    b_prec = b_tp / len(pred_bounds) if pred_bounds else None
    b_rec = b_tp / len(gt_bounds) if gt_bounds else None
    b_f1 = (
        round(2 * b_prec * b_rec / (b_prec + b_rec), 4)
        if b_prec and b_rec and (b_prec + b_rec) > 0
        else (0.0 if pred_bounds or gt_bounds else None)
    )
    diarization_eval = {
        "gt_speakers": sorted({s["speaker"] for s in gt["segments"] if s["speaker"]}),
        "predicted_clusters": [c["cluster_id"] for c in diarization.get("clusters") or []],
        "refined": diarization.get("refined", False),
        "cluster_to_gt": mapping,
        "word_speaker_accuracy": round(correct / attributed, 4) if attributed else None,
        "n_words_attributed": attributed,
        "boundary_f1": b_f1,
        "boundary_precision": round(b_prec, 4) if b_prec is not None else None,
        "boundary_recall": round(b_rec, 4) if b_rec is not None else None,
        "n_pred_speakers": len(diarization.get("clusters") or []),
        "n_gt_speakers": len({s["speaker"] for s in gt["segments"] if s["speaker"]}),
    }

    # ── uncertainty localization checks ──────────────────────────────────
    def _mean_conf(spans: list[tuple[float, float]]) -> float | None:
        vals = [w["confidence"] for w in transcript["words"] if _in_any((w["start"] + w["end"]) / 2.0, spans)]
        return round(sum(vals) / len(vals), 4) if vals else None

    # The trajectory, from the deliverable that now carries it. This read the baseline round's
    # ``summary.json`` out of the belief tree, which is a scorer reconstructing the run's history
    # from an intermediate; ``final/decisions.json`` is the run's own account of it.
    decisions = json.loads((final / "decisions.json").read_text())
    trajectory = (decisions.get("convergence") or {}).get("rounds") or []
    localization = {
        "untranscribed_gt_spans": untranscribed,
        "fused_confidence_in_untranscribed": _mean_conf(untranscribed),
        "fused_confidence_in_transcribed": _mean_conf(transcribed),
        # The mass the *first* recorded round entered with. ``None`` when the run recorded no
        # round, which is a missing measurement and not a mass of zero.
        "baseline_uncertainty_mass": (trajectory[0].get("uncertainty_mass") or {}).get("before")
        if trajectory
        else None,
    }
    # Identity uncertainty at GT speaker boundaries vs inside segments.
    try:
        import pandas as pd  # noqa: PLC0415

        # One row per bucket, folded across perturbations by the writer. The filter this replaces
        # took the transcript's stream, which scored the run against whichever perturbation the
        # transcript came from rather than against the belief the run published — and it read the
        # last round's estimate out of ``L2/`` rather than the deliverable extracted from it.
        ident = pd.read_parquet(estimates / "speaker.parquet")
        boundaries = [g["start"] for g in gt["segments"][1:]]
        at_b: list[float] = []
        inside: list[float] = []
        for _, row in ident.iterrows():
            u = row.get("uncertainty")
            if u is None or u != u:
                continue
            if any(row["start"] <= b < row["end"] for b in boundaries):
                at_b.append(float(u))
            elif _in_any((row["start"] + row["end"]) / 2.0, speech_spans):
                inside.append(float(u))
        localization["speaker_uncertainty_at_gt_boundaries"] = round(sum(at_b) / len(at_b), 4) if at_b else None
        localization["speaker_uncertainty_within_segments"] = round(sum(inside) / len(inside), 4) if inside else None
    except (OSError, ValueError):
        pass

    eval_doc = {
        "ground_truth": str(gt_path),
        "gt_segments": gt["segments"],
        "speech_presence": speech_presence_eval,
        "transcript": transcript_eval,
        "diarization": diarization_eval,
        "localization": localization,
    }
    (final / "eval.json").write_text(json.dumps(eval_doc, indent=2, default=str))
    return eval_doc
