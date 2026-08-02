"""Global single-scalar uncertainty aggregator across all four claims.

Per the workflow's bottom-line goal: produce a single ``[0, 1]`` uncertainty
score that grades whether the audio satisfies four claims simultaneously:

1. **Accurate transcript** — asr axis aggregated over time, plus ASR
   cross-model agreement, minus ASR hallucination penalties.
2. **Single speaker** — ``n_speakers`` from the embedding-derived diar source
   (0 = no speech, 1 = single speaker, ≥2 = multiple), plus speaker-axis
   stability across speech buckets.
3. **High quality** — torchaudio_squim PESQ / STOI / SI-SDR aggregate, plus
   acoustic-feature SNR proxies.
4. **No PII** — boolean from the PII detection module.

Each criterion produces an uncertainty in ``[0, 1]``. The combined scalar is
``max(...)`` over the four — the worst violation drives the bottom-line, and
all four must be low for the audio to read as "compliant on all claims".
Per-criterion sub-scores are exposed alongside so the consumer can drill in.

Hallucination detection
-----------------------

A bucket counts as a likely ASR hallucination when the model's
``no_speech_prob`` is high (≥ 0.5) but the transcript window contains tokens.
This catches Whisper's well-known habit of generating boilerplate
("Thanks for watching!") over silence. The hallucination rate per pass
inflates ``transcript_accuracy_uncertainty``.

n_speakers semantics (per the user's clarification)
---------------------------------------------------

- ``n_speakers == 0`` → recording without anyone speaking. The "single speaker"
  claim is vacuously violated (no speaker exists), so ``single_speaker_uncertainty
  = 1.0`` *unless* the workflow's caller explicitly says it expects empty
  recordings (not currently configurable; default = "speech expected").
- ``n_speakers == 1`` → single speaker confirmed → uncertainty 0.
- ``n_speakers >= 2`` → multi-speaker → uncertainty 1.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from senselab.audio.workflows.audio_analysis.harvesters import (
    seg_attr,
    whisper_chunk_confidence,
)
from senselab.audio.workflows.audio_analysis.types import FusedAxis

PASS_FOLD = "mean over the passes that reported"
"""How per-pass diagnostics are combined into one run-level number.

Named because it is a choice. It is deliberately *not* a minimum: raw and enhanced are the same
recording under a transform, so they are a perturbation sample whose disagreement is evidence —
picking the lower-uncertainty one and reporting it as the run's bottom line discards exactly the
information the second pass was run to obtain.
"""


def _mean_over_speech(
    axis_rows: Sequence[Mapping[str, Any]],
    presence_rows: Sequence[Mapping[str, Any]] | None = None,
) -> float | None:
    """Speech-weighted mean of the fused ``uncertainty`` over an axis's buckets.

    Weighted by the fused presence axis's ``confidence`` at each bucket, projected onto this
    axis's grid via the sanctioned coupling channel (``fuse.project_axis_onto``) rather than by a
    stored ``intensity_weight`` column. Re-deriving it each time is what lets the weighting move
    when the presence belief moves; the stored column froze one round's answer at harvest.

    Returns ``None`` when no bucket carries any weight.
    """
    from senselab.audio.workflows.audio_analysis.fuse import project_axis_onto

    rows = [r for r in axis_rows or () if isinstance(r.get("uncertainty"), (int, float))]
    if not rows:
        return None
    spans = [(float(r["start"]), float(r["end"])) for r in rows]
    confidence = project_axis_onto(
        [
            {"start": r["start"], "end": r["end"], "uncertainty": r["confidence"]}
            for r in presence_rows or ()
            if isinstance(r.get("confidence"), (int, float))
        ],
        spans,
    )
    weighted_sum = 0.0
    weight_total = 0.0
    for row, span in zip(rows, spans):
        # A bucket with no presence measurement is unweighted, not zero-weighted: absent is not
        # the same as "confidently silent", and treating it as silence would delete the bucket.
        w = float(confidence.get(span, 1.0))
        if w <= 0:
            continue
        weighted_sum += w * float(row["uncertainty"])
        weight_total += w
    if weight_total <= 0:
        return None
    return weighted_sum / weight_total


def _fold_over_passes(values: Sequence[float | None]) -> float | None:
    """Combine one per-pass diagnostic into a run-level number under :data:`PASS_FOLD`."""
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    return sum(numeric) / len(numeric) if numeric else None


def _detect_hallucinations(
    asr_resolved: dict[str, Any],
    duration_s: float,
    *,
    no_speech_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute per-pass ASR hallucination indicators.

    For each ASR model that exposes ``no_speech_prob`` (Whisper today), check
    every chunk: if ``no_speech_prob ≥ no_speech_threshold`` and the chunk
    contains text, count it as a likely hallucination. Returns a fraction of
    hallucinated time per ASR model and a pass-level mean.
    """
    per_model_hallu_seconds: dict[str, float] = {}
    per_model_total_text_seconds: dict[str, float] = {}
    for asr_model, resolved in asr_resolved.items():
        items = resolved if isinstance(resolved, list) else [resolved]
        hallu_s = 0.0
        text_s = 0.0
        for line in items:
            chunks = seg_attr(line, "chunks") or []
            for c in chunks:
                cs = seg_attr(c, "start")
                ce = seg_attr(c, "end")
                ct = seg_attr(c, "text") or ""
                if cs is None or ce is None:
                    continue
                dur = max(0.0, float(ce) - float(cs))
                if not ct.strip() or dur <= 0:
                    continue
                text_s += dur
                _, nsp = whisper_chunk_confidence(c)
                if nsp is not None and nsp >= no_speech_threshold:
                    hallu_s += dur
        if text_s > 0:
            per_model_hallu_seconds[asr_model] = hallu_s
            per_model_total_text_seconds[asr_model] = text_s
    rates = {
        m: per_model_hallu_seconds[m] / per_model_total_text_seconds[m]
        for m in per_model_hallu_seconds
        if per_model_total_text_seconds[m] > 0
    }
    pass_rate = float(np.mean(list(rates.values()))) if rates else None
    return {
        "per_model_rate": rates,
        "pass_hallucination_rate": pass_rate,
        "duration_s": duration_s,
    }


def _aggregate_quality(pass_summary: dict[str, Any]) -> dict[str, Any]:
    """Pull torchaudio_squim PESQ / STOI / SI-SDR + acoustic SNR proxies.

    Maps each metric to a [0, 1] uncertainty (lower quality → higher
    uncertainty) using literature-derived acceptance thresholds:

    - **PESQ** (1–4.5): clean speech > 3.5; degraded < 2.5. Uncertainty rises
      below 3.5, saturating below 2.0.
    - **STOI** (0–1): intelligibility. Above 0.85 = uncertainty 0; below 0.5
      saturates at 1.
    - **SI-SDR** (dB): clean speech > 15 dB; below 5 dB poor. Uncertainty
      rises below 15, saturates below 0.

    Combined via mean.
    """
    feat_block = pass_summary.get("features") or {}
    feat_result = feat_block.get("result") if isinstance(feat_block, dict) else None
    squim_rows = feat_result.get("torchaudio_squim", []) if isinstance(feat_result, dict) else []
    stoi_vals: list[float] = []
    pesq_vals: list[float] = []
    sisdr_vals: list[float] = []
    for r in squim_rows:
        if not isinstance(r, dict):
            continue
        for col, store in (("stoi", stoi_vals), ("pesq", pesq_vals), ("si_sdr", sisdr_vals)):
            v = r.get(col)
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(vf):
                store.append(vf)

    def ramp(value: float | None, low: float, high: float) -> float | None:
        """Linear ramp: ``value <= low`` → 1 (max uncertainty); ``>= high`` → 0."""
        if value is None:
            return None
        if value <= low:
            return 1.0
        if value >= high:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (value - low) / (high - low)))

    pesq_mean = float(np.mean(pesq_vals)) if pesq_vals else None
    stoi_mean = float(np.mean(stoi_vals)) if stoi_vals else None
    sisdr_mean = float(np.mean(sisdr_vals)) if sisdr_vals else None

    pesq_unc = ramp(pesq_mean, low=2.0, high=3.5)
    stoi_unc = ramp(stoi_mean, low=0.5, high=0.85)
    sisdr_unc = ramp(sisdr_mean, low=0.0, high=15.0)

    components = [u for u in (pesq_unc, stoi_unc, sisdr_unc) if u is not None]
    combined = float(np.mean(components)) if components else None
    return {
        "uncertainty": combined,
        "pesq_mean": pesq_mean,
        "stoi_mean": stoi_mean,
        "sisdr_mean": sisdr_mean,
        "pesq_uncertainty": pesq_unc,
        "stoi_uncertainty": stoi_unc,
        "sisdr_uncertainty": sisdr_unc,
    }


def compute_run_global_summary(
    *,
    fused_axes: Mapping[str, FusedAxis],
    passes: Mapping[str, dict[str, Any]],
    asr_resolved_by_pass: Mapping[str, dict[str, Any]],
    pii_reports: Mapping[str, Any],
    expects_speech: bool = True,
) -> dict[str, Any]:
    """Aggregate the fused axes plus the per-pass evidence into one run-level four-claim summary.

    One summary per **run**, not per pass. The axes are already folded across passes — an axis has
    no pass — so there is nothing per-pass left to summarise about them. The inputs that *are*
    genuinely per-pass (hallucination scan, squim PESQ/STOI/SI-SDR, PII spans, ``n_speakers``) stay
    per pass in ``by_pass`` and are folded under :data:`PASS_FOLD`.

    Args:
        fused_axes: ``{axis → FusedAxis}`` — the L2 answer.
        passes: ``{perturbation → pass_summary}``, for the per-pass quality and diarization blocks.
        asr_resolved_by_pass: ``{perturbation → {asr_model_id → resolved}}`` for the hallucination
            scan.
        pii_reports: ``{perturbation → PiiPassReport | None}``.
        expects_speech: When ``True`` (default), n_speakers=0 → uncertainty 1.0
            (no-speech recording violates the "single speaker" claim). Set
            ``False`` when the caller wants n=0 to count as compliant.

    Returns:
        Dict with the four sub-uncertainties plus a ``combined`` max() and per-criterion
        diagnostics, with the per-pass inputs kept visible under ``by_pass``.
    """
    presence_rows = fused_axes["speech_presence"].rows if "speech_presence" in fused_axes else []
    utt_mean = _mean_over_speech(fused_axes["asr"].rows if "asr" in fused_axes else [], presence_rows)
    identity_mean = _mean_over_speech(fused_axes["speaker"].rows if "speaker" in fused_axes else [], presence_rows)

    by_pass: dict[str, dict[str, Any]] = {}
    for perturbation, pass_summary in sorted(passes.items()):
        duration_s = float(pass_summary.get("duration_s", 0.0) or 0.0)
        hallu = _detect_hallucinations(dict(asr_resolved_by_pass.get(perturbation) or {}), duration_s)
        diar_blocks = (pass_summary.get("diarization") or {}).get("by_model") or {}
        n_speakers_pass: int | None = None
        for _m, block in diar_blocks.items():
            if isinstance(block, dict) and block.get("status") == "ok" and "n_speakers" in block:
                n_speakers_pass = int(block["n_speakers"])
                break
        by_pass[perturbation] = {
            "hallucination_rate": hallu.get("pass_hallucination_rate"),
            "hallucination_per_model": hallu.get("per_model_rate"),
            "n_speakers": n_speakers_pass,
            "quality": _aggregate_quality(pass_summary),
        }

    hallu_rate = _fold_over_passes([b["hallucination_rate"] for b in by_pass.values()])
    # Combine: asr time-mean (already in [0,1]) + hallucination rate
    # (also [0,1]). max() is the right combiner — either one indicates a
    # transcript problem.
    transcript_components: list[float] = []
    if utt_mean is not None:
        transcript_components.append(utt_mean)
    if hallu_rate is not None:
        transcript_components.append(hallu_rate)
    transcript_uncertainty: float | None = max(transcript_components) if transcript_components else None

    # ─── single_speaker ───
    counts = [b["n_speakers"] for b in by_pass.values() if b["n_speakers"] is not None]
    n_speakers: int | None = max(counts) if counts else None
    if n_speakers is None:
        single_speaker_uncertainty: float | None = None
    elif n_speakers == 1:
        single_speaker_uncertainty = 0.0
    elif n_speakers == 0:
        # No speakers detected. Whether this is "violation" depends on the
        # caller's expectation. Default ``expects_speech=True`` says "we
        # expected a speaker; absence violates the single-speaker claim".
        single_speaker_uncertainty = 1.0 if expects_speech else 0.0
    else:
        single_speaker_uncertainty = 1.0
    if single_speaker_uncertainty is not None and identity_mean is not None:
        # Even when n_speakers == 1, speaker uncertainty over time can flag
        # within-track inconsistencies. Combine via max so speaker drift still surfaces.
        single_speaker_uncertainty = max(single_speaker_uncertainty, identity_mean)

    # ─── quality ───
    quality_block = {
        key: _fold_over_passes([b["quality"].get(key) for b in by_pass.values()])
        for key in (
            "uncertainty",
            "pesq_mean",
            "stoi_mean",
            "sisdr_mean",
            "pesq_uncertainty",
            "stoi_uncertainty",
            "sisdr_uncertainty",
        )
    }

    # ─── no_pii ───
    pii_report = next((pii_reports.get(label) for label in sorted(pii_reports) if pii_reports.get(label)), None)
    # Surface the actual detected PII spans (text + category + detector + ASR
    # source + confidence) so the consumer can audit. The continuous
    # ``detection_confidence`` (per-span score × cross-detector agreement ×
    # cross-ASR agreement) drives the bottom-line uncertainty; the boolean
    # ``contains_pii`` and the span list let a reviewer decide whether each
    # detection is a true positive worth redacting. ``None`` propagation:
    # ``pii_report is None`` (PII stage skipped upstream) or
    # ``pii_report.detector_used is None`` (subprocess crashed / both
    # detectors failed to load / caller passed ``detectors=[]``) both
    # surface as ``no_pii_uncertainty = None`` — distinct from ``0.0``
    # ("ran, found nothing") so a downstream auditor can tell "didn't
    # check" from "checked clean".
    if pii_report is None:
        no_pii_uncertainty: float | None = None
        pii_block: dict[str, Any] | None = None
    elif pii_report.detector_used is None:
        no_pii_uncertainty = None
        pii_block = {
            "contains_pii": pii_report.contains_pii,
            "n_spans": pii_report.n_spans,
            "categories": pii_report.categories,
            "detector_used": None,
            "detection_confidence": None,
            "spans_by_category": {},
            "spans": [],
            "failures": pii_report.failures,
        }
    else:
        no_pii_uncertainty = pii_report.detection_confidence
        # Group spans by category for a quick at-a-glance view; full per-span
        # detail lives alongside.
        from collections import defaultdict

        spans_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for s in pii_report.spans:
            spans_by_category[s.category].append(
                {
                    "text": s.text,
                    "asr_model": s.asr_model,
                    "score": s.score,
                    "source": s.source,
                }
            )
        pii_block = {
            "contains_pii": pii_report.contains_pii,
            "n_spans": pii_report.n_spans,
            "categories": pii_report.categories,
            "detector_used": pii_report.detector_used,
            "detection_confidence": pii_report.detection_confidence,
            "spans_by_category": dict(spans_by_category),
            "spans": [
                {
                    "text": s.text,
                    "category": s.category,
                    "asr_model": s.asr_model,
                    "score": s.score,
                    "source": s.source,
                }
                for s in pii_report.spans
            ],
            "failures": pii_report.failures,
        }

    # ─── combined ───
    components = [
        c
        for c in (
            transcript_uncertainty,
            single_speaker_uncertainty,
            quality_block.get("uncertainty"),
            no_pii_uncertainty,
        )
        if c is not None
    ]
    combined = max(components) if components else None

    return {
        "combined_uncertainty": combined,
        "pass_fold": PASS_FOLD,
        "by_pass": by_pass,
        "transcript_accuracy": {
            "uncertainty": transcript_uncertainty,
            "asr_axis_mean": utt_mean,
            "hallucination_rate": hallu_rate,
        },
        "single_speaker": {
            "uncertainty": single_speaker_uncertainty,
            "n_speakers": n_speakers,
            "speaker_axis_mean": identity_mean,
            "expects_speech": expects_speech,
        },
        "quality": {
            "uncertainty": quality_block.get("uncertainty"),
            "pesq_mean": quality_block.get("pesq_mean"),
            "stoi_mean": quality_block.get("stoi_mean"),
            "sisdr_mean": quality_block.get("sisdr_mean"),
            "pesq_uncertainty": quality_block.get("pesq_uncertainty"),
            "stoi_uncertainty": quality_block.get("stoi_uncertainty"),
            "sisdr_uncertainty": quality_block.get("sisdr_uncertainty"),
        },
        "no_pii": {
            "uncertainty": no_pii_uncertainty,
            **(pii_block or {}),
        },
    }
