"""Global single-scalar uncertainty aggregator across all four claims.

Per the workflow's bottom-line goal: produce a single ``[0, 1]`` uncertainty
score that grades whether the audio satisfies four claims simultaneously:

1. **Accurate transcript** — utterance axis aggregated over time, plus ASR
   cross-model agreement, minus ASR hallucination penalties.
2. **Single speaker** — ``n_speakers`` from the embedding-derived diar source
   (0 = no speech, 1 = single speaker, ≥2 = multiple), plus identity-axis
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

from typing import Any

import numpy as np

from senselab.audio.workflows.audio_analysis.harvesters import (
    seg_attr,
    whisper_chunk_confidence,
)
from senselab.audio.workflows.audio_analysis.types import AxisResult


def _mean_over_voice_buckets(
    rows: list[Any],
    presence_rows: list[Any] | None = None,
) -> float | None:
    """Intensity-weighted mean of ``aggregated_uncertainty`` over voice buckets.

    Uses the per-row ``intensity_weight`` (which already encodes "how much
    confident-voice content is in this bucket" — derived from the presence
    p_voice during ``compute_uncertainty_axes``) as the weight. Buckets in
    confident-silence regions have ``intensity_weight ≈ 0`` and contribute
    nothing to the mean; voice buckets contribute fully.

    The ``presence_rows`` argument is kept for backward compat / debugging
    but is no longer used — the intensity_weight is per-row and pre-computed.

    Returns ``None`` when total weight is 0 (e.g. all rows are confident silence).
    """
    if not rows:
        return None
    weighted_sum = 0.0
    weight_total = 0.0
    for r in rows:
        if r.aggregated_uncertainty is None:
            continue
        # Default to weight 1.0 when a row pre-dates the intensity_weight
        # field (older parquets) — avoids retroactively zeroing those.
        w = 1.0 if r.intensity_weight is None else float(r.intensity_weight)
        if w <= 0:
            continue
        weighted_sum += w * float(r.aggregated_uncertainty)
        weight_total += w
    if weight_total <= 0:
        return None
    return weighted_sum / weight_total


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


def compute_pass_global_summary(
    *,
    pass_label: str,
    pass_summary: dict[str, Any],
    axis_results: dict[tuple[str, Any], AxisResult],
    asr_resolved: dict[str, Any],
    pii_report: Any,  # noqa: ANN401 — PiiPassReport, optional
    expects_speech: bool = True,
) -> dict[str, Any]:
    """Aggregate one pass's per-bucket axes into the four-claim summary.

    Args:
        pass_label: ``"raw_16k"`` or ``"enhanced_16k"``.
        pass_summary: The pass's full per-task summary (used for quality + hallu).
        axis_results: ``{(pass_label, axis) → AxisResult}`` from compute.
        asr_resolved: ``{asr_model_id → resolved_asr_result}`` for hallucination scan.
        pii_report: Optional PiiPassReport for this pass; ``None`` to skip.
        expects_speech: When ``True`` (default), n_speakers=0 → uncertainty 1.0
            (no-speech recording violates the "single speaker" claim). Set
            ``False`` when the caller wants n=0 to count as compliant.

    Returns:
        Dict with the four sub-uncertainties plus a ``combined`` max() and
        per-criterion diagnostics.
    """
    duration_s = float(pass_summary.get("duration_s", 0.0) or 0.0)

    # ─── transcript_accuracy ───
    utt = axis_results.get((pass_label, "utterance"))
    presence = axis_results.get((pass_label, "presence"))
    utt_mean = _mean_over_voice_buckets(utt.rows, presence.rows if presence else []) if utt is not None else None
    hallu = _detect_hallucinations(asr_resolved, duration_s)
    hallu_rate = hallu.get("pass_hallucination_rate")
    # Combine: utterance time-mean (already in [0,1]) + hallucination rate
    # (also [0,1]). max() is the right combiner — either one indicates a
    # transcript problem.
    transcript_components: list[float] = []
    if utt_mean is not None:
        transcript_components.append(utt_mean)
    if hallu_rate is not None:
        transcript_components.append(hallu_rate)
    transcript_uncertainty: float | None = max(transcript_components) if transcript_components else None

    # ─── single_speaker ───
    diar_blocks = (pass_summary.get("diarization") or {}).get("by_model") or {}
    n_speakers: int | None = None
    for m, block in diar_blocks.items():
        if not (isinstance(block, dict) and block.get("status") == "ok"):
            continue
        if "n_speakers" in block:  # the synthetic embedding-derived diar carries it
            n_speakers = int(block["n_speakers"])
            break
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
    identity = axis_results.get((pass_label, "identity"))
    identity_mean = (
        _mean_over_voice_buckets(identity.rows, presence.rows if presence else []) if identity is not None else None
    )
    if single_speaker_uncertainty is not None and identity_mean is not None:
        # Even when n_speakers == 1, identity uncertainty over time can flag
        # within-track inconsistencies. Combine via max so identity drift on
        # a "single-speaker" pass still surfaces.
        single_speaker_uncertainty = max(single_speaker_uncertainty, identity_mean)

    # ─── quality ───
    quality_block = _aggregate_quality(pass_summary)

    # ─── no_pii ───
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
        "pass_label": pass_label,
        "combined_uncertainty": combined,
        "transcript_accuracy": {
            "uncertainty": transcript_uncertainty,
            "utterance_axis_mean": utt_mean,
            "hallucination_rate": hallu_rate,
            "hallucination_per_model": hallu.get("per_model_rate"),
        },
        "single_speaker": {
            "uncertainty": single_speaker_uncertainty,
            "n_speakers": n_speakers,
            "identity_axis_mean": identity_mean,
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
