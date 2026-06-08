"""Score an analyzed recording's windows against a speaker profile.

This is the comparison half of the workflow (the build half lives in
:mod:`build`). Given a profile's per-model centroids and calibration band, it
scores each short detection window of a recording and flags likely
**other-voice** regions:

- :func:`compare_recording_to_profile` — per-window consensus scoring on the
  short detection grid, speech-presence gating, and ``target`` /
  ``other_voice`` / ``unavailable`` flagging.
- :func:`leave_one_file_out_profile` — recompute the centroid excluding a
  contributing file's windows, so a recording that helped build the profile is
  not scored against a centroid that contains itself.
- :func:`within_file_holdout_profile` — the single-file fallback: exclude the
  windows near the window under test instead of a whole file.

The functions are pure: callers supply already-extracted embeddings (and, for
leave-one-out, the pooled source windows re-extracted from the shared cache).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from senselab.audio.workflows.audio_analysis.embeddings import (
    WindowEmbedding,
    calibrate_cosine_uncertainty,
    cos_dist,
)
from senselab.audio.workflows.speaker_profile import constants as C
from senselab.audio.workflows.speaker_profile.build import (
    AggregationResult,
    TaggedWindowEmbedding,
    aggregate_dominant_cluster,
)
from senselab.audio.workflows.speaker_profile.types import (
    ProfileComparisonResult,
    ProfileConfidence,
    RecordingOtherVoiceSummary,
    RecordingQualityIndicator,
)


def score_window(
    window_vectors: Mapping[str, np.ndarray],
    centroids: Mapping[str, Sequence[float]],
    calibration_band: Mapping[str, tuple[float, float]],
    *,
    fusion_weights: Mapping[str, float] | None = None,
) -> tuple[float | None, float | None, dict[str, float]]:
    """Score one window's per-model embeddings against the profile centroids.

    For each model present in both the window and the profile, the cosine
    distance to that model's centroid is mapped to a calibrated other-voice
    uncertainty via the model's own calibration band (so 192-D ECAPA / ResNet
    and 512-D WavLM are never compared directly). The per-model uncertainties
    are fused into a consensus.

    Args:
        window_vectors: ``{model_id -> embedding vector}`` for this time window.
        centroids: ``{model_id -> centroid vector}`` from the profile.
        calibration_band: ``{model_id -> (same_floor, diff_floor)}`` from the
            profile; the literature fallback band is used for any missing model.
        fusion_weights: Optional ``{model_id -> weight}`` for the consensus mean;
            ``None`` is an unweighted mean.

    Returns:
        ``(similarity, other_voice_uncertainty, per_model)`` where ``similarity``
        is ``1 - other_voice_uncertainty`` (calibrated, higher = more like the
        target), ``other_voice_uncertainty`` is the consensus, and ``per_model``
        maps each model to its pre-fusion uncertainty. All three are ``None`` /
        empty when no model overlaps between the window and the profile.
    """
    per_model: dict[str, float] = {}
    for model_id, centroid in centroids.items():
        vec = window_vectors.get(model_id)
        if vec is None:
            continue
        cdist = cos_dist(vec, np.asarray(centroid, dtype=np.float64))
        if cdist is None:
            continue
        same_floor, diff_floor = calibration_band.get(
            model_id, (C.SAME_SPEAKER_FLOOR_FALLBACK, C.DIFF_SPEAKER_FLOOR_FALLBACK)
        )
        per_model[model_id] = calibrate_cosine_uncertainty(
            cdist, same_speaker_floor=same_floor, diff_speaker_floor=diff_floor, direction="same"
        )

    if not per_model:
        return None, None, {}

    if fusion_weights:
        num = sum(per_model[m] * fusion_weights.get(m, 1.0) for m in per_model)
        den = sum(fusion_weights.get(m, 1.0) for m in per_model)
        consensus = float(num / den) if den > 0 else float(np.mean(list(per_model.values())))
    else:
        consensus = float(np.mean(list(per_model.values())))

    similarity = 1.0 - consensus
    return similarity, consensus, per_model


def compare_recording_to_profile(
    detection_windows: Mapping[str, list[WindowEmbedding]],
    centroids: Mapping[str, Sequence[float]],
    calibration_band: Mapping[str, tuple[float, float]],
    *,
    p_voice_by_window: Sequence[float | None] | None = None,
    voice_present_by_window: Sequence[bool] | None = None,
    other_voice_threshold: float | None = C.OTHER_VOICE_THRESHOLD_DEFAULT,
    min_p_voice: float = C.MIN_P_VOICE_FOR_COMPARISON,
    fusion_weights: Mapping[str, float] | None = C.CONSENSUS_FUSION_WEIGHTS_DEFAULT,
) -> list[ProfileComparisonResult]:
    """Score every detection window of a recording against the profile.

    Args:
        detection_windows: ``{model_id -> [WindowEmbedding]}`` on the short
            detection grid; all models share the same grid (as produced by
            ``extract_per_window_embeddings``).
        centroids: ``{model_id -> centroid vector}`` to score against — already
            leave-one-file-out-adjusted by the caller when the recording
            contributed to the profile.
        calibration_band: ``{model_id -> (same_floor, diff_floor)}`` for the same
            profile, used to calibrate each model's uncertainty.
        p_voice_by_window: Optional reused presence value per window index. A
            window below ``min_p_voice`` is scored ``unavailable`` (never
            flagged). ``None`` disables this gate.
        voice_present_by_window: Optional scene-derived per-window voice mask
            (speech / babble / conversation present, foreground OR background).
            When supplied it is the **authoritative** gate — a window with
            ``False`` is ``unavailable`` — so background/secondary voice is
            scored while cough/breath/silence are excluded; ``p_voice`` is then
            recorded for info but not used to gate. When ``None``, the
            ``p_voice`` gate above applies (legacy behavior). Both ``None`` →
            everything is scored.
        other_voice_threshold: Calibrated-uncertainty cutoff for the
            ``other_voice`` flag; ``None`` uses the adaptive
            ``OTHER_VOICE_CALIBRATED_CUTOFF``.
        min_p_voice: Presence gate threshold.
        fusion_weights: Optional per-model consensus weights.

    Returns:
        One :class:`ProfileComparisonResult` per window, time-aligned to the
        detection grid.
    """
    # Reference grid: any model's window list (they share the grid). Prefer a
    # model that also has a centroid so indices line up with scorable vectors.
    grid_model = next((m for m in centroids if m in detection_windows and detection_windows[m]), None)
    if grid_model is None:
        grid_model = next((m for m in detection_windows if detection_windows[m]), None)
    if grid_model is None:
        return []
    grid = detection_windows[grid_model]

    cutoff = other_voice_threshold if other_voice_threshold is not None else C.OTHER_VOICE_CALIBRATED_CUTOFF

    results: list[ProfileComparisonResult] = []
    for i, ref_w in enumerate(grid):
        p_voice = None
        if p_voice_by_window is not None and i < len(p_voice_by_window):
            p_voice = p_voice_by_window[i]

        voice_present: bool | None = None
        if voice_present_by_window is not None and i < len(voice_present_by_window):
            voice_present = bool(voice_present_by_window[i])

        # Presence gate → unavailable, never flagged. The scene-derived voice
        # mask is authoritative when supplied (recall: catch background/secondary
        # voice, exclude cough/breath); otherwise fall back to the p_voice gate.
        if voice_present is not None:
            gated_out = not voice_present
        else:
            gated_out = p_voice is not None and p_voice < min_p_voice
        if gated_out:
            results.append(
                ProfileComparisonResult(
                    start=float(ref_w.start_s),
                    end=float(ref_w.end_s),
                    similarity=None,
                    other_voice_uncertainty=None,
                    flag="unavailable",
                    p_voice=float(p_voice) if p_voice is not None else None,
                    per_model={},
                )
            )
            continue

        window_vectors: dict[str, np.ndarray] = {}
        for model_id, windows in detection_windows.items():
            if i < len(windows) and windows[i].vector.size > 0:
                window_vectors[model_id] = np.asarray(windows[i].vector, dtype=np.float64)

        similarity, uncertainty, per_model = score_window(
            window_vectors, centroids, calibration_band, fusion_weights=fusion_weights
        )

        if uncertainty is None:
            flag: str = "unavailable"
        elif uncertainty >= cutoff:
            flag = "other_voice"
        else:
            flag = "target"

        results.append(
            ProfileComparisonResult(
                start=float(ref_w.start_s),
                end=float(ref_w.end_s),
                similarity=similarity,
                other_voice_uncertainty=uncertainty,
                flag=flag,  # type: ignore[arg-type]
                p_voice=float(p_voice) if p_voice is not None else None,
                per_model=per_model,
            )
        )
    return results


def _window_step_seconds(results: Sequence[ProfileComparisonResult]) -> list[float]:
    """Non-overlapping per-window duration: gap to the next window's start.

    The detection grid overlaps (e.g. 1 s window, 0.5 s hop), so summing raw
    window spans double-counts time. Using each window's *step* to the next
    (and the final window's own span) gives a coverage-correct duration so the
    reported seconds and fractions are not inflated.
    """
    n = len(results)
    steps: list[float] = []
    for i, r in enumerate(results):
        if i + 1 < n:
            steps.append(max(0.0, float(results[i + 1].start) - float(r.start)))
        else:
            steps.append(max(0.0, float(r.end) - float(r.start)))
    return steps


def summarize_other_voice(
    results: Sequence[ProfileComparisonResult],
    profile_confidence: ProfileConfidence,
) -> RecordingOtherVoiceSummary:
    """Roll per-window comparison results up to a recording-level other-voice summary.

    Produces the decision-ready signals that extend ``analyze_audio``'s existing
    ``single_speaker`` claim: the fraction and duration of speech-present audio
    flagged ``other_voice``, the peak and 95th-percentile uncertainty (robust to
    a single spike), the speech-present denominator, and an echo of the profile
    confidence so a downstream gate can fail-safe on a weak profile. No verdict.
    """
    steps = _window_step_seconds(results)
    speech_present_seconds = 0.0
    other_voice_seconds = 0.0
    uncertainties: list[float] = []
    sim_step_weighted = 0.0  # Σ step·similarity over scored windows (for subject_dominance)
    sim_step_total = 0.0  # Σ step over windows that carried a similarity
    for r, step in zip(results, steps, strict=False):
        if r.flag == "unavailable":
            continue
        speech_present_seconds += step
        if r.other_voice_uncertainty is not None:
            uncertainties.append(float(r.other_voice_uncertainty))
        if r.similarity is not None:
            sim_step_weighted += step * float(r.similarity)
            sim_step_total += step
        if r.flag == "other_voice":
            other_voice_seconds += step

    fraction = (other_voice_seconds / speech_present_seconds) if speech_present_seconds > 0 else 0.0
    peak = max(uncertainties) if uncertainties else 0.0
    p95 = float(np.percentile(uncertainties, 95)) if uncertainties else 0.0
    # Continuous subject dominance: voiced-time-weighted mean subject similarity.
    # ``None`` when nothing was scorable — "no signal", not "confidently wrong".
    subject_dominance = (sim_step_weighted / sim_step_total) if sim_step_total > 0 else None

    return RecordingOtherVoiceSummary(
        profile_other_voice_fraction=float(fraction),
        profile_other_voice_seconds=float(other_voice_seconds),
        profile_peak_other_voice_uncertainty=float(peak),
        profile_p95_other_voice_uncertainty=p95,
        profile_speech_present_seconds=float(speech_present_seconds),
        profile_confidence=profile_confidence,
        profile_subject_dominance=subject_dominance,
    )


def compute_target_quality(
    results: Sequence[ProfileComparisonResult],
    profile_confidence: ProfileConfidence,
    *,
    squim_by_window: Sequence[dict[str, float] | None] | None = None,
) -> RecordingQualityIndicator:
    """Estimate how cleanly the target voice is captured in a recording.

    The headline ``profile_target_quality`` is a purely **profile-relative**
    score — the mean of two natively-[0,1] components:

    1. ``target_match_fraction`` — fraction of speech-present duration flagged
       ``target`` (i.e. ``1 - other-voice rate``).
    2. ``mean_target_consistency`` — mean calibrated similarity to the profile
       on the target-matched windows.

    SQUIM (STOI/PESQ/SI-SDR) is reported alongside as **raw means** on the
    target-matched windows (not folded into the headline): those metrics live on
    different scales (STOI ~[0,1], PESQ ~[1,4.5], SI-SDR in dB), the existing
    ``quality`` claim already maps them to uncertainty with its own anchors, and
    a whole-file SQUIM adds little beyond that. The consumer can weigh them.

    Args:
        results: Per-window comparison results for the recording.
        profile_confidence: Echoed onto the indicator; target quality is
            meaningless on an ``insufficient`` profile and should be discounted
            on ``low`` / ``ambiguous`` by the consumer.
        squim_by_window: Optional per-window ``{"stoi","pesq","si_sdr"}`` aligned
            to ``results`` (the caller broadcasts a whole-file score across
            windows when that is all it has). ``None`` skips ``profile_squim``.

    Returns:
        A :class:`RecordingQualityIndicator`. ``profile_target_quality`` is
        ``None`` when no window was scorable (all ``unavailable``), to keep
        "could not assess" distinct from "assessed as poor".
    """
    steps = _window_step_seconds(results)
    speech_present_seconds = 0.0
    matched_seconds = 0.0
    consistencies: list[float] = []
    matched_idx: list[int] = []
    for i, (r, step) in enumerate(zip(results, steps, strict=False)):
        if r.flag == "unavailable":
            continue
        speech_present_seconds += step
        if r.flag == "target":
            matched_seconds += step
            matched_idx.append(i)
            if r.similarity is not None:
                consistencies.append(float(r.similarity))

    if speech_present_seconds <= 0:
        # No scorable (speech-present) windows — the recording could not be
        # assessed (all windows gated ``unavailable``, or no model overlap with
        # the profile). Report ``None`` ("unavailable") rather than ``0.0`` so a
        # consumer (and the global_summary fold) does not read it as a
        # confidently-poor capture.
        target_match_fraction = 0.0
        mean_consistency = 0.0
        target_quality: float | None = None
    else:
        target_match_fraction = matched_seconds / speech_present_seconds
        mean_consistency = float(np.mean(consistencies)) if consistencies else 0.0
        target_quality = float(np.mean([target_match_fraction, mean_consistency]))

    profile_squim: dict[str, float] | None = None
    if squim_by_window is not None and matched_idx:
        per_metric: dict[str, list[float]] = {"stoi": [], "pesq": [], "si_sdr": []}
        for i in matched_idx:
            row = squim_by_window[i] if i < len(squim_by_window) else None
            if not row:
                continue
            for metric in per_metric:
                v = row.get(metric)
                if v is not None:
                    per_metric[metric].append(float(v))
        means = {m: float(np.mean(vals)) for m, vals in per_metric.items() if vals}
        if means:
            profile_squim = means

    return RecordingQualityIndicator(
        profile_target_quality=target_quality,
        profile_target_match_fraction=float(target_match_fraction),
        profile_mean_target_consistency=mean_consistency,
        profile_squim=profile_squim,
        profile_confidence=profile_confidence,
    )


def profile_votes_by_bucket(
    results: Sequence[ProfileComparisonResult],
    bucket_bounds: Sequence[tuple[float, float]],
) -> list[dict[str, dict[str, object]]]:
    """Map detection-grid results onto an identity bucket grid as ``model_votes`` entries.

    For each ``(start, end)`` bucket, the temporally closest comparison result
    (by window center) supplies a ``speaker_profile/consensus`` vote carrying
    ``{similarity, other_voice_uncertainty, flag}`` plus one
    ``speaker_profile/<model>`` vote per model carrying its pre-fusion
    uncertainty. These are additive: the identity aggregator ignores them, so
    the existing per-bucket uncertainty is unchanged — they are an extra
    reference signal in ``model_votes`` and the per-pass sidecar.

    Returns one dict per bucket (empty when no result overlaps it).
    """
    if not results:
        return [{} for _ in bucket_bounds]

    centers = [0.5 * (float(r.start) + float(r.end)) for r in results]
    out: list[dict[str, dict[str, object]]] = []
    for b_start, b_end in bucket_bounds:
        b_center = 0.5 * (b_start + b_end)
        nearest = min(range(len(results)), key=lambda i: abs(centers[i] - b_center))
        r = results[nearest]
        # Only attach when the nearest window actually overlaps the bucket, so a
        # bucket outside the scored span stays empty rather than borrowing a
        # distant window's flag.
        if r.end <= b_start or r.start >= b_end:
            out.append({})
            continue
        vote: dict[str, dict[str, object]] = {
            "speaker_profile/consensus": {
                "similarity": r.similarity,
                "other_voice_uncertainty": r.other_voice_uncertainty,
                "flag": r.flag,
            }
        }
        for model_id, unc in r.per_model.items():
            vote[f"speaker_profile/{model_id}"] = {"other_voice_uncertainty": float(unc)}
        out.append(vote)
    return out


def leave_one_file_out_profile(
    pooled_windows: Sequence[TaggedWindowEmbedding],
    exclude_file_id: str,
    *,
    embedding_models: Sequence[str] = C.DEFAULT_EMBEDDING_MODELS,
    prefer_session: str | None = None,
    session_of_file: Mapping[str, str | None] | None = None,
) -> AggregationResult | None:
    """Recompute the profile centroid excluding one contributing file's windows.

    When the recording being scored helped build the profile, its windows must
    be removed from the centroid so the comparison is not circular. Returns
    ``None`` when nothing remains after the exclusion (a single-file subject —
    use :func:`within_file_holdout_profile` instead).
    """
    remaining = [w for w in pooled_windows if w.file_id != exclude_file_id]
    if not remaining:
        return None
    return aggregate_dominant_cluster(
        remaining,
        embedding_models=embedding_models,
        prefer_session=prefer_session,
        session_of_file=session_of_file,
    )


def within_file_holdout_profile(
    file_windows: Sequence[TaggedWindowEmbedding],
    exclude_start_s: float,
    exclude_end_s: float,
    *,
    embedding_models: Sequence[str] = C.DEFAULT_EMBEDDING_MODELS,
    guard_s: float = C.WITHIN_FILE_HOLDOUT_GUARD_S,
) -> AggregationResult | None:
    """Single-file fallback for leave-one-file-out: hold out near the test window.

    Excludes every window overlapping ``[exclude_start_s - guard_s,
    exclude_end_s + guard_s]`` and re-aggregates the rest, so the window under
    test is never scored against a centroid that contains itself or its
    immediate neighbors. Returns ``None`` when too little remains.
    """
    lo = exclude_start_s - guard_s
    hi = exclude_end_s + guard_s
    remaining = [w for w in file_windows if not (w.window.start_s < hi and w.window.end_s > lo)]
    if not remaining:
        return None
    return aggregate_dominant_cluster(remaining, embedding_models=embedding_models)
