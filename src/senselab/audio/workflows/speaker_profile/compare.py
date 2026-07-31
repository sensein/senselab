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

import warnings
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
    VoiceGroupAssignment,
    VoiceGroupMatch,
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
    and 256-D ResNet-TDNN are never compared directly). The per-model uncertainties
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
    diar_overlap_by_window: Sequence[bool] | None = None,
    other_voice_threshold: float | None = C.OTHER_VOICE_THRESHOLD_DEFAULT,
    min_p_voice: float = C.MIN_P_VOICE_FOR_COMPARISON,
    diar_overlap_floor: float = C.DIAR_OVERLAP_OTHER_VOICE_FLOOR,
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
        diar_overlap_by_window: Optional per-window mask flagging windows where
            diarization sees 2+ distinct speakers active (overlapping speech).
            On such a window a non-subject voice is present by definition, but the
            profile distance is unreliable (mixed-speaker embedding), so the
            consensus other-voice uncertainty is raised to at least
            ``diar_overlap_floor`` (``max(profile_value, floor)``) — a
            reference-free corroborator, applied only to already-scored windows.
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
        diar_overlap_floor: Floor the consensus other-voice uncertainty is raised
            to on a ``diar_overlap_by_window`` window.
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

        # Diarization-overlap corroborator: 2+ speakers active in this window ⇒ a
        # non-subject voice is present even if the profile (unreliable on a
        # mixed-speaker embedding) read it as the target. Raise the consensus
        # uncertainty to at least the floor. Applied only to already-scored
        # windows (does not fabricate a flag where the profile couldn't score).
        if (
            uncertainty is not None
            and diar_overlap_by_window is not None
            and i < len(diar_overlap_by_window)
            and diar_overlap_by_window[i]
        ):
            uncertainty = max(uncertainty, diar_overlap_floor)
            similarity = 1.0 - uncertainty

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

    Produces decision-ready signals for a consumer's own single-speaker claim:
    the fraction and duration of speech-present audio
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


def score_voice_groups(
    group_vectors: Mapping[str, Mapping[str, np.ndarray]],
    centroids: Mapping[str, Sequence[float]],
    calibration_band: Mapping[str, tuple[float, float]],
    *,
    fusion_weights: Mapping[str, float] | None = None,
) -> VoiceGroupAssignment:
    """Name pooled voice groups: which one is the enrolled subject?

    The intended consumer is a speaker-identity step that has already grouped a
    recording into distinct voices (e.g. by pooling per-window embeddings per
    segment and clustering them) and needs to know *which* group is the consented
    speaker. Reference-free grouping can say two voices differ; only a profile can
    say which is the subject.

    Preferred over :func:`compare_recording_to_profile` wherever 2+ groups exist:

    - a group centroid pools many windows, so it is far less noisy than any single
      short detection window;
    - the decision becomes "which group is closer", a *relative* comparison between
      groups drawn from the same recording, so channel and SNR offsets largely
      cancel and no absolute threshold is needed.

    With one group there is no comparison to make and the result falls back to the
    calibrated similarity alone (``basis="absolute"``) — reported explicitly so the
    caller can weight it accordingly rather than mistaking it for a relative call.

    This function does not decide *whether* the subject is present: a low-similarity
    winner is still the winner. Interpreting ``similarity`` and ``margin`` against
    the profile's own ``confidence`` is the caller's job.

    Args:
        group_vectors: ``{group_id -> {model_id -> pooled centroid vector}}``.
        centroids: ``{model_id -> centroid vector}`` from the profile.
        calibration_band: ``{model_id -> (same_floor, diff_floor)}`` from the profile.
        fusion_weights: Optional per-model consensus weights; ``None`` is unweighted.

    Returns:
        A :class:`VoiceGroupAssignment`.
    """
    matches: list[VoiceGroupMatch] = []
    for group_id, vectors in group_vectors.items():
        similarity, uncertainty, per_model = score_window(
            vectors, centroids, calibration_band, fusion_weights=fusion_weights
        )
        matches.append(
            VoiceGroupMatch(
                group_id=group_id,
                similarity=similarity,
                other_voice_uncertainty=uncertainty,
                per_model=per_model,
            )
        )

    # Most- to least-like the subject; unscorable groups sort last.
    matches.sort(key=lambda m: (m.similarity is None, -(m.similarity or 0.0), m.group_id))
    scorable = [m for m in matches if m.similarity is not None]

    if not scorable:
        return VoiceGroupAssignment(matches=matches, subject_group_id=None, margin=None, basis="unavailable")
    if len(scorable) == 1:
        return VoiceGroupAssignment(
            matches=matches, subject_group_id=scorable[0].group_id, margin=None, basis="absolute"
        )
    best, runner_up = scorable[0], scorable[1]
    margin = float((best.similarity or 0.0) - (runner_up.similarity or 0.0))
    return VoiceGroupAssignment(matches=matches, subject_group_id=best.group_id, margin=margin, basis="relative")


class GridMismatchError(ValueError):
    """Raised by ``check_grid_compatibility(..., strict=True)`` on a grid mismatch."""


class GridMismatchWarning(UserWarning):
    """Warned when detection windows were extracted at a different grid than the profile."""


def derive_window_grid(windows: Sequence[WindowEmbedding]) -> tuple[float, float | None]:
    """Infer ``(window_s, hop_s)`` from a window list's timestamps.

    Args:
        windows: A model's per-window embeddings, in time order.

    Returns:
        ``(window_s, hop_s)``; ``hop_s`` is ``None`` when fewer than two windows make the
        hop unobservable. ``(0.0, None)`` for an empty list.
    """
    if not windows:
        return 0.0, None
    window_s = float(windows[0].end_s - windows[0].start_s)
    hop_s = float(windows[1].start_s - windows[0].start_s) if len(windows) > 1 else None
    return window_s, hop_s


def check_grid_compatibility(
    windows: Sequence[WindowEmbedding],
    profile_window_s: float,
    *,
    strict: bool = False,
    tolerance_s: float = 0.01,
) -> None:
    """Verify detection windows match the grid a profile was enrolled at.

    **Why this warns rather than raises.** Comparing a window against a centroid built at a
    different length adds a duration domain gap on top of whatever speaker difference is
    present, and nothing downstream would notice: ``calibration_band`` does **not** adapt to
    the grid — measured, it came out as the fixed ``SAME/DIFF_SPEAKER_FLOOR_FALLBACK`` values
    at both 2.0 s and 0.5 s. So a cross-grid comparison quietly scores against thresholds
    chosen for another grid.

    But measured on constructed intrusions, the cost is **2-10 AUC points** (clean-intrusion
    0.899 -> 0.877, overlapped 0.846 -> 0.751), not a collapse. Degraded, not meaningless — so
    the default surfaces it and lets the run continue. Pass ``strict=True`` where a mismatch
    should be fatal.

    Note this guard covers *mixing* grids. It does not fix the related calibration problem
    that the fixed band is implicitly tuned for the coarse grid: cross-subject centroid
    similarity measures 0.27 at 2.0 s but 0.41 at 0.5 s, while ``DIFF_SPEAKER_FLOOR``
    stays 0.70 — so at 0.5 s genuinely different speakers never reach "confidently
    different". Choosing a grid still requires choosing a band.

    Args:
        windows: Detection windows for one model.
        profile_window_s: The profile's enrollment window length.
        strict: Raise instead of warning.
        tolerance_s: Absolute tolerance, to absorb float noise and end-of-audio clipping.

    Warns:
        GridMismatchWarning: If the window length differs beyond ``tolerance_s``. The *hop* is
            deliberately not checked: ``_window_step_seconds`` derives coverage from the
            results' own timestamps, so duration rollups are already hop-agnostic.

    Raises:
        GridMismatchError: Same condition, when ``strict=True``.
    """
    window_s, _ = derive_window_grid(windows)
    if not windows:
        return
    if abs(window_s - profile_window_s) > tolerance_s:
        msg = (
            f"detection windows are {window_s:g}s but the profile was enrolled at "
            f"{profile_window_s:g}s. The calibration band does not adapt to the grid, so this "
            f"scores against thresholds chosen for another window length — measured cost is "
            f"2-10 AUC points. Re-extract at {profile_window_s:g}s, or enroll at {window_s:g}s."
        )
        if strict:
            raise GridMismatchError(msg)
        warnings.warn(msg, GridMismatchWarning, stacklevel=2)
