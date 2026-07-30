"""Presence axis vote harvesters — "was there a speaker in this bucket?".

Maximally inclusive contributions per FR-002 — every signal the pipeline already
runs casts a vote, each calibrated to what its signal can actually answer:

- **Diar models** (pyannote, Sortformer): ``speaks`` = bucket overlaps a diar
  segment. Binary vote, no native confidence.
- **ASR models**: ``speaks`` = at least one transcript token's timestamp
  overlaps the bucket (post-MMS-alignment for text-only ASR per FR-011).
  Native confidence from Whisper ``avg_logprob`` (other ASRs vote binary).
  When Whisper's ``no_speech_prob`` is ≥ 0.5 but the transcript still has
  tokens here, the bucket's vote is forced to ``speaks=False`` and flagged
  ``hallucinated=True`` so the global aggregator can also penalize.
- **Whisper ``no_speech_prob``** (one extra voter per Whisper model): the
  model's own VAD-like silence head, plumbed as a sibling vote keyed
  ``<asr_model>::no_speech_prob``.
- **AST / YAMNet**: ``speaks`` = top-1 label ∈ ``speech_presence_labels``.
- **Acoustic / loudness** (openSMILE ``Loudness_sma3``): votes ``True`` for
  any audible sound — captures whisper and distorted voice that voicing-based
  signals miss.
- **Acoustic / spectral activity** (openSMILE ``spectralFlux_sma3``): votes
  ``True`` for non-stationary spectrum — also catches whisper + distortion
  even when loudness is low.
- **Acoustic / HNR** (openSMILE ``HNRdBACF_sma3nz``): votes ``True`` for
  clean voiced speech. Calibrated so a *low* HNR contributes ``p_voice ≈ 0.5``
  (uninformative — could be whisper or silence) rather than ``False``, so
  whispered speech isn't pushed down.
- **PPG voice fraction**: fraction of bucket frames whose argmax is *not*
  ``"<silent>"``. Lightly catches whisper too.
- **Embedding silhouette** (windowed speaker embeddings, clustered): each
  window's silhouette coefficient from clustering the pass's window
  embeddings; high → embedding sits inside a coherent speaker cluster (voice);
  low / negative → embedding doesn't fit any cluster (silence / noise /
  transition).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from senselab.audio.workflows.audio_analysis.embeddings import WindowEmbedding
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.harvesters import (
    classification_top1_in_window,
    classification_windows,
    diar_speaks_in_window,
    resolve_asr_result,
    token_overlaps_window,
    whisper_bucket_confidence,
    whisper_bucket_no_speech_prob,
)

if TYPE_CHECKING:
    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import FramePosterior

# Canonical AudioSet labels that count as "speech present" for the AST / YAMNet
# presence voters and the ``speech_window_mask_for_file`` gate. Single source of
# truth so the ``analyze_audio`` identity-axis gate and the ``speaker_profile``
# build-time gate share one speech definition (FR-002: "the same signal the
# clustering step consumes"). Override per-run via the CLI flag.
DEFAULT_SPEECH_PRESENCE_LABELS: tuple[str, ...] = (
    "Speech",
    "Conversation",
    "Narration, monologue",
    "Female speech, woman speaking",
    "Male speech, man speaking",
    "Child speech, kid speaking",
    "Speech synthesizer",
)

# Reporting grids finer than this (seconds) trigger coarse-voter demotion: whole-window
# scene tags / per-segment no-speech probability / sentence-level ASR overlap repeat one
# value across many fine buckets and would otherwise dominate the mean (FR-014). At the
# historical 0.5 s grid no demotion applies, so existing outputs are unchanged.
_FINE_GRID_THRESHOLD_S = 0.5
_COARSE_VOTER_WEIGHT = 0.25


def _row_window_overlap(rows: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    """Return the subset of feature rows whose window overlaps ``[start, end)``."""
    out: list[dict[str, Any]] = []
    for r in rows:
        rs = r.get("start")
        re_ = r.get("end")
        if rs is None or re_ is None:
            continue
        try:
            rs_f, re_f = float(rs), float(re_)
        except (TypeError, ValueError):
            continue
        if rs_f < end and re_f > start:
            out.append(r)
    return out


def _mean_col(rows: list[dict[str, Any]], col: str) -> float | None:
    """Mean of column ``col`` across rows, ignoring None / non-numeric values."""
    vals: list[float] = []
    for r in rows:
        v = r.get(col)
        if v is None:
            continue
        try:
            vf = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(vf):
            continue
        vals.append(vf)
    if not vals:
        return None
    return float(np.mean(vals))


def _calibrate_high(value: float | None, low: float, high: float) -> tuple[bool, float] | None:
    """Map a feature value to ``(speaks, p_voice)`` where higher value → more likely voice.

    ``value <= low`` → ``(False, 0.0)``; ``value >= high`` → ``(True, 1.0)``;
    interpolated linearly in between. ``None`` input → ``None`` output (drop vote).
    """
    if value is None:
        return None
    if value <= low:
        return False, 0.0
    if value >= high:
        return True, 1.0
    p = (value - low) / (high - low)
    return p >= 0.5, max(0.0, min(1.0, p))


def _calibrate_uninformative_low(value: float | None, low: float, high: float) -> tuple[bool, float] | None:
    """Like ``_calibrate_high`` but a low value is uninformative (``p = 0.5``), not negative.

    Used for HNR / phonation_ratio, where low values can mean *either* silence or
    whisper / distorted voice — the signal can't tell the difference, so it
    abstains rather than voting False.
    """
    if value is None:
        return None
    if value >= high:
        return True, 1.0
    if value <= low:
        return False, 0.5  # speaks=False is irrelevant; the 0.5 confidence is what matters
    p = 0.5 + 0.5 * (value - low) / (high - low)
    return p >= 0.5, max(0.0, min(1.0, p))


def _native_classification_grid(block: dict[str, Any]) -> tuple[float, float]:
    """Recover the (win_length, hop_length) the classifier ran with from its first window."""
    windows = classification_windows(block.get("result"))
    if not windows or not isinstance(windows[0], dict):
        return 1.0, 1.0
    w = windows[0]
    win_length = float(w.get("win_length", 0) or 0) or float(w.get("end", 0) - w.get("start", 0))
    hop_length = float(w.get("hop_length", 0) or 0) or win_length
    if win_length <= 0:
        win_length = 1.0
    if hop_length <= 0:
        hop_length = win_length
    return win_length, hop_length


def harvest_presence_votes(
    *,
    pass_summary: dict[str, Any],
    grid: BucketGrid,
    speech_presence_labels: list[str],
    alignment_by_model: dict[str, Any],
    per_window_embeddings: dict[str, list[Any]] | None = None,
    frame_posteriors: dict[str, "FramePosterior"] | None = None,
) -> list[dict[str, Any]]:
    """Yield ``{"start", "end", "votes", "frame_instability"}`` per bucket for presence.

    ``votes`` is a dict ``{model_id → {"speaks": bool, "native_confidence": float | None}}``
    spanning every contributing model. Buckets where no model contributed any vote are
    still emitted (caller drops them in compute._row_emit if status != ok).

    ``frame_posteriors`` maps a voter name → continuous per-frame P(speech)
    (``segmentation-3.0`` raw scores, Brouhaha VAD head). Each contributes a
    fine-resolution voter (bucket-mean P(speech)); the within-bucket frame std
    is aggregated into ``frame_instability`` for the ``presence_uncertainty``
    column. On grids finer than ``_FINE_GRID_THRESHOLD_S`` the coarse voters are
    down-weighted to ``_COARSE_VOTER_WEIGHT`` (FR-014).
    """
    duration_s = float(pass_summary.get("duration_s", 0.0) or 0.0)
    if duration_s <= 0:
        # Try to infer from any model's results. Walk only iterables of segment-shaped
        # objects (with .end / .get('end')) — guard against ScriptLine objects whose
        # iteration would expose pydantic field names rather than segments.
        for task in ("diarization", "asr"):
            block = (pass_summary.get(task) or {}).get("by_model") or {}
            for sub in block.values():
                if not (isinstance(sub, dict) and sub.get("status") == "ok"):
                    continue
                res = sub.get("result")
                if not (isinstance(res, list) and res):
                    continue
                segs = res[0] if isinstance(res[0], list) else res
                if not isinstance(segs, list):
                    continue
                for s in segs:
                    if isinstance(s, dict):
                        end_attr = s.get("end")
                    elif hasattr(s, "end") and not isinstance(s, str):
                        end_attr = s.end
                    else:
                        continue
                    if end_attr is not None:
                        try:
                            duration_s = max(duration_s, float(end_attr))
                        except (TypeError, ValueError):
                            continue

    diar_blocks = (pass_summary.get("diarization") or {}).get("by_model") or {}
    diar_ok = {m: b for m, b in diar_blocks.items() if isinstance(b, dict) and b.get("status") == "ok"}
    asr_blocks = (pass_summary.get("asr") or {}).get("by_model") or {}
    asr_ok = {m: b for m, b in asr_blocks.items() if isinstance(b, dict) and b.get("status") == "ok"}
    asr_resolved = {m: resolve_asr_result(b, alignment_by_model.get(m)) for m, b in asr_ok.items()}

    ast_block = pass_summary.get("ast") or {}
    yam_block = pass_summary.get("yamnet") or {}
    ast_ok = ast_block.get("status") == "ok"
    yam_ok = yam_block.get("status") == "ok"
    if ast_ok:
        ast_win, ast_hop = _native_classification_grid(ast_block)
    if yam_ok:
        yam_win, yam_hop = _native_classification_grid(yam_block)

    # Acoustic features — opensmile rows from the features task. (parselmouth
    # rows are per-utterance not per-bucket; not used as a voter.)
    feat_block = pass_summary.get("features") or {}
    feat_result = feat_block.get("result") if isinstance(feat_block, dict) else None
    opensmile_rows: list[dict[str, Any]] = feat_result.get("opensmile", []) if isinstance(feat_result, dict) else []

    # openSMILE Loudness_sma3 / spectralFlux_sma3 are not absolutely calibrated
    # — they vary with input gain and recording conditions. Compute per-pass
    # percentile anchors so the voter calibrates to "high vs low for this
    # specific recording" rather than fixed-magnitude thresholds. Quiet floor
    # ≈ 10th percentile (mostly silence); active speech anchor ≈ 75th
    # percentile (most loud frames). Falls back to fixed defaults if the
    # opensmile track is too short to estimate percentiles.
    def _per_pass_band(
        rows: list[dict[str, Any]],
        col: str,
        default_low: float,
        default_high: float,
    ) -> tuple[float, float]:
        vals: list[float] = []
        for r in rows:
            v = r.get(col)
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(vf):
                vals.append(vf)
        if len(vals) < 100:  # ~1 s of opensmile frames
            return default_low, default_high
        return float(np.percentile(vals, 10)), float(np.percentile(vals, 75))

    loudness_low, loudness_high = _per_pass_band(opensmile_rows, "Loudness_sma3", 0.05, 0.5)
    flux_low, flux_high = _per_pass_band(opensmile_rows, "spectralFlux_sma3", 0.05, 0.30)
    # HNR uses fixed dB thresholds — those ARE absolutely calibrated (it's a
    # ratio in dB that doesn't depend on input gain). Lowered the high anchor
    # to 10 dB per agent review (#7) — typical conversational HNR is 8–14 dB.
    hnr_low, hnr_high = 2.0, 10.0

    # PPG argmax-per-frame for the voice-fraction signal.
    ppg_block = pass_summary.get("ppgs") or pass_summary.get("ppg") or {}
    ppg_per_frame: list[str] = []
    ppg_frame_hop: float = 0.0
    if isinstance(ppg_block, dict) and ppg_block.get("status") == "ok":
        import sys as _sys

        from senselab.audio.workflows.audio_analysis.harvesters import ppg_argmax_per_frame

        try:
            ppg_per_frame, ppg_frame_hop = ppg_argmax_per_frame(
                ppg_block.get("result"),
                ppg_block.get("phoneme_labels"),
                duration_s,
            )
        except Exception as ppg_exc:  # noqa: BLE001
            # ``_to_2d_frame_major`` raises ValueError on ambiguous tensor
            # shape — that's a real configuration problem, surface it rather
            # than silently disabling the voter.
            print(
                f"warn: PPG argmax decoding failed: {ppg_exc!r} — ppg_voice_fraction voter disabled for this pass",
                file=_sys.stderr,
            )
            ppg_per_frame, ppg_frame_hop = [], 0.0

    # Embedding-cluster silhouette as a voice presence signal. Cluster the
    # windowed embeddings; each window's silhouette coefficient measures how
    # well it fits inside one of the clusters. Voice from a coherent speaker
    # sits firmly inside a cluster (high silhouette → high p_voice); silence
    # or noise lacks the inter-cluster structure, producing low / negative
    # silhouettes. This avoids the previous "embedding norm vs median"
    # heuristic which is dominated by phonetic content rather than voicing.
    # We use whichever embedding model is alphabetically first (typically
    # ECAPA before ResNet); rerun across each model would amount to a vote
    # of voters within the same axis class — leave that to the aggregator if
    # a future caller wants it.
    silhouette_by_emb_model: dict[str, dict[int, float]] = {}
    silhouette_windows: list[Any] = []
    if per_window_embeddings:
        from senselab.audio.workflows.audio_analysis.embeddings import silhouette_voice_score

        # Pick the first embedding model with non-empty windows.
        for emb_model in sorted(per_window_embeddings):
            entries = per_window_embeddings.get(emb_model) or []
            if not entries:
                continue
            scores = silhouette_voice_score(entries)
            if scores is not None:
                silhouette_by_emb_model[emb_model] = scores
                silhouette_windows = entries
                break

    allow = set(speech_presence_labels)
    out: list[dict[str, Any]] = []
    for start, end, _idx in grid.iter_buckets(duration_s):
        votes: dict[str, dict[str, Any]] = {}

        # Diar — no native confidence today (Sortformer/pyannote post-process to labels).
        for m, block in diar_ok.items():
            votes[m] = {
                "speaks": diar_speaks_in_window(block.get("result"), start, end),
                "native_confidence": None,
            }

        # ASR — speaks iff any chunk overlaps; native confidence from Whisper avg_logprob.
        # Hallucination detection: if Whisper's no_speech_prob is high but a
        # transcript token still overlaps the bucket, the ASR vote is
        # downweighted (Whisper sometimes generates "Thanks for watching!"
        # over silence). The hallucination flag is exposed in the vote dict
        # so downstream consumers can audit.
        for m, resolved in asr_resolved.items():
            speaks = token_overlaps_window(resolved, start, end)
            nc = whisper_bucket_confidence(resolved, start, end)
            nsp = whisper_bucket_no_speech_prob(resolved, start, end)
            hallucinated = bool(speaks and nsp is not None and nsp >= 0.5)
            asr_vote: dict[str, Any] = {
                "speaks": speaks and not hallucinated,
                "native_confidence": nc,
                "hallucinated": hallucinated,
                "coarse": True,  # sentence-level: one word spans many fine buckets
            }
            if nsp is not None:
                asr_vote["no_speech_prob"] = float(nsp)
            votes[m] = asr_vote
            # Whisper-only: dedicated VAD-like vote from the model's
            # ``no_speech_prob`` head. Independent of token-overlap (which
            # measures whether the transcript landed in this bucket); this is
            # the model's own "is there silence here?" decision.
            if nsp is not None:
                votes[f"{m}::no_speech_prob"] = {
                    "speaks": nsp < 0.5,
                    "native_confidence": 1.0 - float(nsp),
                    "coarse": True,  # per-~30 s Whisper segment
                }

        # AST / YAMNet — project the bucket's CENTER onto the nearest native
        # window (round-to-nearest, not floor). With AST's 10.24 s windows a
        # bucket straddling a window boundary should use the window that covers
        # most of the bucket, not the one whose start happens to be lower.
        bucket_center = 0.5 * (start + end)
        if ast_ok:
            ast_idx = max(0, int(round(bucket_center / ast_hop))) if ast_hop > 0 else 0
            label, score, _ = classification_top1_in_window(ast_block.get("result"), ast_idx)
            if label is not None:
                votes["ast"] = {
                    "speaks": label in allow,
                    "native_confidence": score,
                    "coarse": True,  # AST native window ~10.24 s
                }
        if yam_ok:
            yam_idx = max(0, int(round(bucket_center / yam_hop))) if yam_hop > 0 else 0
            label, score, _ = classification_top1_in_window(yam_block.get("result"), yam_idx)
            if label is not None:
                votes["yamnet"] = {
                    "speaks": label in allow,
                    "native_confidence": score,
                    "coarse": True,  # YAMNet native window ~0.96 s
                }

        # ── Acoustic features ────────────────────────────────────────────
        # Calibrate functions return ``(speaks_v, p_voice)`` where ``p_voice``
        # is the calibrated probability of voice. The aggregator interprets
        # ``native_confidence`` as the voter's confidence IN ITS OWN ``speaks``
        # direction (so e.g. AST top-1=Speech with score=0.7 means
        # ``speaks=True, nc=0.7``, and AST top-1=Music with score=0.7 means
        # ``speaks=False, nc=0.7`` — both contribute the right p_voice). To
        # match that semantic we convert ``p_voice → confidence-in-direction``
        # via ``p_voice if speaks else (1 − p_voice)``.
        def _vote_from_pvoice(p_voice: float, speaks_v: bool) -> dict[str, Any]:
            return {
                "speaks": speaks_v,
                "native_confidence": p_voice if speaks_v else (1.0 - p_voice),
            }

        if opensmile_rows:
            bucket_rows = _row_window_overlap(opensmile_rows, start, end)
            loud = _mean_col(bucket_rows, "Loudness_sma3")
            flux = _mean_col(bucket_rows, "spectralFlux_sma3")
            hnr = _mean_col(bucket_rows, "HNRdBACF_sma3nz")
            cal = _calibrate_high(loud, low=loudness_low, high=loudness_high)
            if cal is not None:
                speaks_v, p_v = cal
                votes["acoustic_loudness"] = _vote_from_pvoice(p_v, speaks_v)
            cal = _calibrate_high(flux, low=flux_low, high=flux_high)
            if cal is not None:
                speaks_v, p_v = cal
                votes["acoustic_spectral_activity"] = _vote_from_pvoice(p_v, speaks_v)
            cal = _calibrate_uninformative_low(hnr, low=hnr_low, high=hnr_high)
            if cal is not None:
                speaks_v, p_v = cal
                votes["acoustic_hnr"] = _vote_from_pvoice(p_v, speaks_v)

        # Note: parselmouth ``phonation_ratio`` is computed once per utterance
        # (Praat ``Sound: To TextGrid (silences)`` then phonation_time /
        # original_dur over the whole audio), so it doesn't vary by bucket and
        # is not a valid per-bucket voter. Excluded.

        # ── PPG voice fraction ───────────────────────────────────────────
        if ppg_per_frame and ppg_frame_hop > 0:
            first_frame = max(0, int(start / ppg_frame_hop))
            last_frame = min(len(ppg_per_frame), max(first_frame + 1, int(round(end / ppg_frame_hop))))
            n_frames = last_frame - first_frame
            if n_frames > 0:
                voice_count = sum(1 for p in ppg_per_frame[first_frame:last_frame] if p != "<silent>")
                voice_frac = voice_count / n_frames
                speaks_v = voice_frac >= 0.5
                votes["ppg_voice_fraction"] = _vote_from_pvoice(voice_frac, speaks_v)

        # ── Embedding silhouette ─────────────────────────────────────────
        # Per-bucket silhouette score from clustering windowed embeddings.
        # Find the window whose center is closest to the bucket center (same
        # logic as the identity harvester's per-bucket lookup).
        if silhouette_by_emb_model and silhouette_windows:
            scores = next(iter(silhouette_by_emb_model.values()))
            best_idx: int | None = None
            best_dist = float("inf")
            for i, w in enumerate(silhouette_windows):
                wc = 0.5 * (float(w.start_s) + float(w.end_s))
                d = abs(wc - bucket_center)
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx is not None and best_idx in scores:
                p_v = scores[best_idx]
                speaks_v = p_v >= 0.5
                sil_vote = _vote_from_pvoice(p_v, speaks_v)
                sil_vote["coarse"] = True  # embedding window ~1 s+
                votes["embedding_silhouette"] = sil_vote

        # ── Frame-posterior voters (segmentation-3.0, Brouhaha VAD) ──────
        # Continuous per-frame P(speech): fine-resolution voters plus a
        # within-bucket temporal-instability signal for presence_uncertainty.
        frame_stds: list[float] = []
        if frame_posteriors:
            for name, fp in frame_posteriors.items():
                if fp is None:
                    continue
                mean, std = fp.mean_std_in_window(start, end)
                if not np.isfinite(mean):
                    continue
                votes[name] = {"speaks": mean >= 0.5, "native_confidence": float(mean)}
                if np.isfinite(std):
                    frame_stds.append(float(std))

        # Demote coarse voters on fine reporting grids (FR-014). No-op at the
        # historical 0.5 s grid, so legacy outputs are unchanged.
        if grid.win_length < _FINE_GRID_THRESHOLD_S:
            for v in votes.values():
                if isinstance(v, dict) and v.get("coarse"):
                    v["weight"] = _COARSE_VOTER_WEIGHT

        # Within-bucket instability: std of a probability is at most 0.5, so ×2
        # maps to [0, 1]. None when no frame voter contributed.
        frame_instability = float(np.clip(2.0 * float(np.mean(frame_stds)), 0.0, 1.0)) if frame_stds else None

        out.append({"start": start, "end": end, "votes": votes, "frame_instability": frame_instability})

    return out


def speech_window_mask_for_file(
    *,
    entries: list[WindowEmbedding],
    pass_summary: dict[str, Any],
    speech_presence_labels: list[str],
) -> list[bool] | None:
    """Build a per-embedding-window boolean mask of "is this window speech?".

    Public single-file helper used both by the audio_analysis clustering step
    (``compute.py`` consumes it as ``is_speech_per_window``) and by the
    speaker_profile build-time gate (``T009`` per FR-002): it answers
    speech-vs-non-speech for one file from cached diarization-adjacent
    signals — AST/YAMNet speech labels and openSMILE loudness — *without*
    requiring ASR or PPG. Promoted from ``compute._speech_window_mask``
    (T009a) so the speaker_profile package can reuse it.

    **YAMNet veto, not a fallback ladder.** When YAMNet is available, its top-1
    label decision is authoritative — even if loudness is high, AST disagrees,
    or both. AST is only consulted when YAMNet is unavailable; openSMILE
    loudness is consulted only when both classifiers are unavailable. This is
    deliberate: YAMNet is trained on the AudioSet hierarchy with explicit
    speech labels; AST is broader-coverage but noisier on speech specifically;
    loudness is recording-conditional. Trusting YAMNet's "Music" / "Vehicle"
    over a loud-but-non-speech window is the right call for our use case.

    Tradeoff: YAMNet has known confusions (e.g. tagging child voices as "Music"
    or "Singing"). When that happens, a real-speech window gets dropped from
    clustering. The mitigation is upstream: tune ``speech_presence_labels`` to
    include the singing / NORP labels you care about. We deliberately do NOT
    let loudness override YAMNet here, because that would break the
    silent-room-detection guarantee callers rely on (a loud window of music
    must not be allowed to claim "speech" status).

    Returns ``None`` when none of the three signals are available, in which
    case the caller falls back to legacy behavior (cluster every non-zero-norm
    window).
    """
    ast_block = pass_summary.get("ast") or {}
    yam_block = pass_summary.get("yamnet") or {}
    ast_ok = isinstance(ast_block, dict) and ast_block.get("status") == "ok"
    yam_ok = isinstance(yam_block, dict) and yam_block.get("status") == "ok"

    feat_block = pass_summary.get("features") or {}
    feat_result = feat_block.get("result") if isinstance(feat_block, dict) else None
    opensmile_rows: list[dict[str, Any]] = feat_result.get("opensmile", []) if isinstance(feat_result, dict) else []

    if not (ast_ok or yam_ok or opensmile_rows):
        return None

    def _native_grid(block: dict[str, Any]) -> tuple[float, float]:
        windows = classification_windows(block.get("result"))
        if not windows or not isinstance(windows[0], dict):
            return 1.0, 1.0
        w = windows[0]
        win_len = float(w.get("win_length", 0) or 0) or float(w.get("end", 0) - w.get("start", 0))
        hop_len = float(w.get("hop_length", 0) or 0) or win_len
        if win_len <= 0:
            win_len = 1.0
        if hop_len <= 0:
            hop_len = win_len
        return win_len, hop_len

    ast_hop = _native_grid(ast_block)[1] if ast_ok else 0.0
    yam_hop = _native_grid(yam_block)[1] if yam_ok else 0.0

    loudness_q25: float | None = None
    if opensmile_rows:
        vals: list[float] = []
        for r in opensmile_rows:
            v = r.get("Loudness_sma3")
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if math.isfinite(vf):
                vals.append(vf)
        if len(vals) >= 100:  # ~1 s of opensmile frames
            loudness_q25 = float(np.percentile(vals, 25))

    allow = set(speech_presence_labels)
    mask: list[bool] = []
    for w in entries:
        center = 0.5 * (w.start_s + w.end_s)
        # YAMNet is authoritative when available — it's the canonical
        # AudioSet speech-presence detector.
        if yam_ok:
            idx = max(0, int(round(center / yam_hop))) if yam_hop > 0 else 0
            label, _, _ = classification_top1_in_window(yam_block.get("result"), idx)
            if label is not None:
                mask.append(label in allow)
                continue
        # Fall back to AST when YAMNet unavailable.
        if ast_ok:
            idx = max(0, int(round(center / ast_hop))) if ast_hop > 0 else 0
            label, _, _ = classification_top1_in_window(ast_block.get("result"), idx)
            if label is not None:
                mask.append(label in allow)
                continue
        # Final fallback: openSMILE loudness threshold.
        if loudness_q25 is not None and opensmile_rows:
            vals_in: list[float] = []
            for r in opensmile_rows:
                rs = r.get("start") or r.get("frameTime") or r.get("time")
                re_ = r.get("end")
                try:
                    rs_f = float(rs) if rs is not None else None
                    re_f = float(re_) if re_ is not None else (rs_f + 0.01 if rs_f is not None else None)
                except (TypeError, ValueError):
                    continue
                if rs_f is None or re_f is None:
                    continue
                if rs_f < w.end_s and re_f > w.start_s:
                    v = r.get("Loudness_sma3")
                    if v is None:
                        continue
                    try:
                        vf = float(v)
                    except (TypeError, ValueError):
                        continue
                    vals_in.append(vf)
            if vals_in:
                mean_loud = sum(vals_in) / len(vals_in)
                mask.append(mean_loud >= loudness_q25)
                continue
        mask.append(True)
    return mask


def reference_grid_and_speech_mask(
    per_model_windows: Mapping[str, list[WindowEmbedding]],
    *,
    pass_summary: dict[str, Any],
    speech_presence_labels: Sequence[str],
) -> tuple[list[WindowEmbedding], list[bool] | None]:
    """Pick the reference window grid and compute the per-window speech mask once.

    Shared by ``analyze_audio``'s identity-axis clustering (``compute.py``) and
    the ``speaker_profile`` build-time gate (``build.py``) so they apply the
    *same* speech definition (FR-002) via one code path instead of two parallel
    inline copies.

    All models in ``per_model_windows`` share the same window grid (by
    construction in :func:`extract_per_window_embeddings`), and the mask depends
    only on window times + scene/loudness signals — **not** on the embedding
    vectors — so it is computed once on the first non-empty model's grid and
    applies to every model.

    Returns ``(reference_windows, mask)``. ``mask`` is ``None`` when no presence
    signal (AST / YAMNet / loudness) is available; the caller decides the
    "keep every window" fallback, preserving each caller's existing behavior
    (``compute.py`` passes ``None`` straight to ``cluster_pass_speakers``;
    ``build.py`` expands it to an all-``True`` mask).
    """
    reference = next((w for w in per_model_windows.values() if w), [])
    mask = speech_window_mask_for_file(
        entries=reference,
        pass_summary=pass_summary,
        speech_presence_labels=list(speech_presence_labels),
    )
    return reference, mask
