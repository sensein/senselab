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
- **AST / YAMNet**: the share of class-score mass on ``speech_presence_labels``.
  Not ``top-1 ∈ labels``: the argmax discards the rest of several hundred
  scores, so a window topped by ``Music`` at 0.40 with ``Speech`` second at
  0.38 used to vote a confident *no speech*.
- **Acoustic / LUFS** (BS.1770 gated loudness): absolute level, so the same
  loudness always reads the same. Replaces the percentile-ranked
  ``Loudness_sma3`` voter (D-3).
- **Acoustic / level above floor**: dB above this recording's own measured
  noise floor, and therefore gain-invariant — the question LUFS cannot answer.
  Replaces the percentile-ranked ``spectralFlux_sma3`` voter.
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

from typing import TYPE_CHECKING, Any

import numpy as np

from senselab.audio.workflows.audio_analysis.acoustic import (
    level_above_floor_track,
    loudness_confidence,
    lufs_track,
)
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.harvesters import (
    classification_windows,
    diar_covered_fraction,
    diar_speaker_label_in_window,
    diar_speaks_in_window,
    resolve_asr_result,
    token_overlaps_window,
    whisper_bucket_avg_logprob,
    whisper_bucket_confidence,
    whisper_bucket_no_speech_prob,
)
from senselab.audio.workflows.audio_analysis.sound_sources import window_label_mass

if TYPE_CHECKING:
    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import FramePosterior

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


_FRAME_SPEECH_THRESHOLD = 0.5
"""Bucket-mean frame posterior above which the frame voter says speech. Named rather than inlined
so the threshold is findable, and the mean it is applied to is recorded next to the verdict."""

_NO_SPEECH_THRESHOLD = 0.5
"""Whisper ``no_speech_prob`` above which a transcript over this bucket is treated as
hallucinated. A calibration choice, named here rather than inlined so it is findable and so the
measurement it is applied to travels alongside the verdict (register items 3, 5)."""

_WHISPER_SEGMENT_S = 30.0
"""Whisper's segment length. Recorded on the vote so a consumer can see that one value spans many
fine buckets, instead of inferring it from a hard-coded weight (register item 15)."""

SPEECH_EXCESS_DB = 12.0
"""dB above the measured noise floor at which a frame reads as clearly active.

Speech typically sits 12-20 dB above a room's floor. An absolute anchor in dB, so unlike the
percentile band it replaces the same excess always gives the same answer."""


def _excess_confidence(excess_db: float, *, speech_excess_db: float = SPEECH_EXCESS_DB) -> float:
    """Map dB-above-floor to ``P(active)``, abstaining at ``0.5`` when the excess is low.

    Asymmetric on purpose, and for the same reason HNR is: a **low** excess has two
    indistinguishable causes. Either nothing is happening, or a source runs through the whole
    recording and has been absorbed into its own floor estimate — the floor is a percentile of this
    file's own frames, so a signal that never stops *is* the floor. Voting ``False`` there would
    make this signal contradict correct models on any recording without pauses.

    A **high** excess has only one cause, so the upper half of the range is asserted normally.
    LUFS carries the ability to claim absence, since an absolute level of −90 LUFS is unambiguous.
    """
    if not np.isfinite(excess_db):
        return 0.5
    span = max(1e-9, float(speech_excess_db))
    return float(0.5 + 0.5 * max(0.0, min(1.0, float(excess_db) / span)))


def _track_mean_in_window(
    track: tuple[np.ndarray, np.ndarray] | None,
    start: float,
    end: float,
) -> float | None:
    """Mean of a ``(times, values)`` track over samples inside ``[start, end)``.

    ``None`` when the track is absent or no sample falls in the window, so a bucket with no
    measurement drops the vote rather than contributing a fabricated one.
    """
    if track is None:
        return None
    times, values = track
    if times.size == 0 or values.size == 0:
        return None
    sel = (times >= start) & (times < end)
    if not sel.any():
        # Fall back to the nearest sample: a bucket finer than the track's hop would otherwise
        # never be covered, which would silently disable the voter on fine grids.
        idx = int(np.argmin(np.abs(times - 0.5 * (start + end))))
        return float(values[idx])
    return float(np.nanmean(values[sel]))


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


def harvest_speech_presence_votes(
    *,
    pass_summary: dict[str, Any],
    grid: BucketGrid,
    speech_presence_labels: list[str],
    alignment_by_model: dict[str, Any],
    per_window_embeddings: dict[str, list[Any]] | None = None,
    frame_posteriors: dict[str, "FramePosterior"] | None = None,
    waveform: "np.ndarray | None" = None,
    sampling_rate: int | None = None,
) -> list[dict[str, Any]]:
    """Yield ``{"start", "end", "votes", "frame_instability"}`` per bucket for speech_presence.

    ``votes`` is a dict ``{model_id → {"speaks": bool, "native_confidence": float | None}}``
    spanning every contributing model. Buckets where no model contributed any vote are
    still emitted (caller drops them in compute._row_emit if status != ok).

    ``frame_posteriors`` maps a voter name → continuous per-frame P(speech)
    (``segmentation-3.0`` raw scores, Brouhaha VAD head). Each contributes a
    fine-resolution voter (bucket-mean P(speech)); the within-bucket frame std
    is aggregated into ``frame_instability`` for the ``speech_presence_uncertainty``
    column. On grids finer than ``_FINE_GRID_THRESHOLD_S`` the coarse voters are
    down-weighted to ``_COARSE_VOTER_WEIGHT`` (FR-014).

    ``waveform`` / ``sampling_rate`` enable the two absolute-scale acoustic voters
    (``acoustic_lufs``, ``acoustic_level_above_floor``). Both are computed from audio rather than
    from the openSMILE feature table, because the table's ``Loudness_sma3`` has no absolute
    reference and the percentile band that compensated for that is what broke it (D-3). Omit them
    and those voters are simply absent.
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
    # rows are per-asr not per-bucket; not used as a voter.)
    feat_block = pass_summary.get("features") or {}
    feat_result = feat_block.get("result") if isinstance(feat_block, dict) else None
    opensmile_rows: list[dict[str, Any]] = feat_result.get("opensmile", []) if isinstance(feat_result, dict) else []

    # HNR uses fixed dB thresholds — those ARE absolutely calibrated (it's a
    # ratio in dB that doesn't depend on input gain). Lowered the high anchor
    # to 10 dB per agent review (#7) — typical conversational HNR is 8–14 dB.
    #
    # ``Loudness_sma3`` and ``spectralFlux_sma3`` are gone (D-3). Both had no absolute
    # reference, which forced a per-pass percentile band: a 10th-percentile floor and a
    # 75th-percentile ceiling. That makes the value a *rank*, not a level — ~10% of frames pin at
    # 0 and ~25% at 1.0 by construction regardless of the audio, a uniformly quiet recording still
    # spreads to fill [0, 1] so quiet frames read as loud, and the mapping differs per file so the
    # value cannot be compared to dBFS or to a fixed threshold. They are replaced by
    # ``acoustic_lufs`` (absolute level) and ``acoustic_level_above_floor`` (level above this
    # recording's own noise floor, gain-invariant) — two questions the single rank conflated.
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

    # Embedding-cluster silhouette as a voice speech_presence signal. Cluster the
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

    # One pass over the waveform for each absolute acoustic track, before the bucket loop: both
    # are whole-recording measurements (the floor estimate needs the whole distribution).
    lufs: tuple[np.ndarray, np.ndarray] | None = None
    level_above_floor: tuple[np.ndarray, np.ndarray] | None = None
    if waveform is not None and sampling_rate:
        mono = np.asarray(waveform, dtype=np.float64).squeeze()
        if mono.ndim > 1:
            mono = mono.mean(axis=0)
        if mono.size:
            lufs = lufs_track(mono, int(sampling_rate), hop_s=0.05)
            level_above_floor = level_above_floor_track(mono, int(sampling_rate), hop_s=0.05)

    allow = set(speech_presence_labels)
    out: list[dict[str, Any]] = []
    for start, end, _idx in grid.iter_buckets(duration_s):
        votes: dict[str, dict[str, Any]] = {}

        # Diar — the model reports segments, and a segment boundary is a hard claim, so there is
        # no native confidence to report. What the vote carries beyond the bool is *which* speaker
        # was claimed and how much of the bucket the segment covers: a segment overlapping 5% of a
        # bucket and one covering all of it are not the same evidence, and the bool erases that.
        for m, block in diar_ok.items():
            covered = diar_covered_fraction(block.get("result"), start, end)
            votes[m] = {
                "speaks": diar_speaks_in_window(block.get("result"), start, end),
                "native_confidence": None,
                "covered_fraction": covered,
                "speaker_label": diar_speaker_label_in_window(block.get("result"), start, end),
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
            avg_lp = whisper_bucket_avg_logprob(resolved, start, end)
            # The hallucination override is a threshold (nsp >= 0.5) applied to a measurement,
            # so the threshold's *verdict* is recorded but the measurement travels with it: L2 can
            # re-decide where the boundary sits, which it cannot do from the bool alone.
            hallucinated = bool(speaks and nsp is not None and nsp >= _NO_SPEECH_THRESHOLD)
            asr_vote: dict[str, Any] = {
                "speaks": speaks and not hallucinated,
                "native_confidence": nc,
                "hallucinated": hallucinated,
                "coarse": True,  # sentence-level: one word spans many fine buckets
                "native_window_s": None,  # utterance-scoped; no fixed window
            }
            if nsp is not None:
                asr_vote["no_speech_prob"] = float(nsp)
            if avg_lp is not None:
                # The raw log-probability as well as the exp()'d confidence. They are different
                # scales, and only the raw value is the model's own output.
                asr_vote["avg_logprob"] = float(avg_lp)
            votes[m] = asr_vote
            # Whisper-only: dedicated VAD-like vote from the model's own ``no_speech_prob`` head,
            # independent of token overlap (which measures whether the transcript landed here).
            if nsp is not None:
                votes[f"{m}::no_speech_prob"] = {
                    "speaks": nsp < _NO_SPEECH_THRESHOLD,
                    "native_confidence": 1.0 - float(nsp),
                    "no_speech_prob": float(nsp),
                    "coarse": True,  # per-~30 s Whisper segment
                    "native_window_s": _WHISPER_SEGMENT_S,
                }

        # AST / YAMNet — project the bucket's CENTER onto the nearest native
        # window (round-to-nearest, not floor). With AST's 10.24 s windows a
        # bucket straddling a window boundary should use the window that covers
        # most of the bucket, not the one whose start happens to be lower.
        bucket_center = 0.5 * (start + end)
        for name, block, ok, hop, native_win in (
            ("ast", ast_block, ast_ok, ast_hop if ast_ok else 0.0, 10.24),
            ("yamnet", yam_block, yam_ok, yam_hop if yam_ok else 0.0, 0.96),
        ):
            if not ok:
                continue
            idx = max(0, int(round(bucket_center / hop))) if hop > 0 else 0
            windows = classification_windows(block.get("result"))
            if idx >= len(windows):
                continue
            # Mass over the speech label subset, not ``top-1 in subset``. The argmax discards the
            # rest of several hundred class scores: a window topped by ``Music`` at 0.40 with
            # ``Speech`` second at 0.38 previously voted a confident *no speech*.
            mass = window_label_mass(windows[idx], allow)
            if mass is None:
                continue
            votes[name] = {
                "speaks": mass >= 0.5,
                "native_confidence": mass if mass >= 0.5 else 1.0 - mass,
                "speech_label_mass": mass,
                "coarse": True,  # one native window spans many fine buckets
                "native_window_s": native_win,
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
            hnr = _mean_col(bucket_rows, "HNRdBACF_sma3nz")
            cal = _calibrate_uninformative_low(hnr, low=hnr_low, high=hnr_high)
            if cal is not None:
                speaks_v, p_v = cal
                votes["acoustic_hnr"] = _vote_from_pvoice(p_v, speaks_v)

        # Absolute-scale acoustic voters (D-3). Both carry their measurement alongside the vote:
        # LUFS and dB-above-floor mean something outside this recording, unlike the rank they
        # replace, so recording them lets a consumer check the mapping rather than trust it.
        for name, track, to_confidence, units_key in (
            ("acoustic_lufs", lufs, loudness_confidence, "lufs"),
            ("acoustic_level_above_floor", level_above_floor, _excess_confidence, "excess_db"),
        ):
            value = _track_mean_in_window(track, start, end)
            if value is None:
                continue
            p_v = to_confidence(value)
            speaks_v = p_v >= 0.5
            vote = _vote_from_pvoice(p_v, speaks_v)
            vote[units_key] = value
            votes[name] = vote

        # Note: parselmouth ``phonation_ratio`` is computed once per asr
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
        # logic as the speaker harvester's per-bucket lookup).
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
        # within-bucket temporal-instability signal for speech_presence_uncertainty.
        frame_stds: list[float] = []
        if frame_posteriors:
            for name, fp in frame_posteriors.items():
                if fp is None:
                    continue
                mean, std = fp.mean_std_in_window(start, end)
                if not np.isfinite(mean):
                    continue
                frame_vote: dict[str, Any] = {
                    "speaks": mean >= _FRAME_SPEECH_THRESHOLD,
                    "native_confidence": float(mean),
                    # The bucket mean is a reduction over frames, so record what it reduced: the
                    # frame count and the within-bucket dispersion in probability units. A mean of
                    # 0.5 from two steady frames and one from an onset crossing the bucket are
                    # different evidence, and the mean alone cannot distinguish them.
                    "frame_mean": float(mean),
                    "frame_std": float(std) if np.isfinite(std) else None,
                    "native_window_s": float(fp.frame_win_s) or None,
                    "resolution_s": float(fp.frame_hop_s) or None,
                }
                per_channel = fp.per_channel_mean_in_window(start, end)
                if per_channel is not None and per_channel.size > 1:
                    # Per-speaker activations kept intact (D-5): which channel was active is what
                    # the pooled value discards, and it is what per-speaker presence needs.
                    frame_vote["channel_means"] = [float(x) for x in per_channel]
                    frame_vote["channel_labels"] = list(fp.channel_labels)
                votes[name] = frame_vote
                if np.isfinite(std):
                    frame_stds.append(float(std))

        # Demote coarse voters on fine reporting grids (FR-014). No-op at the
        # historical 0.5 s grid, so legacy outputs are unchanged.
        if grid.win_length < _FINE_GRID_THRESHOLD_S:
            for v in votes.values():
                if isinstance(v, dict) and v.get("coarse"):
                    v["weight"] = _COARSE_VOTER_WEIGHT

        # Within-bucket temporal dispersion, in probability units, NOT rescaled.
        #
        # This was ``clip(2 * mean(std), 0, 1)``. The ×2 exists because the std of a value bounded
        # in [0, 1] is at most 0.5, so doubling maps it onto [0, 1] — but that turns a dispersion
        # into something that reads like a probability, and the clip then hides the cases where the
        # rescale was wrong. Per ``statistics.py``, variability is reported in the units of the
        # quantity and deliberately not squeezed into [0, 1]; rescaling makes it a different
        # statistic and invites reading it as a belief. L2 decides how dispersion enters a belief.
        frame_dispersion = float(np.mean(frame_stds)) if frame_stds else None

        out.append({"start": start, "end": end, "votes": votes, "frame_dispersion": frame_dispersion})

    return out
