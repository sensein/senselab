"""L1 speech-presence evidence — what each tool measured, in its own units.

There is no presence at L1, only signals. This module runs the readers that project each model's
output onto a reporting bucket and records the numbers; whether a bucket contains speech is decided
in :mod:`speech_presence_link`, under a named policy. Nothing here thresholds, inverts, ranks, or
selects among estimators.

What each signal contributes, and what the measurement is:

- **Diarization models** — ``covered_fraction`` (union of segment overlap with the bucket, as a
  proportion) and ``speaker_label``. Replaces a ``speaks`` bool that could not distinguish a
  segment grazing 5% of a bucket from one covering all of it.
- **ASR models** — ``word_overlap_s`` and ``n_words`` (how much transcript actually lands here),
  the per-chunk ``avg_logprobs`` and ``no_speech_probs`` unpooled, and the unclipped
  ``claim_span_s`` / ``segment_span_s`` so how *wide* the claim is can be measured rather than
  declared.
- **Whisper's silence head** — ``no_speech_prob`` as a sibling signal keyed
  ``<asr_model>::no_speech_prob``, uninverted.
- **AST / YAMNet** — ``speech_label_mass``, the share of class-score mass on the speech label set.
  Not ``top-1 ∈ labels``: the argmax discards several hundred scores, so a window topped by
  ``Music`` at 0.40 with ``Speech`` second at 0.38 used to read as a confident *no speech*.
- **openSMILE HNR** — ``hnr_db``, a ratio in dB and therefore already absolutely calibrated.
- **LUFS** (BS.1770 gated loudness) — ``lufs``, an absolute level, so the same loudness always
  reads the same.
- **Level above floor** — ``excess_db`` above this recording's own measured noise floor, and
  therefore gain-invariant: the question LUFS cannot answer. Together these two replace a single
  per-pass percentile band that answered neither (D-3), since a rank cannot be compared to a fixed
  threshold or across files.
- **PPG** — ``mean_silence_posterior``, the model's own posterior mass on ``<silent>`` averaged
  over the bucket's frames, plus its dispersion and frame count. Not a count of frames whose
  argmax is not ``<silent>``: that collapses each frame's distribution to a hard verdict, the same
  reduction the scene-classifier top-1 made.
- **Frame posteriors** (``segmentation-3.0`` raw scores, Brouhaha's VAD head) — ``frame_mean``,
  ``frame_std``, ``n_frames``, and the per-speaker ``channel_means`` / ``channel_labels`` kept
  intact (D-5), plus the declared ``resolution_s`` and ``native_window_s``.

Windowed speaker embeddings are **not** read here. Clustering them is a derived signal per D-7 —
it needs the whole pass, and its output (per-window silhouette and cluster label) is a conclusion
about speaker structure rather than a measurement of this bucket. The vectors travel on
``PassHarvest.per_window_embeddings`` and ``speech_presence_link.derive_window_clusters`` clusters
them at L2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from senselab.audio.workflows.audio_analysis.acoustic import (
    level_above_floor_track,
    lufs_track,
)
from senselab.audio.workflows.audio_analysis.grid import BucketGrid
from senselab.audio.workflows.audio_analysis.harvesters import (
    asr_bucket_chunk_evidence,
    classification_windows,
    diar_covered_fraction,
    diar_speaker_label_in_window,
    resolve_asr_result,
)
from senselab.audio.workflows.audio_analysis.sound_sources import window_label_mass

if TYPE_CHECKING:
    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import FramePosterior

ACOUSTIC_TRACK_HOP_S = 0.05
"""Hop for the two whole-recording acoustic tracks, and their declared resolution."""

OPENSMILE_FRAME_S = 0.01
"""openSMILE's frame period, declared so L2 can see the resolution the HNR mean reduced over."""


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


def _track_mean_in_window(
    track: tuple[np.ndarray, np.ndarray] | None,
    start: float,
    end: float,
) -> float | None:
    """Mean of a ``(times, values)`` track over samples inside ``[start, end)``.

    ``None`` when the track is absent or no sample falls in the window, so a bucket with no
    measurement drops the signal rather than contributing a fabricated one.
    """
    if track is None:
        return None
    times, values = track
    if times.size == 0 or values.size == 0:
        return None
    sel = (times >= start) & (times < end)
    if not sel.any():
        # Fall back to the nearest sample: a bucket finer than the track's hop would otherwise
        # never be covered, which would silently disable the signal on fine grids.
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


def harvest_speech_presence_evidence(
    *,
    pass_summary: dict[str, Any],
    grid: BucketGrid,
    speech_presence_labels: list[str],
    alignment_by_model: dict[str, Any],
    frame_posteriors: dict[str, "FramePosterior"] | None = None,
    waveform: "np.ndarray | None" = None,
    sampling_rate: int | None = None,
) -> list[dict[str, Any]]:
    """Yield ``{"start", "end", "evidence", "frame_dispersion"}`` per bucket.

    ``evidence`` maps signal name → the measurements that signal produced for this bucket, in that
    signal's own units. No entry carries a verdict; :func:`~.speech_presence_link.link_speech_presence`
    turns these into votes. A signal that measured nothing in a bucket is absent from ``evidence``
    rather than present with a zero, because "no measurement" and "measured zero" license different
    conclusions.

    Args:
        pass_summary: One pass's per-task results.
        grid: Reporting grid for the speech-presence axis.
        speech_presence_labels: AudioSet labels whose score mass counts as speech.
        alignment_by_model: Per-ASR-model alignment blocks, used to recover timestamps for
            text-only backends.
        frame_posteriors: Signal name → continuous per-frame posteriors.
        waveform: Pass audio, enabling the two absolute-scale acoustic signals. Both are
            whole-recording measurements — the noise floor is a percentile over the whole
            distribution — so neither can be recovered from the per-frame openSMILE table the way
            the percentile-ranked signals they replace were (D-3). Omit and they are simply absent.
        sampling_rate: Sample rate of ``waveform``.

    Returns:
        One row per reporting bucket. ``frame_dispersion`` is the mean within-bucket frame standard
        deviation **in probability units and unrescaled**: per ``statistics.py`` a variability is
        reported in the units of the quantity, and squeezing it into ``[0, 1]`` would make it a
        different statistic and invite reading a dispersion as a belief.
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
    ast_win, ast_hop = _native_classification_grid(ast_block) if ast_ok else (0.0, 0.0)
    yam_win, yam_hop = _native_classification_grid(yam_block) if yam_ok else (0.0, 0.0)

    # Acoustic features — opensmile rows from the features task. (parselmouth rows are per-asr not
    # per-bucket; ``phonation_ratio`` is computed once over the whole audio by Praat's silence
    # TextGrid, so it does not vary by bucket and is not a per-bucket signal at all.)
    feat_block = pass_summary.get("features") or {}
    feat_result = feat_block.get("result") if isinstance(feat_block, dict) else None
    opensmile_rows: list[dict[str, Any]] = feat_result.get("opensmile", []) if isinstance(feat_result, dict) else []

    # PPG argmax-per-frame for the voice-fraction signal.
    ppg_block = pass_summary.get("ppgs") or pass_summary.get("ppg") or {}
    ppg_silence: np.ndarray = np.empty(0)
    ppg_frame_hop: float = 0.0
    if isinstance(ppg_block, dict) and ppg_block.get("status") == "ok":
        import sys as _sys

        from senselab.audio.workflows.audio_analysis.harvesters import ppg_silence_posterior_per_frame

        try:
            ppg_silence, ppg_frame_hop = ppg_silence_posterior_per_frame(
                ppg_block.get("result"),
                ppg_block.get("phoneme_labels"),
                duration_s,
            )
        except Exception as ppg_exc:  # noqa: BLE001
            # ``_to_2d_frame_major`` raises ValueError on ambiguous tensor shape — that's a real
            # configuration problem, surface it rather than silently disabling the signal.
            print(
                f"warn: PPG argmax decoding failed: {ppg_exc!r} — ppg_voice_fraction disabled for this pass",
                file=_sys.stderr,
            )
            ppg_silence, ppg_frame_hop = np.empty(0), 0.0

    # One pass over the waveform for each absolute acoustic track, before the bucket loop: both are
    # whole-recording measurements (the floor estimate needs the whole distribution).
    lufs: tuple[np.ndarray, np.ndarray] | None = None
    level_above_floor: tuple[np.ndarray, np.ndarray] | None = None
    if waveform is not None and sampling_rate:
        mono = np.asarray(waveform, dtype=np.float64).squeeze()
        if mono.ndim > 1:
            mono = mono.mean(axis=0)
        if mono.size:
            lufs = lufs_track(mono, int(sampling_rate), hop_s=ACOUSTIC_TRACK_HOP_S)
            level_above_floor = level_above_floor_track(mono, int(sampling_rate), hop_s=ACOUSTIC_TRACK_HOP_S)

    allow = set(speech_presence_labels)
    out: list[dict[str, Any]] = []
    for start, end, _idx in grid.iter_buckets(duration_s):
        evidence: dict[str, dict[str, Any]] = {}

        # ── Diarization ──────────────────────────────────────────────────
        for m, block in diar_ok.items():
            evidence[m] = {
                "covered_fraction": diar_covered_fraction(block.get("result"), start, end),
                "speaker_label": diar_speaker_label_in_window(block.get("result"), start, end),
                "units": "proportion",
            }

        # ── ASR ──────────────────────────────────────────────────────────
        for m, resolved in asr_resolved.items():
            chunk_ev = asr_bucket_chunk_evidence(resolved, start, end)
            evidence[m] = {
                **chunk_ev,
                "units": "second",
                # How wide the transcript evidence reaching this bucket is, measured. The old
                # harvester hand-marked this signal ``coarse`` with no window at all.
                "native_window_s": chunk_ev.get("claim_span_s"),
            }
            # Whisper-only sibling: the model's own silence head, independent of whether the
            # transcript landed here. Uninverted — ``speaks = nsp < t`` is L2's.
            nsps = chunk_ev.get("no_speech_probs") or []
            if nsps:
                evidence[f"{m}::no_speech_prob"] = {
                    "no_speech_prob": float(np.mean(nsps)),
                    "n_segments": len(nsps),
                    "units": "probability",
                    "native_window_s": chunk_ev.get("segment_span_s"),
                }

        # ── AST / YAMNet ─────────────────────────────────────────────────
        # Project the bucket's CENTER onto the nearest native window (round-to-nearest, not
        # floor). With AST's 10.24 s windows a bucket straddling a boundary should use the window
        # covering most of the bucket, not the one whose start happens to be lower.
        bucket_center = 0.5 * (start + end)
        for name, block, ok, hop, native_win in (
            ("ast", ast_block, ast_ok, ast_hop, ast_win),
            ("yamnet", yam_block, yam_ok, yam_hop, yam_win),
        ):
            if not ok:
                continue
            idx = max(0, int(round(bucket_center / hop))) if hop > 0 else 0
            windows = classification_windows(block.get("result"))
            if idx >= len(windows):
                continue
            mass = window_label_mass(windows[idx], allow)
            if mass is None:
                continue
            evidence[name] = {
                "speech_label_mass": mass,
                "units": "proportion",
                "native_window_s": native_win or None,
                "resolution_s": hop or None,
            }

        # ── openSMILE HNR ────────────────────────────────────────────────
        if opensmile_rows:
            hnr = _mean_col(_row_window_overlap(opensmile_rows, start, end), "HNRdBACF_sma3nz")
            if hnr is not None:
                evidence["acoustic_hnr"] = {
                    "hnr_db": hnr,
                    "units": "decibel",
                    "resolution_s": OPENSMILE_FRAME_S,
                }

        # ── Absolute-scale acoustic signals (D-3) ────────────────────────
        for name, track, value_key, units in (
            ("acoustic_lufs", lufs, "lufs", "lufs"),
            ("acoustic_level_above_floor", level_above_floor, "excess_db", "decibel"),
        ):
            value = _track_mean_in_window(track, start, end)
            if value is None:
                continue
            evidence[name] = {value_key: value, "units": units, "resolution_s": ACOUSTIC_TRACK_HOP_S}

        # ── PPG silence posterior ────────────────────────────────────────
        if ppg_silence.size and ppg_frame_hop > 0:
            first_frame = max(0, int(start / ppg_frame_hop))
            last_frame = min(ppg_silence.size, max(first_frame + 1, int(round(end / ppg_frame_hop))))
            if last_frame > first_frame:
                window = ppg_silence[first_frame:last_frame]
                evidence["ppg_voice_fraction"] = {
                    "mean_silence_posterior": float(np.mean(window)),
                    # The dispersion of the posterior across the bucket, in probability units and
                    # unrescaled, for the same reason the frame signals report theirs.
                    "silence_posterior_std": float(np.std(window)) if window.size > 1 else None,
                    "n_frames": int(window.size),
                    "units": "probability",
                    "resolution_s": ppg_frame_hop,
                }

        # ── Frame posteriors (segmentation-3.0, Brouhaha VAD) ────────────
        frame_stds: list[float] = []
        if frame_posteriors:
            for name, fp in frame_posteriors.items():
                if fp is None:
                    continue
                mean, std = fp.mean_std_in_window(start, end)
                if not np.isfinite(mean):
                    continue
                # The bucket mean is a reduction over frames, so record what it reduced over: a
                # mean of 0.5 from two steady frames and one from an onset crossing the bucket are
                # different evidence, and the mean alone cannot distinguish them.
                frame_ev: dict[str, Any] = {
                    "frame_mean": float(mean),
                    "frame_std": float(std) if np.isfinite(std) else None,
                    "units": "probability",
                    "native_window_s": float(fp.frame_win_s) or None,
                    "resolution_s": float(fp.frame_hop_s) or None,
                }
                per_channel = fp.per_channel_mean_in_window(start, end)
                if per_channel is not None and per_channel.size > 1:
                    # Per-speaker activations kept intact (D-5): which channel was active is what
                    # a pooled value discards, and it is what per-speaker presence needs.
                    frame_ev["channel_means"] = [float(x) for x in per_channel]
                    frame_ev["channel_labels"] = list(fp.channel_labels)
                evidence[name] = frame_ev
                if np.isfinite(std):
                    frame_stds.append(float(std))

        out.append(
            {
                "start": start,
                "end": end,
                "evidence": evidence,
                "frame_dispersion": float(np.mean(frame_stds)) if frame_stds else None,
            }
        )

    return out
