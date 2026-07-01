"""Pause-aware segmentation: split audio into bounded chunks at silences.

A general utility for any task that must feed a model audio no longer than some
window (e.g. ASR backends with a fixed encoder/context limit such as Canary-Qwen
or Granite): it splits audio that exceeds ``max_segment_s`` into ``<=
max_segment_s`` pieces, choosing cut points **at pauses** (minimum short-time RMS
energy) so no word is cut. Pure ``torch`` — no model or external dependency.

Two strategies:

* ``"greedy"`` — from the previous cut, take the *farthest* pause within
  ``max_segment_s``. Minimizes the number of segments, but pins cuts near the cap
  so cut quality is whatever pause happens to sit there.
* ``"dp"`` — optimal segmentation: globally minimize
  ``sum(cut_badness) + cut_penalty * n_cuts`` subject to every segment
  ``<= max_segment_s``. Finds the deepest pauses anywhere and trades an extra
  segment for a much cleaner cut when it is worth it (``cut_penalty`` is the dial).

Both fall back to a forced minimum-energy cut across any pause-less run longer
than ``max_segment_s``. ``"none"`` disables splitting (single full-length span).
"""

from __future__ import annotations

from typing import List, Literal, Tuple

import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing.preprocessing import extract_segments

SegmentStrategy = Literal["none", "greedy", "dp"]

# Documented defaults (override per call):
DEFAULT_MIN_PAUSE_S = 0.20  # minimum silence duration to qualify as a pause candidate
DEFAULT_SILENCE_PERCENTILE = 15.0  # frames below this energy percentile are "quiet" (adaptive)
DEFAULT_CUT_PENALTY = 0.2  # DP per-cut penalty in normalized-badness units [0, 1]
DEFAULT_ENERGY_FRAME_S = 0.02  # short-time RMS frame size used to locate cut points

# Tolerance for the ``<= max_segment_s`` feasibility comparisons. Forced-cut
# positions are built by repeated addition of ``max_segment_s`` (see ``_dp``),
# so a span that is exactly ``max_segment_s`` long can read as e.g.
# ``0.30000000000000004`` and spuriously fail an exact ``> max_segment_s``
# guard — severing the DP chain and collapsing the output to one oversized span.
# 1e-6 s (well below one sample at any real rate, well above float-accumulation
# error over thousands of frames) absorbs the rounding without moving any real
# boundary.
_FEASIBILITY_EPS = 1e-6


def _rms_envelope(audio: Audio, frame_s: float) -> Tuple[torch.Tensor, int, int, int]:
    """Return (per-frame RMS, hop samples, sampling_rate, n_samples)."""
    sr = int(audio.sampling_rate)
    mono = audio.waveform.mean(dim=0)
    n_samples = int(mono.shape[0])
    hop = max(1, int(round(frame_s * sr)))
    n_frames = n_samples // hop
    if n_frames < 1:
        return torch.zeros(0), hop, sr, n_samples
    rms = mono[: n_frames * hop].reshape(n_frames, hop).pow(2).mean(dim=1).sqrt()
    return rms, hop, sr, n_samples


def _normalized(rms: torch.Tensor) -> torch.Tensor:
    """Map RMS to [0, 1] badness via the 95th percentile (speech~1, silence~0)."""
    if rms.numel() == 0:
        return rms
    ref = float(torch.quantile(rms, 0.95).item()) or float(rms.max().item()) or 1.0
    return (rms / ref).clamp(0.0, 1.0)


def _frame_time(frame_idx: int, hop: int, sr: int) -> float:
    """Center time (seconds) of a frame."""
    return (frame_idx * hop + hop / 2) / sr


def _pause_candidates(
    rms: torch.Tensor, hop: int, sr: int, min_pause_s: float, silence_percentile: float
) -> List[Tuple[float, float]]:
    """Return ``(time_s, badness)`` at the deepest frame of each detected pause."""
    if rms.numel() == 0:
        return []
    norm = _normalized(rms)
    thr = float(torch.quantile(rms, silence_percentile / 100.0).item())
    quiet = rms <= thr
    min_frames = max(1, int(round(min_pause_s * sr / hop)))
    out: List[Tuple[float, float]] = []
    n = rms.shape[0]
    i = 0
    while i < n:
        if not bool(quiet[i]):
            i += 1
            continue
        j = i
        while j < n and bool(quiet[j]):
            j += 1
        if (j - i) >= min_frames:
            k = i + int(rms[i:j].argmin().item())
            out.append((_frame_time(k, hop, sr), float(norm[k].item())))
        i = j
    return out


def _forced_cut(rms: torch.Tensor, hop: int, sr: int, lo_s: float, hi_s: float) -> Tuple[float, float]:
    """Quietest ``(time_s, badness)`` in ``[lo_s, hi_s]`` — fallback when no pause exists."""
    norm = _normalized(rms)
    lo_f = max(0, int(lo_s * sr / hop))
    hi_f = min(rms.shape[0], max(lo_f + 1, int(hi_s * sr / hop)))
    seg = rms[lo_f:hi_f]
    if seg.numel() == 0:
        return hi_s, 1.0
    k = lo_f + int(seg.argmin().item())
    return _frame_time(k, hop, sr), float(norm[k].item())


def _spans_from_cuts(cuts: List[float], duration: float) -> List[Tuple[float, float]]:
    pts = sorted({0.0, *[c for c in cuts if 0.0 < c < duration], duration})
    return [(pts[k], pts[k + 1]) for k in range(len(pts) - 1)]


def _greedy(
    rms: torch.Tensor, hop: int, sr: int, duration: float, max_seg: float, cand_times: List[float]
) -> List[float]:
    cuts: List[float] = []
    prev = 0.0
    while duration - prev > max_seg:
        feasible = [t for t in cand_times if prev < t <= prev + max_seg]
        if feasible:
            cut = max(feasible)
        else:
            cut, _ = _forced_cut(rms, hop, sr, prev + 0.5 * max_seg, prev + max_seg)
            cut = min(max(cut, prev + 1.0), prev + max_seg)
        cuts.append(cut)
        prev = cut
    return cuts


def _dp(
    rms: torch.Tensor,
    hop: int,
    sr: int,
    duration: float,
    max_seg: float,
    candidates: List[Tuple[float, float]],
    cut_penalty: float,
) -> List[float]:
    # Candidate positions: start, pauses, end. Insert forced cuts so no two
    # consecutive candidates are farther than max_seg apart (guarantees feasibility).
    raw = [(0.0, 0.0)] + [(t, b) for t, b in candidates if 0.0 < t < duration] + [(duration, 0.0)]
    raw.sort()
    pts: List[Tuple[float, float]] = [raw[0]]
    for idx in range(1, len(raw)):
        while raw[idx][0] - pts[-1][0] > max_seg + _FEASIBILITY_EPS:
            ft, fb = _forced_cut(rms, hop, sr, pts[-1][0] + 0.5 * max_seg, pts[-1][0] + max_seg)
            ft = min(max(ft, pts[-1][0] + 1.0), pts[-1][0] + max_seg)
            pts.append((ft, fb))
        pts.append(raw[idx])

    m = len(pts)
    inf = float("inf")
    cost = [inf] * m
    back = [-1] * m
    cost[0] = 0.0
    for i in range(1, m):
        ti, bi = pts[i]
        is_end = i == m - 1
        for j in range(i - 1, -1, -1):
            if ti - pts[j][0] > max_seg + _FEASIBILITY_EPS:
                break
            add = 0.0 if is_end else (bi + cut_penalty)  # endpoint is not a cut
            c = cost[j] + add
            if c < cost[i]:
                cost[i] = c
                back[i] = j

    cuts: List[float] = []
    i = m - 1
    while i > 0:
        cuts.append(pts[i][0])
        i = back[i]
    return cuts


def pause_aware_boundaries(
    audio: Audio,
    max_segment_s: float,
    strategy: SegmentStrategy = "dp",
    *,
    min_pause_s: float = DEFAULT_MIN_PAUSE_S,
    silence_percentile: float = DEFAULT_SILENCE_PERCENTILE,
    cut_penalty: float = DEFAULT_CUT_PENALTY,
    energy_frame_s: float = DEFAULT_ENERGY_FRAME_S,
) -> List[Tuple[float, float]]:
    """Return ``(start, end)`` second-spans tiling ``audio`` with each span <= ``max_segment_s``.

    Cuts are placed at pauses (see module docstring). Audio at or under
    ``max_segment_s`` (or ``strategy="none"``) returns a single full-length span.
    """
    if max_segment_s <= 0:
        raise ValueError(f"max_segment_s must be strictly positive, got {max_segment_s!r}.")

    duration = audio.waveform.shape[1] / audio.sampling_rate
    if strategy == "none":
        return [(0.0, duration)]
    if strategy not in ("greedy", "dp"):
        raise ValueError(f"Unknown segmentation strategy {strategy!r}; expected 'none', 'greedy', or 'dp'.")

    rms, hop, sr, n = _rms_envelope(audio, energy_frame_s)
    if duration <= max_segment_s or rms.numel() == 0:
        return [(0.0, duration)]
    candidates = _pause_candidates(rms, hop, sr, min_pause_s, silence_percentile)

    if strategy == "greedy":
        cuts = _greedy(rms, hop, sr, duration, max_segment_s, [t for t, _ in candidates])
    else:
        cuts = _dp(rms, hop, sr, duration, max_segment_s, candidates, cut_penalty)
    return _spans_from_cuts(cuts, duration)


def segment_audios_at_pauses(
    audios: List[Audio],
    max_segment_s: float,
    strategy: SegmentStrategy = "dp",
    *,
    min_pause_s: float = DEFAULT_MIN_PAUSE_S,
    silence_percentile: float = DEFAULT_SILENCE_PERCENTILE,
    cut_penalty: float = DEFAULT_CUT_PENALTY,
    energy_frame_s: float = DEFAULT_ENERGY_FRAME_S,
) -> List[List[Audio]]:
    """Split each input audio at pauses into ``<= max_segment_s`` sub-Audios.

    Returns, per input audio, the list of its sub-Audios in time order. An audio
    that needs no splitting is returned as a single-element list holding the
    original object (no copy).
    """
    out: List[List[Audio]] = []
    for audio in audios:
        spans = pause_aware_boundaries(
            audio,
            max_segment_s,
            strategy,
            min_pause_s=min_pause_s,
            silence_percentile=silence_percentile,
            cut_penalty=cut_penalty,
            energy_frame_s=energy_frame_s,
        )
        out.append([audio] if len(spans) == 1 else extract_segments([(audio, spans)])[0])
    return out
