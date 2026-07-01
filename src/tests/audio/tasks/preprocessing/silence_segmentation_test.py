"""Unit tests for general pause-aware audio segmentation (pure CPU, synthetic)."""

from typing import List, Tuple

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import (
    pause_aware_boundaries,
    segment_audios_at_pauses,
)

SR = 16000


def _make_audio(
    duration_s: float, silence_times: Tuple[float, ...] = (), silence_dur: float = 0.6, loud: float = 0.5
) -> Audio:
    """Noise 'speech' of ``loud`` amplitude with exact-zero silence gaps inserted."""
    n = int(duration_s * SR)
    gen = torch.Generator().manual_seed(0)
    wf = (torch.rand(n, generator=gen) * 2 - 1) * loud
    for t in silence_times:
        s, e = int(t * SR), int((t + silence_dur) * SR)
        wf[s:e] = 0.0
    return Audio(waveform=wf.unsqueeze(0), sampling_rate=SR, metadata={})


def _assert_valid_tiling(spans: List[Tuple[float, float]], duration: float, max_seg: float) -> None:
    """Spans must start at 0, end at duration, be contiguous, and each be <= max_seg."""
    assert spans[0][0] == 0.0
    assert abs(spans[-1][1] - duration) < 1e-6
    for i in range(len(spans) - 1):
        assert abs(spans[i][1] - spans[i + 1][0]) < 1e-6, "spans must be contiguous"
    for s, e in spans:
        assert e - s <= max_seg + 1e-3, f"segment {(s, e)} exceeds max {max_seg}"
        assert e > s


def test_short_audio_single_segment() -> None:
    """Audio within the cap returns one full-length span for every strategy."""
    audio = _make_audio(10.0)
    for strategy in ("none", "greedy", "dp"):
        assert pause_aware_boundaries(audio, max_segment_s=38.0, strategy=strategy) == [(0.0, 10.0)], strategy


def test_none_strategy_never_splits() -> None:
    """The 'none' strategy returns a single span regardless of duration."""
    audio = _make_audio(20.0, silence_times=(4, 8, 12, 16))
    assert pause_aware_boundaries(audio, max_segment_s=5.0, strategy="none") == [(0.0, 20.0)]


def test_greedy_and_dp_valid_tiling_with_silences() -> None:
    """Both strategies produce a valid, bounded tiling of audio with pauses."""
    audio = _make_audio(20.0, silence_times=(4, 8, 12, 16))
    for strategy in ("greedy", "dp"):
        spans = pause_aware_boundaries(audio, max_segment_s=5.0, strategy=strategy)
        _assert_valid_tiling(spans, 20.0, 5.0)
        assert len(spans) >= 4


def test_cuts_land_in_silence() -> None:
    """Interior cut points fall near the inserted silences, not mid-'speech'."""
    silence_times = (4, 8, 12, 16)
    centers = [t + 0.3 for t in silence_times]
    audio = _make_audio(20.0, silence_times=silence_times)
    for strategy in ("greedy", "dp"):
        spans = pause_aware_boundaries(audio, max_segment_s=5.0, strategy=strategy)
        for cut in [e for s, e in spans[:-1]]:
            assert min(abs(cut - c) for c in centers) < 0.7, f"{strategy} cut {cut} not near a silence"


def test_pauseless_run_forces_bounded_cuts() -> None:
    """With no silence anywhere, cuts are forced but every segment stays <= max."""
    audio = _make_audio(20.0)
    for strategy in ("greedy", "dp"):
        _assert_valid_tiling(pause_aware_boundaries(audio, max_segment_s=5.0, strategy=strategy), 20.0, 5.0)


def test_greedy_minimizes_segment_count() -> None:
    """Greedy packs to the farthest pause, so it never uses more segments than DP."""
    audio = _make_audio(30.0, silence_times=(4, 8, 12, 16, 20, 24, 28))
    g = pause_aware_boundaries(audio, max_segment_s=5.0, strategy="greedy")
    d = pause_aware_boundaries(audio, max_segment_s=5.0, strategy="dp")
    assert len(g) <= len(d)


def test_cut_penalty_controls_segment_count() -> None:
    """A larger DP cut penalty yields no more segments than a small one."""
    audio = _make_audio(30.0, silence_times=tuple(range(2, 30)))  # many pauses to choose from
    few = pause_aware_boundaries(audio, max_segment_s=8.0, strategy="dp", cut_penalty=1.0)
    many = pause_aware_boundaries(audio, max_segment_s=8.0, strategy="dp", cut_penalty=0.0)
    assert len(few) <= len(many)


def test_unknown_strategy_raises() -> None:
    """An unrecognized strategy name is rejected."""
    with pytest.raises(ValueError, match="strategy"):
        pause_aware_boundaries(_make_audio(5.0), max_segment_s=38.0, strategy="bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize("max_seg", [0.99, 0.8, 0.3, 0.1])
def test_dp_tiles_sub_second_windows(max_seg: float) -> None:
    """DP must tile into <= max_seg spans even for small windows.

    Regression: forced-cut positions are built by repeated addition of
    ``max_seg``, so an exact-float feasibility guard (``ti - t_j > max_seg``)
    tripped on accumulated rounding (e.g. ``0.30000000000000004 > 0.3``),
    severed the DP chain, and returned a SINGLE full-length span far exceeding
    the cap with no error. A comparison tolerance fixes it.
    """
    audio = _make_audio(30.0)  # pause-less -> all cuts are forced at the cap
    spans = pause_aware_boundaries(audio, max_segment_s=max_seg, strategy="dp")
    _assert_valid_tiling(spans, 30.0, max_seg)
    assert len(spans) > 1, "DP collapsed to a single oversized span"


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_nonpositive_max_segment_raises(bad: float) -> None:
    """A non-positive max_segment_s is rejected (would otherwise infinite-loop)."""
    for strategy in ("greedy", "dp"):
        with pytest.raises(ValueError, match="max_segment_s"):
            pause_aware_boundaries(_make_audio(10.0), max_segment_s=bad, strategy=strategy)


def test_segment_audios_at_pauses_returns_subaudios() -> None:
    """segment_audios_at_pauses returns sub-Audios whose durations tile each input."""
    audio = _make_audio(20.0, silence_times=(4, 8, 12, 16))
    out = segment_audios_at_pauses([audio], max_segment_s=5.0, strategy="dp")
    assert len(out) == 1
    segs = out[0]
    assert len(segs) >= 4
    total = sum(s.waveform.shape[1] for s in segs)
    assert abs(total - audio.waveform.shape[1]) <= 1  # samples conserved (±rounding)
    for s in segs:
        assert s.waveform.shape[1] / SR <= 5.0 + 1e-3
