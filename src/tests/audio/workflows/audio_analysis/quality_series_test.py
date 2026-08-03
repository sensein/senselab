"""Quality measurements emitted at native resolution, one series per target (D-20, D-25).

The old path packed seven quantities in three unit systems into one row — `units: "mixed"` was the
honest admission — and resampled them onto a reporting grid handed to the producer. Both are L2
decisions made at L1: which grid, which rule onto it, and which quantities belong together.

Here each target is its own `Series` at the grid this module measures on, and the consumer asks.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.scene_quality.brouhaha import BrouhahaFrames
from senselab.audio.workflows.audio_analysis.quality import (
    QUALITY_ANALYSIS_HOP_S,
    QUALITY_ANALYSIS_WIN_S,
    quality_series,
)
from senselab.audio.workflows.audio_analysis.shapes import Series


def _audio(seconds: float = 2.0, sr: int = 16000) -> Audio:
    rng = np.random.default_rng(0)
    wave = rng.normal(0.0, 0.05, int(seconds * sr)).astype("float32")
    return Audio(waveform=torch.from_numpy(wave).unsqueeze(0), sampling_rate=sr)


def _brouhaha(seconds: float = 2.0, hop: float = 0.01) -> BrouhahaFrames:
    n = int(seconds / hop)
    return BrouhahaFrames(
        vad=np.full(n, 0.9),
        snr_db=np.full(n, 22.0),
        c50_db=np.full(n, 41.0),
        frame_hop_s=hop,
    )


def test_each_target_is_its_own_series_with_its_own_units() -> None:
    """`units: "mixed"` was seven quantities in three unit systems sharing one row."""
    series = quality_series(audio=_audio(), brouhaha=_brouhaha())
    assert series, "expected measurements"
    assert all(isinstance(s, Series) for s in series.values())
    units = {name: s.units for name, s in series.items()}
    assert units["snr_brouhaha_db"] == "dB"
    assert units["proportion_clipped"] != units["snr_brouhaha_db"], "different quantities, different units"
    assert "mixed" not in set(units.values())


def test_the_series_are_at_the_analysis_grid_not_a_reporting_grid() -> None:
    """No grid argument, so no producer-side resampling — the whole of D-25."""
    series = quality_series(audio=_audio(), brouhaha=_brouhaha())
    for s in series.values():
        assert s.hop_s == pytest.approx(QUALITY_ANALYSIS_HOP_S)
        assert s.window_s == pytest.approx(QUALITY_ANALYSIS_WIN_S)


def test_window_and_hop_both_survive_so_overlap_is_visible() -> None:
    """0.5 s windows at a 0.25 s hop share half their audio.

    A consumer treating adjacent values as independent samples is wrong, and can only know that if
    both numbers travel. On a resampled row they did not.
    """
    s = next(iter(quality_series(audio=_audio(), brouhaha=_brouhaha()).values()))
    assert s.window_s > s.hop_s, "these windows overlap, and the series says so"


def test_brouhaha_targets_are_absent_when_the_model_did_not_load() -> None:
    """Absent, not present-and-null: a model that could not load has not measured nothing."""
    series = quality_series(audio=_audio(), brouhaha=None)
    assert "snr_brouhaha_db" not in series
    assert "c50_brouhaha_db" not in series
    assert "proportion_clipped" in series, "the DSP measurements still land"


def test_a_zero_length_audio_yields_no_series() -> None:
    """Nothing to measure is not a measurement of nothing."""
    empty = Audio(waveform=torch.zeros(1, 0), sampling_rate=16000)
    assert quality_series(audio=empty, brouhaha=None) == {}


def test_the_series_can_be_sampled_onto_any_consumer_grid() -> None:
    """One native emission, two consumers, two grids, no producer change."""
    from senselab.audio.workflows.audio_analysis.keys import DerivativeKey, Operator, Route, SignalKey
    from senselab.audio.workflows.audio_analysis.sampler import Sampler

    series = quality_series(audio=_audio(seconds=2.0), brouhaha=_brouhaha(seconds=2.0))
    key = SignalKey("snr", "pyannote/brouhaha", Route())
    sampler = Sampler({key: series["snr_brouhaha_db"]})
    query = DerivativeKey("snr", Operator("resample", "mean"), sources=(key,))
    coarse = sampler.on_grid(query, duration_s=2.0, win_length=0.5, hop_length=0.5)
    fine = sampler.on_grid(query, duration_s=2.0, win_length=0.1, hop_length=0.1)
    assert len(coarse) == 4
    assert len(fine) == 20
    assert coarse[0]["value"] == pytest.approx(22.0), "a constant SNR survives any grid"
