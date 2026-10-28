"""Tests for objective SQUIM over fixed windows of a span, with a stubbed model."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction import torchaudio_squim
from senselab.audio.tasks.features_extraction.torchaudio_squim import (
    extract_objective_quality_features_from_audios,
    plan_windows,
    window_sample_bounds,
)

RATE = 16_000

pytestmark = pytest.mark.skipif(not torchaudio_squim.TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed")


class _StubModel(torch.nn.Module):
    """Reads ``stoi`` as the window's length in seconds, ``pesq`` as its first sample, ``si_sdr`` as its mean."""

    def __init__(self) -> None:
        super().__init__()
        self.lengths: list[int] = []

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One reading per batch row."""
        self.lengths.append(int(waveform.shape[-1]))
        return (
            torch.tensor([waveform.shape[-1] / RATE]),
            waveform[:, 0].clone(),
            waveform.mean(dim=-1),
        )


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> _StubModel:
    """The stub model in place of the SQUIM objective head."""
    model = _StubModel()
    monkeypatch.setattr(torchaudio_squim, "_get_objective_model", lambda: model)
    return model


def _ramp(seconds: float) -> Audio:
    """A waveform whose value at each sample is its time in seconds, scaled to stay under 1."""
    n = int(round(seconds * RATE))
    return Audio(waveform=(torch.arange(n, dtype=torch.float32) / RATE / 100.0).unsqueeze(0), sampling_rate=RATE)


class TestPlanWindows:
    """Full windows from the start; a short tail folds or drops; a short span is one window; a tiny one none."""

    @staticmethod
    def _plan(start: float, end: float, short_tail: Any = "fold", window_s: float | None = 10.0) -> list:  # noqa: ANN401
        return plan_windows(start, end, window_s=window_s, min_window_s=3.0, floor_s=1.0, short_tail=short_tail)

    def test_full_windows_tile_from_the_start(self) -> None:
        """Exactly two windows, no tail."""
        assert self._plan(1.0, 21.0) == [(1.0, 11.0), (11.0, 21.0)]

    def test_a_tail_at_the_minimum_is_its_own_window(self) -> None:
        """A 3 s remainder is a window in both policies."""
        expected = [(0.0, 10.0), (10.0, 20.0), (20.0, 23.0)]
        assert self._plan(0.0, 23.0) == expected
        assert self._plan(0.0, 23.0, "drop") == expected

    def test_a_short_tail_folds_into_the_previous_window(self) -> None:
        """A 2.9 s remainder extends the last window."""
        assert self._plan(0.0, 22.9) == [(0.0, 10.0), (10.0, 22.9)]

    def test_a_short_tail_is_dropped_on_request(self) -> None:
        """Triage's policy: the remainder is not read."""
        assert self._plan(0.0, 22.9, "drop") == [(0.0, 10.0), (10.0, 20.0)]

    def test_a_span_between_the_minimum_and_one_window_is_one_window(self) -> None:
        """A 7 s span is one 7 s window."""
        assert self._plan(2.0, 9.0) == [(2.0, 9.0)]

    def test_a_span_under_the_minimum_is_one_window(self) -> None:
        """A 2 s span is one window over the whole span."""
        assert self._plan(4.0, 6.0) == [(4.0, 6.0)]

    def test_the_floor(self) -> None:
        """Exactly the floor is read; under it nothing is."""
        assert self._plan(4.0, 5.0) == [(4.0, 5.0)]
        assert self._plan(4.0, 4.99) == []

    def test_no_window_length_is_one_pass_above_the_floor(self) -> None:
        """``window_s=None`` reads the span whole, or not at all under the floor."""
        assert self._plan(0.0, 300.0, window_s=None) == [(0.0, 300.0)]
        assert self._plan(0.0, 0.5, window_s=None) == []

    def test_floating_point_edges_do_not_add_a_sliver(self) -> None:
        """30 s of 0.1 s steps is three windows."""
        end = sum([0.1] * 300)
        assert len(self._plan(0.0, end)) == 3
        assert len(self._plan(0.0, end, "drop")) == 3

    def test_inconsistent_lengths_are_refused(self) -> None:
        """The minimum may not exceed the window, and the window must be positive."""
        with pytest.raises(ValueError):
            plan_windows(0.0, 10.0, window_s=2.0, min_window_s=3.0, floor_s=1.0)
        with pytest.raises(ValueError):
            plan_windows(0.0, 10.0, window_s=0.0, min_window_s=0.0, floor_s=1.0)


def test_sample_bounds_round_the_start_down_and_the_end_up() -> None:
    """Clamped to the waveform."""
    assert window_sample_bounds(0.5, 1.00001, RATE, 20_000) == (8_000, 16_001)
    assert window_sample_bounds(1.0, 5.0, RATE, 20_000) == (16_000, 20_000)


class TestExtraction:
    """Medians over windows of a span, each window cut by sample index."""

    def test_the_result_structure(self, stub: _StubModel) -> None:
        """Medians, one record per window, the count and the span length."""
        [result] = extract_objective_quality_features_from_audios([_ramp(21.5)])
        assert set(result) == {"stoi", "pesq", "si_sdr", "windows", "windows_n", "span_s"}
        assert result["windows_n"] == 2 and result["span_s"] == pytest.approx(21.5)
        assert [(w["start_s"], w["end_s"]) for w in result["windows"]] == [(0.0, 10.0), (10.0, 21.5)]
        assert all(set(w) == {"start_s", "end_s", "stoi", "pesq", "si_sdr"} for w in result["windows"])
        assert stub.lengths == [160_000, 184_000]

    def test_the_values_are_the_medians_over_windows(self, stub: _StubModel) -> None:
        """Three windows: the median is the middle reading."""
        [result] = extract_objective_quality_features_from_audios([_ramp(32.0)])
        assert [w["stoi"] for w in result["windows"]] == pytest.approx([10.0, 10.0, 12.0])
        assert result["stoi"] == pytest.approx(10.0)
        assert result["pesq"] == pytest.approx(0.1, abs=1e-6)  # the 10 s window's first sample, 10/100

    def test_a_single_pass_reads_the_whole_span_once(self, stub: _StubModel) -> None:
        """``window_s=None`` is one window whose readings are the result."""
        audio = _ramp(25.0)
        [result] = extract_objective_quality_features_from_audios([audio], window_s=None)
        assert stub.lengths == [400_000]
        assert result["windows_n"] == 1
        assert result["si_sdr"] == pytest.approx(float(audio.waveform.mean()), abs=1e-6)
        assert result["windows"][0]["si_sdr"] == result["si_sdr"]

    def test_the_single_pass_equals_the_model_on_the_whole_waveform(self, stub: _StubModel) -> None:
        """No span and no window is the previous whole-waveform call."""
        audio = _ramp(4.37)
        [result] = extract_objective_quality_features_from_audios([audio], window_s=None, floor_s=0.0)
        stoi, pesq, si_sdr = _StubModel()(audio.waveform)
        assert (result["stoi"], result["pesq"], result["si_sdr"]) == (stoi.item(), pesq.item(), si_sdr.item())

    def test_a_span_is_windowed_in_audio_time(self, stub: _StubModel) -> None:
        """Windows carry the audio's own times, and the cut starts at the span."""
        [result] = extract_objective_quality_features_from_audios([_ramp(30.0)], start_s=2.0, end_s=24.5)
        assert [(w["start_s"], w["end_s"]) for w in result["windows"]] == [(2.0, 12.0), (12.0, 24.5)]
        assert result["windows"][0]["pesq"] == pytest.approx(0.02, abs=1e-6)
        assert result["span_s"] == pytest.approx(22.5)

    def test_a_span_shorter_than_the_minimum_is_one_window(self, stub: _StubModel) -> None:
        """A 2 s span is read whole."""
        [result] = extract_objective_quality_features_from_audios([_ramp(30.0)], start_s=5.0, end_s=7.0)
        assert result["windows_n"] == 1 and stub.lengths == [32_000]

    def test_a_span_under_the_floor_is_nan(self, stub: _StubModel) -> None:
        """Nothing is read and nothing is invented."""
        [result] = extract_objective_quality_features_from_audios([_ramp(0.6)])
        assert result["windows_n"] == 0 and result["windows"] == [] and stub.lengths == []
        assert all(math.isnan(result[head]) for head in ("stoi", "pesq", "si_sdr"))

    @pytest.mark.parametrize("bounds", [(-1.0, 5.0), (5.0, 5.0), (6.0, 5.0), (0.0, 31.0)])
    def test_a_span_outside_the_audio_is_refused(self, stub: _StubModel, bounds: tuple[float, float]) -> None:
        """``0 <= start_s < end_s <= duration`` or a ValueError naming the bounds."""
        with pytest.raises(ValueError, match="0 <= start_s < end_s <= duration"):
            extract_objective_quality_features_from_audios([_ramp(30.0)], start_s=bounds[0], end_s=bounds[1])

    def test_non_mono_and_wrong_rate_are_refused(self, stub: _StubModel) -> None:
        """The model's input contract is unchanged."""
        stereo = Audio(waveform=torch.zeros(2, RATE * 2), sampling_rate=RATE)
        with pytest.raises(ValueError, match="mono"):
            extract_objective_quality_features_from_audios([stereo])
        with pytest.raises(ValueError, match="16000"):
            extract_objective_quality_features_from_audios([Audio(waveform=torch.zeros(1, 8000), sampling_rate=8000)])
