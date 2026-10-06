"""Tests for the cough-onset measure (``cough_pattern.py``) on synthetic subband levels."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from senselab.audio.workflows.triage.cough_pattern import (
    CoughPattern,
    band_levels_db,
    cough_parameters,
    cough_pattern_of,
    in_cough_review_band,
    measure_cough_pattern,
)
from senselab.utils.prov_store import ProvStore

HOP = 0.005
BANDS = 9
FLOOR = -60.0


def _levels(duration_s: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return FLOOR + rng.normal(0.0, 1.0, (BANDS, int(duration_s / HOP)))


def _cough(levels: np.ndarray, at_s: float, *, peak_db: float = 50.0, decay_s: float = 0.25) -> None:
    """A sharp rise across every band at ``at_s`` decaying to the floor over ``decay_s``."""
    start = int(at_s / HOP)
    span = np.arange(int(2 * decay_s / HOP))
    shape = FLOOR + peak_db * np.exp(-span * HOP / (decay_s / 3))
    end = min(levels.shape[1], start + len(span))
    levels[:, start:end] = np.maximum(levels[:, start:end], shape[: end - start])


def _ramp(levels: np.ndarray, start_s: float, end_s: float, level_db: float, bands: slice = slice(None)) -> None:
    """A gradual rise to ``level_db`` above the floor and back, over ``bands``."""
    a, b = int(start_s / HOP), int(end_s / HOP)
    shape = FLOOR + level_db * np.sin(np.linspace(0.0, np.pi, b - a))
    levels[bands, a:b] = np.maximum(levels[bands, a:b], shape)


def test_five_coughs_are_five_onsets() -> None:
    """53dddba6: five coughs about a second apart read as five onsets and one extent over them."""
    levels = _levels(6.0)
    for at in (0.5, 1.5, 2.6, 3.8, 4.9):
        _cough(levels, at)
    reading = measure_cough_pattern(levels, hop_s=HOP)
    assert reading.onsets_n == 5
    assert np.allclose(reading.onsets_s, (0.45, 1.45, 2.55, 3.75, 4.85), atol=0.06)
    assert reading.extent is not None
    assert reading.extent.start_s < 0.5 and reading.extent.end_s > 4.9


def test_fast_coughs_split_at_each_onset() -> None:
    """aad9e6c6: coughs under half a second apart still split at each onset."""
    levels = _levels(3.0)
    for at in (0.4, 0.8, 1.2, 1.6, 2.0):
        _cough(levels, at, decay_s=0.15)
    assert measure_cough_pattern(levels, hop_s=HOP).onsets_n == 5


def test_a_tail_belongs_to_its_cough() -> None:
    """45f1c7ec: a weaker burst out of a cough's decay is its tail, counted once and inside its event."""
    levels = _levels(2.0)
    _cough(levels, 0.5, peak_db=50.0, decay_s=0.3)
    _ramp(levels, 0.62, 1.1, 30.0)
    reading = measure_cough_pattern(levels, hop_s=HOP)
    assert reading.onsets_n == 1
    assert reading.event_spans_s[0][1] > 1.0


def test_a_preparatory_inhale_belongs_to_its_cough() -> None:
    """22c5f400 / 3b7ace99: the inhale before a cough starts that cough's event and is never counted."""
    levels = _levels(2.5)
    _ramp(levels, 0.2, 0.9, 25.0)
    _cough(levels, 1.0)
    reading = measure_cough_pattern(levels, hop_s=HOP)
    assert reading.onsets_n == 1
    assert 0.9 <= reading.onsets_s[0] <= 1.0
    assert reading.event_spans_s[0][0] < 0.35


def test_a_hum_makes_no_onset() -> None:
    """a28e5022: a narrowband hum switching on is stationary and tonal, not a cough."""
    levels = _levels(3.0)
    levels[2, int(1.0 / HOP) :] = FLOOR + 40.0
    assert measure_cough_pattern(levels, hop_s=HOP).onsets_n == 0


def test_silence_makes_no_onset() -> None:
    """cdfa7e4e and the other listened negatives: nothing there, no onset and no extent."""
    reading = measure_cough_pattern(_levels(3.0), hop_s=HOP)
    assert reading.onsets_n == 0
    assert reading.extent is None


def test_a_rise_out_of_leading_digital_silence_is_no_cough() -> None:
    """a28e5022: the recording's start transient rises out of digital silence."""
    levels = _levels(2.0)
    levels[:, :20] = -110.0
    _cough(levels, 0.1, decay_s=0.1)
    assert measure_cough_pattern(levels, hop_s=HOP).onsets_n == 0


def test_an_onset_beside_a_speech_word_is_speech() -> None:
    """A burst inside a spoken word is speech, not a cough."""
    levels = _levels(2.0)
    _cough(levels, 1.0)
    assert measure_cough_pattern(levels, hop_s=HOP, words=[(0.9, 1.3)]).onsets_n == 0


def test_a_single_cough_has_a_tight_extent() -> None:
    """b91cd93f: one cough, an extent over that cough alone."""
    levels = _levels(3.5)
    _cough(levels, 1.3)
    reading = measure_cough_pattern(levels, hop_s=HOP)
    assert reading.onsets_n == 1
    assert reading.extent is not None
    assert 1.1 <= reading.extent.start_s < 1.3 and reading.extent.end_s < 2.0


def test_the_review_band_reads_both_sides_of_the_count() -> None:
    """The decision is low-confidence where the strict and lenient counts straddle the need."""
    assert in_cough_review_band(CoughPattern(onsets_strict_n=2, onsets_lenient_n=3), 3)
    assert not in_cough_review_band(CoughPattern(onsets_strict_n=3, onsets_lenient_n=4), 3)


def test_band_levels_sum_each_band() -> None:
    """Each band sums its bins' power."""
    power = np.ones((161, 4))
    levels = band_levels_db(power, bin_hz=50.0, band_edges_hz=(150.0, 300.0))
    assert np.allclose(levels, 10 * np.log10(4.0))


def test_an_absent_spectrogram_names_itself(tmp_path: Path) -> None:
    """No stored spectrogram: the measure could not look, and says what it lacked."""
    assert cough_pattern_of(ProvStore(run_id="t"), tmp_path, sampling_hz=16000.0) == ("spectrogram_narrowband",)


def test_a_stored_spectrogram_is_read(tmp_path: Path) -> None:
    """The stored narrowband spectrogram is read through its sidecar."""
    p = cough_parameters()
    power = np.full((161, 600), 1e-6)
    power[:, 200:240] = 1.0
    np.savez(tmp_path / "spec.npz", spectrogram=power)
    store = ProvStore(run_id="t")
    store.entity(
        prov_type="measurement",
        extent=None,
        attributes={"name": "spectrogram_narrowband", "path": "spec.npz", "n_fft": 320, "hop_length": 80},
    )
    reading = cough_pattern_of(store, tmp_path, sampling_hz=16000.0)
    assert isinstance(reading, CoughPattern)
    assert reading.onsets_n == 1
    assert p.band_edges_hz[-1] <= 8000.0
