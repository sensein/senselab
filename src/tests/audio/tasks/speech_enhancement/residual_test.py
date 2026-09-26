"""Tests for ``senselab.audio.tasks.speech_enhancement.residual``.

Lag/align/gain/band-fraction sign conventions moved here from the triage PREPROCESS node's own
tests once the computation was extracted into this module; see
``tests/audio/workflows/triage/nodes/preprocess_test.py`` for the node-level behaviour (gates,
stream writing, classification) built on top of it.
"""

import numpy as np
import pytest

from senselab.audio.tasks.speech_enhancement.residual import (
    align,
    band_energy_fractions,
    compute_residual,
    correlation,
    find_lag,
    fit_gain,
)


class TestFindLagAndAlign:
    """The sign conventions the residual subtraction depends on, pinned synthetically."""

    def test_lag_and_gain_cancel_a_delayed_scaled_copy(self) -> None:
        """A signal that is purely a delayed, scaled copy of the reference cancels to ~zero."""
        sr = 16000
        n = sr * 2
        rng = np.random.default_rng(0)
        ref = rng.standard_normal(n)
        true_lag = 37
        true_scale = 0.6
        sig = np.zeros(n)
        sig[true_lag:] = true_scale * ref[: n - true_lag]

        lag = find_lag(ref, sig, sr, max_lag_ms=200.0)
        assert lag == true_lag

        ref_aligned, sig_aligned = align(ref, sig, lag)
        gain = fit_gain(ref_aligned, sig_aligned)
        assert gain == pytest.approx(1.0 / true_scale, rel=1e-6)

        residual = ref_aligned - gain * sig_aligned
        assert np.abs(residual).max() < 1e-8

    def test_lag_sign_flips_when_the_arguments_are_swapped(self) -> None:
        """``sig`` arriving later than ``ref`` is the positive convention; swapping negates it."""
        sr = 16000
        n = sr * 2
        rng = np.random.default_rng(1)
        ref = rng.standard_normal(n)
        sig = np.zeros(n)
        sig[25:] = 0.8 * ref[: n - 25]

        forward = find_lag(ref, sig, sr, max_lag_ms=200.0)
        backward = find_lag(sig, ref, sr, max_lag_ms=200.0)
        assert forward == 25
        assert backward == -forward

    def test_align_drops_the_signals_prefix_on_a_positive_lag(self) -> None:
        """A positive lag trims ``sig``'s head, not ``ref``'s."""
        ref = np.arange(10.0)
        sig = np.arange(10.0) + 100.0
        aligned_ref, aligned_sig = align(ref, sig, 3)
        assert list(aligned_sig) == list(sig[3:])
        assert list(aligned_ref) == list(ref[: len(aligned_sig)])

    def test_align_drops_the_references_prefix_on_a_negative_lag(self) -> None:
        """A negative lag trims ``ref``'s head, not ``sig``'s."""
        ref = np.arange(10.0)
        sig = np.arange(10.0) + 100.0
        aligned_ref, aligned_sig = align(ref, sig, -3)
        assert list(aligned_ref) == list(ref[3:])
        assert list(aligned_sig) == list(sig[: len(aligned_ref)])


class TestFitGain:
    """The least-squares gain fit's sign convention and its zero-energy guard."""

    def test_fit_gain_recovers_a_known_scale(self) -> None:
        """``g = <ref, sig> / <sig, sig>`` recovers the exact scale on a noiseless pair."""
        rng = np.random.default_rng(2)
        sig = rng.standard_normal(1000)
        ref = 0.25 * sig
        assert fit_gain(ref, sig) == pytest.approx(0.25, rel=1e-9)

    def test_fit_gain_is_zero_when_sig_carries_no_energy(self) -> None:
        """A silent ``sig`` cannot be fit a gain against; the denominator is guarded, not divided by."""
        assert fit_gain(np.array([1.0, 2.0, 3.0]), np.zeros(3)) == 0.0


class TestBandEnergyFractions:
    """The band split the residual's summaries and reports read."""

    def test_band_fractions_sum_to_one_and_locate_a_pure_tone(self) -> None:
        """A 300 Hz tone's energy lands almost entirely in the 200-1000 Hz band."""
        sr = 16000
        t = np.arange(sr * 2) / sr
        x = np.sin(2 * np.pi * 300.0 * t)
        bands = [(0.0, 200.0), (200.0, 1000.0), (1000.0, 4000.0), (4000.0, 8000.0)]
        fractions = band_energy_fractions(x, sr, bands)
        assert sum(fractions.values()) == pytest.approx(1.0, abs=1e-6)
        assert fractions["200_1000"] > 0.99

    def test_band_fractions_of_an_empty_signal_are_zero(self) -> None:
        """A zero-length signal reports zero in every band rather than dividing by zero."""
        assert band_energy_fractions(np.array([]), 16000, [(0.0, 200.0)]) == {"0_200": 0.0}


class TestCorrelation:
    """The instrument that told the separation streams apart in the buzz-separation comparison."""

    def test_identical_signals_correlate_at_one(self) -> None:
        """A signal against itself is perfect correlation."""
        rng = np.random.default_rng(3)
        x = rng.standard_normal(1000)
        assert correlation(x, x) == pytest.approx(1.0, rel=1e-9)

    def test_a_negative_scale_flips_the_sign(self) -> None:
        """Correlation is scale-invariant up to sign."""
        rng = np.random.default_rng(4)
        x = rng.standard_normal(1000)
        assert correlation(x, -3.0 * x) == pytest.approx(-1.0, rel=1e-9)

    def test_uncorrelated_signals_land_near_zero(self) -> None:
        """Two independent gaussian draws correlate near zero at this length."""
        rng = np.random.default_rng(5)
        a = rng.standard_normal(50000)
        b = rng.standard_normal(50000)
        assert abs(correlation(a, b)) < 0.05

    def test_a_constant_signal_is_nan(self) -> None:
        """Zero variance in either signal makes the coefficient undefined, not zero."""
        assert np.isnan(correlation(np.ones(10), np.arange(10.0)))

    def test_fewer_than_two_samples_is_nan(self) -> None:
        """Fewer than two samples after trimming to the common length is also undefined."""
        assert np.isnan(correlation(np.array([1.0]), np.array([1.0])))


class TestComputeResidual:
    """The full computation: lag, gain, subtraction, both energy fractions, both correlations."""

    def test_an_identical_signal_absorbs_everything(self) -> None:
        """``signal == reference`` fits gain 1.0 and leaves a zero residual."""
        rng = np.random.default_rng(6)
        ref = rng.standard_normal(16000)
        result = compute_residual(ref, ref.copy(), 16000, max_lag_ms=50.0)
        assert result.lag_samples == 0
        assert result.gain == pytest.approx(1.0, rel=1e-9)
        assert np.abs(result.residual).max() < 1e-9
        assert result.signal_energy_fraction == pytest.approx(1.0, rel=1e-9)
        assert result.residual_energy_fraction == pytest.approx(0.0, abs=1e-9)
        assert result.correlation_signal == pytest.approx(1.0, rel=1e-9)

    def test_a_silent_signal_leaves_the_reference_as_the_residual(self) -> None:
        """A zero ``signal`` fits gain 0.0 (guarded), so the residual is the reference unchanged."""
        rng = np.random.default_rng(7)
        ref = rng.standard_normal(16000)
        sig = np.zeros(16000)
        result = compute_residual(ref, sig, 16000, max_lag_ms=50.0)
        assert result.gain == 0.0
        assert result.signal_energy_fraction == pytest.approx(0.0, abs=1e-9)
        assert result.residual_energy_fraction == pytest.approx(1.0, rel=1e-6)
        np.testing.assert_allclose(result.residual, result.reference_aligned)
        assert np.isnan(result.correlation_signal)  # sig carries zero variance

    def test_a_silent_reference_reports_nan_fractions_rather_than_dividing_by_zero(self) -> None:
        """``input_energy == 0`` makes both fractions undefined, not a raised exception."""
        rng = np.random.default_rng(8)
        ref = np.zeros(1000)
        sig = rng.standard_normal(1000)
        result = compute_residual(ref, sig, 16000, max_lag_ms=50.0)
        assert result.input_energy == 0.0
        assert np.isnan(result.signal_energy_fraction)
        assert np.isnan(result.residual_energy_fraction)

    def test_gain_db_and_lag_ms_are_derived_from_gain_and_lag(self) -> None:
        """The dB/ms conveniences match their linear/sample counterparts."""
        sr = 16000
        rng = np.random.default_rng(9)
        ref = rng.standard_normal(sr)
        sig = np.zeros(sr)
        sig[16:] = 2.0 * ref[: sr - 16]
        result = compute_residual(ref, sig, sr, max_lag_ms=50.0)
        assert result.lag_samples == 16
        assert result.lag_ms == pytest.approx(16 / sr * 1000.0)
        assert result.gain_db == pytest.approx(20.0 * np.log10(abs(result.gain)))

    def test_gain_db_is_negative_infinity_when_gain_is_exactly_zero(self) -> None:
        """A silent ``signal`` reports ``-inf`` dB rather than raising on ``log10(0)``."""
        rng = np.random.default_rng(10)
        ref = rng.standard_normal(1000)
        result = compute_residual(ref, np.zeros(1000), 16000, max_lag_ms=50.0)
        assert result.gain_db == float("-inf")
