"""The content band and the long-term average spectrum, over one STFT."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import butter, sosfiltfilt

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.band_profile import band_profile, long_term_average_spectrum, rolloff_hz
from senselab.audio.tasks.band_profile.api import bin_power

SR = 48000


def _noise(cutoff_hz: float | None, *, sampling_rate: int = SR, seconds: float = 2.0, seed: int = 1) -> Audio:
    """Broadband noise, optionally stopping at ``cutoff_hz``."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(seconds * sampling_rate))
    if cutoff_hz is not None:
        sos = butter(10, cutoff_hz, btype="low", fs=sampling_rate, output="sos")
        x = np.asarray(sosfiltfilt(sos, x))
    x = 0.5 * x / np.abs(x).max()
    return Audio(waveform=torch.from_numpy(x.astype(np.float32)).reshape(1, -1), sampling_rate=sampling_rate)


class TestTheRollOffFollowsTheContent:
    """The measurement is of the signal's band, not of the container it is stored in."""

    @pytest.mark.parametrize("cutoff_hz", [2000.0, 4000.0, 8000.0])
    def test_a_band_limited_signal_reports_its_own_cutoff(self, cutoff_hz: float) -> None:
        """Three cutoffs in one container: a constant or a rate-derived answer fails at least two."""
        measured = rolloff_hz(_noise(cutoff_hz), quantile=0.95, n_fft=960, hop_length=240)
        assert measured is not None
        assert cutoff_hz * 0.9 <= measured <= cutoff_hz * 1.1, f"cutoff {cutoff_hz}, read {measured}"

    def test_full_band_noise_reports_near_nyquist(self) -> None:
        """Unfiltered noise puts 95% of its energy most of the way to Nyquist."""
        measured = rolloff_hz(_noise(None), quantile=0.95, n_fft=960, hop_length=240)
        assert measured is not None
        assert measured > 0.9 * SR / 2

    def test_a_silent_signal_has_no_band_edge(self) -> None:
        """No energy is a missing measurement, not a measured zero."""
        silence = Audio(waveform=torch.zeros(1, SR), sampling_rate=SR)
        assert rolloff_hz(silence, quantile=0.95, n_fft=960, hop_length=240) is None

    def test_a_signal_shorter_than_the_transform_is_a_value_error(self) -> None:
        """Refused rather than padded: a padded transform reports the padding's spectrum."""
        short = Audio(waveform=torch.ones(1, 100), sampling_rate=SR)
        with pytest.raises(ValueError, match="shorter than"):
            rolloff_hz(short, quantile=0.95, n_fft=960, hop_length=240)

    def test_a_lower_quantile_never_reports_a_higher_edge(self) -> None:
        """The quantile is a cumulative fraction, so it is monotone in the edge it names."""
        audio = _noise(6000.0)
        low = rolloff_hz(audio, quantile=0.5, n_fft=960, hop_length=240)
        high = rolloff_hz(audio, quantile=0.95, n_fft=960, hop_length=240)
        assert low is not None and high is not None
        assert low <= high


class TestTheLongTermAverageSpectrum:
    """A short log-spaced vector, and the layout a slope would be taken over."""

    def test_the_bands_are_log_spaced_and_span_the_requested_range(self) -> None:
        """Log spacing is what makes a per-octave slope arithmetic over these points."""
        power, bin_hz = bin_power(_noise(None), n_fft=960, hop_length=240)
        edges, centres, levels = long_term_average_spectrum(
            power, bin_hz=bin_hz, low_hz=50.0, high_hz=SR / 2, n_bands=24
        )
        assert edges.shape == (25,) and centres.shape == (24,) and levels.shape == (24,)
        assert edges[0] == pytest.approx(50.0) and edges[-1] == pytest.approx(SR / 2)
        ratios = edges[1:] / edges[:-1]
        assert np.allclose(ratios, ratios[0]), "the edges are not log-spaced"

    def test_a_band_limited_signal_reads_far_quieter_above_its_cutoff(self) -> None:
        """The vector carries the band limit the roll-off scalar summarises."""
        power, bin_hz = bin_power(_noise(4000.0), n_fft=960, hop_length=240)
        _, centres, levels = long_term_average_spectrum(power, bin_hz=bin_hz, low_hz=50.0, high_hz=SR / 2, n_bands=24)
        below = levels[(centres > 500.0) & (centres < 3000.0)]
        above = levels[centres > 8000.0]
        assert float(below.mean()) - float(above.max()) > 40.0

    def test_an_unrealisable_band_layout_is_a_value_error(self) -> None:
        """Refused rather than silently emptied, so a caller's absence has a reason."""
        power, bin_hz = bin_power(_noise(None), n_fft=960, hop_length=240)
        with pytest.raises(ValueError, match="ascending and positive"):
            long_term_average_spectrum(power, bin_hz=bin_hz, low_hz=24000.0, high_hz=50.0, n_bands=8)
        with pytest.raises(ValueError, match="n_bands must be positive"):
            long_term_average_spectrum(power, bin_hz=bin_hz, low_hz=50.0, high_hz=24000.0, n_bands=0)


class TestTheProfileIsOneObject:
    """Scalar and vector from one transform, carrying the rate they were measured at."""

    def test_the_profile_records_the_rate_it_was_measured_at(self) -> None:
        """A profile that forgets its rate cannot be compared against the declared one."""
        profile = band_profile(
            _noise(4000.0), quantile=0.95, n_fft=960, hop_length=240, ltas_low_hz=50.0, ltas_bands=24
        )
        assert profile.sampling_rate == SR
        assert profile.nyquist_hz == SR / 2
        assert profile.quantile == 0.95
        assert 3600.0 <= profile.rolloff_hz <= 4400.0

    def test_a_silent_signal_is_refused_rather_than_profiled(self) -> None:
        """There is no band edge to report, so there is no profile."""
        silence = Audio(waveform=torch.zeros(1, SR), sampling_rate=SR)
        with pytest.raises(ValueError, match="no energy"):
            band_profile(silence, quantile=0.95, n_fft=960, hop_length=240, ltas_low_hz=50.0, ltas_bands=24)

    def test_a_multi_channel_input_is_averaged_rather_than_refused(self) -> None:
        """Two identical channels must read as the one signal they are."""
        mono = _noise(4000.0)
        stereo = Audio(waveform=mono.waveform.repeat(2, 1), sampling_rate=SR)
        one = band_profile(mono, quantile=0.95, n_fft=960, hop_length=240, ltas_low_hz=50.0, ltas_bands=24)
        two = band_profile(stereo, quantile=0.95, n_fft=960, hop_length=240, ltas_low_hz=50.0, ltas_bands=24)
        assert one.rolloff_hz == two.rolloff_hz
