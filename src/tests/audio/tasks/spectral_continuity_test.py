"""Frame-to-frame spectral continuity: steady spectral shape reads high, a transition reads low."""

from __future__ import annotations

import numpy as np
from scipy.signal import stft

from senselab.audio.tasks.envelope.api import MedianSmoothing
from senselab.audio.tasks.spectral_continuity.api import spectral_continuity

SR = 16000
WINDOW_S = 0.04
HOP_S = 0.01


def _tone(seconds: float, freq: float, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _magnitude(samples: np.ndarray) -> np.ndarray:
    hop = max(1, int(HOP_S * SR))
    nperseg = min(int(WINDOW_S * SR), len(samples))
    noverlap = max(0, min(nperseg - hop, nperseg - 1))
    _, _, z = stft(samples, fs=SR, nperseg=nperseg, noverlap=noverlap, boundary=None)
    return np.abs(z)


def _continuity(samples: np.ndarray) -> np.ndarray:
    magnitude = _magnitude(samples)
    return spectral_continuity(
        magnitude, hop_s=HOP_S, sampling_rate=SR, n_samples=len(samples), smoothing=MedianSmoothing(window_s=0.02)
    )


class TestSteadyToneReadsHigh:
    """A single, unchanging tone has the same spectral shape from one frame to the next."""

    def test_a_sustained_pure_tone_reads_near_one_away_from_its_edges(self) -> None:
        """The middle of a 1 s tone, far from onset/offset transients, is highly self-similar."""
        samples = _tone(1.0, 440.0)
        continuity = _continuity(samples)
        middle = continuity[int(0.3 * SR) : int(0.7 * SR)]
        assert middle.min() > 0.95


class TestATransitionReadsLow:
    """A frequency jump changes the spectral shape abruptly, which the measure must register."""

    def test_a_frequency_jump_dips_continuity_at_the_boundary(self) -> None:
        """Two different pure tones back to back dip continuity right at the switch."""
        samples = np.concatenate([_tone(0.5, 300.0), _tone(0.5, 3000.0)])
        continuity = _continuity(samples)
        boundary = continuity[int(0.5 * SR) - 200 : int(0.5 * SR) + 200]
        steady = continuity[int(0.1 * SR) : int(0.3 * SR)]
        assert boundary.min() < steady.min() - 0.1


class TestSilenceReadsLowNotUndefined:
    """Digital silence must not read as the most continuous case a span gate could latch onto."""

    def test_true_silence_is_finite_and_low(self) -> None:
        """An all-zero recording has no defined spectral shape to hold steady; it reads low, not nan."""
        continuity = _continuity(np.zeros(SR, dtype=np.float32))
        assert np.isfinite(continuity).all()
        assert continuity.max() < 0.5

    def test_silence_raises_no_warning(self) -> None:
        """The 0/0 two all-zero frames would otherwise produce must not surface a RuntimeWarning."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _continuity(np.zeros(SR, dtype=np.float32))


class TestOutputShapeAndRange:
    """The trace is one value per input sample, always in [0, 1]."""

    def test_one_value_per_sample(self) -> None:
        """The returned trace is the same length as the input, not the frame count."""
        samples = _tone(0.73, 220.0)
        assert _continuity(samples).shape == samples.shape

    def test_values_stay_in_unit_range(self) -> None:
        """Floating-point roundoff in the cosine similarity must not push a value outside [0, 1]."""
        samples = np.concatenate([_tone(0.2, 100.0), np.zeros(SR // 10, dtype=np.float32), _tone(0.2, 5000.0)])
        continuity = _continuity(samples)
        assert continuity.min() >= 0.0
        assert continuity.max() <= 1.0 + 1e-9

    def test_a_very_short_recording_does_not_crash(self) -> None:
        """Fewer samples than one STFT window still returns a same-length, finite trace."""
        samples = _tone(0.01, 440.0)
        continuity = _continuity(samples)
        assert continuity.shape == samples.shape
        assert np.isfinite(continuity).all()


class TestReusesCallerSuppliedMagnitude:
    """The function trusts its input array rather than computing its own STFT."""

    def test_a_single_frame_magnitude_reads_as_fully_continuous(self) -> None:
        """With fewer than two frames there is no frame-to-frame comparison to make."""
        magnitude = np.abs(np.random.default_rng(0).normal(size=(257, 1)))
        continuity = spectral_continuity(
            magnitude, hop_s=HOP_S, sampling_rate=SR, n_samples=160, smoothing=MedianSmoothing(window_s=0.05)
        )
        assert continuity.shape == (160,)
        assert np.all(continuity == 1.0)

    def test_identical_frames_read_as_fully_continuous(self) -> None:
        """A magnitude spectrogram that never changes has a perfectly stable spectral shape."""
        rng = np.random.default_rng(1)
        frame = np.abs(rng.normal(size=257))
        magnitude = np.tile(frame[:, None], (1, 10))
        continuity = spectral_continuity(
            magnitude, hop_s=HOP_S, sampling_rate=SR, n_samples=1600, smoothing=MedianSmoothing(window_s=0.02)
        )
        assert continuity.min() > 1.0 - 1e-6
