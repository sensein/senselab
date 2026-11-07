"""`signal.resample`: duration, pass band, stop band and linearity follow from the filter's design."""

import math
from typing import Any

import numpy
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from senselab.signal import resample as rs
from senselab_core.kernel.declare import REGISTRY
from senselab_core.kernel.signals import RegularGroup, audio_channels
from senselab_core.kernel.wire import from_wire_form
from senselab_core.runtime.licenses import packaged_lock_records
from senselab_core.testing.contracts import contract_cases

LOCK_RECORDS = packaged_lock_records("senselab")

RATES = [8000, 11025, 16000, 22050, 32000, 44100, 48000]
ALL_PAIRS = [(a, b) for a in RATES for b in RATES if a != b]
PAIRS = st.sampled_from(ALL_PAIRS)
SLOW = settings(max_examples=20, deadline=None)
DELTA = 10.0 ** (-rs.design_bounds()[1] / 20.0)
N_IN = 4000
EPS = float(numpy.finfo(numpy.float64).eps)


def _ratio(rate: int, target: int) -> tuple[int, int]:
    g = math.gcd(rate, target)
    return target // g, rate // g


def _interior(rate: int, target: int, n_out: int) -> slice:
    """The output samples whose filter support lies wholly inside the signal."""
    up, _ = _ratio(rate, target)
    edge = math.ceil(rs.design(rate, *_ratio(rate, target)).delay / (rate * up) * target) + 2
    return slice(edge, n_out - edge)


def _tone_amplitude(y: numpy.ndarray, target: int, f: float) -> float:
    t = numpy.arange(y.size) / target
    basis = numpy.stack([numpy.cos(2 * numpy.pi * f * t), numpy.sin(2 * numpy.pi * f * t)], axis=1)
    coef, *_ = numpy.linalg.lstsq(basis, y, rcond=None)
    return float(math.hypot(*coef))


def _image_gains(rate: int, target: int, f: float) -> tuple[float, float, float]:
    """The filter's gain at a unit tone `f`, the summed gain at its other images, and the float64 rounding."""
    up, down = _ratio(rate, target)
    h = rs.design(rate, up, down).taps / up
    j = numpy.arange(h.size) - (h.size - 1) // 2
    k = numpy.arange(up + 1)
    images = numpy.concatenate([k * rate + f, k[1:] * rate - f])
    images = images[images <= rate * up / 2]
    gains = numpy.concatenate(
        [
            numpy.abs(numpy.cos(2 * numpy.pi * numpy.outer(chunk, j) / (rate * up)) @ h)
            for chunk in numpy.array_split(images, max(1, images.size // 16))
        ]
    )
    return float(gains[0]), float(gains[1:].sum()), EPS * (h.size + 4 * math.pi * f * N_IN / rate)


def _tone(rate: int, f: float, n: int = N_IN) -> numpy.ndarray:
    return numpy.sin(2 * numpy.pi * f * numpy.arange(n) / rate)[None, :]


@pytest.mark.logic
@SLOW
@given(pair=PAIRS, n=st.integers(1, 3000))
def test_duration_preserved_within_one_sample(pair: tuple[int, int], n: int) -> None:
    """The output spans the input's duration to within one output sample."""
    rate, target = pair
    y = rs.resample_array(numpy.ones((1, n)), rate, target)
    assert y.shape == (1, math.ceil(n * target / rate))
    assert abs(y.shape[1] / target - n / rate) < 1 / target


@pytest.mark.logic
@SLOW
@given(pair=PAIRS, fraction=st.floats(0.01, 1.0))
@example(pair=(8000, 11025), fraction=1.0)
def test_passband_within_design_ripple(pair: tuple[int, int], fraction: float) -> None:
    """A tone up to the pass-band edge keeps unit gain to within the design's ripple."""
    rate, target = pair
    f = fraction * rs.passband_edge(rate, target)
    y = rs.resample_array(_tone(rate, f), rate, target)[0]
    amplitude = _tone_amplitude(y[_interior(rate, target, y.size)], target, f)
    assert abs(20 * math.log10(amplitude)) <= -20 * math.log10(1 - rs.ripple_margin() * DELTA)


def _residual(rate: int, target: int, f: float) -> float:
    """Largest interior deviation of the resampled tone from the ideal sine sampled at `m / target`."""
    y = rs.resample_array(_tone(rate, f), rate, target)[0]
    ideal = numpy.sin(2 * numpy.pi * f * numpy.arange(y.size) / target)
    window = _interior(rate, target, y.size)
    return float(numpy.abs(y[window] - ideal[window]).max())


@pytest.mark.logic
@SLOW
@given(pair=PAIRS, fraction=st.floats(0.01, 1.0))
def test_passband_tone_matches_ideal_sine(pair: tuple[int, int], fraction: float) -> None:
    """A pass-band tone comes out as the ideal sine at `m / target`: no delay or phase error beyond the design."""
    rate, target = pair
    f = fraction * rs.passband_edge(rate, target)
    in_band, images, rounding = _image_gains(rate, target, f)
    assert _residual(rate, target, f) <= abs(in_band - 1) + images + rounding


@pytest.mark.logic
@SLOW
@given(pair=PAIRS.filter(lambda p: p[1] > p[0]), fraction=st.floats(0.5, 1.0))
def test_upsampling_images_below_design_attenuation(pair: tuple[int, int], fraction: float) -> None:
    """Images of a tone near the input's pass-band edge, above the input's Nyquist, stay within their filter gains."""
    rate, target = pair
    f = fraction * rs.passband_edge(rate, target)
    y = rs.resample_array(_tone(rate, f), rate, target)[0]
    window = _interior(rate, target, y.size)
    in_band, images, rounding = _image_gains(rate, target, f)
    t = numpy.arange(y.size) / target
    tone = y - in_band * numpy.sin(2 * numpy.pi * f * t)
    assert numpy.abs(tone[window]).max() <= images + rounding


@pytest.mark.logic
@SLOW
@given(pair=PAIRS.filter(lambda p: p[1] < p[0]), fraction=st.floats(0.0, 1.0))
@example(pair=(48000, 44100), fraction=0.04020100502512563)
def test_stopband_below_design_attenuation(pair: tuple[int, int], fraction: float) -> None:
    """A tone between the target's Nyquist and the input's stays within the filter's summed gain at its images."""
    rate, target = pair
    f = target / 2 + fraction * (rate / 2 - target / 2) * 0.999
    y = rs.resample_array(_tone(rate, f), rate, target)[0]
    interior = y[_interior(rate, target, y.size)]
    assert numpy.abs(interior).max() <= sum(_image_gains(rate, target, f))


@pytest.mark.logic
@SLOW
@given(pair=PAIRS, seed=st.integers(0, 2**16), gain=st.floats(0.01, 100.0))
def test_channel_order_and_gain_invariance(pair: tuple[int, int], seed: int, gain: float) -> None:
    """Channels resample independently and in order, and the output scales with the input."""
    rate, target = pair
    x = numpy.random.default_rng(seed).standard_normal((3, 600))
    y = rs.resample_array(x, rate, target)
    numpy.testing.assert_array_equal(rs.resample_array(x[::-1], rate, target), y[::-1])
    numpy.testing.assert_allclose(rs.resample_array(gain * x, rate, target), gain * y, rtol=1e-12, atol=1e-12)
    numpy.testing.assert_allclose(rs.resample_array(x[1:2], rate, target), y[1:2], rtol=1e-12, atol=1e-12)


@pytest.mark.logic
def test_handle_returns_a_regular_group_wire_form() -> None:
    """A worker result is the wire form of a regular group at the target rate, channels kept."""
    group = RegularGroup(channels=audio_channels(2), sampling_rate=16000.0, data=numpy.zeros((2, 160), numpy.float32))
    [result] = rs.handle(None, [{"id": "a", "inputs": {"series": group}}], {"target_rate": 8000, "backend_params": {}})
    out: Any = from_wire_form(result["outputs"]["series"])
    assert isinstance(out, RegularGroup) and out.sampling_rate == 8000.0
    assert out.channels == group.channels and out.frames == 80 and out.read().dtype == numpy.float32


@pytest.mark.logic
def test_a_non_integer_input_rate_is_a_content_fault() -> None:
    """A rate with no whole-Hz ratio fails that item, not the batch."""
    group = RegularGroup(channels=audio_channels(1), sampling_rate=16000.5, data=numpy.zeros((1, 16), numpy.float32))
    [result] = rs.handle(None, [{"id": "a", "inputs": {"series": group}}], {"target_rate": 8000})
    assert "error" in result


@pytest.mark.contract
@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=name)
        for name, c in contract_cases(REGISTRY, lock_records=LOCK_RECORDS)
        if name.startswith(("signal.resample::", "numpy-sinc::"))
    ],
)
def test_contract(case: Any) -> None:  # noqa: ANN401 - a no-argument contract case
    """The task and backend satisfy the generated contract."""
    case()


@pytest.mark.logic
@pytest.mark.parametrize("pair", ALL_PAIRS, ids=lambda p: f"{p[0]}-{p[1]}")
def test_realised_ripple_within_margin(pair: tuple[int, int]) -> None:
    """The taps' pass-band deviation and stop-band peak are each within `ripple_margin` times the Kaiser target."""
    rate, target = pair
    up, down = _ratio(rate, target)
    h = rs.design(rate, up, down).taps / up
    gain = numpy.abs(numpy.fft.rfft(h, 1 << 21))
    freq = numpy.arange(gain.size) / (1 << 21) * rate * up
    nyquist = min(rate, target) / 2
    limit = rs.ripple_margin() * DELTA
    assert numpy.abs(gain[freq <= rs.passband_edge(rate, target)] - 1).max() <= limit
    assert gain[(freq >= nyquist) & (freq <= rate * up - nyquist)].max() <= limit
