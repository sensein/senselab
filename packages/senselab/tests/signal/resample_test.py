"""`signal.resample`: duration, pass band, stop band and linearity follow from the filter's design."""

import math
from typing import Any

import numpy
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from senselab._api.packaged import LOCK_RECORDS
from senselab.signal import resample as rs
from senselab_core.kernel.declare import REGISTRY
from senselab_core.kernel.signals import RegularGroup, audio_channels
from senselab_core.kernel.wire import from_wire_form
from senselab_core.testing.contracts import contract_cases

RATES = [8000, 11025, 16000, 22050, 32000, 44100, 48000]
PAIRS = st.tuples(st.sampled_from(RATES), st.sampled_from(RATES)).filter(lambda p: p[0] != p[1])
SLOW = settings(max_examples=20, deadline=None)
DELTA = 10.0 ** (-rs.design_bounds()[1] / 20.0)
N_IN = 4000


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
def test_passband_within_design_ripple(pair: tuple[int, int], fraction: float) -> None:
    """A tone up to the pass-band edge keeps unit gain to within the design's ripple."""
    rate, target = pair
    f = fraction * rs.passband_edge(rate, target)
    y = rs.resample_array(_tone(rate, f), rate, target)[0]
    amplitude = _tone_amplitude(y[_interior(rate, target, y.size)], target, f)
    assert abs(20 * math.log10(amplitude)) <= rs.design_bounds()[0]


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
    assert _residual(rate, target, f) <= (math.ceil(target / rate) + 2) * DELTA


@pytest.mark.logic
@SLOW
@given(pair=PAIRS.filter(lambda p: p[1] > p[0]), fraction=st.floats(0.5, 1.0))
def test_upsampling_images_below_design_attenuation(pair: tuple[int, int], fraction: float) -> None:
    """Images of a tone near the input's pass-band edge, above the input's Nyquist, stay at the stop-band level."""
    rate, target = pair
    f = fraction * rs.passband_edge(rate, target)
    y = rs.resample_array(_tone(rate, f), rate, target)[0]
    window = _interior(rate, target, y.size)
    t = numpy.arange(y.size) / target
    basis = numpy.stack([numpy.cos(2 * numpy.pi * f * t), numpy.sin(2 * numpy.pi * f * t)], axis=1)[window]
    coef, *_ = numpy.linalg.lstsq(basis, y[window], rcond=None)
    assert numpy.abs(y[window] - basis @ coef).max() <= (math.ceil(target / rate) + 1) * DELTA


@pytest.mark.logic
@SLOW
@given(pair=PAIRS.filter(lambda p: p[1] < p[0]), fraction=st.floats(0.0, 1.0))
def test_stopband_below_design_attenuation(pair: tuple[int, int], fraction: float) -> None:
    """A tone between the target's Nyquist and the input's leaves the output at least the design attenuation down."""
    rate, target = pair
    f = target / 2 + fraction * (rate / 2 - target / 2) * 0.999
    y = rs.resample_array(_tone(rate, f), rate, target)[0]
    interior = y[_interior(rate, target, y.size)]
    assert numpy.abs(interior).max() <= DELTA


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
    [result] = rs.handle(
        rs.load({}), [{"id": "a", "inputs": {"series": group}}], {"target_rate": 8000, "backend_params": {}}
    )
    out: Any = from_wire_form(result["outputs"]["series"])
    assert isinstance(out, RegularGroup) and out.sampling_rate == 8000.0
    assert out.channels == group.channels and out.frames == 80 and out.read().dtype == numpy.float32


@pytest.mark.logic
def test_a_non_integer_input_rate_is_a_content_fault() -> None:
    """A rate with no whole-Hz ratio fails that item, not the batch."""
    group = RegularGroup(channels=audio_channels(1), sampling_rate=16000.5, data=numpy.zeros((1, 16), numpy.float32))
    [result] = rs.handle(rs.load({}), [{"id": "a", "inputs": {"series": group}}], {"target_rate": 8000})
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
