"""`signal.resample`: rational-ratio resampling of a regularly sampled series by a Kaiser-windowed sinc.

The backend `numpy-sinc` and its worker (`load`, `handle`) are this module. Filter parameters are in
`data/resample.yaml`; `design_bounds` returns the pass-band ripple and stop-band attenuation the design
targets, and `resample_array` is the filter.
"""

import math
from dataclasses import dataclass
from fractions import Fraction
from importlib import resources
from typing import Any

import numpy
from pydantic import Field
from senselab_core.kernel.declare import Env, TaskParams, backend, task
from senselab_core.kernel.signals import RegularSeries
from senselab_core.runtime.audio_io import READER
from senselab_core.runtime.config import load_yaml
from senselab_worker.fake import injected, without_fault
from senselab_worker.faults import ContentFault

WORKER_MODULE = "senselab.signal.resample"
_DESIGN: dict[str, Any] = load_yaml(resources.files("senselab").joinpath("signal/data/resample.yaml"), "resample.yaml")
_STOPBAND_DB: float = float(_DESIGN["stopband_attenuation_db"])
_TRANSITION: float = float(_DESIGN["transition_fraction"])
_MAX_BLOCK: int = int(_DESIGN["max_block_elements"])
_INIT_KEYS = frozenset({"array_dir", "device", "model"})
_KAISER_HIGH_DB = 50.0
_KAISER_LOW_DB = 21.0


class ResampleParams(TaskParams):
    """Params of `signal.resample`: `target_rate` in Hz."""

    target_rate: int = Field(
        gt=0,
        description="The sampling rate to resample to, in Hz.",
        examples=[16_000],
        json_schema_extra={"contract_probe": 16_000},
    )


@task(name="signal.resample", version="1", kind="transform")
class Resample:
    """Resample a regularly sampled series to `target_rate`, keeping its channels and gain."""

    inputs = {"series": RegularSeries}
    outputs = {"series": RegularSeries}
    Params = ResampleParams


@backend(
    task="signal.resample",
    name="numpy-sinc",
    version="1",
    stack="dsp",
    worker=WORKER_MODULE,
    load_stub={},
    batch_invariant=True,
    devices=("cpu",),
    env=Env.host(),
    requirements=("numpy", "pyyaml"),
)
class NumpySinc:
    """Kaiser-windowed sinc over the reduced ratio `target / rate`, in numpy alone."""


@dataclass(frozen=True)
class Design:
    """A designed lowpass: `taps` at the upsampled rate (DC gain `up`), group delay `delay` in them, Kaiser `beta`."""

    taps: numpy.ndarray
    delay: int
    beta: float


def design_bounds() -> tuple[float, float]:
    """Return `(passband_ripple_db, stopband_db)`, the filter's Kaiser design targets.

    The pass band is designed to deviate from unit gain by at most `passband_ripple_db` dB; the stop band to lie at
    least `stopband_db` dB below it.
    """
    delta = 10.0 ** (-_STOPBAND_DB / 20.0)
    return -20.0 * math.log10(1.0 - delta), _STOPBAND_DB


def passband_edge(rate: float, target: float) -> float:
    """Return the highest frequency, in Hz, the design holds within `design_bounds()`."""
    return (1.0 - _TRANSITION) * 0.5 * min(rate, target)


def _kaiser_beta(attenuation_db: float) -> float:
    if attenuation_db > _KAISER_HIGH_DB:
        return 0.1102 * (attenuation_db - 8.7)
    if attenuation_db >= _KAISER_LOW_DB:
        excess = attenuation_db - _KAISER_LOW_DB
        return 0.5842 * math.pow(excess, 0.4) + 0.07886 * excess
    return 0.0


def design(rate: int, up: int, down: int) -> Design:
    """Return the lowpass for resampling `rate` Hz by `up / down`, sampled at `rate * up` Hz.

    The cutoff is the middle of a transition band of `transition_fraction` of the lower Nyquist, ending at that
    Nyquist; the length is Kaiser's `(A - 7.95) / (14.36 * width)` taps for transition `width` in cycles per sample.
    """
    nyquist = 0.5 * min(rate, rate * up / down)
    width = _TRANSITION * nyquist / (rate * up)
    half = math.ceil((_STOPBAND_DB - 7.95) / (14.36 * width) / 2.0)
    cutoff = (nyquist - 0.5 * _TRANSITION * nyquist) / (rate * up)
    beta = _kaiser_beta(_STOPBAND_DB)
    j = numpy.arange(-half, half + 1, dtype=numpy.float64)
    taps = 2.0 * cutoff * numpy.sinc(2.0 * cutoff * j) * numpy.kaiser(2 * half + 1, beta)
    return Design(taps * (up / taps.sum()), half, beta)


def resample_array(x: numpy.ndarray, rate: int, target: int) -> numpy.ndarray:
    """Return `x` `(channels, samples)` resampled from `rate` to `target` Hz, in float64, zero outside the signal.

    The output has `ceil(samples * target / rate)` samples, sample `m` at `m / target` seconds.
    """
    ratio = Fraction(target, rate)
    up, down = ratio.numerator, ratio.denominator
    channels, n = x.shape
    n_out = -(-n * up // down)
    if up == down:
        return x.astype(numpy.float64)
    d = design(rate, up, down)
    q = -(-len(d.taps) // up)
    padded = numpy.zeros(q * up)
    padded[: len(d.taps)] = d.taps
    phases = padded.reshape(q, up).T
    t = numpy.arange(n_out, dtype=numpy.int64) * down + d.delay
    base, phase = t // up, t % up
    xpad = numpy.zeros((channels, q + int(base[-1]) + 1))
    xpad[:, q : q + n] = x
    back = numpy.arange(q)
    out = numpy.empty((channels, n_out))
    block = max(1, _MAX_BLOCK // (channels * q))
    for s in range(0, n_out, block):
        sl = slice(s, s + block)
        gathered = xpad[:, (base[sl, None] - back[None, :]) + q]
        out[:, sl] = numpy.einsum("cbq,bq->cb", gathered, phases[phase[sl]])
    return out


def _series(value: object) -> tuple[list[dict[str, Any]], float, numpy.ndarray]:
    """Return `(channels, sampling_rate, samples)` of a kernel series or its wire form."""
    if isinstance(value, dict):
        form: dict[str, Any] = value
        if "groups" in form:
            (form,) = form["groups"].values()
        return list(form["channels"]), float(form["sampling_rate"]), numpy.asarray(form["data"])
    return [c.model_dump() for c in value.channels], float(value.sampling_rate), value.read(reader=READER)  # type: ignore[attr-defined]


def load(init: dict[str, Any]) -> None:
    """Check `init`, which may hold `array_dir`, `device` and `model`; the backend holds no state.

    Raises:
        ValueError: on any other key.
    """
    unknown = sorted(init.keys() - _INIT_KEYS)
    if unknown:
        raise ValueError(f"numpy-sinc init takes only {sorted(_INIT_KEYS)}, got {unknown}")


def handle(state: None, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one `{"outputs": {"series": <regular-group@3.0 wire form>}}` per item, or its error."""
    target = ResampleParams.model_validate(
        {k: v for k, v in without_fault(params).items() if k != "backend_params"}
    ).target_rate
    faults = injected(params)
    results: list[dict[str, Any]] = []
    for item in items:
        try:
            if item["id"] in faults.fail_items:
                raise ContentFault(f"injected failure for item {item['id']}")
            results.append({"outputs": {"series": _resampled(item["inputs"]["series"], target)}})
        except Exception as exc:  # noqa: BLE001 - every per-item failure is reported as that item's error
            results.append({"error": exc})
    return results


def _resampled(value: object, target: int) -> dict[str, Any]:
    channels, rate, samples = _series(value)
    if rate != int(rate):
        raise ContentFault(f"sampling rate {rate} Hz is not a whole number of Hz")
    out = resample_array(samples, int(rate), target)
    return {
        "format": "regular-group@3.0",
        "channels": channels,
        "sampling_rate": float(target),
        "data": out.astype(samples.dtype if samples.dtype in (numpy.float32, numpy.float64) else numpy.float64),
        "source": None,
        "source_stamp": None,
    }
