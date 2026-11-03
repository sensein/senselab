"""`resample_error`: how far a resampled series is from its reference, as a fraction of the reference's RMS.

The two series must have the same channels, rate and length. The value is the root of the summed squared
difference over every channel and sample, divided by the root of the summed squared reference.
"""

from collections.abc import Mapping
from typing import Any

import numpy
from senselab_core.kernel.declare import InputsAgree, TaskParams, backend, task
from senselab_core.kernel.results import Metric
from senselab_core.kernel.signals import RegularSeries
from senselab_core.runtime.audio_io import READER
from senselab_worker.fake import FakeFaults
from senselab_worker.faults import ContentFault

from senselab.metrics import _worker

WORKER_MODULE = "senselab.metrics.resample_error"
load = _worker.load


SAME = tuple(
    InputsAgree(ports=("reference", "hypothesis"), property=p) for p in ("channels", "sampling_rate", "frames")
)


@task(name="resample_error", version="1", kind="transform", constraints=SAME)
class ResampleError:
    """Score a resampled series against its reference: the relative RMS error."""

    inputs = {"reference": RegularSeries, "hypothesis": RegularSeries}
    outputs = {"metric": Metric}


@backend(task="resample_error", name="numpy-resample-error", worker=WORKER_MODULE, **_worker.HOST_BACKEND)
class NumpyResampleError:
    """Relative L2 distance of the two sample arrays, in numpy."""


def relative_rms_error(reference: RegularSeries, hypothesis: RegularSeries) -> tuple[float, int]:
    """Return the relative RMS error of `hypothesis` against `reference` and the samples compared.

    Raises:
        ContentFault: if the series differ in channels, rate or length, or the reference is silent.
    """
    _worker.require_same("channels", reference.channels, hypothesis.channels)
    _worker.require_same("sampling rate", reference.sampling_rate, hypothesis.sampling_rate)
    _worker.require_same("length", reference.frames, hypothesis.frames)
    a = reference.read(reader=READER).astype(numpy.float64)
    b = hypothesis.read(reader=READER).astype(numpy.float64)
    norm = float(numpy.sqrt(numpy.sum(a * a)))
    if norm == 0.0:
        raise ContentFault("the reference is silent, so a relative error is undefined")
    return float(numpy.sqrt(numpy.sum((b - a) ** 2)) / norm), a.size


def score(inputs: Mapping[str, Any], params: object) -> Metric:  # noqa: ARG001 - the task takes no params
    """Return the `resample_error` of `inputs["hypothesis"]` against `inputs["reference"]`."""
    value, count = relative_rms_error(inputs["reference"], inputs["hypothesis"])
    return Metric(name="resample_error", value=value, unit="ratio", items=count)


def handle(state: FakeFaults, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one `metric@2.0` wire form per item, or its fault."""
    return _worker.serve(state, items, params, TaskParams, score)
