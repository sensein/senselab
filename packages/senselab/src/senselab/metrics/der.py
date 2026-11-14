"""`der`: diarization error rate of hypothesis speaker turns against a reference, as pyannote.metrics defines it.

`DER = (missed speech + false alarm + speaker confusion) / reference speech`, with every duration summed over
speakers (overlap counts once per speaker). Reference and hypothesis speakers are matched one to one to maximise
their co-occurring time. With `collar` seconds, the time within `collar / 2` of each reference boundary is not
scored; with `skip_overlap`, time with more than one reference speaker is not scored.
"""

from collections.abc import Mapping
from typing import Any

import numpy
from pydantic import Field
from senselab_core.kernel.declare import InputsAgree, TaskParams, backend, task
from senselab_core.kernel.results import Intervals, Metric
from senselab_core.runtime.compare import best_assignment
from senselab_worker.faults import ContentFault

from senselab.metrics import _worker

WORKER_MODULE = "senselab.metrics.der"
load = _worker.load
_NS = 1e9


class DerParams(TaskParams):
    """Params of `der`: `collar` in seconds, split over both sides of each reference boundary; `skip_overlap`."""

    collar: float = Field(default=0.0, ge=0.0)
    skip_overlap: bool = False


@task(
    name="der",
    version="1",
    kind="transform",
    constraints=(InputsAgree(ports=("reference", "hypothesis"), property="clock_id"),),
)
class Der:
    """Score hypothesis speaker turns against reference turns: the diarization error rate."""

    inputs = {"reference": Intervals, "hypothesis": Intervals}
    outputs = {"metric": Metric}
    Params = DerParams


@backend(task="der", name="numpy-der", worker=WORKER_MODULE, **_worker.HOST_BACKEND)
class NumpyDer:
    """Elementary-segment accounting with a Hungarian speaker matching, in numpy."""


def _scored_edges(ref: Intervals, hyp: Intervals, params: DerParams) -> numpy.ndarray:
    """Return the sorted segment boundaries, in seconds, that cut every turn and every collar edge."""
    edges = [ref.start_ns / _NS, ref.end_ns / _NS, hyp.start_ns / _NS, hyp.end_ns / _NS]
    if params.collar > 0:
        half = params.collar / 2.0
        edges += [
            ref.start_ns / _NS - half,
            ref.start_ns / _NS + half,
            ref.end_ns / _NS - half,
            ref.end_ns / _NS + half,
        ]
    return numpy.unique(numpy.concatenate(edges))


def _activity(turns: Intervals, labels: numpy.ndarray, mid: numpy.ndarray) -> numpy.ndarray:
    """Return `(labels, segments)`: whether each label speaks at each segment midpoint."""
    start, end = turns.start_ns / _NS, turns.end_ns / _NS
    active = numpy.zeros((len(labels), len(mid)), dtype=bool)
    for k, label in enumerate(labels):
        for s, e in zip(start[turns.label == label], end[turns.label == label]):
            active[k] |= (mid >= s) & (mid < e)
    return active


def _in_collar(ref: Intervals, mid: numpy.ndarray, collar: float) -> numpy.ndarray:
    half = collar / 2.0
    near = numpy.zeros(len(mid), dtype=bool)
    for boundary in numpy.concatenate([ref.start_ns, ref.end_ns]) / _NS:
        near |= numpy.abs(mid - boundary) < half
    return near


def diarization_error(ref: Intervals, hyp: Intervals, params: DerParams) -> tuple[float, float]:
    """Return the seconds of error of `hyp` against `ref` and the seconds of reference speech scored.

    Raises:
        ContentFault: if the clocks differ or the reference has no speech.
    """
    _worker.require_same("clock", ref.clock_id, hyp.clock_id)
    edges = _scored_edges(ref, hyp, params)
    mid, width = (edges[:-1] + edges[1:]) / 2.0, numpy.diff(edges)
    ref_labels, hyp_labels = numpy.unique(ref.label), numpy.unique(hyp.label)
    a, b = _activity(ref, ref_labels, mid), _activity(hyp, hyp_labels, mid)
    keep = numpy.ones(len(mid), dtype=bool)
    if params.collar > 0:
        keep &= ~_in_collar(ref, mid, params.collar)
    if params.skip_overlap:
        keep &= a.sum(axis=0) <= 1
    a, b, width = a[:, keep], b[:, keep], width[keep]
    total = float((a.sum(axis=0) * width).sum())
    if total == 0.0:
        raise ContentFault("the reference has no scored speech")
    n_ref, n_hyp = a.sum(axis=0), b.sum(axis=0)
    together = (a[:, None, :] & b[None, :, :]) * width if len(ref_labels) and len(hyp_labels) else None
    correct = numpy.zeros(len(width))
    if together is not None:
        for r, h in best_assignment(together.sum(axis=2).tolist()):
            correct += (a[r] & b[h]) * 1.0
    missed = numpy.maximum(n_ref - n_hyp, 0) * width
    false_alarm = numpy.maximum(n_hyp - n_ref, 0) * width
    confusion = (numpy.minimum(n_ref, n_hyp) - correct) * width
    return float(missed.sum() + false_alarm.sum() + confusion.sum()), total


def score(inputs: Mapping[str, Any], params: DerParams) -> Metric:
    """Return the `der` of `inputs["hypothesis"]` against `inputs["reference"]`."""
    ref: Intervals = inputs["reference"]
    errors, speech = diarization_error(ref, inputs["hypothesis"], params)
    return Metric(
        name="der",
        value=errors / speech,
        unit="ratio",
        items=max(len(ref.label), 1),
        numerator=errors,
        denominator=speech,
    )


def handle(state: None, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one `metric@2.0` wire form per item, or its fault."""
    return _worker.serve(state, items, params, DerParams, score)
