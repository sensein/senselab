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
from senselab_core.kernel.declare import REGISTRY, TaskParams, backend, task
from senselab_core.kernel.results import Intervals, Metric
from senselab_worker.fake import FakeFaults
from senselab_worker.faults import ContentFault

from senselab.metrics import _worker

WORKER_MODULE = "senselab.metrics.der"
load = _worker.load
_NS = 1e9


class DerParams(TaskParams):
    """Params of `der`: `collar` in seconds, split over both sides of each reference boundary; `skip_overlap`."""

    collar: float = Field(default=0.0, ge=0.0)
    skip_overlap: bool = False


@task(name="der", version="1", kind="transform", registry=REGISTRY)
class Der:
    """Score hypothesis speaker turns against reference turns: the diarization error rate."""

    inputs = {"reference": Intervals, "hypothesis": Intervals}
    outputs = {"metric": Metric}
    Params = DerParams


@backend(task="der", name="numpy-der", worker=WORKER_MODULE, **_worker.HOST_BACKEND)
class NumpyDer:
    """Elementary-segment accounting with a Hungarian speaker matching, in numpy."""


def assign(gain: numpy.ndarray) -> list[tuple[int, int]]:  # noqa: C901 - the Hungarian algorithm is one routine
    """Return the `(row, column)` pairs of the one-to-one matching that maximises the summed `gain`.

    The Hungarian algorithm over the negated, square-padded matrix.
    """
    n = max(gain.shape)
    cost = numpy.zeros((n, n))
    cost[: gain.shape[0], : gain.shape[1]] = -gain
    u, v = numpy.zeros(n + 1), numpy.zeros(n + 1)
    owner = [0] * (n + 1)
    for i in range(1, n + 1):
        owner[0] = i
        j0 = 0
        minv = numpy.full(n + 1, numpy.inf)
        used = [False] * (n + 1)
        way = [0] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = owner[j0], numpy.inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j], way[j] = cur, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[owner[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
    return [(owner[j] - 1, j - 1) for j in range(1, n + 1) if owner[j] - 1 < gain.shape[0] and j - 1 < gain.shape[1]]


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


def diarization_error_rate(ref: Intervals, hyp: Intervals, params: DerParams) -> float:
    """Return the DER of `hyp` against `ref`.

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
        for r, h in assign(together.sum(axis=2)):
            correct += (a[r] & b[h]) * 1.0
    missed = numpy.maximum(n_ref - n_hyp, 0) * width
    false_alarm = numpy.maximum(n_hyp - n_ref, 0) * width
    confusion = (numpy.minimum(n_ref, n_hyp) - correct) * width
    return float((missed.sum() + false_alarm.sum() + confusion.sum()) / total)


def score(inputs: Mapping[str, Any], params: DerParams) -> Metric:
    """Return the `der` of `inputs["hypothesis"]` against `inputs["reference"]`."""
    ref: Intervals = inputs["reference"]
    return Metric(
        name="der",
        value=diarization_error_rate(ref, inputs["hypothesis"], params),
        unit="ratio",
        items=max(len(ref.label), 1),
    )


def handle(state: FakeFaults, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one `metric@1.0` wire form per item, or its typed fault."""
    return _worker.serve(state, items, params, DerParams, score)
