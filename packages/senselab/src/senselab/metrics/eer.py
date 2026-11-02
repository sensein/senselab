"""`eer`: equal error rate of the cosine similarities of `Trials`.

A trial scores the cosine of its two embeddings; a pair is accepted at threshold `t` when its score is at least `t`.
The value is the mean of the false-accept and false-reject rates at the threshold, taken over the trials' own
scores, where they are closest.
"""

from collections.abc import Mapping
from typing import Any

import numpy
from senselab_core.kernel.declare import REGISTRY, TaskParams, backend, task
from senselab_core.kernel.results import Metric, Trials
from senselab_worker.fake import FakeFaults
from senselab_worker.faults import ContentFault

from senselab.metrics import _worker

WORKER_MODULE = "senselab.metrics.eer"
load = _worker.load


@task(name="eer", version="1", kind="transform", registry=REGISTRY)
class Eer:
    """Score speaker-verification trials: the equal error rate of the cosine similarity of each pair."""

    inputs = {"trials": Trials}
    outputs = {"metric": Metric}


@backend(task="eer", name="numpy-eer", worker=WORKER_MODULE, **_worker.HOST_BACKEND)
class NumpyEer:
    """Threshold sweep over the sorted scores, in numpy."""


def cosine(first: numpy.ndarray, second: numpy.ndarray) -> numpy.ndarray:
    """Return the cosine similarity of each row of `first` with the same row of `second`.

    Raises:
        ContentFault: if a row is all zeros.
    """
    a, b = first.astype(numpy.float64), second.astype(numpy.float64)
    norms = numpy.linalg.norm(a, axis=1) * numpy.linalg.norm(b, axis=1)
    if numpy.any(norms == 0.0):
        raise ContentFault("an embedding of a trial is all zeros, so its cosine is undefined")
    return numpy.asarray(numpy.sum(a * b, axis=1) / norms)


def equal_error_rate(scores: numpy.ndarray, same: numpy.ndarray) -> float:
    """Return the mean of the false-reject and false-accept rates at the score threshold where they are closest.

    Raises:
        ContentFault: if the trials have no same pair or no different pair.
    """
    n_same = int(same.sum())
    n_diff = len(same) - n_same
    if n_same == 0 or n_diff == 0:
        raise ContentFault("equal error rate needs trials of both kinds, same and different")
    thresholds = numpy.unique(scores)
    reject = numpy.array([(scores[same] < t).sum() / n_same for t in thresholds])
    accept = numpy.array([(scores[~same] >= t).sum() / n_diff for t in thresholds])
    at = int(numpy.argmin(numpy.abs(reject - accept)))
    return float((reject[at] + accept[at]) / 2.0)


def score(inputs: Mapping[str, Any], params: object) -> Metric:  # noqa: ARG001 - the task takes no params
    """Return the `eer` of `inputs["trials"]`."""
    trials: Trials = inputs["trials"]
    return Metric(
        name="eer",
        value=equal_error_rate(cosine(trials.first, trials.second), trials.same),
        unit="ratio",
        items=len(trials.same),
    )


def handle(state: FakeFaults, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one `metric@1.0` wire form per item, or its fault."""
    return _worker.serve(state, items, params, TaskParams, score)
