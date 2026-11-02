"""`span_f1`: micro F1 of hypothesis `TextSpans` against reference spans of the same text.

A hypothesis span is correct when a reference span has the same start, end and label; each reference span
is matched at most once. Precision is correct over hypothesis spans, recall correct over reference spans.
"""

from collections import Counter
from collections.abc import Mapping
from typing import Any

from senselab_core.kernel.declare import REGISTRY, TaskParams, backend, task
from senselab_core.kernel.results import Metric, TextSpans
from senselab_worker.fake import FakeFaults
from senselab_worker.faults import ContentFault

from senselab.metrics import _worker

WORKER_MODULE = "senselab.metrics.span_f1"
load = _worker.load


@task(name="span_f1", version="1", kind="transform", registry=REGISTRY)
class SpanF1:
    """Score hypothesis text spans against reference spans: exact-match micro F1."""

    inputs = {"reference": TextSpans, "hypothesis": TextSpans}
    outputs = {"metric": Metric}


@backend(task="span_f1", name="numpy-span-f1", worker=WORKER_MODULE, **_worker.HOST_BACKEND)
class NumpySpanF1:
    """Multiset intersection of `(start, end, label)` triples."""


def _triples(spans: TextSpans) -> Counter[tuple[int, int, str]]:
    return Counter(zip(spans.start.tolist(), spans.end.tolist(), (str(x) for x in spans.label.tolist())))


def f1(ref: TextSpans, hyp: TextSpans) -> float:
    """Return the micro F1 of `hyp` against `ref`, 0 when no span matches.

    Raises:
        ContentFault: if the spans lie over different texts, or neither side has a span.
    """
    _worker.require_same("text", ref.text, hyp.text)
    expected, found = _triples(ref), _triples(hyp)
    if not expected and not found:
        raise ContentFault("neither side has a span to score")
    correct = sum((expected & found).values())
    n_ref, n_hyp = sum(expected.values()), sum(found.values())
    return 0.0 if correct == 0 else 2.0 * correct / (n_ref + n_hyp)


def score(inputs: Mapping[str, Any], params: object) -> Metric:  # noqa: ARG001 - the task takes no params
    """Return the `span_f1` of `inputs["hypothesis"]` against `inputs["reference"]`."""
    ref: TextSpans = inputs["reference"]
    hyp: TextSpans = inputs["hypothesis"]
    return Metric(name="span_f1", value=f1(ref, hyp), unit="ratio", items=len(ref.start) + len(hyp.start))


def handle(state: FakeFaults, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one `metric@1.0` wire form per item, or its fault."""
    return _worker.serve(state, items, params, TaskParams, score)
