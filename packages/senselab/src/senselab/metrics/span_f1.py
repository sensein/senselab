"""`span_f1`: micro F1 of hypothesis `TextSpans` against reference spans of the same text.

A hypothesis span is correct when a reference span has the same start, end and label; each reference span
is matched at most once. Precision is correct over hypothesis spans, recall correct over reference spans.
"""

from collections import Counter
from collections.abc import Mapping
from typing import Any

from senselab_core.kernel.declare import InputsAgree, TaskParams, backend, task
from senselab_core.kernel.results import Metric, TextSpans
from senselab_worker.fake import FakeFaults

from senselab.metrics import _worker

WORKER_MODULE = "senselab.metrics.span_f1"
load = _worker.load
EMPTY_VALUE = 1.0


@task(
    name="span_f1",
    version="1",
    kind="transform",
    constraints=(InputsAgree(ports=("reference", "hypothesis"), property="text"),),
)
class SpanF1:
    """Score hypothesis text spans against reference spans: exact-match micro F1."""

    inputs = {"reference": TextSpans, "hypothesis": TextSpans}
    outputs = {"metric": Metric}


@backend(task="span_f1", name="numpy-span-f1", worker=WORKER_MODULE, **_worker.HOST_BACKEND)
class NumpySpanF1:
    """Multiset intersection of `(start, end, label)` triples."""


def _triples(spans: TextSpans) -> Counter[tuple[int, int, str]]:
    return Counter(zip(spans.start.tolist(), spans.end.tolist(), (str(x) for x in spans.label.tolist())))


def f1_parts(ref: TextSpans, hyp: TextSpans) -> tuple[float, float]:
    """Return `(2 * correct, reference spans + hypothesis spans)`, the parts of the micro F1 of `hyp` against `ref`.

    Neither side having a span is `(0, 0)`: a perfect score that adds nothing to a pooled ratio.

    Raises:
        ContentFault: if the spans lie over different texts.
    """
    _worker.require_same("text", ref.text, hyp.text)
    expected, found = _triples(ref), _triples(hyp)
    if not expected and not found:
        return 0.0, 0.0
    correct = sum((expected & found).values())
    return 2.0 * correct, float(sum(expected.values()) + sum(found.values()))


def score(inputs: Mapping[str, Any], params: object) -> Metric:  # noqa: ARG001 - the task takes no params
    """Return the `span_f1` of `inputs["hypothesis"]` against `inputs["reference"]`."""
    ref: TextSpans = inputs["reference"]
    hyp: TextSpans = inputs["hypothesis"]
    numerator, denominator = f1_parts(ref, hyp)
    return Metric(
        name="span_f1",
        value=numerator / denominator if denominator else EMPTY_VALUE,
        unit="ratio",
        items=max(1, len(ref.start) + len(hyp.start)),
        numerator=numerator,
        denominator=denominator,
        empty_value=EMPTY_VALUE,
    )


def handle(state: FakeFaults, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one `metric@2.0` wire form per item, or its fault."""
    return _worker.serve(state, items, params, TaskParams, score)
