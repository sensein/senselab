"""Step ordering and the cross-step checks a single `StepSpec` cannot make.

`toposort` orders a manifest set so every step follows its ``needs``; among steps that are ready at
the same time the lexically smallest name goes first, so the order does not depend on file or step
order. `check_expect_sources` checks every expectation that reads another step's ``metrics.json``
(``from_step``, ``tolerance_from`` and band chains or re-draws) against that step's ``metrics_schema``;
`check_shard_needs` checks how a step reads a sharded step it needs. Errors are a
`DagError` with a fixed code and the names of the steps at fault, never a path or a value.
"""

from __future__ import annotations

import heapq
from collections import Counter
from collections.abc import Sequence

from senselab.utils.reproduce.metrics import metric_address_error, metric_quantity
from senselab.utils.reproduce.spec import StepSpec

CODES = frozenset(
    {
        "E_DAG_DUPLICATE_STEP",
        "E_DAG_UNKNOWN_NEED",
        "E_DAG_CYCLE",
        "E_DAG_FROM_STEP_METRIC",
        "E_DAG_FROM_STEP_UNIT",
        "E_DAG_TOLERANCE_METRIC",
        "E_DAG_SHARD_NEED",
    }
)


class DagError(ValueError):
    """A manifest set that cannot be ordered or whose cross-step expectations do not fit.

    ``code`` is one of `CODES`; ``steps`` names the steps at fault, sorted.
    """

    def __init__(self, code: str, steps: Sequence[str]) -> None:
        """Store the code and the step names; the message is the code and the names only.

        Args:
            code: A fixed code from `CODES`.
            steps: The steps at fault.
        """
        self.code = code
        self.steps = tuple(sorted(steps))
        super().__init__(f"{code}: {', '.join(self.steps)}")


def _index(steps: Sequence[StepSpec]) -> dict[str, StepSpec]:
    dupes = [name for name, n in Counter(s.name for s in steps).items() if n > 1]
    if dupes:
        raise DagError("E_DAG_DUPLICATE_STEP", dupes)
    by_name = {s.name: s for s in steps}
    unknown = [s.name for s in steps if set(s.needs) - by_name.keys()]
    if unknown:
        raise DagError("E_DAG_UNKNOWN_NEED", unknown)
    return by_name


def _cycle(by_name: dict[str, StepSpec], left: set[str]) -> list[str]:
    """One cycle among ``left``, the steps Kahn's algorithm could not place: each has a need in ``left``."""
    start = min(left)
    path: list[str] = []
    seen: dict[str, int] = {}
    node = start
    while node not in seen:
        seen[node] = len(path)
        path.append(node)
        node = min(n for n in by_name[node].needs if n in left)
    return path[seen[node] :]


def check_expect_sources(steps: Sequence[StepSpec]) -> None:
    """Check every expectation that reads an upstream step's metrics against that step's schema.

    A ``from_step`` metric must be an address the upstream ``metrics_schema`` declares (a headline,
    a labelled headline, a count or a per-class field), and name the same quantity as the step's own
    ``metric`` (`metrics.metric_quantity`), so a rate is never compared with a count; that an
    expectation ``unit`` sits only on a rate or ``mcc`` is checked by `StepSpec` itself. Each
    ``tolerance_from`` step and each band chain or re-draw step must declare the expectation's own
    ``metric``, because the spread is taken over that metric.

    Args:
        steps: A manifest set.

    Raises:
        DagError: ``E_DAG_DUPLICATE_STEP`` or ``E_DAG_UNKNOWN_NEED`` for a malformed set,
            ``E_DAG_FROM_STEP_METRIC`` for an undeclared or malformed upstream metric,
            ``E_DAG_FROM_STEP_UNIT`` for a quantity mismatch, ``E_DAG_TOLERANCE_METRIC`` for a
            calibration step that does not declare the metric.
    """
    by_name = _index(steps)
    for step in steps:
        for expect in step.expect:
            if expect.from_step is not None:
                upstream = by_name[expect.from_step.step]
                source = expect.from_step.metric
                if metric_address_error(source, upstream.metrics_schema) is not None:
                    raise DagError("E_DAG_FROM_STEP_METRIC", [step.name])
                if metric_quantity(source) != metric_quantity(expect.metric):
                    raise DagError("E_DAG_FROM_STEP_UNIT", [step.name])
            calibration = list(expect.tolerance_from.steps) if expect.tolerance_from is not None else []
            calibration += expect.band.sources() if expect.band is not None else []
            for name in calibration:
                if metric_address_error(expect.metric, by_name[name].metrics_schema) is not None:
                    raise DagError("E_DAG_TOLERANCE_METRIC", [step.name])


def check_shard_needs(steps: Sequence[StepSpec]) -> None:
    """Check that a sharded step is read only through ``{up:<step>}``, the gather directory of its shards.

    It has no single result dir, so it may not be an ``up:`` root of ``inputs`` or ``stage``, nor a step whose
    metrics an expectation reads (``from_step``, ``tolerance_from``, band chains or re-draws).

    Args:
        steps: A manifest set.

    Raises:
        DagError: ``E_DAG_SHARD_NEED`` naming the steps that read a sharded step otherwise.
    """
    _index(steps)
    sharded = {s.name for s in steps if s.shards is not None}
    bad = []
    for step in steps:
        roots = [i.root for i in step.inputs] + [s.root for s in step.stage]
        read = [r.split(":", 1)[1] for r in roots if r.startswith("up:")]
        for expect in step.expect:
            read += [expect.from_step.step] if expect.from_step is not None else []
            read += list(expect.tolerance_from.steps) if expect.tolerance_from is not None else []
            read += expect.band.sources() if expect.band is not None else []
        if sharded & set(read):
            bad.append(step.name)
    if bad:
        raise DagError("E_DAG_SHARD_NEED", bad)


def toposort(steps: Sequence[StepSpec]) -> list[StepSpec]:
    """Order a manifest set so every step follows its ``needs``, after `check_expect_sources` and `check_shard_needs`.

    Args:
        steps: A manifest set.

    Returns:
        The same steps, needs first; ties go to the lexically smallest name.

    Raises:
        DagError: ``E_DAG_DUPLICATE_STEP`` for a repeated name, ``E_DAG_UNKNOWN_NEED`` for a need
            naming no step in the set, ``E_DAG_CYCLE`` (naming the steps on one cycle) for a cycle, or
            as `check_expect_sources` and `check_shard_needs`.
    """
    by_name = _index(steps)
    waiting = {s.name: len(s.needs) for s in steps}
    dependents: dict[str, list[str]] = {name: [] for name in by_name}
    for step in steps:
        for need in step.needs:
            dependents[need].append(step.name)
    ready = [name for name, n in waiting.items() if n == 0]
    heapq.heapify(ready)
    order: list[StepSpec] = []
    while ready:
        name = heapq.heappop(ready)
        order.append(by_name[name])
        for dependent in dependents[name]:
            waiting[dependent] -= 1
            if waiting[dependent] == 0:
                heapq.heappush(ready, dependent)
    if len(order) != len(steps):
        placed = {s.name for s in order}
        raise DagError("E_DAG_CYCLE", _cycle(by_name, set(by_name) - placed))
    check_expect_sources(steps)
    check_shard_needs(steps)
    return order
