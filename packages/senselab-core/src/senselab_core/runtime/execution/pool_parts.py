"""The parts of the `Pool`: its items and their cuts, what a followed fragment and a unit's probe state hold."""

import contextlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

from senselab_core.runtime.calibration import Estimate, Observation, ProbeItem, ProbePlan
from senselab_core.runtime.execution.executor import FaultRecord, PlanFragment, QueueHeld
from senselab_core.runtime.execution.in_process import job_units, result_ref
from senselab_core.runtime.execution.job import Job, JobOutcome
from senselab_core.runtime.execution.queue import Claim
from senselab_core.runtime.locks import Lease
from senselab_core.runtime.measures import MeasureUnit
from senselab_core.runtime.planner import Plan
from senselab_core.runtime.probe import probe_regime
from senselab_core.runtime.sizing import Sample


def report(  # noqa: PLR0913 - the claim and item, its outcomes, then what was measured of it
    claim: Claim,
    digest: str,
    outcomes: Sequence[JobOutcome],
    sample: Sample | None = None,
    observations: Sequence[Observation] = (),
    survived: Sequence[FaultRecord] = (),
) -> bool:
    """Commit the claimed item `digest` with each job's `ResultRef`, the `sample` measured and its `observations`.

    `survived` are the infrastructure faults of workers lost to a request whose jobs all completed.

    Return whether it committed.
    """
    extra: dict[str, object] = {"jobs": [result_ref(outcome).model_dump(mode="json") for outcome in outcomes]}
    if sample is not None:
        extra["measure"] = sample.model_dump(mode="json")
    if observations:
        extra["observations"] = [o.model_dump(mode="json") for o in observations]
    if survived:
        extra["survived"] = [f.model_dump(mode="json") for f in survived]
    return digest in claim.commit([digest], extra)


_PER_JOB = frozenset({"jobs", "inputs", "upstream", "gathered"})


@dataclass(frozen=True)
class Dumped:
    """A fragment dumped once: the fields its parts share, and its jobs, inputs, upstream and gathers to slice from."""

    shared: dict[str, object]
    jobs: list[object]
    inputs: list[dict[str, str]]
    upstream: dict[str, object]
    gathered: dict[str, list[str]]

    @classmethod
    def of(cls, fragment: PlanFragment) -> "Dumped":
        """Dump `fragment` once."""
        return cls(
            fragment.model_dump(mode="json", exclude=set(_PER_JOB)),
            [j.model_dump(mode="json") for j in fragment.jobs],
            [dict(i) for i in fragment.inputs],
            {i: r.model_dump(mode="json") for i, r in fragment.upstream.items()},
            {g: list(e) for g, e in fragment.gathered.items()},
        )

    def part(self, indices: Sequence[int], **update: object) -> PlanFragment:
        """Return the fragment of jobs `indices`, in that order, with the inputs they read and `update`."""
        named = {i for index in indices for i in self.inputs[index].values()}
        gathered = {g: self.gathered[g] for g in named if g in self.gathered}
        read = {i for i in named if i.startswith("output:")} | {e for es in gathered.values() for e in es}
        return PlanFragment.model_validate(
            {
                **self.shared,
                "jobs": [self.jobs[i] for i in indices],
                "inputs": [self.inputs[i] for i in indices],
                "upstream": {i: self.upstream[i] for i in sorted(read) if i in self.upstream},
                "gathered": gathered,
                **update,
            }
        )


def jobs_of(fragment: PlanFragment, indices: Sequence[int], **update: object) -> PlanFragment:
    """Return the fragment of jobs `indices` of `fragment`, in that order, with the inputs they read and `update`."""
    return Dumped.of(fragment).part(indices, **update)


def split(fragment: PlanFragment, quantum: int, request: int = 1) -> list[PlanFragment]:
    """Return `fragment` as items of at most `quantum` jobs each, in job order; `[fragment]` when it fits in one.

    The fragment is dumped once and each item is built from its own slice, so the work is linear in the jobs.

    `request` is the jobs per request of the fragment's backend (`node_request_size`). When it is at most `quantum`,
    every item but the last holds the most whole requests that fit, so no request is cut; a larger request is cut at
    `quantum` jobs.

    Raises:
        ValueError: `quantum` or `request` is below 1.
    """
    if quantum < 1 or request < 1:
        raise ValueError(f"an item and a request hold at least one job, got {quantum} and {request}")
    count = len(fragment.jobs)
    if count <= quantum:
        return [fragment]
    size = quantum - quantum % request if request <= quantum else quantum
    dumped = Dumped.of(fragment)
    return [dumped.part(range(start, min(start + size, count))) for start in range(0, count, size)]


@dataclass
class Flight:
    """One item on a queue, by its digest."""

    queue: str
    digest: str
    fragment: PlanFragment


@dataclass
class Followed:
    """A submitted fragment: its items in flight, its computed parts held on a unit, each job's charges and moves."""

    flights: list[Flight]
    waiting: list[PlanFragment] = field(default_factory=list)
    charged: Counter[tuple[int, ...]] = field(default_factory=Counter)
    moved: Counter[tuple[int, ...]] = field(default_factory=Counter)
    held: list[QueueHeld] = field(default_factory=list)
    seen: bool = False
    done: bool = False
    cancelled: bool = False


@dataclass(frozen=True)
class ProbedUnit:
    """A unit this pool probed: the probe's id, the digests of the items it put and the jobs they hold."""

    probe: str
    items: frozenset[str]
    jobs: int


@dataclass
class Held:
    """Computed jobs of one fragment waiting on a unit: its owner, its queue and the part.

    `following` holds the digests of the live items whose jobs these already are: the unit sizes them for the scaler
    and the drift reader, and nothing is put.
    """

    followed: Followed
    queue: str
    part: PlanFragment
    following: tuple[str, ...] = ()


@dataclass
class Run:
    """One probe in flight: its queue, plan, the items awaited, the jobs they use and what they observed."""

    id: str
    queue: str
    plan: ProbePlan
    held: Held
    started: float
    prior: tuple[Observation, ...]
    stage: Literal["single", "batched"] = "single"
    awaiting: dict[str, tuple[ProbeItem, Flight]] = field(default_factory=dict)
    used: set[int] = field(default_factory=set)
    put: dict[str, int] = field(default_factory=dict)
    observations: list[Observation] = field(default_factory=list)
    last: float = 0.0
    lease: Lease | None = None
    stack: contextlib.ExitStack | None = None


@dataclass
class State:
    """The `ProbeState` of a unit this run: `peer_wait`, `probing`, `confirming`, `measured` or `unmeasured`.

    A `confirming` unit has no `run` until the next part of the unit arrives to be held; `suspect` is the estimate
    the confirming probe tests. `regimes` are the co-tenancies a regime record was written for this run, `m1_floor`
    the `m1` a device peak above band raised the unit to this run.
    """

    unit: MeasureUnit
    queue: str
    state: Literal["peer_wait", "probing", "confirming", "measured", "unmeasured"]
    since: float
    estimate: Estimate | None = None
    held: list[Held] = field(default_factory=list)
    run: Run | None = None
    checked: float | None = None
    predicted: bool = False
    suspect: Estimate | None = None
    regimes: set[int] = field(default_factory=set)
    confirmed: bool = False
    m1_floor: float | None = None


@dataclass(frozen=True)
class Release:
    """Computed jobs a unit's end lets go: the part, and the unit and the estimate to size it by."""

    held: Held
    estimate: Estimate | None
    unit: MeasureUnit | None


@dataclass(frozen=True)
class Origin:
    """What an item was put under: its unit, the estimate that sized it, its queue and, for a probe item, its probe."""

    unit: MeasureUnit
    estimate: Estimate | None
    queue: str
    probe: str | None


def worth(plan: ProbePlan, estimate: Estimate | None) -> bool:
    """Return whether `plan` is worth running: it has a `SINGLE` item, or warm-ups that a batched phase can use.

    A batched phase needs a measured single fit and a device memory fit (`Probe.plan_batched`); warm-ups alone, with
    neither, measure nothing.
    """
    if any(i.phase == "SINGLE" for i in plan.items):
        return True
    return bool(plan.items and plan.batched and estimate is not None and estimate.device is not None)


def covers(estimate: Estimate | None, jobs: Sequence[Job]) -> bool:
    """Return whether `estimate` measures every length of `jobs`: it has no unit of length, or a slope and the range.

    An `oom` record a probe can lift (`probe_regime`) leaves its stratum uncovered.
    """
    single = None if estimate is None else estimate.single
    if estimate is None or single is None or any(probe_regime(r) for r in estimate.oom):
        return False
    if estimate.unit_kind == "none":
        return True
    lengths = [job_units(j, estimate.unit_kind) for j in jobs]
    return single.u is not None and all(n is None or single.lo <= n <= single.hi for n in lengths)


def placed_backends(placed: Plan) -> dict[str, set[tuple[str, str]]]:
    """Return the backends, as `(task, backend name)`, the plan places on each queue."""
    found: dict[str, set[tuple[str, str]]] = {}
    tasks = {n.id: n.task for n in placed.logical.graph.nodes}
    for node, record in placed.placement.nodes.items():
        model = placed.logical.models.get(node)
        name = record.resource_class if model is None else f"{record.resource_class}#{model.identity}"
        found.setdefault(name, set()).add((tasks[node], record.backend))
    return found
