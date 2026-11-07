"""The `Executor` protocol and the documents it carries: plan fragments in, result references out."""

from collections.abc import Iterator, Mapping
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Protocol

from pydantic import AfterValidator, BeforeValidator, Field, PlainSerializer, StringConstraints, model_validator
from senselab_worker.faults import FaultClass

from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, thawed
from senselab_core.kernel.formats import PLAN_FRAGMENT_FORMAT, FormatTag, Formatted
from senselab_core.kernel.graph import NodeId
from senselab_core.kernel.identity import (
    GatherIdentity,
    InputIdentity,
    OutputIdentity,
    gather_identity,
    port_output_identity,
)
from senselab_core.kernel.placement import NodePlacement
from senselab_core.kernel.plan import ReuseName
from senselab_core.runtime.planner import Plan

if TYPE_CHECKING:
    from senselab_core.runtime.execution.events import EventLog

Hex64 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def _frozen(v: Mapping[str, InputIdentity]) -> Mapping[str, InputIdentity]:
    return FrozenMapping(v)


JobInputs = Annotated[Mapping[str, InputIdentity], AfterValidator(_frozen), PlainSerializer(thawed)]
"""One job's input identities by port."""


class JobRef(Frozen):
    """One job of a plan: the plan hash, the node, and the node's map indices (`()` for an unmapped node)."""

    plan_hash: Hex64
    node: NodeId
    element: tuple[Annotated[int, Field(ge=0)], ...] = ()


def _fault_name(v: object) -> object:
    return v.value if isinstance(v, FaultClass) else v


FaultName = Annotated[Literal["content", "resource", "infrastructure"], BeforeValidator(_fault_name)]
"""A `FaultClass` by its value; given a `FaultClass`, its value is held."""


class FaultRecord(Frozen):
    """A job's failure: its fault class, the exception's type and message, whether it is a gap, whether it is charged.

    Only a content fault that is not a gap is charged against the item's retry budget; a resource or infrastructure
    fault re-places the job, and a gap fault, one the fallback table classified, is never charged.
    """

    fault: FaultName
    type: NonEmptyStr
    message: str
    gap: bool
    charged: bool

    @model_validator(mode="after")
    def _check_charged(self) -> "FaultRecord":
        if self.charged and (self.gap or self.fault != FaultClass.CONTENT.value):
            raise ValueError("only a content fault that is not a gap is charged")
        return self

    @property
    def fault_class(self) -> FaultClass:
        """The fault class, as a `FaultClass`."""
        return FaultClass(self.fault)

    @classmethod
    def of(cls, error: Mapping[str, Any]) -> "FaultRecord":
        """Return the record of a worker's error object (`error_record`): charged for a content fault, not a gap."""
        gap = bool(error.get("gap", True))
        fault = FaultClass(error["fault"])
        return cls(
            fault=fault.value,
            type=str(error["type"]),
            message=str(error.get("message", "")),
            gap=gap,
            charged=fault is FaultClass.CONTENT and not gap,
        )


class UpstreamRef(Frozen):
    """Where an `output:` input is stored: the upstream job's key digest, its entry's output identity, and the port."""

    key_digest: Hex64
    outputs: OutputIdentity
    port: NonEmptyStr

    def identity(self) -> OutputIdentity:
        """Return the input identity of this port (`port_output_identity`)."""
        return port_output_identity(self.key_digest, self.port)


def _upstream(v: Mapping[str, UpstreamRef]) -> Mapping[str, UpstreamRef]:
    return FrozenMapping(v)


def _upstream_form(v: Mapping[str, UpstreamRef]) -> dict[str, object]:
    return {k: ref.model_dump(mode="json") for k, ref in v.items()}


Upstream = Annotated[Mapping[OutputIdentity, UpstreamRef], AfterValidator(_upstream), PlainSerializer(_upstream_form)]
"""Each `output:` identity a fragment's jobs read, alone or gathered, with where the port's value is stored."""


def _gathered(v: Mapping[str, tuple[str, ...]]) -> Mapping[str, tuple[str, ...]]:
    return FrozenMapping(v)


Gathered = Annotated[
    Mapping[GatherIdentity, tuple[OutputIdentity, ...]], AfterValidator(_gathered), PlainSerializer(thawed)
]
"""Each `gather:` input identity a fragment's jobs read, with its elements' port identities in element order."""


class Admission(Frozen):
    """A plan admitted to an executor, with the values of its graph inputs, as the executor names it."""

    id: NonEmptyStr
    plan_hash: Hex64


class PlanFragment(Formatted):
    """A `plan-fragment@1.0` document: jobs of one node of one admitted plan, their input identities, the placement.

    `inputs` holds one mapping per job, in the order of `jobs`; every job names `plan_hash` and `node`. `admission`
    names the admission whose graph-input values the jobs read; `upstream` gives, for each `output:` input
    identity (one port of one upstream job, `port_output_identity`), where that port's value is stored; `gathered`
    gives, for each `gather:` input identity, its elements' port identities in element order, each in `upstream`.
    `attempt` counts the submissions of these jobs.
    """

    format: FormatTag = PLAN_FRAGMENT_FORMAT
    port_value: ClassVar[bool] = False
    plan_hash: Hex64
    node: NodeId
    admission: NonEmptyStr
    jobs: Annotated[tuple[JobRef, ...], Field(min_length=1)]
    inputs: tuple[JobInputs, ...]
    upstream: Upstream = FrozenMapping({})
    gathered: Gathered = FrozenMapping({})
    placement: NodePlacement
    attempt: Annotated[int, Field(ge=1)] = 1

    @model_validator(mode="after")
    def _check_jobs(self) -> "PlanFragment":
        if len(self.inputs) != len(self.jobs):
            raise ValueError(f"a fragment of {len(self.jobs)} jobs holds {len(self.inputs)} input mappings")
        if len(set(self.jobs)) != len(self.jobs):
            raise ValueError("a fragment names each job once")
        if any((j.plan_hash, j.node) != (self.plan_hash, self.node) for j in self.jobs):
            raise ValueError(f"every job of the fragment is of plan {self.plan_hash} node {self.node!r}")
        named = {i for inputs in self.inputs for i in inputs.values()}
        if {i for i in named if i.startswith("gather:")} != set(self.gathered):
            raise ValueError(f"gathered names {sorted(self.gathered)}; the jobs read other gathers")
        if any(gather_identity(elements) != g for g, elements in self.gathered.items()):
            raise ValueError("a gather identity is the identity of its elements in order")
        read = {i for i in named if i.startswith("output:")} | {e for es in self.gathered.values() for e in es}
        if read != set(self.upstream):
            raise ValueError(f"upstream names {sorted(self.upstream)}; the jobs read {sorted(read)}")
        if any(identity != ref.identity() for identity, ref in self.upstream.items()):
            raise ValueError("an upstream identity is the identity of its key digest and port")
        return self


class ResultRef(Frozen):
    """What one job gave: its key's digest and either its output identity or its failure.

    `served` is the reuse policy under which a stored result gave it, None when the job executed.
    """

    job: JobRef
    key_digest: Hex64
    output_identity: OutputIdentity | None = None
    failure: FaultRecord | None = None
    served: ReuseName | None = None

    @model_validator(mode="after")
    def _check_one(self) -> "ResultRef":
        if self.served is not None and self.output_identity is None:
            raise ValueError("only a result that was given can have been served from the store")
        if (self.output_identity is None) == (self.failure is None):
            raise ValueError("a result names exactly one of an output identity and a failure")
        return self


class Handle(Frozen):
    """A submitted fragment, as the executor that took it names it."""

    id: NonEmptyStr


class Status(str, Enum):
    """Where a submitted fragment is: waiting, running, every job done (completed or failed), or cancelled."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"


class QueueHeld(RuntimeError):
    """An executor holds the queue of a fragment's jobs: env `env` failed `failures` times by infrastructure faults.

    The jobs are not failed; they wait in the queue for a later attempt.
    """

    def __init__(self, queue: str, env: str, failures: int) -> None:
        """Name the queue, the env and its failures."""
        super().__init__(f"queue {queue!r} is held: env {env!r} failed {failures} times by infrastructure faults")
        self.queue = queue
        self.env = env
        self.failures = failures


class WorkersBroken(QueueHeld):
    """The workers of a resource class (`*`: the substrate) failed to start `failures` times in a row.

    The jobs are not failed; they wait in the queue for a later epoch, like a held env's. `reason` is the latest cause.
    """

    def __init__(self, queue: str, resource_class: str, failures: int, reason: str) -> None:
        """Name the queue, the class (or `*`), its failures and the latest cause."""
        super().__init__(queue, f"workers:{resource_class}", failures)
        RuntimeError.__init__(
            self, f"queue {queue!r} is held: workers of {resource_class!r} failed to start {failures} times: {reason}"
        )
        self.resource_class = resource_class
        self.reason = reason


class Executor(Protocol):
    """Runs plan fragments of admitted plans: `admit` a plan, `submit` its fragments, follow, stop and read them.

    An admission holds the values of the plan's graph inputs by content identity, until it is released; an
    `output:` input is read from the result store. Two admissions of one plan are independent. `logs_claims` says
    whether the executor's workers log each job `claimed`, in which case the runner logs no `started` as it sends one.
    `splits_fragments` says whether the executor splits a fragment into its own units of work, in which case the runner
    sends the jobs of a node that are ready together as one fragment, not in fragments of the batch size.
    """

    logs_claims: ClassVar[bool]
    splits_fragments: ClassVar[bool]

    def bind_epoch(self, epoch: int, events: "EventLog") -> None:
        """Run what follows as the coordinator of `epoch`, logging to `events` (that epoch's attempt)."""

    def admit(self, plan: Plan, values: Mapping[InputIdentity, object]) -> Admission:
        """Hold `plan` and the value of each graph-input identity its fragments name, and return the admission."""

    def release(self, admission: Admission) -> None:
        """Let go of `admission` and its values."""

    def submit(self, fragment: PlanFragment) -> Handle:
        """Take `fragment` and return its handle."""

    def status(self, handle: Handle) -> Status:
        """Return where the fragment of `handle` is."""

    def cancel(self, handle: Handle) -> None:
        """Stop the jobs of `handle` not yet run."""

    def results(self, handle: Handle) -> Iterator[ResultRef]:
        """Yield one result per job of `handle` as each finishes."""
