"""The `Executor` protocol and the documents it carries: plan fragments in, result references out."""

from collections.abc import Iterator, Mapping
from enum import Enum
from typing import Annotated, Any, ClassVar, Literal, Protocol

from pydantic import AfterValidator, BeforeValidator, Field, PlainSerializer, StringConstraints, model_validator
from senselab_worker.faults import FaultClass

from senselab_core.kernel._base import Frozen, FrozenMapping, NonEmptyStr, thawed
from senselab_core.kernel.formats import PLAN_FRAGMENT_FORMAT, FormatTag, Formatted
from senselab_core.kernel.graph import NodeId
from senselab_core.kernel.identity import InputIdentity, OutputIdentity
from senselab_core.kernel.placement import NodePlacement

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

    A gap fault, one the fallback table classified, is never charged against a fault budget.
    """

    fault: FaultName
    type: NonEmptyStr
    message: str
    gap: bool
    charged: bool

    @model_validator(mode="after")
    def _check_gap(self) -> "FaultRecord":
        if self.gap and self.charged:
            raise ValueError("a gap fault is never charged")
        return self

    @property
    def fault_class(self) -> FaultClass:
        """The fault class, as a `FaultClass`."""
        return FaultClass(self.fault)

    @classmethod
    def of(cls, error: Mapping[str, Any]) -> "FaultRecord":
        """Return the record of a worker's error object (`error_record`): charged exactly when it is not a gap."""
        gap = bool(error.get("gap", True))
        return cls(
            fault=FaultClass(error["fault"]).value,
            type=str(error["type"]),
            message=str(error.get("message", "")),
            gap=gap,
            charged=not gap,
        )


class PlanFragment(Formatted):
    """A `plan-fragment@1.0` document: jobs of one node of one plan, their input identities, the placement, the attempt.

    `inputs` holds one mapping per job, in the order of `jobs`; every job names `plan_hash` and `node`.
    """

    format: FormatTag = PLAN_FRAGMENT_FORMAT
    port_value: ClassVar[bool] = False
    plan_hash: Hex64
    node: NodeId
    jobs: Annotated[tuple[JobRef, ...], Field(min_length=1)]
    inputs: tuple[JobInputs, ...]
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
        return self

    def inputs_of(self, job: JobRef) -> Mapping[str, InputIdentity]:
        """Return the input identities of `job`.

        Raises:
            KeyError: `job` is not in this fragment.
        """
        try:
            return self.inputs[self.jobs.index(job)]
        except ValueError:
            raise KeyError(job) from None


class ResultRef(Frozen):
    """What one job gave: its key's digest and either its output identity or its failure."""

    job: JobRef
    key_digest: Hex64
    output_identity: OutputIdentity | None = None
    failure: FaultRecord | None = None

    @model_validator(mode="after")
    def _check_one(self) -> "ResultRef":
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


class Executor(Protocol):
    """Runs plan fragments: `submit` one, follow it by `status`, stop it by `cancel`, read its jobs by `results`."""

    def submit(self, fragment: PlanFragment) -> Handle:
        """Take `fragment` and return its handle."""

    def status(self, handle: Handle) -> Status:
        """Return where the fragment of `handle` is."""

    def cancel(self, handle: Handle) -> None:
        """Stop the jobs of `handle` not yet run."""

    def results(self, handle: Handle) -> Iterator[ResultRef]:
        """Yield one result per job of `handle` as each finishes."""
