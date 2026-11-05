"""Samples of the job engine's documents and records, for the value contracts."""

from collections.abc import Callable

from senselab_core.kernel.placement import NodePlacement
from senselab_core.runtime.execution.executor import Admission, FaultRecord, Handle, JobRef, PlanFragment, ResultRef

Register = Callable[..., None]

_PLAN = "ab" * 32
_INPUT = "content:" + "cd" * 32
_OUTPUT = "output:" + "ef" * 32


def _placement() -> NodePlacement:
    return NodePlacement(
        backend="b",
        backend_version="1",
        backend_params={},
        stack="s",
        env="host",
        device="cpu",
        batch_size=2,
        resource_class="cpu",
    )


def register_execution_samples(register: Register) -> None:
    """Register the samples of every type here through `register` (`values.register_samples`)."""
    jobs = (JobRef(plan_hash=_PLAN, node="n", element=(0,)), JobRef(plan_hash=_PLAN, node="n", element=(1,)))
    register(JobRef, lambda: list(jobs), invalid_updates=[{"plan_hash": "x"}, {"element": (-1,)}, {"node": "a.b"}])
    faults = (
        FaultRecord(fault="content", type="ContentFault", message="bad input", gap=False, charged=True),
        FaultRecord(fault="infrastructure", type="RuntimeError", message="", gap=True, charged=False),
    )
    register(
        FaultRecord,
        lambda: list(faults),
        invalid_updates=[
            {"fault": "other"},
            {"gap": True, "charged": True},
            {"fault": "resource", "charged": True},
            {"type": ""},
        ],
    )
    register(
        PlanFragment,
        lambda: [
            PlanFragment(
                plan_hash=_PLAN,
                node="n",
                admission="admission-0",
                jobs=jobs,
                inputs=({"x": _INPUT}, {"x": _OUTPUT}),
                upstream={_OUTPUT: _PLAN},
                placement=_placement(),
            ),
            PlanFragment(
                plan_hash=_PLAN,
                node="n",
                admission="admission-1",
                jobs=jobs[:1],
                inputs=({},),
                placement=_placement(),
                attempt=2,
            ),
        ],
        invalid_updates=[
            {"format": "plan-fragment@2.0"},
            {"inputs": ()},
            {"attempt": 0},
            {"jobs": ()},
            {"admission": ""},
            {"upstream": {_OUTPUT: "k"}},
        ],
    )
    register(
        ResultRef,
        lambda: [
            ResultRef(job=jobs[0], key_digest=_PLAN, output_identity=_OUTPUT),
            ResultRef(job=jobs[1], key_digest=_PLAN, failure=faults[0]),
        ],
        invalid_updates=[{"key_digest": "k"}, {"output_identity": _OUTPUT, "failure": faults[0].model_dump()}],
    )
    register(
        Admission,
        lambda: [Admission(id="admission-0", plan_hash=_PLAN), Admission(id="admission-1", plan_hash="cd" * 32)],
        invalid_updates=[{"id": ""}, {"plan_hash": "p"}],
    )
    register(Handle, lambda: [Handle(id="fragment-0"), Handle(id="fragment-1")], invalid_updates=[{"id": ""}])
