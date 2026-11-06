"""Samples of the job engine's documents and records, for the value contracts."""

from collections.abc import Callable

from senselab_core.kernel.identity import gather_identity
from senselab_core.kernel.placement import NodePlacement
from senselab_core.runtime.execution.events import BrokenEnv, Event
from senselab_core.runtime.execution.executor import (
    Admission,
    FaultRecord,
    Handle,
    JobRef,
    PlanFragment,
    ResultRef,
    UpstreamRef,
)

Register = Callable[..., None]

_PLAN = "ab" * 32
_INPUT = "content:" + "cd" * 32
_OUTPUT = "output:" + "ef" * 32
_STORED = UpstreamRef(key_digest=_PLAN, outputs="output:" + "12" * 32, port="value")
_PORT = _STORED.identity()
_GATHER = gather_identity([_PORT])


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
                inputs=({"x": _INPUT}, {"x": _PORT}),
                upstream={_PORT: _STORED},
                placement=_placement(),
            ),
            PlanFragment(
                plan_hash=_PLAN,
                node="n",
                admission="admission-2",
                jobs=jobs[:1],
                inputs=({"xs": _GATHER},),
                upstream={_PORT: _STORED},
                gathered={_GATHER: (_PORT,)},
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
            {"upstream": {_OUTPUT: _STORED}},
            {"gathered": {"gather:" + "0" * 64: (_PORT,)}},
        ],
    )
    register(
        ResultRef,
        lambda: [
            ResultRef(job=jobs[0], key_digest=_PLAN, output_identity=_OUTPUT),
            ResultRef(job=jobs[1], key_digest=_PLAN, failure=faults[0]),
            ResultRef(job=jobs[0], key_digest=_PLAN, output_identity=_OUTPUT, served="exact"),
        ],
        invalid_updates=[
            {"key_digest": "k"},
            {"output_identity": _OUTPUT, "failure": faults[0].model_dump()},
            {"served": "other"},
        ],
    )
    register(
        UpstreamRef,
        lambda: [_STORED, UpstreamRef(key_digest="cd" * 32, outputs=_OUTPUT, port="spans")],
        invalid_updates=[{"key_digest": "k"}, {"outputs": _INPUT}, {"port": ""}],
    )
    register(
        Admission,
        lambda: [Admission(id="admission-0", plan_hash=_PLAN), Admission(id="admission-1", plan_hash="cd" * 32)],
        invalid_updates=[{"id": ""}, {"plan_hash": "p"}],
    )
    register(Handle, lambda: [Handle(id="fragment-0"), Handle(id="fragment-1")], invalid_updates=[{"id": ""}])
    register_event_samples(register, jobs, faults)


def register_event_samples(register: Register, jobs: tuple[JobRef, ...], faults: tuple[FaultRecord, ...]) -> None:
    """Register the samples of the run events and the broken envs, which name `jobs` and `faults`."""
    broken = faults[1]
    register(
        Event,
        lambda: [
            Event(kind="queued", node="n", job=jobs[0], attempt="a-1", seq=0, t_ns=1),
            Event(
                kind="finished",
                node="n",
                job=jobs[1],
                key=_PLAN,
                attempt="a-2",
                seq=3,
                placement_digest="12" * 32,
                env="host",
                output=_OUTPUT,
                t_ns=2,
            ),
            Event(kind="failed", node="n", job=jobs[0], fault="content", failure=faults[0], attempt="a", seq=1, t_ns=3),
            Event(
                kind="cache_hit", node="n", job=jobs[1], reuse="equivalent", output=_OUTPUT, attempt="a", seq=2, t_ns=4
            ),
            Event(kind="verify", node="n", attempt="a", seq=4, t_ns=5),
            Event(kind="blocked", node="n", job=jobs[0], attempt="a", seq=5, t_ns=6),
            Event(kind="cancelled", node="n", job=jobs[1], attempt="a", seq=6, t_ns=7),
        ],
        invalid_updates=[
            {"format": "run-event@2.0"},
            {"kind": "other"},
            {"attempt": ""},
            {"seq": -1},
            {"t_ns": -1},
            {"fault": "resource"},
            {"reuse": "other"},
            {"key": "k"},
            {"placement_digest": "x"},
        ],
    )
    register(
        BrokenEnv,
        lambda: [BrokenEnv(env="venv:a", fault=broken, failures=6), BrokenEnv(env="venv:b", fault=broken, failures=9)],
        invalid_updates=[{"env": ""}, {"failures": 0}],
    )
