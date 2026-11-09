"""A backend out of memory, through `execute` in a real pool worker: the pool's `oom` record is at the request's batch.

Each item's error record names the fault and the request's observation says `resource`; nothing raises on the host.
"""

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
from backends.torch_oom_backend_fixture import HUGE
from lease_clock_fixture import ManualClock
from pool_probe_test import JOBS, SEEN, TEXTY, _plan, _pool
from pool_stack_fixture import packaged, registry
from pool_worker_test import WORKERS, _sender
from senselab_core.kernel.declare import Env
from senselab_core.runtime.calibration import (
    Cotenancy,
    Estimate,
    MemoryFit,
    Observation,
    OomRecord,
    RequestFit,
    StratumBatch,
)
from senselab_core.runtime.execution.executor import Handle, ResultRef
from senselab_core.runtime.execution.in_process import node_fragment
from senselab_core.runtime.execution.pool import Pool
from senselab_core.runtime.execution.pool_worker import PoolWorker
from senselab_core.runtime.execution.queue import RUN_QUEUES
from senselab_core.runtime.execution.run import prepare
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.store import ResultStore
from senselab_core.testing.calibration_samples import UNIT
from senselab_core.testing.measures import FixedEstimates, FixedUnit

pytestmark = pytest.mark.logic

CROWDED = "backends.crowding_backend_fixture"
PEERS = Cotenancy(device=None, host=3)
ESTIMATE = Estimate(
    status="measured",
    unit_kind="chars",
    load_s=1.0,
    load_n=3,
    item_s=0.01,
    item_n=3,
    single=RequestFit(batch=1, r=0.1, u=0.01, n=6, spread=0.1, lo=1.0, hi=100.0),
    batched=(RequestFit(batch=4, r=0.1, u=0.001, n=6, spread=0.1, lo=1.0, hi=400.0),),
    batch=(StratumBatch(lo=1.0, hi=100.0, batch=4),),
    host=MemoryFit(resident=2**20, m0=0.0, m1=1.0, n=6),
)
"""Every job at a batch of 4, which the crowding backend cannot hold."""


@contextmanager
def _crowded(tmp_path: Path, env: Env | None) -> Iterator[ExecutionProfile]:
    """Serve the text task by the crowding backend in `env`; yield a profile granting each cpu worker memory."""
    override = {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {"device": "cpu", "mem_bytes": 2**33, "max_workers": 2}},
        "pool": {"poll_s": 0.2, "first_poll_s": 0.01, "quantum_items": 10},
        "calibration": {"measures_roots": [str(tmp_path / "measures")], "warmup_requests": 1},
        "env_roots": {"isolated": str(tmp_path / "envs"), "server": str(tmp_path / "envs")},
    }
    with packaged(registry(TEXTY, worker=CROWDED, env=env), tmp_path / "cache"):
        yield ExecutionProfile.load("laptop", override)
    WORKERS.close()


def _served(tmp_path: Path, profile: ExecutionProfile, clock: ManualClock, lengths: Sequence[int]) -> Pool:
    """Run one fragment over `lengths` through the pool and one real worker beside three peers; return the pool."""
    placed, values = _plan(profile, lengths)
    prepare(placed, tmp_path / "run", ResultStore(tmp_path / "store"))
    pool = _pool(tmp_path, profile, clock=clock, measures=FixedEstimates(ESTIMATE), resolver=FixedUnit(profile))
    handle = pool.submit(node_fragment(placed, pool.admit(placed, values))).id
    worker = PoolWorker(
        tmp_path / "run" / RUN_QUEUES,
        ["cpu"],
        profile=profile,
        store=pool.store,
        send=_sender(pool.store, profile),
        hardware=lambda _: SEEN,
        device_memory=lambda: None,
        cotenancy=lambda _: PEERS,
    )
    for _ in range(JOBS):
        worker.serve()
        if pool.poll(Handle(id=handle))[1]:
            return pool
    raise AssertionError("the fragment did not end")


def _observed(pool: Pool) -> list[Observation]:
    outcomes = pool.queues.queue("cpu").path / "outcome"
    return [
        Observation.model_validate(o)
        for path in sorted(outcomes.iterdir())
        if not path.name.startswith(".")
        for o in json.loads(path.read_bytes()).get("observations", [])
    ]


@pytest.mark.parametrize("env", [None, Env.isolated("crowd")], ids=["host", "isolated"])
def test_a_backend_oom_at_batch_four_is_recorded_at_batch_four(
    tmp_path: Path, clock: ManualClock, env: Env | None
) -> None:
    """The handle raises inside the worker, yet each request says `resource` and the record is at batch 4.

    It is filed at co-tenancy 1 beside three peers, because each worker is granted its own memory.
    """
    with _crowded(tmp_path, env) as profile:
        pool = _served(tmp_path, profile, clock, [3, 12, 25, 60, 90, 7, 33, 41])
    found = _observed(pool)
    assert {(o.batch, o.outcome, o.fault) for o in found} == {(4, "failed", "resource")}
    [record] = [OomRecord.model_validate(r.record) for r in pool.roots.records(UNIT) if r.kind == "oom"]
    assert (record.resource, record.batch, record.cotenancy, record.grant_bytes) == ("host", 4, 1, 2**33)


TORCH = "backends.torch_oom_backend_fixture"
BATCH = 12
"""The batch the estimate sizes every job at."""
BATCHED = ESTIMATE.model_copy(
    update={"batched": (RequestFit(batch=BATCH, r=0.1, u=0.001, n=6, spread=0.1, lo=1.0, hi=400.0),)}
).model_copy(update={"batch": (StratumBatch(lo=1.0, hi=100.0, batch=BATCH),)})


def test_a_torch_style_oom_at_batch_twelve_bisects_and_does_not_hold_the_env(
    tmp_path: Path, clock: ManualClock
) -> None:
    """The handle raises a plain `OutOfMemoryError`: the halves that fit succeed, the item that cannot dies, no hold.

    The oom record is at the faulting batch, 12, and the env is not held (`poll` raises no `QueueHeld`).
    """
    lengths = [3, 12, 25, 60, 90, 7, 33, 41, 5, 9, 14, 20]
    assert max(lengths) > HUGE
    override = {
        "devices": ["cpu"],
        "resource_classes": {"cpu": {"device": "cpu", "mem_bytes": 2**33, "max_workers": 2}},
        "pool": {"poll_s": 0.2, "first_poll_s": 0.01, "quantum_items": 20},
        "calibration": {"measures_roots": [str(tmp_path / "measures")], "warmup_requests": 1},
        "env_roots": {"isolated": str(tmp_path / "envs"), "server": str(tmp_path / "envs")},
    }
    with packaged(registry(TEXTY, worker=TORCH, env=None), tmp_path / "cache"):
        profile = ExecutionProfile.load("laptop", override)
        placed, values = _plan(profile, lengths)
        prepare(placed, tmp_path / "run", ResultStore(tmp_path / "store"))
        pool = _pool(tmp_path, profile, clock=clock, measures=FixedEstimates(BATCHED), resolver=FixedUnit(profile))
        handle = pool.submit(node_fragment(placed, pool.admit(placed, values))).id
        worker = PoolWorker(
            tmp_path / "run" / RUN_QUEUES,
            ["cpu"],
            profile=profile,
            store=pool.store,
            send=_sender(pool.store, profile),
            hardware=lambda _: SEEN,
            device_memory=lambda: None,
            cotenancy=lambda _: PEERS,
        )
        results: list[ResultRef] = []
        for _ in range(JOBS):
            worker.serve()
            landed, done = pool.poll(Handle(id=handle))
            results.extend(landed)
            if done:
                break
        WORKERS.close()
    assert done
    found = _observed(pool)
    assert any(o.outcome == "bisected" for o in found)
    assert sum(r.failure is None for r in results) == len(lengths) - 1
    [dead] = [r.failure for r in results if r.failure is not None]
    assert (dead.fault, dead.type, dead.gap) == ("resource", "OutOfMemoryError", False)
    assert {o.batch for o in found if o.fault == "resource"} >= {BATCH}
    [record] = [OomRecord.model_validate(r.record) for r in pool.roots.records(UNIT) if r.kind == "oom"]
    assert record.batch == BATCH
    assert not pool.gaps
