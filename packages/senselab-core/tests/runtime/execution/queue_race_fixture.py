"""Processes for the queue race tests: claimers and putters, some stalled or crashed at the `os` boundary.

Each function acts on queue `cpu` under `root` with the profile `override`, and reports
what happened. A stall or crash is installed in the child (`syscall_faults_fixture`) before it acts.
"""

import os
from pathlib import Path
from typing import Any

from senselab_core.kernel.placement import NodePlacement
from senselab_core.runtime.execution.executor import FaultRecord, JobRef, PlanFragment
from senselab_core.runtime.execution.queue import Queue, QueueSet
from senselab_core.runtime.profiles import ExecutionProfile
from syscall_faults_fixture import await_file, exit_at, gate_pause, stall

FAULT = FaultRecord(fault="content", type="ContentFault", message="bad", gap=False, charged=True)


def _queue(root: str, override: dict[str, Any]) -> Queue:
    profile = ExecutionProfile.load(override=override)
    return QueueSet(Path(root), profile, lease=profile.lease).queue("cpu")


def fragment(admission: str) -> PlanFragment:
    """Return a one-job fragment named by `admission`."""
    placement = NodePlacement(
        backend="b",
        backend_version="1",
        backend_params={},
        stack="s",
        env="host",
        device="cpu",
        batch_size=1,
        resource_class="cpu",
    )
    job = (JobRef(plan_hash="ab" * 32, node="n", element=(0,)),)
    return PlanFragment(plan_hash="ab" * 32, node="n", admission=admission, jobs=job, inputs=({},), placement=placement)


def claim_and_finish(
    root: str, override: dict[str, Any], kind: str, gate: str | None = None, wait: tuple[float, float] = (0.0, 0.0)
) -> dict[str, Any]:
    """Claim one item and commit it (`kind` `done`) or kill it (`dead`).

    With `gate`, the claimer's beats wait for `release` there, and its outcome link first creates `stalled` there and
    waits for `release` (`gate_pause`), giving up after `wait[0]` seconds and polling every `wait[1]`.
    """
    if gate is not None:
        release = Path(gate) / "release"
        stall("mkdir", "beat-", lambda: await_file(release, *wait))
        stall("link", "", gate_pause(Path(gate), *wait))
    claim = _queue(root, override).claim(f"w{os.getpid()}", 1)
    if claim is None:
        return {"claimed": False, "finished": [], "outcomes": {}, "broke": 0}
    broke = sum(len(found) for found in claim.broke.values())
    finished = claim.commit() if kind == "done" else claim.dead(FAULT)
    return {"claimed": True, "finished": list(finished), "outcomes": claim.outcomes, "broke": broke}


def put_stalled(root: str, override: dict[str, Any], admission: str, gate: str, wait: tuple[float, float]) -> int:
    """Put one fragment whose first slot `symlink` waits for `release` in `gate`; return the slots taken.

    The wait begins by creating `stalled` in `gate` and gives up after `wait[0]` seconds, polling every `wait[1]`.
    """
    stall("symlink", "slot-", gate_pause(Path(gate), *wait), times=1)
    return _queue(root, override).put_all([fragment(admission)])


def commit_then_crash(root: str, override: dict[str, Any]) -> str:
    """Claim one item and commit it, ending the process when the slot's resolved mark is created."""
    exit_at("open", "res-")
    claim = _queue(root, override).claim("crashed", 1)
    assert claim is not None
    claim.commit()
    return "not reached"
