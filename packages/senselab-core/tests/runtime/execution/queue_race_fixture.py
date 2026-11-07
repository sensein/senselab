"""Processes for the queue race tests: claimers and putters, some stalled or crashed at the `os` boundary.

Each function waits for a wall-clock start, acts on queue `cpu` under `root` with the profile `override`, and reports
what happened. A stall or crash is installed in the child (`syscall_faults_fixture`) before it acts.
"""

import os
import time
from pathlib import Path
from typing import Any

from senselab_core.kernel.placement import NodePlacement
from senselab_core.runtime.execution.executor import FaultRecord, JobRef, PlanFragment
from senselab_core.runtime.execution.queue import Queue, QueueSet
from senselab_core.runtime.profiles import ExecutionProfile
from syscall_faults_fixture import exit_at, stall

FAULT = FaultRecord(fault="content", type="ContentFault", message="bad", gap=False, charged=True)


def _queue(root: str, override: dict[str, Any]) -> Queue:
    return QueueSet(Path(root), ExecutionProfile.load(override=override)).queue("cpu")


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


def claim_and_finish(root: str, override: dict[str, Any], start_at: float, stall_s: float, kind: str) -> dict[str, Any]:
    """Claim one item and commit it (`kind` `done`) or kill it (`dead`); with `stall_s`, beats and links sleep first.

    The link that sleeps is the outcome link, made after the claim's proof.
    """
    if stall_s:
        stall("mkdir", "beat-", stall_s)
        stall("link", "", stall_s)
    queue = _queue(root, override)
    time.sleep(max(0.0, start_at - time.time()))
    claim = queue.claim(f"w{os.getpid()}", 1)
    if claim is None:
        return {"claimed": False, "finished": [], "outcomes": {}, "broke": 0}
    broke = sum(claim.broke(d) for d in claim.items)
    finished = claim.commit() if kind == "done" else claim.dead(FAULT)
    return {"claimed": True, "finished": list(finished), "outcomes": claim.outcomes, "broke": broke}


def _gated_symlink(gate: Path, bound_s: float, poll_s: float) -> None:
    """Make this process's first `os.symlink` of a `slot-` name touch `gate/stalled`, then wait for `gate/release`."""
    real = os.symlink
    left = [1]

    def symlink(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        if left[0] and os.path.basename(os.fsdecode(dst)).startswith("slot-"):
            left[0] = 0
            (gate / "stalled").touch()
            deadline = time.monotonic() + bound_s
            while not (gate / "release").exists():
                assert time.monotonic() < deadline, "the stalled putter was never released"
                time.sleep(poll_s)
        real(src, dst, *args, **kwargs)

    setattr(os, "symlink", symlink)


def put_stalled(root: str, override: dict[str, Any], admission: str, gate: str, wait: tuple[float, float]) -> int:
    """Put one fragment whose first slot `symlink` waits for `release` in `gate`; return the slots taken.

    The wait begins by creating `stalled` in `gate` and gives up after `wait[0]` seconds, polling every `wait[1]`.
    """
    _gated_symlink(Path(gate), *wait)
    return _queue(root, override).put_all([fragment(admission)])


def commit_then_crash(root: str, override: dict[str, Any]) -> str:
    """Claim one item and commit it, ending the process when the slot's resolved mark is created."""
    exit_at("open", "res-")
    claim = _queue(root, override).claim("crashed", 1)
    assert claim is not None
    claim.commit()
    return "not reached"


def ready(hold_s: float) -> int:
    """Return this process's pid after `hold_s` seconds, once its imports are done."""
    time.sleep(hold_s)
    return os.getpid()
