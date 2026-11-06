"""Claimers for the queue race test: separate processes claiming one item, one of them delayed at the `os` boundary.

`claim_and_commit` waits for a wall-clock start, claims one item and commits it. With `delay`, the process first makes
its lease renewer's beats and its outcome links sleep for that long, so another process can break its claim while it
holds it.
"""

import os
import time
from pathlib import Path
from typing import Any

from senselab_core.runtime.execution.queue import QueueSet
from senselab_core.runtime.profiles import ExecutionProfile


def plant_delay(seconds: float) -> None:
    """Make every beat `mkdir` and every hard link in this process sleep `seconds` before it runs."""
    real_mkdir, real_link = os.mkdir, os.link

    def mkdir(path: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        if os.path.basename(os.fsdecode(path)).startswith("beat-"):
            time.sleep(seconds)
        real_mkdir(path, *args, **kwargs)

    def link(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        time.sleep(seconds)
        real_link(src, dst, *args, **kwargs)

    os.mkdir = mkdir
    os.link = link


def claim_and_commit(root: str, override: dict[str, Any], start_at: float, delay: float) -> dict[str, Any]:
    """Wait for `start_at`, claim one item from queue `cpu` and commit it; report what happened."""
    if delay:
        plant_delay(delay)
    queues = QueueSet(Path(root), ExecutionProfile.load(override=override))
    time.sleep(max(0.0, start_at - time.time()))
    claim = queues.queue("cpu").claim(f"w{os.getpid()}", 1)
    if claim is None:
        return {"claimed": False, "committed": [], "outcomes": {}, "broke": 0}
    broke = sum(claim.broke(d) for d in claim.items)
    committed = claim.commit()
    return {"claimed": True, "committed": list(committed), "outcomes": claim.outcomes, "broke": broke}


def ready(hold_s: float) -> int:
    """Return this process's pid after `hold_s` seconds, once its imports are done."""
    time.sleep(hold_s)
    return os.getpid()
