"""Workers for the lease stress test: separate processes contending for one stale lease, round after round.

`setup` runs once per worker process and records, at the `os` boundary, the token set of every lease directory the
process renames. `take` starts at a barrier made of directories in shared storage, contends for the lease once and
returns what it saw. Its clock never advances, so a waiter polls without a deadline.
"""

import os
import time
import uuid
from pathlib import Path
from typing import Any

from lease_clock_fixture import ManualClock
from senselab_core.runtime.locks import Lease, LeaseTimeout

RENAMED: list[list[str]] = []


def setup() -> None:
    """Record the tokens in each `*.lease` directory this process renames, read from where the rename put it."""
    real_rename = os.rename

    def rename(src: Any, dst: Any, *, src_dir_fd: int | None = None, dst_dir_fd: int | None = None) -> None:  # noqa: ANN401
        real_rename(src, dst, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        if os.fsdecode(src).endswith(".lease"):
            try:
                fd = os.open(dst, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dst_dir_fd)
            except OSError:
                return
            try:
                RENAMED.append(sorted(os.listdir(fd)))
            except OSError:
                pass
            finally:
                os.close(fd)

    os.rename = rename


def take(resource: str, gate: str, takers: int) -> dict[str, Any]:
    """Join the barrier at `gate`, take the lease on `resource` once all `takers` have, and report what happened."""
    RENAMED.clear()
    os.mkdir(os.path.join(gate, uuid.uuid4().hex))
    while len(os.listdir(gate)) < takers:
        time.sleep(0)
    report: dict[str, Any] = {"held": False, "broke": 0, "displaced": False, "timeout": None, "token": None}
    try:
        with Lease(Path(resource), timeout=5, stale_after=60, clock=ManualClock(spends=False)) as lease:
            report.update(held=True, broke=len(lease.broke), token=(lease.holder() or {}).get("token"))
            report["displaced"] = not lease.owns()
    except LeaseTimeout as err:
        report["timeout"] = str(err)
    report["renamed"] = list(RENAMED)
    return report
