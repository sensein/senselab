"""Workers for the lease stress test: separate processes contending for one stale lease, round after round.

`setup` runs once per worker process and records, at the `os` boundary, the token set of every lease directory the
process renames. `take` contends for the lease once and returns what it saw.
"""

import os
import time
from pathlib import Path
from typing import Any

from senselab_core.runtime.locks import Lease, LeaseTimeout

HOLD_S = 0.01
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


def take(resource: str, start_at: float) -> dict[str, Any]:
    """Wait for `start_at` (wall clock), take the lease on `resource`, hold it briefly; report what happened."""
    RENAMED.clear()
    time.sleep(max(0.0, start_at - time.time()))
    report: dict[str, Any] = {"held": False, "broke": 0, "displaced": False, "timeout": None, "token": None}
    try:
        with Lease(Path(resource), timeout=5, stale_after=60) as lease:
            report.update(held=True, broke=len(lease.broke), token=(lease.holder() or {}).get("token"))
            time.sleep(HOLD_S)
            report["displaced"] = not lease.owns()
    except LeaseTimeout as err:
        report["timeout"] = str(err)
    report["renamed"] = list(RENAMED)
    return report
