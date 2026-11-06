"""A run's coordinator epochs: `epoch/<n>`, each taken by one `mkdir`, held under a beating lease (ADR-003).

`epoch/<n>/` holds `lease.json` (the stale bound its coordinator's lease is judged by), `coordinator.lease` (the
`Lease` its coordinator keeps alive) and, once that coordinator ends, `ended`. The highest `n` is the current epoch;
a coordinator of an older one releases no further work.
"""

import contextlib
import json
import os
import re
from collections.abc import Iterator
from pathlib import Path

from senselab_core.runtime.execution_settings import JobLease
from senselab_core.runtime.locks import Lease, LeaseState
from senselab_core.runtime.storage import StorageDir

EPOCH_DIR = "epoch"
"""The directory of a run's epochs, under its run directory."""
EPOCH_NAME = re.compile(r"^[1-9][0-9]*$")
"""An epoch: its number, from 1."""
BOUND_FILE = "lease.json"
COORDINATOR = "coordinator"
ENDED = "ended"


class StaleCoordinator(RuntimeError):
    """A newer epoch than this coordinator's was taken; this coordinator releases no further work."""

    def __init__(self, mine: int, current: int) -> None:
        """Name both epochs."""
        super().__init__(f"this coordinator holds epoch {mine}; epoch {current} was taken since, so it stops")
        self.mine = mine
        self.current = current


def _create(path: Path, data: bytes) -> None:
    """Create `path` holding `data` unless something is there."""
    with contextlib.suppress(FileExistsError):
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)


class Epoch:
    """The epochs of the run in `run_dir`."""

    def __init__(self, run_dir: Path) -> None:
        """Hold the epochs under `run_dir/epoch`."""
        self.root = Path(run_dir) / EPOCH_DIR

    def current(self) -> int:
        """Return the highest epoch taken, 0 when none was."""
        if not self.root.is_dir():
            return 0
        return max((int(n) for n in os.listdir(self.root) if EPOCH_NAME.fullmatch(n)), default=0)

    def take(self) -> int:
        """Take the next epoch by `mkdir epoch/<n>` and return `n`; a number another coordinator made is skipped."""
        self.root.mkdir(parents=True, exist_ok=True)
        n = self.current() + 1
        while True:
            try:
                (self.root / str(n)).mkdir()
            except FileExistsError:
                n += 1
                continue
            return n

    @contextlib.contextmanager
    def held(self, n: int, lease: JobLease) -> Iterator[LeaseState]:
        """Hold epoch `n`'s coordinator lease, kept alive under `lease`, for the block; mark the epoch ended after.

        Raises:
            LeaseTimeout: another coordinator holds epoch `n`.
        """
        directory = self.root / str(n)
        bound = {"stale_after_s": lease.stale_after_s, "attribute_delay_s": lease.attribute_delay_s}
        _create(directory / BOUND_FILE, json.dumps(bound).encode())
        held = Lease(directory / COORDINATOR, timeout=0, stale_after=lease.stale_after_s)
        try:
            with held, held.keep_alive(lease.renew_fraction) as state:
                yield state
        finally:
            _create(directory / ENDED, b"")

    def stopped(self) -> bool:
        """Return whether the current epoch's coordinator ended, or its lease is older than its stale bound.

        A lease is aged on the storage clock; an epoch whose lease was never taken is aged by its directory. False
        when no epoch was taken.
        """
        n = self.current()
        if n == 0:
            return False
        directory = self.root / str(n)
        if (directory / ENDED).exists():
            return True
        try:
            bound = json.loads((directory / BOUND_FILE).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return False
        age = Lease(directory / COORDINATOR, stale_after=bound["stale_after_s"]).age()
        if age is None:
            with StorageDir.open_root(directory) as opened:
                age = opened.dir_now() - directory.stat().st_mtime
        return bool(age > float(bound["stale_after_s"]) + float(bound["attribute_delay_s"]))


__all__ = ["EPOCH_DIR", "Epoch", "StaleCoordinator"]
