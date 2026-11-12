"""A run's coordinator epochs: `epoch/<n>`, each taken by one `mkdir`, held under a beating lease (ADR-003).

`epoch/<n>/` holds `lease.json` (its coordinator's whole `lease` section, `run-lease@2`, written as it is taken),
`attempt.json` (the attempt id and ordinal the coordinator claimed under it, written last), `coordinator.lease` (the
`Lease` its coordinator keeps alive) and, once that coordinator ends, `ended`. The highest `n` with an
`attempt.json` is the current epoch; a coordinator of an older one releases no further work.
"""

import contextlib
import json
import os
import re
from collections.abc import Iterator, Sequence
from pathlib import Path

from senselab_core.runtime.clock import Clock, SystemClock
from senselab_core.runtime.execution_settings import JobLease
from senselab_core.runtime.locks import Lease, LeaseState
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.storage import StorageDir, publish

EPOCH_DIR = "epoch"
"""The directory of a run's epochs, under its run directory."""
EPOCH_NAME = re.compile(r"^[1-9][0-9]*$")
"""An epoch: its number, from 1."""
BOUND_FILE = "lease.json"
ATTEMPT_FILE = "attempt.json"
COORDINATOR = "coordinator"
ENDED = "ended"
LEASE_FORMAT = "run-lease@2"
"""The format of an epoch's `lease.json`: `{"format": ..., "lease": <the coordinator's lease section>}`."""


class StaleCoordinator(RuntimeError):
    """A newer epoch than this coordinator's was taken; this coordinator releases no further work."""

    def __init__(self, mine: int, current: int) -> None:
        """Name both epochs."""
        super().__init__(f"this coordinator holds epoch {mine}; epoch {current} was taken since, so it stops")
        self.mine = mine
        self.current = current


def _create(path: Path, data: bytes) -> None:
    """Create `path` holding `data` unless something is there."""
    publish(path, data)


class RunLeaseUnreadable(ValueError):
    """An epoch's `lease.json` is missing, damaged, or lacks a key this build reads: the run is another build's."""

    def __init__(self, path: Path, why: object) -> None:
        """Name the file and what was wrong with it."""
        super().__init__(
            f"{path} is not a {LEASE_FORMAT} lease section ({why}); it was written by another build of senselab, so "
            "the run cannot be resumed; plan and run it again"
        )
        self.path = path


class LeaseMismatch(RuntimeError):
    """A worker's lease section is not the run's newest epoch's; `keys` names the differing keys."""

    def __init__(self, keys: Sequence[str], mine: JobLease, run: JobLease) -> None:
        """Name the differing keys and both sections."""
        super().__init__(
            f"this worker's lease section differs from the run's in {', '.join(keys)} (worker {mine.model_dump()}, "
            f"run {run.model_dump()}); start workers under the profile the coordinator ran with"
        )
        self.keys = tuple(keys)


def _document(lease: JobLease | None) -> bytes:
    """Return `lease.json` for `lease` (the packaged profile's section when None)."""
    if lease is None:
        lease = ExecutionProfile.load().lease
    return json.dumps({"format": LEASE_FORMAT, "lease": lease.model_dump(mode="json")}, sort_keys=True).encode()


def read_section(directory: Path) -> JobLease:
    """Return the lease section `directory/lease.json` holds.

    Raises:
        RunLeaseUnreadable: it is missing, is no `run-lease@2` document, or lacks a key `JobLease` requires.
    """
    path = directory / BOUND_FILE
    try:
        found = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(found, dict) or found.get("format") != LEASE_FORMAT:
            raise ValueError(
                f"its format is {found.get('format') if isinstance(found, dict) else None!r}, not {LEASE_FORMAT}"
            )
        return JobLease.model_validate(found["lease"])
    except (OSError, ValueError, KeyError) as err:
        raise RunLeaseUnreadable(path, err) from err


def combined(sections: Sequence[JobLease]) -> JobLease:
    """Return the lease every judge of a run uses: the longest bounds, the shortest renewal, the rest the newest's.

    `sections` run oldest first; the last is the newest. With none, the packaged profile's section.
    """
    if not sections:
        return ExecutionProfile.load().lease
    return sections[-1].model_copy(
        update={
            "stale_after_s": max(s.stale_after_s for s in sections),
            "attribute_delay_s": max(s.attribute_delay_s for s in sections),
            "renew_fraction": min(s.renew_fraction for s in sections),
        }
    )


def run_lease(run_dir: Path, own: JobLease | None = None) -> JobLease:
    """Return the lease bound every judge and holder of the run in `run_dir` uses (`combined`).

    It is built from the lease section of every epoch with an `attempt.json`, oldest first, and `own` last when given.

    Raises:
        RunLeaseUnreadable: a bound epoch's section is unreadable.
    """
    epochs = Epoch(run_dir)
    sections = [read_section(epochs.root / str(n)) for n in epochs.bound()]
    return combined([*sections, own] if own is not None else sections)


def differing(mine: JobLease, run: JobLease) -> list[str]:
    """Return the keys in which two lease sections differ."""
    return sorted(k for k in JobLease.model_fields if getattr(mine, k) != getattr(run, k))


class Epoch:
    """The epochs of the run in `run_dir`."""

    def __init__(self, run_dir: Path, clock: Clock = SystemClock()) -> None:  # noqa: B008 - stateless
        """Hold the epochs under `run_dir/epoch`; their leases and the storage clock are read on `clock`."""
        self.root = Path(run_dir) / EPOCH_DIR
        self._clock = clock

    def bound(self) -> list[int]:
        """Return the epochs with an `attempt.json`, oldest first."""
        if not self.root.is_dir():
            return []
        found = (int(n) for n in os.listdir(self.root) if EPOCH_NAME.fullmatch(n))
        return sorted(n for n in found if (self.root / str(n) / ATTEMPT_FILE).exists())

    def _taken(self) -> int:
        """Return the highest epoch directory, 0 when none was made."""
        if not self.root.is_dir():
            return 0
        return max((int(n) for n in os.listdir(self.root) if EPOCH_NAME.fullmatch(n)), default=0)

    def current(self) -> int:
        """Return the highest bound epoch (one with an `attempt.json`), 0 when none is."""
        return max(self.bound(), default=0)

    def _make(self) -> int:
        """Make the next epoch directory by `mkdir`, skipping a number another coordinator made, and return `n`."""
        self.root.mkdir(parents=True, exist_ok=True)
        n = self._taken() + 1
        while True:
            try:
                (self.root / str(n)).mkdir()
            except FileExistsError:
                n += 1
                continue
            return n

    def take(self, lease: JobLease | None = None) -> int:
        """Take the next epoch by `mkdir epoch/<n>`, record `lease` in it, and return `n`.

        `lease` is the packaged profile's `lease` section when None. The epoch is current only once `bind` is called.
        """
        n = self._make()
        _create(self.root / str(n) / BOUND_FILE, _document(lease))
        return n

    @contextlib.contextmanager
    def opening(self, lease: JobLease | None = None) -> Iterator[int]:
        """Take the next epoch, recording `lease` in it, and yield it; on exit mark it ended unless it was bound.

        Until `bind`, the epoch is current to no reader: it fences no coordinator and changes no status.
        """
        n = self._make()
        try:
            _create(self.root / str(n) / BOUND_FILE, _document(lease))
            yield n
        finally:
            if not (self.root / str(n) / ATTEMPT_FILE).exists():
                with contextlib.suppress(OSError):
                    _create(self.root / str(n) / ENDED, b"")

    def bind(self, n: int, attempt: str, ordinal: int) -> None:
        """Make epoch `n` current: record the attempt, the `ordinal`-th, its coordinator claimed; written once.

        The last step of a coordinator's setup, so a coordinator that did not become ready was never current.
        """
        _create(self.root / str(n) / ATTEMPT_FILE, json.dumps({"attempt": attempt, "ordinal": ordinal}).encode())

    def attempt(self, n: int) -> tuple[str, int] | None:
        """Return the attempt id and ordinal epoch `n`'s coordinator claimed, None before it claimed one."""
        try:
            found = json.loads((self.root / str(n) / ATTEMPT_FILE).read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            return None
        return str(found["attempt"]), int(found["ordinal"])

    @contextlib.contextmanager
    def held(self, n: int, lease: JobLease) -> Iterator[LeaseState]:
        """Hold epoch `n`'s coordinator lease, kept alive under `lease`, for the block; mark the epoch ended after.

        Raises:
            LeaseTimeout: another coordinator holds epoch `n`.
        """
        directory = self.root / str(n)
        held = Lease(directory / COORDINATOR, settings=lease, timeout=0, clock=self._clock)
        try:
            with held, held.keep_alive(lease.renew_fraction) as state:
                yield state
        finally:
            _create(directory / ENDED, b"")

    def stopped(self) -> bool:
        """Return whether the current epoch's coordinator ended, or its lease is older than the run's stale bound.

        A lease is aged on the storage clock, against `run_lease`. With no epoch bound, the highest epoch taken is
        judged as it is: by its `ended` mark, else by its directory's age against the bound its own `lease.json` holds
        (the packaged profile's when unreadable). False when no epoch was taken.
        """
        n = self.current() or self._taken()
        if n == 0:
            return False
        directory = self.root / str(n)
        if (directory / ENDED).exists():
            return True
        if not (directory / ATTEMPT_FILE).exists():
            try:
                bound = read_section(directory)
            except RunLeaseUnreadable:
                bound = combined([])
        else:
            bound = run_lease(self.root.parent)
        age = Lease(directory / COORDINATOR, settings=bound, clock=self._clock).age()
        if age is None:
            with StorageDir.open_root(self.root) as opened:
                age = self._clock.storage_now(opened) - directory.stat().st_mtime
        return bool(age > bound.stale_after_s + bound.attribute_delay_s)


__all__ = [
    "EPOCH_DIR",
    "Epoch",
    "LeaseMismatch",
    "RunLeaseUnreadable",
    "StaleCoordinator",
    "combined",
    "differing",
    "read_section",
    "run_lease",
]
