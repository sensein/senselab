"""A work queue on shared storage: items, leases, heartbeats and re-publishing, with a worker that runs them.

A pool is a directory that every node can see::

    pool.json                    the pool's settings (schema ``work-pool-1``)
    queue/<item>.json            published, unclaimed items
    leases/<item>/               a claimed item: item.json, owner.json, heartbeat, attempt
    tmp/                         claim, publish and hand-off staging
    interrupted/<item>.<a>.json  attempt ``a`` was released or reaped
    done/<item>.json             the item succeeded (exit code, host, times)
    failed/<item>.<a>.json       attempt ``a`` failed
    dead/<item>.json             the item spent its failure or interruption budget
    conflicts/<item>.<why>.<t>.json  a duplicate copy of an item, kept out of the queue
    workers/<worker>.json        the workers that ran here
    hostcheck/<p>.<worker>.<t>.json  a worker on partition ``p`` failed its host check (before claiming anything)
    controller/                  the controller's lock, the jobs it submitted (jobs/<id>.json), stale markers
    status.json                  written by :meth:`WorkPool.status` (and :class:`Controller`, with ``partitions``)

An item is always in one place. :meth:`WorkPool.claim` takes it with ``rename(queue/<i>.json,
tmp/claim-<token>/item.json)`` into a dir that already holds ``owner.json`` and a heartbeat, and publishes the
lease with ``rename(tmp/claim-<token>, leases/<i>)``. :meth:`WorkPool.complete`, :meth:`WorkPool.fail`,
:meth:`WorkPool.release` and :meth:`WorkPool.reap` all start by renaming ``leases/<i>`` aside, so only one of
them acts on a lease; the next state is written as a new file there and placed with an exclusive ``link(2)``
(``done/``, ``dead/``, or ``queue/`` with attempt+1). After a rename or link reports an error, the destination is
read back for the caller's token, because a retransmitted NFS call can report failure to the process that made
it. Ages are measured on the storage's clock.

``python -m senselab.utils.work_pool work POOL --classes a,b`` runs a worker: claim, run the item's command with
``POOL_ITEM_ID`` and ``POOL_ATTEMPT`` set, heartbeat, then complete or fail; SIGTERM releases the item.

``python -m senselab.utils.work_pool control POOL --classes classes.yaml`` runs a :class:`Controller`: each tick
reaps, counts queued items per class and this pool's Slurm workers per partition, and submits or cancels workers.

Ported and generalised from ``sensein/senselab@feat/task-events`` (f2a806af),
``specs/20261011-slice-pool/pool/slice_pool.py``. Standard library only.
"""

from __future__ import annotations

import argparse
import datetime
import errno
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union

SCHEMA = "work-pool-1"
"""The ``schema`` value of ``pool.json``."""

SUBDIRS = (
    "queue",
    "leases",
    "tmp",
    "interrupted",
    "done",
    "failed",
    "dead",
    "conflicts",
    "workers",
    "hostcheck",
    "controller",
)
"""The pool's subdirectories."""

ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")
"""What an item id may look like (it names files)."""

DEFAULT_MAX_ATTEMPTS = 2
"""Failed attempts an item may have before it goes to ``dead/``."""

DEFAULT_MAX_INTERRUPTS = 6
"""Releases and reaps an item may have before it goes to ``dead/`` (reason ``interrupts``)."""

KILL_GRACE_S = 10.0
"""Seconds between SIGTERM and SIGKILL when a worker stops its child's process group."""

SIGNAL_GRACE_S = 3.0
"""Seconds a worker waits for its own SIGTERM handler after its child died of a signal."""

SIGNAL_EXIT = 143
"""A worker's exit code after SIGTERM."""

PUTBACK_WAIT_S = 1.0
"""Seconds before a lease found absent is checked again: a reaper that renamed a live lease aside puts it back."""

JOIN_GRACE_S = 5.0
"""Extra seconds a worker waits for its heartbeat thread to stop after a run."""

MOVED = ".moved-"
"""Prefix of a marker file created in a dir just renamed into ``tmp/``, so the dir's mtime (its age) counts from
the move. Ages are read from the dir alone: an open of a file inside would make the NFS client keep a
``.nfs*`` copy of it when its owner deletes it, and the owner's removal of the dir would fail."""

REMOVE_RETRY_S = 1.0
"""Seconds a dir removal keeps retrying while an NFS ``.nfs*`` placeholder (a file open elsewhere) blocks it."""

NEXT = "next.json"
"""The next state of an item in a hand-off dir, placed with ``link(2)``."""

HOST_CHECK_EXIT = 3
"""A worker's exit code after its host check failed (it claimed nothing)."""

HOST_CHECK_TIMEOUT_S = 300.0
"""Seconds a host check may run before it counts as failed."""

CONTROLLER_LOCKED_EXIT = 4
"""A controller's exit code when another controller holds the pool's lock."""

PathLike = Union[str, Path]


def _now() -> float:
    return time.time()


def _host() -> str:
    return socket.gethostname().split(".")[0]


GONE_ERRNOS = (errno.ENOENT, errno.ENOTDIR, errno.ESTALE)
"""Errors that mean a path is gone; NFS reports ESTALE for a path another node renamed away mid-lookup."""


def _gone(exc: OSError) -> bool:
    return exc.errno in GONE_ERRNOS


def _names(path: Path) -> list[str]:
    """Directory entries, without dot-files (in-progress writes)."""
    try:
        return [name for name in os.listdir(path) if not name.startswith(".")]
    except OSError as exc:
        if _gone(exc):
            return []
        raise


def _write_json(path: Path, data: Any) -> None:  # noqa: ANN401
    """Write JSON through a temporary file beside it and an atomic rename."""
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:12]}.tmp")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:  # noqa: ANN401
    """The file's JSON; None when it is absent or unreadable as JSON."""
    try:
        with open(path) as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None
    except OSError as exc:
        if _gone(exc):
            return None
        raise


def _exists(path: Path) -> bool:
    """Whether a path exists, asked through an open (revalidated with an NFS server) rather than a cached stat."""
    try:
        os.close(os.open(path, os.O_RDONLY))
    except OSError as exc:
        if _gone(exc):
            return False
        raise
    return True


def _fresh_mtime(path: Path) -> Optional[float]:
    """The file's mtime through an open; None when it is absent."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            return os.fstat(fd).st_mtime
        finally:
            os.close(fd)
    except OSError as exc:
        if _gone(exc):
            return None
        raise


def _fresh_times(path: Path) -> Optional[float]:
    """The later of the path's mtime and ctime through an open; None when it is absent."""
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            st = os.fstat(fd)
            return max(st.st_mtime, st.st_ctime)
        finally:
            os.close(fd)
    except OSError as exc:
        if _gone(exc):
            return None
        raise


def _mark_moved(directory: Path) -> None:
    """Create a fresh marker file in a dir, which sets the dir's mtime on the storage's clock."""
    try:
        os.close(os.open(directory / f"{MOVED}{uuid.uuid4().hex[:12]}", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o660))
    except OSError as exc:
        if not _gone(exc):
            raise


def _remove_dir(path: Path) -> None:
    """Remove a dir tree, retrying briefly while an NFS ``.nfs*`` placeholder blocks it."""
    deadline = time.monotonic() + REMOVE_RETRY_S
    while True:
        shutil.rmtree(path, ignore_errors=True)
        if not _exists(path) or time.monotonic() > deadline:
            return
        time.sleep(0.05)


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def _place(source: Path, target: Path, token: str) -> bool:
    """Link ``source`` to ``target`` only if ``target`` does not exist; True when ``target`` carries ``token``.

    The read-back covers a retransmitted link that landed but reported ``EEXIST``.
    """
    try:
        os.link(source, target)
        return True
    except OSError:
        held = _read_json(target)
        return isinstance(held, dict) and held.get("placed") == token


def _tmp_name(kind: str, token: str) -> str:
    """A ``tmp/`` entry name: its kind, the local time in milliseconds (for reading only) and a token."""
    return f"{kind}-{int(_now() * 1000)}-{token}"


@dataclass(frozen=True)
class Item:
    """One unit of work.

    Attributes:
        id: Unique within the pool; matches :data:`ID_PATTERN`.
        resource_class: A name the caller defines; workers claim the classes they name.
        command: The argv the worker runs.
        env: Extra environment for the command.
        attempt: This run's attempt number, from 1; every failure, release and reap adds one.
        max_attempts: Failed attempts allowed (the pool's ``max_attempts`` when None).
        payload_sha256: sha256 of id, class, command and env (filled by :meth:`WorkPool.publish`).
    """

    id: str
    resource_class: str
    command: Sequence[str]
    env: Mapping[str, str] = field(default_factory=dict)
    attempt: int = 1
    max_attempts: Optional[int] = None
    payload_sha256: str = ""

    def payload_digest(self) -> str:
        """The sha256 of the item's id, class, command and env."""
        payload = {
            "id": self.id,
            "resource_class": self.resource_class,
            "command": list(self.command),
            "env": dict(self.env),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """The item as JSON-ready data."""
        return {
            "id": self.id,
            "resource_class": self.resource_class,
            "command": list(self.command),
            "env": dict(self.env),
            "attempt": self.attempt,
            "max_attempts": self.max_attempts,
            "payload_sha256": self.payload_sha256,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Item:
        """An item from :meth:`to_dict` data (other keys are ignored)."""
        return cls(
            id=data["id"],
            resource_class=data["resource_class"],
            command=tuple(data["command"]),
            env=dict(data.get("env") or {}),
            attempt=int(data.get("attempt", 1)),
            max_attempts=data.get("max_attempts"),
            payload_sha256=data.get("payload_sha256", ""),
        )


@dataclass(frozen=True)
class Lease:
    """A claimed item, held by one worker until it completes, fails, releases or loses it.

    Attributes:
        item: The item, with this run's attempt number.
        token: The claim's token, recorded in ``leases/<item>/owner.json``.
        worker: The claiming worker's name.
        path: ``leases/<item>``.
    """

    item: Item
    token: str
    worker: str
    path: Path


class WorkPool:
    """A pool directory: publish, claim, heartbeat, complete, fail, release, reap and count items."""

    def __init__(self, path: PathLike) -> None:
        """Open an existing pool.

        Args:
            path: The pool directory (made by :meth:`create`).

        Raises:
            ValueError: There is no ``work-pool-1`` pool there.
        """
        self.root = Path(path).absolute()
        cfg = _read_json(self.root / "pool.json")
        if not isinstance(cfg, dict) or cfg.get("schema") != SCHEMA:
            raise ValueError(f"{self.root} is not a {SCHEMA} pool")
        self.lease_s = float(cfg["lease_s"])
        self.heartbeat_s = float(cfg["heartbeat_s"])
        self.max_attempts = int(cfg["max_attempts"])
        self.max_interrupts = int(cfg["max_interrupts"])
        self.success_codes = tuple(int(c) for c in cfg["success_codes"])
        self.host = _host()
        self._classes: dict[str, str] = {}

    @classmethod
    def create(
        cls,
        path: PathLike,
        *,
        lease_s: float,
        heartbeat_s: float,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        max_interrupts: int = DEFAULT_MAX_INTERRUPTS,
        success_codes: Sequence[int] = (0,),
    ) -> WorkPool:
        """Create a pool, or open it when it exists with the same settings.

        Args:
            path: The pool directory, on storage every worker's node can see.
            lease_s: Seconds without a heartbeat after which a lease is reaped.
            heartbeat_s: Seconds between a worker's heartbeats; at most a third of ``lease_s``.
            max_attempts: Failed attempts an item may have (each item may set its own).
            max_interrupts: Releases and reaps an item may have.
            success_codes: Exit codes that complete an item.

        Returns:
            The pool.

        Raises:
            ValueError: A setting is out of range, or the pool exists with other settings.
        """
        if heartbeat_s <= 0 or lease_s < 3 * heartbeat_s:
            raise ValueError("lease_s must be at least three heartbeats, and heartbeat_s positive")
        if max_attempts < 1 or max_interrupts < 1:
            raise ValueError("max_attempts and max_interrupts must be at least 1")
        root = Path(path).absolute()
        cfg = {
            "schema": SCHEMA,
            "lease_s": float(lease_s),
            "heartbeat_s": float(heartbeat_s),
            "max_attempts": int(max_attempts),
            "max_interrupts": int(max_interrupts),
            "success_codes": [int(c) for c in success_codes],
        }
        root.mkdir(parents=True, exist_ok=True)
        for name in SUBDIRS:
            (root / name).mkdir(exist_ok=True)
        old = _read_json(root / "pool.json")
        if old is None:
            _write_json(root / "pool.json", cfg)
        elif old != cfg:
            keys = sorted(k for k in set(old) | set(cfg) if old.get(k) != cfg.get(k))
            raise ValueError(f"{root}/pool.json exists with other settings: {keys}")
        return cls(root)

    def _d(self, name: str) -> Path:
        return self.root / name

    # -- clock and ages -------------------------------------------------------------------------------------------
    def _server_now(self) -> float:
        """The storage's clock: the mtime of a dot-file in ``tmp/`` just touched (a server time on NFS)."""
        path = self._d("tmp") / f".clock-{self.host}-{os.getpid()}-{threading.get_ident()}"
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT, 0o660))
        try:
            os.utime(path)
            stamp = _fresh_mtime(path)
        finally:
            try:
                path.unlink()
            except OSError:
                pass
        return _now() if stamp is None else stamp

    def _entry_time(self, path: Path) -> Optional[float]:
        """When a ``tmp/`` entry last changed: the later of its mtime and ctime (storage clock)."""
        return _fresh_times(path)

    # -- publishing ----------------------------------------------------------------------------------------------
    def _tmp_items(self) -> list[tuple[Path, dict[str, Any]]]:
        """Items held in ``tmp/``: staged claims, hand-offs and publish files."""
        held = []
        for name in _names(self._d("tmp")):
            path = self._d("tmp") / name
            for candidate in (path, path / "item.json", path / NEXT):
                data = _read_json(candidate) if candidate != path or name.endswith(".json") else None
                if isinstance(data, dict) and "id" in data:
                    held.append((path, data))
        return held

    def _known(self, item_id: str) -> bool:
        return self.state(item_id) is not None

    def state(self, item_id: str) -> Optional[str]:
        """Where one item is now: ``done``, ``dead``, ``leased``, ``queued``, ``staged`` (in ``tmp/``) or None.

        A done or dead record wins over any other copy (a ``conflicts/`` record is never counted).

        Args:
            item_id: The item's id.

        Returns:
            The state, or None when the pool holds no such item.

        Raises:
            ValueError: The id does not match :data:`ID_PATTERN`.
        """
        if not ID_PATTERN.fullmatch(item_id):
            raise ValueError(f"item id {item_id!r} does not match {ID_PATTERN.pattern}")
        for state, path in (
            ("done", self._d("done") / f"{item_id}.json"),
            ("dead", self._d("dead") / f"{item_id}.json"),
            ("leased", self._d("leases") / item_id),
            ("queued", self._d("queue") / f"{item_id}.json"),
        ):
            if _exists(path):
                return state
        return "staged" if any(data.get("id") == item_id for _, data in self._tmp_items()) else None

    def publish(self, items: Iterable[Item]) -> list[Item]:
        """Put items in the queue, each with one exclusive ``link(2)`` from ``tmp/``.

        Every item is checked before any is published. An id that appears in the queue between the check and the
        link raises; the items before it stay published.

        Args:
            items: New items; their ids must not be in the pool yet.

        Returns:
            The published items, with ``payload_sha256`` and ``max_attempts`` filled in.

        Raises:
            ValueError: An item is malformed, repeats an id, or is already in the pool.
        """
        ready: list[Item] = []
        seen: set[str] = set()
        for item in items:
            if not ID_PATTERN.fullmatch(item.id):
                raise ValueError(f"item id {item.id!r} does not match {ID_PATTERN.pattern}")
            if not item.resource_class or not item.command or item.attempt < 1:
                raise ValueError(f"item {item.id} needs a resource class, a command and an attempt >= 1")
            if not all(isinstance(a, str) for a in item.command) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in item.env.items()
            ):
                raise ValueError(f"item {item.id}: command and env must be strings")
            if item.id in seen or self._known(item.id):
                raise ValueError(f"item {item.id} is already in the pool")
            digest = item.payload_digest()
            if item.payload_sha256 and item.payload_sha256 != digest:
                raise ValueError(f"item {item.id}: payload_sha256 does not match its payload")
            seen.add(item.id)
            ready.append(
                replace(
                    item,
                    command=tuple(item.command),
                    env=dict(item.env),
                    max_attempts=item.max_attempts or self.max_attempts,
                    payload_sha256=digest,
                )
            )
        for item in ready:
            token = uuid.uuid4().hex
            staged = self._d("tmp") / (_tmp_name("pub", token) + ".json")
            staged.write_text(json.dumps({**item.to_dict(), "placed": token}, indent=1, sort_keys=True) + "\n")
            try:
                if not _place(staged, self._d("queue") / f"{item.id}.json", token):
                    raise ValueError(f"item {item.id} is already in the pool")
            finally:
                _unlink(staged)
        return ready

    # -- claiming ------------------------------------------------------------------------------------------------
    def _class_of(self, item_id: str) -> Optional[str]:
        cls = self._classes.get(item_id)
        if cls is None:
            data = _read_json(self._d("queue") / f"{item_id}.json")
            if not isinstance(data, dict):
                return None
            cls = self._classes[item_id] = str(data["resource_class"])
        return cls

    def _stage(self, worker: str) -> tuple[Path, str, float]:
        """A complete claim directory (owner.json, heartbeat) in ``tmp/``, not yet holding an item."""
        token = uuid.uuid4().hex
        staged = self._d("tmp") / _tmp_name("claim", token)
        staged.mkdir()
        _write_json(
            staged / "owner.json",
            {"token": token, "worker": worker, "host": self.host, "pid": os.getpid(), "claimed": _now()},
        )
        (staged / "heartbeat").touch()
        return staged, token, time.monotonic()

    def _conflict(self, held_item: Path, item_id: str, reason: str, token: str) -> None:
        """Move an item that must not run (a duplicate) to ``conflicts/``."""
        try:
            os.rename(held_item, self._d("conflicts") / f"{item_id}.{reason}.{token[:12]}.json")
        except OSError:
            pass  # left where it is: reap() settles it, and _settle drops a superseded copy

    def _try_claim(self, staged: Path, token: str, item_id: str, worker: str) -> tuple[Optional[Lease], bool]:
        """Take one item into the staged dir, then publish the lease. Returns (lease, staged dir still usable)."""
        source, held_item = self._d("queue") / f"{item_id}.json", staged / "item.json"
        try:
            os.rename(source, held_item)
        except OSError:
            pass  # usually another worker took it; a retransmitted rename that landed is caught by the read-back
        data = _read_json(held_item)
        if not isinstance(data, dict):
            return None, _exists(staged)
        item = Item.from_dict(data)
        for reason in ("done", "dead"):
            if _exists(self._d(reason) / f"{item.id}.json"):
                self._conflict(held_item, item.id, reason, token)
                return None, not _exists(held_item)
        try:
            _write_json(
                staged / "owner.json",
                {
                    "token": token,
                    "worker": worker,
                    "host": self.host,
                    "pid": os.getpid(),
                    "claimed": _now(),
                    "item": item.id,
                    "attempt": item.attempt,
                },
            )
            (staged / "attempt").write_text(f"{item.attempt}\n")
            os.utime(staged / "heartbeat")
        except OSError:
            return None, False  # the staged dir was reaped under a stalled claim; the reaper requeues the item
        target = self._d("leases") / item.id
        error: Optional[OSError] = None
        try:
            os.rename(staged, target)
        except OSError as exc:
            error = exc
        held = _read_json(target / "owner.json")
        if isinstance(held, dict) and held.get("token") == token:
            return Lease(item=item, token=token, worker=worker, path=target), False
        if error is not None and error.errno in (errno.ENOTEMPTY, errno.EEXIST):
            self._conflict(held_item, item.id, "leased", token)  # a lease of this id exists: a duplicate
            return None, not _exists(held_item)
        try:
            os.rename(held_item, source)  # give the item back
        except OSError:
            pass  # the staged dir was reaped (or the rename failed): reap() returns the item after lease_s
        return None, False

    def claim(self, resource_classes: Iterable[str], worker: str) -> Optional[Lease]:
        """Reap, then claim one queued item of the given classes.

        An item that already has a ``done/`` or ``dead/`` record, or whose lease dir already exists, is a duplicate:
        it goes to ``conflicts/`` instead of running.

        Args:
            resource_classes: The classes this worker can run.
            worker: The worker's name, recorded in the lease.

        Returns:
            The lease, or None when no item of those classes could be claimed.
        """
        classes = set(resource_classes)
        self.reap()
        staged: Optional[Path] = None
        token, made = "", 0.0
        try:
            for name in sorted(_names(self._d("queue"))):
                if not name.endswith(".json"):
                    continue
                item_id = name[: -len(".json")]
                if self._class_of(item_id) not in classes:
                    continue
                if staged is not None and time.monotonic() - made > self.lease_s / 2:
                    self._drop_staged(staged)
                    staged = None
                if staged is None:
                    staged, token, made = self._stage(worker)
                lease, usable = self._try_claim(staged, token, item_id, worker)
                if lease is not None:
                    staged = None
                    return lease
                if not usable:
                    self._drop_staged(staged)
                    staged = None
            return None
        finally:
            if staged is not None:
                self._drop_staged(staged)

    def _drop_staged(self, staged: Path) -> None:
        """Remove a staged claim dir unless it holds an item, which reap() then returns to the queue."""
        if not _exists(staged / "item.json"):
            shutil.rmtree(staged, ignore_errors=True)

    def _still_mine(self, lease: Lease) -> bool:
        """Whether the lease is the caller's: a token mismatch says no at once.

        Absence is re-checked once after :data:`PUTBACK_WAIT_S`, because a reaper that renamed a live lease aside
        puts it back.
        """
        for check in range(2):
            held = _read_json(lease.path / "owner.json")
            if isinstance(held, dict):
                return held.get("token") == lease.token
            if check == 0:
                time.sleep(PUTBACK_WAIT_S)
        return False

    def heartbeat(self, lease: Lease) -> bool:
        """Touch the lease's heartbeat if the lease is still this caller's.

        Args:
            lease: The lease.

        Returns:
            False when the lease was lost: another token holds it, or it was absent on two checks.
        """
        for check in range(2):
            if not self._still_mine(lease):
                return False
            try:
                os.utime(lease.path / "heartbeat")
                return True
            except OSError as exc:
                if not _gone(exc):
                    raise
        return False

    # -- finishing -----------------------------------------------------------------------------------------------
    def _take_lease(self, lease: Lease, kind: str) -> Optional[Path]:
        """Rename the caller's lease aside into ``tmp/``; None when it is no longer the caller's."""
        for check in range(2):
            if not self._still_mine(lease):
                return None
            aside = self._d("tmp") / _tmp_name(kind, lease.token)
            try:
                os.rename(lease.path, aside)
            except OSError:
                pass  # read back below
            held = _read_json(aside / "owner.json")
            if not isinstance(held, dict):
                continue  # gone between the check and the rename (a reaper's put-back window): check again
            if held.get("token") != lease.token:  # reaped and claimed again between the check and the rename
                try:
                    os.rename(aside, lease.path)
                except OSError:
                    pass  # left in tmp/: reap() settles it after lease_s
                return None
            _mark_moved(aside)
            return aside
        return None

    def _superseded(self, item: Item, aside: Path) -> bool:
        """Whether a newer copy of the item exists outside this aside dir (so this one is a leftover).

        Checked in the order an item moves (queue, tmp, leases, dead), so a copy moving on is still seen.
        """
        if _exists(self._d("queue") / f"{item.id}.json"):
            return True
        for path, data in self._tmp_items():
            if path != aside and data.get("id") == item.id and int(data.get("attempt", 0)) > item.attempt:
                return True
        return _exists(self._d("leases") / item.id / "item.json") or _exists(self._d("dead") / f"{item.id}.json")

    def _settle(self, aside: Path, kind: str, record: Mapping[str, Any]) -> str:
        """Record an interrupted or failed attempt, then requeue the item with attempt+1 or bury it in dead/.

        The next state is written to a new file in the aside dir and linked into place last, so a crash at any
        point leaves ``item.json`` unchanged for the next reaper, which re-derives the same attempt number.

        Returns ``requeued``, ``dead``, ``done`` (it had completed; dropped), ``superseded`` (a leftover copy;
        dropped), or ``lost`` (another process took the aside dir).
        """
        data = _read_json(aside / "item.json")
        if not isinstance(data, dict):
            _remove_dir(aside)
            return "lost"
        item = Item.from_dict(data)
        if _exists(self._d("done") / f"{item.id}.json"):
            _remove_dir(aside)
            return "done"
        if self._superseded(item, aside):
            _remove_dir(aside)
            return "superseded"
        directory = self._d(kind)
        token = uuid.uuid4().hex
        try:
            _write_json(
                directory / f"{item.id}.{item.attempt}.json",
                {
                    **record,
                    "id": item.id,
                    "attempt": item.attempt,
                    "resource_class": item.resource_class,
                    "host": self.host,
                    "time": _now(),
                },
            )
            count = sum(_exists(directory / f"{item.id}.{a}.json") for a in range(1, item.attempt + 1))
            limit = (item.max_attempts or self.max_attempts) if kind == "failed" else self.max_interrupts
            if count >= limit:
                reason = "failures" if kind == "failed" else "interrupts"
                following = {**item.to_dict(), "dead_reason": reason, "dead_time": _now()}
                outcome, target = "dead", self._d("dead") / f"{item.id}.json"
            else:
                following = replace(item, attempt=item.attempt + 1).to_dict()
                outcome, target = "requeued", self._d("queue") / f"{item.id}.json"
            _write_json(aside / NEXT, {**following, "placed": token})
        except OSError:
            return "lost"
        if not _place(aside / NEXT, target, token):
            return "lost"
        _remove_dir(aside)
        return outcome

    def complete(self, lease: Lease, result: Mapping[str, Any]) -> bool:
        """Mark the lease's item done: ``done/<item>.json`` holds ``result`` plus the item, attempt, worker and host.

        Args:
            lease: The caller's lease.
            result: What to record, e.g. ``exit_code``, ``start``, ``end``.

        Returns:
            False, writing nothing, when the lease is no longer the caller's (so a second call is a no-op).
        """
        aside = self._take_lease(lease, "done")
        if aside is None:
            return False
        item = lease.item
        record = {
            **result,
            "id": item.id,
            "attempt": item.attempt,
            "resource_class": item.resource_class,
            "worker": lease.worker,
            "host": self.host,
            "token": lease.token,
            "completed": _now(),
            "placed": lease.token,
        }
        try:
            _write_json(aside / "done.json", record)
        except OSError:
            return False
        if not _place(aside / "done.json", self._d("done") / f"{item.id}.json", lease.token):
            return False
        _remove_dir(aside)
        return True

    def fail(self, lease: Lease, exit_code: int) -> str:
        """Record a failed attempt; requeue the item with attempt+1, or send it to ``dead/`` once its budget is spent.

        Args:
            lease: The caller's lease.
            exit_code: The command's exit code.

        Returns:
            ``retry``, ``dead``, or ``lost`` (the lease is no longer the caller's; nothing written).
        """
        aside = self._take_lease(lease, "fail")
        if aside is None:
            return "lost"
        outcome = self._settle(aside, "failed", {"exit_code": int(exit_code), "worker": lease.worker})
        return {"requeued": "retry", "dead": "dead"}.get(outcome, "lost")

    def release(self, lease: Lease) -> str:
        """Give the item back without spending a retry; its attempt still goes up and counts toward max_interrupts.

        Args:
            lease: The caller's lease.

        Returns:
            ``requeued``, ``dead`` (``max_interrupts`` reached), or ``lost`` (nothing written).
        """
        aside = self._take_lease(lease, "release")
        if aside is None:
            return "lost"
        outcome = self._settle(aside, "interrupted", {"reason": "release", "worker": lease.worker})
        return outcome if outcome in ("requeued", "dead") else "lost"

    # -- reaping -------------------------------------------------------------------------------------------------
    def _beat(self, lease_dir: Path) -> Optional[float]:
        for name in ("heartbeat", "owner.json", "item.json"):
            mtime = _fresh_mtime(lease_dir / name)
            if mtime is not None:
                return mtime
        return None

    def _settle_reaped(self, aside: Path) -> int:
        _mark_moved(aside)
        owner = _read_json(aside / "owner.json")
        outcome = self._settle(aside, "interrupted", {"reason": "reap", "previous_owner": owner})
        return 1 if outcome in ("requeued", "dead") else 0

    def _reap_lease(self, item_id: str, now: float) -> int:
        """Reap one lease whose heartbeat is older than lease_s; only one reaper wins the rename aside.

        The owner's token is read before the rename and the heartbeat is checked again after it; a lease that
        turns out to be another or a fresh one is put back.
        """
        lease_dir = self._d("leases") / item_id
        held = _read_json(lease_dir / "owner.json")
        token = held.get("token") if isinstance(held, dict) else None
        beat = self._beat(lease_dir)
        if token is None or beat is None or now - beat <= self.lease_s:
            return 0
        aside = self._d("tmp") / _tmp_name("reap", uuid.uuid4().hex)
        try:
            os.rename(lease_dir, aside)
        except OSError:
            pass  # read back below
        if not _exists(aside):
            return 0
        moved = _read_json(aside / "owner.json")
        beat = self._beat(aside)
        mine = isinstance(moved, dict) and moved.get("token") == token
        if not mine or beat is None or self._server_now() - beat <= self.lease_s:
            try:
                os.rename(aside, lease_dir)  # another lease took the name, or this one is fresh: put it back
                return 0
            except OSError:
                pass
        return self._settle_reaped(aside)

    def reap(self) -> int:
        """Re-publish expired leases and abandoned ``tmp/`` entries (older than lease_s) with attempt+1.

        Ages are measured on the storage's clock (heartbeat mtimes against a just-touched file), so a node whose
        clock is off neither reaps live leases nor keeps dead ones. Each reap counts as an interruption, so an
        item reaped ``max_interrupts`` times goes to ``dead/`` instead.

        Returns:
            How many items this call re-published or sent to ``dead/``.
        """
        now = self._server_now()
        count = sum(self._reap_lease(name, now) for name in _names(self._d("leases")))
        for name in _names(self._d("tmp")):
            path = self._d("tmp") / name
            changed = self._entry_time(path)
            if changed is None or now - changed <= self.lease_s:
                continue
            if name.endswith(".json"):  # a publish file whose publisher stalled or died
                _unlink(path)
                continue
            aside = self._d("tmp") / _tmp_name("reap", uuid.uuid4().hex)
            try:
                os.rename(path, aside)
            except OSError:
                pass
            if _exists(aside):
                count += self._settle_reaped(aside)
        return count

    # -- host checks ---------------------------------------------------------------------------------------------
    def note_host_check_failure(self, worker: str, partition: Optional[str], rc: Union[int, str]) -> None:
        """Record that a worker on ``partition`` failed its host check.

        Args:
            worker: The worker's name.
            partition: The worker's partition (``SLURM_JOB_PARTITION``); None outside Slurm.
            rc: The check's exit code, or ``timeout``.
        """
        directory = self._d("hostcheck")
        directory.mkdir(exist_ok=True)
        part = partition or ""
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", part) or "_"
        _write_json(
            directory / f"{safe}.{worker}.{int(_now() * 1000)}.json",
            {"partition": part, "worker": worker, "host": self.host, "rc": rc},
        )

    def host_check_failures(self, window_s: float) -> dict[str, int]:
        """Host-check failures per partition recorded within the last ``window_s`` seconds (storage clock)."""
        now = self._server_now()
        counts: dict[str, int] = {}
        for name in _names(self._d("hostcheck")):
            path = self._d("hostcheck") / name
            changed = _fresh_mtime(path)
            record = _read_json(path)
            if changed is None or now - changed > window_s or not isinstance(record, dict):
                continue
            part = str(record.get("partition") or "")
            if part:
                counts[part] = counts.get(part, 0) + 1
        return counts

    # -- counting ------------------------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Counts per state and per class, also written to ``status.json``.

        Returns:
            ``{"schema", "time", "counts": {queued, leased, done, dead, failed_attempts, interrupted, staged,
            conflicts}, "classes": {class: {queued, leased, done, dead}}, "workers"}``.
        """
        per_class: dict[str, dict[str, int]] = {}

        def add(state: str, data: Any) -> None:  # noqa: ANN401
            if isinstance(data, dict) and data.get("resource_class"):
                row = per_class.setdefault(
                    str(data["resource_class"]), {"queued": 0, "leased": 0, "done": 0, "dead": 0}
                )
                row[state] += 1

        queued = [n for n in _names(self._d("queue")) if n.endswith(".json")]
        leased = _names(self._d("leases"))
        done = [n for n in _names(self._d("done")) if n.endswith(".json")]
        dead = [n for n in _names(self._d("dead")) if n.endswith(".json")]
        for name in queued:
            add("queued", _read_json(self._d("queue") / name))
        for name in leased:
            add("leased", _read_json(self._d("leases") / name / "item.json"))
        for name in done:
            add("done", _read_json(self._d("done") / name))
        for name in dead:
            add("dead", _read_json(self._d("dead") / name))
        staged = sum(_exists(self._d("tmp") / n / "item.json") for n in _names(self._d("tmp")))
        result = {
            "schema": SCHEMA,
            "time": _now(),
            "counts": {
                "queued": len(queued),
                "leased": len(leased),
                "done": len(done),
                "dead": len(dead),
                "failed_attempts": len(_names(self._d("failed"))),
                "interrupted": len(_names(self._d("interrupted"))),
                "staged": staged,
                "conflicts": len(_names(self._d("conflicts"))),
            },
            "classes": per_class,
            "workers": len(_names(self._d("workers"))),
        }
        _write_json(self.root / "status.json", result)
        return result


# -- worker ------------------------------------------------------------------------------------------------------
def _log(message: str) -> None:
    print(f"[work_pool {time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _signal_group(pgid: int, sig: int) -> bool:
    """Signal a process group; False when it no longer exists."""
    try:
        os.killpg(pgid, sig)
    except ProcessLookupError:
        return False
    return True


def _kill_group(pgid: int) -> None:
    """SIGTERM a process group, then SIGKILL whatever is left after :data:`KILL_GRACE_S`; a gone group is fine."""
    if not _signal_group(pgid, signal.SIGTERM):
        return
    deadline = time.monotonic() + KILL_GRACE_S
    while time.monotonic() < deadline:
        if not _signal_group(pgid, 0):
            return
        time.sleep(0.1)
    _signal_group(pgid, signal.SIGKILL)


class _Run:
    """One item's run: its lease, child, and the events its heartbeat thread acts on."""

    def __init__(self, lease: Lease) -> None:
        self.lease = lease
        self.child: Optional[subprocess.Popen[bytes]] = None
        self.lost = threading.Event()
        self.stop = threading.Event()

    def kill(self) -> None:
        """Kill this run's child's whole process group (the child runs in its own session)."""
        if self.child is not None:
            _kill_group(self.child.pid)


class _Worker:
    """Claim, run, heartbeat, finish, repeat."""

    def __init__(
        self,
        pool: WorkPool,
        classes: Sequence[str],
        worker: str,
        linger: int,
        poll_s: float,
        host_check: Optional[Sequence[str]] = None,
        host_check_timeout_s: float = HOST_CHECK_TIMEOUT_S,
    ) -> None:
        self.pool, self.classes, self.worker = pool, list(classes), worker
        self.linger, self.poll_s = linger, poll_s
        self.host_check = list(host_check) if host_check else None
        self.host_check_timeout_s = host_check_timeout_s
        self.run: Optional[_Run] = None
        self.terminating = threading.Event()

    def _check_host(self) -> Union[int, str]:
        """Run the host check in its own session; 0, its exit code, ``timeout`` or 127 (could not start)."""
        try:
            child = subprocess.Popen(self.host_check or [], start_new_session=True)  # noqa: S603
        except OSError:
            return 127
        try:
            return child.wait(timeout=self.host_check_timeout_s)
        except subprocess.TimeoutExpired:
            _signal_group(child.pid, signal.SIGKILL)
            child.wait()
            return "timeout"

    def _on_term(self, signum: int, _frame: Any) -> None:  # noqa: ANN401
        _log(f"signal {signum}: stopping the item in hand and releasing it")
        self.terminating.set()
        run = self.run
        if run is not None:
            threading.Thread(target=run.kill, daemon=True).start()

    def _heartbeats(self, run: _Run) -> None:
        while not run.stop.wait(self.pool.heartbeat_s):
            try:
                mine = self.pool.heartbeat(run.lease)
            except OSError:
                continue
            if run.stop.is_set():
                return
            if not mine:
                item = run.lease.item
                _log(f"lease {item.id} attempt {item.attempt} was lost; stopping its child")
                run.lost.set()
                run.kill()
                return

    def _run(self, lease: Lease) -> None:
        item = lease.item
        if self.terminating.is_set():
            self.pool.release(lease)
            return
        env = {**os.environ, **item.env, "POOL_ITEM_ID": item.id, "POOL_ATTEMPT": str(item.attempt)}
        run = _Run(lease)
        self.run = run
        start = _now()
        _log(f"item {item.id} attempt {item.attempt}: running")
        try:
            child = subprocess.Popen(list(item.command), env=env, start_new_session=True)  # noqa: S603
        except OSError as exc:
            _log(f"item {item.id}: could not start its command ({exc.strerror})")
            rc = 127
        else:
            run.child = child
            beats = threading.Thread(target=self._heartbeats, args=(run,), daemon=True)
            beats.start()
            if self.terminating.is_set():
                threading.Thread(target=run.kill, daemon=True).start()
            rc = child.wait()
            run.kill()  # whatever the child left running in its group
            if (rc < 0 or rc in (137, 143)) and not run.lost.is_set():
                self.terminating.wait(SIGNAL_GRACE_S)  # a job-wide SIGTERM can reach the child first
            run.stop.set()
            beats.join(timeout=self.pool.heartbeat_s + PUTBACK_WAIT_S + JOIN_GRACE_S)
        self.run = None
        end = _now()
        if run.lost.is_set():
            _log(f"item {item.id} attempt {item.attempt}: lease lost (rc {rc}); nothing written")
        elif self.terminating.is_set():
            _log(f"item {item.id} attempt {item.attempt}: released ({self.pool.release(lease)})")
        elif rc in self.pool.success_codes:
            ok = self.pool.complete(
                lease, {"exit_code": rc, "start": start, "end": end, "seconds": round(end - start, 3)}
            )
            _log(f"item {item.id} attempt {item.attempt}: {'done' if ok else 'lease lost before done'}")
        else:
            _log(f"item {item.id} attempt {item.attempt}: failed rc {rc} ({self.pool.fail(lease, rc)})")

    def _record(self, **fields: Any) -> None:  # noqa: ANN401
        path = self.pool.root / "workers" / f"{self.worker}.json"
        base = _read_json(path) or {}
        _write_json(path, {**base, **fields})

    def loop(self) -> int:
        signal.signal(signal.SIGTERM, self._on_term)
        signal.signal(signal.SIGINT, self._on_term)
        env = os.environ
        self._record(
            worker=self.worker,
            host=self.pool.host,
            pid=os.getpid(),
            classes=self.classes,
            job=env.get("SLURM_JOB_ID"),
            partition=env.get("SLURM_JOB_PARTITION"),
            started=_now(),
        )
        if self.host_check is not None:
            rc = self._check_host()
            if rc != 0:
                self.pool.note_host_check_failure(self.worker, env.get("SLURM_JOB_PARTITION"), rc)
                _log(f"host check failed (rc {rc}) on {self.pool.host}; exiting without claiming")
                self._record(exited=_now(), items=0, terminated=False, host_check=rc)
                return HOST_CHECK_EXIT
        idle, ran = 0, 0
        while not self.terminating.is_set():
            lease = self.pool.claim(self.classes, self.worker)
            if lease is None:
                idle += 1
                if idle > self.linger:
                    break
                self.terminating.wait(self.poll_s)
                continue
            idle = 0
            self._run(lease)
            ran += 1
        self._record(exited=_now(), items=ran, terminated=self.terminating.is_set())
        _log(f"worker {self.worker} exits after {ran} item(s)")
        return SIGNAL_EXIT if self.terminating.is_set() else 0


def work(
    pool: PathLike,
    classes: Sequence[str],
    *,
    worker: Optional[str] = None,
    linger: int = 3,
    poll_s: float = 10.0,
    host_check: Optional[Sequence[str]] = None,
    host_check_timeout_s: float = HOST_CHECK_TIMEOUT_S,
) -> int:
    """Run one worker in this process until the queue stays empty (installs SIGTERM and SIGINT handlers).

    Args:
        pool: The pool directory.
        classes: The resource classes this worker claims.
        worker: The worker's name (default ``<host>-<job id or "local">-<pid>``).
        linger: Further empty polls before the worker exits.
        poll_s: Seconds between polls of an empty queue.
        host_check: A command run once, before the first claim; when it fails or outlasts
            ``host_check_timeout_s``, the worker records it under ``hostcheck/`` for its partition and exits with
            :data:`HOST_CHECK_EXIT` without claiming anything.
        host_check_timeout_s: Seconds the host check may run.

    Returns:
        0, :data:`SIGNAL_EXIT` after SIGTERM or SIGINT, or :data:`HOST_CHECK_EXIT`.
    """
    name = worker or f"{_host()}-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return _Worker(WorkPool(pool), classes, name, linger, poll_s, host_check, host_check_timeout_s).loop()


# -- controller --------------------------------------------------------------------------------------------------
Runner = Callable[[Sequence[str]], str]
"""Runs a command (``sbatch``, ``squeue``, ``scancel``) and returns its stdout."""

STALE_REASONS = ("Priority", "QOS", "ReqNodeNotAvail")
"""Pending reasons (prefixes) that make a partition stale after ``pending_s``; ``Resources`` is not one."""

HELD_REASONS = ("JobHeld", "launch failed requeued held")
"""Pending reasons (prefixes) of held jobs, which never start on their own: stale at once."""

LIVE_STATES = ("PENDING", "RUNNING", "CONFIGURING", "SUSPENDED", "REQUEUED")
"""Worker job states that count as live."""

SQUEUE_FORMAT = "%i|%P|%T|%V|%j|%r"
"""Job id, partition, state, submit time, name and reason; the reason last, as it may hold spaces."""


def run_command(argv: Sequence[str]) -> str:
    """The default runner: run ``argv`` and return its stdout.

    Args:
        argv: The command.

    Returns:
        Its stdout.

    Raises:
        subprocess.CalledProcessError: The command failed.
    """
    return subprocess.run(list(argv), check=True, capture_output=True, text=True).stdout  # noqa: S603


@dataclass(frozen=True)
class PartitionSpec:
    """Where and how a resource class's workers are submitted.

    Attributes:
        partition: The Slurm partition.
        cpus: ``--cpus-per-task``.
        mem: ``--mem``.
        time: ``--time``.
        gres: ``--gres``, when set.
        exclude: Nodes for ``--exclude``.
        requeue: Whether to pass ``--requeue``.
        max_workers: Live (pending plus running) workers of the class allowed on this partition.
    """

    partition: str
    cpus: int = 1
    mem: str = "4G"
    time: str = "01:00:00"
    gres: Optional[str] = None
    exclude: tuple[str, ...] = ()
    requeue: bool = False
    max_workers: int = 1


@dataclass(frozen=True)
class ResourceClass:
    """A resource class the caller defines, and the partitions that offer it (in order of preference on ties).

    Attributes:
        name: The class name items carry; matches :data:`ID_PATTERN`.
        partitions: Where its workers may run.
    """

    name: str
    partitions: tuple[PartitionSpec, ...]

    def __post_init__(self) -> None:
        """Check the name and partitions."""
        if not ID_PATTERN.fullmatch(self.name) or not self.partitions:
            raise ValueError(f"resource class {self.name!r} needs a plain name and at least one partition")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResourceClass:
        """A class from ``{name, partitions: [{partition, cpus, mem, time, gres, exclude, requeue, max_workers}]}``."""
        parts = []
        for row in data["partitions"]:
            row = dict(row)
            if "exclude" in row:
                row["exclude"] = tuple(str(n) for n in row["exclude"] or ())
            parts.append(PartitionSpec(**row))
        return cls(name=str(data["name"]), partitions=tuple(parts))


def load_classes(path: PathLike) -> list[ResourceClass]:
    """Resource classes from a YAML (or JSON) file holding ``classes: [{name, partitions: [...]}]``.

    Args:
        path: The file.

    Returns:
        The classes.
    """
    import yaml  # a senselab dependency; imported here so the pool itself stays standard library only

    data = yaml.safe_load(Path(path).read_text())
    return [ResourceClass.from_dict(row) for row in data["classes"]]


def pool_tag(pool: WorkPool) -> str:
    """Eight hex characters naming the pool in its workers' job names (``work-pool-<tag>-<class>``)."""
    return hashlib.sha256(str(pool.root.resolve()).encode()).hexdigest()[:8]


def _submit_time(text: str) -> Optional[float]:
    try:
        return datetime.datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


@dataclass
class _Job:
    id: str
    partition: str
    state: str
    submitted: Optional[float]
    cls: str
    reason: str


class Controller:
    """Keeps Slurm workers on every partition that offers a class, in step with the pool's queue."""

    def __init__(
        self,
        pool: WorkPool,
        classes: Sequence[ResourceClass],
        *,
        runner: Runner = run_command,
        worker_command: Optional[Sequence[str]] = None,
        poll_s: float = 30.0,
        pending_s: float = 600.0,
        release_cap: int = 3,
        release_window_s: float = 3600.0,
        log_dir: Optional[PathLike] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Set up a controller; nothing is submitted until :meth:`tick`.

        Args:
            pool: The pool.
            classes: The resource classes and their partitions.
            runner: Runs ``sbatch``, ``squeue`` and ``scancel``.
            worker_command: The worker's argv up to ``--classes`` (default: this interpreter running
                ``python -m senselab.utils.work_pool work POOL``).
            poll_s: Seconds between ticks in :meth:`run`.
            pending_s: A partition whose workers have been pending this long for a stale reason gets none.
            release_cap: Host-check releases on a partition within ``release_window_s`` that pause it.
            release_window_s: The release cap's window.
            log_dir: Where workers write ``%x-%j.out`` (default: sbatch's default).
            clock: The local clock, against which squeue's submit times are aged.
        """
        self.pool, self.classes, self.runner = pool, list(classes), runner
        self.worker_command = list(worker_command or [sys.executable, "-m", "senselab.utils.work_pool", "work"])
        if worker_command is None:
            self.worker_command.append(str(pool.root))
        self.poll_s, self.pending_s = poll_s, pending_s
        self.release_cap, self.release_window_s = release_cap, release_window_s
        self.log_dir = Path(log_dir).absolute() if log_dir is not None else None
        self.clock = clock
        self.pool8 = pool_tag(pool)

    def job_name(self, cls: str) -> str:
        """The job name of this pool's workers of class ``cls``."""
        return f"work-pool-{self.pool8}-{cls}"

    def _jobs(self) -> list[_Job]:
        names = {self.job_name(c.name): c.name for c in self.classes}
        out = self.runner(["squeue", "--me", "-h", f"--name={','.join(names)}", "-o", SQUEUE_FORMAT])
        jobs = []
        for line in out.splitlines():
            fields = line.strip().split("|", 5)
            if len(fields) < 6 or fields[4] not in names:
                continue
            job_id, partition, state, submitted, name, reason = fields
            jobs.append(_Job(job_id, partition, state, _submit_time(submitted), names[name], reason))
        return jobs

    def _render(self, cls: ResourceClass, spec: PartitionSpec) -> list[str]:
        argv = [
            "sbatch",
            "--parsable",
            f"--partition={spec.partition}",
            f"--job-name={self.job_name(cls.name)}",
            f"--cpus-per-task={spec.cpus}",
            f"--mem={spec.mem}",
            f"--time={spec.time}",
        ]
        if spec.gres:
            argv.append(f"--gres={spec.gres}")
        if spec.exclude:
            argv.append(f"--exclude={','.join(spec.exclude)}")
        if spec.requeue:
            argv.append("--requeue")
        if self.log_dir is not None:
            argv.append(f"--output={self.log_dir}/%x-%j.out")
        # exec: the worker replaces sbatch's wrapper shell, so a cancel or preemption SIGTERM reaches it and it can
        # release its item (a shell dying first ends the step, and Slurm then kills the worker outright).
        argv += ["--wrap", "exec " + shlex.join([*self.worker_command, "--classes", cls.name])]
        return argv

    def _stale(self, job: _Job, now: float) -> bool:
        if job.state != "PENDING":
            return False
        if job.reason.startswith(HELD_REASONS):
            return True
        return (
            job.reason.startswith(STALE_REASONS) and job.submitted is not None and now - job.submitted > self.pending_s
        )

    def _cdir(self, name: str) -> Path:
        path = self.pool.root / "controller" / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _record_job(self, job_id: str, cls: str, partition: str) -> None:
        """Record a job this controller submitted (only recorded jobs are ever cancelled)."""
        _write_json(self._cdir("jobs") / f"{job_id}.json", {"job": job_id, "class": cls, "partition": partition})

    def _recorded(self) -> set[str]:
        return {n[: -len(".json")] for n in _names(self._cdir("jobs")) if n.endswith(".json")}

    def _cancel(self, ids: Sequence[str]) -> list[str]:
        """``scancel --state=PENDING`` the recorded ones of ``ids`` (a worker that started meanwhile is left alone)."""
        recorded = self._recorded()
        mine = [i for i in ids if i in recorded]
        if not mine:
            return []
        try:
            self.runner(["scancel", "--state=PENDING", *mine])
        except (OSError, subprocess.SubprocessError) as exc:
            _log(f"control: scancel failed ({exc}); retrying next tick")
            return []
        return mine

    def _marked_stale(self, partition: str, now: float) -> bool:
        """Whether this controller cancelled stale workers on ``partition`` within ``pending_s`` (storage clock)."""
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", partition)
        changed = _fresh_mtime(self._cdir("stale") / safe)
        return changed is not None and now - changed <= self.pending_s

    def _mark_stale(self, partition: str) -> None:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", partition)
        path = self._cdir("stale") / safe
        os.close(os.open(path, os.O_WRONLY | os.O_CREAT, 0o660))
        os.utime(path)

    def tick(self) -> dict[str, Any]:
        """Reap, count, then submit workers where items exceed live workers and cancel workers nobody needs.

        Pending workers on a stale or paused partition do not count as live: they are cancelled (when this pool's
        controller submitted them) and the partition gets no workers for ``pending_s``, so the work goes elsewhere.
        A failed ``sbatch`` skips its partition for the rest of the tick.

        Returns:
            ``{"reaped", "submitted": [{class, partition, job}], "cancelled": [job ids], "failed": [partitions],
            "partitions": {p: {pending, running, stale, host_check_releases, paused}}, "status"}``; the same
            ``partitions`` go into ``status.json``.

        Raises:
            OSError: The pool could not be read or written.
            subprocess.SubprocessError: ``squeue`` failed.
        """
        reaped = self.pool.reap()
        status = self.pool.status()
        jobs = [j for j in self._jobs() if j.state in LIVE_STATES]
        now = self.clock()
        storage_now = self.pool._server_now()
        releases = self.pool.host_check_failures(self.release_window_s)
        partitions: dict[str, dict[str, Any]] = {}
        for p in dict.fromkeys(spec.partition for c in self.classes for spec in c.partitions):
            mine = [j for j in jobs if j.partition == p]
            partitions[p] = {
                "pending": sum(j.state == "PENDING" for j in mine),
                "running": sum(j.state != "PENDING" for j in mine),
                "stale": any(self._stale(j, now) for j in mine) or self._marked_stale(p, storage_now),
                "host_check_releases": releases.get(p, 0),
                "paused": "host-check" if releases.get(p, 0) >= self.release_cap else None,
            }

        def blocked(p: str) -> bool:
            row = partitions.get(p)
            return row is not None and bool(row["stale"] or row["paused"])

        submitted: list[dict[str, str]] = []
        cancelled: list[str] = []
        failed: list[str] = []
        for cls in self.classes:
            row = status["classes"].get(cls.name, {})
            queued, leased = int(row.get("queued", 0)), int(row.get("leased", 0))
            ours = [j for j in jobs if j.cls == cls.name]
            stuck = [j for j in ours if j.state == "PENDING" and blocked(j.partition)]
            live = [j for j in ours if j not in stuck]
            if queued == 0:
                cancelled += self._cancel([j.id for j in ours if j.state == "PENDING"])
                continue
            gone = self._cancel([j.id for j in stuck])
            cancelled += gone
            for p in dict.fromkeys(j.partition for j in stuck if j.id in gone):
                self._mark_stale(p)
            pending = {
                s.partition: [j for j in live if j.partition == s.partition and j.state == "PENDING"]
                for s in cls.partitions
            }
            count = {s.partition: sum(j.partition == s.partition for j in live) for s in cls.partitions}
            npend = {p: len(js) for p, js in pending.items()}
            oldest = {
                p: max((now - j.submitted for j in js if j.submitted is not None), default=0.0)
                for p, js in pending.items()
            }
            usable = [(k, s) for k, s in enumerate(cls.partitions) if not blocked(s.partition)]
            wanted = queued + leased - len(live)
            while wanted > 0:
                room = [(k, s) for k, s in usable if count[s.partition] < s.max_workers and s.partition not in failed]
                if not room:
                    break
                _, spec = min(room, key=lambda ks: (npend[ks[1].partition], oldest[ks[1].partition], ks[0]))
                try:
                    job_id = self.runner(self._render(cls, spec)).strip().split(";")[0]
                except (OSError, subprocess.SubprocessError) as exc:
                    _log(f"control: sbatch on {spec.partition} failed ({exc}); skipping it this tick")
                    failed.append(spec.partition)
                    continue
                self._record_job(job_id, cls.name, spec.partition)
                wanted -= 1
                submitted.append({"class": cls.name, "partition": spec.partition, "job": job_id})
                count[spec.partition] += 1
                npend[spec.partition] += 1
                partitions[spec.partition]["pending"] += 1
        _write_json(self.pool.root / "status.json", {**status, "partitions": partitions})
        return {
            "reaped": reaped,
            "submitted": submitted,
            "cancelled": cancelled,
            "failed": failed,
            "partitions": partitions,
            "status": status,
        }

    def run(self, once: bool = False) -> int:
        """Hold the pool's controller lock and tick every ``poll_s`` until nothing is queued, leased or staged.

        A tick that fails (a Slurm command, or the pool's storage) is logged and retried after ``poll_s``.

        Args:
            once: Tick once and return.

        Returns:
            0; 1 when ``once`` and the tick failed; :data:`CONTROLLER_LOCKED_EXIT` when another controller runs.
        """
        lock = open(self._cdir("") / "lock", "a")  # noqa: SIM115 — held for the whole run
        try:
            try:
                fcntl.lockf(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                _log(f"control: another controller holds {self.pool.root}/controller/lock; exiting")
                return CONTROLLER_LOCKED_EXIT
            while True:
                try:
                    summary = self.tick()
                except (OSError, subprocess.SubprocessError) as exc:
                    _log(f"control: tick failed ({exc!r}); retrying in {self.poll_s} s")
                    if once:
                        return 1
                    time.sleep(self.poll_s)
                    continue
                counts = summary["status"]["counts"]
                _log(
                    f"control: reaped {summary['reaped']}, submitted {len(summary['submitted'])}, cancelled "
                    f"{len(summary['cancelled'])}; queued {counts['queued']} leased {counts['leased']} done "
                    f"{counts['done']} dead {counts['dead']}"
                )
                if once or counts["queued"] + counts["leased"] + counts["staged"] == 0:
                    return 0
                time.sleep(self.poll_s)
        finally:
            lock.close()


def submit_controller(
    pool: WorkPool,
    classes_file: PathLike,
    *,
    partition: str,
    time: str,
    runner: Runner = run_command,
    poll_s: float = 30.0,
    cpus: int = 1,
    mem: str = "2G",
    output: Optional[PathLike] = None,
    extra_args: Sequence[str] = (),
) -> str:
    """Submit a small job that runs ``control POOL --classes FILE`` until the pool drains.

    Args:
        pool: The pool.
        classes_file: The resource classes' YAML file.
        partition: The controller job's partition.
        time: Its ``--time``.
        runner: Runs ``sbatch``.
        poll_s: Seconds between ticks.
        cpus: Its ``--cpus-per-task``.
        mem: Its ``--mem``.
        output: Its ``--output``, when set.
        extra_args: More ``control`` arguments (e.g. ``--host-check``).

    Returns:
        The controller's job id.
    """
    command = [sys.executable, "-m", "senselab.utils.work_pool", "control", str(pool.root)]
    command += ["--classes", str(Path(classes_file).absolute()), "--poll", str(float(poll_s)), *extra_args]
    argv = [
        "sbatch",
        "--parsable",
        f"--partition={partition}",
        f"--job-name=work-pool-control-{pool_tag(pool)}",
        f"--cpus-per-task={cpus}",
        f"--mem={mem}",
        f"--time={time}",
    ]
    if output is not None:
        argv.append(f"--output={output}")
    argv += ["--wrap", "exec " + shlex.join(command)]
    return runner(argv).strip().split(";")[0]


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line: ``work`` runs one worker, ``control`` runs a controller.

    ``work POOL --classes a,b [--linger N] [--poll S] [--worker NAME] [--host-check CMD] [--host-check-timeout S]``;
    ``control POOL --classes FILE [--once] [--poll S] [--pending-s S] [--release-cap N] [--log-dir D]
    [--host-check CMD] [--host-check-timeout S] [--worker-linger N] [--worker-poll S]``.

    Args:
        argv: Arguments (default ``sys.argv[1:]``).

    Returns:
        The exit code.
    """
    parser = argparse.ArgumentParser(prog="python -m senselab.utils.work_pool")
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("work", help="run one worker")
    run.add_argument("pool")
    run.add_argument("--classes", required=True, help="comma-separated resource classes")
    run.add_argument("--linger", type=int, default=3)
    run.add_argument("--poll", type=float, default=10.0)
    run.add_argument("--worker")
    run.add_argument("--host-check", help="a command (split like a shell) that must succeed on this node")
    run.add_argument("--host-check-timeout", type=float, default=HOST_CHECK_TIMEOUT_S)
    ctl = sub.add_parser("control", help="run a controller that submits and cancels Slurm workers")
    ctl.add_argument("pool")
    ctl.add_argument("--classes", required=True, help="the resource classes' YAML file")
    ctl.add_argument("--once", action="store_true", help="tick once and exit")
    ctl.add_argument("--poll", type=float, default=30.0)
    ctl.add_argument("--pending-s", type=float, default=600.0)
    ctl.add_argument("--release-cap", type=int, default=3)
    ctl.add_argument("--log-dir")
    ctl.add_argument("--host-check", help="passed on to each worker")
    ctl.add_argument("--host-check-timeout", type=float, help="passed on to each worker")
    ctl.add_argument("--worker-linger", type=int, help="passed on to each worker as --linger")
    ctl.add_argument("--worker-poll", type=float, help="passed on to each worker as --poll")
    args = parser.parse_args(argv)
    if args.cmd == "control":
        pool = WorkPool(args.pool)
        command = [sys.executable, "-m", "senselab.utils.work_pool", "work", str(pool.root)]
        if args.host_check:
            command += ["--host-check", args.host_check]
        if args.host_check_timeout is not None:
            command += ["--host-check-timeout", str(args.host_check_timeout)]
        if args.worker_linger is not None:
            command += ["--linger", str(args.worker_linger)]
        if args.worker_poll is not None:
            command += ["--poll", str(args.worker_poll)]
        controller = Controller(
            pool,
            load_classes(args.classes),
            worker_command=command,
            poll_s=args.poll,
            pending_s=args.pending_s,
            release_cap=args.release_cap,
            log_dir=args.log_dir,
        )
        return controller.run(once=args.once)
    classes = [c for c in args.classes.split(",") if c]
    check = shlex.split(args.host_check) if args.host_check else None
    return work(
        args.pool,
        classes,
        worker=args.worker,
        linger=args.linger,
        poll_s=args.poll,
        host_check=check,
        host_check_timeout_s=args.host_check_timeout,
    )


if __name__ == "__main__":
    sys.exit(main())
