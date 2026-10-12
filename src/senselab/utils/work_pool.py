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
    workers/<worker>.json        the workers that ran here
    status.json                  written by :meth:`WorkPool.status`

Every move of an item is one ``rename(2)``, so an item is always in exactly one place, and every lease
directory is staged complete in ``tmp/`` before it is renamed into ``leases/``. :meth:`WorkPool.claim` takes an
item with ``rename(queue/<i>.json, tmp/claim-<token>/item.json)`` and publishes the lease with
``rename(tmp/claim-<token>, leases/<i>)``; :meth:`WorkPool.complete`, :meth:`WorkPool.fail`,
:meth:`WorkPool.release` and :meth:`WorkPool.reap` all start by renaming ``leases/<i>`` aside, so only one of
them acts on a lease. After a rename reports an error, the destination is read back, because a retransmitted
NFS rename can report failure to the process that made it.

``python -m senselab.utils.work_pool work POOL --classes a,b`` runs a worker: claim, run the item's command with
``POOL_ITEM_ID`` and ``POOL_ATTEMPT`` set, heartbeat, then complete or fail; SIGTERM releases the item.

Ported and generalised from ``sensein/senselab@feat/task-events`` (f2a806af),
``specs/20261011-slice-pool/pool/slice_pool.py``. Standard library only.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import re
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
from typing import Any, Iterable, Mapping, Optional, Sequence, Union

SCHEMA = "work-pool-1"
"""The ``schema`` value of ``pool.json``."""

SUBDIRS = ("queue", "leases", "tmp", "interrupted", "done", "failed", "dead", "workers")
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


def _tmp_name(kind: str, token: str) -> str:
    """A ``tmp/`` entry name: its kind, its creation time in milliseconds and a token."""
    return f"{kind}-{int(_now() * 1000)}-{token}"


def _tmp_age(name: str) -> Optional[float]:
    """Seconds since a ``tmp/`` entry was named; None for a name :func:`_tmp_name` did not make."""
    parts = name.split("-")
    if len(parts) < 3 or not parts[1].isdigit():
        return None
    return _now() - int(parts[1]) / 1000


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

    # -- publishing ----------------------------------------------------------------------------------------------
    def _known(self, item_id: str) -> bool:
        return any(
            _exists(p)
            for p in (
                self._d("queue") / f"{item_id}.json",
                self._d("leases") / item_id,
                self._d("done") / f"{item_id}.json",
                self._d("dead") / f"{item_id}.json",
            )
        )

    def publish(self, items: Iterable[Item]) -> list[Item]:
        """Put items in the queue, each with one atomic rename from ``tmp/``.

        Every item is checked before any is published.

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
            staged = self._d("tmp") / (_tmp_name("pub", uuid.uuid4().hex) + ".json")
            staged.write_text(json.dumps(item.to_dict(), indent=1, sort_keys=True) + "\n")
            os.rename(staged, self._d("queue") / f"{item.id}.json")
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
        return staged, token, _now()

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
        for retry in (False, True):
            error: Optional[OSError] = None
            try:
                os.rename(staged, target)
            except OSError as exc:
                error = exc
            held = _read_json(target / "owner.json")
            if isinstance(held, dict) and held.get("token") == token:
                return Lease(item=item, token=token, worker=worker, path=target), False
            if error is None or error.errno not in (errno.ENOTEMPTY, errno.EEXIST) or retry:
                break
            self._reap_lease(item.id)  # a stale lease dir holds the name
        try:
            os.rename(held_item, source)  # give the item back
        except OSError:
            pass  # the staged dir was reaped (or the rename failed): reap() returns the item after lease_s
        return None, False

    def claim(self, resource_classes: Iterable[str], worker: str) -> Optional[Lease]:
        """Reap, then claim one queued item of the given classes.

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
                if staged is not None and _now() - made > self.lease_s / 2:
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

    def _owns(self, lease: Lease) -> bool:
        held = _read_json(lease.path / "owner.json")
        return isinstance(held, dict) and held.get("token") == lease.token

    def heartbeat(self, lease: Lease) -> bool:
        """Touch the lease's heartbeat if the lease is still this caller's.

        Args:
            lease: The lease.

        Returns:
            False when the lease was lost (reaped, or finished).
        """
        if not self._owns(lease):
            return False
        try:
            os.utime(lease.path / "heartbeat")
        except OSError as exc:
            if _gone(exc):
                return False
            raise
        return True

    # -- finishing -----------------------------------------------------------------------------------------------
    def _take_lease(self, lease: Lease, kind: str) -> Optional[Path]:
        """Rename the caller's lease aside into ``tmp/``; None when it is no longer the caller's."""
        if not self._owns(lease):
            return None
        aside = self._d("tmp") / _tmp_name(kind, lease.token)
        try:
            os.rename(lease.path, aside)
        except OSError:
            pass  # read back below
        held = _read_json(aside / "owner.json")
        if not isinstance(held, dict):
            return None
        if held.get("token") != lease.token:  # the lease was reaped and claimed again between check and rename
            try:
                os.rename(aside, lease.path)
            except OSError:
                pass  # left in tmp/: reap() returns its item after lease_s
            return None
        return aside

    def _move_item(self, aside: Path, target: Path) -> bool:
        """Move the item file out of an aside dir; False when a reaper took the dir first."""
        try:
            os.rename(aside / "item.json", target)
            return True
        except OSError:
            return _exists(aside) and not _exists(aside / "item.json") and _exists(target)

    def _settle(self, aside: Path, kind: str, record: Mapping[str, Any]) -> str:
        """Record an interrupted or failed attempt, then requeue the item with attempt+1 or bury it in dead/.

        Returns ``requeued``, ``dead``, ``done`` (it had completed; dropped), or ``lost`` (another process took
        the aside dir).
        """
        data = _read_json(aside / "item.json")
        if not isinstance(data, dict):
            shutil.rmtree(aside, ignore_errors=True)
            return "lost"
        item = Item.from_dict(data)
        if _exists(self._d("done") / f"{item.id}.json"):
            shutil.rmtree(aside, ignore_errors=True)
            return "done"
        directory = self._d(kind)
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
                _write_json(aside / "item.json", {**item.to_dict(), "dead_reason": reason, "dead_time": _now()})
                outcome, target = "dead", self._d("dead") / f"{item.id}.json"
            else:
                _write_json(aside / "item.json", replace(item, attempt=item.attempt + 1).to_dict())
                outcome, target = "requeued", self._d("queue") / f"{item.id}.json"
        except OSError:
            return "lost"
        if not self._move_item(aside, target):
            return "lost"
        shutil.rmtree(aside, ignore_errors=True)
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
        }
        target = self._d("done") / f"{item.id}.json"
        try:
            _write_json(aside / "done.json", record)
            os.rename(aside / "done.json", target)
        except OSError:
            held = _read_json(target)
            if not (isinstance(held, dict) and held.get("token") == lease.token):
                return False
        shutil.rmtree(aside, ignore_errors=True)
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
        owner = _read_json(aside / "owner.json")
        outcome = self._settle(aside, "interrupted", {"reason": "reap", "previous_owner": owner})
        return 1 if outcome in ("requeued", "dead") else 0

    def _reap_lease(self, item_id: str) -> int:
        """Reap one lease if its heartbeat is older than lease_s; only one reaper wins the rename aside."""
        lease_dir = self._d("leases") / item_id
        beat = self._beat(lease_dir)
        if beat is None or _now() - beat <= self.lease_s:
            return 0
        aside = self._d("tmp") / _tmp_name("reap", uuid.uuid4().hex)
        try:
            os.rename(lease_dir, aside)
        except OSError:
            pass  # read back below
        if not _exists(aside):
            return 0
        beat = self._beat(aside)
        if beat is not None and _now() - beat <= self.lease_s:  # a fresh lease took the name before the rename
            try:
                os.rename(aside, lease_dir)
                return 0
            except OSError:
                pass
        return self._settle_reaped(aside)

    def reap(self) -> int:
        """Re-publish expired leases and abandoned ``tmp/`` claims (older than lease_s) with attempt+1.

        Each counts as an interruption, so an item reaped ``max_interrupts`` times goes to ``dead/`` instead.

        Returns:
            How many items this call re-published or sent to ``dead/``.
        """
        count = sum(self._reap_lease(name) for name in _names(self._d("leases")))
        for name in _names(self._d("tmp")):
            age = _tmp_age(name)
            if age is None or age <= self.lease_s:
                continue
            path = self._d("tmp") / name
            if name.startswith("pub-"):
                try:
                    path.unlink()
                except OSError:
                    pass
                continue
            aside = self._d("tmp") / _tmp_name("reap", uuid.uuid4().hex)
            try:
                os.rename(path, aside)
            except OSError:
                pass
            if _exists(aside):
                count += self._settle_reaped(aside)
        return count

    # -- counting ------------------------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Counts per state and per class, also written to ``status.json``.

        Returns:
            ``{"schema", "time", "counts": {queued, leased, done, dead, failed_attempts, interrupted, staged},
            "classes": {class: {queued, leased, done, dead}}, "workers"}``.
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
            },
            "classes": per_class,
            "workers": len(_names(self._d("workers"))),
        }
        _write_json(self.root / "status.json", result)
        return result


# -- worker ------------------------------------------------------------------------------------------------------
def _log(message: str) -> None:
    print(f"[work_pool {time.strftime('%Y-%m-%dT%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


class _Worker:
    """Claim, run, heartbeat, finish, repeat."""

    def __init__(self, pool: WorkPool, classes: Sequence[str], worker: str, linger: int, poll_s: float) -> None:
        self.pool, self.classes, self.worker = pool, list(classes), worker
        self.linger, self.poll_s = linger, poll_s
        self.child: Optional[subprocess.Popen[bytes]] = None
        self.terminating = threading.Event()
        self.lost = threading.Event()

    def _on_term(self, signum: int, _frame: Any) -> None:  # noqa: ANN401
        _log(f"signal {signum}: stopping the item in hand and releasing it")
        self.terminating.set()
        threading.Thread(target=self._kill_child, daemon=True).start()

    def _kill_child(self) -> None:
        child = self.child
        if child is None or child.poll() is not None:
            return
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        deadline = _now() + KILL_GRACE_S
        while child.poll() is None and _now() < deadline:
            time.sleep(0.1)
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def _heartbeats(self, lease: Lease, stop: threading.Event) -> None:
        while not stop.wait(self.pool.heartbeat_s):
            try:
                mine = self.pool.heartbeat(lease)
            except OSError:
                continue
            if not mine:
                _log(f"lease {lease.item.id} attempt {lease.item.attempt} was lost; stopping its child")
                self.lost.set()
                self._kill_child()
                return

    def _run(self, lease: Lease) -> None:
        item = lease.item
        if self.terminating.is_set():
            self.pool.release(lease)
            return
        env = {**os.environ, **item.env, "POOL_ITEM_ID": item.id, "POOL_ATTEMPT": str(item.attempt)}
        self.lost.clear()
        stop = threading.Event()
        start = _now()
        _log(f"item {item.id} attempt {item.attempt}: running")
        try:
            child = subprocess.Popen(list(item.command), env=env, start_new_session=True)  # noqa: S603
        except OSError as exc:
            _log(f"item {item.id}: could not start its command ({exc.strerror})")
            rc = 127
        else:
            self.child = child
            beats = threading.Thread(target=self._heartbeats, args=(lease, stop), daemon=True)
            beats.start()
            if self.terminating.is_set():
                self._kill_child()
            rc = child.wait()
            if (rc < 0 or rc in (137, 143)) and not self.lost.is_set():
                self.terminating.wait(SIGNAL_GRACE_S)  # a job-wide SIGTERM can reach the child first
            stop.set()
            beats.join(timeout=self.pool.heartbeat_s + KILL_GRACE_S + 5)
            self.child = None
        end = _now()
        if self.lost.is_set():
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
) -> int:
    """Run one worker in this process until the queue stays empty (installs SIGTERM and SIGINT handlers).

    Args:
        pool: The pool directory.
        classes: The resource classes this worker claims.
        worker: The worker's name (default ``<host>-<job id or "local">-<pid>``).
        linger: Further empty polls before the worker exits.
        poll_s: Seconds between polls of an empty queue.

    Returns:
        0, or :data:`SIGNAL_EXIT` after SIGTERM or SIGINT.
    """
    name = worker or f"{_host()}-{os.environ.get('SLURM_JOB_ID', 'local')}-{os.getpid()}"
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return _Worker(WorkPool(pool), classes, name, linger, poll_s).loop()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command line: ``work POOL --classes a,b [--linger N] [--poll S] [--worker NAME]``.

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
    args = parser.parse_args(argv)
    classes = [c for c in args.classes.split(",") if c]
    return work(args.pool, classes, worker=args.worker, linger=args.linger, poll_s=args.poll)


if __name__ == "__main__":
    sys.exit(main())
