"""The supervisor of the processes a host launches: its table of them, their orphans, and a keeper outliving the host.

Every process senselab starts goes through `Supervisor.popen`, in a new session, so it leads its own process group.
The supervisor in force (`current`) is the innermost one entered in this process, else a process-wide one that
supervises by process group alone.

An entered supervisor on Linux sets `PR_SET_CHILD_SUBREAPER`, so a descendant orphaned by its parent's exit (a fork
that never execs, a `setsid` escapee) is re-parented to this process and enters the table as an orphan. Orphans are
found in the kernel's list of this process's children. An orphan in a launched group is a member of it; one in any
other group is held for the groups launched when it was adopted and is SIGKILLed once each of them is `released`,
at once when there were none. A child started in its own session by code that bypasses the supervisor is taken for
such an orphan.
A group is live while its leader has not exited or one of its orphans has not (`group_live`). Orphans are reaped
one by one, by pid; a launched process is never adopted or reaped here until its owner has reaped it. Children that
were already running when it was entered, their groups, and children of this process's own group are never adopted.

Elsewhere, and in the process-wide supervisor, a group is supervised by its process group: a `setsid` escapee is not
reached, and liveness is probed by group (`live_members`).

Every launch, and every orphan an entered supervisor adopts, is also recorded with the keeper, one process per
supervisor, in its own session, that reads the records on its stdin. Once this process has died, by any signal, the
keeper SIGKILLs every group and orphan still recorded.
Leaving an entered supervisor SIGKILLs every group and orphan it holds, reaps the orphans and the leaders, stops the
keeper and restores the subreaper flag.
"""

import collections
import contextlib
import ctypes
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import TracebackType
from typing import IO, Any, Self

logger = logging.getLogger(__name__)

PR_SET_CHILD_SUBREAPER = 36
PR_GET_CHILD_SUBREAPER = 37
REAP_POLL_S = 0.05
"""Seconds between the reaper's passes over the orphans."""
EXIT_BOUND_S = 5.0
"""Seconds leaving a supervisor waits for its killed orphans, leaders and keeper to be reaped."""
REAPED_KEPT = 256
"""How many reaped orphans' exit statuses the table keeps, the oldest dropped first."""

KEEPER = r"""
import os, select, signal, subprocess, sys

owner = int(sys.argv[1])
try:
    os.setsid()
except OSError:
    pass
held = {"g": {}, "p": {}}
rest = b""


def on_term(signum, frame):
    pass


def started(pid):
    if os.path.isdir("/proc/self"):
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                data = f.read()
        except OSError:
            return None
        return data[data.rfind(b")") + 2 :].split()[19].decode()
    try:
        out = subprocess.run(["ps", "-o", "lstart=", "-p", str(pid)], capture_output=True, text=True, timeout=5)
    except Exception:
        return "?"
    return " ".join(out.stdout.split()) or None


def apply(data):
    global rest
    *lines, rest = (rest + data).split(b"\n")
    for line in lines:
        sign, kind, (number, _, given) = line[:1], line[1:2].decode(), line[2:].decode().partition(" ")
        number = int(number)
        if sign == b"+":
            held[kind][number] = given or started(number)
        else:
            held[kind].pop(number, None)


def drain():
    os.set_blocking(0, False)
    while True:
        try:
            data = os.read(0, 65536)
        except OSError:
            return
        if not data:
            return
        apply(data)


def kill(kind, number, recorded):
    now = started(number)
    if now is not None and now != recorded:
        return
    if now is None and kind == "p":
        return
    try:
        (os.killpg if kind == "g" else os.kill)(number, signal.SIGKILL)
    except OSError:
        pass


signal.signal(signal.SIGTERM, on_term)
if sys.platform == "linux":
    try:
        import ctypes
        ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    except Exception:
        pass
while os.getppid() == owner:
    if select.select([0], [], [], 0.2)[0]:
        data = os.read(0, 65536)
        if not data:
            break
        apply(data)
drain()
for kind in ("g", "p"):
    for number, recorded in held[kind].items():
        kill(kind, number, recorded)
"""
"""Run as ``python -I -c KEEPER <owner pid>`` with records on stdin: ``+g``/``-g`` a group, ``+p``/``-p`` a pid.

A ``+`` record may carry the start time after the number; otherwise the keeper reads the process's start time
(`/proc/<pid>/stat` field 22, else `ps -o lstart=`). At the end it SIGKILLs a recorded pid only while a process of
that pid has the recorded start time, and a recorded group while its leader has it or no process has the group's
number.
"""

_STAT_STATE = 0
_STAT_PGRP = 2
"""Field indices in `/proc/<pid>/stat` after the parenthesised command name: the state and the process group."""


def live_members(pgid: int) -> bool:
    """Whether group `pgid` has a member that is neither a zombie nor exiting: by `/proc`, else by `killpg(pgid, 0)`."""
    if os.path.isdir("/proc/self/task"):
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/stat", "rb") as f:
                    data = f.read()
            except OSError:
                continue
            fields = data[data.rfind(b")") + 2 :].split()
            if len(fields) > _STAT_PGRP and int(fields[_STAT_PGRP]) == pgid and fields[_STAT_STATE] not in (b"Z", b"X"):
                return True
        return False
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


def own_children() -> set[int]:
    """The pids of this process's children, from the kernel's list of each of its threads' children."""
    found: set[int] = set()
    try:
        tasks = os.listdir("/proc/self/task")
    except OSError:
        return found
    for task in tasks:
        try:
            with open(f"/proc/self/task/{task}/children") as f:
                found.update(int(pid) for pid in f.read().split())
        except OSError:
            continue
    return found


def subreaper_supported() -> bool:
    """Whether this host can make a process a subreaper whose orphans it can list: Linux with a children list."""
    return sys.platform == "linux" and os.path.exists(f"/proc/self/task/{threading.get_native_id()}/children")


def _prctl(option: int, value: object) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, value, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))


def _subreaper_flag() -> int:
    value = ctypes.c_int(0)
    _prctl(PR_GET_CHILD_SUBREAPER, ctypes.byref(value))
    return value.value


def _exited(pid: int) -> bool:
    """Whether child `pid` has exited, without reaping it; a pid that is no longer an unreaped child has (Linux)."""
    if sys.platform != "linux":
        raise NotImplementedError("orphans are adopted on Linux only")
    try:
        return os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is not None
    except ChildProcessError:
        return True


def _reap(pid: int) -> int | None:
    """Reap exited child `pid` and return its exit status, negative for the signal that ended it; else None (Linux)."""
    if sys.platform != "linux":
        raise NotImplementedError("orphans are adopted on Linux only")
    try:
        info = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG)
    except ChildProcessError:
        return None
    if info is None:
        return None
    return info.si_status if info.si_code == os.CLD_EXITED else -info.si_status


def _kill(pid: int, sig: int, *, group: bool = False) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        (os.killpg if group else os.kill)(pid, sig)


@dataclass
class Orphan:
    """A descendant re-parented to this process: its pid, its group, and the launched groups it is held for.

    A member of a launched group has no owners; it is that group's.
    """

    pid: int
    pgid: int
    owners: set[int] = field(default_factory=set)


class Keeper:
    """The keeper process of this process, started on the first record; records are written to its stdin.

    In a process forked from its owner, the next record starts that process's own keeper.
    """

    def __init__(self) -> None:
        """Start no process yet."""
        self.pid: int | None = None
        self._owner = os.getpid()
        self._stdin: IO[bytes] | None = None
        self._broken = False

    def record(self, sign: str, kind: str, number: int, start: str | None = None) -> None:
        """Record (`+`) or drop (`-`) group or pid `number` (`kind` `g` or `p`); a dead keeper is logged once.

        `start` is the recorded start time of the group leader or pid; the keeper reads it when None.
        """
        if os.getpid() != self._owner:
            self._disown()
        if self._broken or (self.pid is None and sign == "-"):
            return
        try:
            stdin = self._stdin if self._stdin is not None else self._start()
            stdin.write(f"{sign}{kind}{number}{'' if start is None else ' ' + start}\n".encode())
            stdin.flush()
        except (OSError, ValueError) as exc:
            self._broken = True
            logger.warning("the keeper is gone; launches are no longer kept: %s", exc)

    def stop(self, timeout: float) -> None:
        """Close the keeper's stdin, so it kills what is still recorded and exits; reap it within `timeout`."""
        if os.getpid() != self._owner:
            self._disown()
        pid, stdin, self.pid, self._stdin = self.pid, self._stdin, None, None
        if stdin is not None:
            with contextlib.suppress(OSError):
                stdin.close()
        if pid is None:
            return
        deadline = time.monotonic() + timeout
        with contextlib.suppress(ChildProcessError):
            while os.waitpid(pid, os.WNOHANG) == (0, 0):
                if time.monotonic() > deadline:
                    _kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                    return
                time.sleep(REAP_POLL_S)

    def _start(self) -> IO[bytes]:
        read, write = os.pipe()
        try:
            self.pid = os.posix_spawn(
                sys.executable,
                [sys.executable, "-I", "-c", KEEPER, str(os.getpid())],
                os.environ,
                file_actions=[
                    (os.POSIX_SPAWN_DUP2, read, 0),
                    (os.POSIX_SPAWN_OPEN, 1, os.devnull, os.O_WRONLY, 0),
                ],
            )
        finally:
            os.close(read)
        self._stdin = os.fdopen(write, "wb")
        return self._stdin

    def _disown(self) -> None:
        """In a forked copy, close this process's copy of the owner's keeper stdin and keep nothing of it."""
        if self._stdin is not None:
            with contextlib.suppress(OSError):
                self._stdin.close()
        self.pid, self._stdin, self._broken, self._owner = None, None, False, os.getpid()


class Supervisor:
    """The table of the processes launched through it and of the orphans it adopted; a context manager.

    Attributes:
        reaped: The latest reaped orphans' pids and exit statuses (a negative status is the signal that ended it).
    """

    def __init__(
        self,
        *,
        subreaper: bool | None = None,
        children: Callable[[], set[int]] = own_children,
        poll_s: float = REAP_POLL_S,
    ) -> None:
        """Supervise; once entered, as a subreaper where `subreaper` (by default where `subreaper_supported`).

        `children` lists this process's children; `poll_s` is the reaper's interval.
        """
        self._want_subreaper = subreaper_supported() if subreaper is None else subreaper
        self._children = children
        self._poll_s = poll_s
        self._lock = threading.RLock()
        self._groups: dict[int, subprocess.Popen[bytes]] = {}
        self._launched: dict[int, subprocess.Popen[bytes]] = {}
        self._orphans: dict[int, Orphan] = {}
        self._ended: set[int] = set()
        self._foreign: set[int] = set()
        self._foreign_groups: set[int] = set()
        self._own_pgid = os.getpgid(0)
        self._keeper = Keeper()
        self._entered = False
        self._adopting = False
        self._previous_flag = 0
        self._stop = threading.Event()
        self._reaper: threading.Thread | None = None
        self.reaped: collections.deque[tuple[int, int]] = collections.deque(maxlen=REAPED_KEPT)

    @property
    def adopting(self) -> bool:
        """Whether this supervisor is entered as a subreaper, so orphans re-parent to it."""
        return self._adopting

    def __enter__(self) -> Self:
        """Become the supervisor in force; as a subreaper, start reaping orphans."""
        with self._lock:
            self._entered = True
            if self._want_subreaper:
                try:
                    self._previous_flag = _subreaper_flag()
                    _prctl(PR_SET_CHILD_SUBREAPER, 1)
                except OSError as exc:
                    logger.warning("cannot become a subreaper; supervising by process group only: %s", exc)
                    self._want_subreaper = False
            if self._want_subreaper:
                self._foreign = self._children() - self._launched.keys()
                self._foreign_groups = {g for g in map(_group_of, self._foreign) if g is not None}
                self._adopting = True
                self._stop.clear()
                self._reaper = threading.Thread(target=self._reap_loop, name="senselab-supervisor", daemon=True)
                self._reaper.start()
        _push(self)
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        """SIGKILL every group and orphan held, reap them, stop the keeper and restore the subreaper flag."""
        _pop(self)
        self._stop.set()
        if self._reaper is not None:
            self._reaper.join()
            self._reaper = None
        deadline = time.monotonic() + EXIT_BOUND_S
        with self._lock:
            for pgid, proc in self._groups.items():
                if proc.returncode is None:
                    _kill(pgid, signal.SIGKILL, group=True)
            while self._adopting:
                self._adopt()
                for orphan in self._orphans.values():
                    _kill(orphan.pid, signal.SIGKILL)
                self.reap()
                leaders = any(p.returncode is None and not _exited(g) for g, p in self._groups.items())
                if (not self._orphans and not leaders) or time.monotonic() > deadline:
                    break
                time.sleep(self._poll_s)
            for proc in self._groups.values():
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(max(deadline - time.monotonic(), 0.0))
            self._prune()
            self._keeper.stop(max(deadline - time.monotonic(), 0.0))
            self._groups.clear()
            self._launched.clear()
            if self._adopting:
                with contextlib.suppress(OSError):
                    _prctl(PR_SET_CHILD_SUBREAPER, self._previous_flag)
            self._adopting = self._entered = False

    def popen(self, argv: Sequence[str], **popen: Any) -> "subprocess.Popen[bytes]":  # noqa: ANN401
        """Start `argv` in a new session as `subprocess.Popen` does with `popen`; enter its group in the table.

        The group is recorded with the keeper.
        """
        with self._lock:
            proc: subprocess.Popen[bytes] = subprocess.Popen(argv, **{**popen, "start_new_session": True})  # noqa: S603
            self._prune()
            self._groups[proc.pid] = self._launched[proc.pid] = proc
            self._ended.discard(proc.pid)
            self._keeper.record("+", "g", proc.pid)
            return proc

    def released(self, pgid: int) -> None:
        """Drop launched group `pgid` before its leader is reaped; SIGKILL its orphans and those held only for it."""
        with self._lock:
            self._prune()
            if pgid in self._groups:
                self._drop(pgid)

    def group_live(self, pgid: int) -> bool:
        """Whether group `pgid` can still run code: from the table when adopting, else by probing the group."""
        if not self._adopting:
            return live_members(pgid)
        with self._lock:
            self._adopt()
            leader = self._groups.get(pgid)
            if leader is not None and leader.returncode is None and not _exited(pgid):
                return True
            return any(o.pgid == pgid and not _exited(o.pid) for o in self._orphans.values())

    def orphans(self) -> dict[int, int]:
        """Each adopted orphan not yet reaped, by pid, with its process group."""
        with self._lock:
            self._adopt()
            return {pid: orphan.pgid for pid, orphan in self._orphans.items()}

    def reap(self) -> None:
        """Adopt new orphans and reap each that has exited, recording its status in `reaped`."""
        with self._lock:
            self._prune()
            self._adopt()
            for pid in list(self._orphans):
                if not _exited(pid):
                    continue
                self._keeper.record("-", "p", pid)
                status = _reap(pid)
                del self._orphans[pid]
                if status is not None:
                    self.reaped.append((pid, status))

    def _adopt(self) -> None:
        if not self._adopting:
            return
        listed = self._children()
        self._foreign &= listed
        self._launched = {pid: proc for pid, proc in self._launched.items() if proc.returncode is None}
        known = self._launched.keys() | self._orphans.keys() | self._foreign
        if self._keeper.pid is not None:
            known.add(self._keeper.pid)
        for pid in listed - known:
            pgid = _group_of(pid)
            if pgid is None or pgid == self._own_pgid or pgid in self._foreign_groups:
                continue
            owners = set() if pgid in self._groups else set(self._groups)
            orphan = self._orphans[pid] = Orphan(pid, pgid, owners)
            self._keeper.record("+", "p", pid)
            if pgid not in self._groups and (pgid in self._ended or not owners):
                self._end(orphan)

    def _prune(self) -> None:
        """Drop each launched group whose leader its owner has reaped."""
        for pgid in [g for g, proc in self._groups.items() if proc.returncode is not None]:
            self._drop(pgid)

    def _drop(self, pgid: int) -> None:
        """Forget group `pgid`, with the keeper too; SIGKILL its orphans and those held only for it."""
        del self._groups[pgid]
        self._keeper.record("-", "g", pgid)
        self._ended.add(pgid)
        if not self._adopting:
            return
        self._adopt()
        for orphan in self._orphans.values():
            if orphan.pgid == pgid or pgid in orphan.owners:
                orphan.owners.discard(pgid)
                if orphan.pgid == pgid or not orphan.owners:
                    self._end(orphan)

    def _end(self, orphan: Orphan) -> None:
        """SIGKILL `orphan` and, when it leads a group of its own, that group."""
        self._ended.add(orphan.pgid)
        _kill(orphan.pid, signal.SIGKILL)
        if orphan.pgid == orphan.pid:
            _kill(orphan.pgid, signal.SIGKILL, group=True)

    def _reap_loop(self) -> None:
        while not self._stop.wait(self._poll_s):
            try:
                self.reap()
            except Exception:  # noqa: BLE001 - the reaper keeps running
                logger.exception("the supervisor's reaper pass failed")


def _group_of(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None


_STACK: list[Supervisor] = []
_STACK_LOCK = threading.Lock()
_DEFAULT: list[Supervisor] = []


def _push(supervisor: Supervisor) -> None:
    with _STACK_LOCK:
        _STACK.append(supervisor)


def _pop(supervisor: Supervisor) -> None:
    with _STACK_LOCK:
        if supervisor in _STACK:
            _STACK.remove(supervisor)


def current() -> Supervisor:
    """The innermost supervisor entered in this process, else the process-wide one, which supervises by group."""
    with _STACK_LOCK:
        if _STACK:
            return _STACK[-1]
        if not _DEFAULT:
            _DEFAULT.append(Supervisor(subreaper=False))
        return _DEFAULT[0]


__all__ = [
    "EXIT_BOUND_S",
    "KEEPER",
    "REAPED_KEPT",
    "REAP_POLL_S",
    "Keeper",
    "Orphan",
    "Supervisor",
    "current",
    "live_members",
    "own_children",
    "subreaper_supported",
]
