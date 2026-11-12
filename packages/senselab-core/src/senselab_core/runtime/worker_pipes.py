"""The pipes of a worker process: its stdout and stderr split into lines by a thread, and the lines read in order."""

import contextlib
import os
import queue
import selectors
import socket
import threading
import time
import weakref
from collections.abc import Callable, Iterator
from typing import Final

from senselab_worker.codec import Deadline

from senselab_core.runtime.clock import Clock

HINT_BYTE: Final = b"h"
WAKE_BYTE: Final = b"x"
READ_BYTES: Final = 2**16  # invariant: the size of one pipe read; lines are reassembled whatever it is


def _close_fds(*fds: int) -> None:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def _close_once(box: list[tuple[int, ...]]) -> None:
    """Close the descriptors in `box` unless another caller has already taken them; they are closed once."""
    try:
        fds = box.pop()
    except IndexError:
        return
    _close_fds(*fds)


class SharedFds:
    """Descriptors that threads and signal handlers use without a lock, closed by the last user after `release`.

    A call brackets its use in `enter` and `leave`, and uses the descriptors only when `enter` returned a token. After
    `release`, the last call to leave closes them, once, so no call ever uses a number that was reused elsewhere.
    """

    def __init__(self, *fds: int) -> None:
        """Own `fds`; they are also closed when this object is collected."""
        self.fds = fds
        self.released = False
        self._box = [fds]
        self._users: set[object] = set()
        closer = weakref.finalize(self, _close_once, self._box)
        setattr(closer, "atexit", False)  # noqa: B010 - mypy's weakref stub has no `atexit` slot

    def enter(self) -> object | None:
        """Start a use; return its token, or None once released, when the descriptors must not be used."""
        token = object()
        self._users.add(token)
        if self.released:
            self.leave(token)
            return None
        return token

    def leave(self, token: object) -> None:
        """End the use `token`."""
        self._users.discard(token)
        self._close_if_unused()

    def release(self) -> None:
        """Give the descriptors up; the last use in flight, or this call, closes them. Safe to call twice."""
        self.released = True
        self._close_if_unused()

    def write(self, data: bytes) -> None:
        """Write `data` to the first descriptor unless released, ignoring a full or closed peer."""
        token = self.enter()
        if token is None:
            return
        try:
            os.write(self.fds[0], data)
        except OSError:
            pass
        finally:
            self.leave(token)

    def _close_if_unused(self) -> None:
        if self.released and not self._users:
            _close_once(self._box)


class WakeEvent(threading.Event):
    """A `threading.Event` that takes no lock: a flag and a pipe, so a signal handler can `set` and `wait` on it.

    The main thread can be interrupted inside any `threading.Event`, which holds its condition's lock there, and a
    handler that then waited on or set the same event would block for good. A `WakeEvent` waits in a selector.
    `close` wakes every waiter; a closed event keeps its flag and no longer blocks. Its pipe is a `SharedFds`.
    """

    def __init__(self) -> None:
        """Start clear."""
        self._flag = False
        self._r, self._w = os.pipe()
        os.set_blocking(self._r, False)
        os.set_blocking(self._w, False)
        self._pipe = SharedFds(self._w, self._r)

    def is_set(self) -> bool:
        """Whether the flag is set."""
        return self._flag

    def set(self) -> None:
        """Set the flag and wake every waiter."""
        self._flag = True
        self._pipe.write(WAKE_BYTE)

    def clear(self) -> None:
        """Clear the flag."""
        self._flag = False
        token = self._pipe.enter()
        if token is not None:
            try:
                self._drain()
            finally:
                self._pipe.leave(token)

    def _drain(self) -> None:
        try:
            while os.read(self._r, READ_BYTES):
                pass
        except OSError:
            pass

    def close(self) -> None:
        """Wake every waiter and give up the pipe; safe to call twice."""
        token = self._pipe.enter()
        if token is not None:
            try:
                self._pipe.release()  # before the wake, so a waiter it wakes sees it; this call still holds the pipe
                os.write(self._w, WAKE_BYTE)
            except OSError:
                pass
            finally:
                self._pipe.leave(token)

    @contextlib.contextmanager
    def held(self) -> Iterator[int]:
        """The read end, kept open for the block, to wait on in a selector; -1 once closed. Not for signal handlers."""
        token = self._pipe.enter()
        try:
            yield -1 if token is None else self._r
        finally:
            if token is not None:
                self._pipe.leave(token)

    def wait(self, timeout: float | None = None) -> bool:
        """Wait up to `timeout` seconds (forever when None) for the flag; return whether it is set."""
        end = None if timeout is None else time.monotonic() + timeout
        token = self._pipe.enter()
        try:
            while token is not None and not self._flag and not self._pipe.released:
                left = None if end is None else end - time.monotonic()
                if left is not None and left <= 0:
                    break
                with selectors.DefaultSelector() as sel:
                    sel.register(self._r, selectors.EVENT_READ)
                    if sel.select(left) and not self._flag and not self._pipe.released:
                        self._drain()  # a byte left by a `set` that a `clear` then undid
        finally:
            if token is not None:
                self._pipe.leave(token)
        return self._flag


class Pump:
    """A daemon thread splitting one pipe into lines for `sink` until EOF, or until `finish` drains it.

    Lines longer than `limit` are passed on in pieces of `limit` bytes. At the end `sink(None)` is
    called; a trailing piece without a newline is passed on first only if `keep_partial`.
    """

    def __init__(self, fd: int, limit: int, sink: Callable[[bytes | None], None], *, keep_partial: bool) -> None:
        """Start reading `fd`."""
        self._fd = fd
        self._limit = limit
        self._sink = sink
        self._keep_partial = keep_partial
        self._wake_r, self._wake_w = os.pipe()
        self._lock = threading.Lock()
        self._closed = False
        self.ended = WakeEvent()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def finish(self) -> None:
        """Ask the thread to read what is already in the pipe, then stop."""
        with self._lock:
            if not self._closed:
                try:
                    os.write(self._wake_w, b"x")
                except OSError:
                    pass

    def close(self) -> None:
        """Release the wake pipe; call only once the thread has ended."""
        with self._lock:
            if not self._closed:
                self._closed = True
                os.close(self._wake_r)
                os.close(self._wake_w)
                self.ended.close()

    def _run(self) -> None:
        buf = bytearray()
        scanned = 0
        try:
            with selectors.DefaultSelector() as sel:
                sel.register(self._fd, selectors.EVENT_READ)
                sel.register(self._wake_r, selectors.EVENT_READ)
                draining = False
                while True:
                    if not draining and any(key.fd == self._wake_r for key, _ in sel.select()):
                        draining = True
                        os.set_blocking(self._fd, False)
                    try:
                        chunk = os.read(self._fd, READ_BYTES)
                    except BlockingIOError:
                        if draining:
                            break
                        continue
                    except OSError:
                        break
                    if not chunk:
                        break
                    buf += chunk
                    scanned = self._split(buf, scanned)
        finally:
            if buf and self._keep_partial:
                self._sink(bytes(buf))
            self._sink(None)
            self.ended.set()

    def _split(self, buf: bytearray, start: int) -> int:
        """Pass on every complete line in `buf`, searching from `start`; return where the next search starts."""
        while True:
            newline = buf.find(b"\n", start, self._limit)
            if newline >= 0:
                self._sink(bytes(buf[: newline + 1]))
                del buf[: newline + 1]
            elif len(buf) >= self._limit:
                self._sink(bytes(buf[: self._limit]))
                del buf[: self._limit]
            else:
                return len(buf)
            start = 0


def _relay_wakes(source: socket.socket, ready: threading.Event, on_hint: Callable[[], None] | None) -> None:
    """Set `ready` for each batch of bytes `source` delivers, and call `on_hint` for each hint byte in it."""
    try:
        while data := source.recv(READ_BYTES):
            if on_hint is not None:
                for _ in range(data.count(HINT_BYTE)):
                    on_hint()
            ready.set()
    except OSError:
        pass
    finally:
        source.close()


class Lines:
    """The lines the stdout pump produced, taken by one reader at a time, each wait read on `clock`.

    `put` and `request_hint` take no lock, so they are safe from a signal handler and never wait on a lock the reader
    holds: `put` adds to a `SimpleQueue` and, while a reader waits, writes a byte to a socket pair; a relay thread of
    this object reads it, sets the wake event the reader waits on and calls `on_hint` for a hint. `close` ends the
    relay thread without waiting for it.
    """

    def __init__(
        self, clock: Clock, ready: threading.Event | None = None, *, on_hint: Callable[[], None] | None = None
    ) -> None:
        """Start empty; `ready` is the event a reader waits on (a new one when None).

        `on_hint` is what the relay thread calls for each `request_hint`.
        """
        self._clock = clock
        self._queue: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()
        self._ready = threading.Event() if ready is None else ready
        self._waiting = False
        relay_r, relay_w = socket.socketpair()
        relay_w.setblocking(False)
        self._relay = SharedFds(relay_w.detach())
        threading.Thread(
            target=_relay_wakes, args=(relay_r, self._ready, on_hint), name="senselab-lines-relay", daemon=True
        ).start()

    def put(self, line: bytes | None) -> None:
        """Add a line; `None` is the end of the stream."""
        self._queue.put(line)
        if self._waiting:  # else the reader's next look finds the line
            self._send(WAKE_BYTE)

    def request_hint(self) -> None:
        """Have the relay thread call `on_hint`."""
        self._send(HINT_BYTE)

    def _send(self, byte: bytes) -> None:
        self._relay.write(byte)  # a full socket already has a wake pending

    def close(self) -> None:
        """End the relay thread once no write is in flight; safe to call twice."""
        self._relay.release()

    def get(self, deadline: Deadline) -> bytes | None:
        """Return the next line, waiting until `deadline`; raise `queue.Empty` if none came.

        The wake flag is cleared before each look and never between the last look and the wait, so a line put after
        a look sets it again.
        """
        try:
            while True:
                self._waiting = True
                self._ready.clear()
                try:
                    return self._queue.get_nowait()
                except queue.Empty:
                    pass
                remaining = deadline.remaining()
                if remaining is not None and remaining <= 0:
                    raise queue.Empty
                self._clock.wait(self._ready, threading.TIMEOUT_MAX if remaining is None else remaining)
        finally:
            self._waiting = False
