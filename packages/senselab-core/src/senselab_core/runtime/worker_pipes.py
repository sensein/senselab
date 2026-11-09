"""The pipes of a worker process: its stdout and stderr split into lines by a thread, and the lines read in order."""

import os
import queue
import selectors
import threading
from collections.abc import Callable

from senselab_core.runtime.clock import Clock, Deadline

READ_BYTES = 2**16


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
        self.ended = threading.Event()
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


class Lines:
    """The lines the stdout pump produced, taken by one reader at a time, each wait read on `clock`."""

    def __init__(self, clock: Clock) -> None:
        """Start empty."""
        self._clock = clock
        self._queue: queue.SimpleQueue[bytes | None] = queue.SimpleQueue()
        self._ready = threading.Event()

    def put(self, line: bytes | None) -> None:
        """Add a line; `None` is the end of the stream."""
        self._queue.put(line)
        self._ready.set()

    def get(self, deadline: Deadline) -> bytes | None:
        """Return the next line, waiting until `deadline`; raise `queue.Empty` if none came."""
        while True:
            for _ in range(2):
                try:
                    return self._queue.get_nowait()
                except queue.Empty:
                    self._ready.clear()
            remaining = deadline.remaining()
            if remaining is not None and remaining <= 0:
                raise queue.Empty
            self._clock.wait(self._ready, threading.TIMEOUT_MAX if remaining is None else remaining)
