"""The one classifier of a fault as transient (a retry may succeed) or permanent."""

import errno
import socket
import urllib.error

import httpx

TRANSIENT_ERRNOS: frozenset[int] = frozenset(
    {
        errno.EAGAIN,
        errno.EWOULDBLOCK,
        errno.EINTR,
        errno.EBUSY,
        errno.ETIMEDOUT,
        errno.ECONNRESET,
        errno.ECONNABORTED,
        errno.ECONNREFUSED,
        errno.EPIPE,
        errno.ENETDOWN,
        errno.ENETUNREACH,
        errno.ENETRESET,
        errno.EHOSTDOWN,
        errno.EHOSTUNREACH,
        errno.ENOBUFS,
        errno.ENOLCK,
        errno.EIO,
        errno.ESTALE,
    }
)
"""The errnos of an `OSError` that is transient."""


def http_status(exc: BaseException) -> int | None:
    """Return the HTTP status a `urllib.error.HTTPError` or an `httpx.HTTPError` with a response carries, else None."""
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code
    if isinstance(exc, httpx.HTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status if isinstance(status, int) and not isinstance(status, bool) else None
    return None


def is_transient(exc: BaseException) -> bool:
    """Return whether retrying the operation that raised `exc` may succeed.

    - An HTTP error with a status (`http_status`): transient for 429 and for 500 and above.
    - `httpx.TransportError`, `ConnectionError` and `TimeoutError`: transient.
    - `urllib.error.URLError` without a status: its `reason` when that is an exception, else permanent.
    - `socket.gaierror`: transient only for `EAI_AGAIN`.
    - Any other `OSError`: transient when its errno is in `TRANSIENT_ERRNOS`; with no errno, its
      `__cause__` decides when it has one.
    - Anything else: permanent.
    """
    status = http_status(exc)
    if status is not None:
        return status == 429 or status >= 500
    if isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, urllib.error.URLError):
        return isinstance(exc.reason, BaseException) and is_transient(exc.reason)
    if isinstance(exc, socket.gaierror):
        return exc.errno == socket.EAI_AGAIN
    if isinstance(exc, OSError):
        if exc.errno is None:
            return exc.__cause__ is not None and is_transient(exc.__cause__)
        return exc.errno in TRANSIENT_ERRNOS
    return False
