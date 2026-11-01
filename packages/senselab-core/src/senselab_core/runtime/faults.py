"""The one classifier of a fault as transient (a retry may succeed) or permanent."""

import errno
import socket
import urllib.error

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

HTTP_CLIENTS: frozenset[str] = frozenset({"httpx", "httpx2"})
"""The HTTP client packages whose `TransportError` subclasses are transient: those `huggingface_hub` 1.x and 2.x use."""


def http_status(exc: BaseException) -> int | None:
    """Return the HTTP status an exception carries, else None.

    The status is `code` of a `urllib.error.HTTPError`, and otherwise `response.status_code`, which
    httpx, httpx2, requests and `huggingface_hub` errors all carry.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def _is_transport_error(exc: BaseException) -> bool:
    return any(
        cls.__name__ == "TransportError" and cls.__module__.partition(".")[0] in HTTP_CLIENTS for cls in type(exc).__mro__
    )


def is_transient(exc: BaseException) -> bool:
    """Return whether retrying the operation that raised `exc` may succeed.

    - An HTTP error with a status (`http_status`): transient for 429 and for 500 and above.
    - A `TransportError` of an HTTP client in `HTTP_CLIENTS`, `ConnectionError` and `TimeoutError`: transient.
    - `urllib.error.URLError` without a status: its `reason` when that is an exception, else permanent.
    - `socket.gaierror`: transient only for `EAI_AGAIN`.
    - Any other `OSError`: transient when its errno is in `TRANSIENT_ERRNOS`; with no errno, its
      `__cause__` decides when it has one.
    - Anything else: permanent.
    """
    status = http_status(exc)
    if status is not None:
        return status == 429 or status >= 500
    if _is_transport_error(exc) or isinstance(exc, (ConnectionError, TimeoutError)):
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
