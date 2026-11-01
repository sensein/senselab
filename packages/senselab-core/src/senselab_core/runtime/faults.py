"""The one classifier of a fault as transient (a retry may succeed) or permanent."""

import errno
import socket
import ssl
import urllib.error
from http import HTTPStatus

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

_TRANSIENT_EAI: frozenset[int] = frozenset({socket.EAI_AGAIN})

TRANSIENT_SSL_ERRORS: tuple[type[ssl.SSLError], ...] = (
    ssl.SSLEOFError,
    ssl.SSLZeroReturnError,
    ssl.SSLSyscallError,
    ssl.SSLWantReadError,
    ssl.SSLWantWriteError,
)
"""The `ssl.SSLError` classes that are transient: a connection cut or closed mid-TLS, or a socket not ready."""


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
        cls.__name__ == "TransportError" and cls.__module__.partition(".")[0] in HTTP_CLIENTS
        for cls in type(exc).__mro__
    )


def is_transient(exc: BaseException) -> bool:
    """Return whether retrying the operation that raised `exc` may succeed.

    - An HTTP error with a status (`http_status`): transient for 429 and for 500 and above; every other
      status, 408 and 425 included, is permanent.
    - A `TransportError` of an HTTP client in `HTTP_CLIENTS`, `ConnectionError` and `TimeoutError`: transient.
    - `urllib.error.URLError` without a status: its `reason` when that is an exception, else permanent.
    - An `ssl.SSLError`, whose errno is an SSL code: transient when it is one of `TRANSIENT_SSL_ERRORS`.
    - Any other `OSError`: transient when its errno is in `TRANSIENT_ERRNOS`, or for a `socket.gaierror`, whose
      errno is an `EAI_*` code, when it is `EAI_AGAIN`; with no errno, its `__cause__` decides when it has one.
    - Anything else: permanent, as is a chain of reasons and causes that returns to an exception already seen.
    """
    decided: bool | BaseException = exc
    seen: set[int] = set()
    while isinstance(decided, BaseException):
        if id(decided) in seen:
            return False
        seen.add(id(decided))
        decided = _verdict(decided)
    return decided


def _verdict(exc: BaseException) -> bool | BaseException:
    """Return whether `exc` is transient, or the exception that decides for it."""
    status = http_status(exc)
    if status is not None:
        return status == HTTPStatus.TOO_MANY_REQUESTS or status >= HTTPStatus.INTERNAL_SERVER_ERROR
    if _is_transport_error(exc) or isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, urllib.error.URLError):
        return exc.reason if isinstance(exc.reason, BaseException) else False
    if not isinstance(exc, OSError):
        return False
    if isinstance(exc, ssl.SSLError):
        return isinstance(exc, TRANSIENT_SSL_ERRORS)
    transient = _TRANSIENT_EAI if isinstance(exc, socket.gaierror) else TRANSIENT_ERRNOS
    return exc.errno in transient if exc.errno is not None else (exc.__cause__ or False)
