"""Retry on transient network errors (HF Hub and other HTTP clients)."""

from __future__ import annotations

import logging
import os
import time
from typing import Callable, Optional, TypeVar

logger = logging.getLogger("senselab")


_TRANSIENT_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    OSError,
)

try:
    from urllib.error import HTTPError as UrllibHTTPError

    _TRANSIENT_EXCEPTIONS = (*_TRANSIENT_EXCEPTIONS, UrllibHTTPError)  # type: ignore[assignment]
except ImportError:
    pass

try:
    from requests.exceptions import ConnectionError as RequestsConnectionError
    from requests.exceptions import HTTPError as RequestsHTTPError
    from requests.exceptions import Timeout as RequestsTimeout

    _TRANSIENT_EXCEPTIONS = (*_TRANSIENT_EXCEPTIONS, RequestsConnectionError, RequestsHTTPError, RequestsTimeout)  # type: ignore[assignment]
except ImportError:
    pass

try:
    # huggingface_hub 1.x transports over httpx, whose transport errors descend from
    # httpx.HTTPError -- plain Exception, not OSError -- so none of the entries above match
    # them. Without this, a dropped connection or read timeout mid-`model_info` was classified
    # non-transient and raised on the first attempt, while this module's whole reason to exist
    # is retrying exactly that. Status errors (httpx.HTTPStatusError and huggingface_hub's
    # HfHubHTTPError, which also subclasses OSError) are judged on their status below, not here.
    from httpx import TransportError as HttpxTransportError

    _TRANSIENT_EXCEPTIONS = (*_TRANSIENT_EXCEPTIONS, HttpxTransportError)  # type: ignore[assignment]
except ImportError:
    pass


def _http_status(exc: BaseException) -> Optional[int]:
    """Return the HTTP status code carried by *exc*, or None if it carries none.

    Three attribute shapes, because three clients are in play: ``urllib`` records it on
    ``.code``, some SDKs on ``.status_code``, and ``requests``/``httpx`` -- the transport
    ``huggingface_hub`` actually uses -- only on ``.response.status_code``. Reading the first
    two alone made every ``huggingface_hub`` error look statusless: ``RepositoryNotFoundError``
    subclasses ``httpx.HTTPError`` *and* ``OSError``, so it matched ``_TRANSIENT_EXCEPTIONS``
    and then fell through to the no-status branch, and a definitive 404 was retried three times
    with backoff before failing.

    ``bool`` is excluded explicitly because ``isinstance(True, int)`` holds, and a truthy
    non-status attribute would otherwise be read as status 1.

    Args:
        exc: The exception to inspect.

    Returns:
        The status code, or None if the exception carries no HTTP status.
    """
    response = getattr(exc, "response", None)
    for candidate in (
        getattr(exc, "code", None),
        getattr(exc, "status_code", None),
        getattr(response, "status_code", None),
    ):
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    return None


def _is_transient(exc: Exception) -> bool:
    """Return True if the exception looks like a transient network error.

    Args:
        exc: The exception raised by the call being retried.

    Returns:
        True if retrying the same call could plausibly succeed.
    """
    if not isinstance(exc, _TRANSIENT_EXCEPTIONS):
        return False
    status = _http_status(exc)
    if status is None:
        # Connection resets, DNS failures and timeouts never reach a status. Retrying is
        # the point of this function, so an unclassifiable member of the tuple retries.
        return True
    # 429 is the one 4xx worth retrying: it is the server asking for backoff, not a verdict
    # on the request. Every other 4xx -- 401 no token, 403 gated, 404 no such repo -- is a
    # verdict no amount of waiting changes, and retrying it only delays the real error by the
    # backoff schedule while hammering a Hub that already answered.
    return status == 429 or status >= 500


_T = TypeVar("_T")


def retry_on_transient_error(
    fn: Callable[..., _T],
    *args: object,
    max_retries: Optional[int] = None,
    **kwargs: object,
) -> _T:
    """Call *fn* with retries on transient network errors.

    Retries with exponential backoff (1s, 2s, 4s, ...) on connection errors,
    timeouts, and HTTP 5xx/429 responses.  Non-transient exceptions (including
    HTTP 4xx other than 429) are raised immediately.

    Args:
        fn: The callable to invoke.
        *args: Positional arguments forwarded to *fn*.
        max_retries: Override for ``SENSELAB_HF_MAX_RETRIES`` (default 3).
        **kwargs: Keyword arguments forwarded to *fn*.

    Returns:
        The return value of *fn*.
    """
    retries = max_retries if max_retries is not None else int(os.environ.get("SENSELAB_HF_MAX_RETRIES", "3"))
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if _is_transient(exc) and attempt < retries - 1:
                wait = 2**attempt
                logger.warning(
                    "Transient error on attempt %d/%d: %s. Retrying in %ds...",
                    attempt + 1,
                    retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Unreachable")  # pragma: no cover
