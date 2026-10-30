"""Transient-error classification and retry."""

from collections.abc import Callable

import httpx
import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from senselab_core.runtime import retry
from senselab_core.runtime.retry import _is_transient, retry_on_transient_error


@pytest.mark.logic
def test_a_definitive_404_is_not_retried(hub_error: Callable[..., Exception]) -> None:
    """A missing repo is a verdict, not a blip: retrying it only delays the real error.

    ``RepositoryNotFoundError`` subclasses ``httpx.HTTPError`` -> ``OSError``, so it matches
    ``_TRANSIENT_EXCEPTIONS``, and it carries its status only on ``.response.status_code``. Before
    ``_http_status`` read that attribute, this classified as transient and every genuine 404 cost
    three attempts and 3s of backoff.
    """
    exc = hub_error(404, RepositoryNotFoundError)
    assert not hasattr(exc, "code") and not hasattr(exc, "status_code"), (
        "the status lives on .response.status_code alone -- that is the whole point of this test"
    )
    assert _is_transient(exc) is False


@pytest.mark.logic
def test_a_gated_403_is_not_retried(hub_error: Callable[..., Exception]) -> None:
    """403 means the token lacks access; waiting does not grant it."""
    assert _is_transient(hub_error(403, GatedRepoError)) is False


@pytest.mark.logic
def test_a_429_is_retried(hub_error: Callable[..., Exception]) -> None:
    """The one 4xx that is transient: the server asking for backoff is what backoff is for."""
    assert _is_transient(hub_error(429)) is True


@pytest.mark.logic
def test_a_500_is_retried(hub_error: Callable[..., Exception]) -> None:
    """Server-side failures are the canonical retryable case."""
    assert _is_transient(hub_error(500)) is True
    assert _is_transient(hub_error(503)) is True


@pytest.mark.logic
def test_connection_and_timeout_errors_are_retried() -> None:
    """The statusless cases, in the shapes each client in play actually raises them.

    ``httpx`` is huggingface_hub 1.x's transport and its transport errors descend from
    ``httpx.HTTPError``, not ``OSError``, so they matched nothing in ``_TRANSIENT_EXCEPTIONS``
    until it learned about them.
    """
    assert _is_transient(httpx.ConnectError("connection refused")) is True
    assert _is_transient(httpx.ReadTimeout("timed out")) is True
    assert _is_transient(ConnectionError("reset by peer")) is True
    assert _is_transient(TimeoutError("timed out")) is True


@pytest.mark.logic
def test_a_non_network_error_is_never_retried() -> None:
    """A programming error must fail on the first attempt, not three seconds later."""
    assert _is_transient(ValueError("bad argument")) is False


@pytest.mark.logic
def test_retry_stops_on_a_404_and_backs_off_on_a_429(
    monkeypatch: pytest.MonkeyPatch, hub_error: Callable[..., Exception]
) -> None:
    """The classification is only worth anything if ``retry_on_transient_error`` acts on it."""
    monkeypatch.setattr(retry.time, "sleep", lambda *_a, **_k: None)

    calls = {"n": 0}

    def _not_found() -> str:
        calls["n"] += 1
        raise hub_error(404, RepositoryNotFoundError)

    with pytest.raises(RepositoryNotFoundError):
        retry_on_transient_error(_not_found)
    assert calls["n"] == 1, "a 404 must fail on the first attempt"

    calls["n"] = 0

    def _throttled_once() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise hub_error(429)
        return "ok"

    assert retry_on_transient_error(_throttled_once) == "ok"
    assert calls["n"] == 2, "a 429 must be retried"
