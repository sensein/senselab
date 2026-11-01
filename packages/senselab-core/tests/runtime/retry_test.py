"""The transient-fault classifier, the retry policy and retry."""

import ast
import errno
import inspect
import socket
import urllib.error
from collections.abc import Callable
from email.message import Message
from pathlib import Path

import httpx
import pytest
import senselab_core.runtime.retry as retry_module
import yaml
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from senselab_core.runtime.faults import is_transient
from senselab_core.runtime.retry import RetryPolicy, retry_on_transient_error
from senselab_core.runtime.storage import storage_fault

POLICY = RetryPolicy(attempts=3, first_backoff_s=1.0)


def _http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.org/f", status, "status", Message(), None)


@pytest.mark.logic
def test_a_definitive_404_is_not_retried(hub_error: Callable[..., Exception]) -> None:
    """A missing repo, whose status is only on `.response.status_code`, is permanent."""
    exc = hub_error(404, RepositoryNotFoundError)
    assert not hasattr(exc, "code") and not hasattr(exc, "status_code")
    assert is_transient(exc) is False


@pytest.mark.logic
def test_a_gated_403_is_not_retried(hub_error: Callable[..., Exception]) -> None:
    """403 means the token lacks access; waiting does not grant it."""
    assert is_transient(hub_error(403, GatedRepoError)) is False


@pytest.mark.logic
@pytest.mark.parametrize(("status", "transient"), [(429, True), (500, True), (503, True), (404, False), (401, False)])
def test_an_http_status_decides_for_every_client(
    hub_error: Callable[..., Exception], status: int, transient: bool
) -> None:
    """429 and 5xx are transient and every other status is permanent, from the Hub, httpx and urllib alike."""
    request = httpx.Request("GET", "https://example.org/f")
    response = httpx.Response(status, request=request)
    assert is_transient(hub_error(status)) is transient
    assert is_transient(httpx.HTTPStatusError("s", request=request, response=response)) is transient
    assert is_transient(_http_error(status)) is transient


@pytest.mark.logic
def test_connection_and_timeout_errors_are_retried() -> None:
    """Statusless network failures, in the shapes each client raises them, are transient."""
    assert is_transient(httpx.ConnectError("connection refused")) is True
    assert is_transient(httpx.ReadTimeout("timed out")) is True
    assert is_transient(ConnectionResetError("reset by peer")) is True
    assert is_transient(TimeoutError("timed out")) is True
    assert is_transient(urllib.error.URLError(ConnectionRefusedError(errno.ECONNREFUSED, "refused"))) is True
    assert is_transient(urllib.error.URLError("unknown url type: ftp")) is False


@pytest.mark.logic
@pytest.mark.parametrize(
    "code", [errno.EAGAIN, errno.EINTR, errno.ETIMEDOUT, errno.ECONNRESET, errno.EIO, errno.ESTALE, errno.ENETUNREACH]
)
def test_a_transient_errno_is_retried(code: int) -> None:
    """An `OSError` whose errno names a fault that can clear by itself is transient."""
    assert is_transient(OSError(code, "x")) is True


@pytest.mark.logic
@pytest.mark.parametrize(
    "exc",
    [
        FileNotFoundError(errno.ENOENT, "no such file"),
        PermissionError(errno.EACCES, "denied"),
        IsADirectoryError(errno.EISDIR, "is a directory"),
        OSError(errno.ENOSPC, "full"),
        OSError("no errno"),
        ValueError("bad argument"),
    ],
)
def test_a_permanent_fault_is_not_retried(exc: Exception) -> None:
    """A missing file, a refused permission, a full disk or a programming error is permanent."""
    assert is_transient(exc) is False


@pytest.mark.logic
def test_only_a_temporary_name_resolution_failure_is_transient() -> None:
    """`socket.gaierror` carries an `EAI_*` code, not an errno: only `EAI_AGAIN` is transient."""
    assert is_transient(socket.gaierror(socket.EAI_AGAIN, "try again")) is True
    assert is_transient(socket.gaierror(socket.EAI_NONAME, "unknown host")) is False


@pytest.mark.logic
def test_retry_surfaces_a_permanent_os_error_on_the_first_attempt() -> None:
    """A `FileNotFoundError` inside a retried call raises at once, with no backoff."""
    slept: list[float] = []
    calls = {"n": 0}

    def _missing() -> str:
        calls["n"] += 1
        raise FileNotFoundError(errno.ENOENT, "no such file", "/models/w.bin")

    with pytest.raises(FileNotFoundError):
        retry_on_transient_error(_missing, POLICY, sleep=slept.append)
    assert calls["n"] == 1 and slept == []


@pytest.mark.logic
def test_retry_stops_on_a_404_and_backs_off_on_a_429(hub_error: Callable[..., Exception]) -> None:
    """A 404 raises on the first attempt; a 429 is retried after the policy's first backoff."""
    slept: list[float] = []
    calls = {"n": 0}

    def _not_found() -> str:
        calls["n"] += 1
        raise hub_error(404, RepositoryNotFoundError)

    with pytest.raises(RepositoryNotFoundError):
        retry_on_transient_error(_not_found, POLICY, sleep=slept.append)
    assert calls["n"] == 1

    calls["n"] = 0

    def _throttled_once() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise hub_error(429)
        return "ok"

    assert retry_on_transient_error(_throttled_once, POLICY, sleep=slept.append) == "ok"
    assert calls["n"] == 2 and slept == [1.0]


@pytest.mark.logic
def test_the_retry_policy_is_the_argument_and_no_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    """The attempts and the doubling backoff come from the policy passed; `SENSELAB_HF_MAX_RETRIES` is not read."""
    monkeypatch.setenv("SENSELAB_HF_MAX_RETRIES", "1")
    slept: list[float] = []
    calls = {"n": 0}

    def _down() -> str:
        calls["n"] += 1
        raise ConnectionResetError(errno.ECONNRESET, "reset")

    with pytest.raises(ConnectionResetError):
        retry_on_transient_error(_down, RetryPolicy(attempts=4, first_backoff_s=0.5), sleep=slept.append)
    assert calls["n"] == 4 and slept == [0.5, 1.0, 2.0]
    assert "max_retries" not in inspect.signature(retry_on_transient_error).parameters


@pytest.mark.logic
def test_the_packaged_retry_policy_carries_a_derivation_per_value() -> None:
    """`RetryPolicy.load_default` reads `data/retry_policy.yaml`, whose every value has a derivation."""
    path = Path(inspect.getfile(retry_module)).parent / "data" / "retry_policy.yaml"
    raw = yaml.safe_load(path.read_text("utf-8"))
    assert raw.pop("format") == "retry-policy@1.0"
    assert all(entry["derivation"].strip() for entry in raw.values())
    assert RetryPolicy.load_default() == RetryPolicy(**{name: entry["value"] for name, entry in raw.items()})


@pytest.mark.logic
def test_retry_imports_no_undeclared_http_client() -> None:
    """The retry and fault modules import neither `requests` nor anything optionally."""
    for module in (retry_module, inspect.getmodule(is_transient)):
        tree = ast.parse(inspect.getsource(module))  # type: ignore[arg-type]
        imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        imported |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert "requests" not in imported
        guarded = [n for t in ast.walk(tree) if isinstance(t, ast.Try) for n in ast.walk(t)]
        assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in guarded)


@pytest.mark.logic
@pytest.mark.parametrize(
    "err",
    [
        OSError(errno.EIO, "i/o error"),
        OSError(errno.ESTALE, "stale handle"),
        PermissionError(errno.EACCES, "denied"),
        FileNotFoundError(errno.ENOENT, "missing"),
        TimeoutError("timed out"),
    ],
)
def test_storage_faults_are_classified_by_the_same_classifier(err: OSError) -> None:
    """A storage fault is transient exactly when its cause is, and only a transient one tells the caller to retry."""
    fault = storage_fault(err, Path("/store"))
    assert is_transient(fault) is is_transient(err)
    assert ("retry" in str(fault)) is is_transient(err)
