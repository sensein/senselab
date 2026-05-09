"""Lazy, cached availability checks for optional dependencies and HF model caching."""

import json
import logging
import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, TypeVar

logger = logging.getLogger("senselab")


@lru_cache(maxsize=1)
def torchaudio_available() -> bool:
    """Return True if torchaudio can be imported without errors."""
    try:
        import torchaudio  # noqa: F401

        return True
    except (ImportError, RuntimeError):
        return False


# ---------------------------------------------------------------------------
# General-purpose retry for transient network errors
# ---------------------------------------------------------------------------

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


def _is_transient(exc: Exception) -> bool:
    """Return True if the exception looks like a transient network error."""
    if isinstance(exc, _TRANSIENT_EXCEPTIONS):
        # For HTTP errors, only retry on server-side codes (5xx) and 429 (rate limit)
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if status is not None:
            return int(status) >= 429
        return True
    return False


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


# ---------------------------------------------------------------------------
# HuggingFace model caching utilities
# ---------------------------------------------------------------------------


def _senselab_cache_dir() -> Path:
    """Return the directory used by senselab for cross-process caching.

    Defaults to ``{HF_HOME}/senselab_cache``.  Created on first access.
    """
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_dir = hf_home / "senselab_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def speechbrain_savedir(repo_id: str, revision: Optional[str] = None) -> Path:
    """Return a stable on-disk location for SpeechBrain ``from_hparams(savedir=...)``.

    SpeechBrain's ``from_hparams`` defaults ``savedir`` to ``./pretrained_models/...``
    (or the ``MODULES_NEEDED`` directory specified in hyperparams.yaml, e.g.
    ``./wav2vec2_checkpoints``). That dumps multi-hundred-MB checkpoints in the
    user's CWD — which on this repo lands inside the working tree as untracked
    files. Pinning savedir under the senselab cache keeps SpeechBrain artifacts
    co-located with the HuggingFace cache.
    """
    rev = revision or "main"
    return _senselab_cache_dir() / "speechbrain" / _safe_key(repo_id, rev)


def _safe_key(repo_id: str, revision: str) -> str:
    """Return a filesystem-safe key for a (repo_id, revision) pair."""
    return f"{repo_id.replace('/', '--')}--{revision}"


def is_hf_model_cached(repo_id: str, revision: str = "main", repo_type: str = "model") -> bool:
    """Check whether a HuggingFace model snapshot exists in the local cache.

    This is a **filesystem-only** check — no network calls are made.
    Returns ``True`` when ``HF_HUB_OFFLINE=1`` is set.
    """
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        return True

    try:
        from huggingface_hub import try_to_load_from_cache

        result = try_to_load_from_cache(
            repo_id=repo_id,
            filename="config.json",
            revision=revision,
            repo_type=repo_type,
        )
        return isinstance(result, str)
    except Exception:
        return False


def _get_cached_commit_hash(repo_id: str, revision: str = "main") -> str:
    """Read the resolved commit hash from the local HF cache directory structure."""
    from huggingface_hub import try_to_load_from_cache

    result = try_to_load_from_cache(
        repo_id=repo_id,
        filename="config.json",
        revision=revision,
    )
    if isinstance(result, str):
        # Path looks like: .../snapshots/<commit_hash>/config.json
        return Path(result).parent.name
    return revision


def _read_result_cache(repo_id: str, revision: str) -> Optional[dict]:
    """Read a cached validation/download result from the filesystem."""
    cache_file = _senselab_cache_dir() / f"{_safe_key(repo_id, revision)}.json"
    if not cache_file.is_file():
        return None
    try:
        return json.loads(cache_file.read_text())  # type: ignore[no-any-return]
    except Exception:
        return None


def _write_result_cache(repo_id: str, revision: str, **data: object) -> None:
    """Write a validation/download result to the filesystem cache."""
    cache_file = _senselab_cache_dir() / f"{_safe_key(repo_id, revision)}.json"
    try:
        cache_file.write_text(json.dumps(data))
    except Exception:
        pass  # Best-effort; failure to cache is not fatal


class _HeartbeatLock:
    """A file lock with a heartbeat mechanism for long-running operations.

    While the lock is held, a background thread touches a heartbeat file every
    ``heartbeat_interval`` seconds.  Waiting processes check the heartbeat when
    their initial timeout expires: if the heartbeat is recent, the download is
    still in progress and they keep waiting; if it's stale, the holder likely
    crashed and the lock can be broken.
    """

    def __init__(
        self,
        lock_path: Path,
        heartbeat_interval: int = 30,
        stale_threshold: int = 90,
    ) -> None:
        from filelock import FileLock

        self._lock_path = lock_path
        self._heartbeat_path = lock_path.with_suffix(".heartbeat")
        self._heartbeat_interval = heartbeat_interval
        self._stale_threshold = stale_threshold
        self._lock = FileLock(str(lock_path))
        self._stop_event = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self._heartbeat_interval):
            try:
                self._heartbeat_path.touch()
            except Exception:
                pass

    def _is_heartbeat_stale(self) -> bool:
        if not self._heartbeat_path.exists():
            return True
        try:
            age = time.time() - self._heartbeat_path.stat().st_mtime
            return age > self._stale_threshold
        except Exception:
            return True

    def __enter__(self) -> "_HeartbeatLock":
        initial_timeout = 60
        while True:
            try:
                self._lock.acquire(timeout=initial_timeout)
                break
            except TimeoutError:
                if self._is_heartbeat_stale():
                    logger.warning(
                        "Stale lock detected (heartbeat expired) at %s — breaking lock",
                        self._lock_path,
                    )
                    try:
                        self._lock_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    self._lock.acquire(timeout=initial_timeout)
                    break
                else:
                    logger.info(
                        "Download in progress (heartbeat active) at %s — continuing to wait",
                        self._lock_path,
                    )
                    # Keep waiting in 60s increments
                    continue

        # Start heartbeat once we hold the lock
        self._stop_event.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        self._heartbeat_path.touch()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop_event.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=5)
        try:
            self._heartbeat_path.unlink(missing_ok=True)
        except Exception:
            pass
        self._lock.release()


def hf_local_files_only(repo_id: str, revision: str = "main") -> bool:
    """Return True if the model is cached and ``local_files_only=True`` is safe.

    Call this before any ``from_pretrained`` / ``pipeline`` invocation.
    If the model is not yet cached, triggers :func:`ensure_hf_model` to
    download it (with cross-process locking), then returns True.
    Returns False only if the download fails, allowing normal (online) loading.
    """
    if is_hf_model_cached(repo_id, revision):
        return True
    try:
        ensure_hf_model(repo_id, revision)
        return True
    except Exception:
        return False


def _cached_error(cached: dict) -> Exception:
    """Reconstruct an exception from a cached error result."""
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

    msg = cached.get("error_message", "")
    if cached.get("error_type") == "RepositoryNotFoundError":
        return RepositoryNotFoundError(msg)
    return RevisionNotFoundError(msg)


def ensure_hf_model(repo_id: str, revision: str = "main", token: Optional[str] = None) -> str:
    """Ensure a HuggingFace model is available locally.

    Uses file locking so only one process per ``(repo_id, revision)`` does the
    API check and download.  All other processes wait on the lock and then reuse
    the cached result.

    Both successes *and* definitive failures (repository/revision not found) are
    cached so that subsequent processes avoid redundant API calls.  Transient
    failures (network errors, rate limits) are **not** cached and are retried
    with exponential backoff.

    Returns:
        The resolved commit hash of the downloaded snapshot.

    Raises:
        RepositoryNotFoundError: If the repository does not exist (cached).
        RevisionNotFoundError: If the revision does not exist (cached).
        Exception: On transient failures after exhausting retries.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

    from senselab.utils.data_structures.model import get_huggingface_token

    # Fast path 1: model already downloaded
    if is_hf_model_cached(repo_id, revision):
        return _get_cached_commit_hash(repo_id, revision)

    # Fast path 2: result cached from a prior process (success or definitive failure)
    cached = _read_result_cache(repo_id, revision)
    if cached is not None:
        if cached.get("status") == "ok":
            return str(cached["commit_hash"])
        raise _cached_error(cached)

    # Slow path: acquire lock, re-check, then download
    lock_path = _senselab_cache_dir() / f"{_safe_key(repo_id, revision)}.lock"
    with _HeartbeatLock(lock_path):
        # Re-check after acquiring lock
        if is_hf_model_cached(repo_id, revision):
            return _get_cached_commit_hash(repo_id, revision)
        cached = _read_result_cache(repo_id, revision)
        if cached is not None:
            if cached.get("status") == "ok":
                return str(cached["commit_hash"])
            raise _cached_error(cached)

        # Download with retries on transient errors
        resolved_token = token or get_huggingface_token()
        try:
            retry_on_transient_error(
                snapshot_download,
                repo_id=repo_id,
                revision=revision,
                token=resolved_token,
            )
            commit_hash = _get_cached_commit_hash(repo_id, revision)
            _write_result_cache(repo_id, revision, status="ok", commit_hash=commit_hash)
            return commit_hash
        except (RepositoryNotFoundError, RevisionNotFoundError) as exc:
            # GatedRepoError (subclass of RepositoryNotFoundError) means the repo
            # exists but requires auth — do NOT cache as definitive failure.
            from huggingface_hub.errors import GatedRepoError

            if isinstance(exc, GatedRepoError):
                raise
            # Definitive failure — cache so other processes don't repeat the API call
            _write_result_cache(
                repo_id,
                revision,
                status="error",
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            raise
    # Should never reach here, but satisfy mypy
    raise RuntimeError("Unreachable")  # pragma: no cover
