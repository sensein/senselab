"""Lazy, cached availability checks for optional dependencies and HF model caching."""

import contextlib
import json
import logging
import os
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, TypeVar

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


# ``os.chdir`` is process-global: while this context manager holds the lock and
# CWD is pinned to ``savedir``, any *other* thread doing CWD-relative I/O sees
# the same redirect. To prevent concurrent SpeechBrain loaders from racing on
# the working directory (e.g. loading two models from different threads), the
# context serializes via this module-level lock. Code that needs the original
# CWD inside the body must not run concurrently from another thread — that's
# the documented trade-off. Non-SpeechBrain code paths are unaffected because
# the lock is private to this function.
_speechbrain_cwd_lock = threading.Lock()


@contextlib.contextmanager
def speechbrain_loading_cwd(savedir: Path) -> Iterator[Path]:
    """Run a SpeechBrain ``from_hparams`` call with CWD pinned to ``savedir``.

    Some SpeechBrain hparams.yaml files declare CWD-relative ``save_path`` values
    on inner lobes (e.g. ``save_path: wav2vec2_checkpoints`` on the Wav2Vec2 lobe
    under ``speechbrain/emotion-recognition-wav2vec2-IEMOCAP``), which the outer
    ``savedir=`` argument does not redirect. Wrapping the loader in this context
    causes those relative paths to resolve under ``savedir`` instead of the
    process CWD, keeping the artifacts inside the senselab cache.

    **Threading caveat.** ``os.chdir`` is process-global. To prevent two threads
    concurrently entering this context from racing on the working directory,
    the implementation serializes via a module-level lock — concurrent
    SpeechBrain loads will block, not interleave. This is correct but means
    parallel-model-init use cases will load sequentially. If you need
    concurrent SpeechBrain model construction, use multiple processes (each
    has its own CWD) rather than threads.
    """
    savedir = Path(savedir).resolve()
    savedir.mkdir(parents=True, exist_ok=True)
    with _speechbrain_cwd_lock:
        prev = Path.cwd()
        os.chdir(savedir)
        try:
            yield savedir
        finally:
            os.chdir(prev)


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


# Env vars that switch HuggingFace libraries to local-cache-only mode. Both are
# honored at call time (read fresh on each download/HEAD), so toggling them
# around a load reliably suppresses the network revision-check that 429s under
# many parallel jobs.
_HF_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")

# ``HF_HUB_OFFLINE`` is process-global. ``hf_offline_loading`` serializes its
# temporary toggle through this lock so concurrent loaders in the same process
# don't observe each other's env mutation. Model *loads* (not inference) thus
# run sequentially while the env is flipped — acceptable since loads are
# one-time and cached. (Mirrors the global-state-lock precedent of
# ``speechbrain_loading_cwd``.)
_hf_offline_lock = threading.RLock()


def hf_subprocess_env(
    repo_id: "str | os.PathLike[str]",
    revision: str = "main",
    *,
    also: Optional[Iterable[tuple[str, str]]] = None,
    base_env: Optional[dict] = None,
) -> dict:
    """Build an environment dict for a subprocess that loads HF model(s) offline.

    Ensures every referenced snapshot is present locally first (download-once
    across processes/nodes via :func:`ensure_hf_model`), then returns a copy of
    ``base_env`` (default ``os.environ``) with ``HF_HUB_OFFLINE`` /
    ``TRANSFORMERS_OFFLINE`` set to ``"1"`` so the child's ``from_pretrained``
    skips the network revision check that rate-limits (429) under many parallel
    jobs.

    The offline flag is only set when **all** referenced models are cached — if
    any is missing, the env is returned unchanged so the child can still
    download it online. Use ``also`` for workers that load more than one model
    (e.g. Qwen3-ASR + its companion forced aligner).

    This is the subprocess analogue of :func:`hf_offline_loading`: in-process
    loaders use the context manager, subprocess-venv workers (NeMo / Qwen /
    Granite) inherit the offline flag through this env. No caller-set env vars
    are required.
    """
    env = dict(os.environ if base_env is None else base_env)
    repos = [(os.fspath(repo_id), revision), *(also or [])]
    if all(hf_local_files_only(rid, rev) for rid, rev in repos):
        for var in _HF_OFFLINE_ENV_VARS:
            env[var] = "1"
    return env


@contextlib.contextmanager
def hf_offline_loading(repo_id: "str | os.PathLike[str]", revision: str = "main") -> Iterator[bool]:
    """Force local-cache-only HuggingFace loading for the duration of the block.

    Ensures the snapshot is present first (download-once across processes/nodes
    via :func:`ensure_hf_model`: cross-process heartbeat lock + retry-on-429),
    then sets ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` so ``from_pretrained``
    / ``pipeline`` make **no** network revision-check calls during the load —
    the source of 429 storms when many jobs load the same model at once.
    Requires **no** caller configuration.

    If the model cannot be cached (e.g. genuinely offline and never downloaded),
    the env is left untouched and the block runs normally (online), so behavior
    degrades gracefully rather than failing.

    Yields:
        ``True`` if offline mode was engaged (model is cached), else ``False``.
    """
    repo_id = os.fspath(repo_id)
    if not hf_local_files_only(repo_id, revision):
        yield False
        return
    with _hf_offline_lock:
        saved = {var: os.environ.get(var) for var in _HF_OFFLINE_ENV_VARS}
        for var in _HF_OFFLINE_ENV_VARS:
            os.environ[var] = "1"
        try:
            yield True
        finally:
            for var, prev in saved.items():
                if prev is None:
                    os.environ.pop(var, None)
                else:
                    os.environ[var] = prev


def load_hf_resilient(
    loader: Callable[..., _T],
    *args: object,
    repo_id: "str | os.PathLike[str]",
    revision: str = "main",
    **kwargs: object,
) -> _T:
    """Load an HF model resiliently: cache-once, load local-only, retry on 429.

    Loader-agnostic wrapper for any in-process model constructor
    (``transformers.pipeline``, ``*.from_pretrained``, SpeechBrain
    ``from_hparams``, ...). It:

    1. Ensures the snapshot is present exactly once across processes/nodes
       (:func:`ensure_hf_model` — cross-process heartbeat lock + retry).
    2. Runs ``loader(*args, **kwargs)`` with HF libs in local-cache-only mode
       (:func:`hf_offline_loading`) so no network revision-check fires.
    3. Retries the load on any residual transient error (5xx / 429 / timeout).

    ``repo_id`` / ``revision`` identify the model for the cache step and are
    **not** forwarded to ``loader`` — pass the loader's own model/revision
    arguments via ``*args`` / ``**kwargs`` as usual.
    """

    def _call() -> _T:
        with hf_offline_loading(repo_id, revision):
            return loader(*args, **kwargs)

    return retry_on_transient_error(_call)


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
