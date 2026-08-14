"""Lazy, cached availability checks for optional dependencies and HF model caching."""

import contextlib
import json
import logging
import os
import re
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Tuple, TypeVar

from senselab.utils.file_lock import SharedFileLock

logger = logging.getLogger("senselab")

# Chosen to reproduce the effective behaviour of the _HeartbeatLock this module used to
# define: it polled with a 60s acquire timeout and looped indefinitely whenever the
# heartbeat still looked fresh, so a live multi-GB download was never abandoned. Under
# SharedFileLock the "is the holder dead" check moved onto the uncontended-acquire path
# (see file_lock.py) rather than the timeout path, so a poll either succeeds immediately
# (holder gone -- possibly a stale takeover, logged by SharedFileLock itself) or times out
# having *proven* the holder is still alive, in which case ensure_hf_model's retry loop
# below waits again. HEARTBEAT_INTERVAL matches _HeartbeatLock's old default.
#
# stale_after is deliberately left at SharedFileLock's own default (120s) rather than
# _HeartbeatLock's old 90s: _HeartbeatLock's 90s was only ever exercised same-host, where
# "heartbeat age" and "wall-clock time" share one clock. _senselab_cache_dir() (below) is
# explicitly meant to be redirected via SENSELAB_CACHE onto a group-writable tree shared
# across cluster nodes, which is exactly the cross-node case file_lock.py's module
# docstring warns about: stale_after must exceed plausible clock skew between the holder's
# node and the checker's, or a live holder on a slow/behind clock reads as dead and gets
# displaced mid-download. 120s (SharedFileLock's own derivation: 4x heartbeat_interval,
# with headroom for both missed beats and skew) costs at most 30s of extra wait before
# taking over a genuinely dead holder -- negligible against a multi-GB download -- and is
# not weaker than what this call site actually needs.
_LOCK_POLL_TIMEOUT = 60.0
_LOCK_HEARTBEAT_INTERVAL = 30.0


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

    Defaults to ``~/.cache/senselab/hf``, a sibling of the venv cache
    (``~/.cache/senselab/venvs`` in ``subprocess_venv.py``). Override with ``SENSELAB_CACHE``.

    Deliberately NOT derived from ``HF_HOME``: on shared HPC clusters, ``HF_HOME`` is routinely
    redirected to a large group-writable tree so model weights are downloaded once and reused
    across users. This directory also holds per-process coordination lock files
    (``SharedFileLock``, see ``resolve_model`` below) — putting those in a tree shared
    by a whole group means unrelated users contend on each other's locks over NFS. Keep this
    directory in senselab's own, per-user namespace no matter where ``HF_HOME`` points.

    Created on first access.
    """
    cache_dir = Path(os.environ.get("SENSELAB_CACHE", str(Path.home() / ".cache" / "senselab" / "hf")))
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

    This is a **filesystem-only** check — no network calls are made, so it is already
    safe under ``HF_HUB_OFFLINE`` and does not special-case it. It used to answer ``True``
    unconditionally when that flag was set, which inverted the question: offline means "do
    not use the network", not "everything you might ask for is present". A caller that
    believed it skipped the download and handed back a SHA anyway, and
    :func:`resolve_model` then wrote ``refs/<ref>`` for a snapshot that had never been
    staged — a cache entry that looks resolvable and is not.
    """
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


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _snapshot_dir(repo_id: str, sha: str) -> Path:
    """Return the ``snapshots/<sha>/`` directory for ``repo_id`` in the *current* HF cache."""
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "snapshots" / sha


def _snapshot_is_present(repo_id: str, sha: str) -> bool:
    """Whether ``repo_id``@``sha`` is actually staged in the HF cache this process will read.

    The question every fast path in :func:`ensure_hf_model` needs answered, and the one none
    of them used to ask. A recorded success lives under ``SENSELAB_CACHE``, which is
    deliberately decoupled from ``HF_HOME`` (see :func:`_senselab_cache_dir`) — so it says
    "this was downloaded", not "this was downloaded *into the cache you are about to read*".
    On a cluster those differ routinely: weights go to a shared group tree while coordination
    state stays per-user, and a job that switches ``HF_HOME`` replays a success recorded
    against a different tree entirely.

    Args:
        repo_id: The HF repository id.
        sha: A resolved 40-hex commit.

    Returns:
        ``True`` when the snapshot directory exists and is non-empty. Empty counts as absent:
        ``_point_ref_at`` and an interrupted download can both leave the directory behind
        with nothing in it, and returning ``True`` there reintroduces the same failure one
        level deeper.
    """
    snapshot = _snapshot_dir(repo_id, sha)
    try:
        return snapshot.is_dir() and any(snapshot.iterdir())
    except OSError:
        return False


def _get_cached_commit_hash(repo_id: str, revision: str = "main") -> str:
    """Resolve the immutable commit SHA of a locally cached model — no network.

    Resolution order (all filesystem-only):
      1. ``revision`` is already a full 40-hex commit SHA -> return it unchanged;
      2. the authoritative ``refs/<revision>`` pointer file -> its SHA (works
         regardless of which files the repo ships, e.g. SpeechBrain/pyannote/NeMo
         repos with no ``config.json``);
      3. a cached file's ``snapshots/<sha>/`` parent directory.

    Raises:
        RevisionResolutionError: If none of the above resolves ``revision`` to a
            SHA. Callers needing an immutable identity cannot be handed the
            mutable ``revision`` back as if it were one; the caller can retry
            against the Hub instead (see ``model_revision._resolve_uncached``).
    """
    if _SHA_RE.match(revision):
        return revision

    from huggingface_hub import constants, try_to_load_from_cache

    from senselab.utils.model_revision import RevisionResolutionError

    ref_file = Path(constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "refs" / revision
    try:
        if ref_file.is_file():
            sha = ref_file.read_text().strip()
            if _SHA_RE.match(sha):
                return sha
    except Exception:
        pass

    for filename in ("config.json", "hyperparams.yaml", "preprocessor_config.json", "model.safetensors"):
        try:
            result = try_to_load_from_cache(repo_id=repo_id, filename=filename, revision=revision)
            if isinstance(result, str):
                # Path looks like: .../snapshots/<commit_hash>/<filename>
                return Path(result).parent.name
        except Exception:
            continue

    # Returning `revision` here would hand a mutable ref to callers that will
    # record it as a commit -- provenance that is confidently wrong, which is
    # worse than none. Refuse instead; the caller can go to the Hub.
    raise RevisionResolutionError(
        f"{repo_id}@{revision} is not resolvable from the local cache "
        f"(no refs/{revision} pointer and no snapshot directory)."
    )


# A cached "not found" expires; a cached success does not. HuggingFace returns a plain 404 for a
# private repo the caller cannot see -- deliberately, so repo existence does not leak -- which makes
# "does not exist" and "exists but you lack access" indistinguishable at the API. Caching either as
# permanent means an access change never takes effect: measured on ORCD, a repo that 404'd under a
# narrowly-scoped token stayed "missing" on that host after the token was re-scoped AND the repo was
# made public, because every later call read this file instead of asking again. GatedRepoError is
# already exempt from caching, but that only covers gated repos that announce themselves.
#
# One hour: long enough to still absorb a retry storm across a job array, short enough that fixing a
# token or flipping a repo public is not a mystery.
_NEGATIVE_CACHE_TTL_S = 3600.0


def _read_result_cache(repo_id: str, revision: str) -> Optional[dict]:
    """Read a cached validation/download result, expiring stale negative entries.

    Returns ``None`` for a "not found" entry older than :data:`_NEGATIVE_CACHE_TTL_S` so the caller
    re-asks the Hub, because such an entry may only mean "not visible to the token used then".
    """
    cache_file = _senselab_cache_dir() / f"{_safe_key(repo_id, revision)}.json"
    if not cache_file.is_file():
        return None
    try:
        payload = json.loads(cache_file.read_text())
    except Exception:
        return None
    if isinstance(payload, dict) and payload.get("status") == "error":
        try:
            if time.time() - cache_file.stat().st_mtime > _NEGATIVE_CACHE_TTL_S:
                logger.info(
                    "Ignoring a stale cached failure for %s@%s (older than %.0fs); re-checking the Hub",
                    repo_id,
                    revision,
                    _NEGATIVE_CACHE_TTL_S,
                )
                return None
        except OSError:
            return None
    return payload  # type: ignore[no-any-return]


def _write_result_cache(repo_id: str, revision: str, **data: object) -> None:
    """Write a validation/download result to the filesystem cache."""
    cache_file = _senselab_cache_dir() / f"{_safe_key(repo_id, revision)}.json"
    try:
        cache_file.write_text(json.dumps(data))
    except Exception:
        pass  # Best-effort; failure to cache is not fatal


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
    """Reconstruct an exception from a cached error result.

    ``huggingface_hub``'s ``HfHubHTTPError`` subclasses take ``response`` as a
    *required* keyword-only argument, so constructing them bare raises
    ``TypeError`` — which would surface to callers instead of the repo/revision
    error they catch. A synthetic 404 response stands in for the original HTTP
    response, which the cache does not retain. The response needs its ``request``
    set too: ``HfHubHTTPError`` reads ``response.request`` while building its
    message, and httpx raises ``RuntimeError`` for a response without one.
    """
    import httpx
    from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

    msg = cached.get("error_message", "")
    response = httpx.Response(
        status_code=404,
        request=httpx.Request("GET", "https://huggingface.co"),
    )
    if cached.get("error_type") == "RepositoryNotFoundError":
        return RepositoryNotFoundError(msg, response=response)
    return RevisionNotFoundError(msg, response=response)


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

    # Fast path 2: result cached from a prior process (success or definitive failure).
    # A recorded *success* is only usable if the weights are in the cache this process reads
    # -- see _snapshot_is_present for why the two can disagree. When they do, fall through and
    # download rather than returning a SHA whose snapshot is not here; the alternative is a
    # FileNotFoundError surfacing inside a subprocess worker, far from the cause.
    # A recorded *failure* still short-circuits: a 404 does not become truer or falser
    # depending on which HF cache is mounted.
    cached = _read_result_cache(repo_id, revision)
    if cached is not None:
        if cached.get("status") == "ok":
            recorded_sha = str(cached["commit_hash"])
            if _snapshot_is_present(repo_id, recorded_sha):
                return recorded_sha
        else:
            raise _cached_error(cached)

    # Slow path: acquire lock, re-check, then download. SharedFileLock derives the lock
    # filename by appending (never replacing) a fixed suffix onto this resource path (see
    # file_lock.py), so passing the bare _safe_key(...) here is safe even though repo_id
    # or revision can legitimately contain a dot: the derivation is injective, so no two
    # distinct (repo_id, revision) pairs can ever collide onto the same lock file.
    resource_path = _senselab_cache_dir() / _safe_key(repo_id, revision)
    lock = SharedFileLock(
        resource_path,
        timeout=_LOCK_POLL_TIMEOUT,
        heartbeat_interval=_LOCK_HEARTBEAT_INTERVAL,
    )
    while True:
        try:
            lock.__enter__()
            break
        except TimeoutError:
            # Reaching this proves the flock was held continuously for the whole
            # poll window, which SharedFileLock's own contract treats as proof of
            # a live holder (a crashed process's flock is kernel-released, so it
            # cannot hold continuously). SharedFileLock deliberately never retries
            # this internally -- retrying here, unboundedly, is what reproduces
            # _HeartbeatLock's old "still waiting" loop, so a live multi-GB
            # download is never abandoned mid-transfer.
            logger.info(
                "Still waiting for another process to resolve %s@%s (lock held for the last %.0fs)",
                repo_id,
                revision,
                _LOCK_POLL_TIMEOUT,
            )
            continue
    try:
        # Re-check after acquiring lock, under the same "is it actually here?" rule as above:
        # the holder we waited on may have recorded a success against a different HF cache.
        if is_hf_model_cached(repo_id, revision):
            return _get_cached_commit_hash(repo_id, revision)
        cached = _read_result_cache(repo_id, revision)
        if cached is not None:
            if cached.get("status") == "ok":
                recorded_sha = str(cached["commit_hash"])
                if _snapshot_is_present(repo_id, recorded_sha):
                    return recorded_sha
            else:
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
    finally:
        # __exit__ ignores its exc_info arguments (it never suppresses an exception),
        # so passing None here is equivalent to the (exc_type, exc, tb) a `with` block
        # would supply.
        lock.__exit__(None, None, None)
    # Should never reach here, but satisfy mypy
    raise RuntimeError("Unreachable")  # pragma: no cover


def resolve_model(repo_id: str, revision: str = "main", *, token: Optional[str] = None) -> Tuple[str, Path]:
    """Resolve ``(repo_id, revision)`` to its immutable commit SHA + local snapshot dir.

    Ensures the model is present locally (download-once, cross-process, via
    :func:`ensure_hf_model`), then returns the resolved 40-hex commit SHA and the
    path to its ``snapshots/<sha>/`` directory. The SHA is derived from the cache
    (``refs/<ref>`` / snapshot dir), never a mutable ref, so callers can pin loads
    to it and skip the Hub version check that rate-limits (429) under parallelism.

    Loaders with a ``revision`` parameter should pass ``revision=sha,
    local_files_only=True``; loaders without one (SpeechBrain/pyannote/NeMo) should
    be pointed at the returned snapshot path.
    """
    from huggingface_hub import constants

    from senselab.utils.model_revision import resolve_revision

    # Resolve through the run manifest BEFORE staging, so the commit this loads is the commit the
    # run agreed on -- not whatever this host's refs/<ref> happens to point at.
    #
    # Passing `revision` straight to ensure_hf_model made the loader a second, independent
    # resolver: cache keys and provenance consult the manifest, while the load consulted the local
    # ref. On one warm node they agree, which is why nothing caught it. On the multi-node sweep the
    # manifest exists for, they diverge -- node A records repo@main -> SHA1 at t0, upstream pushes
    # SHA2, and node B (cold) six hours later keys and stamps SHA1 while ensure_hf_model downloads
    # and loads SHA2. The artifact then names a commit that did not run, which is the one outcome
    # worse than naming none.
    #
    # Resolving first also makes the manifest outrank the local cache, as the docs claim: a node
    # missing the pinned commit has is_hf_model_cached(repo, <sha>) return False and fetches
    # exactly that commit, rather than reusing an older snapshot its own ref still points at.
    sha = ensure_hf_model(repo_id, resolve_revision(repo_id, revision, token=token), token=token)
    repo_root = Path(constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}"
    snapshot_path = repo_root / "snapshots" / sha

    # Verified before the ref is written, not after. Callers take this path away into another
    # venv and open files under it, so a path to nothing surfaces as "[Errno 2] ...
    # atten_unet_vctk.toml" inside a worker -- a message that names the file rather than the
    # staging that never happened. Writing refs/<ref> first would additionally leave a pointer
    # to an absent snapshot behind, which is what made a later offline load report the model as
    # uncached-and-unreachable instead of simply missing.
    if not _snapshot_is_present(repo_id, sha):
        raise RuntimeError(
            f"{repo_id}@{sha} resolved but its snapshot is not in this HF cache "
            f"({constants.HF_HUB_CACHE}): expected {snapshot_path}. Staging did not complete here. "
            "If HF_HOME changed since this model was first resolved, senselab's own result cache "
            "(SENSELAB_CACHE) may still record the earlier download -- the two are deliberately "
            "separate trees."
        )

    _point_ref_at(repo_root, revision, sha)
    return sha, snapshot_path


def _point_ref_at(repo_root: Path, ref: str, sha: str) -> None:
    """Make ``refs/<ref>`` name ``sha``, so a loader that cannot take a revision still gets it.

    ``snapshot_download(revision=<sha>)`` writes ``snapshots/<sha>/`` and **no** ``refs/`` entry at
    all — refs exist only for named revisions. Measured, not assumed: staging a repo by SHA into an
    empty cache leaves no ``refs`` directory, and a subsequent bare
    ``AutoConfig.from_pretrained(repo)`` under ``HF_HUB_OFFLINE=1`` then fails outright with
    "couldn't find them in the cached files".

    That matters because several subprocess backends load bare — NeMo's
    ``Model.from_pretrained``, ``DiariZenPipeline.from_pretrained`` — since their loaders accept no
    revision argument. Resolving the ref before staging (which is what makes the run agree on one
    commit) would otherwise *break* them, because the pointer they rely on stops being written.

    Writing it here fixes that and upgrades those backends from "the parent staged the right commit
    but the worker reads whatever the ref says" to genuinely pinned: the ref now says the pinned
    commit.

    On a shared cache this mutates state other processes read, which deserves stating plainly — but
    it is strictly better than what it replaced. Ref-addressed staging already overwrote
    ``refs/<ref>`` on every call; it just wrote whatever upstream happened to be serving at that
    moment, rather than the commit this run pinned.

    A failure is ignored: on a group-owned tree the file may belong to another user, and the pinned
    load path (an explicit ``revision=<sha>``) does not depend on this pointer.
    """
    if _SHA_RE.match(ref):
        return
    refs_dir = repo_root / "refs"
    try:
        refs_dir.mkdir(parents=True, exist_ok=True)
        ref_file = refs_dir / ref
        current = ref_file.read_text().strip() if ref_file.is_file() else None
        if current != sha:
            ref_file.write_text(sha)
    except OSError as exc:
        logger.debug("Could not point refs/%s at %s under %s: %s", ref, sha, repo_root, exc)


def load_hf_resilient(
    loader: Callable[..., _T],
    *args: object,
    repo_id: str,
    revision: str = "main",
    pass_revision: bool = True,
    token: Optional[str] = None,
    **kwargs: object,
) -> _T:
    """Resolve+pin then load, so a cached model makes no Hub version-check call.

    Resolves ``(repo_id, revision)`` to an immutable SHA (download-once), then calls
    ``loader(*args, **kwargs)``. When ``pass_revision`` (the default, for loaders that
    accept a ``revision`` — ``transformers`` ``from_pretrained``/``pipeline``,
    ``sentence-transformers``), injects only ``revision=<sha>``: a full commit SHA
    triggers huggingface_hub's commit-hash shortcut, which returns cached files with
    **zero** network (no HEAD) without needing ``local_files_only``. We deliberately do
    NOT inject ``local_files_only`` — ``transformers.pipeline`` routes unknown kwargs
    into the model's ``generate`` params and raises ``ValueError: model_kwargs are not
    used by the model: ['local_files_only']`` at inference.
    Set ``pass_revision=False`` for loaders without a ``revision`` argument and point
    them at :func:`resolve_model`'s snapshot path instead. Transient errors are retried.

    ``repo_id``/``revision``/``token`` identify the model for resolution and are not
    forwarded to ``loader`` unless injected as above.
    """

    def _call() -> _T:
        sha, _ = resolve_model(repo_id, revision, token=token)
        if pass_revision:
            kwargs.setdefault("revision", sha)
        # Forward the token to the loader too (not only to resolve_model): gated
        # repos / first-download paths need it at load time. setdefault so an
        # explicit token already in kwargs wins.
        if token is not None:
            kwargs.setdefault("token", token)
        return loader(*args, **kwargs)

    return retry_on_transient_error(_call)


_HF_OFFLINE_ENV_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def hf_subprocess_env(
    repo_id: str,
    revision: str = "main",
    *,
    also: Optional[Iterable[Tuple[str, str]]] = None,
    base_env: Optional[dict] = None,
    token: Optional[str] = None,
) -> dict:
    """Build an environment for a subprocess-venv worker that loads HF model(s) offline.

    Ensures every referenced model is present locally (download-once, cross-process,
    via :func:`resolve_model`), then returns a copy of ``base_env`` (default
    ``os.environ``) with ``HF_HUB_OFFLINE`` / ``TRANSFORMERS_OFFLINE`` = ``"1"`` so the
    child's ``from_pretrained`` loads from the local cache with **no** Hub version
    check — the source of 429 storms under many parallel jobs. Use ``also`` for
    workers that load more than one model (e.g. Qwen3-ASR + its forced-aligner
    companion).

    The offline flag is set only when **all** referenced models are cacheable; if any
    cannot be staged, the env is returned unchanged so the child can still download it
    online. This is the subprocess analogue of :func:`load_hf_resilient`: unlike an
    in-process env toggle (a no-op, since huggingface_hub freezes offline mode at
    import), the child imports fresh with the flag already set, so it is honored.

    The returned env also carries ``SENSELAB_RUN_ID``, so the child process joins this
    run's resolution manifest (see ``model_revision.py``) instead of starting its own.
    Without this, a worker that resolves a ref itself would do so against an empty
    manifest and could pin to a different commit than the rest of the run if upstream
    moved between the two resolutions.
    """
    env = dict(os.environ if base_env is None else base_env)
    # Set unconditionally, before either return path below: a worker that falls back to
    # online loading (the early `return env` on a staging failure) must still inherit the
    # parent's run identity, not just the happy path that reaches the offline flags.
    from senselab.utils.model_revision import run_id

    env["SENSELAB_RUN_ID"] = run_id()
    repos: list[Tuple[str, str]] = [(str(repo_id), revision), *(also or [])]
    for rid, rev in repos:
        try:
            resolve_model(rid, rev, token=token)
        except Exception as exc:
            # A model is missing/undownloadable -> let the child try online instead
            # of failing outright, but say so: silently dropping the offline flag
            # here means the child reverts to the exact per-call Hub version-check
            # path (the 429 source) this function exists to remove. `rid`/`rev` are
            # bound to the pair that actually failed here, inside the loop — reading
            # the loop variables after the loop exits would be correct only by
            # accident of `repos` being non-empty by construction.
            logger.warning(
                f"hf_subprocess_env: failed to stage {rid!r}@{rev!r} for offline use ({exc}); "
                "the worker will fall back to online Hub loading for all referenced repos."
            )
            return env
    for var in _HF_OFFLINE_ENV_VARS:
        env[var] = "1"
    return env
