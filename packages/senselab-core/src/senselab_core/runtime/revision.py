"""Run-scoped resolution of HuggingFace refs to immutable commit SHAs.

A run resolves each `(repo_id, ref)` once, in `runs/<run id>/resolutions@2.json` under the cache root,
and every participant binds to that answer; entries are append-if-absent. This module imports nothing from
`senselab.utils.data_structures`. Rationale: `specs/20260809-200421-hf-revision-pinning/design.md`.
"""

import json
import logging
import os
import re
import uuid
from pathlib import Path
from typing import Optional

from senselab_worker.canonical import jcs_dumps

from senselab_core.runtime.locks import SharedFileLock
from senselab_core.runtime.retry import retry_on_transient_error
from senselab_core.runtime.storage import (
    STAGING_FILE_MODE,
    TEMPORARY_NAME,
    StorageDir,
    StorePolicy,
    WrongKindError,
    committed_modes,
    path_component,
    shared_dir_mode,
)

logger = logging.getLogger("senselab")

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

MANIFEST_NAME = "resolutions@2.json"
"""The run manifest's file name; its `@<major>` is the format of its keys (`manifest_key`)."""
MANIFEST = re.compile(re.escape(MANIFEST_NAME))
LEGACY_TEMPORARY = re.compile(re.escape(MANIFEST_NAME) + r"\.tmp\.[0-9]+")
"""A temporary the manifest writer named before H9b: `resolutions@2.json.tmp.<pid>`."""
WRITTEN_NAMES = frozenset({MANIFEST})
"""The grammars of the file names this module writes (besides `storage` and `locks`)."""
MANIFEST_LIMIT = 1 << 24
"""The most bytes a run manifest is read to."""

# Process-local. _RUN_ID caches the run identity; _MEMO avoids re-reading the
# manifest from disk for a pair already resolved in this process.
_RUN_ID: Optional[str] = None
_MEMO: dict[str, str] = {}


class RevisionResolutionError(RuntimeError):
    """A ref could not be resolved to an immutable commit SHA.

    Raised rather than falling back to the ref: a result whose provenance says
    "unknown commit" is worth less than no result, and it would still be cached
    under a key that cannot distinguish it from any other commit.
    """


def manifest_key(repo_id: str, ref: str) -> str:
    """Return the manifest key for a ``(repo_id, ref)`` pair: the JCS text of ``[repo_id, ref]``."""
    return jcs_dumps([repo_id, ref]).decode()


def run_id() -> str:
    """Return this run's id, generating and exporting one if unset.

    The id is `SENSELAB_RUN_ID`, else `SLURM_ARRAY_JOB_ID`, else `SLURM_JOB_ID`, else a fresh UUID4, and is
    exported as `SENSELAB_RUN_ID` so subprocesses inherit it.

    Raises:
        ValueError: the chosen id is not one path component.
    """
    global _RUN_ID
    if _RUN_ID is None:
        chosen = (
            os.environ.get("SENSELAB_RUN_ID")
            or os.environ.get("SLURM_ARRAY_JOB_ID")
            or os.environ.get("SLURM_JOB_ID")
            or str(uuid.uuid4())
        )
        try:
            _RUN_ID = path_component(chosen)
        except ValueError:
            raise ValueError(f"run id {chosen!r} (from SENSELAB_RUN_ID or Slurm) must be one path component") from None
        os.environ["SENSELAB_RUN_ID"] = _RUN_ID
    return _RUN_ID


def _cache_root() -> Path:
    """Return senselab's cache root, honouring ``SENSELAB_CACHE``.

    Duplicated from ``dependencies._senselab_cache_dir`` rather than imported:
    ``dependencies`` imports *this* module, and importing back would close a cycle.
    """
    override = os.environ.get("SENSELAB_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "senselab" / "hf"


def cache_root() -> Path:
    """Return senselab's cache root, honouring ``SENSELAB_CACHE``."""
    return _cache_root()


def manifest_path(run: Optional[str] = None) -> Path:
    """Return the path to run ``run``'s resolution manifest (this run's when ``run`` is None).

    Raises:
        ValueError: `run` is not one path component.
    """
    return _cache_root() / "runs" / path_component(run_id() if run is None else run) / MANIFEST_NAME


def _run_dir(run: Optional[str], *, create: bool, policy: Optional[StorePolicy] = None) -> StorageDir:
    """Open `runs/<run>/` under the cache root (this run's when `run` is None), without following any symlink.

    With `create`, missing directories get the policy's `shared_dir_mode`, and an object of the wrong kind at
    `runs/` or the run's directory is moved aside beside itself.
    """
    name = manifest_path(run).parent.name
    mode = shared_dir_mode(policy) if policy is not None else None
    with StorageDir.open_root(_cache_root(), create=create, mode=mode) as root:
        if not create:
            return root.walk("runs", name)
        return root.walk("runs", name, create=True, mode=mode, aside=lambda: None)


def _parse_manifest(data: Optional[bytes], where: Path) -> dict[str, str]:
    if not data or not data.strip():
        return {}
    try:
        payload = json.loads(data.decode("utf-8"))
    except ValueError:
        logger.warning("Run manifest %s is non-empty but unparsable; treating as empty", where)
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): str(v) for k, v in payload.items()}


def _read_in(run_dir: StorageDir) -> dict[str, str]:
    try:
        data = run_dir.read_regular(MANIFEST_NAME, limit=MANIFEST_LIMIT)
    except (WrongKindError, ValueError) as err:
        logger.warning("Run manifest %s cannot be read (%s); treating as empty", run_dir.path / MANIFEST_NAME, err)
        return {}
    return _parse_manifest(data, run_dir.path / MANIFEST_NAME)


def read_manifest(run: Optional[str] = None) -> dict[str, str]:
    """Return a run's recorded resolutions, or an empty mapping.

    The manifest is read through `runs/<run>/`'s descriptor without following or blocking. A missing,
    empty, unreadable or unparsable manifest, or anything but a regular file at its name, reads as empty;
    a non-empty one that fails to parse is logged at warning.

    Raises:
        ValueError: `run` (or this run's id) is not one path component.
    """
    try:
        run_dir = _run_dir(run, create=False)
    except (OSError, WrongKindError):
        return {}
    with run_dir:
        try:
            return _read_in(run_dir)
        except OSError:
            return {}


def record_resolution(repo_id: str, ref: str, sha: str, run: Optional[str] = None) -> str:
    """Record ``(repo_id, ref) -> sha`` for this run and return the binding SHA.

    Append-if-absent under a `SharedFileLock` on the manifest: a concurrent writer's entries are kept,
    and an existing commit SHA for the key wins over ``sha``. ``runs/<run>/`` is reached through
    `StorageDir` and made with the `StorePolicy`'s `shared_dir_mode`; the manifest and the lock's files
    get the policy's file modes, whatever the umask. Under the lock, a dead writer's temporaries beside the
    manifest are removed, as are objects moved aside under ``runs/`` past the quarantine age.

    Raises:
        RevisionResolutionError: ``sha`` is not a 40-hex commit SHA.
        ValueError: ``run`` (or this run's id) is not one path component.
        LockTimeout: the manifest's lock could not be taken within its timeout.
        WrongKindError: an object of the wrong kind at ``runs/``, the run's directory or a lock name
            cannot be moved aside.
    """
    if not _SHA_RE.fullmatch(sha):
        raise RevisionResolutionError(
            f"Refusing to record {sha!r} as the commit for {manifest_key(repo_id, ref)}: "
            "a run manifest entry must be a 40-hex commit SHA."
        )
    policy = StorePolicy.load_default()
    with _run_dir(run, create=True, policy=policy) as run_dir:
        return _record_in(run_dir, manifest_key(repo_id, ref), sha, policy)


def _record_in(run_dir: StorageDir, key: str, sha: str, policy: StorePolicy) -> str:
    """Append-if-absent `key -> sha` to the manifest in `run_dir`, under the manifest's lock."""
    lock_mode = STAGING_FILE_MODE | (0o004 if policy.world_readable else 0)
    lock = SharedFileLock(run_dir.path / MANIFEST_NAME, manage_dir_mode=False, file_mode=lock_mode, directory=run_dir)
    with lock:
        _remove_leftovers(run_dir, policy)
        current = _read_in(run_dir)
        winner = current.get(key)
        if winner is not None and _SHA_RE.fullmatch(winner):
            return winner
        current[key] = sha
        run_dir.write_atomically(
            MANIFEST_NAME,
            json.dumps(current, indent=2, sort_keys=True).encode("utf-8"),
            grammar=MANIFEST,
            mode=committed_modes(policy)[0],
        )
        return sha


def _remove_leftovers(run_dir: StorageDir, policy: StorePolicy) -> None:
    """Remove dead writers' leftovers under the manifest's lock.

    That is the manifest's temporaries in `run_dir` (any there under the lock is a dead writer's), the
    temporaries writers before H9b left, and objects moved aside under `runs/` past the quarantine age.
    """
    prefix = f".{MANIFEST_NAME}."
    for name in run_dir.listdir():
        if name.startswith(prefix) and TEMPORARY_NAME.fullmatch(name):
            run_dir.remove_owned(name, pattern=TEMPORARY_NAME)
        elif LEGACY_TEMPORARY.fullmatch(name):
            run_dir.remove_owned(name, pattern=LEGACY_TEMPORARY)
    try:
        with StorageDir.open_root(_cache_root()) as root, root.walk("runs") as runs:
            runs.remove_older(run_dir.fs_now() - policy.quarantine_retention_s)
    except (OSError, WrongKindError) as err:
        logger.warning("cannot collect what was set aside under runs/: %s", err)


def get_cached_commit_hash(repo_id: str, revision: str = "main") -> str:
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
            against the Hub instead (see ``revision._resolve_uncached``).
    """
    if _SHA_RE.fullmatch(revision):
        return revision

    from huggingface_hub import constants, try_to_load_from_cache

    ref_file = Path(constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "refs" / revision
    try:
        if ref_file.is_file():
            sha = ref_file.read_text().strip()
            if _SHA_RE.fullmatch(sha):
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


def _resolve_uncached(repo_id: str, ref: str, token: Optional[str] = None) -> str:
    """Resolve a ref to a SHA without consulting the manifest or the memo.

    The local cache first, the Hub only on a miss, through `retry_on_transient_error`.
    """
    try:
        local = get_cached_commit_hash(repo_id, ref)
    except Exception as exc:  # noqa: BLE001 — any local-read failure just means "ask the Hub"
        logger.debug("Local cache lookup failed for %s@%s: %s", repo_id, ref, exc)
        local = None
    if local and _SHA_RE.fullmatch(local):
        return local

    try:
        from huggingface_hub import HfApi

        sha = retry_on_transient_error(lambda: HfApi(token=token).model_info(repo_id=repo_id, revision=ref).sha)
    except Exception as exc:
        raise RevisionResolutionError(
            f"Cannot resolve {repo_id}@{ref} to a commit SHA: {exc}. "
            "Refusing to load through a mutable ref — the result's provenance would name no commit."
        ) from exc
    if not sha or not _SHA_RE.fullmatch(str(sha)):
        raise RevisionResolutionError(f"Hub returned no usable commit SHA for {repo_id}@{ref} (got {sha!r})")
    return str(sha)


def resolve_revision(repo_id: str, ref: str = "main", *, token: Optional[str] = None) -> str:
    """Return the immutable commit SHA this run uses for ``(repo_id, ref)``.

    Order: a ref that is already a full SHA short-circuits with no I/O; then this
    process's memo; then the run manifest; then local cache and Hub. A freshly
    resolved SHA is recorded so the rest of the run follows it.

    Raises:
        RevisionResolutionError: If no SHA can be obtained.
    """
    if _SHA_RE.fullmatch(ref):
        return ref

    key = manifest_key(repo_id, ref)
    memoized = _MEMO.get(key)
    if memoized is not None:
        return memoized

    recorded = read_manifest().get(key)
    if recorded is not None and _SHA_RE.fullmatch(recorded):
        _MEMO[key] = recorded
        return recorded
    if recorded is not None:
        logger.warning(
            "Run manifest entry for %s is not a commit SHA (%r); re-resolving and ignoring it", key, recorded
        )

    sha = _resolve_uncached(repo_id, ref, token)
    binding = record_resolution(repo_id, ref, sha)
    _MEMO[key] = binding
    return binding
