"""Run-scoped resolution of HuggingFace refs to immutable commit SHAs.

A run resolves each `(repo_id, ref)` once, in `2x/runs/<run id>/resolutions@2.json` under the cache root,
and every participant binds to that answer; entries are append-if-absent. `2x/` is the 2.x namespace: 2.x
reads and writes no manifest outside it. Rationale: `specs/20261030-140022-senselab-2x-api/adr/ADR-006-formats.md`.
"""

import contextlib
import contextvars
import functools
import json
import logging
import os
import re
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from senselab_worker.canonical import jcs_dumps
from senselab_worker.digest import COMMIT_HEX

from senselab_core.runtime.locks import Lease, LeaseUnavailable
from senselab_core.runtime.network import Network, require_online
from senselab_core.runtime.profiles import ExecutionProfile
from senselab_core.runtime.retry import RetryPolicy, retry_on_transient_error
from senselab_core.runtime.storage import (
    ASIDE_NAME,
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


NAMESPACE = "2x"
"""The directory under the cache root that holds every run manifest 2.x writes and reads."""
MANIFEST_NAME = "resolutions@2.json"
"""The run manifest's file name; its `@<major>` is the format of its keys (`manifest_key`)."""
MANIFEST = re.compile(re.escape(MANIFEST_NAME))
_MANIFEST_TEMPORARY = re.compile(re.escape(f".{MANIFEST_NAME}.") + r"[0-9]+\.[0-9]+\.[0-9a-f]{32}\.tmp")
"""The manifest's own temporaries, and not its lease's, which a live taker may be writing."""
MANIFEST_LIMIT = 1 << 24
"""The most bytes a run manifest is read to."""
RECORD_ATTEMPTS = 3
"""How many times `record_resolution` takes the manifest's lease before a displaced holder gives up."""


class RevisionResolutionError(RuntimeError):
    """A ref could not be resolved to an immutable commit SHA.

    Raised rather than falling back to the ref: a result whose provenance says
    "unknown commit" is worth less than no result, and it would still be cached
    under a key that cannot distinguish it from any other commit.
    """


def manifest_key(repo_id: str, ref: str) -> str:
    """Return the manifest key for a ``(repo_id, ref)`` pair: the JCS text of ``[repo_id, ref]``."""
    return jcs_dumps([repo_id, ref]).decode()


def run_component(run: str) -> str:
    """Return `run` if it can name a run directory: one path component that is not a set-aside name (`ASIDE_NAME`).

    Raises:
        ValueError: otherwise.
    """
    if ASIDE_NAME.fullmatch(path_component(run)):
        raise ValueError(f"run id {run!r} has the form of a set-aside name (32 lowercase hex); use a dashed uuid")
    return run


def choose_attempt_id(environ: Mapping[str, str]) -> str:
    """Return the attempt id `environ` names: `SENSELAB_RUN_ID`, else `SLURM_ARRAY_JOB_ID`, `SLURM_JOB_ID` or a UUID4.

    Raises:
        ValueError: the chosen id is refused by `run_component`.
    """
    chosen = (
        environ.get("SENSELAB_RUN_ID")
        or environ.get("SLURM_ARRAY_JOB_ID")
        or environ.get("SLURM_JOB_ID")
        or str(uuid.uuid4())
    )
    try:
        return run_component(chosen)
    except ValueError as err:
        raise ValueError(f"attempt id from SENSELAB_RUN_ID or Slurm: {err}") from None


def fresh_attempt_id(environ: Mapping[str, str] | None = None) -> str:
    """Return the attempt id of one run or resume: the id `environ` pins, else a new UUID4 every call.

    Raises:
        ValueError: the id is refused by `run_component`.
    """
    return choose_attempt_id(os.environ if environ is None else environ)


_ATTEMPT: contextvars.ContextVar[str | None] = contextvars.ContextVar("senselab_attempt", default=None)


@contextlib.contextmanager
def attempt_scope(attempt: str) -> Iterator[str]:
    """Make `attempt` the id `attempt_id()` returns in this context until the block exits."""
    token = _ATTEMPT.set(run_component(attempt))
    try:
        yield attempt
    finally:
        _ATTEMPT.reset(token)


def attempt_id() -> str:
    """Return the id of the attempt running in this context (`attempt_scope`), else `process_attempt_id()`.

    Raises:
        ValueError: the chosen id is refused by `run_component`.
    """
    current = _ATTEMPT.get()
    return current if current is not None else process_attempt_id()


@functools.cache
def process_attempt_id() -> str:
    """Return this process's attempt id (`choose_attempt_id` of the environment), chosen once and exported.

    It is exported as `SENSELAB_RUN_ID` so subprocesses inherit it.

    Raises:
        ValueError: the chosen id is refused by `run_component`.
    """
    chosen = choose_attempt_id(os.environ)
    os.environ["SENSELAB_RUN_ID"] = chosen
    return chosen


def cache_root() -> Path:
    """Return senselab's cache root: ``SENSELAB_CACHE``, else ``~/.cache/senselab/hf``."""
    override = os.environ.get("SENSELAB_CACHE")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "senselab" / "hf"


def manifest_path(run: Optional[str] = None, *, root: Optional[Path] = None) -> Path:
    """Return the path to run ``run``'s resolution manifest (this run's when ``run`` is None).

    `root` is the cache root holding it, `cache_root()` when None.

    Raises:
        ValueError: `run` is refused by `run_component`.
    """
    base = cache_root() if root is None else Path(root)
    return base / NAMESPACE / "runs" / _run_name(run) / MANIFEST_NAME


def _open_runs(root: Optional[Path], policy: Optional[StorePolicy] = None) -> StorageDir:
    """Open `2x/runs/` under `root` (`cache_root()` when None), without following any symlink.

    With a `policy`, missing directories are made with its `shared_dir_mode`, and an object of the wrong kind
    at `2x/` or `runs/` is moved aside beside itself.
    """
    base = cache_root() if root is None else Path(root)
    if policy is None:
        with StorageDir.open_root(base) as top:
            return top.walk(NAMESPACE, "runs")
    mode = shared_dir_mode(policy)
    with StorageDir.open_root(base, create=True, mode=mode) as top:
        return top.walk(NAMESPACE, "runs", create=True, mode=mode, aside=lambda: None)


def _run_name(run: Optional[str]) -> str:
    """Return the directory name of run `run` (this run's when None), refused by `run_component` before any I/O."""
    return run_component(attempt_id() if run is None else run)


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
    except (WrongKindError, ValueError, PermissionError) as err:
        logger.warning("Run manifest %s cannot be read (%s); treating as empty", run_dir.path / MANIFEST_NAME, err)
        return {}
    return _parse_manifest(data, run_dir.path / MANIFEST_NAME)


def read_manifest(run: Optional[str] = None, *, root: Optional[Path] = None) -> dict[str, str]:
    """Return a run's recorded resolutions under `root` (`cache_root()` when None), or an empty mapping.

    The manifest is read through `2x/runs/<run>/`'s descriptor without following or blocking. A missing,
    empty, unreadable or unparsable manifest, or anything but a regular file at its name, reads as empty;
    a non-empty one that fails to parse is logged at warning.

    Raises:
        ValueError: `run` (or this run's id) is refused by `run_component`.
    """
    name = _run_name(run)
    try:
        with _open_runs(root) as runs:
            run_dir = runs.walk(name)
    except (OSError, WrongKindError):
        return {}
    with run_dir:
        try:
            return _read_in(run_dir)
        except OSError:
            return {}


def record_resolution(
    repo_id: str, ref: str, sha: str, run: Optional[str] = None, *, root: Optional[Path] = None
) -> str:
    """Record ``(repo_id, ref) -> sha`` for this run under `root` (`cache_root()` when None); return the binding SHA.

    Append-if-absent under a `Lease` on the manifest: a concurrent writer's entries are kept, and an
    existing commit SHA for the key wins over ``sha``. ``2x/runs/<run>/`` is reached through `StorageDir`
    and made with the `StorePolicy`'s `shared_dir_mode`; the manifest and its lease get the policy's file
    modes, whatever the umask. Under the lease, a dead writer's temporaries beside the manifest are removed,
    as are objects moved aside under ``2x/runs/`` past the quarantine age.

    Raises:
        RevisionResolutionError: ``sha`` is not a 40-hex commit SHA.
        ValueError: ``run`` (or this run's id) is refused by `run_component`.
        LeaseTimeout: the manifest's lease could not be taken within its timeout.
        LeaseUnavailable: the manifest's lease cannot be taken in the run's directory, what is at its name
            cannot be moved aside, or it was displaced on each of `RECORD_ATTEMPTS` takes.
        WrongKindError: an object of the wrong kind at ``2x/``, ``2x/runs/`` or the run's directory cannot be
            moved aside.
    """
    if not COMMIT_HEX.fullmatch(sha):
        raise RevisionResolutionError(
            f"Refusing to record {sha!r} as the commit for {manifest_key(repo_id, ref)}: "
            "a run manifest entry must be a 40-hex commit SHA."
        )
    policy = StorePolicy.load_default()
    name = _run_name(run)
    with (
        _open_runs(root, policy) as runs,
        runs.walk(name, create=True, mode=shared_dir_mode(policy), aside=lambda: None) as run_dir,
    ):
        return _record_in(runs, run_dir, manifest_key(repo_id, ref), sha, policy)


def _record_in(runs: StorageDir, run_dir: StorageDir, key: str, sha: str, policy: StorePolicy) -> str:
    """Append-if-absent `key -> sha` to the manifest in `run_dir`, under the manifest's lease.

    A holder whose lease was displaced before or after its write (whose temporary the new holder may have
    removed) takes the lease again and re-reads and merges the manifest, so its entry is kept beside the
    entries of the holder that displaced it, up to `RECORD_ATTEMPTS` takes.

    Raises:
        LeaseUnavailable: the lease was displaced on every one of `RECORD_ATTEMPTS` takes.
    """
    lease_mode = STAGING_FILE_MODE | (0o004 if policy.world_readable else 0)
    path = run_dir.path / MANIFEST_NAME
    for _ in range(RECORD_ATTEMPTS):
        with Lease(path, file_mode=lease_mode, directory=run_dir) as lease:
            _remove_leftovers(runs, run_dir, policy)
            current = _read_in(run_dir)
            winner = current.get(key)
            if winner is not None and COMMIT_HEX.fullmatch(winner):
                return winner
            if lease.owns():
                current[key] = sha
                try:
                    run_dir.write_atomically(
                        MANIFEST_NAME,
                        json.dumps(current, indent=2, sort_keys=True).encode("utf-8"),
                        grammar=MANIFEST,
                        mode=committed_modes(policy)[0],
                    )
                except FileNotFoundError:
                    if lease.owns():
                        raise
                if lease.owns():
                    return sha
        logger.warning("run manifest %s: lease displaced; re-reading and merging", path)
    raise LeaseUnavailable(
        path.with_name(MANIFEST_NAME + ".lease"),
        f"the lease on {path} was displaced on each of {RECORD_ATTEMPTS} takes; its stale_after is shorter than a "
        "manifest write, or another process keeps breaking it",
    )


def _remove_leftovers(runs: StorageDir, run_dir: StorageDir, policy: StorePolicy) -> None:
    """Remove dead writers' leftovers under the manifest's lease.

    That is the manifest's temporaries in `run_dir` (any there under the lease is a dead writer's), and objects
    moved aside in `2x/runs/` or in `run_dir` past the quarantine age.
    """
    for name in run_dir.listdir():
        if _MANIFEST_TEMPORARY.fullmatch(name):
            run_dir.remove_owned(name, pattern=TEMPORARY_NAME)
    try:
        cutoff = run_dir.fs_now() - policy.quarantine_retention_s
    except (OSError, WrongKindError) as err:
        logger.warning("cannot read the clock of %s; not collecting: %s", run_dir.path, err)
        return
    for directory in (runs, run_dir):
        directory.remove_older(cutoff)


def get_cached_commit_hash(repo_id: str, revision: str = "main", *, cache_dir: Optional[Path] = None) -> str:
    """Resolve the immutable commit SHA of a locally cached model — no network.

    `cache_dir` is the HF hub cache to read, `huggingface_hub.constants.HF_HUB_CACHE` when None.
    Resolution order (all filesystem-only):
      0. ``revision`` holding ``/``, a backslash or ``..``, or equal to ``.``, is refused before any read;
      1. ``revision`` is already a full 40-hex commit SHA -> return it unchanged;
      2. the authoritative ``refs/<revision>`` pointer file -> its SHA (works
         regardless of which files the repo ships, e.g. SpeechBrain/pyannote/NeMo
         repos with no ``config.json``);
      3. a cached file's ``snapshots/<sha>/`` parent directory.

    Raises:
        RevisionResolutionError: If ``revision`` is refused, or none of the above resolves it to a
            SHA. Callers needing an immutable identity cannot be handed the
            mutable ``revision`` back as if it were one; the caller can retry
            against the Hub instead (see `Resolver`).
    """
    if any(part in revision for part in ("/", "\\", "..")) or revision in ("", "."):
        raise RevisionResolutionError(f"{repo_id}@{revision} names no single refs/ entry; the local cache is not read")
    if COMMIT_HEX.fullmatch(revision):
        return revision

    from huggingface_hub import constants, try_to_load_from_cache  # noqa: PLC0415 - keeps the Hub client out of import

    hub_cache = Path(cache_dir) if cache_dir is not None else Path(constants.HF_HUB_CACHE)
    ref_file = hub_cache / f"models--{repo_id.replace('/', '--')}" / "refs" / revision
    try:
        if ref_file.is_file():
            sha = ref_file.read_text().strip()
            if COMMIT_HEX.fullmatch(sha):
                return sha
    except Exception:
        pass

    for filename in ("config.json", "hyperparams.yaml", "preprocessor_config.json", "model.safetensors"):
        try:
            result = try_to_load_from_cache(repo_id=repo_id, filename=filename, revision=revision, cache_dir=hub_cache)
            if isinstance(result, str):
                # Path looks like: .../snapshots/<commit_hash>/<filename>
                return Path(result).parent.name
        except Exception:
            continue

    raise RevisionResolutionError(
        f"{repo_id}@{revision} is not resolvable from the local cache "
        f"(no refs/{revision} pointer and no snapshot directory)."
    )


def hub_commit(repo_id: str, ref: str, token: Optional[str]) -> object:
    """Return the commit SHA the Hub's `model_info` reports for `repo_id@ref`."""
    from huggingface_hub import HfApi  # noqa: PLC0415 - keeps the Hub client out of import

    return HfApi(token=token).model_info(repo_id=repo_id, revision=ref).sha


@dataclass(frozen=True)
class Resolver:
    """Binds `(repo_id, ref)` to an immutable commit SHA for a run.

    It answers from its memo, then from the run manifest under `cache_root`, then from the local HF cache
    at `cache_dir`, then from `hub`; a fresh answer is recorded in the manifest and the memo. Two resolvers
    share an answer only through a manifest they both read.
    """

    cache_dir: Optional[Path] = None
    """The HF hub cache to read (`get_cached_commit_hash`); `HF_HUB_CACHE` when None."""
    hub: Callable[[str, str, Optional[str]], object] = hub_commit
    """Asked for the commit of `(repo_id, ref, token)` when the local cache has none."""
    network: Network = "online"
    """When `offline`, `hub` is never asked; a ref the local cache cannot resolve raises `Offline`."""
    cache_root: Optional[Path] = None
    """The root of the run manifests (`manifest_path`); `cache_root()` when None."""
    retry: RetryPolicy = field(default_factory=lambda: ExecutionProfile.load().retry)
    """How `hub` is retried on a transient fault; the packaged `laptop` profile's by default."""
    memo: dict[str, str] = field(default_factory=dict, compare=False, repr=False)
    """This resolver's answers so far, by `manifest_key`."""

    def resolve(self, repo_id: str, ref: str = "main", token: Optional[str] = None) -> str:
        """Return the commit SHA this run uses for ``(repo_id, ref)``.

        A ref that is already a full SHA returns with no I/O. A manifest entry that is not a SHA is
        logged and resolved again.

        Raises:
            RevisionResolutionError: no 40-hex commit SHA can be obtained.
        """
        if COMMIT_HEX.fullmatch(ref):
            return ref
        key = manifest_key(repo_id, ref)
        memoized = self.memo.get(key)
        if memoized is not None:
            return memoized
        recorded = read_manifest(root=self.cache_root).get(key)
        if recorded is not None and COMMIT_HEX.fullmatch(recorded):
            self.memo[key] = recorded
            return recorded
        if recorded is not None:
            logger.warning(
                "Run manifest entry for %s is not a commit SHA (%r); re-resolving and ignoring it", key, recorded
            )
        binding = record_resolution(repo_id, ref, self.lookup(repo_id, ref, token), root=self.cache_root)
        self.memo[key] = binding
        return binding

    def lookup(self, repo_id: str, ref: str, token: Optional[str] = None) -> str:
        """Resolve `ref` from the local cache, else the Hub, retried under `self.retry`; record nothing.

        Raises:
            RevisionResolutionError: neither answers with a 40-hex commit SHA.
        """
        try:
            local: Optional[str] = get_cached_commit_hash(repo_id, ref, cache_dir=self.cache_dir)
        except Exception as exc:  # noqa: BLE001 — any local-read failure just means "ask the Hub"
            logger.debug("Local cache lookup failed for %s@%s: %s", repo_id, ref, exc)
            local = None
        if local and COMMIT_HEX.fullmatch(local):
            return local
        require_online(self.network, f"resolving {repo_id}@{ref} on the Hub")

        try:
            sha = retry_on_transient_error(lambda: self.hub(repo_id, ref, token), self.retry)
        except Exception as exc:
            raise RevisionResolutionError(
                f"Cannot resolve {repo_id}@{ref} to a commit SHA: {exc}. "
                "Refusing to load through a mutable ref — the result's provenance would name no commit."
            ) from exc
        if not sha or not COMMIT_HEX.fullmatch(str(sha)):
            raise RevisionResolutionError(f"Hub returned no usable commit SHA for {repo_id}@{ref} (got {sha!r})")
        return str(sha)


PROCESS_RESOLVER = Resolver()
"""The process's resolver: the default HF hub cache and cache root, and the process's memo."""


def resolve_revision(
    repo_id: str, ref: str = "main", *, token: Optional[str] = None, resolver: Resolver = PROCESS_RESOLVER
) -> str:
    """Return ``resolver.resolve(repo_id, ref, token)``, the commit SHA this run uses for ``(repo_id, ref)``.

    Raises:
        RevisionResolutionError: If no SHA can be obtained.
    """
    return resolver.resolve(repo_id, ref, token)
