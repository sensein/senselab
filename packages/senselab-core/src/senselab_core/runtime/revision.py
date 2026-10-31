"""Run-scoped resolution of HuggingFace refs to immutable commit SHAs.

A cluster sweep is an array of jobs across nodes, each spawning subprocess venvs,
running over hours or days. If upstream pushes to ``main`` partway through, tasks
that resolve after the push get different weights from those before it — every
task recording its own SHA correctly, and the run as a whole quietly
inhomogeneous. Per-task provenance documents that split; it does not prevent it.

So a run resolves each ``(repo_id, ref)`` exactly once and every participant binds
to that answer. Entries are append-if-absent and immutable for the run's life;
that immutability is the entire guarantee.

This module imports nothing from ``senselab.utils.data_structures`` —
``dependencies.py`` imports it, and that module carries a circular-import
constraint.
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
    TEMPORARY_NAME,
    StorageDir,
    StorePolicy,
    committed_modes,
    path_component,
    shared_dir_mode,
)

logger = logging.getLogger("senselab")

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

MANIFEST_NAME = "resolutions@2.json"
"""The run manifest's file name; its `@<major>` is the format of its keys (`manifest_key`)."""

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

    Inherited from ``SENSELAB_RUN_ID`` when present, so a single Slurm submission
    that exports one value shares it across every node, task and subprocess venv.
    When absent, the Slurm job identity is used so a submission is self-consistent
    with no configuration required — ``SLURM_ARRAY_JOB_ID`` (shared by every task of
    an array) is preferred over per-task ``SLURM_JOB_ID`` so an N-task array reuses
    one ``runs/<id>/`` manifest dir instead of leaking N of them into an
    inode-quota'd cache root. Only a truly bare launch falls through to a fresh
    UUID4. Whatever is chosen is exported, so any subprocess senselab spawns inherits
    it through the environment.

    That shrinks the manifest-dir leak without closing it: an array goes from N dirs to 1, but a
    non-array job still leaves one dir per job and a bare launch is unchanged (a fresh UUID4 dir
    every time). Run dirs are small and cleanup remains manual, as ``audio_analysis/doc.md``
    documents.

    Two consequences of preferring the Slurm identity, both real behaviour changes:

    - **Same-allocation merging.** Every senselab invocation inside one allocation now shares a
      manifest, where previously each process minted its own identity. Entries are immutable for
      the run's life (see :func:`record_resolution`), so a second experiment later in the same
      batch script — or unrelated work in a long-lived ``salloc`` — is silently pinned to the SHAs
      the first one resolved, with no way to re-resolve. Setting ``SENSELAB_RUN_ID`` explicitly is
      the escape hatch: it outranks everything here, so give each experiment its own value when
      they must resolve independently.
    - **Job-id reuse (known edge, unmitigated).** Slurm ids wrap at ``MaxJobId`` and reset when a
      controller is reinstalled, and nothing garbage-collects ``runs/``. A recycled id can
      therefore land on a months-old ``runs/<jobid>/`` manifest and adopt its SHAs as
      authoritative — confidently wrong provenance, which the module docstring names as the
      outcome worse than no provenance at all. Low probability, and deliberately not handled here:
      the cheap mitigations (folding ``SLURM_CLUSTER_NAME`` into the id, or ageing a manifest out
      by mtime) each change what an id *means* and belong in their own change, measured.
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
    """Return the path to a run's resolution manifest.

    Raises:
        ValueError: `run` is not one path component.
    """
    return _cache_root() / "runs" / path_component(run or run_id()) / MANIFEST_NAME


def read_manifest(run: Optional[str] = None) -> dict[str, str]:
    """Return a run's recorded resolutions, or an empty mapping.

    A manifest that is missing, empty, or unparsable reads as empty rather than
    raising: a corrupt manifest must degrade to "resolve again", never to a crash
    that takes down every job in the run. Missing/empty is the expected steady
    state for a run's first resolution and is silent; a *non-empty* file that
    fails to parse is a materially worse signal (a truncated write, disk
    corruption) and is logged at warning so it doesn't disappear silently.
    """
    path = manifest_path(run)
    try:
        text = path.read_text()
    except OSError:
        return {}
    if not text.strip():
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Run manifest %s is non-empty but unparsable; treating as empty", path)
        return {}
    if not isinstance(payload, dict):
        return {}
    return {str(k): str(v) for k, v in payload.items()}


def record_resolution(repo_id: str, ref: str, sha: str, run: Optional[str] = None) -> str:
    """Record ``(repo_id, ref) -> sha`` for this run and return the binding SHA.

    Append-if-absent under a :class:`SharedFileLock`: two nodes resolving the same
    model concurrently must not lose one another's entries, and the loser of the
    race adopts the winner's SHA rather than overwriting it. The returned value is
    therefore the *authoritative* SHA for the run, which may not be ``sha``.

    ``runs/<run>/`` is reached through `StorageDir` and made with the `StorePolicy`'s
    `shared_dir_mode`; the manifest is replaced atomically with its committed file mode, whatever
    the umask. Temporaries a dead writer left beside it are removed under the lock.

    Raises:
        RevisionResolutionError: ``sha`` is not a 40-hex commit SHA.
        ValueError: ``run`` (or this run's id) is not one path component.
        WrongKindError: an object of the wrong kind at ``runs/`` or the run's directory cannot be moved aside.
    """
    # Reject a non-SHA at the door rather than only filtering it on read: an entry is immutable
    # for the run's life, so one bad write poisons every later participant that adopts it.
    if not _SHA_RE.fullmatch(sha):
        raise RevisionResolutionError(
            f"Refusing to record {sha!r} as the commit for {manifest_key(repo_id, ref)}: "
            "a run manifest entry must be a 40-hex commit SHA."
        )

    path = manifest_path(run)
    policy = StorePolicy.load_default()
    dir_mode = shared_dir_mode(policy)
    with StorageDir.open_root(_cache_root(), create=True, mode=dir_mode) as root:
        with root.walk("runs", path.parent.name, create=True, mode=dir_mode, aside=lambda: None) as run_dir:
            return _record_in(run_dir, path, manifest_key(repo_id, ref), sha, run, committed_modes(policy)[0])


def _record_in(run_dir: StorageDir, path: Path, key: str, sha: str, run: Optional[str], file_mode: int) -> str:
    """Append-if-absent `key -> sha` to the manifest at `path` in `run_dir`, under the manifest's lock."""
    lock = SharedFileLock(path, manage_dir_mode=False)
    while True:
        try:
            lock.__enter__()
            break
        except TimeoutError:
            # A timeout proves a live holder: a crashed process's flock is
            # kernel-released, so it cannot hold continuously through the window.
            logger.info("Waiting to record a resolution for %s in the run manifest", key)
            continue
    try:
        _remove_leftover_temporaries(run_dir)
        current = read_manifest(run)
        winner = current.get(key)
        # Adopt an existing entry only if it is a real commit; a non-SHA entry is the one replaced.
        if winner is not None and _SHA_RE.fullmatch(winner):
            return winner
        current[key] = sha
        run_dir.write_atomically(
            MANIFEST_NAME, json.dumps(current, indent=2, sort_keys=True).encode("utf-8"), mode=file_mode
        )
        return sha
    finally:
        lock.__exit__(None, None, None)


def _remove_leftover_temporaries(run_dir: StorageDir) -> None:
    """Remove the manifest's temporary files in `run_dir`; under the manifest's lock any there is a dead writer's."""
    prefix = f".{MANIFEST_NAME}."
    for name in run_dir.listdir():
        if name.startswith(prefix) and TEMPORARY_NAME.fullmatch(name):
            run_dir.remove_owned(name, pattern=TEMPORARY_NAME)


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

    Filesystem first, network only on a miss: re-checking the Hub on every
    resolution would reintroduce the 429 rate-limiting that ``load_hf_resilient``
    exists to avoid.
    """
    try:
        local = get_cached_commit_hash(repo_id, ref)
    except Exception as exc:  # noqa: BLE001 — any local-read failure just means "ask the Hub"
        # Never surfaces a wrong answer, but a *persistent* local problem (bad
        # HF_HUB_CACHE, permissions) would otherwise force every resolution onto
        # the network with zero diagnostic trail. debug, not warning: a cold
        # cache is the normal first-resolution case, not an anomaly.
        logger.debug("Local cache lookup failed for %s@%s: %s", repo_id, ref, exc)
        local = None
    if local and _SHA_RE.fullmatch(local):
        return local

    try:
        from huggingface_hub import HfApi

        # The mandatory Hub round-trip on the cold-cache path, and the *first* network call a
        # cold run makes: resolve_model calls resolve_revision before ensure_hf_model. Wrap it in
        # the same transient-retry every other Hub call uses, so a rate-limit burst during a
        # parallel batch is survived rather than fatal at cache-key computation. It also has to be
        # retried here rather than left to load_hf_resilient's outer retry, because the wrapper
        # below turns any failure into RevisionResolutionError -- a RuntimeError, which
        # _is_transient classifies as non-transient, so a 429 escaping this line would defeat that
        # outer retry as well.
        #
        # What is retried is _is_transient's judgement (retry.py): 429, 5xx and statusless
        # connection/timeout failures. A definitive 4xx -- 401, 403, 404 -- is not retried and
        # surfaces below as RevisionResolutionError on the first attempt. That last part only
        # became true when _is_transient learned to read exc.response.status_code, which is the
        # only place huggingface_hub records a status; before that every one of its errors read as
        # statusless and a genuine 404 cost three attempts and 3s of backoff.
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
    # Validate rather than trust: read_manifest coerces every value with str(), so a JSON null
    # becomes the string "None", and a hand-edit or a partial write on a filesystem where
    # os.replace is not atomic can leave anything at all here. An unvalidated entry would flow
    # straight into cache keys, provenance and worker payloads -- "confidently wrong" with the
    # loud-failure path bypassed, which is the one outcome this module exists to prevent. A
    # non-SHA entry is therefore treated as absent and re-resolved.
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
