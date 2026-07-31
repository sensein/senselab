"""Content-addressable caching for expensive model inference (T051).

Lifted out of ``scripts/analyze_audio.py`` so the cache contract is importable,
unit-testable, and reusable by the adaptive loop rather than living in a 2500-line
CLI script.

A cache entry is keyed on everything that can change the result:

    (schema version, audio signature, task, model id, params,
     code version, senselab version)

``code_version`` is a caller-supplied string identifying the *behavior* that
produced an entry. It deliberately replaced an earlier ``wrapper_hash`` that was
a sha256 of the CLI script's own source. Source hashing was the wrong tool:
editing a comment, a docstring, or letting ``ruff-format`` rotate a line
invalidated every cached model result, while the blast radius grew as more stages
shared one file. Callers now pass a coarse, hand-managed identifier (see
``STAGE_VERSIONS`` in ``workflows/audio_analysis/stage_context.py``) whose
counterpart obligation is explicit: bump a stage's version when the stored shape
of its outcome changes.

``senselab_version`` still participates, and covers the larger surface — most
stages are thin pass-throughs to a ``tasks/`` API, so library-side changes are
what usually matter.

Cache keys are NOT stable across senselab versions and are not intended to be:
:data:`CACHE_SCHEMA_VERSION` is the deliberate global invalidation lever, and
:func:`sync_cache_with_schema_version` wipes stale entries automatically on every
host rather than requiring anyone to delete a directory by hand.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "align_cache_key",
    "audio_signature",
    "cache_key",
    "cache_lookup",
    "cache_store",
    "canonical_params",
    "run_alignment_cached",
    "run_cached",
    "run_task",
    "run_task_cached",
    "senselab_version",
    "serialize",
    "prune_unreachable_entries",
    "sync_cache_with_schema_version",
    "transcript_signature",
    "write_json",
]

CACHE_SCHEMA_VERSION = 6
"""Bump to invalidate every on-disk entry (see :func:`sync_cache_with_schema_version`).

Bumped 1 → 2 when ``wrapper_hash`` became ``code_version``: the key payload
changed shape, so every pre-existing entry is unreadable by construction. The
wipe is automatic on every host, not a manual ``rm -rf``.

Bumped 2 → 3 for the scene-quality level work: AST classification scores are no
longer softmaxed across all 527 AudioSet classes (the head is multi-label, and
the competition was structurally suppressing secondary background categories),
and the YAMNet path now applies gain before serializing its temp WAV. Both change
stored classification output, so every pre-existing entry is stale rather than
merely re-derivable.

Bumped 3 → 4 as that work continued: every stage outcome now carries the audio
variant and gain it was computed from, the noise-floor estimate gained a ``status``
field and moved to a 100 ms analysis frame (changing which bands exist at all), and
the background mask and source stages emit new shapes. Rather than reason about which
subset of entries survives, wipe — cache invalidation is free, and a stale entry that
*looks* readable is far more expensive than recomputing one.
"""


def senselab_version() -> str:
    """Return the installed senselab version, or ``"unknown"`` if metadata is missing."""
    try:
        return importlib.metadata.version("senselab")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def serialize(obj: Any) -> Any:  # noqa: ANN401 — recursive heterogeneous serializer
    """Convert senselab outputs (ScriptLine, tensor, etc.) to JSON-friendly types.

    ``torch`` is imported lazily so that merely deriving a cache key or writing a
    JSON sidecar does not pull in the ML stack — which in turn keeps
    ``stage_context`` importable without torch (there is a test asserting that).
    After the first call it is a ``sys.modules`` lookup.
    """
    import torch

    if isinstance(obj, dict):
        return {k: serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize(x) for x in obj]
    if isinstance(obj, torch.Tensor):
        return {
            "_tensor_shape": list(obj.shape),
            "_dtype": str(obj.dtype),
            "values": obj.detach().cpu().tolist(),
        }
    if hasattr(obj, "model_dump"):
        return serialize(obj.model_dump())
    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        return {k: serialize(v) for k, v in vars(obj).items() if not k.startswith("_")}
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def canonical_params(params: dict[str, Any]) -> str:
    """Stable JSON encoding of params for cache keying. Sorted, no whitespace."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


@runtime_checkable
class _HasWaveform(Protocol):
    """Structural type for :func:`audio_signature`.

    Typed structurally rather than as ``senselab.audio.data_structures.Audio``
    on purpose: ``utils/`` must not import from ``audio/`` (that would invert the
    dependency direction and risk an import cycle), and it lets callers pass any
    waveform-carrying object — which the tests do.
    """

    @property
    def waveform(self) -> Any: ...  # noqa: ANN401 — a torch.Tensor in practice

    @property
    def sampling_rate(self) -> int: ...


def audio_signature(audio: _HasWaveform) -> str:
    """Return a deterministic sha256 of the audio waveform PCM + sampling rate.

    Two identical-sounding files produce the same signature regardless of
    their on-disk format (e.g., WAV vs FLAC) — what matters is the post-
    resample, post-downmix waveform that each task actually sees. Extra
    metadata (file path, mtime, encoding) is intentionally excluded.

    This is the join key between a run's ``summary.json``
    (``passes[label].audio_signature``) and each cache entry's
    ``provenance.audio_signature``; the adaptive loop indexes cached results on
    it, so the two must be produced by this one function.
    """
    arr = audio.waveform.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(audio.sampling_rate).encode())
    h.update(b"|")
    h.update(str(arr.shape).encode())
    h.update(b"|")
    h.update(arr.tobytes())
    return h.hexdigest()


def write_json(path: Path, payload: Any) -> None:  # noqa: ANN401 — heterogeneous senselab outputs
    """Write a JSON file with senselab-aware serialization.

    Same operation as :func:`cache_store` minus the key→filename convention;
    lives here because it shares :func:`serialize`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(serialize(payload), fh, indent=2, default=str)


def cache_key(
    *,
    audio_sig: str,
    task: str,
    model_id: str | None,
    params: dict[str, Any],
    code_version: str,
    senselab_ver: str,
) -> str:
    """Compute the deterministic cache key for one (audio, task, model, params) combo."""
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "audio_signature": audio_sig,
        "task": task,
        "model": model_id,
        "params": params,
        "code_version": code_version,
        "senselab_version": senselab_ver,
    }
    return hashlib.sha256(canonical_params(payload).encode()).hexdigest()


def align_cache_key(
    *,
    audio_sig: str,
    transcript_sha: str,
    language: str | None,
    aligner_model_id: str,
    aligner_params: dict[str, Any],
    code_version: str,
    senselab_ver: str,
) -> str:
    """Cache key for one (audio, transcript, language, aligner) alignment call.

    Independent from the ASR cache: an alignment cache hit replays prior
    timestamps without invoking the aligner; an ASR-cache miss + alignment-cache
    hit (or vice versa) is supported by construction.
    """
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "audio_signature": audio_sig,
        "task": "alignment",
        "transcript_sha": transcript_sha,
        "language": language,
        "aligner_model": aligner_model_id,
        "aligner_params": aligner_params,
        "code_version": code_version,
        "senselab_version": senselab_ver,
    }
    return hashlib.sha256(canonical_params(payload).encode()).hexdigest()


def transcript_signature(text: str) -> str:
    """sha256 of an ASR transcript — anchors an alignment outcome to its exact input.

    The alignment cache uses this as one of its keys: re-aligning the same
    transcript on the same audio with the same params returns the cached
    timestamps without re-loading the aligner model.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cache_lookup(cache_dir: Path, key: str) -> dict[str, Any] | None:
    """Return the cached result dict for ``key``, or ``None`` on miss.

    A corrupt or unreadable entry counts as a miss rather than an error — the
    caller recomputes and overwrites it.
    """
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def cache_store(cache_dir: Path, key: str, payload: dict[str, Any]) -> None:
    """Persist ``payload`` for ``key`` under the cache dir."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(json.dumps(serialize(payload), indent=2, default=str), encoding="utf-8")


def prune_unreachable_entries(cache_dir: Path, *, senselab_ver: str) -> int:
    """Delete cache entries no current key can ever hit; return how many were removed.

    ``senselab_version`` and ``code_version`` are *inside* the cache key, so a
    senselab release orphans every entry and a ``STAGE_VERSIONS`` bump orphans that
    stage's. Nothing previously reclaimed them: ``CACHE_SCHEMA_VERSION`` only wipes
    on a schema change, so the directory grew monotonically across releases and a
    cache that looked healthy could be entirely dead weight.

    An entry is unreachable when its recorded ``provenance.senselab_version``
    differs from the running one, or its ``provenance.code_version`` no longer
    matches the declared version for that task. Entries without provenance are
    kept — absence of evidence isn't evidence of staleness, and a hit on them is
    still correct.

    Args:
        cache_dir: The cache directory.
        senselab_ver: The running senselab version.

    Returns:
        Number of entries removed.
    """
    try:
        from senselab.audio.workflows.audio_analysis.stage_context import stage_code_version
    except ImportError:  # pragma: no cover — stage versions live in the audio workflow
        return 0

    removed = 0
    for entry in cache_dir.glob("*.json"):
        try:
            prov = (json.loads(entry.read_text(encoding="utf-8")) or {}).get("provenance") or {}
        except (json.JSONDecodeError, OSError):
            continue  # corrupt entries already read as a miss; leave them to be overwritten
        if not prov:
            continue
        recorded_ver = prov.get("senselab_version")
        stale = recorded_ver is not None and recorded_ver != senselab_ver
        if not stale:
            task = prov.get("task")
            recorded_code = prov.get("code_version")
            if task and recorded_code is not None:
                try:
                    stale = recorded_code != stage_code_version(str(task))
                except KeyError:
                    stale = True  # task no longer declares a version → unreachable
        if stale:
            try:
                entry.unlink()
                removed += 1
            except OSError:
                continue
    if removed:
        print(
            f"Cache: pruned {removed} unreachable entr{'y' if removed == 1 else 'ies'} "
            f"(senselab/stage version drift) in {cache_dir}",
            file=sys.stderr,
        )
    return removed


def sync_cache_with_schema_version(cache_dir: Path) -> None:
    """Keep the on-disk cache state and :data:`CACHE_SCHEMA_VERSION` in sync.

    The cache directory carries a ``.schema_version`` marker file. On each run:

    - If the directory is empty / missing the marker → the cache was just
      created (or manually cleared). Write the current schema version. No
      data wipe is needed because there's nothing to wipe.
    - If the marker exists and matches the current code version → keep cache.
    - If the marker exists but doesn't match → the code has bumped the
      schema since the cache was populated. Wipe all cache entries and
      rewrite the marker with the current version.

    Bidirectional invariant: clearing the cache resets the version to current
    automatically (since the marker is recreated); bumping the version in
    code wipes the cache automatically (since the marker mismatch triggers
    the wipe). The user never has to manually delete cache files when they
    edit the schema number.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    marker = cache_dir / ".schema_version"
    on_disk_version: int | None = None
    if marker.exists():
        try:
            on_disk_version = int(marker.read_text().strip())
        except (ValueError, OSError):
            on_disk_version = None

    # Has the cache been populated with non-marker entries?
    has_entries = any(p.name != ".schema_version" for p in cache_dir.iterdir())

    if on_disk_version == CACHE_SCHEMA_VERSION:
        prune_unreachable_entries(cache_dir, senselab_ver=senselab_version())
        return

    if on_disk_version is None and not has_entries:
        # Fresh / cleared cache. Write current version, no wipe needed.
        marker.write_text(str(CACHE_SCHEMA_VERSION))
        print(
            f"Cache: initialized {cache_dir} at schema_version={CACHE_SCHEMA_VERSION}",
            file=sys.stderr,
        )
        return

    # Mismatch — wipe and rewrite the marker.
    n_removed = 0
    for p in cache_dir.iterdir():
        if p.name == ".schema_version":
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
            n_removed += 1
        except OSError:
            continue
    marker.write_text(str(CACHE_SCHEMA_VERSION))
    print(
        f"Cache: schema_version {on_disk_version} → {CACHE_SCHEMA_VERSION}; "
        f"wiped {n_removed} stale entr{'y' if n_removed == 1 else 'ies'} in {cache_dir}",
        file=sys.stderr,
    )


# ── Task runners ──────────────────────────────────────────────────────


def run_task(
    name: str,
    fn: Any,  # noqa: ANN401 — generic dispatcher
    *args: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Run a task with timing + structured error capture.

    Never raises: a failing model becomes ``{"status": "failed", ...}`` with the
    traceback captured, so one broken backend can't abort an hours-long
    multi-model run.
    """
    print(f"  [{name}] running...", flush=True)
    started = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — diagnostic capture by design
        elapsed = time.perf_counter() - started
        print(f"  [{name}] FAILED in {elapsed:.1f}s: {exc}", flush=True)
        return {
            "status": "failed",
            "elapsed_s": round(elapsed, 3),
            "error": repr(exc),
            "traceback": traceback.format_exc(limit=5),
        }
    elapsed = time.perf_counter() - started
    print(f"  [{name}] ok in {elapsed:.1f}s", flush=True)
    return {"status": "ok", "elapsed_s": round(elapsed, 3), "result": result}


def run_cached(
    name: str,
    fn: Any,  # noqa: ANN401
    *args: Any,  # noqa: ANN401
    cache_dir: Path | None,
    cache_key_str: str,
    provenance: dict[str, Any],
    hit_label: str = "cache",
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Cache lookup → run → store, attaching provenance to fresh results.

    On a hit the stored outcome is returned with ``cache="hit"`` and the task is
    not invoked. On a miss the task runs and, **only if it succeeded**, the
    outcome is stored — failures stay retryable so a fixed backend or a senselab
    upgrade triggers a fresh attempt rather than replaying a cached error.
    ``cache_dir=None`` disables caching entirely (``cache="disabled"``).

    Args:
        name: Log label for this task.
        fn: The callable to invoke on a miss.
        *args: Positional arguments forwarded to ``fn``.
        cache_dir: Cache directory, or ``None`` to disable caching.
        cache_key_str: Precomputed key (see :func:`cache_key` / :func:`align_cache_key`).
        provenance: Recorded on fresh outcomes for reproducibility.
        hit_label: Wording used in the cache-hit log line.
        **kwargs: Keyword arguments forwarded to ``fn``.

    Returns:
        The task outcome dict, annotated with ``cache`` and ``cache_key``.
    """
    if cache_dir is not None:
        hit = cache_lookup(cache_dir, cache_key_str)
        if hit is not None:
            print(f"  [{name}] {hit_label} HIT ({cache_key_str[:12]}...)", flush=True)
            hit["cache"] = "hit"
            hit["cache_key"] = cache_key_str
            return hit
    outcome = run_task(name, fn, *args, **kwargs)
    outcome["provenance"] = provenance
    outcome["cache"] = "miss" if cache_dir is not None else "disabled"
    outcome["cache_key"] = cache_key_str
    if cache_dir is not None and outcome.get("status") == "ok":
        cache_store(cache_dir, cache_key_str, outcome)
    return outcome


def run_task_cached(
    name: str,
    fn: Any,  # noqa: ANN401
    *args: Any,  # noqa: ANN401
    cache_dir: Path | None,
    cache_key_str: str,
    provenance: dict[str, Any],
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Run a model task through the cache. Thin alias over :func:`run_cached`."""
    return run_cached(
        name,
        fn,
        *args,
        cache_dir=cache_dir,
        cache_key_str=cache_key_str,
        provenance=provenance,
        **kwargs,
    )


def run_alignment_cached(
    name: str,
    fn: Any,  # noqa: ANN401
    *args: Any,  # noqa: ANN401
    cache_dir: Path | None,
    cache_key_str: str,
    provenance: dict[str, Any],
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Run an alignment step through the cache.

    Control flow is identical to :func:`run_task_cached` — the two were literal
    duplicates before this consolidation. The distinction is semantic: the
    provenance carries alignment-specific fields (``transcript_sha``,
    ``language``, ``parent_asr_cache_key``) and the key came from
    :func:`align_cache_key`, keeping the alignment cache independent of the
    parent ASR cache. Kept as a separate name so call sites still read as
    alignment, and so the log line says so.
    """
    return run_cached(
        name,
        fn,
        *args,
        cache_dir=cache_dir,
        cache_key_str=cache_key_str,
        provenance=provenance,
        hit_label="alignment cache",
        **kwargs,
    )
