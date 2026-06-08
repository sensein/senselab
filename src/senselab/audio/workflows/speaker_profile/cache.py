"""Shared content-addressable cache helpers (T007, FR-015 / research R1).

This module is the **library-side** of `analyze_audio`'s task cache. Today,
`scripts/analyze_audio.py` defines its own ``cache_key`` and ``wrapper_hash``
(sha256 of the script source). That keys cache entries to the calling script,
so a second consumer — e.g. a future ``scripts/build_speaker_profile.py`` — would
*miss* the cache on identical tasks just because the script source differs.

The fix is to key the "wrapper hash" to the **stable library modules** that
implement each task rather than to whichever script ran them. Both
``analyze_audio`` and ``build_speaker_profile`` then produce identical
``cache_key`` values for shared tasks (diarization, speaker embeddings, scene
classification) and reuse each other's entries.

For Phase 2 this module ships the helpers; wiring ``analyze_audio.py`` to call
``task_wrapper_hash`` happens later (when the second consumer actually exists),
because that wiring is the riskier touch and benefits from a real consumer.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
from typing import Any

import torch  # only needed for type hints / Audio waveform handling

from senselab.audio.data_structures import Audio

CACHE_SCHEMA_VERSION: int = 1
"""Schema version stamped into every cache entry's key payload.

Matches ``scripts/analyze_audio.py``'s ``_CACHE_SCHEMA_VERSION`` so entries
remain interoperable across the script and this library helper. Bump (or
reset) on any breaking change to cache key composition.
"""


# ──────────────────────────────────────────────────────────────────────────
# Task → implementing-module map.
#
# A "wrapper hash" derived from the library modules below means changing one of
# those modules invalidates the relevant cache entries — regardless of which
# script invoked the task. Keep this map in sync with the actual task surface.

_TASK_MODULES: dict[str, tuple[str, ...]] = {
    "speaker_embeddings": (
        "senselab.audio.tasks.speaker_embeddings.api",
        "senselab.audio.tasks.speaker_embeddings.speechbrain",
        "senselab.audio.tasks.speaker_embeddings.wavlm",
        # The per-window extraction / windowing orchestration also determines the
        # cached result, so a change to it must invalidate the cache (avoids the
        # under-invalidation risk of hashing only the backend modules).
        "senselab.audio.workflows.audio_analysis.embeddings",
    ),
    "diarization": ("senselab.audio.tasks.speaker_diarization.api",),
    "classification": ("senselab.audio.tasks.classification.api",),
    "features": ("senselab.audio.tasks.features_extraction.api",),
    "asr": ("senselab.audio.tasks.speech_to_text.api",),
}


# ──────────────────────────────────────────────────────────────────────────
# Public helpers


def senselab_version() -> str:
    """Return the installed senselab version, or ``"unknown"`` if metadata is missing."""
    try:
        return importlib.metadata.version("senselab")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def audio_signature(audio: Audio) -> str:
    """Deterministic sha256 of the audio waveform PCM + sampling rate.

    Matches ``scripts/analyze_audio.py``'s ``audio_signature`` so the two
    callers compute the same signature for identical-sounding audio
    regardless of on-disk format (WAV vs FLAC vs ...). Extra metadata
    (path, mtime, encoding) is intentionally excluded.
    """
    arr = audio.waveform.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(str(audio.sampling_rate).encode())
    h.update(b"|")
    h.update(str(arr.shape).encode())
    h.update(b"|")
    h.update(arr.tobytes())
    return h.hexdigest()


def task_wrapper_hash(task: str) -> str:
    """sha256 of the library modules that implement ``task``.

    Caller-agnostic — two different scripts invoking the same task get the
    same hash. Unknown tasks fall back to a sentinel keyed on the senselab
    version + task name (still caller-agnostic, but invalidates on senselab
    upgrade).
    """
    modules = _TASK_MODULES.get(task)
    if not modules:
        return hashlib.sha256(f"{senselab_version()}:{task}".encode()).hexdigest()
    h = hashlib.sha256()
    for mod_name in sorted(modules):
        h.update(mod_name.encode("utf-8"))
        h.update(b"|")
        try:
            spec = importlib.util.find_spec(mod_name)
            if spec is not None and spec.origin:
                h.update(Path(spec.origin).read_bytes())
            else:
                h.update(b"<unresolved>")
        except (OSError, ValueError, ImportError, ModuleNotFoundError):
            h.update(b"<error>")
        h.update(b"|")
    return h.hexdigest()


def _canonical_params(params: dict[str, Any]) -> str:
    """Stable JSON encoding of params for cache keying. Sorted, no whitespace."""
    return json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)


def cache_key(
    *,
    audio_sig: str,
    task: str,
    model_id: str | None,
    params: dict[str, Any],
    wrapper_hash: str | None = None,
    senselab_ver: str | None = None,
    schema_version: int = CACHE_SCHEMA_VERSION,
) -> str:
    """Compute the deterministic cache key for one (audio, task, model, params) combo.

    Both ``wrapper_hash`` and ``senselab_ver`` default to caller-agnostic
    helpers (:func:`task_wrapper_hash` for the task, :func:`senselab_version`
    for the runtime), matching the FR-015 cross-stage reuse contract. Callers
    that want to pin a different wrapper hash (e.g. the legacy
    ``analyze_audio.py`` behavior) can pass an explicit value.
    """
    if wrapper_hash is None:
        wrapper_hash = task_wrapper_hash(task)
    if senselab_ver is None:
        senselab_ver = senselab_version()
    payload = {
        "schema": schema_version,
        "audio_signature": audio_sig,
        "task": task,
        "model": model_id,
        "params": params,
        "wrapper_hash": wrapper_hash,
        "senselab_version": senselab_ver,
    }
    return hashlib.sha256(_canonical_params(payload).encode()).hexdigest()


def cache_lookup(cache_dir: Path, key: str) -> dict[str, Any] | None:
    """Return the cached result dict for ``key``, or ``None`` on miss."""
    path = Path(cache_dir) / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def cache_store(cache_dir: Path, key: str, payload: dict[str, Any]) -> None:
    """Persist ``payload`` for ``key`` under ``cache_dir``."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# Re-export torch alias to satisfy mypy on the top-level import (used by
# downstream consumers that import this module for both cache + Audio).
__all__ = [
    "CACHE_SCHEMA_VERSION",
    "audio_signature",
    "cache_key",
    "cache_lookup",
    "cache_store",
    "senselab_version",
    "task_wrapper_hash",
    "torch",
]
