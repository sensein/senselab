"""Tests for shared cache helpers (T008, FR-015 / research R1).

Validates that the library-level cache helpers in
``senselab.audio.workflows.speaker_profile.cache`` are caller-agnostic — i.e.,
two processes invoking the same task on the same audio with the same params
produce identical ``cache_key`` values regardless of which script called them.

Note: the full cross-stage assertion (run build_speaker_profile then run
analyze_audio and assert ``cache: "hit"``) is wired in a follow-up patch once
the second consumer exists. This file covers the helpers' own contract:
determinism, sensitivity to task / model / params, and the lookup/store
round-trip.
"""

from __future__ import annotations

from pathlib import Path

import torch

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.speaker_profile.cache import (
    CACHE_SCHEMA_VERSION,
    audio_signature,
    cache_key,
    cache_lookup,
    cache_store,
    senselab_version,
    task_wrapper_hash,
)


def _stub_audio(seed: int = 0, samples: int = 16000) -> Audio:
    """Deterministic 1 s mono 16 kHz Audio (sine-like via fixed seed)."""
    g = torch.Generator().manual_seed(seed)
    waveform = torch.randn(1, samples, generator=g)
    return Audio(waveform=waveform, sampling_rate=16000)


# ──────────────────────────────────────────────────────────────────────────
# audio_signature


def test_audio_signature_is_deterministic() -> None:
    """Same waveform → same signature, repeatable across calls."""
    a = _stub_audio(seed=42)
    assert audio_signature(a) == audio_signature(a)


def test_audio_signature_distinguishes_content() -> None:
    """Different waveforms → different signatures."""
    a = _stub_audio(seed=1)
    b = _stub_audio(seed=2)
    assert audio_signature(a) != audio_signature(b)


def test_audio_signature_distinguishes_sampling_rate() -> None:
    """Same waveform array but different sampling_rate → different signature."""
    w = torch.zeros(1, 16000)
    a = Audio(waveform=w, sampling_rate=16000)
    b = Audio(waveform=w, sampling_rate=8000)
    assert audio_signature(a) != audio_signature(b)


# ──────────────────────────────────────────────────────────────────────────
# task_wrapper_hash


def test_task_wrapper_hash_is_deterministic() -> None:
    """The hash for a known task is stable across calls (same process)."""
    h1 = task_wrapper_hash("speaker_embeddings")
    h2 = task_wrapper_hash("speaker_embeddings")
    assert h1 == h2 and len(h1) == 64


def test_task_wrapper_hash_differs_across_tasks() -> None:
    """Different tasks hash different library code → different hashes."""
    a = task_wrapper_hash("speaker_embeddings")
    b = task_wrapper_hash("diarization")
    assert a != b


def test_task_wrapper_hash_unknown_task_is_stable_and_caller_agnostic() -> None:
    """Unknown tasks still return a stable, deterministic, 64-char sha256."""
    h1 = task_wrapper_hash("totally-fictional-task")
    h2 = task_wrapper_hash("totally-fictional-task")
    h3 = task_wrapper_hash("totally-fictional-task-2")
    assert h1 == h2 and len(h1) == 64
    assert h1 != h3


# ──────────────────────────────────────────────────────────────────────────
# cache_key


def test_cache_key_is_caller_agnostic_by_default() -> None:
    """Two callers passing the same (audio, task, model, params) get the same key.

    The default ``wrapper_hash`` comes from :func:`task_wrapper_hash`, which is
    library-derived and identical for any caller — this is the FR-015 invariant.
    """
    a = _stub_audio()
    sig = audio_signature(a)
    k1 = cache_key(audio_sig=sig, task="speaker_embeddings", model_id="m1", params={"device": "cpu"})
    k2 = cache_key(audio_sig=sig, task="speaker_embeddings", model_id="m1", params={"device": "cpu"})
    assert k1 == k2


def test_cache_key_sensitive_to_audio() -> None:
    """Different audio signatures → different cache keys."""
    a = _stub_audio(seed=1)
    b = _stub_audio(seed=2)
    k_a = cache_key(audio_sig=audio_signature(a), task="speaker_embeddings", model_id="m1", params={})
    k_b = cache_key(audio_sig=audio_signature(b), task="speaker_embeddings", model_id="m1", params={})
    assert k_a != k_b


def test_cache_key_sensitive_to_task_model_and_params() -> None:
    """Each of (task, model_id, params) materially changes the key."""
    sig = "deadbeef"
    base = cache_key(audio_sig=sig, task="speaker_embeddings", model_id="m1", params={"device": "cpu"})
    diff_task = cache_key(audio_sig=sig, task="diarization", model_id="m1", params={"device": "cpu"})
    diff_model = cache_key(audio_sig=sig, task="speaker_embeddings", model_id="m2", params={"device": "cpu"})
    diff_params = cache_key(audio_sig=sig, task="speaker_embeddings", model_id="m1", params={"device": "cuda"})
    assert base != diff_task
    assert base != diff_model
    assert base != diff_params


def test_cache_key_explicit_wrapper_hash_overrides_default() -> None:
    """A caller passing its own ``wrapper_hash`` opts out of the library default.

    This preserves the legacy ``analyze_audio.py`` behavior (per-script hash) as
    a one-line opt-out so the migration can be staged.
    """
    sig = "deadbeef"
    legacy = cache_key(audio_sig=sig, task="speaker_embeddings", model_id="m1", params={}, wrapper_hash="legacy")
    canonical = cache_key(audio_sig=sig, task="speaker_embeddings", model_id="m1", params={})
    assert legacy != canonical


# ──────────────────────────────────────────────────────────────────────────
# cache lookup / store round-trip


def test_cache_store_lookup_round_trip(tmp_path: Path) -> None:
    """A stored payload reads back identically from the same cache_dir/key."""
    sig = audio_signature(_stub_audio())
    key = cache_key(audio_sig=sig, task="speaker_embeddings", model_id="m1", params={})
    payload = {"result": [1, 2, 3], "status": "ok"}
    cache_store(tmp_path, key, payload)
    assert cache_lookup(tmp_path, key) == payload


def test_cache_lookup_miss_returns_none(tmp_path: Path) -> None:
    """Unknown key → ``None`` rather than an exception."""
    assert cache_lookup(tmp_path, "nonexistent") is None


def test_cache_store_creates_parent_dirs(tmp_path: Path) -> None:
    """Store creates the cache directory tree if missing."""
    nested = tmp_path / "a" / "b" / "c"
    cache_store(nested, "k", {"v": 1})
    assert (nested / "k.json").exists()


# ──────────────────────────────────────────────────────────────────────────
# Misc helpers


def test_cache_schema_version_is_int() -> None:
    """The schema-version constant is an int (stamped into every key payload)."""
    assert isinstance(CACHE_SCHEMA_VERSION, int) and CACHE_SCHEMA_VERSION >= 1


def test_senselab_version_returns_string() -> None:
    """The version helper always returns a string (``"unknown"`` when metadata is absent)."""
    v = senselab_version()
    assert isinstance(v, str) and v
