"""Cache-contract tests for `utils/tasks/cached_inference` (T051).

The golden digests below were captured from `scripts/analyze_audio.py` *before*
the cache layer moved out of it. They are the safety net for the extraction: if
key derivation drifts, every entry in `artifacts/analyze_audio_cache/` silently
becomes a miss and a full re-run costs hours of model time. Treat a failure here
as "the cache was invalidated", not "update the constant".
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from senselab.utils.tasks.cached_inference import (
    CACHE_SCHEMA_VERSION,
    align_cache_key,
    cache_key,
    cache_lookup,
    cache_store,
    canonical_params,
    run_alignment_cached,
    run_task,
    run_task_cached,
    senselab_version,
    serialize,
    sync_cache_with_schema_version,
    transcript_signature,
)

# Captured from the pre-refactor script (analyze_audio.py @ 88e812fc).
GOLDEN_CACHE_KEY = "2ad59bc61873cac6b9f5438012742dda370abab5733c8166f3a69a83463eae82"
GOLDEN_TRANSCRIPT_SIG = "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"


def test_cache_key_matches_pre_refactor_digest() -> None:
    """Key derivation is byte-identical to the script's — no cache invalidation."""
    assert (
        cache_key(
            audio_sig="sig123",
            task="diarization",
            model_id="pyannote/x",
            params={"b": 2, "a": 1},
            wrapper_hash="WH",
            senselab_ver="1.2.3",
        )
        == GOLDEN_CACHE_KEY
    )


def test_transcript_signature_matches_pre_refactor_digest() -> None:
    """Plain sha256 of the utf-8 text (the well-known 'hello world' digest)."""
    assert transcript_signature("hello world") == GOLDEN_TRANSCRIPT_SIG


def test_schema_version_unchanged() -> None:
    """Bumping this wipes every user's cache — it must be a deliberate act."""
    assert CACHE_SCHEMA_VERSION == 1


def test_param_order_does_not_change_the_key() -> None:
    """Params are canonicalized, so dict ordering can't fragment the cache."""
    a = cache_key(audio_sig="s", task="t", model_id="m", params={"a": 1, "b": 2}, wrapper_hash="w", senselab_ver="v")
    b = cache_key(audio_sig="s", task="t", model_id="m", params={"b": 2, "a": 1}, wrapper_hash="w", senselab_ver="v")
    assert a == b


@pytest.mark.parametrize(
    "field",
    ["audio_sig", "task", "model_id", "params", "wrapper_hash", "senselab_ver"],
)
def test_every_keyed_field_changes_the_key(field: str) -> None:
    """Each component genuinely participates — none is silently ignored."""
    base = {
        "audio_sig": "s",
        "task": "t",
        "model_id": "m",
        "params": {"a": 1},
        "wrapper_hash": "w",
        "senselab_ver": "v",
    }
    altered = dict(base)
    altered[field] = {"a": 2} if field == "params" else "CHANGED"
    assert cache_key(**base) != cache_key(**altered)  # type: ignore[arg-type]


def test_alignment_key_is_independent_of_the_task_key() -> None:
    """ASR and alignment caches must not collide (separable per the design)."""
    align = align_cache_key(
        audio_sig="s",
        transcript_sha="ts",
        language="en",
        aligner_model_id="mms",
        aligner_params={"x": 1},
        wrapper_hash="w",
        senselab_ver="v",
    )
    task = cache_key(
        audio_sig="s", task="alignment", model_id="mms", params={"x": 1}, wrapper_hash="w", senselab_ver="v"
    )
    assert align != task
    assert len(align) == 64


def test_alignment_key_tracks_the_transcript() -> None:
    """A different transcript must re-align rather than replay stale timestamps."""
    kwargs = {
        "audio_sig": "s",
        "language": "en",
        "aligner_model_id": "mms",
        "aligner_params": {},
        "wrapper_hash": "w",
        "senselab_ver": "v",
    }
    assert align_cache_key(transcript_sha="a", **kwargs) != align_cache_key(transcript_sha="b", **kwargs)  # type: ignore[arg-type]


# ── store / lookup round-trip ─────────────────────────────────────────


def test_store_then_lookup_round_trip(tmp_path: Path) -> None:
    """A stored payload comes back equal through the JSON round-trip."""
    cache_store(tmp_path, "k1", {"status": "ok", "result": [1, 2, 3]})
    assert cache_lookup(tmp_path, "k1") == {"status": "ok", "result": [1, 2, 3]}


def test_lookup_miss_returns_none(tmp_path: Path) -> None:
    """An absent key is a miss, not an error."""
    assert cache_lookup(tmp_path, "nope") is None


def test_corrupt_entry_counts_as_a_miss(tmp_path: Path) -> None:
    """Truncated JSON must degrade to recompute rather than crash a long run."""
    (tmp_path / "bad.json").write_text("{not json")
    assert cache_lookup(tmp_path, "bad") is None


def test_store_creates_the_cache_dir(tmp_path: Path) -> None:
    """First write into a fresh run dir shouldn't require pre-creation."""
    nested = tmp_path / "a" / "b"
    cache_store(nested, "k", {"v": 1})
    assert (nested / "k.json").exists()


def test_tensors_serialize_with_shape_and_dtype(tmp_path: Path) -> None:
    """Tensors round-trip as a shape/dtype/values envelope, not repr()."""
    cache_store(tmp_path, "t", {"emb": torch.ones(2, 2)})
    entry = cache_lookup(tmp_path, "t")
    assert entry is not None
    assert entry["emb"]["_tensor_shape"] == [2, 2]
    assert entry["emb"]["values"] == [[1.0, 1.0], [1.0, 1.0]]
    assert "float" in entry["emb"]["_dtype"]


def test_serialize_handles_pydantic_and_unknown_objects() -> None:
    """`model_dump` is preferred; genuinely opaque objects fall back to repr."""

    class _Dumpable:
        def model_dump(self) -> dict[str, int]:
            return {"a": 1}

    assert serialize(_Dumpable()) == {"a": 1}
    assert serialize({1, 2}) == repr({1, 2})  # sets aren't JSON-able → repr


def test_canonical_params_is_stable_and_compact() -> None:
    """Sorted keys, no whitespace — the form the digest is taken over."""
    assert canonical_params({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_senselab_version_returns_a_string() -> None:
    """Never raises, even when package metadata is unavailable."""
    assert isinstance(senselab_version(), str)


# ── schema-version sync ───────────────────────────────────────────────


def test_sync_initializes_a_fresh_cache_without_wiping(tmp_path: Path) -> None:
    """Fresh dir → marker written, nothing removed."""
    sync_cache_with_schema_version(tmp_path)
    assert (tmp_path / ".schema_version").read_text().strip() == str(CACHE_SCHEMA_VERSION)


def test_sync_keeps_entries_when_version_matches(tmp_path: Path) -> None:
    """A matching marker must not touch cached entries."""
    cache_store(tmp_path, "keep", {"v": 1})
    (tmp_path / ".schema_version").write_text(str(CACHE_SCHEMA_VERSION))
    sync_cache_with_schema_version(tmp_path)
    assert (tmp_path / "keep.json").exists()


def test_sync_wipes_entries_on_version_mismatch(tmp_path: Path) -> None:
    """A stale marker means the schema moved → entries are unusable, so wipe."""
    cache_store(tmp_path, "stale", {"v": 1})
    (tmp_path / ".schema_version").write_text(str(CACHE_SCHEMA_VERSION + 1))
    sync_cache_with_schema_version(tmp_path)
    assert not (tmp_path / "stale.json").exists()
    assert (tmp_path / ".schema_version").read_text().strip() == str(CACHE_SCHEMA_VERSION)


def test_sync_wipes_when_marker_is_unreadable_but_entries_exist(tmp_path: Path) -> None:
    """A garbage marker alongside data is treated as a mismatch, not as fresh."""
    cache_store(tmp_path, "orphan", {"v": 1})
    (tmp_path / ".schema_version").write_text("not-a-number")
    sync_cache_with_schema_version(tmp_path)
    assert not (tmp_path / "orphan.json").exists()


def test_sync_wipes_nested_directories(tmp_path: Path) -> None:
    """Directory-shaped entries are removed too (shutil path)."""
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "f.json").write_text("{}")
    (tmp_path / ".schema_version").write_text(str(CACHE_SCHEMA_VERSION + 1))
    sync_cache_with_schema_version(tmp_path)
    assert not (tmp_path / "subdir").exists()


def test_stored_entry_is_valid_json_on_disk(tmp_path: Path) -> None:
    """Entries stay human-inspectable — the cache is a debugging surface."""
    cache_store(tmp_path, "k", {"a": 1})
    assert json.loads((tmp_path / "k.json").read_text()) == {"a": 1}


# ── Cached task runners (T051 part 2) ─────────────────────────────────


def test_run_task_reports_ok_with_timing() -> None:
    """A successful call is wrapped with status/elapsed/result."""
    out = run_task("demo", lambda x: x * 2, 21)
    assert out["status"] == "ok"
    assert out["result"] == 42
    assert isinstance(out["elapsed_s"], float)


def test_run_task_captures_failure_without_raising() -> None:
    """Errors become structured diagnostics — one bad task can't abort a long run."""

    def boom() -> None:
        raise ValueError("nope")

    out = run_task("demo", boom)
    assert out["status"] == "failed"
    assert "nope" in out["error"]
    assert "ValueError" in out["traceback"]


def test_run_task_cached_miss_runs_and_stores(tmp_path: Path) -> None:
    """First call misses, executes, and persists the outcome with provenance."""
    calls = []

    def fn() -> str:
        calls.append(1)
        return "v"

    out = run_task_cached("t", fn, cache_dir=tmp_path, cache_key_str="k", provenance={"p": 1})
    assert out["cache"] == "miss" and out["result"] == "v" and out["provenance"] == {"p": 1}
    assert len(calls) == 1
    assert cache_lookup(tmp_path, "k") is not None


def test_run_task_cached_hit_skips_execution(tmp_path: Path) -> None:
    """A stored entry short-circuits the model call entirely — the whole point."""
    calls = []

    def fn() -> str:
        calls.append(1)
        return "v"

    run_task_cached("t", fn, cache_dir=tmp_path, cache_key_str="k", provenance={})
    out = run_task_cached("t", fn, cache_dir=tmp_path, cache_key_str="k", provenance={})
    assert out["cache"] == "hit"
    assert len(calls) == 1, "cache hit must not re-run the task"
    assert out["cache_key"] == "k"


def test_run_task_cached_disabled_never_stores(tmp_path: Path) -> None:
    """cache_dir=None runs every time and writes nothing."""
    out = run_task_cached("t", lambda: "v", cache_dir=None, cache_key_str="k", provenance={})
    assert out["cache"] == "disabled"
    assert not list(tmp_path.iterdir())


def test_failed_task_is_not_cached(tmp_path: Path) -> None:
    """Failures must stay retryable — a fixed aligner/upgrade should re-attempt."""

    def boom() -> None:
        raise RuntimeError("x")

    out = run_task_cached("t", boom, cache_dir=tmp_path, cache_key_str="k", provenance={})
    assert out["status"] == "failed"
    assert cache_lookup(tmp_path, "k") is None


def test_run_alignment_cached_shares_the_runner_semantics(tmp_path: Path) -> None:
    """Alignment differs only in provenance/keying, not control flow."""
    calls = []

    def fn() -> str:
        calls.append(1)
        return "aligned"

    first = run_alignment_cached("a", fn, cache_dir=tmp_path, cache_key_str="ak", provenance={"transcript_sha": "s"})
    second = run_alignment_cached("a", fn, cache_dir=tmp_path, cache_key_str="ak", provenance={"transcript_sha": "s"})
    assert first["cache"] == "miss" and second["cache"] == "hit"
    assert len(calls) == 1


def test_failed_alignment_is_not_cached(tmp_path: Path) -> None:
    """Explicit contract from the original docstring."""

    def boom() -> None:
        raise RuntimeError("aligner down")

    out = run_alignment_cached("a", boom, cache_dir=tmp_path, cache_key_str="ak", provenance={})
    assert out["status"] == "failed"
    assert cache_lookup(tmp_path, "ak") is None
