"""Tests for HF model-cache resolution helpers in ``senselab.utils.dependencies``.

MVP goal: stop the per-call Hub version check (the 429 source) by resolving a
requested ref to an immutable commit SHA once and loading pinned by that SHA with
``local_files_only=True`` (huggingface_hub's commit-hash shortcut then does no HEAD).
"""

from pathlib import Path

import huggingface_hub
import pytest

from senselab.utils.dependencies import (
    _get_cached_commit_hash,
    load_hf_resilient,
    resolve_model,
)


def _fake_cache(monkeypatch: pytest.MonkeyPatch, root: Path, repo_id: str, revision: str, sha: str) -> Path:
    """Fabricate an HF cache with ``refs/<revision> -> <sha>`` and a snapshot dir; return the snapshot path."""
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(root))
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / revision).write_text(sha)
    snap = repo_dir / "snapshots" / sha
    snap.mkdir(parents=True)
    return snap


def test_get_cached_commit_hash_resolves_real_sha_from_refs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """_get_cached_commit_hash returns the commit SHA from refs/<ref>, not the mutable branch name."""
    sha = "a" * 40
    _fake_cache(monkeypatch, tmp_path, "org/model", "main", sha)
    assert _get_cached_commit_hash("org/model", "main") == sha


def test_sha_revision_is_passed_through(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A full-SHA revision is returned as-is (already immutable)."""
    sha = "e" * 40
    _fake_cache(monkeypatch, tmp_path, "org/model", "main", "f" * 40)
    assert _get_cached_commit_hash("org/model", sha) == sha


def test_resolve_model_returns_sha_and_snapshot_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """resolve_model returns the immutable SHA + local snapshot dir without re-downloading a cached model."""
    sha = "b" * 40
    snap = _fake_cache(monkeypatch, tmp_path, "org/model", "main", sha)
    (snap / "config.json").write_text("{}")
    monkeypatch.setattr("senselab.utils.dependencies.is_hf_model_cached", lambda *a, **k: True)
    got_sha, got_path = resolve_model("org/model", "main")
    assert got_sha == sha
    assert Path(got_path) == snap


def test_load_hf_resilient_pins_sha_and_local_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_hf_resilient injects revision=<sha> + local_files_only=True into the loader, preserving caller kwargs."""
    sha = "c" * 40
    monkeypatch.setattr(
        "senselab.utils.dependencies.resolve_model",
        lambda repo_id, revision="main", **k: (sha, "/tmp/snap"),
    )
    captured: dict = {}

    def loader(**kwargs: object) -> str:
        captured.update(kwargs)
        return "MODEL"

    out = load_hf_resilient(loader, repo_id="org/model", revision="main", task="asr")
    assert out == "MODEL"
    assert captured["revision"] == sha
    assert captured["local_files_only"] is True
    assert captured["task"] == "asr"


def test_load_hf_resilient_no_hub_call_when_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached model loads with zero HfApi.model_info (Hub version-check) calls."""
    sha = "d" * 40
    monkeypatch.setattr("senselab.utils.dependencies.is_hf_model_cached", lambda *a, **k: True)
    monkeypatch.setattr("senselab.utils.dependencies._get_cached_commit_hash", lambda *a, **k: sha)

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("HfApi.model_info was called for a cached model")

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", boom, raising=False)
    assert load_hf_resilient(lambda **k: "M", repo_id="org/model", revision="main") == "M"
