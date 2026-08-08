"""Tests for HF model-cache resolution helpers in ``senselab.utils.dependencies``.

MVP goal: stop the per-call Hub version check (the 429 source) by resolving a
requested ref to an immutable commit SHA once and loading pinned by that SHA
(``revision=<sha>`` — huggingface_hub's commit-hash shortcut then does no HEAD).
"""

import logging
from pathlib import Path

import huggingface_hub
import pytest

import senselab.utils.dependencies as dep
from senselab.utils.dependencies import (
    _get_cached_commit_hash,
    hf_subprocess_env,
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
    monkeypatch.setattr(dep, "is_hf_model_cached", lambda *a, **k: True)
    got_sha, got_path = resolve_model("org/model", "main")
    assert got_sha == sha
    assert Path(got_path) == snap


def test_load_hf_resilient_pins_sha_without_local_files_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """load_hf_resilient injects revision=<sha> ONLY (never local_files_only, which breaks pipeline)."""
    sha = "c" * 40
    monkeypatch.setattr(
        dep,
        "resolve_model",
        lambda repo_id, revision="main", **k: (sha, "/tmp/snap"),
    )
    captured: dict = {}

    def loader(**kwargs: object) -> str:
        captured.update(kwargs)
        return "MODEL"

    out = load_hf_resilient(loader, repo_id="org/model", revision="main", task="asr")
    assert out == "MODEL"
    assert captured["revision"] == sha
    # local_files_only must NOT be injected: transformers.pipeline forwards it to
    # generate and raises "model_kwargs are not used by the model: ['local_files_only']".
    assert "local_files_only" not in captured
    assert captured["task"] == "asr"


def test_load_hf_resilient_no_hub_call_when_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached model loads with zero HfApi.model_info (Hub version-check) calls."""
    sha = "d" * 40
    monkeypatch.setattr(dep, "is_hf_model_cached", lambda *a, **k: True)
    monkeypatch.setattr(dep, "_get_cached_commit_hash", lambda *a, **k: sha)

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("HfApi.model_info was called for a cached model")

    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", boom, raising=False)
    assert load_hf_resilient(lambda **k: "M", repo_id="org/model", revision="main") == "M"


# --------------------------------------------------------------------------- #
# hf_subprocess_env — offline env for subprocess-venv workers (fresh import)
# --------------------------------------------------------------------------- #


def test_hf_subprocess_env_sets_offline_when_all_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every referenced model is cacheable, the child env gets HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE = '1'."""
    monkeypatch.setattr(
        dep,
        "resolve_model",
        lambda repo_id, revision="main", **k: ("0" * 40, "/tmp/snap"),
    )
    env = hf_subprocess_env("Qwen/Qwen3-ASR-1.7B", "main", base_env={})
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["TRANSFORMERS_OFFLINE"] == "1"


def test_hf_subprocess_env_left_unchanged_when_uncacheable(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If a model cannot be staged, the env is returned unchanged so the child may still download online.

    It must also warn: the silent version of this fallback reverts to the per-call Hub version-check
    path, which is the 429 source ``hf_subprocess_env`` exists to remove. A future refactor that drops
    the ``logger.warning`` call would restore that silent revert with nothing to catch it, so the
    assertion checks the message content (names the failing repo, states the online-fallback
    consequence) rather than merely that some record was emitted — a record that just said "error"
    would pass a count-only check without proving the regression this delta guards against.
    """

    def boom(*a: object, **k: object) -> None:
        raise RuntimeError("cannot download")

    monkeypatch.setattr(dep, "resolve_model", boom)
    with caplog.at_level(logging.WARNING, logger="senselab"):
        env = hf_subprocess_env("org/model", "main", base_env={})
    assert "HF_HUB_OFFLINE" not in env
    assert "TRANSFORMERS_OFFLINE" not in env

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, "hf_subprocess_env must warn when a model can't be staged for offline use"
    message = warnings[0].getMessage()
    assert "org/model" in message, "warning must name the repo that failed to stage"
    assert "online" in message.lower(), "warning must state the fallback consequence (online Hub loading)"


def test_hf_subprocess_env_stages_companion_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """`also` companions (e.g. the Qwen forced aligner) are staged alongside the primary model."""
    staged: list = []

    def _stub_resolve(repo_id: str, revision: str = "main", **k: object) -> tuple:
        staged.append(repo_id)
        return ("0" * 40, "/tmp/snap")

    monkeypatch.setattr(dep, "resolve_model", _stub_resolve)
    env = hf_subprocess_env(
        "Qwen/Qwen3-ASR-1.7B", "main", also=[("Qwen/Qwen3-ForcedAligner-0.6B", "main")], base_env={}
    )
    assert staged == ["Qwen/Qwen3-ASR-1.7B", "Qwen/Qwen3-ForcedAligner-0.6B"]
    assert env["HF_HUB_OFFLINE"] == "1"
