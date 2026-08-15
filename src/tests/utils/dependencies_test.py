"""Tests for HF model-cache resolution helpers in ``senselab.utils.dependencies``.

MVP goal: stop the per-call Hub version check (the 429 source) by resolving a
requested ref to an immutable commit SHA once and loading pinned by that SHA
(``revision=<sha>`` — huggingface_hub's commit-hash shortcut then does no HEAD).
"""

import logging
import os
from pathlib import Path

import huggingface_hub
import pytest
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError

import senselab.utils.dependencies as dep
from senselab.utils.dependencies import (
    _get_cached_commit_hash,
    hf_subprocess_env,
    load_hf_resilient,
    resolve_model,
)
from senselab.utils.file_lock import lock_holder
from tests.utils.conftest import hub_error


def _fake_cache(monkeypatch: pytest.MonkeyPatch, root: Path, repo_id: str, revision: str, sha: str) -> Path:
    """Fabricate an HF cache with ``refs/<revision> -> <sha>`` and a snapshot dir; return the snapshot path."""
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(root))
    repo_dir = root / f"models--{repo_id.replace('/', '--')}"
    (repo_dir / "refs").mkdir(parents=True)
    (repo_dir / "refs" / revision).write_text(sha)
    snap = repo_dir / "snapshots" / sha
    snap.mkdir(parents=True)
    # A real snapshot is never empty, and an empty one is exactly what an interrupted
    # download leaves behind — `_snapshot_is_present` treats it as absent on purpose, so a
    # fixture that omits this would be fabricating a state the code is right to reject.
    (snap / "config.json").write_text("{}")
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


def test_cached_commit_hash_raises_rather_than_returning_a_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cold cache must raise, not hand back the mutable ref as if it were a SHA."""
    from senselab.utils.dependencies import _get_cached_commit_hash
    from senselab.utils.model_revision import RevisionResolutionError

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    with pytest.raises(RevisionResolutionError):
        _get_cached_commit_hash("org/never-downloaded", "main")


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


def test_load_hf_resilient_no_hub_call_when_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cached model loads with zero HfApi.model_info (Hub version-check) calls.

    The cache is fabricated on disk rather than only stubbed: "cached" now has to be true of
    the filesystem, since that is the disagreement — a stub saying yes while the snapshot is
    absent — that let a ref be written for weights that were never staged.
    """
    sha = "d" * 40
    _fake_cache(monkeypatch, tmp_path, "org/model", "main", sha)
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


def test_senselab_cache_dir_honors_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SENSELAB_CACHE, when set, is used verbatim as the cache directory."""
    custom = tmp_path / "custom_cache"
    monkeypatch.setenv("SENSELAB_CACHE", str(custom))
    result = dep._senselab_cache_dir()
    assert result == custom


def test_senselab_cache_dir_defaults_under_senselab_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Absent SENSELAB_CACHE, the cache lives under ~/.cache/senselab, not ~/.cache/huggingface."""
    monkeypatch.delenv("SENSELAB_CACHE", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    result = dep._senselab_cache_dir()
    assert result == tmp_path / ".cache" / "senselab" / "hf"
    assert "huggingface" not in str(result)


def test_senselab_cache_dir_ignores_hf_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting HF_HOME to a shared tree must not relocate senselab's own cache.

    This is the regression guard: on a shared HPC cluster, HF_HOME is routinely pointed at a
    large group-writable tree so model weights are downloaded once and reused. This directory
    also holds per-process lock files (see resolve_model), so deriving it from HF_HOME would put
    those locks in a tree contended by every other user of the shared tree.
    """
    monkeypatch.delenv("SENSELAB_CACHE", raising=False)
    shared_hf_home = tmp_path / "shared_group_tree" / "huggingface"
    monkeypatch.setenv("HF_HOME", str(shared_hf_home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path / "home")
    result = dep._senselab_cache_dir()
    assert "shared_group_tree" not in str(result)
    assert result == tmp_path / "home" / ".cache" / "senselab" / "hf"


def test_senselab_cache_dir_created_on_first_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache directory is created if it does not already exist."""
    custom = tmp_path / "not_yet_created"
    monkeypatch.setenv("SENSELAB_CACHE", str(custom))
    assert not custom.exists()
    result = dep._senselab_cache_dir()
    assert result.is_dir()


def test_heartbeat_lock_class_removed() -> None:
    """Pre-alpha rename-and-replace: `_HeartbeatLock` must not survive alongside `SharedFileLock`."""
    assert not hasattr(dep, "_HeartbeatLock")


def test_ensure_hf_model_locks_via_shared_file_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The slow path in ensure_hf_model must acquire via SharedFileLock, not a bare filelock.FileLock.

    SharedFileLock is what stamps a JSON holder identity (user/host/pid) into the lock file
    while it is held; a bare `filelock.FileLock` never writes any content there. Reading that
    identity back from *inside* the locked `snapshot_download` call proves the real class is
    wired in as the thing serialising the download, not merely imported and unused.
    """
    monkeypatch.setattr(dep, "_senselab_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(dep, "is_hf_model_cached", lambda *a, **k: False)
    monkeypatch.setattr(dep, "_read_result_cache", lambda *a, **k: None)
    monkeypatch.setattr(dep, "_write_result_cache", lambda *a, **k: None)
    monkeypatch.setattr(dep, "_get_cached_commit_hash", lambda *a, **k: "f" * 40)
    monkeypatch.setattr("senselab.utils.data_structures.model.get_huggingface_token", lambda: None)

    captured: dict = {}

    def fake_snapshot_download(**kwargs: object) -> None:
        lock_path = tmp_path / f"{dep._safe_key('org/model', 'main')}.lock"
        captured["holder"] = lock_holder(lock_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    dep.ensure_hf_model("org/model", "main")

    assert captured["holder"] is not None, "no SharedFileLock holder identity was recorded during the download"
    assert captured["holder"]["pid"] == os.getpid()


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


# ── A recorded success is a fact about one HF cache, not a global one ──


def test_a_recorded_success_is_not_reused_when_the_snapshot_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The result cache and the weights live in different trees, so it can lie.

    ``_senselab_cache_dir`` is deliberately NOT derived from ``HF_HOME`` — the weights go to a
    group tree, the coordination state stays per-user. That means a success recorded while
    ``HF_HOME`` pointed at cache A gets replayed when it points at cache B, and the fast path
    returns a SHA for a snapshot that was never downloaded here. Measured on ORCD: a job left
    ``models--openai--whisper-base/refs/main`` written and no ``snapshots/`` directory at all,
    and the worker then failed offline with "couldn't find it in the cached files".
    """
    import huggingface_hub.constants as hf_constants

    sha = "c" * 40
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "empty-hf-cache"))
    monkeypatch.setattr(dep, "is_hf_model_cached", lambda *a, **k: False)
    monkeypatch.setattr(dep, "_read_result_cache", lambda *a, **k: {"status": "ok", "commit_hash": sha})
    monkeypatch.setattr(dep, "_write_result_cache", lambda *a, **k: None)
    monkeypatch.setattr(dep, "_senselab_cache_dir", lambda: tmp_path / "senselab")

    downloaded: list[str] = []

    def fake_snapshot_download(repo_id: str, **kwargs: object) -> str:
        downloaded.append(repo_id)
        snap = Path(hf_constants.HF_HUB_CACHE) / f"models--{repo_id.replace('/', '--')}" / "snapshots" / sha
        snap.mkdir(parents=True, exist_ok=True)
        return str(snap)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)

    dep.ensure_hf_model("org/model", sha)

    assert downloaded == ["org/model"], "a recorded success must not stand in for weights that are not here"


def test_resolve_model_refuses_to_return_a_snapshot_path_that_does_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Handing back a path to nothing turns a staging failure into a confusing FileNotFoundError.

    The caller is a worker in another venv that will open files under this path, so the error
    surfaces there — far from the code that failed to stage — as "[Errno 2] ... atten_unet_vctk.toml".
    """
    import huggingface_hub.constants as hf_constants

    sha = "d" * 40
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "hf"))
    monkeypatch.setattr("senselab.utils.model_revision.resolve_revision", lambda *a, **k: sha)
    monkeypatch.setattr(dep, "ensure_hf_model", lambda *a, **k: sha)

    with pytest.raises(RuntimeError, match="snapshot"):
        resolve_model("org/model", "main")


def test_offline_mode_does_not_make_every_model_report_as_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``HF_HUB_OFFLINE=1`` means "do not use the network", not "everything is present".

    Answering True for an absent model makes ``ensure_hf_model`` skip the download and hand back
    a SHA anyway, which is how a ref gets written for a snapshot that was never staged.
    """
    import huggingface_hub.constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(tmp_path / "empty"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert dep.is_hf_model_cached("org/definitely-not-here", "main") is False


def test_a_definitive_404_is_not_retried() -> None:
    """A missing repo is a verdict, not a blip: retrying it only delays the real error.

    ``RepositoryNotFoundError`` subclasses ``httpx.HTTPError`` -> ``OSError``, so it matches
    ``_TRANSIENT_EXCEPTIONS``, and it carries its status only on ``.response.status_code``. Before
    ``_http_status`` read that attribute, this classified as transient and every genuine 404 cost
    three attempts and 3s of backoff.
    """
    exc = hub_error(404, RepositoryNotFoundError)
    assert not hasattr(exc, "code") and not hasattr(exc, "status_code"), (
        "the status lives on .response.status_code alone -- that is the whole point of this test"
    )
    assert dep._is_transient(exc) is False


def test_a_gated_403_is_not_retried() -> None:
    """403 means the token lacks access; waiting does not grant it."""
    assert dep._is_transient(hub_error(403, GatedRepoError)) is False


def test_a_429_is_retried() -> None:
    """The one 4xx that is transient: the server asking for backoff is what backoff is for."""
    assert dep._is_transient(hub_error(429)) is True


def test_a_500_is_retried() -> None:
    """Server-side failures are the canonical retryable case."""
    assert dep._is_transient(hub_error(500)) is True
    assert dep._is_transient(hub_error(503)) is True


def test_connection_and_timeout_errors_are_retried() -> None:
    """The statusless cases, in the shapes each client in play actually raises them.

    ``httpx`` is huggingface_hub 1.x's transport and its transport errors descend from
    ``httpx.HTTPError``, not ``OSError``, so they matched nothing in ``_TRANSIENT_EXCEPTIONS``
    until it learned about them.
    """
    import httpx

    assert dep._is_transient(httpx.ConnectError("connection refused")) is True
    assert dep._is_transient(httpx.ReadTimeout("timed out")) is True
    assert dep._is_transient(ConnectionError("reset by peer")) is True
    assert dep._is_transient(TimeoutError("timed out")) is True


def test_a_non_network_error_is_never_retried() -> None:
    """A programming error must fail on the first attempt, not three seconds later."""
    assert dep._is_transient(ValueError("bad argument")) is False


def test_retry_stops_on_a_404_and_backs_off_on_a_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """The classification is only worth anything if ``retry_on_transient_error`` acts on it."""
    monkeypatch.setattr(dep.time, "sleep", lambda *_a, **_k: None)

    calls = {"n": 0}

    def _not_found() -> str:
        calls["n"] += 1
        raise hub_error(404, RepositoryNotFoundError)

    with pytest.raises(RepositoryNotFoundError):
        dep.retry_on_transient_error(_not_found)
    assert calls["n"] == 1, "a 404 must fail on the first attempt"

    calls["n"] = 0

    def _throttled_once() -> str:
        calls["n"] += 1
        if calls["n"] == 1:
            raise hub_error(429)
        return "ok"

    assert dep.retry_on_transient_error(_throttled_once) == "ok"
    assert calls["n"] == 2, "a 429 must be retried"
