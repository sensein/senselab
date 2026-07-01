"""Unit tests for the HuggingFace-resilient loading helpers in ``dependencies``.

Covers the offline-loading context, the subprocess env builder, and the
``load_hf_resilient`` wrapper. No network calls and no real model downloads:
``hf_local_files_only`` (the cache-ensuring step) is monkeypatched so the tests
exercise only the env-toggling / retry logic this module owns.
"""

import os

import pytest

from senselab.utils import dependencies
from senselab.utils.dependencies import (
    hf_offline_loading,
    hf_subprocess_env,
    load_hf_resilient,
    retry_on_transient_error,
)

_OFFLINE_VARS = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


@pytest.fixture
def cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend every model is already cached (no download, no network)."""
    monkeypatch.setattr(dependencies, "hf_local_files_only", lambda *a, **k: True)


@pytest.fixture
def not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend no model is cached / cannot be cached."""
    monkeypatch.setattr(dependencies, "hf_local_files_only", lambda *a, **k: False)


# ── hf_offline_loading ─────────────────────────────────────────────


def test_offline_loading_sets_and_restores_when_cached(cached: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inside the block the offline vars are "1"; afterwards the prior state is restored."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "preexisting")

    with hf_offline_loading("some/model") as engaged:
        assert engaged is True
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"

    # Unset var goes back to unset; preexisting value is restored verbatim.
    assert "HF_HUB_OFFLINE" not in os.environ
    assert os.environ["TRANSFORMERS_OFFLINE"] == "preexisting"


def test_offline_loading_noop_when_not_cached(not_cached: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """If the model can't be cached, env is untouched and the block runs online."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with hf_offline_loading("some/model") as engaged:
        assert engaged is False
        assert "HF_HUB_OFFLINE" not in os.environ
    assert "HF_HUB_OFFLINE" not in os.environ


def test_offline_loading_restores_on_exception(cached: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Env is restored even when the wrapped block raises."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(ValueError):
        with hf_offline_loading("some/model"):
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            raise ValueError("boom")
    assert "HF_HUB_OFFLINE" not in os.environ


def test_offline_loading_nested_keeps_env_until_outermost_exit(cached: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested blocks keep the offline vars set until the OUTERMOST block exits."""
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    with hf_offline_loading("model_a") as engaged_a:
        assert engaged_a is True
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        with hf_offline_loading("model_b") as engaged_b:
            assert engaged_b is True
            assert os.environ["HF_HUB_OFFLINE"] == "1"
        # Inner exit must NOT clear the flag while the outer block is still active.
        assert os.environ["HF_HUB_OFFLINE"] == "1"

    assert "HF_HUB_OFFLINE" not in os.environ


def test_offline_loading_concurrent_does_not_serialize(cached: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two cached loads can be inside the block simultaneously (no whole-load serialization).

    A rendezvous barrier only releases if both threads are inside their
    ``hf_offline_loading`` block at once. A design that holds a lock across the
    whole block would keep the second thread out until the first exits, so the
    barrier would time out (BrokenBarrierError).
    """
    import threading

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    barrier = threading.Barrier(2, timeout=5)
    errors: list = []

    def worker() -> None:
        try:
            with hf_offline_loading("some/model") as engaged:
                assert engaged is True
                assert os.environ["HF_HUB_OFFLINE"] == "1"
                barrier.wait()  # both threads must reach here together
        except Exception as exc:  # BrokenBarrierError if the loads serialized
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"concurrent cached loads were serialized: {errors!r}"
    assert "HF_HUB_OFFLINE" not in os.environ  # env cleared once the last holder exits


# ── hf_subprocess_env ──────────────────────────────────────────────


def test_subprocess_env_offline_when_all_cached(cached: None) -> None:
    """All referenced models cached → offline vars set in the returned env."""
    env = hf_subprocess_env("a/model", "main", also=[("b/aligner", "main")], base_env={"PATH": "/x"})
    assert env["PATH"] == "/x"  # base_env preserved
    for var in _OFFLINE_VARS:
        assert env[var] == "1"


def test_subprocess_env_online_when_any_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If any referenced model is uncached, offline vars are NOT set (child may download)."""
    # First repo cached, companion not.
    calls = {"a/model": True, "b/aligner": False}
    monkeypatch.setattr(dependencies, "hf_local_files_only", lambda rid, rev="main": calls[rid])
    env = hf_subprocess_env("a/model", "main", also=[("b/aligner", "main")], base_env={})
    for var in _OFFLINE_VARS:
        assert var not in env


def test_subprocess_env_does_not_mutate_base(cached: None) -> None:
    """The passed base_env dict is copied, not mutated in place."""
    base = {"PATH": "/x"}
    hf_subprocess_env("a/model", base_env=base)
    assert base == {"PATH": "/x"}


# ── load_hf_resilient ──────────────────────────────────────────────


def test_load_hf_resilient_returns_loader_result(not_cached: None) -> None:
    """The wrapper forwards args/kwargs to the loader but consumes repo_id/revision."""
    sentinel = object()

    def loader(model_id: str, *, device: str) -> object:
        assert model_id == "openai/whisper-tiny"
        assert device == "cpu"
        return sentinel

    # repo_id/revision steer the cache step and are NOT passed to the loader;
    # the loader's own args go through *args / **kwargs.
    out = load_hf_resilient(loader, "openai/whisper-tiny", repo_id="openai/whisper-tiny", revision="main", device="cpu")
    assert out is sentinel


def test_load_hf_resilient_retries_transient(not_cached: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """A transient failure on first attempt is retried and then succeeds."""
    monkeypatch.setattr(dependencies.time, "sleep", lambda _s: None)  # don't actually back off
    attempts = {"n": 0}

    def flaky() -> str:
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("transient")
        return "ok"

    out = load_hf_resilient(flaky, repo_id="x/y")
    assert out == "ok"
    assert attempts["n"] == 2


def test_retry_reraises_non_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-transient errors (e.g. ValueError) are raised immediately, not retried."""
    attempts = {"n": 0}

    def boom() -> None:
        attempts["n"] += 1
        raise ValueError("permanent")

    with pytest.raises(ValueError):
        retry_on_transient_error(boom)
    assert attempts["n"] == 1
