"""Tests for run-scoped HF revision resolution."""

import multiprocessing as mp
import os
from pathlib import Path

import pytest

from senselab.utils.model_revision import (
    RevisionResolutionError,
    manifest_key,
    manifest_path,
    read_manifest,
    record_resolution,
    resolve_revision,
    run_id,
)

SHA_A = "a" * 40
SHA_B = "b" * 40


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point SENSELAB_CACHE at a temp dir and clear any inherited run id."""
    monkeypatch.setenv("SENSELAB_CACHE", str(tmp_path / "senselab"))
    monkeypatch.delenv("SENSELAB_RUN_ID", raising=False)
    import senselab.utils.model_revision as mr

    mr._RUN_ID = None
    mr._MEMO.clear()


def test_run_id_is_generated_once_and_exported() -> None:
    """A generated run id is stable within the process and exported to the environment."""
    first = run_id()
    second = run_id()
    assert first == second, "run_id must be stable within a process"
    assert os.environ["SENSELAB_RUN_ID"] == first, "run id must be exported so subprocesses inherit it"


def test_run_id_is_inherited_when_already_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pre-set SENSELAB_RUN_ID is reused rather than replaced with a fresh UUID."""
    monkeypatch.setenv("SENSELAB_RUN_ID", "job-12345")
    import senselab.utils.model_revision as mr

    mr._RUN_ID = None
    assert run_id() == "job-12345"


def test_a_full_sha_short_circuits_without_touching_the_manifest() -> None:
    """A ref that is already a full commit SHA resolves with no filesystem I/O."""
    assert resolve_revision("org/model", SHA_A) == SHA_A
    assert not manifest_path().exists(), "a 40-hex ref must resolve with no I/O at all"


def test_record_then_read_round_trips() -> None:
    """A recorded resolution is readable back from the manifest."""
    record_resolution("org/model", "main", SHA_A)
    assert read_manifest()[manifest_key("org/model", "main")] == SHA_A


def test_manifest_entries_are_immutable_for_the_run() -> None:
    """A second recording for the same key adopts the first SHA instead of overwriting it."""
    record_resolution("org/model", "main", SHA_A)
    returned = record_resolution("org/model", "main", SHA_B)
    assert returned == SHA_A, "the loser of a race adopts the winner's SHA, never overwrites it"
    assert read_manifest()[manifest_key("org/model", "main")] == SHA_A


def test_the_manifest_pins_across_an_upstream_move(monkeypatch: pytest.MonkeyPatch) -> None:
    """Once recorded, a run keeps its SHA even after upstream moves the ref."""
    record_resolution("org/model", "main", SHA_A)

    def _moved(repo_id: str, ref: str, token: object = None) -> str:
        raise AssertionError("resolution must not be attempted when the manifest already has an answer")

    monkeypatch.setattr("senselab.utils.model_revision._resolve_uncached", _moved)
    assert resolve_revision("org/model", "main") == SHA_A


def test_an_unresolvable_ref_raises_rather_than_falling_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ref that cannot be resolved raises instead of silently returning the ref itself."""

    def _fails(repo_id: str, ref: str, token: object = None) -> str:
        raise RevisionResolutionError("cold cache, hub unreachable")

    monkeypatch.setattr("senselab.utils.model_revision._resolve_uncached", _fails)
    with pytest.raises(RevisionResolutionError):
        resolve_revision("org/model", "main")


def test_a_resolved_ref_is_recorded_so_the_rest_of_the_run_follows(monkeypatch: pytest.MonkeyPatch) -> None:
    """A freshly resolved SHA is written to the manifest so later calls skip resolution."""
    calls = []

    def _once(repo_id: str, ref: str, token: object = None) -> str:
        calls.append((repo_id, ref))
        return SHA_A

    monkeypatch.setattr("senselab.utils.model_revision._resolve_uncached", _once)
    assert resolve_revision("org/model", "main") == SHA_A
    import senselab.utils.model_revision as mr

    mr._MEMO.clear()  # force the manifest, not the memo, to answer the second call
    assert resolve_revision("org/model", "main") == SHA_A
    assert len(calls) == 1, "the manifest must answer after the first resolution"


def test_a_corrupt_manifest_does_not_crash_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unparsable manifest file degrades to 'resolve again' rather than raising."""
    manifest_path().parent.mkdir(parents=True, exist_ok=True)
    manifest_path().write_text("{not json")
    monkeypatch.setattr("senselab.utils.model_revision._resolve_uncached", lambda *a, **k: SHA_A)
    assert resolve_revision("org/model", "main") == SHA_A


def test_cache_root_agrees_with_dependencies_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``_cache_root`` must not drift from ``dependencies._senselab_cache_dir``.

    The two duplicate the same env var and default path -- duplication forced by
    the circular-import constraint (``dependencies`` imports this module) -- and
    nothing else catches the two definitions drifting apart, which would put the
    run manifest in a different tree from the rest of the HF cache.
    """
    import senselab.utils.dependencies as deps
    import senselab.utils.model_revision as mr

    override = tmp_path / "explicit-cache"
    monkeypatch.setenv("SENSELAB_CACHE", str(override))
    assert mr._cache_root() == deps._senselab_cache_dir()

    # Unset case: also pin HOME so dependencies._senselab_cache_dir()'s mkdir
    # lands under tmp_path rather than creating ~/.cache/senselab/hf for real.
    monkeypatch.delenv("SENSELAB_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert mr._cache_root() == deps._senselab_cache_dir()


def _record_resolution_worker(sha: str, cache_dir: str, out: "mp.Queue[str]") -> None:
    """Run ``record_resolution`` in a spawned process and report the winning SHA.

    Module-level, not a closure inside the test: the ``spawn`` start method
    pickles the target by qualified name, and a function nested inside the test
    is a local object that plain ``pickle`` cannot resolve by name, so the
    original nested-closure version of this test failed with
    ``AttributeError: Can't pickle local object`` before any locking logic ran.
    The cache directory is passed as an explicit argument rather than captured
    from an enclosing scope for the same reason.
    """
    os.environ["SENSELAB_CACHE"] = cache_dir
    os.environ["SENSELAB_RUN_ID"] = "shared-run"
    import importlib

    import senselab.utils.model_revision as mr

    importlib.reload(mr)
    out.put(mr.record_resolution("org/model", "main", sha))


def test_concurrent_first_resolution_agrees_on_one_sha(tmp_path: Path) -> None:
    """Two processes recording different SHAs must converge on one winner."""
    cache = tmp_path / "shared"

    ctx = mp.get_context("spawn")
    q: "mp.Queue[str]" = ctx.Queue()
    procs = [ctx.Process(target=_record_resolution_worker, args=(s, str(cache), q)) for s in (SHA_A, SHA_B)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    results = {q.get(timeout=5) for _ in procs}
    assert len(results) == 1, f"both processes must adopt one SHA, got {results}"
