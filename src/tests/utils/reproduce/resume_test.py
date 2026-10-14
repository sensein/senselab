"""Tests for resumable steps: what an attempt is, the kept store, its discard, and the claim's leftover test.

Every site is synthetic (`World`); a killed holder is simulated by leaving its ``.holder`` and heartbeat files
behind with no process holding the lock.
"""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from senselab.utils.file_lock import SharedFileLock
from senselab.utils.reproduce import cache, digest, run
from senselab.utils.reproduce.cache import CacheError
from senselab.utils.reproduce.spec import Manifest
from tests.utils.reproduce.run_test import World, needs_userns
from tests.utils.reproduce.slurm_test import _profile, _spec

SHA = "ab" * 32

ITEMS = """\
import json, os, pathlib, sys
out = pathlib.Path(sys.argv[1])
store = out / "items"
store.mkdir(exist_ok=True)
present = sorted(p.stem for p in store.glob("*.done"))
made = []
for item in ("a", "b", "c", "d"):
    if (store / (item + ".done")).exists():
        continue
    (store / (item + ".tmp")).write_text(item)
    os.rename(store / (item + ".tmp"), store / (item + ".json"))
    (store / (item + ".done")).write_text("")
    made.append(item)
(out / "made.json").write_text(json.dumps({"present": present, "made": made}))
(out / "result.json").write_text(json.dumps({"value": float(len(made))}))
"""


@pytest.fixture
def world(tmp_path: Path) -> World:
    """A fresh synthetic site."""
    return World(tmp_path)


def _escaped(code: str) -> str:
    return code.replace("{", "{{").replace("}", "}}")


def _manifest(world: World, **change: Any) -> Manifest:  # noqa: ANN401
    argv = ["{venv}/bin/python", "-c", _escaped(ITEMS), "{out}"]
    fields: dict[str, Any] = {"resumable": "items", "resume_dir": "items", **change}
    manifest = world.manifest(world.step("infer", "prep", "", argv=argv, **{k: v for k, v in fields.items() if v}))
    world.prefetch(manifest)
    return manifest


def _key(world: World, manifest: Manifest, **change: Any) -> str:  # noqa: ANN401
    [plan] = run.plan_steps([manifest], world.config(**change))
    assert plan.key is not None, plan.codes
    return plan.key


def _killed(world: World, key: str, *, holder: bool = True, summary: dict[str, Any] | None = None) -> Path:
    """What a killed attempt leaves: two finished items and one unmarked, an attempts line, the claim's files."""
    out = world.tmp / "results" / key
    store = out / run.WORK / "items"
    store.mkdir(parents=True)
    for item in ("a", "b", "c"):
        (store / f"{item}.json").write_text(item)
    for item in ("a", "b"):
        (store / f"{item}.done").write_text("")
    (out / run.ATTEMPTS).write_text(json.dumps({"attempt": 1, "started": "t0", "host": "h"}) + "\n")
    if summary is not None:
        (out / cache.SUMMARY).write_text(json.dumps(summary))
    if holder:
        claims = world.tmp / "results" / cache.CLAIMS
        claims.mkdir(exist_ok=True)
        payload = {"user": "u", "host": socket.gethostname(), "pid": 2**22 + 7, "taken_at": time.time() - 30}
        (claims / f"{key}.holder").write_text(json.dumps({**payload, "token": "dead"}))
        beat = claims / f"{key}.heartbeat"
        beat.write_text("")
        os.utime(beat, (time.time() - 10, time.time() - 10))
    return out


def _made(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out / run.WORK / "made.json").read_text())
    return data


def _summary(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out / cache.SUMMARY).read_text())
    return data


def _attempts(out: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (out / run.ATTEMPTS).read_text().splitlines()]


@needs_userns
def test_a_killed_attempt_keeps_the_item_store(world: World) -> None:
    """A leftover holder and no summary is an attempt: the step sees its two finished items, attempt 2."""
    died = world.tmp / "died" / SHA
    (died / cache.RESUME_KEEP).mkdir(parents=True)
    (died / cache.RESUME_KEEP / "a.done").write_text("")
    (died / "logs").mkdir()
    assert cache.reset_result_dir(died.parent, SHA, keep="work/items", keep_attempts=True) == died
    assert sorted(p.relative_to(died).as_posix() for p in died.rglob("*")) == [
        "work",
        "work/items",
        "work/items/a.done",
    ]

    manifest = _manifest(world)
    out = _killed(world, _key(world, manifest))
    outcome = world.run(manifest, parent_env={"SLURM_JOB_ID": "77", "SLURM_RESTART_COUNT": "1"})["infer"]
    assert outcome.ok, outcome.codes
    assert _made(out) == {"present": ["a", "b"], "made": ["c", "d"]}
    assert _summary(out)["attempt"] == 2
    attempts = _attempts(out)
    assert [a["attempt"] for a in attempts] == [1, 2]
    assert set(attempts[1]) == {"attempt", "started", "host", "slurm_job_id", "slurm_restart_count"}
    assert (attempts[1]["slurm_job_id"], attempts[1]["slurm_restart_count"]) == ("77", "1")
    assert not (out / ".resume-keep").exists()


@needs_userns
def test_a_failed_run_with_a_summary_is_reset_completely(world: World) -> None:
    """A result dir with a failed summary is no attempt, leftover holder or not: everything goes, attempt 1."""
    manifest = _manifest(world)
    out = _killed(world, _key(world, manifest), summary={"exit": 1, "codes": ["E_STEP_FAILED"]})
    outcome = world.run(manifest)["infer"]
    assert outcome.ok, outcome.codes
    assert _made(out) == {"present": [], "made": ["a", "b", "c", "d"]}
    assert _summary(out)["attempt"] == 1 and [a["attempt"] for a in _attempts(out)] == [1]


@needs_userns
def test_integrity_codes_discard_the_store(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run ending with an integrity code removes the store before its summary is written."""
    assert run.INTEGRITY_CODES == {
        "E_INPUT_MUTATED",
        "E_SRC_MUTATED",
        "E_ENV_CHANGED",
        "E_ENV_MISSING",
        "E_ENV_UNCHECKED",
        "E_ADAPTER_CHANGED",
        "E_ENGINE_FIT",
    }
    manifest = _manifest(world)
    out = _killed(world, _key(world, manifest))
    monkeypatch.setattr(digest, "digest_unchanged", lambda before, after: False)
    outcome = world.run(manifest)["infer"]
    assert outcome.codes == ("E_INPUT_MUTATED",) and outcome.exit != 0
    assert _summary(out)["attempt"] == 2
    assert not (out / run.WORK / "items").exists() and (out / run.WORK / "made.json").exists()

    world.env_map = {"MODE": "b"}
    manifest = _manifest(world)
    refused = _killed(world, _key(world, manifest), holder=False)

    def changed(*args: Any) -> Any:  # noqa: ANN401
        raise run.RunError("E_ENV_CHANGED")

    monkeypatch.setattr(run, "_ready", changed)
    outcome = world.run(manifest)["infer"]
    assert outcome.codes == ("E_ENV_CHANGED",) and not (refused / cache.SUMMARY).exists()
    assert not (refused / run.WORK / "items").exists()


@needs_userns
def test_a_non_resumable_step_is_reset_on_an_attempt(world: World) -> None:
    """An attempt of a step that declares no store keeps nothing of ``work/``, but is still counted."""
    manifest = _manifest(world, resumable=None, resume_dir=None)
    out = _killed(world, _key(world, manifest))
    outcome = world.run(manifest)["infer"]
    assert outcome.ok, outcome.codes
    assert _made(out) == {"present": [], "made": ["a", "b", "c", "d"]}
    assert _summary(out)["attempt"] == 2

    world.env_map = {"MODE": "b"}
    manifest = _manifest(world, resumable=None, resume_dir=None)
    full = _killed(world, _key(world, manifest))

    def no_space(*args: Any) -> None:  # noqa: ANN401
        raise OSError(28, "No space left on device")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(run, "_record_attempt", no_space)
        outcome = world.run(manifest)["infer"]
    assert outcome.codes == ("E_DRIVER",) and _summary(full)["codes"] == ["E_DRIVER"]
    assert _summary(full)["attempt"] == 2


@needs_userns
def test_a_waiter_after_a_clean_release_is_not_an_attempt(world: World) -> None:
    """A claim that waited for a live holder's clean release is no attempt, though it saw that holder's identity."""
    root = world.tmp / "results"
    held, release = threading.Event(), threading.Event()

    def holder(key: str) -> None:
        with cache.claim(root, key):
            held.set()
            release.wait(10)
            time.sleep(2)

    thread = threading.Thread(target=holder, args=("cd" * 32,))
    thread.start()
    held.wait(10)
    release.set()
    with cache.claim(root, "cd" * 32) as (_, leftover):
        assert leftover is False
    thread.join()

    manifest = _manifest(world)
    key = _key(world, manifest)
    out = _killed(world, key, holder=False)
    held.clear()
    thread = threading.Thread(target=holder, args=(key,))
    thread.start()
    held.wait(10)
    outcome = world.run(manifest)["infer"]
    thread.join()
    assert outcome.ok, outcome.codes
    assert _made(out) == {"present": [], "made": ["a", "b", "c", "d"]}
    assert _summary(out)["attempt"] == 1


def test_a_fresh_leftover_holder_is_taken_over_without_error(tmp_path: Path) -> None:
    """Regression pin of `SharedFileLock.__enter__`: a leftover holder with a 10 s old heartbeat is taken over.

    The one refusing branch is a leftover with no heartbeat and no ``taken_at``, which a killed job does not
    leave, since its heartbeat file stays.
    """
    root = tmp_path / "results"
    claims = root / cache.CLAIMS
    claims.mkdir(parents=True)
    holder = {"user": "u", "host": "h", "pid": 1, "taken_at": time.time() - 60, "token": "dead"}
    (claims / f"{SHA}.holder").write_text(json.dumps(holder))
    beat = claims / f"{SHA}.heartbeat"
    beat.write_text("")
    os.utime(beat, (time.time() - 10, time.time() - 10))
    real_write = SharedFileLock._write_holder

    def slow_write(self: SharedFileLock) -> None:
        time.sleep(1.0)
        real_write(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(SharedFileLock, "_write_holder", slow_write)
        with cache.claim(root, SHA) as (out, leftover):
            assert leftover is True and out == root / SHA
    with cache.claim(root, SHA) as (_, leftover):
        assert leftover is False
    (claims / f"{SHA}.holder").write_text(json.dumps({"user": "u", "host": "h", "pid": 1}))
    with pytest.raises(CacheError, match="E_CLAIM_HELD"):
        with cache.claim(root, SHA):
            pass


def test_resume_dir_is_required_and_keyed(tmp_path: Path) -> None:
    """``resumable`` and ``resume_dir`` come together, the dir is a plain relative path, and both are keyed."""
    for bad in (
        {"resumable": "items"},
        {"resume_dir": "items"},
        {"resumable": "items", "resume_dir": "/abs"},
        {"resumable": "items", "resume_dir": "../up"},
        {"resumable": "items", "resume_dir": "stage/x"},
        {"resumable": "items", "resume_dir": "notebooks"},
        {"resumable": "items", "resume_dir": "."},
        {"resumable": "items", "resume_dir": "metrics.json"},
        {"resumable": "items", "resume_dir": "store/summary.json"},
        {"resumable": "nope", "resume_dir": "items"},
    ):
        with pytest.raises(ValueError):
            _spec("infer", **bad)
    assert run.RESERVED_NAMES == {cache.SUMMARY, run.METRICS, run.RUN_JSON, run.EXTRACTED}
    profile = _profile(tmp_path)
    kwargs: dict[str, Any] = {
        "upstream": {},
        "submodules": {},
        "adapter_sha256": SHA,
        "adapter_env_sha256": SHA,
        "input_digests": [SHA],
        "native": {},
        "sources_lock_sha256": SHA,
    }
    keys = {
        cache.step_key(_spec("infer", **change), profile, **kwargs)
        for change in (
            {},
            {"resumable": "items", "resume_dir": "items"},
            {"resumable": "checkpoint", "resume_dir": "items"},
            {"resumable": "items", "resume_dir": "store"},
        )
    }
    assert len(keys) == 4
    payload = cache.step_key_payload(_spec("infer", resumable="checkpoint", resume_dir="ckpt"), profile, **kwargs)
    assert (payload["resumable"], payload["resume_dir"]) == ("checkpoint", "ckpt")


@needs_userns
def test_a_republished_pool_item_is_an_attempt(world: World) -> None:
    """``POOL_ATTEMPT`` above 1 and no summary is an attempt with no leftover holder; ``1`` is not."""
    manifest = _manifest(world)
    first = _killed(world, _key(world, manifest), holder=False)
    assert world.run(manifest, parent_env={"POOL_ATTEMPT": "1"})["infer"].ok
    assert _made(first)["present"] == [] and _summary(first)["attempt"] == 1

    world.env_map = {"MODE": "b"}
    manifest = _manifest(world)
    second = _killed(world, _key(world, manifest), holder=False)
    assert second != first
    outcome = world.run(manifest, parent_env={"POOL_ATTEMPT": "2"})["infer"]
    assert outcome.ok, outcome.codes
    assert _made(second) == {"present": ["a", "b"], "made": ["c", "d"]}
    assert _summary(second)["attempt"] == 2
