"""Tests that close the mutation gaps the review found, the lock exclusion tests and the CLI-level cache test.

Each test names the behaviour a surviving mutant had broken without any test failing. Every site is
synthetic, under ``tmp_path`` (or the NFS dir named by ``SENSELAB_REPRO_NFS_DIR``).
"""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from senselab.utils.reproduce import build, cache, cli, env, run, sandbox
from senselab.utils.reproduce.run import evaluate_expect
from senselab.utils.reproduce.spec import Manifest, StepSpec
from tests.utils.reproduce.cache_test import OTHER, REV, SHA, _key, _step_data
from tests.utils.reproduce.run_test import World, _metrics, _py, _r_step, _schema_step, _session, _sha

needs_userns = pytest.mark.skipif(not sandbox.userns_available(), reason="unprivileged user namespaces are unavailable")


@pytest.fixture
def world(tmp_path: Path) -> World:
    """A fresh synthetic site."""
    return World(tmp_path)


# --- 1. is_done needs an integer 0 ---------------------------------------------------------------------


@pytest.mark.parametrize("value", [False, 0.0, "0", None, 1])
def test_is_done_refuses_every_exit_but_the_integer_zero(tmp_path: Path, value: object) -> None:
    """``exit: false``, ``0.0``, ``"0"``, null and 1 are not done; only the integer 0 is."""
    (tmp_path / cache.SUMMARY).write_text(json.dumps({"exit": value}))
    assert cache.is_done(tmp_path) is False
    (tmp_path / cache.SUMMARY).write_text(json.dumps({"exit": 0}))
    assert cache.is_done(tmp_path) is True


# --- 2. an hf_weights file sha is keyed --------------------------------------------------------------------


def test_an_hf_weights_file_sha_changes_the_key(tmp_path: Path) -> None:
    """Changing one pinned file's sha256 (same repo, revision and path) changes the step key."""

    def weights(sha: str) -> list[dict[str, Any]]:
        return [{"repo": "org/model", "revision": REV, "files": [{"path": "model.safetensors", "sha256": sha}]}]

    assert _key(tmp_path, _step_data(hf_weights=weights(SHA))) != _key(tmp_path, _step_data(hf_weights=weights(OTHER)))


# --- 3. the in-claim recheck: a waiter skips -----------------------------------------------------------------


SLOW = (
    "import json, sys, time; time.sleep(3); open(sys.argv[1] + '/result.json', 'w').write(json.dumps({'value': 2.0}))"
)


def _race_child(base: str, queue: Any, submission: str) -> None:  # noqa: ANN401
    world = World.__new__(World)
    state = json.loads(Path(base, "world.json").read_text())
    world.__dict__.update(state)
    for name in ("tmp", "repo", "base", "data", "cache", "hook"):
        setattr(world, name, Path(getattr(world, name)))
    world.restricted_dirs = [Path(p) for p in world.restricted_dirs]
    manifest = Manifest.model_validate({"name": "demo", "sources": world.sources, "steps": [_py(world, SLOW)]})
    outcomes = run.run_steps([manifest], world.config(), only=["prep"], submission=submission)
    queue.put([(o.step, o.status, o.exit, list(o.codes)) for o in outcomes])


@needs_userns
def test_concurrent_drivers_of_one_step_execute_it_once(tmp_path: Path) -> None:
    """Three drivers of one step (a job and its duplicates): one executes, the waiters re-check and skip."""
    world = World(tmp_path)
    manifest = world.manifest(_py(world, SLOW))
    world.prefetch(manifest)
    state = {
        k: (str(v) if isinstance(v, Path) else [str(p) for p in v] if k == "restricted_dirs" else v)
        for k, v in world.__dict__.items()
    }
    (tmp_path / "world.json").write_text(json.dumps(state))
    submission = "abcdef123456"
    run.reserve_run_record(world.profile().runs_root, world.config().date, "demo", ["prep"], submission)
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    workers = [context.Process(target=_race_child, args=(str(tmp_path), queue, submission)) for _ in range(3)]
    for worker in workers:
        worker.start()
    results = [queue.get(timeout=240) for _ in workers]
    for worker in workers:
        worker.join(30)
    statuses = sorted(r[0][1] for r in results)
    assert statuses == ["executed", "skipped", "skipped"], results
    assert all(r[0][2] == 0 for r in results), results


# --- 4. a refused remount aborts ---------------------------------------------------------------------------


def test_the_sweep_fails_when_a_non_pseudo_mount_refuses(tmp_path: Path) -> None:
    """Outside a namespace every remount is refused: the sweep exits 1 naming the types left writable."""
    pseudo = ",".join(sorted(sandbox.PSEUDO_FS))
    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", sandbox.SWEEP, pseudo, str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert done.returncode == 1 and "mounts left writable" in done.stderr


@needs_userns
def test_a_failing_sweep_aborts_the_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the sweep fails (faked), ``set -e`` ends the script and the step never runs."""
    monkeypatch.setattr(sandbox, "SWEEP", "import sys\nsys.exit(1)\n")
    marker = tmp_path / "out" / "ran"
    (tmp_path / "out").mkdir()
    done = subprocess.run(
        sandbox.sandbox_argv(["/bin/sh", "-c", f"touch {marker}"], [], [tmp_path / "out"]),
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        check=False,
    )
    assert done.returncode != 0 and not marker.exists()


# --- 5. release_reservation touches only its own submission -------------------------------------------------


def test_release_reservation_leaves_another_submissions_rows(tmp_path: Path) -> None:
    """A row that another submission took over since the reservation is not restored or settled."""
    runs, day = tmp_path / "runs", datetime.date(2026, 3, 4)
    snapshot = run.reserve_run_record(runs, day, "demo", ["a", "b"], "s1")
    other: dict[str, Any] = {
        "step": "b",
        "key": None,
        "status": "pending",
        "exit": None,
        "codes": [],
        "submission": "s2",
    }
    run.update_run_record(runs, day, "demo", [other], order=["a", "b"])
    run.release_reservation(runs, day, "demo", ["a", "b"], "s1", snapshot, ("E_SUBMIT_ABORTED",))
    rows = {r["step"]: r for r in run.read_run_record(runs / "2026-03-04" / "demo" / run.RUN_JSON)}
    assert rows["a"]["codes"] == ["E_SUBMIT_ABORTED"] and rows["a"]["submission"] == "s1"
    assert {k: rows["b"][k] for k in other} == other


# --- 6. both checks are needed ---------------------------------------------------------------------------


@pytest.mark.parametrize(("pip_ok", "smoke_ok"), [(False, True), (True, False), (False, False), (True, True)])
def test_venv_checked_needs_both_checks(world: World, pip_ok: bool, smoke_ok: bool) -> None:
    """A failed pip check is unchecked even when the smoke imports passed, and the other way round."""
    step = world.manifest(_py(world, SLOW)).steps[0]
    path = world.cache / "venvs" / "v"
    path.mkdir(parents=True)
    data = json.dumps({"packages": {}, "freeze": []}).encode()
    (path / env.INSTALLED).write_bytes(data)
    marker = {
        "env_key": env.env_key(step.env),
        "overlay_sha256": None,
        "wheels": {},
        "installed_sha256": hashlib.sha256(data).hexdigest(),
        "checks": {"digest": env.checks_digest(step.env), "pip_check_ok": pip_ok, "smoke_ok": smoke_ok},
    }
    (path / env.MARKER).write_text(json.dumps(marker))
    assert env.venv_checked(path, step.env) is (pip_ok and smoke_ok)


# --- 7. the tolerance boundary passes -----------------------------------------------------------------------


def test_a_delta_equal_to_the_tolerance_passes() -> None:
    """``|value - target| == tolerance`` passes (binary-exact numbers, so no rounding decides it)."""
    step = _schema_step(expect=[{"metric": "balanced_accuracy", "value": 0.5, "tolerance": 0.25, "kind": "abs"}])
    final, passed, codes = evaluate_expect(step, _metrics(0.75, 0.7), {})
    assert codes == [] and passed is True and final is not None
    assert final["headline"][0]["delta"] == 0.25 and final["headline"][0]["pass"] is True
    _, failed, _ = evaluate_expect(step, _metrics(0.765625, 0.7), {})
    assert failed is False


# --- 8. status: an executed row with a non-zero exit failed ---------------------------------------------------


def test_status_exits_1_for_an_executed_row_with_a_non_zero_exit(tmp_path: Path) -> None:
    """``executed`` with exit 3 is a failure for ``status``, not a success."""
    day = tmp_path / "runs" / "2026-03-04" / "demo"
    day.mkdir(parents=True)
    rows = [{"step": "prep", "key": None, "status": "executed", "exit": 3, "codes": ["E_STEP_FAILED"]}]
    (day / run.RUN_JSON).write_text(json.dumps({"steps": rows}))
    assert cli.main(["status", str(day)]) == cli.EXIT_FAILED


# --- locks: exclusion asserted ---------------------------------------------------------------------------


def _try_lock(path: str, shared: bool, queue: Any) -> None:  # noqa: ANN401
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o660)
    try:
        fcntl.flock(fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        queue.put("got")
    except BlockingIOError:
        queue.put("blocked")
    finally:
        os.close(fd)


def _lock_dir(tmp_path: Path, where: str) -> Path:
    """``tmp_path``, or a fresh sibling of ``SENSELAB_REPRO_NFS_DIR`` (skipped when the variable is unset).

    A sibling, not the dir itself, which the ``nfs_dir`` fixture creates and removes.
    """
    if where == "local":
        return tmp_path
    value = os.environ.get("SENSELAB_REPRO_NFS_DIR", "")
    if not value:
        pytest.skip("SENSELAB_REPRO_NFS_DIR is not set")
    path = Path(value).parent / f"{Path(value).name}-locks-{os.getpid()}-{tmp_path.name}"
    path.mkdir(parents=True)
    return path


@pytest.mark.parametrize("where", ["local", "nfs"])
def test_an_exclusive_lock_excludes_and_shared_locks_coexist(tmp_path: Path, where: str) -> None:
    """Another process cannot take a held exclusive lock in any mode; two shared locks are held together."""
    directory = _lock_dir(tmp_path, where)
    lock = directory / "x.lock"
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()

    def other(shared: bool) -> str:
        process = context.Process(target=_try_lock, args=(str(lock), shared, queue))
        process.start()
        process.join(60)
        return str(queue.get(timeout=10))

    with build.file_lock(lock):
        assert other(False) == "blocked" and other(True) == "blocked"
    with build.file_lock(lock, shared=True):
        assert other(True) == "got" and other(False) == "blocked"
    assert other(False) == "got"
    if where == "nfs":
        shutil.rmtree(directory)


# --- CLI-level C5: the cache across a 4-step DAG -----------------------------------------------------------


def _dag(world: World, score_expect: list[dict[str, Any]] | None = None) -> Any:  # noqa: ANN401
    prep = world.step(
        "prep",
        "prep",
        "{root:data}/in/a.txt",
        patches=[{"file": "patches/p.patch", "sha256": _sha(world.base / "patches" / "p.patch")}],
        adapter="adapter_prep.py",
    )
    score = world.step(
        "score",
        "score",
        "{up:prep}/result.json",
        needs=["prep"],
        inputs=[{"root": "up:prep", "patterns": ["result.json"]}],
        expect=score_expect or [],
    )
    rstep = _r_step(world, _session(world))
    rstep.update(name="rstep", needs=["score"], inputs=[{"root": "up:score", "patterns": ["result.json"]}])
    notebook = world.step("nb", "prep", "{root:data}/in/a.txt", kind="notebook", needs=["prep"])
    notebook["env"]["notebook_tools"] = {"lock": "nb.lock", "lock_sha256": "ef" * 32}
    return world.manifest(prep, score, rstep, notebook)


def _write_inputs(world: World, manifest: Any) -> tuple[Path, Path]:  # noqa: ANN401
    profile = world.tmp / "profile.yaml"
    profile.write_text(yaml.safe_dump(world.profile().model_dump(mode="json")))
    path = world.base / "demo.yaml"
    path.write_text(yaml.safe_dump(manifest.model_dump(mode="json")))
    return profile, path


def _statuses(text: str) -> dict[str, str]:
    return {line.split("\t")[0]: line.split("\t")[2] for line in text.splitlines() if line.strip()}


@needs_userns
def test_cli_run_caches_a_four_step_dag(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    """Four executed, then four skipped; a patch reruns its step and dependents; an expect change, its own.

    The patch is on ``prep``, which every other step depends on; the expectation is on ``score``, whose only
    dependent is ``rstep``.
    """
    manifest = _dag(world)
    world.prefetch(manifest)
    profile, path = _write_inputs(world, manifest)
    argv = ["run", str(path), "--profile", str(profile), "--base-dir", str(world.base), "--cache", str(world.cache)]

    def run_cli() -> dict[str, str]:
        assert cli.main(argv) == cli.EXIT_OK, capsys.readouterr()
        return _statuses(capsys.readouterr().out)

    assert run_cli() == {"prep": "executed", "score": "executed", "rstep": "executed", "nb": "executed"}
    assert set(run_cli().values()) == {"skipped"}

    world.write_patch("# v7")
    patched = _dag(world)
    world.prefetch(patched)
    _write_inputs(world, patched)
    assert set(run_cli().values()) == {"executed"}
    assert set(run_cli().values()) == {"skipped"}

    expecting = _dag(world, [{"metric": "f_stat", "value": 4.0, "tolerance": 0.5, "kind": "abs"}])
    world.prefetch(expecting)
    _write_inputs(world, expecting)
    assert run_cli() == {"prep": "skipped", "score": "executed", "rstep": "executed", "nb": "skipped"}


# --- parse-before-pin --------------------------------------------------------------------------------------


@pytest.mark.parametrize("change", ["parsable", "unparsable"])
def test_a_changed_manifest_is_refused_before_it_is_parsed(
    world: World, change: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """A job whose manifest changed, parsable or not, writes ``E_MANIFEST_CHANGED`` on its own row."""
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    profile, path = _write_inputs(world, manifest)
    pin = hashlib.sha256(path.read_bytes()).hexdigest()
    date = world.config().date
    runs = world.profile().runs_root
    run.reserve_run_record(runs, date, "demo", ["prep"], "s1")
    run.update_run_record(runs, date, "demo", [{"step": "prep", "job": "42"}], order=["prep"], submission="s1")
    path.write_text(path.read_text() + "\n# edited\n" if change == "parsable" else "steps: [\n")
    argv = ["run", str(path), "--profile", str(profile), "--base-dir", str(world.base), "--cache", str(world.cache)]
    argv += ["--date", date.isoformat(), "--submission", "s1", "--pin", pin, "--step", "prep"]
    assert cli.main(argv) == cli.EXIT_FAILED
    assert capsys.readouterr().out.splitlines() == ["prep\t-\tfailed\t1\tE_MANIFEST_CHANGED"]
    (row,) = run.read_run_record(runs / date.isoformat() / "demo" / run.RUN_JSON)
    assert row["status"] == "failed" and row["codes"] == ["E_MANIFEST_CHANGED"] and row["submission"] == "s1"


def test_step_spec_helper_is_valid() -> None:
    """The cache tests' step data still validates (helper sanity for the key test above)."""
    assert StepSpec.model_validate(_step_data()).name
