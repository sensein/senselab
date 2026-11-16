"""Tests for a sharded step's ``host_probe``: run in the step's sandbox by each work-pool worker before it claims.

A probe that fails, for each reason, leaves the worker without a claim: the pool records the failure under
``hostcheck/`` with its partition and reason, and the item stays queued with its attempt count unchanged. A probe
that passes lets the worker claim and run the item.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from senselab.utils import work_pool
from senselab.utils.reproduce import cache, run, slurm
from senselab.utils.reproduce.spec import Manifest, StepSpec
from tests.utils.reproduce.run_test import World, needs_userns, write_manifest_file
from tests.utils.reproduce.slurm_test import CLASSES, POOL_SETTINGS, _profile

SHARDS = {"count": 2, "resource_class": "small"}
PY = "{venv}/bin/python"


@pytest.fixture
def world(tmp_path: Path) -> World:
    """A fresh synthetic site with the pool settings and resource classes."""
    w = World(tmp_path)
    w.profile_extra = {"resource_classes": CLASSES, "work_pool": POOL_SETTINGS}
    return w


def _probe_step(world: World, probe: list[str], **change: Any) -> dict[str, Any]:  # noqa: ANN401
    return world.step("parts", "prep", "{root:data}/in/a.txt", shards=SHARDS, host_probe=probe, **change)


def _site(world: World, probe: list[str], **change: Any) -> tuple[Manifest, run.RunConfig, Path]:  # noqa: ANN401
    """The manifest (as a file), its run config and the manifest path; the step venv is prefetched."""
    manifest, ids = write_manifest_file(world, world.manifest(_probe_step(world, probe, **change)))
    world.prefetch(manifest)
    return manifest, world.config(manifest_ids=ids), world.base / "analyses" / "demo.json"


def _cli_probe(world: World, path: Path) -> list[str]:
    """The worker's host check as the pool job builds it (`slurm.host_probe_command`)."""
    profile = world.tmp / "profile.yaml"
    profile.write_text(yaml.safe_dump(world.profile().model_dump(mode="json")))
    base = slurm.driver_run_argv(
        manifest_paths=[path],
        profile_path=profile,
        base_dir=world.base,
        cache=world.cache,
        date=world.config().date,
        submission="sub",
        pins=[],
    )
    return slurm.host_probe_command(base, "parts", sys.executable)


def _pool(world: World) -> work_pool.WorkPool:
    pool = work_pool.WorkPool.create(world.tmp / "pool", lease_s=30.0, heartbeat_s=5.0)
    pool.publish([work_pool.Item(id="sub-0", resource_class="small", command=("true",))])
    return pool


def _work(pool: work_pool.WorkPool, check: list[str], monkeypatch: pytest.MonkeyPatch) -> int:
    monkeypatch.setenv("SLURM_JOB_PARTITION", "main")
    return work_pool.work(
        pool.root,
        ["small"],
        worker="w1",
        linger=0,
        poll_s=0.01,
        host_check=check,
        host_check_reasons={code: reason for reason, code in run.HOST_PROBE_EXIT.items()},
    )


def _hostcheck_records(pool: work_pool.WorkPool) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((pool.root / "hostcheck").glob("*.json"))]


def _queued_attempt(pool: work_pool.WorkPool, item_id: str) -> int:
    attempt: int = json.loads((pool.root / "queue" / f"{item_id}.json").read_text())["attempt"]
    return attempt


# --- the probe itself ---------------------------------------------------------------------------------------


@needs_userns
def test_a_passing_probe_returns_none_and_touches_no_result(world: World) -> None:
    """The probe runs in the step's sandbox (it sees its input and its venv) and leaves no result dir or link."""
    probe = [
        PY,
        "-c",
        "import pathlib, sys; sys.exit(0 if pathlib.Path(sys.argv[1]).is_file() else 5)",
        "{root:data}/in/a.txt",
    ]
    manifest, config, _ = _site(world, probe)
    assert run.host_probe([manifest], config, "parts", 0) is None
    results = world.tmp / "results"
    assert [p.name for p in results.iterdir()] == [run.HOST_PROBE_DIR]
    assert not any((results / run.HOST_PROBE_DIR).iterdir())
    assert not (world.tmp / "runs").exists()


@needs_userns
@pytest.mark.parametrize(
    ("probe", "timeout", "reason"),
    [
        ([PY, "-c", "import sys; sys.exit(3)"], 120.0, "exit-nonzero"),
        (["{venv}/bin/no-such-program"], 120.0, "exec-error"),
        ([PY, "-c", "import time; time.sleep(120)"], 10.0, "timeout"),
    ],
)
def test_a_failing_probe_names_its_reason(world: World, probe: list[str], timeout: float, reason: str) -> None:
    """A non-zero exit, a program that cannot be executed and a probe past its timeout each give their reason."""
    manifest, config, _ = _site(world, probe, host_probe_timeout_s=timeout)
    assert run.host_probe([manifest], config, "parts", 0) == reason


@needs_userns
def test_a_host_without_a_sandbox_is_sandbox_unavailable(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """A refusal before the probe (here the sandbox sweep) is ``sandbox-unavailable``; the probe never runs."""
    manifest, config, _ = _site(world, [PY, "-c", "pass"])

    def refused(*_: object) -> None:
        raise run.RunError("E_SANDBOX_UNAVAILABLE")

    monkeypatch.setattr(run, "_ready", refused)
    assert run.host_probe([manifest], config, "parts", 0) == "sandbox-unavailable"


@needs_userns
def test_a_sandbox_that_does_not_start_is_sandbox_unavailable(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """When the sandbox check fails the probe is not run: ``sandbox-unavailable`` even for a passing probe."""
    manifest, config, _ = _site(world, [PY, "-c", "pass"])
    monkeypatch.setattr(run, "HOST_PROBE_SANDBOX_CHECK", "/bin/false")
    assert run.host_probe([manifest], config, "parts", 0) == "sandbox-unavailable"


@needs_userns
def test_failures_that_do_not_depend_on_the_host_pass(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """A keying error or an env refusal is the same on every host: the probe passes and the shard fails with it."""
    manifest, config, _ = _site(world, [PY, "-c", "import sys; sys.exit(3)"])

    def key_error(*_: object, **__: object) -> None:
        raise run.RunError("E_KEY_INVALID")

    with monkeypatch.context() as patch:
        patch.setattr(run, "_key", key_error)
        assert run.host_probe([manifest], config, "parts", 0) is None

    def env_missing(*_: object) -> None:
        raise run.RunError("E_ENV_MISSING")

    monkeypatch.setattr(run, "_ready", env_missing)
    assert run.host_probe([manifest], config, "parts", 0) is None


# --- the worker ---------------------------------------------------------------------------------------------


@needs_userns
@pytest.mark.parametrize(
    ("probe", "reason"),
    [([PY, "-c", "import sys; sys.exit(3)"], "exit-nonzero"), (["{venv}/bin/no-such-program"], "exec-error")],
)
def test_a_worker_whose_probe_fails_claims_nothing_and_records_the_reason(
    world: World, monkeypatch: pytest.MonkeyPatch, probe: list[str], reason: str
) -> None:
    """The worker exits `work_pool.HOST_CHECK_EXIT`; ``hostcheck/`` names partition and reason; the item waits."""
    _, _, path = _site(world, probe)
    pool = _pool(world)
    assert _work(pool, _cli_probe(world, path), monkeypatch) == work_pool.HOST_CHECK_EXIT
    records = _hostcheck_records(pool)
    assert [(r["partition"], r["rc"], r["reason"]) for r in records] == [("main", run.HOST_PROBE_EXIT[reason], reason)]
    assert pool.state("sub-0") == "queued" and _queued_attempt(pool, "sub-0") == 1
    assert pool.host_check_failures(3600.0) == {"main": 1}


@needs_userns
def test_a_worker_whose_probe_passes_claims_and_runs(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """With a passing probe the worker claims the item and it ends ``done``; nothing is recorded."""
    _, _, path = _site(world, [PY, "-c", "pass"])
    pool = _pool(world)
    assert _work(pool, _cli_probe(world, path), monkeypatch) == 0
    assert pool.state("sub-0") == "done" and _hostcheck_records(pool) == []


def test_a_worker_records_a_timeout_and_an_unnamed_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``timeout`` is always named; an exit code without a name is recorded with reason None."""
    pool = work_pool.WorkPool.create(tmp_path / "pool", lease_s=30.0, heartbeat_s=5.0)
    pool.publish([work_pool.Item(id="i-0", resource_class="small", command=("true",))])
    monkeypatch.setenv("SLURM_JOB_PARTITION", "main")
    sleep = [sys.executable, "-c", "import time; time.sleep(30)"]
    code = work_pool.work(
        pool.root, ["small"], worker="a", linger=0, poll_s=0.01, host_check=sleep, host_check_timeout_s=0.5
    )
    assert code == work_pool.HOST_CHECK_EXIT
    exit7 = [sys.executable, "-c", "import sys; sys.exit(7)"]
    code = work_pool.work(
        pool.root, ["small"], worker="b", linger=0, poll_s=0.01, host_check=exit7, host_check_reasons={21: "x"}
    )
    assert code == work_pool.HOST_CHECK_EXIT
    records = {r["worker"]: (r["rc"], r["reason"]) for r in _hostcheck_records(pool)}
    assert records == {"a": ("timeout", "timeout"), "b": (7, None)}
    assert pool.state("i-0") == "queued"


# --- the pool job passes the probe to its workers -----------------------------------------------------------


def test_the_pool_drain_gives_each_worker_the_host_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`slurm.run_pool` with ``host_probe`` starts workers with ``--host-check``, its timeout and the reason names."""
    seen: dict[str, Any] = {}

    class Controller:
        def __init__(self, pool: work_pool.WorkPool, classes: Any, *, worker_command: list[str], **_: Any) -> None:  # noqa: ANN401
            seen["command"] = worker_command

        def run(self) -> int:
            return 0

    monkeypatch.setattr(work_pool, "Controller", Controller)
    profile = _profile(tmp_path, resource_classes=CLASSES, work_pool=POOL_SETTINGS)
    probe = [
        "/opt/py",
        "-P",
        "-m",
        "senselab.utils.reproduce",
        "run",
        "m.yaml",
        "--step",
        "parts",
        "--shard",
        "0",
        "--host-probe",
    ]
    args = slurm.host_check_args(probe, 2 * 120.0 + slurm.HOST_PROBE_MARGIN_S)
    drain = slurm.pool_drain(profile, datetime.date(2026, 1, 2), "/opt/py", args)
    pool = work_pool.WorkPool.create(tmp_path / "pool", lease_s=30.0, heartbeat_s=5.0)
    drain(pool, [])
    command = seen["command"]
    at = command.index("--host-check")
    assert command[at + 1].split() == probe
    assert command[command.index("--host-check-timeout") + 1] == "840.0"
    named = sorted(a.split("=", 1)[1] for a in command if a.startswith("--host-check-reason="))
    assert named == sorted(f"{code}={reason}" for reason, code in run.HOST_PROBE_EXIT.items())


def test_the_work_command_line_records_the_named_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``work … --host-check CMD --host-check-reason CODE=NAME`` records that name; a bad pair is refused."""
    pool = work_pool.WorkPool.create(tmp_path / "pool", lease_s=30.0, heartbeat_s=5.0)
    pool.publish([work_pool.Item(id="i-0", resource_class="small", command=("true",))])
    monkeypatch.setenv("SLURM_JOB_PARTITION", "main")
    check = f"{sys.executable} -c 'import sys; sys.exit(21)'"
    argv = ["work", str(pool.root), "--classes", "small", "--linger", "0", "--poll", "0.01", "--worker", "w"]
    assert (
        work_pool.main([*argv, "--host-check", check, "--host-check-reason=21=exit-nonzero"])
        == work_pool.HOST_CHECK_EXIT
    )
    assert [(r["rc"], r["reason"]) for r in _hostcheck_records(pool)] == [(21, "exit-nonzero")]
    with pytest.raises(SystemExit):
        work_pool.main([*argv, "--host-check", check, "--host-check-reason=x=y"])


# --- the spec -----------------------------------------------------------------------------------------------


def test_host_probe_is_sharded_only_and_not_keyed(world: World) -> None:
    """Only a public sharded step may declare a probe (``E_SPEC_HOST_PROBE``); the probe and timeout are not keyed."""
    with pytest.raises(ValueError, match="E_SPEC_HOST_PROBE"):
        StepSpec.model_validate(world.step("one", "prep", "{root:data}/in/a.txt", host_probe=[PY, "-c", "pass"]))
    with pytest.raises(ValueError, match="E_SPEC_HOST_PROBE"):
        StepSpec.model_validate(_probe_step(world, [PY, "-c", "pass"], restricted=True))
    with pytest.raises(ValueError, match="E_SPEC_ABSOLUTE_PATH"):
        StepSpec.model_validate(_probe_step(world, ["/usr/bin/nvidia-smi"]))
    with_probe = StepSpec.model_validate(_probe_step(world, [PY, "-c", "pass"], host_probe_timeout_s=30))
    without = StepSpec.model_validate(world.step("parts", "prep", "{root:data}/in/a.txt", shards=SHARDS))
    common: dict[str, Any] = {
        "upstream": {},
        "submodules": {},
        "adapter_sha256": "a" * 64,
        "adapter_env_sha256": "b" * 64,
        "input_digests": ["c" * 64],
        "native": {},
        "sources_lock_sha256": "d" * 64,
        "shard": 0,
    }
    profile = world.profile()
    assert cache.step_key(with_probe, profile, **common) == cache.step_key(without, profile, **common)
