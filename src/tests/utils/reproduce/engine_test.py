"""Tests for engine steps: the vLLM launcher in the sandbox, its status codes, the engine key and provenance.

The ``vllm`` is the fake `vllm_server_test` drives, installed as ``{venv}/bin/vllm`` with a wrapper that gives it a
record dir under ``{out}/fake`` and holds a lock file there for as long as any of its processes lives. Weights are
a small fake hub cache; no GPU is used (the host probe is faked).
"""

from __future__ import annotations

import fcntl
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from senselab_core.runtime import hardware

from senselab.utils.reproduce import cache, engine, host, run, sandbox
from senselab.utils.reproduce.spec import Manifest, StepSpec
from tests.utils.reproduce.resume_test import _killed
from tests.utils.reproduce.run_test import World, _git, needs_userns
from tests.utils.reproduce.weights_test import REPO, REVISION, WEIGHTS, _hub, _weights, make_snapshot
from tests.utils.vllm_server_test import _FAKE

DRIVER = "580.95.05"

RECORDER = """\
import fcntl, json, os, sys
_record = os.path.join(os.environ["REPRO_OUT"], "fake")
os.makedirs(_record, exist_ok=True)
os.environ["FAKE_VLLM_DIR"] = _record
_alive = open(os.path.join(_record, "alive-%d" % os.getpid()), "w")
fcntl.flock(_alive, fcntl.LOCK_EX)
with open(os.path.join(_record, "argv-%d.json" % os.getpid()), "w") as _handle:
    json.dump(sys.argv[1:], _handle)


def _stopped(signum, frame):
    with open(os.path.join(_record, "events.log"), "a") as handle:
        handle.write("stopped %d 0\\n" % os.getpid())
    os._exit(0)


import signal
signal.signal(signal.SIGTERM, _stopped)
"""
"""Prepended to the fake: its record dir, a lock held while it (or its forked child) lives, and its argv."""

ENGINE_STEP = """\
import json, os, pathlib, sys, time, urllib.request
out, mode = pathlib.Path(sys.argv[1]), sys.argv[2]
urls = os.environ["REPRO_ENGINE_URLS"].split(",")
roots = []
for url in urls:
    with urllib.request.urlopen(url + "/v1/models", timeout=10) as response:
        roots.append(json.loads(response.read())["data"][0]["root"].rsplit("/", 1)[-1])
report = {
    "value": float(len(urls)),
    "roots": roots,
    "identity": json.loads(os.environ["REPRO_ENGINE_IDENTITY"]),
    "no_proxy": [os.environ["no_proxy"], os.environ["NO_PROXY"]],
}
(out / "result.json").write_text(json.dumps(report))
if mode == "exit94":
    sys.exit(94)
if mode == "term":
    import signal
    os.kill(os.getpid(), signal.SIGTERM)
if mode == "sleep":
    time.sleep(600)
"""


def _engine(**change: Any) -> dict[str, Any]:  # noqa: ANN401
    return {
        "name": "vllm",
        "version": "0.31.0",
        "model": {"repo": REPO, "revision": REVISION},
        "concurrency": 2,
        "batch_size": 4,
        "servers_per_card": 2,
        "max_num_seqs": 4,
        "max_model_len": 2048,
        **change,
    }


def _world(tmp_path: Path) -> World:
    world = World(tmp_path)
    world.profile_extra = {"hf_hub_cache": str(_hub(world))}
    (world.repo / "engine_step.py").write_text(ENGINE_STEP)
    _git(world.repo, "add", "-A")
    _git(world.repo, "commit", "-q", "-m", "engine")
    world.commit = _git(world.repo, "rev-parse", "HEAD").strip()
    make_snapshot(_hub(world), REPO, REVISION, {"weights.bin": WEIGHTS})
    return world


@pytest.fixture
def world(tmp_path: Path) -> World:
    """A synthetic site with an engine step script and a fake hub snapshot."""
    return _world(tmp_path)


def _step(world: World, mode: str = "ok", **change: Any) -> dict[str, Any]:  # noqa: ANN401
    fields: dict[str, Any] = {
        "argv": ["{venv}/bin/python", "{checkout}/engine_step.py", "{out}", mode],
        "hf_weights": _weights({"weights.bin": WEIGHTS}),
        "resources": {"profile": "cpu", "gpu_arch": ["sm_90"], "min_driver": "580.65.06"},
        "engine": _engine(),
        **change,
    }
    return world.step("infer", "prep", "", **fields)


def _manifest(world: World, mode: str = "ok", **change: Any) -> Manifest:  # noqa: ANN401
    manifest = world.manifest(_step(world, mode, **change))
    world.prefetch(manifest)
    venv_bin = run._step_venv(manifest.steps[0], manifest.name, world.cache)[0] / "bin"
    (venv_bin / "fake_vllm.py").write_text(RECORDER + _FAKE)
    exe = venv_bin / "vllm"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{venv_bin / "fake_vllm.py"}" "$@"\n')
    exe.chmod(0o755)
    return manifest


def _raises() -> hardware.HostCuda:
    raise PermissionError("nvidia-smi")


def _gpu(argv: Sequence[str]) -> tuple[int, str]:
    return (0, f"Test GPU, {DRIVER}\n") if argv[0] == "nvidia-smi" else (1, "")


def _run(world: World, manifest: Manifest, **change: Any) -> run.StepOutcome:  # noqa: ANN401
    return world.run(manifest, host_runner=_gpu, **change)["infer"]


def _record(out: Path) -> Path:
    return out / run.WORK / "fake"


def _all_gone(out: Path, timeout_s: float = 30.0) -> bool:
    """Whether, within the timeout, no fake server process holds its lock (each one and its forked child exited).

    A kill is asynchronous: the kernel ends the sandbox's processes after the driver's wait returns.
    """
    alive = sorted(_record(out).glob("alive-*"))
    assert alive, "no server started"
    deadline = time.monotonic() + timeout_s
    for path in alive:
        with open(path) as handle:
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() > deadline:
                        return False
                    time.sleep(0.2)
    return True


def _summary(out: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((out / cache.SUMMARY).read_text())
    return data


@needs_userns
def test_servers_start_one_at_a_time_and_urls_reach_the_step(world: World) -> None:
    """Two servers start one after the other with the declared args; the step reaches both through loopback."""
    make_snapshot(_hub(world), REPO, REVISION, {"d.csv": b"1\n"}, repo_type="dataset")
    both = [*_weights({"d.csv": b"1\n"}, repo_type="dataset"), *_weights({"weights.bin": WEIGHTS})]
    manifest = _manifest(world, hf_weights=both)
    outcome = _run(world, manifest)
    assert outcome.ok and outcome.status == "executed", outcome.codes
    assert outcome.out is not None
    report = json.loads((outcome.out / run.WORK / "result.json").read_text())
    assert report["roots"] == [REVISION, REVISION]
    assert report["identity"] == engine.engine_identity(manifest.steps[0].engine)  # type: ignore[arg-type]
    assert report["no_proxy"] == ["127.0.0.1,localhost", "127.0.0.1,localhost"]
    lines = (_record(outcome.out) / "events.log").read_text().splitlines()
    events = [line.split() for line in lines if not line.startswith("stopped")]
    assert [e[0] for e in sorted(events, key=lambda e: float(e[2]))] == ["start", "ready", "start", "ready"]
    first, second = sorted(events, key=lambda e: float(e[2]))[:2], sorted(events, key=lambda e: float(e[2]))[2:]
    assert first[0][1] == first[1][1] and second[0][1] == second[1][1] and first[0][1] != second[0][1]
    for path in _record(outcome.out).glob("argv-*.json"):
        argv = json.loads(path.read_text())
        assert argv[0] == "serve" and Path(argv[1]).parts[-3:] == ("models--org--model", "snapshots", REVISION)
        assert argv[argv.index("--host") + 1] == "127.0.0.1"
        for flag, value in (("--gpu-memory-utilization", "0.45"), ("--max-num-seqs", "4"), ("--max-model-len", "2048")):
            assert argv[argv.index(flag) + 1] == value
        assert argv[-1] == "--enable-prefix-caching"
    logs = sorted((outcome.out / run.LOGS).glob("engine-*.log"))
    assert len(logs) == 2 and all("GET /v1/models" in p.read_text() for p in logs)
    assert not (outcome.out / run.TMP).exists()


@needs_userns
def test_a_server_that_does_not_fit_is_e_engine_fit(world: World) -> None:
    """A server that exits before it is healthy is ``E_ENGINE_FIT``: the step never runs and the store is discarded."""
    manifest = _manifest(world, engine=_engine(server_args=["--fake-nofit"]), resumable="items", resume_dir="items")
    [plan] = run.plan_steps([manifest], world.config())
    assert plan.key is not None
    out = _killed(world, plan.key)
    outcome = _run(world, manifest)
    assert outcome.status == "failed" and outcome.codes == ("E_ENGINE_FIT",) and outcome.exit != 0
    assert not (out / run.WORK / "items").exists()
    assert not (out / run.WORK / "result.json").exists()
    summary = _summary(out)
    assert summary["executed"] is False and summary["attempt"] == 2
    [log] = (out / run.LOGS).glob("engine-*.log")
    assert "KV cache cannot hold" in log.read_text()


def test_only_an_exited_server_is_e_engine_fit() -> None:
    """``E_ENGINE_FIT`` (which discards a store) is only a server that exited before it was healthy.

    An absent or malformed launcher report says nothing about the engine's fit: it is ``E_ENGINE_LAUNCH``, which is
    not an integrity code, so a resumable store is kept.
    """
    deep = b"[" * 100000 + b"]" * 100000 + b"\n"
    for report, code in (
        (b'{"status": "ready"}\n', None),
        (b"", "E_ENGINE_LAUNCH"),
        (b'{"status": "ready"}\n{"status": "ready"}\n', "E_ENGINE_LAUNCH"),
        (b'{"status": "exited"}\n', "E_ENGINE_FIT"),
        (b'{"status": "snapshot"}\n', "E_ENGINE_VERSION"),
        (b'{"status": "error"}\n', "E_ENGINE_LAUNCH"),
        (b'{"status": "other"}\n', "E_ENGINE_LAUNCH"),
        (b"\xff\n", "E_ENGINE_LAUNCH"),
        (deep, "E_ENGINE_LAUNCH"),
    ):
        assert engine.status_code(report) == code, report[:40]
    assert "E_ENGINE_LAUNCH" not in run.INTEGRITY_CODES and "E_ENGINE_FIT" in run.INTEGRITY_CODES


@needs_userns
def test_a_slow_start_is_e_engine_start_timeout_and_keeps_the_store(world: World) -> None:
    """A server not healthy within ``startup_timeout_s`` is ``E_ENGINE_START_TIMEOUT``; an attempt's store stays."""
    slow = _engine(server_args=["--fake-slow", "30"], startup_timeout_s=2, servers_per_card=1)
    manifest = _manifest(world, engine=slow, resumable="items", resume_dir="items")
    [plan] = run.plan_steps([manifest], world.config())
    assert plan.key is not None
    out = _killed(world, plan.key)
    outcome = _run(world, manifest)
    assert outcome.codes == ("E_ENGINE_START_TIMEOUT",) and outcome.exit != 0
    assert sorted(p.name for p in (out / run.WORK / "items").iterdir()) == [
        "a.done",
        "a.json",
        "b.done",
        "b.json",
        "c.json",
    ]
    assert not (out / run.WORK / "result.json").exists()
    assert _all_gone(out)

    manifest = _manifest(world, engine=_engine(servers_per_card=1), resumable="items", resume_dir="items")
    (run._step_venv(manifest.steps[0], manifest.name, world.cache)[0] / "bin" / "vllm").unlink()
    [plan] = run.plan_steps([manifest], world.config())
    assert plan.key is not None
    out = _killed(world, plan.key)
    outcome = _run(world, manifest)
    assert outcome.codes == ("E_ENGINE_LAUNCH",) and outcome.exit != 0
    assert (out / run.WORK / "items" / "a.done").is_file()


@needs_userns
def test_a_version_mismatch_is_e_engine_version(world: World) -> None:
    """Another served version, or another served snapshot, is ``E_ENGINE_VERSION``; the step never runs."""
    for change in ({"version": "0.30.0"}, {"server_args": ["--fake-root", "x/snapshots/" + "d" * 40]}):
        manifest = _manifest(world, engine=_engine(servers_per_card=1, **change))
        outcome = _run(world, manifest)
        assert outcome.codes == ("E_ENGINE_VERSION",), (change, outcome.codes)
        assert outcome.out is not None and not (outcome.out / run.WORK / "result.json").exists()
        assert _summary(outcome.out)["executed"] is False


@needs_userns
def test_a_step_that_itself_exits_94_is_e_step_failed(world: World) -> None:
    """Once the engine is ready, the step's own exit is its own: 94 is ``E_STEP_FAILED`` with exit 94."""
    outcome = _run(world, _manifest(world, "exit94"))
    assert outcome.status == "executed" and outcome.codes == ("E_STEP_FAILED",) and outcome.exit == 94
    assert outcome.out is not None and (outcome.out / run.WORK / "result.json").is_file()
    assert _summary(outcome.out)["executed"] is True
    termed = _run(world, _manifest(world, "term"))
    assert termed.codes == ("E_STEP_FAILED",) and termed.exit == 143


@needs_userns
def test_servers_die_with_the_step(world: World) -> None:
    """The servers are gone when the step ends, and when the step is killed for its timeout."""
    outcome = _run(world, _manifest(world))
    assert outcome.ok, outcome.codes
    assert outcome.out is not None and _all_gone(outcome.out)
    events = [line.split() for line in (_record(outcome.out) / "events.log").read_text().splitlines()]
    started = {e[1] for e in events if e[0] == "start"}
    assert len(started) == 2 and started <= {e[1] for e in events if e[0] == "stopped"}
    killed = _run(world, _manifest(world, "sleep"), timeout=15)
    assert killed.codes == ("E_TIMEOUT",)
    assert killed.out is not None and _all_gone(killed.out)


@needs_userns
def test_engine_identity_is_keyed_and_in_provenance(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every identity field re-keys the step and reaches the keyed env and the record; bad engines are refused."""
    spec = StepSpec.model_validate(_step(world))
    assert spec.engine is not None
    identity = engine.engine_identity(spec.engine)
    assert identity == {
        "name": "vllm",
        "version": "0.31.0",
        "server_args": [
            "--max-model-len",
            "2048",
            "--max-num-seqs",
            "4",
            "--gpu-memory-utilization",
            "0.45",
            "--kv-cache-dtype",
            "auto",
            "--generation-config",
            "vllm",
            "--limit-mm-per-prompt",
            "{}",
            "--enable-prefix-caching",
        ],
        "concurrency": 2,
        "batch_size": 4,
        "servers_per_card": 2,
    }
    keyed = sandbox.keyed_templates(spec, world.profile())
    assert json.loads(keyed["REPRO_ENGINE_IDENTITY"]) == identity
    assert (keyed["no_proxy"], keyed["NO_PROXY"]) == ("127.0.0.1,localhost", "127.0.0.1,localhost")
    for name in ("VLLM_CACHE_ROOT", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "FLASHINFER_WORKSPACE_BASE"):
        assert keyed[name].startswith("{tmp}/")
    plain = sandbox.keyed_templates(StepSpec.model_validate(world.step("p", "prep", "x")), world.profile())
    assert plain["no_proxy"] == "" and "REPRO_ENGINE_IDENTITY" not in plain and "VLLM_CACHE_ROOT" not in plain

    def key(spec: dict[str, Any]) -> str:
        manifest = world.manifest(_step(world, engine=spec))
        world.prefetch(manifest)
        [plan] = run.plan_steps([manifest], world.config())
        assert plan.key is not None, plan.codes
        return plan.key

    base = key(_engine())
    assert key(_engine(startup_timeout_s=60)) == base
    for change in (
        {"version": "0.31.1"},
        {"concurrency": 1},
        {"batch_size": 8},
        {"servers_per_card": 1},
        {"max_num_seqs": 8},
        {"max_model_len": 4096},
        {"enable_prefix_caching": False},
        {"limit_mm_per_prompt": {"image": 2}},
        {"server_args": ["--seed", "0"]},
        {"kv_cache_dtype": "fp8"},
        {"generation_config": "auto"},
    ):
        assert key(_engine(**change)) != base, change
    batched = {"name": "transformers-batched", "version": "4.57.1", "model": _engine()["model"], "concurrency": 1}
    assert key({**batched, "batch_size": 4}) != key({**batched, "batch_size": 2})

    bad: list[dict[str, Any]] = [
        {"engine": _engine(model={"repo": "org/other", "revision": REVISION})},
        {"engine": _engine(model={"repo": REPO, "revision": "e" * 40})},
        *[
            {"engine": _engine(server_args=[arg])}
            for arg in (
                "--port",
                "--max-model-len=8",
                "--max_model_len=8",
                "--gpu_memory_utilization=0.99",
                "--gpu-memory",
                "--enable-log-requests",
                "--enable_log_requests",
                "--enable-log-r",
                "--config=x",
                "/x",
                '--hf-overrides={"x":"/etc"}',
                "--download-dir=x:/abs",
            )
        ],
        {"engine": _engine(max_num_seqs=None)},
        {"resources": {"profile": "cpu", "gpu_arch": ["sm_90"]}},
        {"resources": {"profile": "cpu", "min_driver": "580.65.06"}},
        {"resources": {"profile": "cpu", "gpu_arch": ["sm_90"], "min_driver": "570.86.15"}},
        {"engine": {**batched, "batch_size": 1, "server_args": ["--seed", "0"]}},
    ]
    for change in bad:
        with pytest.raises(ValidationError, match="E_SPEC_ENGINE"):
            StepSpec.model_validate(_step(world, **change))

    monkeypatch.setattr(
        hardware, "detect_host_cuda", lambda: hardware.HostCuda((13, 0), "nvidia-smi", "CUDA Version: 13.0")
    )
    record = host.host_record(None, runner=_gpu)
    assert (record["gpu_name"], record["gpu_driver"], record["host_cuda"]) == ("Test GPU", DRIVER, "13.0")
    for probe in (lambda: hardware.HostCuda((12, 9), "nvcc", "release 12.9"), _raises):
        with monkeypatch.context() as patched:
            patched.setattr(hardware, "detect_host_cuda", probe)
            assert host.host_record(None, runner=_gpu)["host_cuda"] is None
    manifest = _manifest(world)
    outcome = _run(world, manifest)
    assert outcome.ok and outcome.out is not None, outcome.codes
    records = [json.loads(line) for line in (outcome.out / "prov" / "store.jsonl").read_text().splitlines()]
    [parameters] = [r["parameters"] for r in records if "parameters" in r]
    assert parameters["details"]["engine"] == identity
    assert parameters["host"]["host_cuda"] == "13.0" and parameters["host"]["gpu_driver"] == DRIVER
