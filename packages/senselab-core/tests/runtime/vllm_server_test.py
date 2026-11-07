"""Tests for `senselab_core.runtime.vllm_server`: lifecycle and attestation against `fake_vllm_fixture.py`."""

import json
import os
import signal
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pytest
from lease_time_fixture import install
from senselab_core.kernel.identity import ExecutionContext, ModelRef, ResolvedModel
from senselab_core.runtime import vllm_server as vs
from senselab_core.runtime.envs import EnvHandle
from senselab_core.runtime.execution_settings import JobLease
from senselab_core.testing.processes import group_gone
from senselab_worker.digest import identity_digest

pytestmark = pytest.mark.logic

SHA = "a" * 40
MODEL = ResolvedModel(ref=ModelRef.parse(f"org/model@{SHA}"), identity=SHA)
SETTINGS: dict[str, Any] = {
    "max_model_len": 4096,
    "max_num_seqs": 4,
    "gpu_memory_utilization": 0.9,
    "kv_cache_dtype": "auto",
    "generation_config": "vllm",
    "limit_mm_per_prompt": {"image": 0},
    "enable_prefix_caching": False,
    "flashinfer_sampler": False,
}
FAKE = Path(__file__).with_name("fake_vllm_fixture.py")
LEASE = JobLease(stale_after_s=30, renew_fraction=0.01, attribute_delay_s=0)

_HOLDER = r"""
import sys, time
from pathlib import Path
from senselab_core.runtime.envs import EnvHandle
from senselab_worker.digest import identity_digest
from senselab_core.runtime.vllm_server import VllmServer
from senselab_core.kernel.identity import ModelRef, ResolvedModel
from senselab_core.runtime.execution_settings import JobLease

python, locks, sha, settings = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
env = EnvHandle(python=python, python_version="3.12", lock_sha="l", wheel_set="w", torch_index=None)
model = ResolvedModel(ref=ModelRef.parse(f"org/model@{sha}"), identity=sha)
lease = JobLease(stale_after_s=30, renew_fraction=0.1, attribute_delay_s=0)
kwargs = {"start_lock_dir": Path(locks), "startup_timeout_s": 30, "start_margin_s": 1, "lease": lease}
with VllmServer(env, model, __import__("json").loads(settings), **kwargs):
    print("up", flush=True)
    time.sleep(600)
"""


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EnvHandle:
    """A server env whose `bin/vllm` is the fake; the fake's records go to `<tmp>/fake`."""
    out = tmp_path / "fake"
    out.mkdir()
    exe = tmp_path / "env" / "bin" / "vllm"
    exe.parent.mkdir(parents=True)
    knobs = 'FAKE_VLLM_KNOBS="$(dirname "$0")/fake_vllm.json"'
    exe.write_text(f'#!/bin/sh\n{knobs} exec "{sys.executable}" "{FAKE}" "$@"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    _knob(tmp_path, FAKE_VLLM_DIR=str(out))
    return EnvHandle(
        python=str(exe.with_name("python")),
        python_version="3.12",
        lock_sha="l" * 8,
        wheel_set="w" * 8,
        torch_index=None,
    )


def _knob(tmp_path: Path, **knobs: str) -> None:
    """Set `knobs` for the fake server of the `env` fixture."""
    path = tmp_path / "env" / "bin" / "fake_vllm.json"
    path.write_text(json.dumps({**(json.loads(path.read_text()) if path.exists() else {}), **knobs}))


def _server(env: EnvHandle, tmp_path: Path, settings: dict[str, Any] = SETTINGS) -> vs.VllmServer:
    return vs.VllmServer(
        env, MODEL, settings, start_lock_dir=tmp_path / "locks", startup_timeout_s=30, start_margin_s=1, lease=LEASE
    )


def _records(tmp_path: Path) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted((tmp_path / "fake").glob("*.json"))]


def _context(fields: dict[str, Any]) -> ExecutionContext:
    return ExecutionContext(
        **fields,
        platform="linux-x86_64",
        numerics={},
        decoder=None,
        host="h",
        protocol="p",
        model_source="hf",
        env_license=None,
    )


def test_attestation_enters_context(env: EnvHandle, tmp_path: Path) -> None:
    """The served version, the model commit and the config digest reach the context and every reply."""
    with _server(env, tmp_path) as server:
        attestation = server.attestation
        assert attestation == vs.Attestation(
            engine_version="0.31.0",
            served_model="org/model",
            model_identity=SHA,
            config_digest=identity_digest(vs.CONFIG_TAG, [*server.args, "VLLM_USE_FLASHINFER_SAMPLER=0"]),
        )
        driver = _context(server.context_fields()).driver
        assert {k: driver[k] for k in attestation.driver()} == attestation.driver()
        assert server.chat([{"role": "user", "content": "hi"}], sampling={}).attestation == attestation
    (record,) = _records(tmp_path)
    assert record["argv"][1] == "org/model"
    assert {f"--revision={SHA}", f"--tokenizer-revision={SHA}", "--max-num-seqs=4"} <= set(record["argv"])


def test_an_ambient_engine_variable_never_reaches_the_server(
    env: EnvHandle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the declared engine variables reach the server; an ambient `VLLM_*` neither passes nor moves the digest."""
    launches = []
    for backend in (None, "FLASH_ATTN"):
        if backend is not None:
            monkeypatch.setenv("VLLM_ATTENTION_BACKEND", backend)
        with _server(env, tmp_path) as server:
            launches.append((server.launch, _context(server.context_fields()).driver["launch_env"]))
    assert "VLLM_ATTENTION_BACKEND" not in launches[1][0].variables
    assert launches[0][0].variables["VLLM_USE_FLASHINFER_SAMPLER"] == "0" and launches[0][1] == launches[1][1]


def test_restart_with_new_config_changes_attestation(env: EnvHandle, tmp_path: Path) -> None:
    """A restart with another config attests another digest and context; the same config attests the same."""
    contexts = []
    for settings in (SETTINGS, {**SETTINGS, "max_num_seqs": 8}, SETTINGS):
        with _server(env, tmp_path, settings) as server:
            contexts.append(_context(server.context_fields()).fingerprint())
    assert contexts[0] != contexts[1] and contexts[0] == contexts[2]


def test_engine_variables_reach_the_server_and_its_digest(env: EnvHandle, tmp_path: Path) -> None:
    """`flashinfer_sampler` is set on the server process as `VLLM_USE_FLASHINFER_SAMPLER` and changes the digest."""
    digests, seen = [], []
    for on in (False, True):
        with _server(env, tmp_path, {**SETTINGS, "flashinfer_sampler": on}) as server:
            digests.append(server.config_digest)
        seen.append({r["sampler"] for r in _records(tmp_path)})
    assert seen == [{"0"}, {"0", "1"}] and digests[0] != digests[1]


def test_server_stops_with_owner(env: EnvHandle, tmp_path: Path) -> None:
    """Killing the owning process with SIGKILL takes the server's whole process group with it."""
    holder = subprocess.Popen(  # noqa: S603 - this interpreter
        [sys.executable, "-c", _HOLDER, env.python, str(tmp_path / "locks"), SHA, json.dumps(SETTINGS)],
        stdout=subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
    )
    try:
        assert holder.stdout is not None and holder.stdout.readline() == b"up\n"
        (record,) = _records(tmp_path)
        holder.send_signal(signal.SIGKILL)
        holder.wait()
        assert group_gone(record["pgid"], 30)
    finally:
        holder.kill()
        holder.wait()


def test_server_stops_on_exit(env: EnvHandle, tmp_path: Path) -> None:
    """Leaving the context stops the group and ends the attestation."""
    with _server(env, tmp_path) as server:
        pass
    (record,) = _records(tmp_path)
    assert group_gone(record["pgid"], 5)
    assert server.attestation is None
    with pytest.raises(vs.StaleAttestation):
        server.context_fields()


def test_reply_from_another_model_is_stale(env: EnvHandle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A reply naming another model does not carry the attestation."""
    _knob(tmp_path, FAKE_VLLM_REPLY_MODEL="org/other")
    with _server(env, tmp_path) as server, pytest.raises(vs.StaleAttestation, match="another model than org/model@"):
        server.chat([{"role": "user", "content": "hi"}], sampling={})


@pytest.mark.parametrize(
    ("variable", "kind"), [("FAKE_VLLM_EXIT", "exited"), ("FAKE_VLLM_ROOT", "model"), ("FAKE_VLLM_VERSION", "version")]
)
def test_start_failure_is_typed_and_stops(
    env: EnvHandle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variable: str, kind: str
) -> None:
    """A server that exits, serves another model or reports no version raises its kind and leaves nothing."""
    _knob(tmp_path, **{variable: {"FAKE_VLLM_EXIT": "1", "FAKE_VLLM_ROOT": "org/other"}.get(variable, "")})
    server = _server(env, tmp_path)
    with pytest.raises(vs.VllmStartError) as raised:
        server.__enter__()
    assert raised.value.kind == kind and server.url is None
    for record in _records(tmp_path):
        assert group_gone(record["pgid"], 5)


@pytest.mark.parametrize(
    "model",
    [
        ResolvedModel(ref=ModelRef.parse("url:https://h/m.bin#sha256=" + "b" * 64), identity="b" * 64),
        ResolvedModel(ref=ModelRef.parse("org/model@main"), identity="main", unresolved_reason="a ref"),
    ],
)
def test_url_or_unpinned_model_refused(env: EnvHandle, tmp_path: Path, model: ResolvedModel) -> None:
    """Only a pinned hub or local model is served."""
    with pytest.raises(ValueError, match="pinned|url"):
        vs.VllmServer(env, model, SETTINGS, start_lock_dir=tmp_path, startup_timeout_s=1, start_margin_s=1, lease=LEASE)


@pytest.mark.parametrize("settings", [{k: v for k, v in SETTINGS.items() if k != "max_num_seqs"}, {**SETTINGS, "x": 1}])
def test_settings_must_be_the_known_keys(settings: dict[str, Any]) -> None:
    """A missing or an unknown setting is refused, never dropped."""
    with pytest.raises(ValueError, match="vLLM settings"):
        vs.server_args(settings)


def test_server_args_bind_values_to_flags() -> None:
    """Each value is joined to its flag, so no value is read as another flag."""
    args = vs.server_args({**SETTINGS, "kv_cache_dtype": "--enable-log-requests"})
    assert "--kv-cache-dtype=--enable-log-requests" in args
    assert args[-2:] == ['--limit-mm-per-prompt={"image":0}', "--no-enable-prefix-caching"]


def test_local_model_errors_name_no_path(env: EnvHandle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A local model is served by its path, and an error names it by identity, never by that path."""
    weights = tmp_path / "private-weights"
    model = ResolvedModel(ref=ModelRef.parse(f"local:{weights}"), identity="c" * 64)
    _knob(tmp_path, FAKE_VLLM_ROOT="elsewhere")
    server = vs.VllmServer(
        env, model, SETTINGS, start_lock_dir=tmp_path / "locks", startup_timeout_s=30, start_margin_s=1, lease=LEASE
    )
    with pytest.raises(vs.VllmStartError) as raised:
        server.__enter__()
    assert "private-weights" not in str(raised.value) and "c" * 12 in str(raised.value)
    (record,) = _records(tmp_path)
    assert record["argv"][1] == str(weights) and not any(a.startswith("--revision") for a in record["argv"])


@pytest.mark.parametrize("kind", ["audio", "image"])
def test_unreadable_media_error_names_no_path(tmp_path: Path, kind: str) -> None:
    """A missing media file is refused by its suffix, never its path."""
    with pytest.raises(ValueError, match="cannot read a '.wav' file") as raised:
        vs.content_part(kind, tmp_path / "secret-name.wav")
    assert "secret-name" not in str(raised.value) and raised.value.__cause__ is None


def test_the_start_lease_outlasts_its_stale_bound_while_it_beats(
    env: EnvHandle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """While the server starts, the storage clock passes the start lease's stale bound; it beats and is never broken."""
    clock, real_popen = install(monkeypatch), subprocess.Popen
    started: list[int] = []

    class Popen(real_popen):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            if not started:
                started.append(1)
                clock.outlast(tmp_path / "locks", LEASE.stale_after_s)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", Popen)
    with _server(env, tmp_path) as server:
        assert server.attestation is not None
    assert clock.attempts == 4 and clock.breaks == 0


def test_a_start_lease_lost_after_the_launch_keeps_the_healthy_server(
    env: EnvHandle, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The start lease only orders starts: lost while the launched server comes up, it is logged and the server kept."""
    real_mkdir, real_popen = os.mkdir, subprocess.Popen
    failing, failed = threading.Event(), threading.Event()

    def mkdir(path: Any, mode: int = 0o777, *, dir_fd: int | None = None) -> None:  # noqa: ANN401
        if failing.is_set() and os.fsdecode(path).startswith("beat-"):
            failed.set()
            raise FileNotFoundError(2, "No such file or directory", os.fsdecode(path))
        real_mkdir(path, mode, dir_fd=dir_fd)

    class Popen(real_popen):  # type: ignore[misc,valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
            if not failing.is_set():
                failing.set()
                assert failed.wait(10)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(os, "mkdir", mkdir)
    monkeypatch.setattr(subprocess, "Popen", Popen)
    with _server(env, tmp_path) as server:
        assert server.attestation is not None
