"""B14 re-check: a confined R library, the venv-bin symlink guard, masked submounts, the sandbox-unavailable code.

Every site is synthetic, under ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

from senselab.utils.reproduce import cache, env, run, sandbox
from tests.utils.reproduce.env_test import FakeR, _profile, _r_env
from tests.utils.reproduce.run_test import World, _py, _r_step, _session

needs_userns = pytest.mark.skipif(not sandbox.userns_available(), reason="unprivileged user namespaces are unavailable")
WRITE = "import json, sys; open(sys.argv[1] + '/result.json', 'w').write(json.dumps({'value': 2.0}))"
BAD_R_LIBS = [
    "{cache}/../victim",
    "{cache}/..",
    "{cache}/venvs/repro-demo-00000000/bin",
    "{cache}/r/a/b",
    "{cache}/r/..",
]


@pytest.fixture
def world(tmp_path: Path) -> World:
    """A fresh synthetic site."""
    return World(tmp_path)


# --- r_lib confinement ----------------------------------------------------------------------------------


@pytest.mark.parametrize("r_lib", BAD_R_LIBS)
def test_r_lib_must_be_one_plain_name_under_cache_r(r_lib: str) -> None:
    """Anything but ``{cache}/r/<Name>`` is ``E_SPEC_BAD_R_LIB``."""
    with pytest.raises(ValidationError, match="E_SPEC_BAD_R_LIB"):
        _r_env(r_lib=r_lib)
    assert _r_env(r_lib="{cache}/r/lme4_1.1").r_lib == "{cache}/r/lme4_1.1"


@pytest.mark.parametrize("r_lib", BAD_R_LIBS)
def test_prefetch_r_never_touches_a_library_outside_cache_r(tmp_path: Path, r_lib: str) -> None:
    """With validation bypassed, prefetch still refuses before any removal, write or lock; the victim survives."""
    cache_dir = tmp_path / "cache"
    (cache_dir / "venvs" / "repro-demo-00000000" / "bin").mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("precious")
    bad = _r_env().model_copy(update={"r_lib": r_lib})
    with pytest.raises(ValueError, match="E_SPEC_BAD_R_LIB"):
        env.prefetch_r(bad, cache_dir, _profile(tmp_path), runner=FakeR({}))
    assert (victim / "keep.txt").read_text() == "precious"
    assert (cache_dir / "venvs" / "repro-demo-00000000" / "bin").is_dir()
    assert not list(tmp_path.glob("*.lock")) and not list(cache_dir.glob("*.lock"))


def test_a_symlinked_r_dir_cannot_lead_out_of_the_cache(tmp_path: Path) -> None:
    """``{cache}/r`` itself a symlink elsewhere: the real path is outside, so the library is refused."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (cache_dir / "r").symlink_to(elsewhere)
    with pytest.raises(ValueError, match="E_SPEC_BAD_R_LIB"):
        env.confined_r_lib("{cache}/r/demo", cache_dir)


@needs_userns
def test_a_run_refuses_an_r_lib_outside_cache_r(world: World) -> None:
    """A bypassed manifest naming ``{cache}/../victim`` fails the step before anything is read or locked."""
    manifest = world.manifest(_r_step(world, _session(world)))
    world.prefetch(manifest)
    step = manifest.steps[0]
    bad_env = step.env.model_copy(update={"r_lib": "{cache}/../victim"})
    bad = manifest.model_copy(update={"steps": [step.model_copy(update={"env": bad_env})]})
    outcome = world.run(bad)["prep"]
    assert outcome.status == "failed" and outcome.codes == ("E_SPEC_BAD_R_LIB",)


# --- the venv-bin symlink guard ---------------------------------------------------------------------------


@needs_userns
def test_a_venv_bin_symlink_to_a_system_python_is_refused(world: World) -> None:
    """``<venv>/bin/evilpy -> /usr/bin/python3`` is found in the venv's bin but leads outside it."""
    data = _py(world, WRITE)
    manifest = world.manifest(data)
    world.prefetch(manifest)
    path, _ = run._step_venv(manifest.steps[0], manifest.name, world.cache)
    (path / "bin" / "evilpy").symlink_to("/usr/bin/python3")
    data["argv"][0] = "evilpy"
    outcome = world.run(world.manifest(data))["prep"]
    assert "E_STEP_INTERPRETER" in outcome.codes and outcome.status == "failed"
    data["argv"][0] = "python"
    assert world.run(world.manifest(data))["prep"].ok


def test_python_home_is_read_from_pyvenv_cfg(tmp_path: Path) -> None:
    """The uv-managed install is the parent of ``home`` in ``pyvenv.cfg``; nothing is executed."""
    (tmp_path / "pyvenv.cfg").write_text("home = /opt/py/cpython-3.12/bin\ninclude-system-site-packages = false\n")
    assert run.python_install_root(tmp_path) == "/opt/py/cpython-3.12"
    (tmp_path / "pyvenv.cfg").write_text("version = 3.12\n")
    assert run.python_install_root(tmp_path) is None


# --- the sweep and masked submounts ----------------------------------------------------------------------


@needs_userns
def test_a_mount_under_a_masked_dir_does_not_fail_the_sweep(tmp_path: Path) -> None:
    """A tmpfs under a masked dir (like ``/run/user/<uid>`` under ``/run``) is skipped: the sandbox still runs."""
    (tmp_path / "m" / "sub").mkdir(parents=True)
    (tmp_path / "w").mkdir()
    inner = sandbox.sandbox_argv(
        ["/bin/sh", "-c", f"touch {tmp_path / 'w' / 'ran'}"], [], [tmp_path / "w"], masked=[tmp_path / "m"]
    )
    outer = ["/usr/bin/unshare", "-rm", "/bin/sh", "-c", f'mount -t tmpfs tmpfs {tmp_path / "m" / "sub"} && exec "$@"']
    done = subprocess.run([*outer, "sh", *inner], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"})
    assert done.returncode == 0, done.stderr
    assert (tmp_path / "w" / "ran").exists()


# --- the sandbox-unavailable code ------------------------------------------------------------------------


def test_sandbox_status_separates_no_userns_from_a_refused_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    """A sweep that leaves a mount writable is ``sweep-refused`` with its message; other failures ``no-userns``."""

    def fake(stderr: str, code: int) -> object:
        return lambda *a, **k: subprocess.CompletedProcess(a, code, b"", stderr.encode())

    monkeypatch.setattr(sandbox.subprocess, "run", fake("reproduce sandbox: mounts left writable: fuse\n", 1))
    assert sandbox.sandbox_status() == ("sweep-refused", "reproduce sandbox: mounts left writable: fuse")
    assert sandbox.userns_available() is False
    monkeypatch.setattr(sandbox.subprocess, "run", fake("unshare: unshare failed: Operation not permitted\n", 1))
    assert sandbox.sandbox_status()[0] == "no-userns"
    monkeypatch.setattr(sandbox.subprocess, "run", fake("", 0))
    assert sandbox.sandbox_status() == ("ok", "")


def test_a_refused_sweep_is_e_sandbox_unavailable_with_driver_log(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A node whose mounts cannot all be made read-only refuses the step with its own code and logs why."""
    manifest = world.manifest(_py(world, WRITE))
    world.prefetch(manifest)
    message = "reproduce sandbox: mounts left writable: fuse"
    monkeypatch.setattr(sandbox, "sandbox_status", lambda: ("sweep-refused", message))
    outcome = world.run(manifest)["prep"]
    assert outcome.status == "failed" and outcome.codes == ("E_SANDBOX_UNAVAILABLE",)
    assert outcome.key is not None
    log = cache.result_dir(world.profile().results_root, outcome.key) / run.LOGS / run.DRIVER_LOG
    assert message in log.read_text()
    monkeypatch.setattr(sandbox, "sandbox_status", lambda: ("no-userns", "unshare failed"))
    assert world.run(manifest)["prep"].codes == ("E_NO_USERNS",)


# --- ir_notebook kernel specs ------------------------------------------------------------------------------


def test_kernel_specs_are_recorded_and_a_change_is_seen(tmp_path: Path) -> None:
    """``record_kernels`` stores the kernels digest in installed.json and keeps the marker in step."""
    venv_dir = tmp_path / "v"
    kernel = venv_dir / "share" / "jupyter" / "kernels" / "ir"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text('{"argv": ["R"]}')
    data = json.dumps({"packages": {}, "freeze": []}).encode()
    (venv_dir / env.INSTALLED).write_bytes(data)
    marker: dict[str, object] = {
        "env_key": "k",
        "overlay_sha256": None,
        "wheels": {},
        "installed_sha256": hashlib.sha256(data).hexdigest(),
    }
    (venv_dir / env.MARKER).write_text(json.dumps(marker))
    env.record_kernels(venv_dir)
    installed = json.loads((venv_dir / env.INSTALLED).read_text())
    assert installed["kernels_sha256"] == env.kernels_sha256(venv_dir)
    assert json.loads((venv_dir / env.MARKER).read_text())["installed_sha256"] == env.installed_sha256(venv_dir)
    (kernel / "kernel.json").write_text('{"argv": ["/elsewhere/R"]}')
    assert env.kernels_sha256(venv_dir) != installed["kernels_sha256"]


@needs_userns
def test_a_changed_kernel_spec_is_e_env_changed(world: World) -> None:
    """An ``ir_notebook`` step refuses when the notebook venv's kernel specs changed since prefetch."""
    data = _r_step(world, _session(world))
    data["kind"] = "ir_notebook"
    data["argv"][0] = "{venv}/bin/python"
    data["env"]["notebook_tools"] = {"lock": "nb.lock", "lock_sha256": "ef" * 32}
    data["env"]["r_packages"].append({"name": "IRkernel", "version": "1.3.2"})
    manifest = world.manifest(data)
    world.prefetch(manifest)
    path, _ = run._step_venv(manifest.steps[0], manifest.name, world.cache)
    kernel = path / "share" / "jupyter" / "kernels" / "ir"
    kernel.mkdir(parents=True)
    (kernel / "kernel.json").write_text("{}")
    env.record_kernels(path)
    (kernel / "kernel.json").write_text('{"argv": ["/elsewhere"]}')
    assert world.run(manifest)["prep"].codes == ("E_ENV_CHANGED",)
    (kernel / "kernel.json").write_text("{}")
    assert world.run(manifest)["prep"].ok
