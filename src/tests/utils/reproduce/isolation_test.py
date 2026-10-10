"""Isolation tests: the step interpreter, R startup files, sandboxed venv checks and R installs, host sockets.

Every repository, venv, data root and result tree is synthetic, under ``tmp_path``.
"""

from __future__ import annotations

import fcntl
import io
import json
import os
import shutil
import subprocess
import tarfile
import venv as stdlib_venv
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from senselab.utils.reproduce import build, env, run, sandbox, source
from senselab.utils.reproduce.spec import StepSpec
from tests.utils.reproduce.run_test import FIXTURE_ENV, World, _py, _r_step, _session

needs_userns = pytest.mark.skipif(not sandbox.userns_available(), reason="unprivileged user namespaces are unavailable")
WRITE = "import json, sys; open(sys.argv[1] + '/result.json', 'w').write(json.dumps({'value': 2.0}))"


@pytest.fixture
def world(tmp_path: Path) -> World:
    """A fresh synthetic site."""
    return World(tmp_path)


# --- B1: the step interpreter -------------------------------------------------------------------------


def test_spec_refuses_absolute_literal_paths(world: World) -> None:
    """An argv or pythonpath entry that is an absolute path with no placeholder is refused."""
    data = world.step("prep", "prep", "{root:data}/in/a.txt", argv=["/usr/bin/python3", "-c", "1"])
    with pytest.raises(ValidationError, match="E_SPEC_ABSOLUTE_PATH"):
        StepSpec.model_validate(data)
    data = world.step("prep", "prep", "{root:data}/in/a.txt", pythonpath=["/opt/lib"])
    with pytest.raises(ValidationError, match="E_SPEC_ABSOLUTE_PATH"):
        StepSpec.model_validate(data)
    StepSpec.model_validate(world.step("prep", "prep", "{root:data}/in/a.txt", pythonpath=["{checkout}/src"]))


@needs_userns
@pytest.mark.parametrize("argv0", ["/usr/bin/python3", "sh"])
def test_an_interpreter_outside_the_venv_never_runs(world: World, argv0: str) -> None:
    """The reviewer's ``/usr/bin/python3`` (spec bypassed) and a bare ``sh`` on PATH are ``E_STEP_INTERPRETER``."""
    data = _py(world, WRITE)
    manifest = world.manifest(data)
    step = manifest.steps[0].model_copy(update={"argv": [argv0, "-c", "true", "{out}"]})
    manifest = manifest.model_copy(update={"steps": [step]})
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.status == "failed" and "E_STEP_INTERPRETER" in outcome.codes and outcome.exit != 0
    assert outcome.out is not None and not (outcome.out / run.WORK / "result.json").exists()


@needs_userns
def test_a_bare_python_resolves_into_the_venv(world: World) -> None:
    """``python`` found on the step's PATH is the venv's interpreter and runs."""
    data = _py(world, WRITE)
    data["argv"][0] = "python"
    manifest = world.manifest(data)
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.ok, outcome.codes


@needs_userns
def test_an_r_step_runs_only_an_r_module_or_native_program(world: World) -> None:
    """An R step may start a program from the R module's bin; the venv's python is ``E_STEP_INTERPRETER``."""
    data = _r_step(world, _session(world))
    data["argv"][0] = "{venv}/bin/python"
    manifest = world.manifest(data)
    world.prefetch(manifest)
    refused = world.run(manifest)["prep"]
    assert "E_STEP_INTERPRETER" in refused.codes
    data["argv"][0] = "rpy"
    manifest = world.manifest(data)
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.ok, outcome.codes


# --- S1: R startup files ---------------------------------------------------------------------------------


@needs_userns
def test_r_startup_files_are_pinned_and_the_session_runs_in_the_step_cwd(world: World) -> None:
    """``R_ENVIRON_USER``/``R_PROFILE_USER`` name an empty driver file; a planted ``.Renviron`` is not it."""
    (world.repo / ".Renviron").write_text("R_LIBS_USER=/elsewhere\n")
    (world.repo / ".Rprofile").write_text(".libPaths('/elsewhere')\n")
    from tests.utils.reproduce.run_test import _git

    _git(world.repo, "add", "-A")
    _git(world.repo, "commit", "-q", "-m", "startup files")
    world.commit = _git(world.repo, "rev-parse", "HEAD").strip()
    data = _r_step(world, _session(world))
    data["argv"][0] = "rpy"
    code = (
        "import json, os, sys\n"
        "vals = {k: os.environ.get(k) for k in ('R_ENVIRON_USER', 'R_PROFILE_USER')}\n"
        "vals['sizes'] = [os.path.getsize(v) for v in list(vals.values()) if v]\n"
        "vals['cwd'] = os.getcwd()\n"
        "open(sys.argv[1] + '/startup.json', 'w').write(json.dumps(vals))\n"
        "open(sys.argv[1] + '/result.json', 'w').write(json.dumps({'value': 2.0}))\n"
    )
    data["argv"][2] = code.replace("{", "{{").replace("}", "}}")
    manifest = world.manifest(data)
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.ok, outcome.codes
    assert outcome.out is not None
    seen = json.loads((outcome.out / run.WORK / "startup.json").read_text())
    assert seen["R_ENVIRON_USER"] == seen["R_PROFILE_USER"] and seen["sizes"] == [0, 0]
    assert not seen["R_ENVIRON_USER"].startswith(seen["cwd"])
    keyed = sandbox.keyed_templates(manifest.steps[0], world.profile())
    assert keyed["R_ENVIRON_USER"] == keyed["R_PROFILE_USER"] == sandbox.R_STARTUP_TEMPLATE


@needs_userns
def test_the_r_session_check_runs_in_the_step_cwd(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """The fake Rscript records its working directory: the step's cwd, the checkout."""
    data = _r_step(world, _session(world))
    rscript = world.tmp / "rbin" / "Rscript"
    rscript.write_text(rscript.read_text() + 'pwd > "$out/cwd.txt"\n')
    data["argv"][0] = "rpy"
    manifest = world.manifest(data)
    world.prefetch(manifest)
    seen: dict[str, str] = {}
    original = env.read_r_session

    def read(out: Path) -> dict[str, Any]:
        seen["cwd"] = (out / "cwd.txt").read_text().strip()
        return original(out)

    monkeypatch.setattr(env, "read_r_session", read)
    outcome = world.run(manifest)["prep"]
    assert outcome.ok, outcome.codes
    checkout = source.prefetched_checkout(manifest.steps[0], cache=world.cache).worktree
    assert seen["cwd"] == os.path.realpath(checkout)


# --- S2: venv checks in the sandbox ------------------------------------------------------------------------


def _pth_venv(tmp_path: Path) -> tuple[Path, Path]:
    venv_dir = tmp_path / "venv"
    stdlib_venv.EnvBuilder(symlinks=True, with_pip=False).create(venv_dir)
    marker = tmp_path / "pth-ran"
    (site,) = list((venv_dir / "lib").glob("python*/site-packages"))
    (site / "zz_marker.pth").write_text(f"import pathlib; pathlib.Path({str(marker)!r}).write_text('ran')\n")
    return venv_dir, marker


@needs_userns
@pytest.mark.skipif(shutil.which("uv") is None and not os.environ.get("REPRO_UV"), reason="uv is not available")
def test_pip_check_and_freeze_run_pth_code_only_inside_the_sandbox(tmp_path: Path) -> None:
    """A ``.pth`` file that writes a marker cannot write it: uv's interpreter query runs read-only, offline."""
    venv_dir, marker = _pth_venv(tmp_path)
    control = subprocess.run([str(venv_dir / "bin" / "python"), "-c", "pass"], check=False)
    assert control.returncode == 0 and marker.exists()
    marker.unlink()
    assert env.pip_check(venv_dir) == []
    assert env.pip_freeze(venv_dir) == ""
    assert not marker.exists()


# --- S4: the installed set at run time ----------------------------------------------------------------------


def test_installed_set_from_metadata(tmp_path: Path) -> None:
    """Names and versions come from dist-info METADATA, normalised, local CUDA tags dropped; nothing runs."""
    site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages"
    for name, version in (("Foo_Bar", "1.0"), ("torch", "2.8.0+cu128")):
        meta = site / f"{name}-{version}.dist-info"
        meta.mkdir(parents=True)
        (meta / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\nbody\n")
    (site / "legacy.egg-info").mkdir()
    assert env.installed_from_metadata(tmp_path / "venv") == {"foo-bar": "1.0", "torch": "2.8.0"}


@needs_userns
def test_an_out_of_band_dist_info_is_e_env_changed(world: World) -> None:
    """A distribution added to the venv after prefetch makes the run refuse with ``E_ENV_CHANGED``."""
    manifest = world.manifest(_py(world, WRITE))
    world.prefetch(manifest)
    path, _ = run._step_venv(manifest.steps[0], manifest.name, world.cache)
    (site,) = list((path / "lib").glob("python*/site-packages"))
    meta = site / "intruder-9.9.dist-info"
    meta.mkdir()
    (meta / "METADATA").write_text("Metadata-Version: 2.1\nName: intruder\nVersion: 9.9\n")
    outcome = world.run(manifest)["prep"]
    assert outcome.codes == ("E_ENV_CHANGED",) and outcome.status == "failed"


# --- S5 and the cache mask: hidden host paths ----------------------------------------------------------------


@needs_userns
def test_host_socket_dirs_are_masked(tmp_path: Path) -> None:
    """``/run`` (and the other masked dirs that exist) are empty inside the sandbox."""
    present = [p for p in sandbox.MASKED_HOST_DIRS if os.path.isdir(p) and not os.path.islink(p) and os.listdir(p)]
    if not present:
        pytest.skip("no non-empty host socket dir to mask")
    script = "; ".join(f'test -z "$(ls -A {p})"' for p in present)
    done = subprocess.run(
        sandbox.sandbox_argv(["/bin/sh", "-c", script], [], [tmp_path]),
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr


@needs_userns
def test_the_hugging_face_cache_is_hidden_from_steps_without_weights(world: World) -> None:
    """With ``hf_weights: []`` the step sees ``{cache}/huggingface`` empty."""
    hf = world.cache / "huggingface"
    hf.mkdir(parents=True)
    (hf / "secret-model").write_text("weights")
    code = (
        "import json, os, sys\n"
        "open(sys.argv[1] + '/hf.json', 'w').write(json.dumps(os.listdir(sys.argv[2] + '/huggingface')))\n" + WRITE
    )
    manifest = world.manifest(_py(world, code))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.ok, outcome.codes
    assert outcome.out is not None and json.loads((outcome.out / run.WORK / "hf.json").read_text()) == []


# --- nits ---------------------------------------------------------------------------------------------------


def test_more_loader_and_r_variables_are_reserved(world: World) -> None:
    """``LD_AUDIT``, ``GCONV_PATH``, ``R_ENVIRON*`` and ``R_PROFILE*`` cannot come from the profile env."""
    step = world.manifest(_py(world, WRITE)).steps[0]
    for name in ("LD_AUDIT", "GCONV_PATH", "R_ENVIRON", "R_ENVIRON_USER", "R_PROFILE", "R_PROFILE_USER"):
        world.env_map = {name: "x"}
        with pytest.raises(ValueError, match="driver-owned"):
            sandbox.keyed_templates(step, world.profile())


@needs_userns
def test_a_symlink_out_of_work_fails_the_step(world: World) -> None:
    """An output that is a symlink pointing outside ``work/`` is ``E_OUTPUT_SYMLINK`` and is not followed."""
    code = "import os, sys\nos.symlink('/etc/hostname', sys.argv[1] + '/result.json')\n"
    manifest = world.manifest(_py(world, code))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert "E_OUTPUT_SYMLINK" in outcome.codes and outcome.exit != 0


@needs_userns
def test_a_step_holds_a_shared_lock_on_its_venv(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """While the step runs, an exclusive lock on the venv's lock file (a rebuild) cannot be taken."""
    manifest = world.manifest(_py(world, WRITE))
    world.prefetch(manifest)
    path, _ = run._step_venv(manifest.steps[0], manifest.name, world.cache)
    lock = path.with_name(path.name + ".lock")
    seen: dict[str, bool] = {}
    real = run._Execution._exec

    def probe(self: Any) -> None:  # noqa: ANN401
        fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o660)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            seen["exclusive"] = True
            fcntl.flock(fd, fcntl.LOCK_UN)
        except BlockingIOError:
            seen["exclusive"] = False
        finally:
            os.close(fd)
        real(self)

    monkeypatch.setattr(run._Execution, "_exec", probe)
    outcome = world.run(manifest)["prep"]
    assert outcome.ok, outcome.codes
    assert seen == {"exclusive": False}


def test_verify_checkout_counts_ignored_files(world: World) -> None:
    """An ignored file planted in the worktree makes it dirty."""
    (world.repo / ".gitignore").write_text("planted*\n")
    from tests.utils.reproduce.run_test import _git

    _git(world.repo, "add", "-A")
    _git(world.repo, "commit", "-q", "-m", "ignore")
    world.commit = _git(world.repo, "rev-parse", "HEAD").strip()
    step = world.manifest(_py(world, WRITE)).steps[0]
    co = source.checkout_step(step, cache=world.cache, base_dir=world.base)
    source.verify_checkout(co)
    (co.worktree / "planted.py").write_text("x = 1\n")
    with pytest.raises(source.ReproduceSourceError) as info:
        source.verify_checkout(co)
    assert info.value.code == source.WORKTREE_DIRTY


def _tar(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as archive:
        for info, data in members:
            archive.addfile(info, io.BytesIO(data) if data is not None else None)
    return buf.getvalue()


def _member(name: str, kind: bytes = tarfile.REGTYPE, link: str = "", data: bytes = b"x") -> tuple[Any, Any]:
    info = tarfile.TarInfo(name)
    info.type = kind
    info.linkname = link
    info.mode = 0o644
    if kind == tarfile.REGTYPE:
        info.size = len(data)
        return info, data
    return info, None


@pytest.mark.parametrize("filtered", [True, False])
def test_safe_extraction_with_and_without_the_data_filter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filtered: bool
) -> None:
    """Plain members extract; ``..``, linking-out and device members are refused on both paths.

    An absolute name is made relative by the ``data`` filter and refused by the manual check.
    """
    if not filtered:
        monkeypatch.setattr(build, "_HAS_DATA_FILTER", False)
    elif not build._HAS_DATA_FILTER:
        pytest.skip("this Python has no tarfile.data_filter")
    good = _tar([_member("pkg-1.0/setup.py"), _member("pkg-1.0/sub", tarfile.DIRTYPE)])
    dest = tmp_path / "ok"
    build.safe_extract(good, dest)
    assert (dest / "pkg-1.0" / "setup.py").read_bytes() == b"x"
    absolute = _tar([_member("/etc/evil")])
    if filtered:
        build.safe_extract(absolute, tmp_path / "abs")
        assert (tmp_path / "abs" / "etc" / "evil").is_file()
    else:
        with pytest.raises(build.ReproduceLockError):
            build.safe_extract(absolute, tmp_path / "abs")
    bad = [
        [_member("pkg/../../evil")],
        [_member("pkg/link", tarfile.SYMTYPE, "/etc/passwd")],
        [_member("pkg/link", tarfile.SYMTYPE, "../../outside")],
        [_member("pkg/hard", tarfile.LNKTYPE, "/etc/passwd")],
        [_member("pkg/dev", tarfile.CHRTYPE)],
    ]
    for index, members in enumerate(bad):
        with pytest.raises((tarfile.TarError, ValueError, build.ReproduceLockError)):
            build.safe_extract(_tar(members), tmp_path / f"bad{index}")


def test_fixture_env_is_reused() -> None:
    """The run tests' git environment is importable here (helper sanity)."""
    assert FIXTURE_ENV["GIT_CONFIG_NOSYSTEM"] == "1"
