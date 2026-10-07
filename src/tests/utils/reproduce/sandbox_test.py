"""Tests for the step sandbox: namespaces, read-only roots and the allowlisted env."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from senselab.utils.reproduce.sandbox import build_env, canary, keyed_templates, sandbox_argv, userns_available
from senselab.utils.reproduce.spec import Profile, StepSpec, render

COMMIT = "0123456789abcdef0123456789abcdef01234567"
SHA = "ab" * 32

needs_userns = pytest.mark.skipif(not userns_available(), reason="unprivileged user namespaces are unavailable here")


def _step(**extra: Any) -> StepSpec:  # noqa: ANN401
    data: dict[str, Any] = {
        "name": "s",
        "repo": "https://git.example/r.git",
        "commit": COMMIT,
        "kind": "python",
        "env": {
            "python": "3.12",
            "python_exact": "3.12.11",
            "torch": None,
            "body": {"lock": "b.lock", "lock_sha256": SHA, "source": "uv-export", "export_cmd": ["uv", "export"]},
            "exclude_newer": "2026-01-01",
        },
        "argv": ["{venv}/bin/python", "run.py", "--out", "{out}"],
        "pythonpath": ["{checkout}/lib", "{checkout}"],
        "resources": {"profile": "cpu"},
        "adapter": "a.py",
    }
    data.update(extra)
    return StepSpec.model_validate(data)


def _profile(root: Path, env: dict[str, str] | None = None) -> Profile:
    return Profile.model_validate(
        {
            "roots": {"data": str(root)},
            "results_root": str(root / "results"),
            "runs_root": str(root.parent / "runs"),
            "slurm_logs_root": str(root.parent / "logs"),
            "rw_under_roots": [str(root / "results")],
            "env": {"DATA_DIR": str(root)} if env is None else env,
            "partitions": {"cpu": {"partition": "p", "cpus": 1, "mem": "1G", "time": "0:10:00"}},
            "build": "cpu",
            "deny_patterns_file": str(root.parent / "deny.txt"),
            "canary_host": "pypi.org",
        }
    )


def _paths(base: Path) -> dict[str, Path]:
    return {name: base / name for name in ("out", "tmp", "checkout", "venv", "cache")}


def _run(argv: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, capture_output=True, text=True, timeout=60, env=env)


def test_render_uses_the_placeholder_grammar() -> None:
    """Placeholders expand; doubled braces stay literal; an unknown or missing value is refused."""
    assert render("{out}/a/{{x}}/{root:data}", {"out": "/o", "root:data": "/d"}) == "/o/a/{x}//d"
    with pytest.raises(ValueError):
        render("{nope}", {})
    with pytest.raises(KeyError):
        render("{out}", {})


def test_sandbox_argv_shape(tmp_path: Path) -> None:
    """Tools are absolute, ``sh`` is ``$0``, the script fails closed, rw binds precede the ro remount, caps drop."""
    root, out = tmp_path / "r", tmp_path / "r" / "out"
    argv = sandbox_argv(["true"], [root], [out], [tmp_path / "ck"])
    unshare = [
        "/usr/bin/unshare",
        "-rnm",
        "--pid",
        "--fork",
        "--kill-child",
        "--mount-proc",
        "--propagation",
        "private",
    ]
    assert argv[:10] == [*unshare, "/bin/sh", "-c"]
    assert argv[11:] == ["sh", "true"]
    script = argv[10]
    assert script.startswith("set -e")
    mount = "/usr/bin/mount"
    assert script.index(f"{mount} --bind {out} {out}") < script.index(f"{mount} -o remount,bind,ro {root}")
    assert script.index(f"{mount} -o remount,bind,ro {root}") < script.index(f"{mount} --bind {tmp_path / 'ck'}")
    assert "/usr/sbin/ip link set lo up" in script
    assert script.rstrip().endswith(
        'exec /usr/bin/setpriv --bounding-set=-all --inh-caps=-all --ambient-caps=-all --no-new-privs "$@"'
    )
    assert all(line.startswith(("set -e", "/usr/", "exec /usr/")) for line in script.splitlines())


def test_nested_paths_are_bound_shallow_first(tmp_path: Path) -> None:
    """An inner root listed first is still bound after its parent."""
    outer, inner = tmp_path / "r", tmp_path / "r" / "in"
    script = sandbox_argv(["true"], [inner, outer], [], [])[10]
    assert script.index(f"--bind {outer} ") < script.index(f"--bind {inner} ")


def test_sandbox_argv_refuses_a_ro_path_under_a_rw_dir(tmp_path: Path) -> None:
    """A rw bind over a ro mount would hide it, so that layout is refused."""
    with pytest.raises(ValueError):
        sandbox_argv(["true"], [tmp_path / "w" / "r"], [tmp_path / "w"], [])
    with pytest.raises(ValueError):
        sandbox_argv(["true"], [], [tmp_path / "ck" / "out"], [tmp_path / "ck"])
    with pytest.raises(ValueError):
        sandbox_argv(["true"], ["relative"], [], [])


@needs_userns
def test_argv_passes_through_unchanged() -> None:
    """The child receives argv exactly, first word included."""
    done = _run(sandbox_argv(["printf", "%s\\n", "a", "b c", "d"], [], [], []))
    assert done.returncode == 0, done.stderr
    assert done.stdout == "a\nb c\nd\n"


@needs_userns
def test_loopback_up_and_canary_fails() -> None:
    """Loopback connects inside the namespace; the canary to the profile host does not."""
    loop = (
        "import socket\n"
        "s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(1)\n"
        "c = socket.create_connection(s.getsockname(), timeout=5); print('ok')\n"
    )
    done = _run(sandbox_argv([sys.executable, "-c", loop], [], [], []))
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "ok"
    for host in ("pypi.org", "1.1.1.1", "example.org:443"):
        done = _run(sandbox_argv([sys.executable, "-c", canary(host)], [], [], []))
        assert done.returncode == 0, done.stderr
        assert json.loads(done.stdout) == {"canary_failed": True}


@needs_userns
def test_ro_roots_rw_out_ro_checkout_and_venv(tmp_path: Path) -> None:
    """A data root is read-only, ``{out}`` under it is writable, the checkout and venv are read-only."""
    root = tmp_path / "root"
    out, checkout, venv = root / "out", tmp_path / "ck", tmp_path / "venv"
    for d in (out, checkout, venv):
        d.mkdir(parents=True)
    probe = (
        "import json, sys\n"
        "res = {}\n"
        "for d in sys.argv[1:]:\n"
        "    try:\n"
        "        open(d + '/w.txt', 'w').write('x'); res[d] = True\n"
        "    except OSError:\n"
        "        res[d] = False\n"
        "print(json.dumps(res))\n"
    )
    targets = [str(root), str(out), str(checkout), str(venv)]
    done = _run(sandbox_argv([sys.executable, "-c", probe, *targets], [root], [out], [checkout, venv]))
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == {str(root): False, str(out): True, str(checkout): False, str(venv): False}
    assert (out / "w.txt").exists()
    assert not (root / "w.txt").exists()


@needs_userns
def test_nested_roots_mount(tmp_path: Path) -> None:
    """Nested roots given inner-first mount cleanly: both read-only, ``{out}`` under the inner one writable."""
    outer, inner = tmp_path / "r", tmp_path / "r" / "in"
    out = inner / "out"
    out.mkdir(parents=True)
    probe = 'for d in "$@"; do touch "$d/w" 2>/dev/null && echo 1 || echo 0; done'
    argv = sandbox_argv(["/bin/sh", "-c", probe, "sh", str(outer), str(inner), str(out)], [inner, outer], [out], [])
    done = _run(argv)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["0", "0", "1"]


ESCAPE = """
w() { touch "$1" 2>/dev/null && echo "$2=written" || echo "$2=blocked"; }
w "$2/o" out
/usr/bin/mount -o remount,bind,rw "$1" 2>/dev/null || true
w "$1/a" remount
/usr/bin/umount "$2" 2>/dev/null || true
/usr/bin/umount "$1" 2>/dev/null || true
w "$1/b" umount
/usr/bin/umount "$3" 2>/dev/null || true
w "$3/c" checkout
grep CapEff /proc/self/status
"""


@needs_userns
def test_step_cannot_undo_the_mounts(tmp_path: Path) -> None:
    """With capabilities dropped, remounting rw or unmounting the binds leaves the data read-only."""
    root, checkout = tmp_path / "root", tmp_path / "ck"
    out = root / "out"
    for d in (out, checkout):
        d.mkdir(parents=True)
    argv = sandbox_argv(["/bin/sh", "-c", ESCAPE, "sh", str(root), str(out), str(checkout)], [root], [out], [checkout])
    done = _run(argv)
    assert done.returncode == 0, done.stderr
    lines = done.stdout.split("\n")
    assert lines[:4] == ["out=written", "remount=blocked", "umount=blocked", "checkout=blocked"]
    assert lines[4].split() == ["CapEff:", "0000000000000000"]
    assert not any((root / n).exists() for n in ("a", "b")) and not (checkout / "c").exists()


@needs_userns
def test_pid_namespace_hides_and_protects_host_processes(tmp_path: Path) -> None:
    """The step is pid 1, sees only sandbox processes in /proc, and cannot signal a host process."""
    host = subprocess.Popen(["sleep", "60"])
    try:
        probe = (
            'echo "self=$$"; '
            "ls /proc | grep -E '^[0-9]+$' | sort -n | tr '\\n' ' '; echo; "
            f"kill {host.pid} 2>/dev/null && echo kill=sent || echo kill=failed"
        )
        done = _run(sandbox_argv(["/bin/sh", "-c", probe], [], [], []))
        assert done.returncode == 0, done.stderr
        lines = done.stdout.splitlines()
        assert lines[0] == "self=1"
        pids = [int(p) for p in lines[1].split()]
        assert str(host.pid) not in lines[1].split()
        assert max(pids) < 100
        assert lines[2] == "kill=failed"
        assert host.poll() is None
    finally:
        host.kill()
        host.wait()


NOTEBOOK = {
    "cells": [{"cell_type": "code", "execution_count": None, "id": "c1", "metadata": {}, "outputs": [],
               "source": "print(6 * 7)"}],
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}  # fmt: skip


@needs_userns
def test_notebook_executes_in_the_sandbox(tmp_path: Path) -> None:
    """A one-cell notebook runs through ``python -m nbconvert --execute`` (kernel over loopback) with the step env."""
    pytest.importorskip("nbconvert")
    pytest.importorskip("ipykernel")
    paths = {**_paths(tmp_path), "venv": Path(sys.prefix)}
    out = paths["out"]
    for d in (out, paths["tmp"] / "home", paths["checkout"]):
        d.mkdir(parents=True)
    nb = paths["checkout"] / "nb.ipynb"
    nb.write_text(json.dumps(NOTEBOOK))
    env, _, _ = build_env(_step(), _profile(tmp_path / "root"), paths, parent={})
    argv = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--output-dir", str(out), str(nb)]
    done = subprocess.run(
        sandbox_argv(argv, [], [out], [paths["checkout"]]), capture_output=True, text=True, timeout=180, env=env
    )
    assert done.returncode == 0, done.stderr[-2000:]
    executed = json.loads((out / "nb.ipynb").read_text())
    assert "".join(executed["cells"][0]["outputs"][0]["text"]) == "42\n"


@needs_userns
def test_failing_mount_aborts_before_the_step(tmp_path: Path) -> None:
    """A ro root that does not exist aborts the script; the step never runs."""
    marker = tmp_path / "marker"
    done = _run(sandbox_argv(["touch", str(marker)], [tmp_path / "missing"], [], []))
    assert done.returncode != 0
    assert not marker.exists()


@needs_userns
def test_parent_env_does_not_leak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Only the allowlist reaches the child: ``FOO_LEAK`` stays behind, ``SLURM_*`` is passed and recorded."""
    monkeypatch.setenv("FOO_LEAK", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    monkeypatch.setenv("SLURM_NTASKS", "4")
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("no_proxy", "localhost")
    paths = _paths(tmp_path)
    env, _, _ = build_env(_step(), _profile(tmp_path / "root"), paths)
    done = _run(sandbox_argv(["/usr/bin/env"], [], [], []), env=env)
    assert done.returncode == 0, done.stderr
    child = dict(line.split("=", 1) for line in done.stdout.splitlines())
    assert "FOO_LEAK" not in child
    assert child["SLURM_JOB_ID"] == "42"
    assert "SLURM_NTASKS" not in child and "SLURM_PROCID" not in child
    assert child["REPRO_OUT"] == str(paths["out"])
    assert child["REPRO_TMP"] == str(paths["tmp"])
    assert child["PYTHONPATH"] == f"{paths['checkout']}/lib:{paths['checkout']}"
    assert child["PATH"].startswith(f"{paths['venv']}/bin:")
    assert child["https_proxy"] == child["HTTP_PROXY"] == "http://127.0.0.1:9"
    assert child["no_proxy"] == child["NO_PROXY"] == ""
    assert child["TZ"] == "UTC"
    assert child["DATA_DIR"] == str(tmp_path / "root")


def test_keyed_sha_ignores_volatile_values_and_follows_the_profile(tmp_path: Path) -> None:
    """Job ids and expanded paths change only the recorded sha; the profile env map changes the keyed sha."""
    step, profile = _step(), _profile(tmp_path / "root")
    _, keyed1, rec1 = build_env(step, profile, _paths(tmp_path / "a"), parent={"SLURM_JOB_ID": "1"})
    _, keyed2, rec2 = build_env(step, profile, _paths(tmp_path / "b"), parent={"SLURM_JOB_ID": "2"})
    assert keyed1 == keyed2
    assert rec1 != rec2
    other = _profile(tmp_path / "root", env={"DATA_DIR": str(tmp_path / "elsewhere")})
    _, keyed3, _ = build_env(step, other, _paths(tmp_path / "a"), parent={"SLURM_JOB_ID": "1"})
    assert keyed3 != keyed1
    _, keyed4, _ = build_env(_step(pythonpath=["{checkout}"]), profile, _paths(tmp_path / "a"), parent={})
    assert keyed4 != keyed1


def test_r_libs(tmp_path: Path) -> None:
    """``r_lib`` keys ``R_LIBS`` as a template, with ``R_LIBS_SITE`` empty and no ``R_LIBS_USER``."""
    r_step = _step(
        kind="r",
        env={**_step().env.model_dump(mode="json"), "r_packages": [{"name": "x", "version": "1"}]}
        | {"r_snapshot": "2026-01-01", "r_lib": "{cache}/R/e"},
    )
    keyed = keyed_templates(r_step, _profile(tmp_path / "root"))
    assert keyed["R_LIBS"] == "{cache}/R/e"
    assert keyed["R_LIBS_SITE"] == ""
    assert "R_LIBS_USER" not in keyed
    env, _, _ = build_env(r_step, _profile(tmp_path / "root"), _paths(tmp_path), parent={})
    assert env["R_LIBS"] == f"{tmp_path}/cache/R/e"


NATIVE = {
    "ffmpeg": {"bin_dirs": ["/opt/ff/bin"], "lib_dirs": ["/opt/ff/curated-lib"]},
    "sox": {"bin_dirs": ["/opt/sox/bin"], "lib_dirs": ["/opt/sox/lib", "/opt/sox/lib64"]},
}


def test_native_entries_are_opt_in(tmp_path: Path) -> None:
    """A step without ``native`` gets no native dirs; selecting one gives only that entry's dirs, in order."""
    profile = Profile.model_validate({**_profile(tmp_path / "root").model_dump(mode="json"), "native": NATIVE})
    plain = keyed_templates(_step(), profile)
    assert "LD_LIBRARY_PATH" not in plain
    assert plain["PATH"] == "{venv}/bin:/usr/bin:/bin"
    sox = keyed_templates(_step(resources={"profile": "cpu", "native": ["sox"]}), profile)
    assert sox["LD_LIBRARY_PATH"] == "/opt/sox/lib:/opt/sox/lib64"
    assert sox["PATH"] == "{venv}/bin:/usr/bin:/bin:/opt/sox/bin"
    both = keyed_templates(_step(resources={"profile": "cpu", "native": ["ffmpeg", "sox"]}), profile)
    assert both["LD_LIBRARY_PATH"] == "/opt/ff/curated-lib:/opt/sox/lib:/opt/sox/lib64"
    _, k_plain, _ = build_env(_step(), profile, _paths(tmp_path), parent={})
    _, k_sox, _ = build_env(
        _step(resources={"profile": "cpu", "native": ["sox"]}), profile, _paths(tmp_path), parent={}
    )
    assert k_plain != k_sox


def test_unknown_native_entry_is_refused(tmp_path: Path) -> None:
    """A step may select only entries the profile defines."""
    with pytest.raises(ValueError):
        build_env(_step(resources={"profile": "cpu", "native": ["ffmpeg"]}), _profile(tmp_path / "root"),
                  _paths(tmp_path), parent={})  # fmt: skip


def test_pythonpath_entries_with_a_colon_are_refused(tmp_path: Path) -> None:
    """A ``:`` inside one entry would split it into two path entries."""
    with pytest.raises(ValueError):
        build_env(_step(pythonpath=["{checkout}/a:/b"]), _profile(tmp_path / "root"), _paths(tmp_path), parent={})


def test_profile_env_may_not_override_driver_variables(tmp_path: Path) -> None:
    """Every driver-owned name is refused in the profile map, whatever the step uses."""
    keys = (
        "PATH", "PYTHONPATH", "R_LIBS", "R_LIBS_SITE", "R_LIBS_USER", "JUPYTER_PATH", "HF_HOME", "HF_TOKEN_PATH",
        "TORCH_HOME", "XDG_CACHE_HOME", "CUDA_CACHE_PATH", "REPRO_OUT", "https_proxy", "ALL_PROXY", "no_proxy",
        "TMPDIR", "HOME", "IPYTHONDIR", "SLURM_JOB_ID", "SLURM_NTASKS", "LD_PRELOAD", "LD_LIBRARY_PATH",
        "PYTHONHOME", "PYTHONSTARTUP", "PYTHONNOUSERSITE", "TZ", "CUDA_VISIBLE_DEVICES",
    )  # fmt: skip
    for key in keys:
        with pytest.raises(ValueError):
            build_env(_step(), _profile(tmp_path / "root", env={key: "x"}), _paths(tmp_path), parent={})
