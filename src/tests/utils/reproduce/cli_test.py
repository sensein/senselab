"""Tests for ``python -m senselab.utils.reproduce``: the seven subcommands, lint exit codes, status, run chains.

Manifests, profiles, repositories and run trees are synthetic files in ``tmp_path``. The lock, prefetch
and build entry points are checked with their builders replaced by recorders, so nothing is installed.
"""

from __future__ import annotations

import builtins
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from senselab.utils.reproduce import build, cache, cli, env, lockgen, run, sandbox, slurm, source
from senselab.utils.reproduce.spec import load_manifest
from tests.utils.reproduce.run_test import ADAPTER, World
from tests.utils.reproduce.slurm_test import FakeSbatch, _step

SUBCOMMANDS = ["lock", "prefetch", "build-one", "run", "submit", "lint", "status"]
ZERO = "0" * 64
DATE = "2026-03-04"
needs_userns = pytest.mark.skipif(not sandbox.userns_available(), reason="unprivileged user namespaces are unavailable")

FIXTURE_ENV = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "fixture",
    "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
    "GIT_COMMITTER_NAME": "fixture",
    "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    "GIT_AUTHOR_DATE": "@1000000000 +0000",
    "GIT_COMMITTER_DATE": "@1000000000 +0000",
}


def _git(cwd: Path, *args: str) -> str:
    argv = ["git", "-c", "init.defaultBranch=main", "-C", str(cwd), *args]
    env_vars = {**FIXTURE_ENV, "HOME": str(cwd)}
    return subprocess.run(argv, env=env_vars, check=True, capture_output=True, text=True).stdout


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Site:
    """A manifest repository with an adapter, a patch, a deny-pattern file and a profile."""

    def __init__(self, tmp: Path) -> None:
        """Lay the site out under ``tmp``."""
        self.tmp = tmp
        self.base = tmp / "manifests"
        (self.base / "patches").mkdir(parents=True)
        (self.base / "adapter.py").write_text(ADAPTER)
        self.patch("print(2)")
        self.deny = tmp / "deny.txt"
        self.deny.write_text("# test patterns\nFORBIDDEN-[0-9]+\n")
        (tmp / "data").mkdir()
        self.cache = tmp / "cache"
        self.profile_path = tmp / "profile.yaml"
        self.write_profile()

    def write_profile(self, **change: Any) -> None:  # noqa: ANN401
        """Write the profile file."""
        data: dict[str, Any] = {
            "roots": {"data": str(self.tmp / "data")},
            "results_root": str(self.tmp / "results"),
            "runs_root": str(self.tmp / "runs"),
            "slurm_logs_root": str(self.tmp / "slurm"),
            "partitions": {
                "cpu": {"partition": "main", "cpus": 1, "mem": "1G", "time": "00:10:00"},
                "build": {"partition": "quick", "cpus": 1, "mem": "1G", "time": "00:10:00"},
            },
            "build": "build",
            "deny_patterns_file": str(self.deny),
            "canary_host": "192.0.2.1:9",
        }
        data.update(change)
        self.profile_path.write_text(yaml.safe_dump(data))

    def patch(self, line: str) -> dict[str, str]:
        """Write ``patches/p.patch`` adding ``line`` and return its manifest reference."""
        path = self.base / "patches" / "p.patch"
        path.write_text(f"--- a/step.py\n+++ b/step.py\n@@ -1 +1 @@\n-print(1)\n+{line}\n")
        return {"file": "patches/p.patch", "sha256": _sha(path)}

    def manifest(self, name: str, *steps: dict[str, Any], file: str = "m.yaml", header: str = "") -> Path:
        """Write a manifest file under the manifest repository."""
        path = self.base / file
        path.write_text(header + yaml.safe_dump({"name": name, "steps": list(steps)}, sort_keys=False))
        return path

    def main(self, *argv: str) -> int:
        """Run the CLI with ``--profile``, ``--base-dir`` and ``--cache`` filled in where they apply."""
        return cli.main(list(argv))

    def common(self) -> list[str]:
        """The site options every manifest subcommand takes."""
        return ["--profile", str(self.profile_path), "--base-dir", str(self.base), "--cache", str(self.cache)]


@pytest.fixture
def site(tmp_path: Path) -> Site:
    """A fresh synthetic manifest repository and profile."""
    return Site(tmp_path)


@pytest.fixture
def world(tmp_path: Path) -> World:
    """The run tests' synthetic site: one repository, a data root, a profile and a cache."""
    return World(tmp_path)


# --- the command line ---------------------------------------------------------------------------------


def test_help_lists_the_seven_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    """``--help`` names exactly lock, prefetch, build-one, run, submit, lint and status."""
    with pytest.raises(SystemExit) as info:
        cli.main(["--help"])
    assert info.value.code == 0
    text = capsys.readouterr().out
    for name in SUBCOMMANDS:
        assert name in text
    assert cli.subcommands() == SUBCOMMANDS


def test_python_dash_m_runs_the_cli() -> None:
    """``python -m senselab.utils.reproduce --help`` works from a minimal environment."""
    src = Path(__file__).resolve().parents[3]
    env_vars = {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONPATH": str(src)}
    if "LD_LIBRARY_PATH" in os.environ:
        env_vars["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    done = subprocess.run(
        [sys.executable, "-m", "senselab.utils.reproduce", "--help"],
        env=env_vars,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert all(name in done.stdout for name in SUBCOMMANDS)


# --- lint ------------------------------------------------------------------------------------------------


def test_lint_exit_codes(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    """0 when clean, 1 for findings (each printed as a code), 2 for usage errors and unreadable files."""
    clean = site.manifest("demo", _step("prep", patches=[site.patch("print(2)")]))
    assert site.main("lint", str(clean), *site.common()) == cli.EXIT_OK
    assert capsys.readouterr().out == ""

    rooted = site.manifest("demo", _step("prep", inputs=[{"root": "nowhere", "patterns": ["a"]}]), file="r.yaml")
    assert site.main("lint", str(rooted), *site.common()) == cli.EXIT_FAILED
    assert "root-unknown" in capsys.readouterr().out

    secret = "FORBIDDEN-12345"
    denied = site.manifest("demo", _step("prep", patches=[site.patch(secret)]), file="d.yaml")
    assert site.main("lint", str(denied), *site.common()) == cli.EXIT_FAILED
    out = capsys.readouterr().out
    assert "patch-lint deny-pattern line 5 pattern 0" in out and secret not in out

    stale = site.manifest("demo", _step("prep", patches=[{"file": "patches/p.patch", "sha256": ZERO}]), file="s.yaml")
    assert site.main("lint", str(stale), *site.common()) == cli.EXIT_FAILED
    assert "patch-sha" in capsys.readouterr().out

    invalid = site.manifest("demo", _step("prep", commit="abc"), file="i.yaml")
    assert site.main("lint", str(invalid), *site.common()) == cli.EXIT_FAILED
    assert "manifest-invalid" in capsys.readouterr().out

    cycle = site.manifest("demo", _step("a", needs=["b"]), _step("b", needs=["a"]), file="c.yaml")
    assert site.main("lint", str(cycle), *site.common()) == cli.EXIT_FAILED
    assert "dag-invalid" in capsys.readouterr().out

    no_adapter = site.manifest("demo", _step("prep", adapter="missing.py"), file="a.yaml")
    assert site.main("lint", str(no_adapter), *site.common()) == cli.EXIT_FAILED
    assert "adapter-missing" in capsys.readouterr().out

    assert site.main("lint", str(site.base / "absent.yaml"), *site.common()) == cli.EXIT_USAGE
    with pytest.raises(SystemExit) as info:
        site.main("lint", "--no-such-option")
    assert info.value.code == cli.EXIT_USAGE


# --- prefetch -------------------------------------------------------------------------------------------


NOTEBOOK = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [{"output_type": "stream", "name": "stdout", "text": ["an output\n"]}],
            "source": ["print('an output')\n"],
        }
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}


def test_prefetch_sources_only_on_a_draft_manifest(
    site: Site, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A draft with no ``env`` gets a cleared worktree; nothing is locked, built or installed."""
    repo = site.tmp / "upstream" / "analysis"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    (repo / "nb.ipynb").write_text(json.dumps(NOTEBOOK))
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "x")
    commit = _git(repo, "rev-parse", "HEAD").strip()

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("sources-only installed something")

    monkeypatch.setattr(env, "prefetch_env", refuse)
    monkeypatch.setattr(env, "prefetch_r", refuse)
    monkeypatch.setattr(build, "ensure_wheel", refuse)
    draft = {"name": "draft", "steps": [{"name": "row", "repo": str(repo), "commit": commit, "argv": ["python"]}]}
    path = site.base / "draft.yaml"
    path.write_text(yaml.safe_dump(draft))
    assert site.main("prefetch", "--sources-only", str(path), "--cache", str(site.cache)) == cli.EXIT_OK
    worktree = source.worktree_dir(site.cache, str(repo), commit, source.EMPTY_PATCH_SET_SHA256)
    cleared = json.loads((worktree / "nb.ipynb").read_text())["cells"][0]
    assert cleared["outputs"] == [] and cleared["execution_count"] is None
    assert not (site.cache / env.VENVS).exists()
    assert "row" in capsys.readouterr().out


def _venv_result(path: Path) -> env.VenvResult:
    return env.VenvResult(path, "k" * 64, None, False, {}, True)


def _sdist(site: Site) -> dict[str, Any]:
    (site.base / "build.lock").write_text("x\n")
    return {
        "name": "pkg",
        "url": "https://files.example/pkg-1.0.tar.gz",
        "sha256": "1" * 64,
        "build_env": {"lock": "build.lock", "lock_sha256": _sha(site.base / "build.lock")},
    }


def test_prefetch_passes_the_profile_and_deny_patterns_and_records_wheels(
    site: Site, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checkouts get the profile's deny patterns, ``prefetch_env`` the profile; a new wheels.lock sha is recorded."""
    seen: dict[str, Any] = {}

    def checkout_step(step: Any, *, cache: Path, base_dir: Path, deny_patterns: Any) -> Any:  # noqa: ANN401
        seen["deny"] = list(deny_patterns)
        return type("Co", (), {"worktree": site.tmp})()

    def prefetch_env(env_spec: Any, base_dir: Path, cache_dir: Path, **kwargs: Any) -> env.PrefetchResult:  # noqa: ANN401
        seen["calls"] = seen.get("calls", 0) + 1
        seen.update(kwargs)
        return env.PrefetchResult(_venv_result(cache_dir / "v"), {}, "e" * 64, True, ("pkg",))

    monkeypatch.setattr(source, "checkout_step", checkout_step)
    monkeypatch.setattr(env, "prefetch_env", prefetch_env)
    sdist_env = dict(_step("x")["env"], sdist_allow=[_sdist(site)])
    path = site.manifest("demo", _step("prep", env=sdist_env), _step("again", env=sdist_env))
    assert site.main("prefetch", str(path), *site.common()) == cli.EXIT_OK
    assert seen["deny"] == ["FORBIDDEN-[0-9]+"]
    assert seen["profile"] is not None and seen["profile"].deny_patterns_file == site.deny
    assert seen["calls"] == 1 and seen["wheels_lock"] == site.base / "wheels.lock"
    assert seen["wheelhouse"] == site.cache / cli.WHEELHOUSE
    for step in load_manifest(path).steps:
        assert step.env.wheels is not None
        assert (step.env.wheels.lock, step.env.wheels.lock_sha256) == ("wheels.lock", "e" * 64)


def test_a_changed_wheels_sha_is_written_in_place_keeping_comments(site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    """A recorded sha that changed is replaced in the file text, so comments and layout survive."""
    monkeypatch.setattr(source, "checkout_step", lambda step, **kw: type("Co", (), {"worktree": site.tmp})())
    monkeypatch.setattr(
        env,
        "prefetch_env",
        lambda e, b, c, **kw: env.PrefetchResult(_venv_result(c / "v"), {}, "f" * 64, True, ()),
    )
    sdist_env = dict(
        _step("x")["env"], sdist_allow=[_sdist(site)], wheels={"lock": "wheels.lock", "lock_sha256": "e" * 64}
    )
    path = site.manifest("demo", _step("prep", env=sdist_env), header="# keep this comment\n")
    assert site.main("prefetch", str(path), *site.common()) == cli.EXIT_OK
    text = path.read_text()
    assert text.startswith("# keep this comment\n") and "f" * 64 in text and "e" * 64 not in text


def test_prefetch_refuses_hf_weights_and_lint_findings(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    """Steps that load Hugging Face weights are not prefetched yet; lint findings stop a prefetch."""
    weights = [{"repo": "org/model", "revision": "a" * 40, "files": [{"path": "w.bin", "sha256": "2" * 64}]}]
    path = site.manifest("demo", _step("prep", hf_weights=weights))
    assert site.main("prefetch", str(path), *site.common()) == cli.EXIT_FAILED
    assert "hf_weights" in capsys.readouterr().err
    bad = site.manifest("demo", _step("prep", resources={"profile": "tpu"}), file="b.yaml")
    assert site.main("prefetch", str(bad), *site.common()) == cli.EXIT_FAILED


def test_build_one_records_the_new_wheels_sha(site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    """``build-one`` builds through ``ensure_wheel`` and writes the new ``wheels.lock`` sha into the manifest."""
    seen: dict[str, Any] = {}

    def ensure_wheel(sdist: Any, base_dir: Path, wheelhouse: Path, wheels_lock: Path, *a: Any, **kw: Any) -> Any:  # noqa: ANN401
        seen.update(kw, wheels_lock=wheels_lock, wheelhouse=wheelhouse, name=sdist.name)
        wheels_lock.write_text("lock\n")
        return build.WheelResult(wheelhouse / "w.whl", "3" * 64, _sha(wheels_lock), True)

    monkeypatch.setattr(build, "ensure_wheel", ensure_wheel)
    sdist_env = dict(_step("x")["env"], sdist_allow=[_sdist(site)])
    path = site.manifest("demo", _step("prep", env=sdist_env))
    argv = ["build-one", str(path), *site.common(), "--step", "prep", "--sdist", "pkg"]
    assert site.main(*argv) == cli.EXIT_OK
    assert seen["name"] == "pkg" and seen["wheels_lock"] == site.base / "wheels.lock"
    assert seen["expected_sha256"] is None and seen["rebuild"] is False
    (step,) = load_manifest(path).steps
    assert step.env.wheels is not None and step.env.wheels.lock_sha256 == _sha(site.base / "wheels.lock")


# --- lock -----------------------------------------------------------------------------------------------


def test_lock_runs_the_builders_in_order_and_fills_in_the_shas(site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    """resolve, torch, body, then overlays; each written lock's sha256 goes into the manifest."""
    order: list[str] = []

    def recorder(name: str, rel: Any) -> Any:  # noqa: ANN401
        def builder(env_spec: Any, base_dir: Path, *args: Any, **kwargs: Any) -> list[str]:  # noqa: ANN401
            label = f"{name}:{args[0]}" if name == "overlay_lock" else name
            order.append(label)
            target = base_dir / rel(env_spec if name != "overlay_lock" else base_dir, args)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f"# {label}\n")
            return ["uv", label]

        return builder

    monkeypatch.setattr(lockgen, "first_resolve", recorder("first_resolve", lambda e, a: e.first_resolve.file))
    monkeypatch.setattr(lockgen, "torch_lock_from_index", recorder("torch_lock_from_index", lambda e, a: e.torch.lock))
    monkeypatch.setattr(
        lockgen, "body_lock_from_requirements", recorder("body_lock_from_requirements", lambda e, a: e.body.lock)
    )

    def overlay(kind: str, env_spec: Any, base_dir: Path, **kwargs: Any) -> list[str]:  # noqa: ANN401
        order.append(f"overlay_lock:{kind}")
        assert kwargs["requirements"] == "envs/nb.in"
        ref = env_spec.notebook_tools
        (base_dir / ref.lock).write_text(f"# {kind}\n")
        return ["uv", kind]

    monkeypatch.setattr(lockgen, "overlay_lock", overlay)
    (site.base / "envs").mkdir()
    (site.base / "envs" / "requirements.in").write_text("numpy\n")
    env_spec = {
        "python": "3.12",
        "python_exact": "3.12.11",
        "torch": {"lock": "envs/torch.lock", "lock_sha256": ZERO, "index_url": "https://download.example/whl"},
        "body": {
            "lock": "envs/body.lock",
            "lock_sha256": ZERO,
            "source": "requirements.in",
            "requirements_in": "envs/requirements.in",
            "requirements_in_sha256": _sha(site.base / "envs" / "requirements.in"),
        },
        "first_resolve": {"file": "envs/resolve.txt", "sha256": ZERO},
        "notebook_tools": {"lock": "envs/nb.lock", "lock_sha256": ZERO},
        "exclude_newer": "2026-01-01",
    }
    path = site.manifest("demo", _step("prep", kind="notebook", env=env_spec))
    assert site.main("lock", str(path), *site.common()) == cli.EXIT_OK
    assert order == [
        "first_resolve",
        "torch_lock_from_index",
        "body_lock_from_requirements",
        "overlay_lock:notebook_tools",
    ]
    (step,) = load_manifest(path).steps
    assert step.env.torch is not None and step.env.first_resolve is not None and step.env.notebook_tools is not None
    assert step.env.torch.lock_sha256 == _sha(site.base / "envs" / "torch.lock")
    assert step.env.body.lock_sha256 == _sha(site.base / "envs" / "body.lock")
    assert step.env.first_resolve.sha256 == _sha(site.base / "envs" / "resolve.txt")
    assert step.env.notebook_tools.lock_sha256 == _sha(site.base / "envs" / "nb.lock")


def test_lock_from_a_repository_uv_lock(site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    """A uv-export env locks torch from the checkout's ``uv.lock``, then exports the body and records the command."""
    checkout = site.tmp / "checkout"
    checkout.mkdir()
    (checkout / "uv.lock").write_text('[[package]]\nname = "torch"\nversion = "2.8.0"\n')
    monkeypatch.setattr(source, "checkout_step", lambda step, **kw: type("Co", (), {"worktree": checkout})())
    order: list[str] = []

    def torch_lock(env_spec: Any, base_dir: Path, uv_lock: Path, roots: Any) -> list[str]:  # noqa: ANN401
        order.append(f"torch:{','.join(roots)}:{uv_lock == checkout / 'uv.lock'}")
        (base_dir / env_spec.torch.lock).write_text("t\n")
        return []

    def body_lock(env_spec: Any, base_dir: Path, project_dir: Path, **kwargs: Any) -> list[str]:  # noqa: ANN401
        order.append(f"body:{list(kwargs['extras'])}:{list(kwargs['groups'])}")
        (base_dir / env_spec.body.lock).write_text("b\n")
        return ["/abs/uv", "export", "--frozen", "--extra", "gpu"]

    monkeypatch.setattr(lockgen, "torch_lock_from_uvlock", torch_lock)
    monkeypatch.setattr(lockgen, "body_lock_from_uvlock_export", body_lock)
    env_spec = {
        "python": "3.12",
        "python_exact": "3.12.11",
        "torch": {"lock": "torch.lock", "lock_sha256": ZERO, "index_url": "https://download.example/whl"},
        "body": {
            "lock": "body.lock",
            "lock_sha256": ZERO,
            "source": "uv-export",
            "export_cmd": ["uv", "export", "--extra", "gpu", "--group", "eval"],
        },
        "exclude_newer": "2026-01-01",
    }
    path = site.manifest("demo", _step("prep", env=env_spec))
    assert site.main("lock", str(path), *site.common()) == cli.EXIT_OK
    assert order == ["torch:torch:True", "body:['gpu']:['eval']"]
    (step,) = load_manifest(path).steps
    assert step.env.body.export_cmd == ["uv", "export", "--frozen", "--extra", "gpu"]
    assert step.env.body.lock_sha256 == _sha(site.base / "body.lock")


# --- status -----------------------------------------------------------------------------------------------


def _run_tree(site: Site, rows: list[dict[str, Any]], summaries: dict[str, dict[str, Any]]) -> Path:
    day = site.tmp / "runs" / DATE / "demo"
    day.mkdir(parents=True)
    for step, summary in summaries.items():
        key = next(r["key"] for r in rows if r["step"] == step)
        out = site.tmp / "results" / key
        (out / "logs").mkdir(parents=True)
        (out / "logs" / "step.stdout.log").write_text("MARKER-LOG-LINE\n")
        (out / cache.SUMMARY).write_text(json.dumps(summary))
        (day / step).symlink_to(os.path.relpath(out, day))
    (day / run.RUN_JSON).write_text(json.dumps({"steps": rows}))
    return day


def test_status_prints_codes_from_run_and_summary_only(
    site: Site, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One line per step: name, key8, status, exit, codes; never a log; exit 1 when something failed."""
    key = "a1b2c3d4" + "0" * 56
    rows: list[dict[str, Any]] = [
        {"step": "prep", "key": key, "status": "executed", "exit": 0, "codes": []},
        {"step": "score", "key": None, "status": "never-ran", "exit": None, "codes": []},
        {"step": "odd", "key": None, "status": "failed", "exit": 1, "codes": ["E_SRC_MISSING", "rm -rf /"]},
    ]
    day = _run_tree(site, rows, {"prep": {"exit": 0, "codes": ["W_HOST_VALUE_DROPPED"]}})
    opened: list[str] = []
    real_open, real_os_open = builtins.open, os.open

    def spy_open(file: Any, *a: Any, **kw: Any) -> Any:  # noqa: ANN401
        opened.append(str(file))
        return real_open(file, *a, **kw)

    def spy_os_open(path: Any, *a: Any, **kw: Any) -> int:  # noqa: ANN401
        opened.append(str(path))
        return real_os_open(path, *a, **kw)

    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(os, "open", spy_os_open)
    assert site.main("status", str(day)) == cli.EXIT_FAILED
    monkeypatch.undo()
    out = capsys.readouterr().out
    assert out.splitlines() == [
        "prep\ta1b2c3d4\texecuted\t0\tW_HOST_VALUE_DROPPED",
        "score\t-\tnever-ran\t-\t-",
        "odd\t-\tfailed\t1\tE_SRC_MISSING,?",
    ]
    assert "MARKER-LOG-LINE" not in out
    assert not any("logs" in Path(p).parts for p in opened)


def test_status_exit_codes(site: Site) -> None:
    """0 when every step succeeded, 3 while one is pending, 2 for a missing run record."""
    key = "f" * 64
    rows: list[dict[str, Any]] = [{"step": "prep", "key": key, "status": "skipped", "exit": 0, "codes": []}]
    day = _run_tree(site, rows, {"prep": {"exit": 0, "codes": []}})
    assert site.main("status", str(day)) == cli.EXIT_OK
    rows.append({"step": "score", "key": None, "status": "pending", "exit": None, "codes": [], "job": "7"})
    (day / run.RUN_JSON).write_text(json.dumps({"steps": rows}))
    assert site.main("status", str(day)) == cli.EXIT_PENDING
    assert site.main("status", str(site.tmp / "nowhere")) == cli.EXIT_USAGE


# --- run: one step per job, upstream resolved ------------------------------------------------------------


def _rows(world: World) -> dict[str, tuple[str, int | None]]:
    path = world.tmp / "runs" / world.config().date.isoformat() / "demo" / run.RUN_JSON
    return {r["step"]: (r["status"], r["exit"]) for r in json.loads(path.read_text())["steps"]}


@needs_userns
def test_run_only_resolves_upstream_without_running_it(world: World) -> None:
    """A job for ``score`` reuses ``prep``'s done result; with ``prep`` not done, ``score`` never runs."""
    manifest = world.dag()
    world.prefetch(manifest)
    (first,) = run.run_steps([manifest], world.config(), only={"score"})
    assert first.step == "score" and first.status == "never-ran"
    assert not (world.tmp / "results").exists() or not any(cache.is_done(p) for p in (world.tmp / "results").iterdir())
    (prep,) = run.run_steps([manifest], world.config(), only={"prep"})
    assert prep.status == "executed" and prep.ok
    assert _rows(world) == {"prep": ("executed", 0), "score": ("never-ran", None)}
    assert prep.out is not None
    stamp = (prep.out / cache.SUMMARY).stat().st_mtime_ns
    (score,) = run.run_steps([manifest], world.config(), only={"score"})
    assert score.status == "executed" and score.ok, score.codes
    assert (prep.out / cache.SUMMARY).stat().st_mtime_ns == stamp
    assert _rows(world) == {"prep": ("executed", 0), "score": ("executed", 0)}


@needs_userns
def test_a_failed_job_marks_its_pending_dependents_never_ran(world: World) -> None:
    """After a failed step, its dependents' pending rows read ``never-ran``; ``--finalize`` settles the rest."""
    manifest = world.dag(prep_expect=[{"metric": "f_stat", "value": 99.0, "tolerance": 0.5, "kind": "abs"}])
    world.prefetch(manifest)
    config = world.config()
    pending: list[dict[str, Any]] = [
        {"step": s, "key": None, "status": "pending", "exit": None, "codes": [], "job": "1"} for s in ("prep", "score")
    ]
    run.update_run_record(config.profile.runs_root, config.date, "demo", pending, order=["prep", "score"])
    (prep,) = run.run_steps([manifest], config, only={"prep"})
    assert prep.status == "executed" and not prep.ok
    rows = _rows(world)
    assert rows["prep"][0] == "executed" and rows["prep"][1] != 0 and rows["score"] == ("never-ran", None)
    record = world.tmp / "runs" / config.date.isoformat() / "demo" / run.RUN_JSON
    score_row = next(r for r in json.loads(record.read_text())["steps"] if r["step"] == "score")
    assert "E_EXPECT" in score_row["codes"]

    run.update_run_record(config.profile.runs_root, config.date, "demo", pending, order=["prep", "score"])
    run.finalize_run_records([manifest], config)
    assert _rows(world) == {"prep": ("never-ran", None), "score": ("never-ran", None)}


@needs_userns
def test_the_run_command_prints_codes_and_exits_non_zero_on_failure(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    """``run --step`` prints one status line per step and exits 1 unless every selected step succeeded."""
    manifest = world.dag()
    world.prefetch(manifest)
    base = world.base
    path = base / "m.json"
    path.write_text(json.dumps(manifest.model_dump(mode="json")))
    profile = world.tmp / "profile.json"
    profile.write_text(json.dumps(world.profile().model_dump(mode="json")))
    date = world.config().date.isoformat()
    common = ["--profile", str(profile), "--base-dir", str(base), "--cache", str(world.cache), "--date", date]
    assert cli.main(["run", str(path), *common, "--step", "score"]) == cli.EXIT_FAILED
    assert capsys.readouterr().out.splitlines() == ["score\t-\tnever-ran\t-\tE_UPSTREAM_NOT_DONE"]
    assert cli.main(["run", str(path), *common, "--step", "prep"]) == cli.EXIT_OK
    line = capsys.readouterr().out.strip().split("\t")
    assert line[0] == "prep" and line[2] == "executed" and line[3] == "0" and len(line[1]) == 8


def test_submit_refuses_lint_findings_and_submits_a_clean_chain(
    site: Site, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``submit`` lints first (no sbatch call on findings), then prints one job id per step."""
    fake = FakeSbatch()
    monkeypatch.setattr(slurm, "default_runner", fake)
    monkeypatch.setenv("UV_CACHE_DIR", "/c/uv")
    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", "/c/py")
    bad = site.manifest("demo", _step("prep", resources={"profile": "tpu"}), file="b.yaml")
    assert site.main("submit", str(bad), *site.common(), "--date", DATE) == cli.EXIT_FAILED
    assert fake.calls == []
    capsys.readouterr()
    good = site.manifest("demo", _step("prep"), _step("score", needs=["prep"]))
    assert site.main("submit", str(good), *site.common(), "--date", DATE) == cli.EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("submission\t") and lines[1:] == ["prep\t101", "score\t102", "finalize\t103"]
    assert fake.commands("scontrol") == [["scontrol", "release", "101,102,103"]]
    script = fake.submissions()[1]
    assert f"--pin {_sha(good)}" in script.splitlines()[-1]
    assert "export SENSELAB_REPRO_CACHE=" + str(site.cache) in script and "export UV_LINK_MODE=hardlink" in script


# --- review fixes: builds, pins, error mapping, names, finalize scope ------------------------------------


def _cuda_manifest(site: Site) -> Path:
    sdist = dict(_sdist(site), build_modules=["cuda/12.9.1"])
    return site.manifest("demo", _step("prep", env=dict(_step("x")["env"], sdist_allow=[sdist])))


def _no_install(monkeypatch: pytest.MonkeyPatch, site: Site) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(source, "checkout_step", lambda step, **kw: type("Co", (), {"worktree": site.tmp})())

    def prefetch_env(e: Any, b: Path, c: Path, **kw: Any) -> env.PrefetchResult:  # noqa: ANN401
        calls.append("prefetch_env")
        return env.PrefetchResult(_venv_result(c / "v"), {}, None, False, ())

    monkeypatch.setattr(env, "prefetch_env", prefetch_env)
    monkeypatch.setenv("UV_CACHE_DIR", "/c/uv")
    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", "/c/py")
    return calls


def test_batch_builds_wait_without_timeout_and_pass_the_base_dir(site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    """A CUDA sdist is built by a waited ``build-one`` job given prefetch's ``--base-dir``; then the env installs."""
    calls = _no_install(monkeypatch, site)
    waited: list[tuple[list[str], str]] = []

    def wait(argv: list[str], script: str) -> subprocess.CompletedProcess[str]:
        waited.append((argv, script))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(slurm, "wait_runner", wait)
    path = _cuda_manifest(site)
    assert site.main("prefetch", str(path), *site.common()) == cli.EXIT_OK
    ((argv, script),) = waited
    assert argv[:3] == ["sbatch", "--wait", "--parsable"] and argv[3].startswith("--job-name=reproduce-build-pkg-")
    assert f"--base-dir {site.base}" in script.splitlines()[-1] and "-P -m" in script.splitlines()[-1]
    assert calls == ["prefetch_env"]


def test_a_failed_build_job_fails_the_prefetch(site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    """A build job that exits non-zero stops the prefetch before any env is installed."""
    calls = _no_install(monkeypatch, site)
    monkeypatch.setattr(slurm, "wait_runner", lambda argv, script: subprocess.CompletedProcess(argv, 1))
    assert site.main("prefetch", str(_cuda_manifest(site)), *site.common()) == cli.EXIT_FAILED
    assert calls == []


def test_an_interrupted_build_wait_cancels_the_job(site: Site, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-C while waiting cancels the build job by name, so no orphan rewrites the manifest later."""
    calls = _no_install(monkeypatch, site)
    names: list[str] = []
    cancels: list[list[str]] = []

    def interrupted(argv: list[str], script: str) -> subprocess.CompletedProcess[str]:
        names.append(argv[3].split("=", 1)[1])
        raise KeyboardInterrupt

    monkeypatch.setattr(slurm, "wait_runner", interrupted)
    monkeypatch.setattr(slurm, "default_runner", lambda argv, script, timeout=None: cancels.append(argv))
    path = _cuda_manifest(site)
    before = path.read_text()
    assert site.main("prefetch", str(path), *site.common()) == cli.EXIT_FAILED
    assert cancels == [["scancel", f"--name={names[0]}", cancels[0][2]]] and cancels[0][2].startswith("--user=")
    assert calls == [] and path.read_text() == before


def test_a_job_refuses_a_manifest_changed_since_submission(site: Site, capsys: pytest.CaptureFixture[str]) -> None:
    """``run --pin`` with another sha256 records ``E_MANIFEST_CHANGED`` and exits 1 without running anything."""
    path = site.manifest("demo", _step("prep"))
    argv = ["run", str(path), *site.common(), "--date", DATE, "--submission", "s1", "--pin", "0" * 64, "--step", "prep"]
    assert site.main(*argv) == cli.EXIT_FAILED
    assert capsys.readouterr().out.splitlines() == ["prep\t-\tfailed\t1\tE_MANIFEST_CHANGED"]
    rows = json.loads((site.tmp / "runs" / DATE / "demo" / run.RUN_JSON).read_text())["steps"]
    assert [(r["step"], r["status"], r["codes"]) for r in rows] == [("prep", "failed", ["E_MANIFEST_CHANGED"])]


def test_driver_errors_are_codes_without_a_traceback(
    site: Site, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ``CacheError`` (or any other driver error) is one code on stderr and exit 1."""
    path = site.manifest("demo", _step("prep"))

    def broken(*args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        raise cache.CacheError("E_RESULT_UNREADABLE")

    monkeypatch.setattr(run, "run_steps", broken)
    assert site.main("run", str(path), *site.common(), "--date", DATE) == cli.EXIT_FAILED
    err = capsys.readouterr().err
    assert err == "error: E_RESULT_UNREADABLE\n"
    monkeypatch.setattr(run, "run_steps", lambda *a, **k: (_ for _ in ()).throw(KeyError("/secret/path")))
    assert site.main("run", str(path), *site.common(), "--date", DATE) == cli.EXIT_FAILED
    err = capsys.readouterr().err
    assert err == "error: E_DRIVER (KeyError)\n" and "secret" not in err


@pytest.mark.parametrize("name", ["..", ".", ".hidden"])
def test_dot_names_fail_lint(site: Site, name: str, capsys: pytest.CaptureFixture[str]) -> None:
    """A manifest or step named ``.``, ``..`` or with a leading dot is ``manifest-invalid``."""
    path = site.manifest(name, _step("prep"), file="n.yaml")
    assert site.main("lint", str(path), *site.common()) == cli.EXIT_FAILED
    assert "manifest-invalid" in capsys.readouterr().out
    step = site.manifest("demo", _step(name), file="s.yaml")
    assert site.main("lint", str(step), *site.common()) == cli.EXIT_FAILED


def test_finalize_touches_only_its_own_submission(site: Site) -> None:
    """``--finalize --submission`` settles only that submission's pending rows; a requeued job's row wins after."""
    path = site.manifest("demo", _step("a"), _step("b"))
    runs = site.tmp / "runs"
    rows: list[dict[str, Any]] = [
        {"step": "a", "key": None, "status": "pending", "exit": None, "codes": [], "submission": "s1"},
        {"step": "b", "key": None, "status": "pending", "exit": None, "codes": [], "submission": "s2"},
    ]
    run.update_run_record(runs, datetime.date.fromisoformat(DATE), "demo", rows, order=["a", "b"])
    argv = ["run", str(path), *site.common(), "--date", DATE, "--finalize", "--submission", "s1"]
    assert site.main(*argv) == cli.EXIT_OK
    record = runs / DATE / "demo" / run.RUN_JSON
    got = {r["step"]: (r["status"], r["submission"]) for r in json.loads(record.read_text())["steps"]}
    assert got == {"a": ("never-ran", "s1"), "b": ("pending", "s2")}
    done = {"step": "a", "key": "f" * 64, "status": "executed", "exit": 0, "codes": []}
    run.update_run_record(runs, datetime.date.fromisoformat(DATE), "demo", [done], order=["a", "b"])
    got = {r["step"]: (r["status"], r["submission"]) for r in json.loads(record.read_text())["steps"]}
    assert got["a"] == ("executed", "s1")


def test_manual_settle_cancels_pending_jobs_and_records_why(
    site: Site, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--finalize --submission`` scancels the pending jobs; a job that ended unrecorded gets its sacct state."""
    calls: list[list[str]] = []
    scancel_fails = {"now": False}

    def fake(argv: list[str], script: str, timeout: float | None = None) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        if argv[0] == "sacct":
            return subprocess.CompletedProcess(argv, 0, "TIMEOUT\n", "")
        return subprocess.CompletedProcess(argv, 1 if argv[0] == "scancel" and scancel_fails["now"] else 0)

    monkeypatch.setattr(slurm, "default_runner", fake)
    path = site.manifest("demo", _step("a"), _step("b", needs=["a"]), _step("c"))
    runs, day = site.tmp / "runs", datetime.date.fromisoformat(DATE)
    rows: list[dict[str, Any]] = [
        {"step": s, "key": None, "status": "pending", "exit": None, "codes": [], "job": j, "submission": "s1"}
        for s, j in (("a", "11"), ("b", "12"))
    ]
    rows.append({"step": "c", "key": "f" * 64, "status": "executed", "exit": 0, "codes": [], "submission": "s1"})
    run.update_run_record(runs, day, "demo", rows, order=["a", "b", "c"])
    argv = ["run", str(path), *site.common(), "--date", DATE, "--finalize", "--submission", "s1"]
    assert site.main(*argv) == cli.EXIT_OK
    assert calls[0] == ["scancel", "11", "12"]
    got = {r["step"]: r for r in json.loads((runs / DATE / "demo" / run.RUN_JSON).read_text())["steps"]}
    assert got["a"]["codes"] == ["E_JOB_ENDED_WITHOUT_RECORD"] and got["a"]["slurm_state"] == "TIMEOUT"
    assert got["b"]["codes"] == ["E_UPSTREAM_NOT_DONE"] and "slurm_state" not in got["b"]
    assert got["c"]["status"] == "executed"

    run.update_run_record(runs, day, "demo", rows[:2], order=["a", "b", "c"])
    scancel_fails["now"] = True
    assert site.main(*argv) == cli.EXIT_OK
    assert "E_SCANCEL_FAILED: job(s) 11,12 may still be queued" in capsys.readouterr().err


def test_a_changed_manifest_marks_dependents_and_keeps_row_order(
    site: Site, capsys: pytest.CaptureFixture[str]
) -> None:
    """``E_MANIFEST_CHANGED`` goes on the job's row and its pending dependents', in the record's order."""
    path = site.manifest("demo", _step("a"), _step("b", needs=["a"]), _step("c"))
    runs, day = site.tmp / "runs", datetime.date.fromisoformat(DATE)
    rows: list[dict[str, Any]] = [
        {"step": s, "key": None, "status": "pending", "exit": None, "codes": [], "job": "1", "submission": "s1"}
        for s in ("c", "a", "b")
    ]
    run.update_run_record(runs, day, "demo", rows, order=["c", "a", "b"])
    argv = ["run", str(path), *site.common(), "--date", DATE, "--submission", "s1", "--pin", "0" * 64, "--step", "a"]
    assert site.main(*argv) == cli.EXIT_FAILED
    got = json.loads((runs / DATE / "demo" / run.RUN_JSON).read_text())["steps"]
    assert [(r["step"], r["status"], r["codes"]) for r in got] == [
        ("c", "pending", []),
        ("a", "failed", ["E_MANIFEST_CHANGED"]),
        ("b", "never-ran", ["E_MANIFEST_CHANGED"]),
    ]
