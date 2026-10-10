"""``run --dry-run``: keys and statuses without executing, and nothing written.

Every site is synthetic, under ``tmp_path``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from senselab.utils.reproduce import cli, env, run, sandbox
from tests.utils.reproduce.mutation_test import _dag, _write_inputs
from tests.utils.reproduce.run_test import World

needs_userns = pytest.mark.skipif(not sandbox.userns_available(), reason="unprivileged user namespaces are unavailable")


@pytest.fixture
def world(tmp_path: Path) -> World:
    """A fresh synthetic site."""
    return World(tmp_path)


def _argv(world: World, path: Path, profile: Path, *extra: str) -> list[str]:
    common = ["--profile", str(profile), "--base-dir", str(world.base), "--cache", str(world.cache)]
    return ["run", str(path), *common, *extra]


def _lines(text: str) -> dict[str, list[str]]:
    return {line.split("\t")[0]: line.split("\t") for line in text.splitlines() if line.strip()}


def _tree(root: Path) -> dict[str, tuple[int, float, str]]:
    """Every entry under ``root`` (links not followed): kind, mtime and, for links, the target."""
    found: dict[str, tuple[int, float, str]] = {}
    if not root.exists():
        return found
    for top, dirs, files in os.walk(root, followlinks=False):
        for name in [*dirs, *files]:
            path = Path(top) / name
            st = os.lstat(path)
            found[str(path.relative_to(root))] = (
                st.st_mode,
                st.st_mtime_ns,
                os.readlink(path) if path.is_symlink() else "",
            )
    return found


@needs_userns
def test_dry_run_reports_skips_and_what_a_patch_would_rerun(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    """After a run every step reads ``skipped``; after a patch, ``prep`` will execute and its dependents wait on it."""
    manifest = _dag(world)
    world.prefetch(manifest)
    profile, path = _write_inputs(world, manifest)
    dry = _argv(world, path, profile, "--dry-run")
    assert cli.main(dry) == cli.EXIT_OK
    first = _lines(capsys.readouterr().out)
    assert [first[s][2] for s in ("prep", "score", "rstep", "nb")] == [
        "will-execute",
        "will-execute (upstream)",
        "will-execute (upstream)",
        "will-execute (upstream)",
    ]
    assert len(first["prep"][1]) == 8 and first["score"][1] == "-"
    assert first["prep"][3:5] == ["env:ok", "src:ok"]
    assert cli.main(_argv(world, path, profile)) == cli.EXIT_OK
    run_keys = {k: v[1] for k, v in _lines(capsys.readouterr().out).items()}
    assert cli.main(dry) == cli.EXIT_OK
    again = _lines(capsys.readouterr().out)
    assert {s: line[2] for s, line in again.items()} == {s: "skipped" for s in run_keys}
    assert {s: line[1] for s, line in again.items()} == run_keys

    world.write_patch("# v8")
    patched = _dag(world)
    world.prefetch(patched)
    _write_inputs(world, patched)
    assert cli.main(dry) == cli.EXIT_OK
    after = _lines(capsys.readouterr().out)
    assert after["prep"][2] == "will-execute" and after["prep"][1] != run_keys["prep"]
    assert {after[s][2] for s in ("score", "rstep", "nb")} == {"will-execute (upstream)"}
    assert str(world.tmp) not in "".join("\t".join(v) for v in after.values())


@needs_userns
def test_a_missing_venv_is_blocked(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    """A step whose venv was never prefetched reads ``blocked env:missing`` and the exit is 1."""
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    profile, path = _write_inputs(world, manifest)
    venv_path, _ = run._step_venv(manifest.steps[0], manifest.name, world.cache)
    (venv_path / env.MARKER).unlink()
    assert cli.main(_argv(world, path, profile, "--dry-run")) == cli.EXIT_FAILED
    (line,) = _lines(capsys.readouterr().out).values()
    assert line[2] == "blocked" and line[3] == "env:missing" and line[4] == "src:ok" and "E_ENV_MISSING" in line[5]


@needs_userns
def test_a_dry_run_writes_nothing(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    """Results, runs and cache trees are identical before and after a dry run, fresh and after a real run."""
    manifest = _dag(world)
    world.prefetch(manifest)
    profile, path = _write_inputs(world, manifest)
    roots = [world.tmp / "results", world.tmp / "runs", world.cache]
    before = [_tree(r) for r in roots]
    assert cli.main(_argv(world, path, profile, "--dry-run")) == cli.EXIT_OK
    assert [_tree(r) for r in roots] == before and not (world.tmp / "results").exists()
    assert cli.main(_argv(world, path, profile)) == cli.EXIT_OK
    before = [_tree(r) for r in roots]
    assert cli.main(_argv(world, path, profile, "--dry-run")) == cli.EXIT_OK
    assert [_tree(r) for r in roots] == before
    capsys.readouterr()
