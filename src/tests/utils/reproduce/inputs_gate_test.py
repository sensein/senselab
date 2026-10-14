"""Tests for the registered-input gate: the pre-check, the post-run check and the link rule.

Every site is synthetic, under ``tmp_path``.
"""

from __future__ import annotations

import builtins
import io
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from senselab.utils.reproduce import cli, inputs, run, sandbox, source
from senselab.utils.reproduce.inputs import RegisteredInput
from tests.utils.reproduce.run_test import World
from tests.utils.reproduce.sources_fixture import register_input

needs_userns = pytest.mark.skipif(not sandbox.userns_available(), reason="unprivileged user namespaces are unavailable")


@pytest.fixture
def world(tmp_path: Path) -> World:
    """A fresh synthetic site."""
    return World(tmp_path)


def _tables(world: World, relpath: str = "derivatives/tables") -> Path:
    tables = world.data / relpath
    (tables / "sub").mkdir(parents=True)
    (tables / "t.csv").write_text("3\n")
    (tables / "sub" / "u.csv").write_text("abc\n")
    return tables


def _register(world: World, input_id: str = "tables", relpath: str = "derivatives/tables", **over: Any) -> None:  # noqa: ANN401
    world.registered[input_id] = register_input(
        world.base, {"data": world.data}, input_id, root="data", relpath=relpath, **over
    )


def _entry(world: World, input_id: str = "tables") -> RegisteredInput:
    return inputs.read_registry_entry(world.base, input_id)[0]


def _precheck(world: World, input_id: str = "tables", overlay: dict[str, int] | None = None) -> list[str]:
    entry = _entry(world, input_id)
    return inputs.precheck(entry, world.profile(), {input_id: entry}, overlay)


def _manifest(world: World, lock_inputs: tuple[str, ...] = ("tables",)) -> Any:  # noqa: ANN401
    step = world.step("prep", "prep", "{input:tables}/t.csv", inputs=[], lock_inputs=list(lock_inputs))
    manifest = world.manifest(step)
    world.prefetch(manifest)
    return manifest


def _plan(world: World, manifest: Any) -> run.Plan:  # noqa: ANN401
    (plan,) = run.plan_steps([manifest], world.config())
    return plan


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Every path passed to ``builtins.open``, ``io.open`` or ``os.open`` while the test runs."""
    seen: list[str] = []
    real_open, real_io_open, real_os_open = builtins.open, io.open, os.open

    def wrap(real: Any) -> Any:  # noqa: ANN401
        def opener(file: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if isinstance(file, (str, os.PathLike)):
                seen.append(os.path.abspath(os.fspath(file)))
            return real(file, *args, **kwargs)

        return opener

    monkeypatch.setattr(builtins, "open", wrap(real_open))
    monkeypatch.setattr(io, "open", wrap(real_io_open))
    monkeypatch.setattr(os, "open", wrap(real_os_open))
    yield seen


@needs_userns
def test_dry_run_with_a_deleted_file_is_blocked_e_input_missing(
    world: World, opened: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SC1(b): a deleted manifest file makes the dry run ``blocked E_INPUT_MISSING``; no input file is opened.

    The gate's codes are listed beside a missing checkout or a key refusal too.
    """
    tables = _tables(world)
    _register(world)
    manifest = _manifest(world)
    first = _plan(world, manifest)
    assert first.status == "will-execute" and first.codes == ()
    (tables / "sub" / "u.csv").unlink()
    opened.clear()
    plan = _plan(world, manifest)
    assert plan.status == "blocked" and "E_INPUT_MISSING" in plan.codes
    assert not [p for p in opened if p == str(tables) or p.startswith(str(tables) + os.sep)]

    (world.base / "adapter.py").rename(world.base / "adapter.moved")
    assert set(_plan(world, manifest).codes) >= {"E_ADAPTER_MISSING", "E_INPUT_MISSING"}
    (world.base / "adapter.moved").rename(world.base / "adapter.py")

    def no_checkout(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise source.ReproduceSourceError(source.WORKTREE_MISSING)

    monkeypatch.setattr(source, "prefetched_checkout", no_checkout)
    assert set(_plan(world, manifest).codes) >= {"E_SRC_MISSING", "E_INPUT_MISSING"}


def test_a_size_change_is_e_input_mismatch(world: World) -> None:
    """A file one byte longer than its manifest row is ``E_INPUT_MISMATCH``."""
    tables = _tables(world)
    _register(world)
    assert _precheck(world) == []
    with open(tables / "t.csv", "a") as handle:
        handle.write("4\n")
    assert _precheck(world) == ["E_INPUT_MISMATCH"]


def test_an_mtime_change_alone_is_e_input_mismatch_without_overlay(world: World) -> None:
    """Same bytes, another mtime: ``E_INPUT_MISMATCH``, unless an overlay accepts exactly that mtime."""
    tables = _tables(world)
    _register(world)
    st = os.stat(tables / "t.csv")
    os.utime(tables / "t.csv", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    moved = os.stat(tables / "t.csv").st_mtime_ns
    assert _precheck(world) == ["E_INPUT_MISMATCH"]
    assert _precheck(world, overlay={"t.csv": moved}) == []
    assert _precheck(world, overlay={"t.csv": moved + 1}) == ["E_INPUT_MISMATCH"]
    assert _precheck(world, overlay={"sub/u.csv": moved}) == ["E_INPUT_MISMATCH"]


def test_a_one_byte_change_with_size_and_mtime_restored_passes_the_precheck(world: World) -> None:
    """The pre-check reads no content: a same-size edit with the mtime put back passes it (verify catches it)."""
    tables = _tables(world)
    _register(world)
    path = tables / "sub" / "u.csv"
    st = os.stat(path)
    path.write_text("abd\n")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert _precheck(world) == []


def test_a_changed_link_target_is_e_input_mismatch(world: World) -> None:
    """A link re-pointed, or replaced by a file, is ``E_INPUT_MISMATCH``; a symlinked parent dir is too."""
    tables = _tables(world)
    (tables / "alias.csv").symlink_to("t.csv")
    _register(world)
    assert _precheck(world) == []
    (tables / "alias.csv").unlink()
    (tables / "alias.csv").symlink_to("sub/u.csv")
    assert _precheck(world) == ["E_INPUT_MISMATCH"]
    (tables / "alias.csv").unlink()
    (tables / "alias.csv").write_text("3\n")
    assert _precheck(world) == ["E_INPUT_MISMATCH"]
    (tables / "alias.csv").unlink()
    (tables / "alias.csv").symlink_to("t.csv")
    assert _precheck(world) == []
    copy = world.data / "copy-of-sub"
    (tables / "sub").rename(copy)
    (tables / "sub").symlink_to(copy)
    assert _precheck(world) == ["E_INPUT_MISMATCH"]
    (tables / "sub").unlink()
    assert _precheck(world) == ["E_INPUT_MISSING"]


def test_a_broken_hardlink_between_derivative_and_original_is_e_input_mismatch(world: World) -> None:
    """Derivative and original must share each file's inode; a rewrite-and-rename of the original breaks it."""
    original = world.data / "orig" / "tables"
    (original / "sub").mkdir(parents=True)
    (original / "t.csv").write_text("3\n")
    (original / "sub" / "u.csv").write_text("abc\n")
    tables = world.data / "derivatives" / "tables"
    (tables / "sub").mkdir(parents=True)
    os.link(original / "t.csv", tables / "t.csv")
    os.link(original / "sub" / "u.csv", tables / "sub" / "u.csv")
    (original / "sub" / "dataset_description.json").write_text("{}\n")
    os.link(original / "sub" / "dataset_description.json", tables / "sub" / "dataset_description.json")
    (tables / "dataset_description.json").write_text('{"DatasetType": "derivative"}\n')
    _register(world, original={"root": "data", "relpath": "orig/tables"})
    assert _precheck(world) == []
    (original / "sub" / "dataset_description.json").unlink()
    assert _precheck(world) == ["E_INPUT_MISMATCH"]
    os.link(tables / "sub" / "dataset_description.json", original / "sub" / "dataset_description.json")
    assert _precheck(world) == []

    target = original / "sub" / "u.csv"
    st = os.stat(target)
    temp = original / "sub" / ".u.csv.tmp"
    temp.write_bytes(target.read_bytes())
    os.utime(temp, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.replace(temp, target)
    assert os.stat(target).st_size == st.st_size and os.stat(target).st_mtime_ns == st.st_mtime_ns
    assert _precheck(world) == ["E_INPUT_MISMATCH"]
    target.unlink()
    assert _precheck(world) == ["E_INPUT_MISMATCH"]
    os.link(tables / "sub" / "u.csv", target)
    assert _precheck(world) == []

    _register(world, "orphan", original={"root": "elsewhere", "relpath": "orig/tables"})
    manifest = world.manifest(world.step("prep", "prep", "{input:orphan}/t.csv", inputs=[], lock_inputs=["orphan"]))
    assert run.gate(manifest.steps[0], "demo", world.config()).codes == ("E_ROOT_MISSING",)


@needs_userns
def test_a_registered_input_changed_during_the_step_is_e_input_mutated(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pre-check is repeated after the step: an mtime moved while it ran fails it with ``E_INPUT_MUTATED``."""
    tables = _tables(world)
    _register(world)
    manifest = _manifest(world)
    real_exec = run._Execution._exec

    def exec_then_touch(self: run._Execution) -> None:
        real_exec(self)
        st = os.stat(tables / "t.csv")
        os.utime(tables / "t.csv", ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))

    monkeypatch.setattr(run._Execution, "_exec", exec_then_touch)
    outcome = world.run(manifest)["prep"]
    assert outcome.status == "executed" and "E_INPUT_MUTATED" in outcome.codes and outcome.exit != 0
    assert outcome.out is not None and not (outcome.out / run.METRICS).exists()


@needs_userns
def test_a_link_into_an_unregistered_tree_blocks_and_is_a_lint_finding(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    """A link whose target is in no registered input of the step blocks it and is a lint finding (count only)."""
    tables = _tables(world)
    outside = world.data / "outside"
    outside.mkdir()
    (outside / "x.csv").write_text("9\n")
    (tables / "x.csv").symlink_to("../../outside/x.csv")
    (tables / "own.csv").symlink_to("sub/u.csv")
    _register(world)
    _register(world, "outside", relpath="outside")
    manifest = _manifest(world)
    plan = _plan(world, manifest)
    assert plan.status == "blocked" and "E_INPUT_LINK_UNREGISTERED" in plan.codes

    (world.tmp / "deny.txt").write_text("FORBIDDEN-[0-9]+\n")
    profile = world.tmp / "profile.yaml"
    profile.write_text(yaml.safe_dump(world.profile().model_dump(mode="json")))

    def lint(manifest: Any) -> tuple[int, str]:  # noqa: ANN401
        path = world.base / "m.yaml"
        path.write_text(yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False))
        code = cli.main(["lint", str(path), "--profile", str(profile), "--base-dir", str(world.base)])
        return code, capsys.readouterr().out

    code, out = lint(manifest)
    assert code == cli.EXIT_FAILED and "input-link-unregistered tables 1" in out
    assert "x.csv" not in out and str(world.tmp) not in out.replace(str(world.base / "m.yaml"), "")

    both = _manifest(world, ("tables", "outside"))
    assert "E_INPUT_LINK_UNREGISTERED" not in _plan(world, both).codes
    code, out = lint(both)
    assert code == cli.EXIT_OK, out
