"""Tests for the registered-input gate: the pre-check, the post-run check and the link rule.

Every site is synthetic, under ``tmp_path``.
"""

from __future__ import annotations

import builtins
import datetime
import hashlib
import io
import json
import os
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from senselab.utils.reproduce import cli, inputs, run, sandbox, slurm, source
from senselab.utils.reproduce.inputs import RegisteredInput
from tests.utils.reproduce.run_test import World
from tests.utils.reproduce.slurm_test import FakeSbatch
from tests.utils.reproduce.sources_fixture import register_input, verify_input

NOW = datetime.datetime(2026, 3, 4, 12, 0, tzinfo=datetime.UTC)
LINE = re.compile(
    r"^[A-Za-z0-9_.-]+\t(ok|mismatch)\t"
    r"(files|missing|size|hash|mtime_only|links|extra|links_out|unlinked|paired|purged)=\d+"
    r"( (files|missing|size|hash|mtime_only|links|extra|links_out|unlinked|paired|purged)=\d+)*$"
)
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
    verify_input(world.base, world.profile(), "tables")
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
    broken = inputs.verify(_entry(world), world.profile(), now=NOW).record
    assert broken.status == "mismatch" and broken.counts["unlinked"] == 1 and broken.counts["paired"] == 3
    assert broken.counts["hash"] == 0
    target.unlink()
    assert _precheck(world) == ["E_INPUT_MISMATCH"]
    os.link(tables / "sub" / "u.csv", target)
    assert _precheck(world) == []
    restored = inputs.verify(_entry(world), world.profile(), now=NOW).record
    assert restored.status == "ok" and restored.counts["unlinked"] == 0

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
    verify_input(world.base, world.profile(), "tables")
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


# --- A6: verify-inputs, records, overlay, submit rule, mismatch trigger -------------------------------------


def _files(world: World) -> tuple[Path, Path]:
    """The profile file (with a deny file) and the manifest file of the latest `_manifest`."""
    (world.tmp / "deny.txt").write_text("FORBIDDEN-[0-9]+\n")
    profile = world.tmp / "profile.yaml"
    profile.write_text(yaml.safe_dump(world.profile().model_dump(mode="json")))
    return profile, world.base / "demo.yaml"


def _write_manifest(world: World, manifest: Any) -> Path:  # noqa: ANN401
    path = world.base / "demo.yaml"
    path.write_text(yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False))
    return path


def _verify_cli(world: World, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, list[str]]:
    profile, _ = _files(world)
    code = cli.main(["verify-inputs", "--profile", str(profile), "--base-dir", str(world.base), *args])
    return code, capsys.readouterr().out.splitlines()


def _counts(line: str) -> dict[str, int]:
    return {k: int(v) for k, v in (field.split("=") for field in line.split("\t")[2].split(" "))}


def _submit_config(world: World, manifest: Any) -> slurm.SubmitConfig:  # noqa: ANN401
    profile, _ = _files(world)
    path = _write_manifest(world, manifest)
    return slurm.SubmitConfig(
        profile=world.profile(),
        profile_path=profile,
        manifest_paths=[path],
        base_dir=world.base,
        cache=world.cache,
        date=datetime.date(2026, 3, 4),
        exports={},
        driver_python=sys.executable,
        submission="sub1",
    )


def _rows(world: World) -> dict[str, dict[str, Any]]:
    path = world.tmp / "runs" / "2026-03-04" / "demo" / run.RUN_JSON
    return {r["step"]: r for r in run.read_run_record(path)}


@needs_userns
def test_one_byte_change_with_size_and_mtime_restored_is_caught_by_verify_and_blocks(
    world: World, capsys: pytest.CaptureFixture[str]
) -> None:
    """SC1(a): a same-size edit with the mtime put back is caught by ``verify-inputs`` and then blocks the step.

    The pre-check passes, ``verify-inputs`` prints ``mismatch`` with ``hash=1``, the dry run is ``blocked
    E_INPUT_UNVERIFIED``, and submit issues no sbatch.
    """
    tables = _tables(world)
    _register(world)
    manifest = _manifest(world)
    code, lines = _verify_cli(world, capsys, "tables")
    assert code == cli.EXIT_OK and lines[0].split("\t")[:2] == ["tables", "ok"]
    assert _plan(world, manifest).status == "will-execute"

    path = tables / "sub" / "u.csv"
    st = os.stat(path)
    path.write_text("abd\n")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert _precheck(world) == []
    code, lines = _verify_cli(world, capsys, "tables")
    assert code == cli.EXIT_FAILED and lines[0].split("\t")[:2] == ["tables", "mismatch"]
    assert _counts(lines[0])["hash"] == 1 and _counts(lines[0])["size"] == 0

    plan = _plan(world, manifest)
    assert plan.status == "blocked" and plan.codes == ("E_INPUT_UNVERIFIED",)
    profile, _ = _files(world)
    dry = ["run", str(_write_manifest(world, manifest)), "--profile", str(profile), "--base-dir", str(world.base)]
    assert cli.main([*dry, "--cache", str(world.cache), "--dry-run"]) == cli.EXIT_FAILED
    assert capsys.readouterr().out.split("\t")[2::3][:1] == ["blocked"]
    assert cli.main([*dry, "--cache", str(world.cache), "--dry-run"]) == cli.EXIT_FAILED
    assert capsys.readouterr().out.rstrip("\n").endswith("\tE_INPUT_UNVERIFIED")
    fake = FakeSbatch()
    submission = slurm.submit([manifest], _submit_config(world, manifest), runner=fake)
    assert fake.calls == [] and submission.jobs == [] and submission.verify_jobs == ()
    assert submission.refused == {"prep": ("E_INPUT_UNVERIFIED",)}
    assert {s: (r["status"], r["codes"]) for s, r in _rows(world).items()} == {
        "prep": ("never-ran", ["E_INPUT_UNVERIFIED"])
    }
    run.reserve_run_record(world.tmp / "runs", datetime.date(2026, 3, 4), "demo", ["prep"], "other")
    config = _submit_config(world, manifest)
    assert slurm.submit([manifest], config, runner=FakeSbatch()).refused == {"prep": ("E_INPUT_UNVERIFIED",)}
    assert (_rows(world)["prep"]["status"], _rows(world)["prep"]["submission"]) == ("pending", "other")


@needs_userns
def test_no_verification_record_blocks_e_input_unverified(world: World) -> None:
    """SC1(c): a registered input with no record is ``blocked`` at dry run and refused by a run, until verified."""
    _tables(world)
    _register(world)
    manifest = _manifest(world)
    plan = _plan(world, manifest)
    assert plan.status == "blocked" and plan.codes == ("E_INPUT_UNVERIFIED",)
    outcome = world.run(manifest)["prep"]
    assert outcome.status == "failed" and outcome.codes == ("E_INPUT_UNVERIFIED",) and outcome.out is None
    verify_input(world.base, world.profile(), "tables")
    assert _plan(world, manifest).status == "will-execute"
    assert world.run(manifest)["prep"].ok


def test_mtime_only_drift_records_ok_with_an_overlay_and_keeps_the_key(world: World) -> None:
    """Only mtimes drifted: the record is ``ok`` with an overlay the pre-check honours; the key is unchanged."""
    tables = _tables(world)
    _register(world)
    manifest = _manifest(world)
    step, config = manifest.steps[0], world.config()
    verify_input(world.base, world.profile(), "tables")
    key = run._key(step, "demo", config, {}).key
    assert run.gate(step, "demo", config).codes == ()
    st = os.stat(tables / "t.csv")
    os.utime(tables / "t.csv", ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert run.gate(step, "demo", config).codes == ("E_INPUT_MISMATCH",)

    record = verify_input(world.base, world.profile(), "tables")
    assert record.status == "ok" and record.counts["mtime_only"] == 1 and record.overlay_sha256 is not None
    overlay = world.tmp / "input-verify" / "tables.mtime-overlay.tsv"
    assert overlay.is_file() and (overlay.stat().st_mode & 0o777) == 0o660
    assert run.gate(step, "demo", config).codes == ()
    assert run._key(step, "demo", config, {}).key == key
    manifest_file = world.data / "_manifests" / "tables.xxh3.tsv"
    assert _entry(world).manifest.sha256 == hashlib.sha256(manifest_file.read_bytes()).hexdigest()

    overlay.write_text(overlay.read_text() + "sub/u.csv\t1\n")
    assert run.gate(step, "demo", config).codes == ("E_INPUT_MISMATCH", "E_INPUT_UNVERIFIED")


def test_a_record_older_than_31_days_blocks(world: World) -> None:
    """A record is current for 31 days; a day later the input is unverified."""
    _tables(world)
    _register(world)
    entry, profile = _entry(world), world.profile()
    verify_input(world.base, profile, "tables", now=NOW)
    assert inputs.verified_codes(entry, profile, now=NOW + datetime.timedelta(days=31)) == []
    assert inputs.verified_codes(entry, profile, now=NOW + datetime.timedelta(days=32)) == ["E_INPUT_UNVERIFIED"]
    verify_input(world.base, profile, "tables", now=NOW + datetime.timedelta(days=32))
    assert inputs.verified_codes(entry, profile, now=NOW + datetime.timedelta(days=32)) == []
    assert inputs.verified_codes(entry, profile, now=NOW + datetime.timedelta(days=31)) == ["E_INPUT_UNVERIFIED"]


def test_an_extra_file_is_a_mismatch_unless_extras_ignore(world: World) -> None:
    """A selected file absent from the manifest is ``extra``; ``extras: ignore`` and excluded files do not count."""
    tables = _tables(world)
    _register(world, exclude=["**/*.tmp"])
    entry, profile = _entry(world), world.profile()
    assert inputs.verify(entry, profile, now=NOW).record.status == "ok"
    (tables / "sub" / "scratch.tmp").write_text("x\n")
    assert inputs.verify(entry, profile, now=NOW).record.counts["extra"] == 0
    (tables / "sub" / "new.csv").write_text("x\n")
    record = inputs.verify(entry, profile, now=NOW).record
    assert record.status == "mismatch" and record.counts["extra"] == 1
    ignored = entry.model_copy(update={"extras": "ignore"})
    assert inputs.verify(ignored, profile, now=NOW).record.status == "ok"

    one = world.data / "single"
    one.mkdir()
    (one / "a.csv").write_text("1\n")
    (one / "b.csv").write_text("2\n")
    _register(world, "single", relpath="single", include=["a.csv"])
    assert inputs.verify(_entry(world, "single"), profile, now=NOW).record.counts["files"] == 1
    (one / "c.csv").write_text("3\n")
    assert inputs.verify(_entry(world, "single"), profile, now=NOW).record.status == "ok"


def test_a_newer_mismatch_record_supersedes_an_older_ok(world: World) -> None:
    """Only the latest record counts: ok, then mismatch, blocks; a later ok clears it; another manifest sha blocks."""
    tables = _tables(world)
    _register(world)
    entry, profile = _entry(world), world.profile()
    verify_input(world.base, profile, "tables", now=NOW)
    assert inputs.verified_codes(entry, profile, now=NOW) == []
    path = tables / "t.csv"
    st = os.stat(path)
    path.write_text("4\n")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert verify_input(world.base, profile, "tables", now=NOW).status == "mismatch"
    assert inputs.verified_codes(entry, profile, now=NOW) == ["E_INPUT_UNVERIFIED"]
    records = json.loads((world.tmp / "input-verify" / "tables.json").read_text())["records"]
    assert [r["status"] for r in records] == ["ok", "mismatch"]
    path.write_text("3\n")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    later = NOW + datetime.timedelta(seconds=1)
    verify_input(world.base, profile, "tables", now=later)
    assert inputs.verified_codes(entry, profile, now=later) == []
    other = entry.model_copy(update={"manifest": entry.manifest.model_copy(update={"sha256": "cd" * 32})})
    assert inputs.verified_codes(other, profile, now=later) == ["E_INPUT_UNVERIFIED"]

    directory = world.tmp / "input-verify"
    newer = inputs.verify(entry, profile, now=later + datetime.timedelta(seconds=2))
    stale_mismatch = newer._replace(record=newer.record.model_copy(update={"status": "mismatch"}))
    inputs.append_record(directory, "tables", stale_mismatch)
    inputs.append_record(directory, "tables", inputs.verify(entry, profile, now=later + datetime.timedelta(seconds=1)))
    assert inputs.verified_codes(entry, profile, now=later + datetime.timedelta(seconds=3)) == ["E_INPUT_UNVERIFIED"]
    tie = newer._replace(record=newer.record.model_copy(update={"status": "ok"}))
    inputs.append_record(directory, "tables", tie)
    assert inputs.verified_codes(entry, profile, now=later + datetime.timedelta(seconds=3)) == ["E_INPUT_UNVERIFIED"]

    (directory / "tables.json").write_text("{not json")
    with pytest.raises(inputs.InputError) as info:
        verify_input(world.base, profile, "tables", now=later)
    assert info.value.code == "E_INPUT_RECORD" and (directory / "tables.json").read_text() == "{not json"


def test_verify_output_has_counts_only(world: World, capsys: pytest.CaptureFixture[str]) -> None:
    """One line per id of counts only; ``links_out`` counts links outside every registry entry."""
    tables = _tables(world)
    elsewhere = world.data / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "e.csv").write_text("1\n")
    shared = world.data / "shared"
    shared.mkdir()
    (shared / "s.csv").write_text("1\n")
    (tables / "out.csv").symlink_to("../../elsewhere/e.csv")
    (tables / "in.csv").symlink_to("../../shared/s.csv")
    _register(world)
    _register(world, "shared", relpath="shared")
    code, lines = _verify_cli(world, capsys, "--all")
    assert code == cli.EXIT_OK and [line.split("\t")[0] for line in lines] == ["shared", "tables"]
    assert all(LINE.match(line) for line in lines), lines
    assert _counts(lines[1])["links_out"] == 1 and _counts(lines[1])["links"] == 0
    assert _counts(lines[0])["links_out"] == 0
    text = "\n".join(lines)
    assert str(world.tmp) not in text and "csv" not in text
    code, lines = _verify_cli(world, capsys)
    assert code == cli.EXIT_USAGE

    directory = world.tmp / "input-verify"
    for path in [directory, *directory.iterdir()]:
        path.unlink() if path.is_file() else None
    directory.rmdir()
    saved = os.umask(0o022)
    try:
        assert _verify_cli(world, capsys, "tables")[0] == cli.EXIT_OK
    finally:
        os.umask(saved)
    assert directory.stat().st_mode & 0o070 == 0o070
    assert (directory / "tables.json").stat().st_mode & 0o777 == 0o660
    # The records lock is an exclusive `SharedFileLock` (A17 fix round 1): released, it leaves no lock files.
    assert not [p.name for p in directory.iterdir() if p.name.startswith("tables.json.")]
    directory.chmod(0o500)
    try:
        code, lines = _verify_cli(world, capsys, "tables")
        err = capsys.readouterr().err
    finally:
        directory.chmod(0o770)
    assert code == cli.EXIT_FAILED and lines == ["tables\tE_DRIVER"] and str(world.tmp) not in err


def test_registry_only_mode_hashes_nothing(
    world: World, capsys: pytest.CaptureFixture[str], opened: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--registry-only`` checks each registry file, its manifest sha and root line, and opens no input file."""
    tables = _tables(world)
    _register(world)
    spaced = world.data / "derivatives" / "two words"
    spaced.mkdir()
    (spaced / "w.csv").write_text("1\n")
    _register(world, "spaced", relpath="derivatives/two words")

    def refuse(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise AssertionError("hashed")

    monkeypatch.setattr(inputs, "_hashes", refuse)
    opened.clear()
    code, lines = _verify_cli(world, capsys, "--all", "--registry-only")
    assert code == cli.EXIT_OK and lines == ["spaced\tregistry-ok", "tables\tregistry-ok"]
    assert not [p for p in opened if p.startswith((str(tables) + os.sep, str(spaced) + os.sep))]
    assert not (world.tmp / "input-verify").exists()

    registry = world.base / "inputs" / "bad.yaml"
    registry.write_text(
        yaml.safe_dump({**yaml.safe_load((world.base / "inputs" / "tables.yaml").read_text()), "id": "bad"})
    )
    data = yaml.safe_load(registry.read_text())
    data["manifest"]["sha256"] = "cd" * 32
    registry.write_text(yaml.safe_dump(data))
    moved = world.base / "inputs" / "moved.yaml"
    data = yaml.safe_load((world.base / "inputs" / "tables.yaml").read_text())
    moved.write_text(yaml.safe_dump({**data, "id": "moved", "relpath": "derivatives"}))
    code, lines = _verify_cli(world, capsys, "--all", "--registry-only")
    assert code == cli.EXIT_FAILED
    assert lines == ["bad\tE_INPUT_REGISTRY", "moved\tE_INPUT_REGISTRY", "spaced\tregistry-ok", "tables\tregistry-ok"]


def test_a_precheck_mismatch_at_submit_submits_a_verify_job(
    world: World, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-check mismatch at submit: one ``verify-inputs <id>`` job on the build partition, no step job."""
    tables = _tables(world)
    _register(world)
    manifest = _manifest(world)
    verify_input(world.base, world.profile(), "tables")
    st = os.stat(tables / "t.csv")
    os.utime(tables / "t.csv", ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    fake = FakeSbatch()
    monkeypatch.setattr(slurm, "default_runner", fake)
    monkeypatch.setenv("UV_CACHE_DIR", str(world.tmp / "uv-cache"))
    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", str(world.tmp / "uv-python"))
    profile, _ = _files(world)
    path = _write_manifest(world, manifest)
    argv = ["submit", str(path), "--profile", str(profile), "--base-dir", str(world.base), "--cache", str(world.cache)]
    assert cli.main([*argv, "--date", "2026-03-04"]) == cli.EXIT_FAILED
    out = capsys.readouterr().out.splitlines()
    scripts = fake.submissions()
    assert len(scripts) == 1 and len(fake.calls) == 1
    assert "--hold" not in fake.calls[0][0]
    last = scripts[0].splitlines()[-1]
    assert " verify-inputs " in last and last.endswith(" tables") and "--step" not in last
    assert "#SBATCH --partition=p" in scripts[0] and "#SBATCH --export=NONE" in scripts[0]
    assert "verify-submitted\ttables\t101" in out
    assert any(line.startswith("prep\t-\tnever-ran\t-\tE_INPUT_MISMATCH") for line in out), out
    assert _rows(world)["prep"]["codes"] == ["E_INPUT_MISMATCH"]
    assert any(a.startswith("--comment=") for a in fake.calls[0][0])
    head = scripts[0].splitlines()
    assert {"#SBATCH --cpus-per-task=1", "#SBATCH --mem=1G", "#SBATCH --time=0:10:00"} <= set(head)

    failing = FakeSbatch(fail_at=1)
    monkeypatch.setattr(slurm, "default_runner", failing)
    assert cli.main([*argv, "--date", "2026-03-04"]) == cli.EXIT_FAILED
    captured = capsys.readouterr()
    assert "verify-not-submitted\ttables" in captured.out.splitlines()
    assert any(line.startswith("prep\t-\tnever-ran") for line in captured.out.splitlines())

    date = datetime.date(2026, 3, 4)
    done = {"step": "prep", "key": "ab" * 32, "status": "executed", "exit": 0, "codes": []}
    run.update_run_record(world.tmp / "runs", date, "demo", [done], order=["prep"])
    assert cli.main([*argv, "--date", "2026-03-04"]) == cli.EXIT_FAILED
    capsys.readouterr()
    assert {k: _rows(world)["prep"][k] for k in done} == done
    failed = {**done, "status": "failed", "exit": 1, "codes": ["E_STEP_FAILED"]}
    run.update_run_record(world.tmp / "runs", date, "demo", [failed], order=["prep"])
    assert cli.main([*argv, "--date", "2026-03-04"]) == cli.EXIT_FAILED
    capsys.readouterr()
    assert (_rows(world)["prep"]["status"], _rows(world)["prep"]["codes"]) == ("never-ran", ["E_INPUT_MISMATCH"])


def test_an_added_file_is_a_mismatch_before_any_full_verify(world: World) -> None:
    """The manifest's dirs sidecar lets the ``lstat`` pre-check see an added file or dir (``E_INPUT_MISMATCH``).

    Each dir holding a row is listed: a selected entry the manifest lacks, or a new subdir that could hold one, is
    a mismatch; an entry the input's ``exclude`` leaves out is not. ``extras: ignore`` skips the check.
    """
    tables = _tables(world)
    _register(world, exclude=["**/*.tmp"])
    assert inputs.dirs_sidecar(_entry(world), world.profile()).is_file()
    assert _precheck(world) == []
    for added in (tables / "new.csv", tables / "sub" / "v.csv", tables / "sub" / "deeper" / "w.csv"):
        added.parent.mkdir(exist_ok=True)
        added.write_text("1\n")
        assert _precheck(world) == ["E_INPUT_MISMATCH"], added.name
        added.unlink()
        if added.parent.name == "deeper":
            added.parent.rmdir()
        assert _precheck(world) == []
    (tables / "sub" / "scratch.tmp").write_text("x\n")
    assert _precheck(world) == []
    (tables / "new.csv").write_text("1\n")
    _register(world, exclude=["**/*.tmp"], extras="ignore")
    (tables / "other.csv").write_text("1\n")
    assert _precheck(world) == []


@needs_userns
def test_a_file_added_after_verify_blocks_the_dry_run(world: World) -> None:
    """A file added to a verified registered input blocks the dry run with ``E_INPUT_MISMATCH``."""
    tables = _tables(world)
    _register(world)
    verify_input(world.base, world.profile(), "tables")
    manifest = _manifest(world)
    assert _plan(world, manifest).status == "will-execute"
    (tables / "sub" / "added.csv").write_text("1\n")
    plan = _plan(world, manifest)
    assert (plan.status, plan.codes) == ("blocked", ("E_INPUT_MISMATCH",))
