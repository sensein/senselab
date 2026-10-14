"""Tests for sources locks: loading by sha256, the pins a step must match, staleness and provenance."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from senselab.utils.reproduce import cli, lock, run
from senselab.utils.reproduce.lock import LockError, SourcesLock
from senselab.utils.reproduce.spec import SourcesRef, StepSpec
from tests.utils.reproduce.cache_test import OTHER, REV, SHA, _step_data
from tests.utils.reproduce.run_test import World, _summary, needs_userns
from tests.utils.reproduce.sources_fixture import lock_data, register_input, verify_input, write_lock

WEIGHTS = [{"repo": "org/model", "revision": REV, "files": [{"path": "model.safetensors", "sha256": SHA}]}]
TABLES = {"registry_sha256": SHA, "manifest_sha256": SHA}


@pytest.fixture
def world(tmp_path: Path) -> World:
    """A fresh synthetic site."""
    return World(tmp_path)


def _codes(step: StepSpec, data: dict[str, Any]) -> list[str]:
    return sorted({code for code, _ in lock.step_findings(step, SourcesLock.model_validate(data))})


def test_lock_references_and_pins_must_match(tmp_path: Path) -> None:
    """Unknown ids and unpinned weights or patterns are ``E_LOCK_REF``; differing pins ``E_LOCK_MISMATCH``."""
    step = StepSpec.model_validate(_step_data(lock_inputs=["tables"], hf_weights=WEIGHTS))
    good = lock_data([step], registered={"tables": TABLES})
    assert _codes(step, good) == []
    staged = StepSpec.model_validate(
        _step_data(
            lock_inputs=["tables"], hf_weights=WEIGHTS, stage=[{"root": "data", "patterns": ["x/*.csv"], "dest": "s"}]
        )
    )
    assert _codes(staged, good) == []
    unstaged = copy.deepcopy(good)
    unstaged["data"] = [e for e in good["data"] if e["kind"] != "pattern"]
    assert sorted(d for c, d in lock.step_findings(staged, SourcesLock.model_validate(unstaged))) == [
        "inputs[0]",
        "stage[0]",
    ]
    for patterns in (["/abs"], ["a/../b"], [""]):
        with pytest.raises(ValueError):
            SourcesLock.model_validate({**good, "data": [{"kind": "pattern", "root": "data", "patterns": patterns}]})
    unknown = StepSpec.model_validate(_step_data(lock_inputs=["other"], hf_weights=WEIGHTS))
    assert _codes(unknown, good) == ["E_LOCK_REF"]
    pattern_only = copy.deepcopy(good)
    pattern_only["data"] = [d for d in good["data"] if d.get("id") != "tables"]
    pattern_only["external"] = [{"id": "tables", "kind": "registered", **TABLES}]
    assert _codes(step, pattern_only) == []

    def change(edit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
        data = copy.deepcopy(good)
        edit(data)
        return data

    refs = [
        lambda d: d["models"][0]["files"][0].update(sha256=OTHER),
        lambda d: d["models"][0].update(revision="2" * 40),
        lambda d: d.update(models=[]),
        lambda d: d.update(data=[e for e in d["data"] if e["kind"] != "pattern"]),
    ]
    for edit in refs:
        assert _codes(step, change(edit)) == ["E_LOCK_REF"]
    mismatches = [
        lambda d: d["code"][0].update(commit="f" * 40),
        lambda d: d["code"][0].update(patches=[OTHER]),
        lambda d: d["envs"][0].update(sha256=OTHER),
        lambda d: d.update(envs=d["envs"][1:]),
    ]
    for edit in mismatches:
        assert _codes(step, change(edit)) == ["E_LOCK_MISMATCH"]

    ref = SourcesRef.model_validate(write_lock(tmp_path, "demo", good))
    loaded, sha = lock.load_lock(tmp_path, ref)
    assert loaded == SourcesLock.model_validate(good) and sha == ref.sha256
    for bad in (ref.model_copy(update={"sha256": OTHER}), ref.model_copy(update={"lock": "sources/none.lock.yaml"})):
        with pytest.raises(LockError) as info:
            lock.load_lock(tmp_path, bad)
        assert info.value.code == "E_LOCK_SHA" and str(info.value) == "E_LOCK_SHA"
    invalid = SourcesRef.model_validate(write_lock(tmp_path, "bad", {**good, "schema": "other"}))
    with pytest.raises(LockError) as info:
        lock.load_lock(tmp_path, invalid)
    assert info.value.code == "E_LOCK_INVALID"
    for wrong in ("locks/demo.lock.yaml", "sources/demo.yaml", "sources/../demo.lock.yaml"):
        with pytest.raises(ValueError):
            SourcesRef.model_validate({"lock": wrong, "sha256": SHA})


def test_a_registry_newer_than_its_lock_is_e_lock_stale(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A registry file or manifest sha that moved on while the lock did not is ``E_LOCK_STALE``, also at lint."""
    root = tmp_path / "main"
    (root / "tables").mkdir(parents=True)
    (root / "tables" / "t.csv").write_text("1\n")
    entry = register_input(tmp_path, {"main": root}, "tables", root="main", relpath="tables")
    data = lock_data([], registered={"tables": entry})
    resolved = lock.resolve_inputs(tmp_path, SourcesLock.model_validate(data), ["tables"])
    assert resolved["tables"].relpath == "tables"
    ref = write_lock(tmp_path, "demo", data)
    argv = ["lint", "--lock", str(tmp_path / ref["lock"]), "--base-dir", str(tmp_path)]
    assert cli.main(argv) == cli.EXIT_OK

    stale_manifest = copy.deepcopy(data)
    stale_manifest["data"][0]["manifest_sha256"] = OTHER
    with pytest.raises(LockError) as info:
        lock.resolve_inputs(tmp_path, SourcesLock.model_validate(stale_manifest), ["tables"])
    assert info.value.code == "E_LOCK_STALE"

    registry = tmp_path / "inputs" / "tables.yaml"
    registry.write_text(registry.read_text().replace("CC0-1.0", "CC-BY-4.0"))
    with pytest.raises(LockError) as info:
        lock.resolve_inputs(tmp_path, SourcesLock.model_validate(data), ["tables"])
    assert info.value.code == "E_LOCK_STALE"
    capsys.readouterr()
    assert cli.main(argv) == cli.EXIT_FAILED
    out = capsys.readouterr().out
    assert "E_LOCK_STALE tables" in out and str(root) not in out
    with pytest.raises(LockError) as info:
        lock.resolve_inputs(tmp_path, SourcesLock.model_validate(data), ["missing"])
    assert info.value.code == "E_LOCK_REF"


@needs_userns
def test_provenance_records_the_lock_sha(world: World) -> None:
    """A run records the lock's sha256 and file name; ``{input:<id>}`` names the registered tree; staleness fails."""
    tables = world.data / "derivatives" / "tables"
    tables.mkdir(parents=True)
    (tables / "t.csv").write_text("3\n")
    world.registered["tables"] = register_input(
        world.base, {"data": world.data}, "tables", root="data", relpath="derivatives/tables"
    )
    verify_input(world.base, world.profile(), "tables")
    step = world.step("prep", "prep", "{input:tables}/t.csv", inputs=[], lock_inputs=["tables"])
    manifest = world.manifest(step)
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.ok, outcome.codes
    assert outcome.out is not None
    assert json.loads((outcome.out / run.METRICS).read_text())["headline"][0]["value"] == 3.0
    text = "".join(p.read_text() for p in (outcome.out / "prov").iterdir())
    assert manifest.sources.sha256 in text and "demo.lock.yaml" in text
    assert str(world.tmp) not in text and "t.csv" not in text

    pins = lock_data(manifest.steps, world.registered)
    pins["code"][0]["submodules"] = {"lib": "1" * 40}
    data = {**manifest.model_dump(mode="json"), "sources": write_lock(world.base, "demo", pins)}
    path = world.base / "m.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    (world.tmp / "deny.txt").write_text("FORBIDDEN-[0-9]+\n")
    profile = world.tmp / "profile.yaml"
    profile.write_text(yaml.safe_dump(world.profile().model_dump(mode="json")))
    common = ["--profile", str(profile), "--base-dir", str(world.base)]
    assert cli.main(["lint", str(path), *common, "--cache", str(world.tmp / "empty-cache")]) == cli.EXIT_OK
    assert cli.main(["lint", str(path), *common, "--cache", str(world.cache)]) == cli.EXIT_FAILED
    world.sources = manifest.sources.model_dump()
    write_lock(world.base, "demo", lock_data(manifest.steps, world.registered))

    registry = world.base / "inputs" / "tables.yaml"
    registry.write_text(yaml.safe_dump({**yaml.safe_load(registry.read_text()), "licence": "CC-BY-4.0"}))
    stale = world.run(manifest)["prep"]
    assert stale.status == "failed" and stale.codes == ("E_LOCK_STALE",)
    assert _summary(outcome)["exit"] == 0
