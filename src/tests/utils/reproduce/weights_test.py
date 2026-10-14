"""Tests for pinned weights: the hub-cache presence check, sha256 verification at prefetch, and the hub view.

Every hub cache here is a small fake in the Hugging Face layout under ``tmp_path``; no weights are downloaded.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from senselab.utils.reproduce import inputs, run, sandbox, slurm, weights
from senselab.utils.reproduce.spec import StepSpec
from tests.utils.reproduce.inputs_gate_test import _rows, _submit_config
from tests.utils.reproduce.run_test import World, _git
from tests.utils.reproduce.slurm_test import FakeSbatch
from tests.utils.reproduce.sources_fixture import register_input, verify_input

needs_userns = pytest.mark.skipif(not sandbox.userns_available(), reason="unprivileged user namespaces are unavailable")
REPO = "org/model"
REVISION = "c" * 40
WEIGHTS = b"fake weights 0123456789\n"
CONFIG = b'{"model_type": "fake"}\n'
INPUT = "mc-org__model"

LOAD_SCRIPT = """\
import json, os, pathlib, sys
out = pathlib.Path(sys.argv[1])
hub = pathlib.Path(os.environ["HF_HUB_CACHE"])
repo = hub / "models--org--model"
revision = (repo / "refs" / "main").read_text().strip()
data = (repo / "snapshots" / revision / "weights.bin").read_bytes()
try:
    (repo / "blobs" / "new").write_bytes(b"x")
    blobs_writable = True
except OSError:
    blobs_writable = False
home = pathlib.Path(os.environ["HF_HOME"])
(home / "modules").mkdir(parents=True, exist_ok=True)
(home / "modules" / "m.py").write_text("")
report = {
    "value": float(len(data)),
    "blobs_writable": blobs_writable,
    "same_hub": os.environ["HUGGINGFACE_HUB_CACHE"] == str(hub),
    "offline": os.environ["HF_HUB_OFFLINE"],
    "hub_in_tmp": str(hub).startswith(os.environ["REPRO_TMP"]),
}
(out / "result.json").write_text(json.dumps(report))
"""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_snapshot(hub: Path, repo: str, revision: str, files: dict[str, bytes], repo_type: str = "model") -> Path:
    """A repository in Hugging Face layout: content-named blobs, and snapshot files linking to them."""
    org, name = repo.split("/")
    top = hub / f"{repo_type}s--{org}--{name}"
    (top / "blobs").mkdir(parents=True, exist_ok=True)
    for path, data in files.items():
        blob = top / "blobs" / _sha(data)
        blob.write_bytes(data)
        link = top / "snapshots" / revision / path
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
    return top


def _hub(world: World) -> Path:
    return world.data / "model-cache" / "hub"


def _world(tmp_path: Path) -> World:
    world = World(tmp_path)
    world.profile_extra = {"hf_hub_cache": str(_hub(world))}
    return world


def _weights(files: dict[str, bytes], **change: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    declared = [{"path": p, "sha256": _sha(d)} for p, d in files.items()]
    return [{"repo": REPO, "revision": REVISION, "files": declared, **change}]


def _step(world: World, name: str = "prep", **change: Any) -> dict[str, Any]:  # noqa: ANN401
    change.setdefault("hf_weights", _weights({"weights.bin": WEIGHTS}))
    return world.step(name, "prep", "{root:data}/in/a.txt", **change)


def _spec(world: World, **change: Any) -> StepSpec:  # noqa: ANN401
    return StepSpec.model_validate(_step(world, **change))


def _counts(out: str) -> str:
    return [line for line in out.splitlines() if line.startswith("weights ")][-1]


def test_prefetch_verifies_every_declared_file_sha256(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Every declared file is hashed and compared; a changed blob or a wrong declared sha256 is a mismatch."""
    world = _world(tmp_path)
    top = make_snapshot(_hub(world), REPO, REVISION, {"weights.bin": WEIGHTS, "config.json": CONFIG})
    step = StepSpec.model_validate(
        world.step("prep", "prep", "x", hf_weights=_weights({"weights.bin": WEIGHTS, "config.json": CONFIG}))
    )
    assert weights.verify_hf([step], world.profile(), {}) == []
    assert _counts(capsys.readouterr().out) == "weights files=2 hashed=2 accepted=0"
    wrong = StepSpec.model_validate(world.step("prep", "prep", "x", hf_weights=_weights({"weights.bin": b"other"})))
    assert weights.verify_hf([wrong], world.profile(), {}) == [f"E_WEIGHTS_MISMATCH {REPO}@cccccccc weights.bin"]
    (top / "blobs" / _sha(WEIGHTS)).write_bytes(b"fake weights 0123456788\n")
    assert weights.verify_hf([step], world.profile(), {}) == [f"E_WEIGHTS_MISMATCH {REPO}@cccccccc weights.bin"]
    other = StepSpec.model_validate(world.step("prep", "prep", "x", hf_weights=_weights({"absent.bin": WEIGHTS})))
    assert weights.verify_hf([other], world.profile(), {}) == [f"E_WEIGHTS_MISSING {REPO}@cccccccc absent.bin"]


@needs_userns
def test_a_missing_weight_blocks_the_dry_run_and_submit(tmp_path: Path) -> None:
    """SC5 negative: a declared file absent from the hub cache blocks the dry run and refuses the submission."""
    world = _world(tmp_path)
    manifest = world.manifest(_step(world))
    world.prefetch(manifest)
    (plan,) = run.plan_steps([manifest], world.config())
    assert (plan.status, plan.codes) == ("blocked", ("E_WEIGHTS_MISSING",))
    fake = FakeSbatch()
    submission = slurm.submit([manifest], _submit_config(world, manifest), runner=fake)
    assert fake.calls == [] and submission.jobs == [] and submission.refused == {"prep": ("E_WEIGHTS_MISSING",)}
    assert _rows(world)["prep"]["codes"] == ["E_WEIGHTS_MISSING"]
    outcome = world.run(manifest)["prep"]
    assert outcome.status == "failed" and outcome.codes == ("E_WEIGHTS_MISSING",)
    make_snapshot(_hub(world), REPO, REVISION, {"weights.bin": WEIGHTS})
    (plan,) = run.plan_steps([manifest], world.config())
    assert plan.status == "will-execute", plan.codes


@needs_userns
def test_the_step_loads_through_the_view_offline(tmp_path: Path) -> None:
    """The step reads its weights through ``{tmp}/hf-hub`` (``refs/main`` → the revision); blobs are read-only."""
    world = _world(tmp_path)
    (world.repo / "load.py").write_text(LOAD_SCRIPT)
    _git(world.repo, "add", "-A")
    _git(world.repo, "commit", "-q", "-m", "load")
    world.commit = _git(world.repo, "rev-parse", "HEAD").strip()
    make_snapshot(_hub(world), REPO, REVISION, {"weights.bin": WEIGHTS, "config.json": CONFIG})
    step = _step(world, argv=["{venv}/bin/python", "{checkout}/load.py", "{out}"])
    step["hf_weights"] = _weights({"weights.bin": WEIGHTS, "config.json": CONFIG})
    manifest = world.manifest(step)
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.status == "executed" and outcome.ok, outcome.codes
    assert outcome.out is not None
    report = json.loads((outcome.out / "work" / "result.json").read_text())
    assert report == {
        "value": float(len(WEIGHTS)),
        "blobs_writable": False,
        "same_hub": True,
        "offline": "1",
        "hub_in_tmp": True,
    }
    assert not (_hub(world) / "models--org--model" / "refs").exists()
    keyed = sandbox.keyed_templates(StepSpec.model_validate(step), world.profile())
    assert (keyed["HF_HOME"], keyed["HF_HUB_CACHE"], keyed["HUGGINGFACE_HUB_CACHE"]) == (
        "{tmp}/hf",
        "{tmp}/hf-hub",
        "{tmp}/hf-hub",
    )
    plain = sandbox.keyed_templates(StepSpec.model_validate(world.step("p", "prep", "x")), world.profile())
    assert plain["HF_HOME"] == "{cache}/huggingface" and "HF_HUB_CACHE" not in plain

    from huggingface_hub import try_to_load_from_cache

    view = tmp_path / "view"
    weights.build_hub_view(view, StepSpec.model_validate(step), world.profile())
    found = try_to_load_from_cache(REPO, "weights.bin", cache_dir=view)
    assert isinstance(found, str) and Path(found).read_bytes() == WEIGHTS


def test_two_steps_with_the_same_weights_hash_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One prefetch over two steps declaring the same file hashes its blob once."""
    world = _world(tmp_path)
    make_snapshot(_hub(world), REPO, REVISION, {"weights.bin": WEIGHTS})
    calls: list[int] = []
    real = hashlib.sha256

    def counting(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        calls.append(1)
        return real(*args, **kwargs)

    steps = [_spec(world), StepSpec.model_validate(_step(world, "again"))]
    profile = world.profile()
    monkeypatch.setattr(hashlib, "sha256", counting)
    assert weights.verify_hf(steps, profile, {}) == []
    assert len(calls) == 1
    assert _counts(capsys.readouterr().out) == "weights files=1 hashed=1 accepted=0"


def _registered_world(tmp_path: Path) -> tuple[World, StepSpec, dict[str, Any]]:
    """A world whose model-cache repository is a registered, verified input with a sha256 column."""
    world = _world(tmp_path)
    make_snapshot(_hub(world), REPO, REVISION, {"weights.bin": WEIGHTS})
    world.registered[INPUT] = register_input(
        world.base, {"data": world.data}, INPUT, root="data", relpath="model-cache/hub/models--org--model", sha256=True
    )
    verify_input(world.base, world.profile(), INPUT)
    step = _spec(world, lock_inputs=[INPUT], hf_weights=_weights({"weights.bin": WEIGHTS}, input=INPUT))
    registry = {INPUT: inputs.read_registry_entry(world.base, INPUT)[0]}
    return world, step, registry


def test_a_registered_verified_snapshot_is_accepted_without_hashing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file whose registered input carries its sha256 and a current ``ok`` record is not hashed."""
    world, step, registry = _registered_world(tmp_path)
    assert weights.verify_hf([step], world.profile(), registry) == []
    assert _counts(capsys.readouterr().out) == "weights files=1 hashed=0 accepted=1"
    stale = datetime.datetime.now(datetime.UTC) + datetime.timedelta(days=40)
    assert weights.verify_hf([step], world.profile(), registry, now=stale) == []
    assert _counts(capsys.readouterr().out) == "weights files=1 hashed=1 accepted=0"
    undeclared = StepSpec.model_validate(_step(world, lock_inputs=[INPUT]))
    assert weights.verify_hf([undeclared], world.profile(), registry) == []
    assert _counts(capsys.readouterr().out) == "weights files=1 hashed=1 accepted=0"
    with pytest.raises(ValueError, match="E_SPEC_INPUT_UNDECLARED"):
        _spec(world, hf_weights=_weights({"weights.bin": WEIGHTS}, input=INPUT))


def test_a_blob_whose_size_or_mtime_drifted_is_rehashed(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A blob whose ``mtime_ns`` no longer matches its manifest row is hashed again, and still checks."""
    world, step, registry = _registered_world(tmp_path)
    blob = _hub(world) / "models--org--model" / "blobs" / _sha(WEIGHTS)
    st = os.stat(blob)
    os.utime(blob, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    assert weights.verify_hf([step], world.profile(), registry) == []
    assert _counts(capsys.readouterr().out) == "weights files=1 hashed=1 accepted=0"
    os.utime(blob, ns=(st.st_atime_ns, st.st_mtime_ns))
    blob.write_bytes(WEIGHTS + b"x")
    os.utime(blob, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert weights.verify_hf([step], world.profile(), registry) == [f"E_WEIGHTS_MISMATCH {REPO}@cccccccc weights.bin"]
    assert _counts(capsys.readouterr().out) == "weights files=1 hashed=1 accepted=0"
