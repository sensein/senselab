"""Tests for pinned weights: presence checks, sha256 verification at prefetch, the hub view and URL placement.

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

from senselab.utils.reproduce import cli, env, inputs, lock, run, sandbox, slurm, source, weights
from senselab.utils.reproduce.spec import StepSpec
from tests.utils.reproduce.cli_test import _venv_result
from tests.utils.reproduce.inputs_gate_test import _files, _rows, _submit_config, _write_manifest
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
    return [line for line in out.splitlines() if line.startswith(("weights ", "url-weights "))][-1]


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
    assert weights.verify_hf([wrong], world.profile(), {}) == [
        f"E_WEIGHTS_MISMATCH {REPO}@cccccccc weights.bin {_sha(b'other')[:8]}"
    ]
    (top / "blobs" / _sha(WEIGHTS)).write_bytes(b"fake weights 0123456788\n")
    assert weights.verify_hf([step], world.profile(), {}) == [
        f"E_WEIGHTS_MISMATCH {REPO}@cccccccc weights.bin {_sha(WEIGHTS)[:8]}"
    ]
    other = StepSpec.model_validate(world.step("prep", "prep", "x", hf_weights=_weights({"absent.bin": WEIGHTS})))
    assert weights.verify_hf([other], world.profile(), {}) == [f"E_WEIGHTS_MISSING {REPO}@cccccccc absent.bin"]
    make_snapshot(_hub(world), "org/data", REVISION, {"d.csv": b"1\n"}, repo_type="dataset")
    data = _weights({"d.csv": b"1\n"}, repo="org/data", repo_type="dataset")
    assert (
        weights.verify_hf([StepSpec.model_validate(world.step("d", "prep", "x", hf_weights=data))], world.profile(), {})
        == []
    )
    world.profile_extra = {"hf_hub_cache": str(world.restricted_dirs[0] / "hub")}
    with pytest.raises(ValueError, match="E_PROFILE_RESTRICTED"):
        world.profile()


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
    with pytest.raises(weights.WeightsError) as info:
        weights.build_hub_view(view, StepSpec.model_validate(step), world.profile())
    assert info.value.code == "E_WEIGHTS_VIEW"
    make_snapshot(_hub(world), "org/data", REVISION, {"d.csv": b"1\n"}, repo_type="dataset")
    data = world.step("d", "prep", "x", hf_weights=_weights({"d.csv": b"1\n"}, repo="org/data", repo_type="dataset"))
    weights.build_hub_view(tmp_path / "view2", StepSpec.model_validate(data), world.profile())
    found = try_to_load_from_cache("org/data", "d.csv", cache_dir=tmp_path / "view2", repo_type="dataset")
    assert isinstance(found, str) and Path(found).read_bytes() == b"1\n"


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
    make_snapshot(_hub(world), REPO, REVISION, {"copy.bin": WEIGHTS})
    both = [StepSpec.model_validate(_step(world, hf_weights=_weights({"weights.bin": WEIGHTS, "copy.bin": WEIGHTS})))]
    wrong = StepSpec.model_validate(_step(world, "wrong", hf_weights=_weights({"weights.bin": b"other"})))
    expected = [f"E_WEIGHTS_MISMATCH {REPO}@cccccccc weights.bin {_sha(b'other')[:8]}"]
    calls.clear()
    found = weights.verify_hf([*both, wrong], profile, {})
    assert found == expected and len(calls) == 1
    assert _counts(capsys.readouterr().out) == "weights files=2 hashed=1 accepted=0"
    near = "c" * 39 + "d"
    make_snapshot(_hub(world), REPO, near, {"weights.bin": b"other"})
    pair = [
        _spec(world),
        StepSpec.model_validate(_step(world, "near", hf_weights=_weights({"weights.bin": WEIGHTS}, revision=near))),
    ]
    assert weights.verify_hf(pair, profile, {}) == [
        f"E_WEIGHTS_MISMATCH {REPO}@cccccccc weights.bin {_sha(WEIGHTS)[:8]}"
    ]
    assert _counts(capsys.readouterr().out) == "weights files=2 hashed=2 accepted=0"


def _registered_world(tmp_path: Path) -> tuple[World, StepSpec, dict[str, Any]]:
    """A world whose model-cache repository is a registered, verified input with a sha256 column."""
    world = _world(tmp_path)
    make_snapshot(_hub(world), REPO, REVISION, {"weights.bin": WEIGHTS, "config.json": CONFIG})
    world.registered[INPUT] = register_input(
        world.base, {"data": world.data}, INPUT, root="data", relpath="model-cache/hub/models--org--model", sha256=True
    )
    verify_input(world.base, world.profile(), INPUT)
    step = _spec(world, lock_inputs=[INPUT], hf_weights=_weights({"weights.bin": WEIGHTS}, input=INPUT))
    registry = {INPUT: inputs.read_registry_entry(world.base, INPUT)[0]}
    return world, step, registry


def test_a_registered_verified_snapshot_is_accepted_without_hashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A file whose registered input carries its sha256 and a current ``ok`` record is not hashed."""
    world, step, registry = _registered_world(tmp_path)
    reads: list[str] = []
    real = inputs.manifest_rows

    def counted(entry: Any, profile: Any) -> Any:  # noqa: ANN401
        reads.append(entry.id)
        return real(entry, profile)

    monkeypatch.setattr(inputs, "manifest_rows", counted)
    two = _spec(
        world, lock_inputs=[INPUT], hf_weights=_weights({"weights.bin": WEIGHTS, "config.json": CONFIG}, input=INPUT)
    )
    assert weights.verify_hf([two], world.profile(), registry) == []
    assert _counts(capsys.readouterr().out) == "weights files=2 hashed=0 accepted=2" and reads == [INPUT]
    assert weights.verify_hf([step], world.profile(), registry) == []
    assert _counts(capsys.readouterr().out) == "weights files=1 hashed=0 accepted=1"
    make_snapshot(_hub(world), "org/other", REVISION, {"weights.bin": WEIGHTS})
    foreign = _spec(
        world, lock_inputs=[INPUT], hf_weights=_weights({"weights.bin": WEIGHTS}, repo="org/other", input=INPUT)
    )
    assert weights.verify_hf([foreign], world.profile(), registry) == []
    assert _counts(capsys.readouterr().out) == "weights files=1 hashed=1 accepted=0"
    claimed = _spec(world, lock_inputs=[INPUT], hf_weights=_weights({"weights.bin": b"other"}, input=INPUT))
    assert len(weights.verify_hf([claimed], world.profile(), registry)) == 1
    assert _counts(capsys.readouterr().out) == "weights files=1 hashed=1 accepted=0"
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
    verify_input(world.base, world.profile(), INPUT)
    assert weights.verify_hf([step], world.profile(), registry) == []
    assert _counts(capsys.readouterr().out) == "weights files=1 hashed=0 accepted=1"
    os.utime(blob, ns=(st.st_atime_ns, st.st_mtime_ns))
    blob.write_bytes(WEIGHTS + b"x")
    os.utime(blob, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert weights.verify_hf([step], world.profile(), registry) == [
        f"E_WEIGHTS_MISMATCH {REPO}@cccccccc weights.bin {_sha(WEIGHTS)[:8]}"
    ]
    assert _counts(capsys.readouterr().out) == "weights files=1 hashed=1 accepted=0"


# --- A11: URL weights ---------------------------------------------------------------------------------------

URL_FILES = {"whisper/small.pt": b"whisper weights\n", "torch/resnet.pth": b"resnet weights\n", "other/m.ckpt": b"m\n"}
URL_INPUT = "mc-urls"

PLACE_SCRIPT = """\
import json, os, pathlib, sys
out = pathlib.Path(sys.argv[1])
tmp = os.environ["REPRO_TMP"]
paths = {
    "whisper": pathlib.Path(os.environ["XDG_CACHE_HOME"]) / "whisper" / "small.pt",
    "torch_hub": pathlib.Path(os.environ["TORCH_HOME"]) / "hub" / "checkpoints" / "resnet.pth",
    "path": pathlib.Path(tmp) / "weights" / "ckpt" / "m.ckpt",
}
report = {name: path.read_text() for name, path in paths.items()}
try:
    paths["path"].write_text("x")
    report["writable"] = True
except OSError:
    report["writable"] = False
report["torch_in_tmp"] = os.environ["TORCH_HOME"].startswith(tmp)
report["value"] = 1.0
(out / "result.json").write_text(json.dumps(report))
"""


def _url_world(tmp_path: Path) -> World:
    world = _world(tmp_path)
    root = world.data / "model-cache" / "urls"
    world.profile_extra = {**world.profile_extra, "url_weights_root": str(root)}
    for rel, data in URL_FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    return world


def _url_weights(**sha: bytes) -> list[dict[str, str]]:
    places = {"whisper/small.pt": "whisper", "torch/resnet.pth": "torch_hub", "other/m.ckpt": "path:ckpt/m.ckpt"}
    return [{"file": f, "sha256": _sha(sha.get(f.split("/")[0], d)), "place": places[f]} for f, d in URL_FILES.items()]


def _url_step(world: World, name: str = "prep", **change: Any) -> dict[str, Any]:  # noqa: ANN401
    change.setdefault("url_weights", _url_weights())
    return world.step(name, "prep", "{root:data}/in/a.txt", **change)


@needs_userns
def test_url_weights_are_placed_where_each_loader_looks(tmp_path: Path) -> None:
    """Whisper, torch hub and declared-path weights appear read-only where their loaders look, under ``{tmp}``."""
    world = _url_world(tmp_path)
    (world.repo / "place.py").write_text(PLACE_SCRIPT)
    _git(world.repo, "add", "-A")
    _git(world.repo, "commit", "-q", "-m", "place")
    world.commit = _git(world.repo, "rev-parse", "HEAD").strip()
    manifest = world.manifest(_url_step(world, argv=["{venv}/bin/python", "{checkout}/place.py", "{out}"]))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.status == "executed" and outcome.ok, outcome.codes
    assert outcome.out is not None
    report = json.loads((outcome.out / "work" / "result.json").read_text())
    assert report == {
        "whisper": "whisper weights\n",
        "torch_hub": "resnet weights\n",
        "path": "m\n",
        "writable": False,
        "torch_in_tmp": True,
        "value": 1.0,
    }
    keyed = sandbox.keyed_templates(StepSpec.model_validate(_url_step(world)), world.profile())
    assert keyed["TORCH_HOME"] == "{tmp}/torch" and keyed["XDG_CACHE_HOME"] == "{tmp}/cache"
    plain = sandbox.keyed_templates(StepSpec.model_validate(world.step("p", "prep", "x")), world.profile())
    assert plain["TORCH_HOME"] == "{cache}/torch"


@needs_userns
def test_a_url_weight_sha_mismatch_fails_prefetch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A URL weight whose bytes differ from its declared sha256 fails prefetch; a verified input skips the hash."""
    world = _url_world(tmp_path)
    good = StepSpec.model_validate(_url_step(world))
    assert weights.verify_url([good], world.profile(), {}) == []
    assert _counts(capsys.readouterr().out) == "url-weights files=3 hashed=3 accepted=0"
    bad = StepSpec.model_validate(_url_step(world, url_weights=_url_weights(other=b"changed\n")))
    expected = f"E_WEIGHTS_MISMATCH url other/m.ckpt {_sha(b'changed' + bytes([10]))[:8]}"
    assert weights.verify_url([bad], world.profile(), {}) == [expected]
    world.registered[URL_INPUT] = register_input(
        world.base, {"data": world.data}, URL_INPUT, root="data", relpath="model-cache/urls", sha256=True
    )
    verify_input(world.base, world.profile(), URL_INPUT)
    registry = {URL_INPUT: inputs.read_registry_entry(world.base, URL_INPUT)[0]}
    capsys.readouterr()
    registered = StepSpec.model_validate(_url_step(world, lock_inputs=[URL_INPUT]))
    assert weights.verify_url([registered], world.profile(), registry) == []
    assert _counts(capsys.readouterr().out) == "url-weights files=3 hashed=0 accepted=3"
    (world.data / "tables").mkdir()
    (world.data / "tables" / "t.csv").write_text("1\n")
    world.registered["tables"] = register_input(
        world.base, {"data": world.data}, "tables", root="data", relpath="tables"
    )
    verify_input(world.base, world.profile(), "tables")
    registry["tables"] = inputs.read_registry_entry(world.base, "tables")[0]
    reads: list[str] = []
    real_rows = inputs.manifest_rows

    def counted(entry: Any, profile: Any) -> Any:  # noqa: ANN401
        reads.append(entry.id)
        return real_rows(entry, profile)

    monkeypatch.setattr(inputs, "manifest_rows", counted)
    both = StepSpec.model_validate(_url_step(world, lock_inputs=["tables", URL_INPUT]))
    assert weights.verify_url([both], world.profile(), registry) == []
    assert _counts(capsys.readouterr().out) == "url-weights files=3 hashed=0 accepted=3" and reads == [URL_INPUT]
    monkeypatch.setattr(inputs, "manifest_rows", real_rows)
    other = StepSpec.model_validate(_url_step(world, lock_inputs=["tables"]))
    assert weights.verify_url([other], world.profile(), registry) == []
    assert _counts(capsys.readouterr().out) == "url-weights files=3 hashed=3 accepted=0"
    secret = world.restricted_dirs[0] / "w.pt"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_bytes(b"w")
    (world.data / "model-cache" / "urls" / "leak.pt").symlink_to(secret)
    leaking = StepSpec.model_validate(
        _url_step(world, url_weights=[{"file": "leak.pt", "sha256": _sha(b"w"), "place": "torch_hub"}])
    )
    assert weights.check_url_present(leaking, world.profile()) == ["E_WEIGHTS_MISSING"]
    assert weights.verify_url([leaking], world.profile(), {}) == ["E_WEIGHTS_MISSING url leak.pt"]
    capsys.readouterr()
    profile_extra = dict(world.profile_extra)
    world.profile_extra = {}
    assert weights.check_url_present(good, world.profile()) == ["E_WEIGHTS_MISSING"]
    outside = tmp_path / "outside"
    outside.mkdir()
    (world.data / "linked").symlink_to(outside)
    for root, match in (
        (tmp_path / "elsewhere", "data root"),
        (world.data / "linked", "data root"),
        (secret.parent, "E_PROFILE_RESTRICTED"),
    ):
        world.profile_extra = {"url_weights_root": str(root)}
        with pytest.raises(ValueError, match=match):
            world.profile()
    world.profile_extra = profile_extra

    monkeypatch.setattr(
        source, "checkout_step", lambda step, **kw: type("Co", (), {"worktree": world.tmp, "submodules": {}})()
    )
    monkeypatch.setattr(
        env, "prefetch_env", lambda e, b, c, **kw: env.PrefetchResult(_venv_result(c / "v"), {}, None, False, ())
    )
    del world.registered[URL_INPUT]
    manifest = world.manifest(_url_step(world, url_weights=_url_weights(other=b"changed\n")))
    profile, _ = _files(world)
    path = _write_manifest(world, manifest)
    argv = [
        "prefetch",
        str(path),
        "--profile",
        str(profile),
        "--base-dir",
        str(world.base),
        "--cache",
        str(world.cache),
    ]
    assert cli.main(argv) == cli.EXIT_FAILED
    assert expected in capsys.readouterr().err


@needs_userns
def test_a_missing_url_weight_blocks(tmp_path: Path) -> None:
    """A declared URL weight absent from the profile's root blocks the dry run, the run and the submission."""
    world = _url_world(tmp_path)
    manifest = world.manifest(_url_step(world))
    world.prefetch(manifest)
    (world.data / "model-cache" / "urls" / "torch" / "resnet.pth").unlink()
    (plan,) = run.plan_steps([manifest], world.config())
    assert (plan.status, plan.codes) == ("blocked", ("E_WEIGHTS_MISSING",))
    fake = FakeSbatch()
    submission = slurm.submit([manifest], _submit_config(world, manifest), runner=fake)
    assert fake.calls == [] and submission.refused == {"prep": ("E_WEIGHTS_MISSING",)}
    outcome = world.run(manifest)["prep"]
    assert outcome.status == "failed" and outcome.codes == ("E_WEIGHTS_MISSING",)


def test_url_weights_change_the_key(tmp_path: Path) -> None:
    """Adding a URL weight, or changing its sha256 or place, moves the key; the lock must pin each file."""
    from tests.utils.reproduce.cache_test import _key, _step_data

    weight = {"file": "w/model.pt", "sha256": "1" * 64, "place": "torch_hub"}
    keys = {_key(tmp_path)}
    for change in (
        weight,
        {**weight, "sha256": "2" * 64},
        {**weight, "place": "whisper"},
        {**weight, "place": "path:a/b"},
    ):
        data = _step_data()
        data["url_weights"] = [change]
        keys.add(_key(tmp_path, data))
    assert len(keys) == 5
    for place in ("cache", "path:../x", "path:/abs", "path:"):
        with pytest.raises(ValueError):
            StepSpec.model_validate({**_step_data(), "url_weights": [{**weight, "place": place}]})
    clash = [weight, {**weight, "file": "v/model.pt"}]
    with pytest.raises(ValueError, match="url_weights"):
        StepSpec.model_validate({**_step_data(), "url_weights": clash})
    nested = [{**weight, "place": "path:a"}, {**weight, "file": "v/model.pt", "place": "path:a/b"}]
    with pytest.raises(ValueError, match="url_weights"):
        StepSpec.model_validate({**_step_data(), "url_weights": nested})
    step = StepSpec.model_validate({**_step_data(), "url_weights": [weight]})
    unpinned = lock.SourcesLock.model_validate({"schema": lock.LOCK_SCHEMA})
    assert ("E_LOCK_REF", "url_weights w/model.pt") in lock.step_findings(step, unpinned, None)
