"""Tests for step keys, result dirs, the completion marker, run claims and run links."""

from __future__ import annotations

import datetime
import json
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from senselab.utils.reproduce import cache
from senselab.utils.reproduce.cache import (
    CacheError,
    claim,
    driver_env_sha256,
    is_done,
    link_run,
    native_digests,
    reset_result_dir,
    result_dir,
    run_dir,
    step_key,
    step_key_payload,
    write_summary,
)
from senselab.utils.reproduce.digest import DigestError
from senselab.utils.reproduce.sandbox import build_env
from senselab.utils.reproduce.spec import Profile, StepSpec

COMMIT = "0123456789abcdef0123456789abcdef01234567"
SUB = "fedcba9876543210fedcba9876543210fedcba98"
REV = "1111111111111111111111111111111111111111"
SHA = "ab" * 32
OTHER = "cd" * 32
UP = "ee" * 32


def _step_data(**over: Any) -> dict[str, Any]:  # noqa: ANN401
    data: dict[str, Any] = {
        "name": "train",
        "repo": "https://git.example/org/analysis.git",
        "replicate": 0,
        "commit": COMMIT,
        "kind": "python",
        "env": {
            "python": "3.12",
            "python_exact": "3.12.11",
            "torch": {"lock": "envs/a/torch.lock", "lock_sha256": SHA, "index_url": "https://idx.example/cu128"},
            "body": {"lock": "envs/a/body.lock", "lock_sha256": SHA, "source": "uv-export", "export_cmd": ["uv"]},
            "sdist_allow": [
                {
                    "name": "psf",
                    "url": "https://files.example/psf-0.6.tar.gz",
                    "sha256": SHA,
                    "build_env": {"lock": "envs/a/build.lock", "lock_sha256": SHA},
                }
            ],
            "wheels": {"lock": "envs/a/wheels.lock", "lock_sha256": SHA},
            "test": {"lock": "envs/a/test.lock", "lock_sha256": SHA},
            "exclude_newer": "2026-01-01",
        },
        "argv": ["{venv}/bin/python", "run.py", "--data", "{root:data}/x", "--out", "{out}", "{up:prep}/y"],
        "cwd": "src",
        "pythonpath": ["{checkout}/src"],
        "inputs": [{"root": "data", "patterns": ["x/*.csv"]}, {"root": "up:prep", "patterns": ["y/*.json"]}],
        "needs": ["prep"],
        "resources": {"profile": "cpu"},
        "patches": [{"file": "patches/train/0001.diff", "sha256": SHA}],
        "adapter": "adapters/train.py",
        "hf_weights": [],
        "metrics_schema": {"kind": "statistic", "headline": ["f_stat"]},
        "expect": [{"metric": "f_stat", "value": 3.5, "tolerance": 0.1, "kind": "abs"}],
    }
    data.update(over)
    return data


def _r_step_data(**over: Any) -> dict[str, Any]:  # noqa: ANN401
    data = _step_data(kind="r")
    data["env"] = {
        **data["env"],
        "r_packages": [{"name": "lme4", "version": "1.1-35"}],
        "r_snapshot": "2026-01-01",
        "r_lib": "{cache}/r/train",
    }
    data.update(over)
    return data


def _profile(tmp_path: Path, native: dict[str, Any] | None = None, **env: str) -> Profile:
    return Profile.model_validate(
        {
            "native": native or {},
            "roots": {"data": str(tmp_path / "data")},
            "results_root": str(tmp_path / "reruns" / "results"),
            "runs_root": str(tmp_path / "reruns" / "runs"),
            "slurm_logs_root": str(tmp_path / "reruns" / "slurm"),
            "env": {"DATA_VAR": "d", **env},
            "partitions": {"cpu": {"partition": "p", "cpus": 1, "mem": "1G", "time": "0:10:00"}},
            "build": "cpu",
            "deny_patterns_file": str(tmp_path / "deny.txt"),
            "canary_host": "pypi.org",
        }
    )


def _kwargs(**over: Any) -> dict[str, Any]:  # noqa: ANN401
    kwargs: dict[str, Any] = {
        "upstream": {"prep": UP},
        "submodules": {"third_party/lib": SUB},
        "adapter_sha256": SHA,
        "adapter_env_sha256": SHA,
        "input_digests": [SHA, SHA],
        "native": {},
    }
    kwargs.update(over)
    return kwargs


def _key(tmp_path: Path, data: dict[str, Any] | None = None, profile: Profile | None = None, **over: Any) -> str:  # noqa: ANN401
    step = StepSpec.model_validate(_step_data() if data is None else data)
    return step_key(step, _profile(tmp_path) if profile is None else profile, **_kwargs(**over))


def _env_change(path: tuple[str, ...], value: object) -> Callable[[dict[str, Any]], None]:
    def change(data: dict[str, Any]) -> None:
        target = data
        for part in path[:-1]:
            target = target[part]
        target[path[-1]] = value

    return change


# --- step key --------------------------------------------------------------------------------------

SPEC_CHANGES: dict[str, Callable[[dict[str, Any]], None]] = {
    "torch-lock-sha": _env_change(("env", "torch", "lock_sha256"), OTHER),
    "body-lock-sha": _env_change(("env", "body", "lock_sha256"), OTHER),
    "wheels-lock-sha": _env_change(("env", "wheels", "lock_sha256"), OTHER),
    "overlay-lock-sha": _env_change(("env", "test", "lock_sha256"), OTHER),
    "sdist-sha": lambda d: d["env"]["sdist_allow"][0].update(sha256=OTHER),
    "python-exact": _env_change(("env", "python_exact"), "3.12.12"),
    "patch-sha": lambda d: d["patches"][0].update(sha256=OTHER),
    "patch-added": lambda d: d["patches"].append({"file": "patches/train/0002.diff", "sha256": SHA}),
    "argv-template": lambda d: d["argv"].append("--fast"),
    "cwd": lambda d: d.update(cwd="."),
    "pythonpath": lambda d: d.update(pythonpath=["{checkout}"]),
    "commit": lambda d: d.update(commit="f" * 40),
    "replicate": lambda d: d.update(replicate=1),
    "hf-weights": lambda d: d.update(
        hf_weights=[{"repo": "org/model", "revision": REV, "files": [{"path": "model.safetensors", "sha256": SHA}]}]
    ),
    "expect": lambda d: d["expect"][0].update(value=3.6),
    "input-patterns": lambda d: d["inputs"][0].update(patterns=["x/*.tsv"]),
    "stage": lambda d: d.update(stage=[{"root": "checkout", "patterns": ["c/*.yaml"], "dest": "c"}]),
    "outputs": lambda d: d.update(outputs=["r/*.json"]),
    "metrics-schema": lambda d: d.update(metrics_schema={"kind": "statistic", "headline": ["f_stat"], "counts": ["n"]}),
    "gpu-arch": lambda d: d["resources"].update(gpu_arch=["sm_80"]),
    "notebook-overlay": lambda d: d["env"].update(notebook_tools={"lock": "n.lock", "lock_sha256": SHA}),
}


@pytest.mark.parametrize("change", list(SPEC_CHANGES), ids=list(SPEC_CHANGES))
def test_key_changes_with_each_manifest_input(tmp_path: Path, change: str) -> None:
    """Every keyed manifest field moves the key."""
    data = _step_data()
    SPEC_CHANGES[change](data)
    assert _key(tmp_path, data) != _key(tmp_path)


ARG_CHANGES: dict[str, dict[str, Any]] = {
    "upstream-key": {"upstream": {"prep": OTHER}},
    "adapter-sha": {"adapter_sha256": OTHER},
    "adapter-env-sha": {"adapter_env_sha256": OTHER},
    "input-digest": {"input_digests": [SHA, OTHER]},
    "submodule-sha": {"submodules": {"third_party/lib": "e" * 40}},
}


@pytest.mark.parametrize("change", list(ARG_CHANGES), ids=list(ARG_CHANGES))
def test_key_changes_with_each_run_input(tmp_path: Path, change: str) -> None:
    """The upstream keys, adapter, input-set and submodule digests move the key."""
    assert _key(tmp_path, **ARG_CHANGES[change]) != _key(tmp_path)


def test_key_changes_with_keyed_env(tmp_path: Path) -> None:
    """A profile env value is keyed, so changing it moves the key."""
    assert _key(tmp_path, profile=_profile(tmp_path, EXTRA="1")) != _key(tmp_path)
    assert _key(tmp_path, profile=_profile(tmp_path, DATA_VAR="e")) != _key(tmp_path)


def test_key_changes_with_r_session_and_module_env(tmp_path: Path) -> None:
    """An R step's ``SESSION.json`` and module-env sha256s are keyed."""
    base = _key(tmp_path, _r_step_data(), r_session_sha256=SHA, r_module_env_sha256=SHA)
    assert _key(tmp_path, _r_step_data(), r_session_sha256=OTHER, r_module_env_sha256=SHA) != base
    assert _key(tmp_path, _r_step_data(), r_session_sha256=SHA, r_module_env_sha256=OTHER) != base


def test_key_ignores_volatile_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Job ids, visible GPUs, the clock, data-root and result locations, partition and step name are not keyed."""
    base = _key(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "7")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("TMPDIR", "/elsewhere")
    monkeypatch.setattr(time, "time", lambda: 4102444800.0)
    assert _key(tmp_path) == base
    moved = tmp_path / "moved"
    assert _key(tmp_path, profile=_profile(moved)) == base
    assert _key(tmp_path, _step_data(name="train_again")) == base
    assert _key(tmp_path, _step_data(repo="https://mirror.example/analysis.git")) == base
    profile = _profile(tmp_path).model_copy(
        update={"partitions": {"cpu": {"partition": "q", "cpus": 4, "mem": "8G", "time": "1:00:00"}}}
    )
    assert _key(tmp_path, profile=profile) == base
    assert _key(tmp_path, _step_data(resources={"profile": "cpu", "min_driver": "550"})) == base


def test_key_ignores_order_of_set_like_lists(tmp_path: Path) -> None:
    """Pattern, GPU arch, HF repository and HF file order do not move the key; -0.0 keys as 0.0."""
    files = [{"path": "a.bin", "sha256": SHA}, {"path": "b.bin", "sha256": OTHER}]
    one = {"repo": "org/one", "revision": REV, "files": files}
    two = {"repo": "org/two", "revision": REV, "files": files[::-1]}
    a = _step_data(hf_weights=[one, two], resources={"profile": "cpu", "gpu_arch": ["sm_80", "sm_90"]})
    b = _step_data(
        hf_weights=[{**two, "files": files}, {**one, "files": files[::-1]}],
        resources={"profile": "cpu", "gpu_arch": ["sm_90", "sm_80"]},
    )
    a["inputs"][0]["patterns"] = ["x/*.csv", "x/*.tsv"]
    b["inputs"][0]["patterns"] = ["x/*.tsv", "x/*.csv"]
    assert _key(tmp_path, a) == _key(tmp_path, b)
    zero = _step_data(expect=[{"metric": "f_stat", "value": 0.0, "tolerance": 0.0, "kind": "abs"}])
    negative = _step_data(expect=[{"metric": "f_stat", "value": -0.0, "tolerance": -0.0, "kind": "abs"}])
    assert _key(tmp_path, zero) == _key(tmp_path, negative)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_expect_numbers_must_be_finite(bad: float) -> None:
    """NaN and infinite targets or tolerances are refused, so they never reach a key."""
    for expect in (
        {"metric": "f_stat", "value": bad, "tolerance": 0.1, "kind": "abs"},
        {"metric": "f_stat", "value": 1.0, "tolerance": bad, "kind": "abs"},
    ):
        with pytest.raises(ValueError):
            StepSpec.model_validate(_step_data(expect=[expect]))


def test_staged_data_root_content_is_keyed(tmp_path: Path) -> None:
    """A data-root stage repeats an input entry, so a change to the staged files moves the key."""
    stage = [{"root": "data", "patterns": ["x/*.csv"], "dest": "x"}]
    data = _step_data(stage=stage)
    assert _key(tmp_path, data, input_digests=[OTHER, SHA]) != _key(tmp_path, data)


def test_data_root_stage_without_input_is_refused() -> None:
    """A data-root stage with no inputs entry of the same root and patterns is refused with a fixed code."""
    for stage in (
        {"root": "data", "patterns": ["z/*.csv"], "dest": "z"},
        {"root": "other", "patterns": ["x/*.csv"], "dest": "x"},
        {"root": "data", "patterns": ["x/*.csv", "x/*.tsv"], "dest": "x"},
    ):
        with pytest.raises(ValueError, match="E_SPEC_STAGE_UNDIGESTED"):
            StepSpec.model_validate(_step_data(stage=[stage]))
    reordered = _step_data(stage=[{"root": "data", "patterns": ["x/*.csv"], "dest": "x"}])
    reordered["inputs"][0]["patterns"] = ["x/*.tsv", "x/*.csv"]
    reordered["stage"][0]["patterns"] = ["x/*.csv", "x/*.tsv"]
    StepSpec.model_validate(reordered)
    for root in ("checkout", "up:prep"):
        StepSpec.model_validate(_step_data(stage=[{"root": root, "patterns": ["q/*"], "dest": "q"}]))


def _native_profile(tmp_path: Path) -> tuple[Profile, Path]:
    lib = tmp_path / "ff" / "lib"
    lib.mkdir(parents=True)
    (lib / "libav.so.60").write_bytes(b"one")
    bin_dir = tmp_path / "ff" / "bin"
    bin_dir.mkdir()
    (bin_dir / "ffmpeg").write_bytes(b"exe")
    native = {"ffmpeg": {"bin_dirs": [str(bin_dir)], "lib_dirs": [str(lib)]}}
    return _profile(tmp_path, native=native), lib


def test_native_dir_content_is_keyed(tmp_path: Path) -> None:
    """Each selected native dir's content digest is keyed; changing a library moves the key."""
    profile, lib = _native_profile(tmp_path)
    step = StepSpec.model_validate(_step_data(resources={"profile": "cpu", "native": ["ffmpeg"]}))
    first = native_digests(step, profile)
    assert list(first) == ["ffmpeg"] and len(first["ffmpeg"]) == 2
    base = step_key(step, profile, **_kwargs(native=first))
    (lib / "libav.so.60").write_bytes(b"two")
    second = native_digests(step, profile)
    assert second["ffmpeg"][0] == first["ffmpeg"][0] and second["ffmpeg"][1] != first["ffmpeg"][1]
    assert step_key(step, profile, **_kwargs(native=second)) != base
    for bad in ({}, {"ffmpeg": first["ffmpeg"][:1]}, {"ffmpeg": [SHA, "x"]}, {"ffmpeg": first["ffmpeg"], "q": []}):
        with pytest.raises(CacheError) as info:
            step_key(step, profile, **_kwargs(native=bad))
        assert info.value.code == "E_KEY_NATIVE"
    (lib / "libav.so.60").unlink()
    with pytest.raises(DigestError) as digest_info:
        native_digests(step, profile)
    assert digest_info.value.code == "E_DIGEST_EMPTY"


def test_driver_semantics_and_driver_env_identity(tmp_path: Path) -> None:
    """The payload carries the driver-semantics number; the driver env identity is stable and hex."""
    step = StepSpec.model_validate(_step_data())
    payload = step_key_payload(step, _profile(tmp_path), **_kwargs())
    assert payload["driver_semantics"] == cache.DRIVER_SEMANTICS
    identity = driver_env_sha256()
    assert identity == driver_env_sha256() and len(identity) == 64
    assert _key(tmp_path, adapter_env_sha256=identity) != _key(tmp_path)


def test_keyed_env_is_the_sandbox_keyed_set(tmp_path: Path) -> None:
    """The payload's env is what `build_env` keys: two different job envs give the same keyed sha and key."""
    step = StepSpec.model_validate(_step_data())
    profile = _profile(tmp_path)
    paths = {n: tmp_path / n for n in ("out", "tmp", "checkout", "venv", "cache")} | {"up:prep": tmp_path / "u"}
    _, keyed_a, recorded_a = build_env(step, profile, paths, parent={"SLURM_JOB_ID": "1"})
    _, keyed_b, recorded_b = build_env(step, profile, paths, parent={"SLURM_JOB_ID": "2"})
    assert keyed_a == keyed_b and recorded_a != recorded_b
    payload = step_key_payload(step, profile, **_kwargs())
    assert not any(k.startswith("SLURM_") or k in ("CUDA_VISIBLE_DEVICES", "TMPDIR", "HOME") for k in payload["env"])


def test_key_is_deterministic_and_documented_serialization(tmp_path: Path) -> None:
    """The key is sha256 over sorted-key compact JSON of the payload, with a schema tag."""
    step = StepSpec.model_validate(_step_data())
    payload = step_key_payload(step, _profile(tmp_path), **_kwargs())
    assert payload["schema"] == cache.STEP_KEY_SCHEMA
    import hashlib

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    assert _key(tmp_path) == hashlib.sha256(text.encode()).hexdigest()
    # Dict order of the run inputs does not matter.
    two = {"a/x": SUB, "b/y": "e" * 40}
    assert _key(tmp_path, submodules=dict(reversed(two.items()))) == _key(tmp_path, submodules=two)


REFUSALS: dict[str, tuple[dict[str, Any] | None, dict[str, Any], str]] = {
    "upstream-missing": (None, {"upstream": {}}, "E_KEY_UPSTREAM"),
    "upstream-extra": (None, {"upstream": {"prep": UP, "other": UP}}, "E_KEY_UPSTREAM"),
    "upstream-not-hex": (None, {"upstream": {"prep": "x"}}, "E_KEY_UPSTREAM"),
    "inputs-count": (None, {"input_digests": [SHA]}, "E_KEY_INPUTS"),
    "inputs-not-hex": (None, {"input_digests": [SHA, "zz"]}, "E_KEY_INPUTS"),
    "adapter-not-hex": (None, {"adapter_sha256": "abc"}, "E_KEY_ADAPTER"),
    "adapter-env-not-hex": (None, {"adapter_env_sha256": "abc"}, "E_KEY_ADAPTER"),
    "submodule-not-hex": (None, {"submodules": {"lib": "abc"}}, "E_KEY_SUBMODULES"),
    "submodules-when-disabled": (_step_data(submodules=False), {}, "E_KEY_SUBMODULES"),
    "r-session-on-python": (None, {"r_session_sha256": SHA, "r_module_env_sha256": SHA}, "E_KEY_R_SESSION"),
    "r-session-missing": (_r_step_data(), {}, "E_KEY_R_SESSION"),
    "r-module-env-missing": (_r_step_data(), {"r_session_sha256": SHA}, "E_KEY_R_SESSION"),
}


@pytest.mark.parametrize("case", list(REFUSALS), ids=list(REFUSALS))
def test_key_refuses_inconsistent_inputs(tmp_path: Path, case: str) -> None:
    """Run inputs that do not fit the step are refused with a fixed code."""
    data, over, code = REFUSALS[case]
    with pytest.raises(CacheError) as info:
        _key(tmp_path, data, **over)
    assert info.value.code == code
    assert str(tmp_path) not in str(info.value)


def test_key_refuses_unrecorded_wheels(tmp_path: Path) -> None:
    """An env with sdist builds but no recorded ``wheels.lock`` cannot be keyed yet."""
    data = _step_data()
    data["env"]["wheels"] = None
    with pytest.raises(CacheError) as info:
        _key(tmp_path, data)
    assert info.value.code == "E_KEY_NO_WHEELS"


# --- hf_weights field ------------------------------------------------------------------------------


def test_hf_weights_is_required_and_commit_pinned() -> None:
    """``hf_weights`` must be written out; each entry pins a 40-hex revision and hashes every file."""
    data = _step_data()
    del data["hf_weights"]
    with pytest.raises(ValueError, match="hf_weights"):
        StepSpec.model_validate(data)
    good = {"repo": "org/model", "revision": REV, "files": [{"path": "model.safetensors", "sha256": SHA}]}
    assert StepSpec.model_validate(_step_data(hf_weights=[good])).hf_weights[0].revision == REV
    for bad in (
        {**good, "revision": "main"},
        {**good, "revision": "v1.0"},
        {**good, "repo": "model"},
        {**good, "files": []},
        {**good, "files": [{"path": "../x", "sha256": SHA}]},
        {**good, "files": [{"path": "a", "sha256": SHA}, {"path": "a", "sha256": SHA}]},
        {**good, "extra": 1},
    ):
        with pytest.raises(ValueError):
            StepSpec.model_validate(_step_data(hf_weights=[bad]))
    with pytest.raises(ValueError, match="duplicate"):
        StepSpec.model_validate(_step_data(hf_weights=[good, good]))


# --- result dirs and the completion marker ---------------------------------------------------------


def test_result_dir_is_content_addressed(tmp_path: Path) -> None:
    """The result dir is ``<results_root>/<step_key>``; a key that is not 64-hex is refused."""
    assert result_dir(tmp_path, SHA) == tmp_path / SHA
    for bad in ("", "../x", "AB" * 32, SHA + "0"):
        with pytest.raises(CacheError) as info:
            result_dir(tmp_path, bad)
        assert info.value.code == "E_KEY_BAD"


def test_is_done_only_for_a_complete_exit_zero_summary(tmp_path: Path) -> None:
    """Absent dir, no summary, a failed exit, a bool exit or a torn file are not done."""
    out = result_dir(tmp_path, SHA)
    assert not is_done(out)
    out.mkdir()
    assert not is_done(out)
    (out / "metrics.json").write_text("{}")
    assert not is_done(out)
    (out / cache.SUMMARY).write_text('{"exit": 0')
    assert not is_done(out)
    (out / cache.SUMMARY).write_text('{"exit": true}')
    assert not is_done(out)
    (out / cache.SUMMARY).write_text("[0]")
    assert not is_done(out)
    write_summary(out, {"exit": 1, "codes": ["E_EXPECT"]})
    assert not is_done(out)
    write_summary(out, {"exit": 0, "codes": []})
    assert is_done(out)
    assert json.loads((out / cache.SUMMARY).read_text()) == {"exit": 0, "codes": []}
    assert sorted(p.name for p in out.iterdir()) == ["metrics.json", cache.SUMMARY]


def test_write_summary_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash before the rename leaves no summary and no temporary file; the old summary survives."""
    out = tmp_path / SHA
    out.mkdir()

    def boom(*_: object) -> None:
        raise OSError("disk gone")

    monkeypatch.setattr(cache.os, "replace", boom)
    with pytest.raises(OSError):
        write_summary(out, {"exit": 0})
    assert list(out.iterdir()) == []
    assert not is_done(out)
    monkeypatch.undo()
    write_summary(out, {"exit": 1})
    monkeypatch.setattr(cache.os, "replace", boom)
    with pytest.raises(OSError):
        write_summary(out, {"exit": 0})
    assert [p.name for p in out.iterdir()] == [cache.SUMMARY]
    assert json.loads((out / cache.SUMMARY).read_text()) == {"exit": 1}


def test_write_summary_refuses_non_json(tmp_path: Path) -> None:
    """NaN and non-JSON values are refused before anything is written."""
    out = tmp_path / SHA
    out.mkdir()
    with pytest.raises(ValueError):
        write_summary(out, {"exit": 0, "x": float("nan")})
    assert list(out.iterdir()) == []


def test_reset_result_dir_clears_an_incomplete_run_and_keeps_a_done_one(tmp_path: Path) -> None:
    """An incomplete or failed result dir is emptied; a done one is refused."""
    out = result_dir(tmp_path, SHA)
    assert reset_result_dir(tmp_path, SHA) == out and list(out.iterdir()) == []
    (out / "logs").mkdir()
    (out / "logs" / "x.log").write_text("x")
    write_summary(out, {"exit": 2})
    reset_result_dir(tmp_path, SHA)
    assert out.is_dir() and list(out.iterdir()) == []
    write_summary(out, {"exit": 0})
    with pytest.raises(CacheError) as info:
        reset_result_dir(tmp_path, SHA)
    assert info.value.code == "E_RESULT_DONE"
    assert is_done(out)


def test_reset_result_dir_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad key, a symlink or a file at the path, and a failed removal give fixed codes and no path."""
    with pytest.raises(CacheError) as info:
        reset_result_dir(tmp_path, "../x")
    assert info.value.code == "E_KEY_BAD"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / SHA).symlink_to(elsewhere)
    (tmp_path / OTHER).write_text("x")
    for key in (SHA, OTHER):
        with pytest.raises(CacheError) as info:
            reset_result_dir(tmp_path, key)
        assert info.value.code == "E_RESULT_NOT_DIR"
    assert elsewhere.is_dir() and (tmp_path / OTHER).is_file()
    (tmp_path / UP).mkdir()

    def boom(*_: object, **__: object) -> None:
        raise OSError(13, "denied", str(tmp_path))

    monkeypatch.setattr(cache.shutil, "rmtree", boom)
    with pytest.raises(CacheError) as info:
        reset_result_dir(tmp_path, UP)
    assert info.value.code == "E_RESULT_RESET"
    assert str(tmp_path) not in str(info.value) and info.value.__suppress_context__


# --- claims ----------------------------------------------------------------------------------------

_HOLDER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from senselab.utils.reproduce.cache import claim, result_dir, write_summary
    root, key, ready, release = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
    with claim(root, key):
        ready.write_text("held")
        deadline = time.monotonic() + 60
        while not release.exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        out = result_dir(root, key)
        out.mkdir(parents=True, exist_ok=True)
        write_summary(out, {"exit": 0})
    """
)


def _wait_for(path: Path, proc: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 60
    while not path.exists():
        assert proc.poll() is None, proc.communicate()
        assert time.monotonic() < deadline
        time.sleep(0.05)


def test_claim_excludes_a_second_process(tmp_path: Path) -> None:
    """While one process holds a step's claim another cannot take it; after release it sees the step done."""
    root = tmp_path / "results"
    ready, release = tmp_path / "ready", tmp_path / "release"
    proc = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(root), SHA, str(ready), str(release)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        _wait_for(ready, proc)
        with pytest.raises(CacheError) as info:
            with claim(root, SHA, timeout=0.5):
                pass
        assert info.value.code == "E_CLAIM_HELD"
        assert str(tmp_path) not in str(info.value)
        assert info.value.__cause__ is None and info.value.__suppress_context__
        with claim(root, OTHER, timeout=0.5):
            pass
        release.write_text("go")
        with claim(root, SHA, timeout=60) as out:
            assert out == result_dir(root, SHA)
            assert is_done(out)
    finally:
        release.touch()
        _, err = proc.communicate(timeout=60)
    assert proc.returncode == 0, err


def test_claim_files_live_outside_result_dirs(tmp_path: Path) -> None:
    """The claim's lock files sit under ``<results_root>/.claims``, never in a result dir or beside it."""
    root = tmp_path / "results"
    root.mkdir(mode=0o700)
    with claim(root, SHA):
        assert sorted(p.name for p in root.iterdir()) == [cache.CLAIMS]
    assert not (root / SHA).exists()
    assert root.stat().st_mode & 0o7777 == 0o700


# --- run dirs and links ----------------------------------------------------------------------------


def test_run_dir_shape(tmp_path: Path) -> None:
    """``<runs_root>/<YYYY-MM-DD>/<analysis>``; a path-like analysis name is refused."""
    assert run_dir(tmp_path, datetime.date(2026, 10, 7), "rerun") == tmp_path / "2026-10-07" / "rerun"
    for bad in ("", "..", "a/b", ".hidden"):
        with pytest.raises(CacheError) as info:
            run_dir(tmp_path, datetime.date(2026, 10, 7), bad)
        assert info.value.code == "E_RUN_BAD_NAME"


def test_link_run_points_the_step_at_its_result(tmp_path: Path) -> None:
    """A relative symlink ``<run>/<step>`` → result dir; relinking is idempotent and repoints atomically."""
    results = tmp_path / "reruns" / "results"
    first, second = result_dir(results, SHA), result_dir(results, OTHER)
    first.mkdir(parents=True)
    second.mkdir()
    run = run_dir(tmp_path / "reruns" / "runs", datetime.date(2026, 10, 7), "rerun")
    link = link_run(run, "train", first)
    assert link == run / "train" and link.is_symlink()
    assert not os.path.isabs(os.readlink(link))
    assert link.resolve() == first.resolve()
    assert link_run(run, "train", first) == link and link.resolve() == first.resolve()
    link_run(run, "train", second)
    assert link.resolve() == second.resolve()
    assert sorted(p.name for p in run.iterdir()) == ["train"]


def test_link_run_refusals(tmp_path: Path) -> None:
    """A non-symlink at the link path, a bad step name and a missing result dir are refused."""
    results = tmp_path / "results"
    out = result_dir(results, SHA)
    run = tmp_path / "runs" / "2026-10-07" / "rerun"
    with pytest.raises(CacheError) as info:
        link_run(run, "train", out)
    assert info.value.code == "E_RESULT_MISSING"
    out.mkdir(parents=True)
    for bad in ("", "..", "a/b"):
        with pytest.raises(CacheError) as info:
            link_run(run, bad, out)
        assert info.value.code == "E_RUN_BAD_NAME"
    (run / "train").mkdir(parents=True)
    with pytest.raises(CacheError) as info:
        link_run(run, "train", out)
    assert info.value.code == "E_LINK_NOT_SYMLINK"
    assert str(tmp_path) not in str(info.value)
