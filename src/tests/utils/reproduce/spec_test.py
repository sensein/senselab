"""Tests for the manifest and profile schema."""

from __future__ import annotations

import copy
import datetime
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml
from pydantic import ValidationError

from senselab.utils.reproduce.spec import (
    EnvSpec,
    Manifest,
    Profile,
    SourceSpec,
    StepSpec,
    check_step_profile,
    load_manifest,
    load_manifest_set,
    load_profile,
    load_sources,
    placeholders,
)

COMMIT = "0123456789abcdef0123456789abcdef01234567"
SHA = "ab" * 32
SOURCES = {"lock": "sources/analysis.lock.yaml", "sha256": SHA}


def _env() -> dict[str, Any]:
    return {
        "python": "3.11",
        "python_exact": "3.11.9",
        "torch": {"index_url": "https://download.example/whl/cu128", "lock": "envs/a/torch.lock", "lock_sha256": SHA},
        "body": {
            "lock": "envs/a/body.lock",
            "lock_sha256": SHA,
            "source": "requirements.in",
            "requirements_in": "envs/a/requirements.in",
            "requirements_in_sha256": SHA,
            "index_url": "https://pypi.org/simple",
        },
        "sdist_allow": [
            {
                "name": "pkg",
                "url": "https://files.example/pkg-1.0.tar.gz",
                "sha256": SHA,
                "patches": [{"file": "envs/a/pkg.diff", "sha256": SHA}],
                "build_env": {"lock": "envs/a/build.lock", "lock_sha256": SHA},
                "build_env_vars": {"FORCE_CUDA": "1"},
                "build_modules": ["gcc/12.2.0", "cuda/12.9"],
                "torch_cuda_arch_list": "8.0;9.0",
            }
        ],
        "exclude_newer": "2026-01-01",
        "extra_no_emit": ["opencv-python"],
        "first_resolve": {"file": "envs/a/first_resolve.txt", "sha256": SHA},
        "wheels": {"lock": "envs/a/wheels.lock", "lock_sha256": SHA},
        "test": None,
        "notebook_tools": {"lock": "envs/a/notebook_tools.lock", "lock_sha256": SHA},
        "pip_check_expected": [{"package": "decord", "kind": "platform", "detail": "not supported"}],
        "smoke_imports": ["torch"],
        "max_cuda": "12.8",
    }


def _metrics_schema(**over: Any) -> dict[str, Any]:  # noqa: ANN401
    schema: dict[str, Any] = {
        "kind": "classification",
        "headline": ["balanced_accuracy"],
        "secondary": ["top1"],
        "headline_labels": ["sys_a"],
        "counts": ["n_systems"],
        "per_class": ["walk", "run", "0"],
    }
    schema.update(over)
    return schema


def _step(name: str = "train", **over: Any) -> dict[str, Any]:  # noqa: ANN401
    step: dict[str, Any] = {
        "name": name,
        "repo": "https://git.example/org/analysis.git",
        "replicate": 0,
        "commit": COMMIT,
        "submodules": True,
        "kind": "python",
        "env": _env(),
        "argv": ["{venv}/bin/python", "run.py", "--data", "{root:proj}/x", "--out", "{out}", "--tmp", "{tmp}"],
        "cwd": "src",
        "pythonpath": ["{checkout}/src"],
        "inputs": [{"root": "proj", "patterns": ["x/**/*.csv"]}],
        "stage": [{"root": "checkout", "patterns": ["configs/*.yaml"], "dest": "configs"}],
        "outputs": ["results/*.json"],
        "needs": [],
        "resources": {"profile": "gpu", "gpu_arch": ["sm_80"], "min_driver": "550", "native": ["ffmpeg"]},
        "patches": [{"file": "patches/train/0001.diff", "sha256": SHA}],
        "adapter": "adapters/train.py",
        "hf_weights": [],
        "restricted": False,
        "metrics_schema": _metrics_schema(),
        "expect": [{"metric": "top1", "value": 77.56, "tolerance": 0.01, "unit": "pp", "kind": "abs"}],
    }
    step.update(over)
    return step


def _scoring_steps() -> list[dict[str, Any]]:
    return [
        _step("target"),
        _step("s0a"),
        _step(
            "s0b",
            replicate=1,
            needs=["s0a"],
            expect=[{"metric": "top1", "from_step": {"step": "s0a", "metric": "top1"}, "kind": "exact"}],
        ),
        _step("s1"),
        _step("s2"),
        _step(
            "score",
            needs=["target", "s0a", "s1", "s2"],
            argv=["{venv}/bin/python", "score.py", "{up:s0a}/results"],
            expect=[
                {
                    "metric": "top1",
                    "from_step": {"step": "target", "metric": "top1"},
                    "tolerance_from": {"steps": ["s1", "s2"], "rule": "max(a, k*sd)", "a": 0.5, "k": 2},
                    "unit": "pp",
                    "kind": "abs",
                }
            ],
        ),
    ]


def _manifest() -> dict[str, Any]:
    return {"name": "analysis", "sources": SOURCES, "steps": _scoring_steps()}


def _profile(**over: Any) -> dict[str, Any]:  # noqa: ANN401
    profile: dict[str, Any] = {
        "roots": {"proj": "/data/root", "scratch": "/scratch/root"},
        "results_root": "/data/root/reruns/results",
        "runs_root": "/data/root/reruns",
        "slurm_logs_root": "/data/root/reruns/slurm",
        "rw_under_roots": ["/data/root/reruns"],
        "env": {"PROJ_DATA_ROOT": "/data/root"},
        "partitions": {
            "gpu": {
                "partition": "gpu-a",
                "gres": "gpu:1",
                "cpus": 8,
                "mem": "64G",
                "time": "4:00:00",
                "fallback": "gpu-b",
            },
            "cpu": {"partition": "cpu-a", "cpus": 16, "mem": "32G", "time": "1:00:00"},
        },
        "build": "cpu",
        "native": {"ffmpeg": {"bin_dirs": ["/opt/tools/bin"], "lib_dirs": ["/opt/tools/curated-lib"]}},
        "r": {"module": "R/4.5.2", "version": "4.5.2"},
        "deny_patterns_file": "/etc/repro/deny.txt",
        "canary_host": "example.org:443",
        "policy": {
            "min_cell": 5,
            "headline_rule": "balanced_accuracy_only",
            "require_deny_patterns": True,
            "require_canary": True,
            "confidential_provenance": True,
            "umask": "007",
        },
    }
    profile.update(over)
    return profile


def _write(path: Path, obj: object) -> Path:
    path.write_text(yaml.safe_dump(obj, sort_keys=False))
    return path


def test_manifest_round_trips_through_yaml(tmp_path: Path) -> None:
    """A valid manifest loads, dumps to YAML and loads back equal."""
    first = load_manifest(_write(tmp_path / "m.yaml", _manifest()))
    second = load_manifest(_write(tmp_path / "m2.yaml", first.model_dump(mode="json")))
    assert first == second
    assert [s.name for s in second.steps] == ["target", "s0a", "s0b", "s1", "s2", "score"]
    assert second.steps[2].replicate == 1


def test_profile_round_trips_through_yaml(tmp_path: Path) -> None:
    """A valid profile loads, dumps to YAML and loads back equal."""
    first = load_profile(_write(tmp_path / "p.yaml", _profile()))
    second = load_profile(_write(tmp_path / "p2.yaml", first.model_dump(mode="json")))
    assert first == second
    assert second.partitions["gpu"].fallback == "gpu-b"


def test_manifest_and_profile_load_from_json(tmp_path: Path) -> None:
    """JSON files load the same as YAML."""
    (tmp_path / "m.json").write_text(json.dumps(_manifest()))
    (tmp_path / "p.json").write_text(json.dumps(_profile()))
    assert load_manifest(tmp_path / "m.json") == Manifest.model_validate(_manifest())
    assert load_profile(tmp_path / "p.json") == Profile.model_validate(_profile())


def test_unknown_suffix_is_refused(tmp_path: Path) -> None:
    """Only YAML and JSON files are read."""
    (tmp_path / "m.toml").write_text("")
    with pytest.raises(ValueError):
        load_manifest(tmp_path / "m.toml")


def _set_commit(m: dict[str, Any]) -> None:
    m["steps"][0]["commit"] = COMMIT[:8]


def _set_sha(m: dict[str, Any]) -> None:
    m["steps"][0]["env"]["body"]["lock_sha256"] = SHA[:40]


def _set_placeholder(m: dict[str, Any]) -> None:
    m["steps"][0]["argv"].append("{home}/x")


def _set_up(m: dict[str, Any]) -> None:
    m["steps"][1]["argv"].append("{up:target}")


def _set_from_step(m: dict[str, Any]) -> None:
    m["steps"][-1]["expect"][0]["from_step"]["step"] = "s0b"


def _set_tolerance_from(m: dict[str, Any]) -> None:
    m["steps"][-1]["expect"][0]["tolerance_from"]["steps"].append("s0b")


def _set_duplicate(m: dict[str, Any]) -> None:
    m["steps"][1]["name"] = "target"


def _set_notebook(m: dict[str, Any]) -> None:
    m["steps"][0]["kind"] = "notebook"
    m["steps"][0]["env"]["notebook_tools"] = None


def _set_ir_notebook(m: dict[str, Any]) -> None:
    m["steps"][0]["kind"] = "ir_notebook"
    m["steps"][0]["env"].update(
        notebook_tools=None,
        r_packages=[{"name": "lme4", "version": "1.1-35"}],
        r_snapshot="2026-01-01",
        r_lib="{cache}/r/a",
    )


MANIFEST_COUNTER_EXAMPLES: dict[str, Callable[[dict[str, Any]], None]] = {
    "sources-missing": lambda m: m.pop("sources"),
    "commit-not-40-hex": _set_commit,
    "sha256-not-64-hex": _set_sha,
    "unknown-placeholder": _set_placeholder,
    "up-outside-needs": _set_up,
    "from-step-outside-needs": _set_from_step,
    "tolerance-from-outside-needs": _set_tolerance_from,
    "duplicate-step-names": _set_duplicate,
    "notebook-without-notebook-tools": _set_notebook,
    "ir-notebook-without-notebook-tools": _set_ir_notebook,
}


@pytest.mark.parametrize("case", sorted(MANIFEST_COUNTER_EXAMPLES))
def test_manifest_validator_rejects_counter_example(case: str) -> None:
    """Each manifest validator rejects the one change that breaks it."""
    manifest = copy.deepcopy(_manifest())
    MANIFEST_COUNTER_EXAMPLES[case](manifest)
    with pytest.raises(ValidationError):
        Manifest.model_validate(manifest)


PROFILE_COUNTER_EXAMPLES: dict[str, dict[str, Any]] = {
    "secret-like-env-key": {"env": {"PROJ_DATA_ROOT": "/data/root", "HF_TOKEN": "x"}},
    "results-root-inside-data-root": {"results_root": "/scratch/root/results"},
    "runs-root-inside-data-root": {"runs_root": "/data/root/runs"},
    "slurm-logs-root-inside-data-root": {"slurm_logs_root": "/data/root/slurm"},
}


@pytest.mark.parametrize("case", sorted(PROFILE_COUNTER_EXAMPLES))
def test_profile_validator_rejects_counter_example(case: str) -> None:
    """Each profile validator rejects the one change that breaks it."""
    with pytest.raises(ValidationError):
        Profile.model_validate(_profile(**PROFILE_COUNTER_EXAMPLES[case]))


def test_roots_outside_every_data_root_need_no_rw_entry() -> None:
    """Results, runs and log roots outside every data root are accepted without ``rw_under_roots``."""
    profile = Profile.model_validate(
        _profile(results_root="/out/results", runs_root="/out/runs", slurm_logs_root="/out/slurm", rw_under_roots=[])
    )
    assert profile.rw_under_roots == []


def test_secret_like_build_env_var_is_rejected() -> None:
    """The secret-key rule also covers an sdist's build environment map."""
    manifest = _manifest()
    manifest["steps"][0]["env"]["sdist_allow"][0]["build_env_vars"]["API_KEY"] = "x"
    with pytest.raises(ValidationError):
        Manifest.model_validate(manifest)


@pytest.mark.parametrize(
    "modules", [["cuda/12.9; rm -rf x"], ["gcc/12.2.0 cuda/12.9"], ["cuda/12.9", "cuda/12.9"], [""]]
)
def test_bad_build_modules_are_rejected(modules: list[str]) -> None:
    """``build_modules`` entries are single module names without shell syntax, and unique."""
    manifest = _manifest()
    manifest["steps"][0]["env"]["sdist_allow"][0]["build_modules"] = modules
    with pytest.raises(ValidationError):
        Manifest.model_validate(manifest)


def test_r_kind_requires_r_fields() -> None:
    """An ``r`` step needs ``r_packages``, ``r_snapshot`` and ``r_lib``; with them it validates."""
    step = _step(kind="r")
    with pytest.raises(ValidationError):
        StepSpec.model_validate(step)
    step["env"].update(r_packages=[{"name": "lme4", "version": "1.1-35"}], r_snapshot="2026-01-01", r_lib="{cache}/r/a")
    assert StepSpec.model_validate(step).env.r_lib == "{cache}/r/a"


def test_expect_needs_one_target_and_one_tolerance() -> None:
    """An ``abs`` expectation takes exactly one of value/from_step and one of tolerance/tolerance_from."""
    both = {
        "metric": "top1",
        "value": 1.0,
        "from_step": {"step": "x", "metric": "top1"},
        "tolerance": 0.1,
        "kind": "abs",
    }
    neither_tol = {"metric": "top1", "value": 1.0, "kind": "abs"}
    for expect in (both, neither_tol):
        with pytest.raises(ValidationError):
            StepSpec.model_validate(_step(needs=["x"], expect=[expect]))


def test_seed_band_shape_rho_and_needs_validators() -> None:
    """A ``seed_band`` takes only a band, ``rho < max_width_frac``, chains in needs, values in range."""
    band = {"stored": 70.0, "unit": "pp", "floor": 50.0, "floor_source": "expected_floor", "chains": ["a", "b"]}
    good = {"metric": "top1", "kind": "seed_band", "band": band}
    redrawn = {k: v for k, v in band.items() if k != "chains"} | {"n_subset": 200, "n_full": 2000}
    step = StepSpec.model_validate(_step(needs=["a", "b"], expect=[good]))
    assert step.expect[0].band is not None and step.expect[0].band.a_min == 0.2
    cases: list[tuple[dict[str, Any], list[str], str]] = [
        ({"metric": "top1", "kind": "seed_band"}, ["a", "b"], "E_SPEC_BAND_SHAPE"),
        ({**good, "tolerance": 0.5}, ["a", "b"], "E_SPEC_BAND_SHAPE"),
        ({**good, "band": {**band, "rho": 0.3}}, ["a", "b"], "E_SPEC_BAND_RHO"),
        ({**good, "band": {**band, "chains": ["a", "c"]}}, ["a", "b"], "outside its needs"),
        ({**good, "band": {**band, "stored": 0.7, "unit": "fraction"}}, ["a", "b"], "outside its range"),
        ({**good, "metric": "counts.n_systems"}, ["a", "b"], "E_SPEC_EXPECT_UNIT"),
        ({**good, "band": {**redrawn, "redraws": ["a", "b", "c"]}}, ["a", "b"], "outside its needs"),
    ]
    for expect, needs, message in cases:
        with pytest.raises(ValidationError, match=message):
            StepSpec.model_validate(_step(needs=needs, expect=[expect]))


def test_escaped_braces_are_not_placeholders() -> None:
    """Doubled braces are literal text, not placeholders."""
    step = StepSpec.model_validate(_step(argv=["awk", "{{print $1}}", "{out}/a"]))
    assert step.argv[1] == "{{print $1}}"


def test_manifest_set_requires_unique_names_across_files(tmp_path: Path) -> None:
    """Step names are unique across every manifest in a set."""
    a = _write(tmp_path / "a.yaml", {"name": "a", "sources": SOURCES, "steps": [_step("one"), _step("two")]})
    b = _write(tmp_path / "b.yaml", {"name": "b", "sources": SOURCES, "steps": [_step("three")]})
    c = _write(tmp_path / "c.yaml", {"name": "c", "sources": SOURCES, "steps": [_step("two")]})
    assert [s.name for s in load_manifest_set([a, b])] == ["one", "two", "three"]
    with pytest.raises(ValueError, match="two"):
        load_manifest_set([a, c])


def test_draft_sources_load_without_env(tmp_path: Path) -> None:
    """A draft manifest needs only name, repo and commit; submodules default to true."""
    path = _write(tmp_path / "d.yaml", {"name": "row", "repo": "https://git.example/org/a.git", "commit": COMMIT})
    (source,) = load_sources(path)
    assert source == SourceSpec(name="row", repo="https://git.example/org/a.git", commit=COMMIT, submodules=True)
    listed = _write(tmp_path / "l.yaml", [{"name": "r1", "repo": "u", "commit": COMMIT, "submodules": False}])
    assert load_sources(listed)[0].submodules is False
    with pytest.raises(ValidationError):
        load_sources(_write(tmp_path / "bad.yaml", {"name": "row", "repo": "u", "commit": "abc"}))


def _env_with(**over: Any) -> dict[str, Any]:  # noqa: ANN401
    env = _env()
    for key, value in over.items():
        target, _, field = key.partition("__")
        if field:
            env[target][field] = value
        else:
            env[target] = value
    return env


ENV_COUNTER_EXAMPLES: dict[str, tuple[dict[str, Any], str]] = {
    "requirements-in-without-file": ({"body__requirements_in": None}, "requirements_in and requirements_in_sha256"),
    "requirements-in-without-sha": ({"body__requirements_in_sha256": None}, "requirements_in and requirements_in"),
    "uv-export-without-cmd": ({"body__source": "uv-export"}, "needs export_cmd"),
    "config-file-without-sha": ({"body__config_file": "envs/a/uv.toml"}, "config_file and config_file_sha256"),
    "overrides-sha-without-file": ({"body__overrides_sha256": SHA}, "overrides and overrides_sha256"),
    "torch-without-first-resolve": ({"first_resolve": None}, "needs first_resolve"),
    "r-lib-outside-cache": ({"r_lib": "{out}/r/a"}, "E_SPEC_BAD_R_LIB"),
    "r-snapshot-int": ({"r_snapshot": 20260101}, "YYYY-MM-DD"),
    "r-snapshot-datetime": ({"r_snapshot": datetime.datetime(2026, 1, 1)}, "YYYY-MM-DD"),
    "r-snapshot-short": ({"r_snapshot": "2026-1-1"}, "YYYY-MM-DD"),
    "exclude-newer-int": ({"exclude_newer": 20260101}, "exclude_newer must be a YYYY-MM-DD"),
    "exclude-newer-datetime": ({"exclude_newer": datetime.datetime(2026, 1, 1)}, "exclude_newer must be a YYYY-MM-DD"),
    "exclude-newer-short": ({"exclude_newer": "2026-1-1"}, "exclude_newer must be a YYYY-MM-DD"),
    "lock-absolute": ({"body__lock": "/etc/body.lock"}, "path must be relative"),
    "lock-parent": ({"torch__lock": "envs/../../body.lock"}, "path must be relative"),
    "first-resolve-parent": ({"first_resolve": {"file": "../r.txt", "sha256": SHA}}, "path must be relative"),
    "requirements-in-absolute": ({"body__requirements_in": "/r.in"}, "path must be relative"),
    "patch-unhashed": (
        {"sdist_allow": [{**_env()["sdist_allow"][0], "patches": ["envs/a/pkg.diff"]}]},
        "valid dictionary",
    ),
    "patch-parent": (
        {"sdist_allow": [{**_env()["sdist_allow"][0], "patches": [{"file": "../pkg.diff", "sha256": SHA}]}]},
        "path must be relative",
    ),
}


@pytest.mark.parametrize("case", sorted(ENV_COUNTER_EXAMPLES))
def test_env_validator_rejects_counter_example(case: str) -> None:
    """Hashed files come with their sha256s, R fields are well formed, torch locks have a first resolve."""
    over, message = ENV_COUNTER_EXAMPLES[case]
    with pytest.raises(ValidationError, match=message):
        EnvSpec.model_validate(_env_with(**over))


def test_uv_export_body_without_torch_needs_no_first_resolve() -> None:
    """A uv-export body with its command, and a torch-free requirements.in env, need no first resolve."""
    uv = _env_with(body__source="uv-export", body__export_cmd=["uv", "export", "--frozen"], first_resolve=None)
    assert EnvSpec.model_validate(uv).body.export_cmd == ["uv", "export", "--frozen"]
    assert EnvSpec.model_validate(_env_with(torch=None, first_resolve=None)).torch is None
    assert EnvSpec.model_validate(_env_with(r_snapshot="2026-01-01")).r_snapshot == datetime.date(2026, 1, 1)
    assert EnvSpec.model_validate(_env()).exclude_newer == datetime.date(2026, 1, 1)
    assert EnvSpec.model_validate(_env_with(exclude_newer=datetime.date(2026, 1, 1))).exclude_newer.day == 1


STEP_COUNTER_EXAMPLES: dict[str, tuple[dict[str, Any], str]] = {
    "stage-up-outside-needs": (
        {"stage": [{"root": "up:other", "patterns": ["*"], "dest": "x"}]},
        r"outside its needs: \['other'\]",
    ),
    "input-up-outside-needs": ({"inputs": [{"root": "up:other", "patterns": ["*"]}]}, "outside its needs"),
    "stage-root-malformed": ({"stage": [{"root": "up:", "patterns": ["*"], "dest": "x"}]}, "should match pattern"),
    "stage-dest-placeholder": (
        {"stage": [{"root": "checkout", "patterns": ["*"], "dest": "{home}"}]},
        "unknown placeholder",
    ),
    "stage-dest-up-outside-needs": (
        {"stage": [{"root": "checkout", "patterns": ["*"], "dest": "{up:other}"}]},
        "outside its needs",
    ),
    "lone-open-brace": ({"argv": ["run", "{out"]}, "unmatched brace"),
    "lone-close-brace": ({"argv": ["run", "a}b"]}, "unmatched brace"),
    "needs-self": ({"needs": ["train"]}, "needs itself"),
    "needs-duplicate": ({"needs": ["a", "a"]}, "duplicate needs"),
    "step-patch-unhashed": ({"patches": ["patches/train/0001.diff"]}, "valid dictionary"),
    "step-patch-absolute": ({"patches": [{"file": "/tmp/0001.diff", "sha256": SHA}]}, "path must be relative"),
    "tolerance-from-duplicate-steps": (
        {
            "needs": ["a"],
            "expect": [
                {
                    "metric": "top1",
                    "value": 1.0,
                    "tolerance_from": {"steps": ["a", "a"], "rule": "max(a, k*sd)", "a": 0.5, "k": 2},
                    "kind": "abs",
                }
            ],
        },
        "duplicates",
    ),
    "tolerance-from-negative-a": (
        {
            "needs": ["a", "b"],
            "expect": [
                {
                    "metric": "top1",
                    "value": 1.0,
                    "tolerance_from": {"steps": ["a", "b"], "rule": "max(a, k*sd)", "a": -0.5, "k": 2},
                    "kind": "abs",
                }
            ],
        },
        "greater than or equal to 0",
    ),
    "metrics-schema-missing": ({"metrics_schema": None}, "valid dictionary"),
    "metrics-schema-unknown-headline": ({"metrics_schema": _metrics_schema(headline=["top5"])}, "headline"),
    "metrics-schema-empty-headline": ({"metrics_schema": _metrics_schema(headline=[])}, "at least 1 item"),
    "metrics-schema-duplicate-headline": (
        {"metrics_schema": _metrics_schema(headline=["top1", "top1"])},
        "headline has duplicates",
    ),
    "metrics-schema-duplicate-label": (
        {"metrics_schema": _metrics_schema(per_class=["walk", "walk"])},
        "per_class has duplicates",
    ),
    "metrics-schema-bad-label": ({"metrics_schema": _metrics_schema(per_class=["a b"])}, "should match pattern"),
    "metrics-schema-bad-count-name": ({"metrics_schema": _metrics_schema(counts=["N-1"])}, "should match pattern"),
    "metrics-schema-classification-without-classes": (
        {"metrics_schema": _metrics_schema(per_class=[])},
        "needs per_class labels",
    ),
    "metrics-schema-unknown-kind": ({"metrics_schema": _metrics_schema(kind="regression")}, "kind"),
    "metrics-schema-unknown-key": ({"metrics_schema": _metrics_schema(confusion=True)}, "Extra inputs"),
    "metrics-schema-headline-of-other-kind": (
        {"metrics_schema": _metrics_schema(headline=["f_stat"])},
        "E_SPEC_HEADLINE_NOT_BA",
    ),
    "metrics-schema-macro-recall-removed": (
        {"metrics_schema": _metrics_schema(headline=["macro_recall"])},
        r"headline\.0\n  Input should be 'balanced_accuracy'",
    ),
    "metrics-schema-pr-auc-without-positive": (
        {"metrics_schema": _metrics_schema(per_class=["pos", "neg"], secondary=["pr_auc"])},
        "E_SPEC_POSITIVE_REQUIRED",
    ),
    "metrics-schema-f1-without-positive": (
        {"metrics_schema": _metrics_schema(per_class=["pos", "neg"], secondary=["f1"])},
        "E_SPEC_POSITIVE_REQUIRED",
    ),
    "metrics-schema-pr-auc-multi-class": (
        {"metrics_schema": _metrics_schema(secondary=["pr_auc"])},
        "E_SPEC_SECONDARY_BINARY_ONLY",
    ),
    "metrics-schema-f1-multi-class": (
        {"metrics_schema": _metrics_schema(secondary=["top1", "f1"])},
        "E_SPEC_SECONDARY_BINARY_ONLY",
    ),
    "metrics-schema-ba-plus-accuracy-headline": (
        {"metrics_schema": _metrics_schema(headline=["balanced_accuracy", "accuracy"])},
        "E_SPEC_HEADLINE_NOT_BA",
    ),
    "metrics-schema-ba-as-secondary": (
        {"metrics_schema": _metrics_schema(secondary=["balanced_accuracy"])},
        "E_SPEC_SECONDARY_NOT_ALLOWED",
    ),
    "metrics-schema-secondary-on-statistic": (
        {
            "metrics_schema": _metrics_schema(kind="statistic", headline=["f_stat"], per_class=[], secondary=["mcc"]),
            "expect": [],
        },
        "E_SPEC_SECONDARY_NOT_ALLOWED",
    ),
    "metrics-schema-detection-with-classification-name": (
        {"metrics_schema": _metrics_schema(kind="detection", headline=["avg_map", "top1"]), "expect": []},
        "do not belong to kind detection",
    ),
    "metrics-schema-one-class": (
        {"metrics_schema": _metrics_schema(per_class=["walk"])},
        "at least two per_class labels",
    ),
    "metrics-schema-accuracy-headline": (
        {"metrics_schema": _metrics_schema(headline=["top1"], secondary=[])},
        "E_SPEC_HEADLINE_NOT_BA",
    ),
    "metrics-schema-positive-not-declared": (
        {"metrics_schema": _metrics_schema(per_class=["walk", "run"], positive="jump")},
        "positive must be one of exactly two",
    ),
    "metrics-schema-positive-three-classes": (
        {"metrics_schema": _metrics_schema(positive="walk")},
        "positive must be one of exactly two",
    ),
    "metrics-schema-statistic-with-classes": (
        {"metrics_schema": _metrics_schema(kind="statistic", headline=["f_stat"], secondary=[]), "expect": []},
        "takes no per_class labels",
    ),
    "metrics-schema-count-with-classes": (
        {
            "metrics_schema": _metrics_schema(kind="count", headline=["count"], secondary=[], per_class=["a", "b"]),
            "expect": [],
        },
        "takes no per_class labels",
    ),
    "metrics-schema-statistic-with-positive": (
        {
            "metrics_schema": _metrics_schema(
                kind="statistic", headline=["f_stat"], secondary=[], per_class=[], positive="a"
            ),
            "expect": [],
        },
        "takes no per_class labels",
    ),
    "metrics-schema-split-bad-reason": (
        {"metrics_schema": _metrics_schema(split={"none": "too_small"})},
        "whole_dataset",
    ),
    "expect-unit-unknown": (
        {"expect": [{"metric": "top1", "value": 1.0, "tolerance": 0.1, "unit": "percent", "kind": "abs"}]},
        "'fraction' or 'pp'",
    ),
    "expect-metric-undeclared": (
        {"expect": [{"metric": "auroc", "value": 0.5, "tolerance": 0.1, "kind": "abs"}]},
        "E_METRICS_UNDECLARED_HEADLINE",
    ),
    "expect-metric-bad-address": (
        {"expect": [{"metric": "per_class.walk.auc", "value": 0.5, "kind": "exact"}]},
        "E_METRICS_BAD_ADDRESS",
    ),
    "expect-metric-undeclared-label": (
        {"expect": [{"metric": "top1@sys_z", "value": 0.5, "kind": "exact"}]},
        "E_METRICS_UNDECLARED_LABEL",
    ),
    "tolerance-from-negative-k": (
        {
            "needs": ["a", "b"],
            "expect": [
                {
                    "metric": "top1",
                    "value": 1.0,
                    "tolerance_from": {"steps": ["a", "b"], "rule": "max(a, k*sd)", "a": 0.5, "k": -2},
                    "kind": "abs",
                }
            ],
        },
        "greater than or equal to 0",
    ),
}


@pytest.mark.parametrize("case", sorted(STEP_COUNTER_EXAMPLES))
def test_step_validator_rejects_counter_example(case: str) -> None:
    """Stage and input roots, templates, needs and tolerance_from are checked."""
    over, message = STEP_COUNTER_EXAMPLES[case]
    with pytest.raises(ValidationError, match=message):
        StepSpec.model_validate(_step(**over))


@pytest.mark.parametrize(
    "metric", ["top1", "balanced_accuracy", "top1@sys_a", "counts.n_systems", "per_class.walk.recall", "confusion.fn"]
)
def test_expect_metric_addresses_resolve_against_the_schema(metric: str) -> None:
    """Every addressable form of a declared metric validates as an ``expect`` metric."""
    step = StepSpec.model_validate(_step(expect=[{"metric": metric, "value": 1.0, "kind": "exact"}]))
    assert step.expect[0].metric == metric


def test_binary_schema_with_positive_and_split_opt_out() -> None:
    """A binary classification step may declare its positive class and opt out of the split report."""
    schema = _metrics_schema(per_class=["pos", "neg"], positive="pos", split={"none": "whole_dataset"})
    step = StepSpec.model_validate(_step(metrics_schema=schema))
    assert step.metrics_schema.positive == "pos"
    assert step.metrics_schema.split is not None and step.metrics_schema.split.none == "whole_dataset"


def test_statistic_schema_needs_no_classes() -> None:
    """A test-statistic step declares only its headline names."""
    schema = _metrics_schema(
        kind="statistic", headline=["f_stat"], secondary=[], headline_labels=[], counts=[], per_class=[]
    )
    step = StepSpec.model_validate(_step(metrics_schema=schema, expect=[]))
    assert step.metrics_schema.headline == ["f_stat"]
    assert step.metrics_schema.per_class == []


def test_up_stage_root_and_escaped_braces_validate() -> None:
    """``up:<step>`` stage roots inside needs, and ``{{{out}}}``, are accepted; submodules default to true."""
    step = _step(
        needs=["prep"],
        stage=[{"root": "up:prep", "patterns": ["*.csv"], "dest": "{tmp}/x"}],
        argv=["echo", "{{{out}}}"],
    )
    del step["submodules"]
    parsed = StepSpec.model_validate(step)
    assert parsed.submodules is True
    assert placeholders("{{{out}}}") == ["out"]


@pytest.mark.parametrize("key", ["MONKEY", "KEYED", "TOKENIZERS_PARALLELISM", "PROJ_DATA_ROOT"])
def test_secret_rule_accepts_ordinary_keys(key: str) -> None:
    """Keys that merely contain a secret-like word are accepted."""
    assert key in Profile.model_validate(_profile(env={key: "1"})).env


@pytest.mark.parametrize("key", ["HF_TOKEN", "API_KEY", "DB_PASSWORD", "AWS_CREDENTIALS", "GITHUB_TOKENS"])
def test_secret_rule_refuses_secret_keys(key: str) -> None:
    """Keys with a secret-like word as a whole ``_``-separated part are refused."""
    with pytest.raises(ValidationError, match="secret-like"):
        Profile.model_validate(_profile(env={key: "1"}))


ROOT_COUNTER_EXAMPLES: dict[str, tuple[dict[str, Any], str]] = {
    "root-named-checkout": ({"roots": {"proj": "/data/root", "checkout": "/x"}}, "reserved root names"),
    "rw-equal-to-root": ({"rw_under_roots": ["/data/root"]}, "strictly under a data root"),
    "rw-slash": ({"rw_under_roots": ["/"]}, "strictly under a data root"),
    "rw-outside-roots": ({"rw_under_roots": ["/elsewhere/reruns"]}, "strictly under a data root"),
    "double-slash-inside-root": ({"results_root": "//data/root/results"}, "results_root lies inside"),
    "rw-entry-in-other-root": (
        {
            "roots": {"proj": "/data/root", "inner": "/data/root/inner"},
            "rw_under_roots": ["/data/root/inner"],
            "results_root": "/data/root/inner/results",
            "runs_root": "/out/runs",
            "slurm_logs_root": "/out/slurm",
        },
        "results_root lies inside",
    ),
}


@pytest.mark.parametrize("case", sorted(ROOT_COUNTER_EXAMPLES))
def test_root_validator_rejects_counter_example(case: str) -> None:
    """Reserved root names, rw entries outside or equal to a root, and leading ``//`` are caught."""
    over, message = ROOT_COUNTER_EXAMPLES[case]
    with pytest.raises(ValidationError, match=message):
        Profile.model_validate(_profile(**over))


@pytest.mark.parametrize(
    "native",
    [{"ff": {"bin_dirs": ["relative/bin"]}}, {"ff": {"lib_dirs": ["/opt/a:b"]}}, {"bad name": {}}],
)
def test_native_entries_validate(native: dict[str, Any]) -> None:
    """Native dirs are absolute with no ':', and entry names are plain names."""
    with pytest.raises(ValidationError):
        Profile.model_validate(_profile(native=native))


def test_step_may_select_only_profile_native_entries() -> None:
    """``check_step_profile`` refuses a native name or partition key the profile lacks."""
    profile = Profile.model_validate(_profile())
    step = StepSpec.model_validate(_step())
    check_step_profile(step, profile)
    with pytest.raises(ValueError):
        check_step_profile(
            step.model_copy(update={"resources": step.resources.model_copy(update={"native": ["x"]})}), profile
        )
    with pytest.raises(ValueError):
        check_step_profile(
            step.model_copy(update={"resources": step.resources.model_copy(update={"profile": "x"})}), profile
        )


def test_a_root_path_no_declared_pattern_covers_is_refused_at_load() -> None:
    """``{root:X}`` in argv, cwd, pythonpath or a stage dest must be covered by an ``inputs`` pattern of root X.

    A path a pattern matches, a dir a pattern reaches below, and the root itself are covered. A step with
    ``lock_inputs`` defers the paths no pattern covers to the registry gate (`undeclared_root_paths`).
    """
    from senselab.utils.reproduce.spec import undeclared_root_paths

    def argv(path: str) -> dict[str, Any]:
        return {"argv": ["{venv}/bin/python", "run.py", path]}

    for ok in ("{root:proj}/x", "{root:proj}", "{root:proj}/x/a/b.csv", "--data={root:proj}/x/"):
        StepSpec.model_validate(_step(**argv(ok)))
    refused = [
        argv("{root:proj}/y/a.csv"),
        argv("{root:other}/x"),
        argv("{root:proj}/x/../y"),
        argv("{root:proj}/x:/../../y"),
        argv("--a={root:proj}/x,../z"),
        {"cwd": "{root:other}"},
        {"pythonpath": ["{root:other}/lib"]},
        {"stage": [{"root": "checkout", "patterns": ["configs/*.yaml"], "dest": "{root:other}/d"}]},
    ]
    for change in refused:
        with pytest.raises(ValidationError, match="E_SPEC_UNDECLARED_ROOT"):
            StepSpec.model_validate(_step(**change))
    deferred = StepSpec.model_validate(_step(**argv("{root:other}/t/f.csv"), lock_inputs=["t"]))
    assert undeclared_root_paths(deferred, {"other": ["t"]}) == []
    assert undeclared_root_paths(deferred, {"other": ["t/f.csv/x"]}) == []
    assert undeclared_root_paths(deferred, {"other": ["u"]}) == ["{root:other}/t/f.csv"]
    assert undeclared_root_paths(deferred, {}) == ["{root:other}/t/f.csv"]
