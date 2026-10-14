"""Tests for the step runner: keys and skips, the sandboxed run, expectations, adapters and provenance.

Every repository, venv, data root and result tree here is built in ``tmp_path`` from synthetic files.
Venvs are stdlib venvs of the test interpreter with a hand-written marker, so nothing is installed.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import socket
import subprocess
import sys
import uuid
import venv
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import pytest

import senselab
from senselab.utils.reproduce import cache, digest, env, host, provenance, restricted, run, sandbox, source, weights
from senselab.utils.reproduce.digest import input_set_digest
from senselab.utils.reproduce.run import RunConfig, StepOutcome, evaluate_expect, run_step, run_steps
from senselab.utils.reproduce.spec import Manifest, Profile, SourcesRef, StepSpec
from tests.utils.reproduce.sources_fixture import lock_data, register_input, write_lock

SHA = "ab" * 32
MARKER = "MARKER-STDOUT-7731"
OFFLINE_CANARY = "192.0.2.1:9"
REFUSED_CANARY = "127.0.0.1:9"
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

STEP_SCRIPT = """\
# v0
import json, os, pathlib, sys
mode, out, src = sys.argv[1], pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
print("MARKER-STDOUT-7731")
if mode == "write-root":
    (src.parent / "evil.txt").write_text("x")
if mode == "write-cwd":
    pathlib.Path("written-in-cwd.txt").write_text("x")
if mode == "prep":
    value = float(src.read_text().strip())
elif mode == "stage":
    value = float((out / "stage" / "data" / "in" / "a.txt").read_text()) + len(list((out / "stage").rglob("*.txt")))
else:
    value = 2 * json.loads(src.read_text())["value"]
(out / "result.json").write_text(json.dumps({"value": value}))
"""

ADAPTER = """\
import json, pathlib, sys
from senselab.utils.reproduce.metrics import METRIC_NAMES
assert "f_stat" in METRIC_NAMES
out, dest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
value = json.loads((out / "result.json").read_text())["value"]
dest.write_text(json.dumps({"headline": [{"name": "f_stat", "value": value, "numdf": 1, "dendf": 10}]}))
"""


SCAN_HOOK = """\
#!/bin/sh
if /usr/bin/grep -q CANARY-7f3a "$1"; then exit 1; fi
exit 0
"""
"""The fixture scan hook: refuses a file holding the canary ``CANARY-7f3a``, passes any other."""


def _git(cwd: Path, *args: str) -> str:
    argv = ["git", "-c", "init.defaultBranch=main", "-C", str(cwd), *args]
    env_vars = {**FIXTURE_ENV, "HOME": str(cwd)}
    return subprocess.run(argv, env=env_vars, check=True, capture_output=True, text=True).stdout


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class World:
    """A synthetic site: one repository, a manifest repository, a data root, a profile and a cache."""

    def __init__(self, tmp: Path) -> None:
        """Build the site under ``tmp``."""
        self.tmp = tmp
        self.repo = tmp / "upstream" / "analysis"
        self.repo.mkdir(parents=True)
        _git(self.repo, "init", "-q")
        (self.repo / "step.py").write_text(STEP_SCRIPT)
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "analysis")
        self.commit = _git(self.repo, "rev-parse", "HEAD").strip()
        self.base = tmp / "manifests"
        (self.base / "patches").mkdir(parents=True)
        (self.base / "adapter.py").write_text(ADAPTER)
        (self.base / "adapter_prep.py").write_text(ADAPTER)
        self.write_patch("# v1")
        self.data = tmp / "data"
        (self.data / "in").mkdir(parents=True)
        (self.data / "in" / "a.txt").write_text("2\n")
        self.cache = tmp / "cache"
        self.env_map = {"MODE": "a"}
        self.canary = OFFLINE_CANARY
        self.wheels_sha: str | None = None
        self.registered: dict[str, dict[str, str]] = {}
        self.sources: dict[str, str] | None = None
        self.hook = tmp / "hook.sh"
        self.hook.write_text(SCAN_HOOK)
        self.hook.chmod(0o755)
        self.restricted_dirs: list[Path] = [self.data / "restricted"]
        self.restricted_fields = True
        self.profile_extra: dict[str, Any] = {}
        self.extra_roots: dict[str, str] = {}

    def write_patch(self, line: str) -> None:
        """Write the patch that changes the step script's first line to ``line``."""
        context = "".join(f" {c}\n" for c in STEP_SCRIPT.splitlines()[1:4])
        text = f"--- a/step.py\n+++ b/step.py\n@@ -1,4 +1,4 @@\n-# v0\n+{line}\n{context}"
        (self.base / "patches" / "p.patch").write_text(text)

    def restricted_profile_fields(self) -> dict[str, Any]:
        """The restricted fields: ``<first restricted dir>/reruns/results``, released metrics under the data root."""
        first = self.restricted_dirs[0]
        return {
            "restricted_dirs": [str(p) for p in self.restricted_dirs],
            "restricted_results_root": str(first / "reruns" / "results"),
            "released_metrics_root": str(self.data / "reruns" / "restricted-metrics"),
            "scan_hook": [str(self.hook)],
            "rw_under_roots": [str(first / "reruns"), str(self.data / "reruns")],
        }

    def profile(self) -> Profile:
        """The profile, rebuilt from the current env map, canary host and restricted fields."""
        extra = self.restricted_profile_fields() if self.restricted_fields else {}
        return Profile.model_validate(
            {
                **extra,
                **self.profile_extra,
                "roots": {"data": str(self.data), **self.extra_roots},
                "results_root": str(self.tmp / "results"),
                "runs_root": str(self.tmp / "runs"),
                "slurm_logs_root": str(self.tmp / "slurm"),
                "env": dict(self.env_map),
                "partitions": {"cpu": {"partition": "p", "cpus": 1, "mem": "1G", "time": "0:10:00"}},
                "build": "cpu",
                "deny_patterns_file": str(self.tmp / "deny.txt"),
                "canary_host": self.canary,
                "policy": {
                    "min_cell": 5,
                    "headline_rule": "balanced_accuracy_only",
                    "require_deny_patterns": True,
                    "require_canary": True,
                    "confidential_provenance": True,
                    "umask": "007",
                },
            }
        )

    def config(self, **change: Any) -> RunConfig:  # noqa: ANN401
        """A run configuration for this world; its sources are those of the latest `manifest`."""
        sources = {} if self.sources is None else {"demo": SourcesRef.model_validate(self.sources)}
        return RunConfig(profile=self.profile(), cache=self.cache, base_dir=self.base, sources=sources, **change)

    def step(self, name: str, mode: str, src: str, **change: Any) -> dict[str, Any]:  # noqa: ANN401
        """A step running ``step.py <mode> {out} <src>``."""
        data: dict[str, Any] = {
            "name": name,
            "repo": str(self.repo),
            "commit": self.commit,
            "kind": "python",
            "env": {
                "python": "3.12",
                "python_exact": "3.12.11",
                "torch": None,
                "body": {"lock": "b.lock", "lock_sha256": SHA, "source": "uv-export", "export_cmd": ["uv"]},
                "exclude_newer": "2026-01-01",
            },
            "argv": ["{venv}/bin/python", "{checkout}/step.py", mode, "{out}", src],
            "inputs": [{"root": "data", "patterns": ["in/a.txt"]}],
            "outputs": ["result.json"],
            "resources": {"profile": "cpu"},
            "adapter": "adapter.py",
            "hf_weights": [],
            "restricted": False,
            "metrics_schema": {"kind": "statistic", "headline": ["f_stat"]},
        }
        if self.wheels_sha is not None:
            data["env"]["wheels"] = {"lock": "wheels.lock", "lock_sha256": self.wheels_sha}
        data.update(change)
        return data

    def manifest(self, *steps: dict[str, Any]) -> Manifest:
        """A manifest named ``demo``, with a sources lock that pins its steps and `registered` inputs."""
        validated = [StepSpec.model_validate(s) for s in steps]
        self.sources = write_lock(self.base, "demo", lock_data(validated, self.registered))
        return Manifest.model_validate({"name": "demo", "sources": self.sources, "steps": list(steps)})

    def dag(self, prep_expect: Sequence[dict[str, Any]] = ()) -> Manifest:
        """Two steps: ``prep`` reads the data root, ``score`` reads ``prep``'s result dir through ``{up:prep}``."""
        prep = self.step(
            "prep",
            "prep",
            "{root:data}/in/a.txt",
            patches=[{"file": "patches/p.patch", "sha256": _sha(self.base / "patches" / "p.patch")}],
            adapter="adapter_prep.py",
            expect=list(prep_expect),
        )
        score = self.step(
            "score",
            "score",
            "{up:prep}/result.json",
            needs=["prep"],
            inputs=[{"root": "up:prep", "patterns": ["result.json"]}],
            expect=[{"metric": "f_stat", "value": 4.0, "tolerance": 0.5, "kind": "abs"}],
        )
        return self.manifest(prep, score)

    def prefetch(self, manifest: Manifest, *, checked: bool = True) -> None:
        """Check out every step and give it a stdlib venv with a marker, as `env.prefetch_env` would.

        Each weights file present is stamped as verified, as prefetch's sha256 check would (`weights.write_stamp`).
        """
        profile = self.profile()
        for step in manifest.steps:
            for file, sha256 in weights.declared_files(step, profile):
                if file is not None and file.is_file():
                    weights.write_stamp(self.cache / weights.STAMPS_DIR, file, sha256)
            source.checkout_step(step, cache=self.cache, base_dir=self.base)
            path, overlay = run._step_venv(step, manifest.name, self.cache)
            if not (path / "bin" / "python").exists():
                venv.EnvBuilder(symlinks=True, with_pip=False).create(path)
            data = json.dumps({"packages": {}, "freeze": []}).encode()
            (path / env.INSTALLED).write_bytes(data)
            marker = {
                "env_key": env.env_key(step.env),
                "overlay_sha256": None if overlay is None else overlay.lock_sha256,
                "wheels": {},
                "installed_sha256": hashlib.sha256(data).hexdigest(),
                "checks": {"digest": env.checks_digest(step.env), "pip_check_ok": checked, "smoke_ok": checked},
            }
            (path / env.MARKER).write_text(json.dumps(marker))
            if overlay is not None:
                env.record_kernels(path)

    def run(self, manifest: Manifest, **change: Any) -> dict[str, StepOutcome]:  # noqa: ANN401
        """Run the manifest and return the outcomes by step name."""
        return {o.step: o for o in run_steps([manifest], self.config(**change))}


@pytest.fixture
def world(tmp_path: Path) -> World:
    """A fresh synthetic site."""
    return World(tmp_path)


def _executed(outcomes: dict[str, StepOutcome]) -> list[str]:
    return sorted(name for name, o in outcomes.items() if o.status == "executed")


def _summary(outcome: StepOutcome) -> dict[str, Any]:
    assert outcome.out is not None
    data: dict[str, Any] = json.loads((outcome.out / cache.SUMMARY).read_text())
    return data


# --- the DAG and the cache ----------------------------------------------------------------------


@needs_userns
def test_two_step_dag_runs_then_skips_and_reruns_on_each_keyed_change(world: World) -> None:
    """The hand-off works; a rerun executes nothing; a patch, wheels.lock, adapter or env-map change reruns both."""
    manifest = world.dag()
    world.prefetch(manifest)
    first = world.run(manifest)
    assert _executed(first) == ["prep", "score"], {n: o.codes for n, o in first.items()}
    assert all(o.ok for o in first.values())
    score_out = first["score"].out
    assert score_out is not None
    metrics_json = json.loads((score_out / run.METRICS).read_text())
    assert metrics_json["headline"][0] | {"pass": True} == metrics_json["headline"][0]
    assert metrics_json["headline"][0]["target"] == 4.0 and metrics_json["pass"] is True
    summary = _summary(first["score"])
    assert summary["exit"] == 0 and summary["executed"] is True and summary["pass"] is True
    assert summary["canary_failed"] is True and summary["sys_prefix_ok"] is True
    assert set(summary) == {
        "exit",
        "exc_class",
        "frame",
        "codes",
        "pass",
        "sys_prefix_ok",
        "torch",
        "torch_cuda",
        "canary_failed",
        "executed",
        "verdict",
        "extracted",
        "released",
        "input_closure",
        "attempt",
    }
    link = world.tmp / "runs" / world.config().date.strftime("%Y-%m-%d") / "demo" / "score"
    assert link.is_symlink() and link.resolve() == score_out.resolve()
    rows = json.loads((link.parent / run.RUN_JSON).read_text())["steps"]
    assert [(r["step"], r["status"], r["exit"]) for r in rows] == [("prep", "executed", 0), ("score", "executed", 0)]

    second = world.run(manifest)
    assert _executed(second) == [] and {o.status for o in second.values()} == {"skipped"}
    assert {n: o.key for n, o in second.items()} == {n: o.key for n, o in first.items()}

    world.write_patch("# v2")
    patched = world.dag()
    world.prefetch(patched)
    assert _executed(world.run(patched)) == ["prep", "score"]
    assert _executed(world.run(patched)) == []

    world.wheels_sha = "cd" * 32
    wheeled = world.dag()
    world.prefetch(wheeled)
    assert _executed(world.run(wheeled)) == ["prep", "score"]

    (world.base / "adapter_prep.py").write_text(ADAPTER + "# edited\n")
    assert _executed(world.run(wheeled)) == ["prep", "score"]

    world.env_map = {"MODE": "b"}
    assert _executed(world.run(wheeled)) == ["prep", "score"]
    assert _executed(world.run(wheeled)) == []


@needs_userns
def test_a_failed_expectation_exits_non_zero_and_its_dependent_never_runs(world: World) -> None:
    """``E_EXPECT`` gives a non-zero exit and an undone result dir; the afterok dependent is ``never-ran``."""
    manifest = world.dag(prep_expect=[{"metric": "f_stat", "value": 99.0, "tolerance": 0.5, "kind": "abs"}])
    world.prefetch(manifest)
    outcomes = world.run(manifest)
    prep = outcomes["prep"]
    assert prep.status == "executed" and prep.exit != 0 and "E_EXPECT" in prep.codes
    assert prep.out is not None and not cache.is_done(prep.out)
    summary = _summary(prep)
    assert summary["pass"] is False and summary["exit"] != 0
    entry = json.loads((prep.out / run.METRICS).read_text())["headline"][0]
    assert entry["pass"] is False and entry["delta"] == pytest.approx(2.0 - 99.0)
    assert outcomes["score"].status == "never-ran" and outcomes["score"].exit is None
    again = world.run(manifest)
    assert again["prep"].status == "executed" and again["score"].status == "never-ran"


@needs_userns
def test_step_stdout_reaches_only_the_log(world: World, capfd: pytest.CaptureFixture[str]) -> None:
    """A marker the step prints is absent from the caller's stdout and stderr and present in ``{out}/logs``."""
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    capfd.readouterr()
    outcome = world.run(manifest)["prep"]
    captured = capfd.readouterr()
    assert outcome.ok, outcome.codes
    assert MARKER not in captured.out and MARKER not in captured.err
    assert outcome.out is not None
    assert MARKER in (outcome.out / run.LOGS / "step.stdout.log").read_text()


@needs_userns
def test_a_step_writing_into_a_read_only_root_fails_and_leaves_its_inputs(world: World) -> None:
    """The write fails with EROFS, the step exits non-zero, and the input digest is unchanged."""
    manifest = world.manifest(world.step("prep", "write-root", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    before = input_set_digest(world.data, ["**"])
    outcome = world.run(manifest)["prep"]
    assert outcome.status == "executed" and outcome.exit != 0
    assert "E_STEP_FAILED" in outcome.codes and "E_INPUT_MUTATED" not in outcome.codes
    assert input_set_digest(world.data, ["**"]) == before
    assert not (world.data / "in" / "evil.txt").exists()
    summary = _summary(outcome)
    assert summary["exc_class"] == "OSError" and summary["frame"].startswith("step.py:")


@needs_userns
def test_the_step_cwd_is_the_read_only_checkout(world: World) -> None:
    """The step starts in the checkout (``cwd: .``) on its read-only bind, so a relative write fails."""
    manifest = world.manifest(world.step("prep", "write-cwd", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.exit != 0 and "E_STEP_FAILED" in outcome.codes
    co = source.prefetched_checkout(manifest.steps[0], cache=world.cache)
    assert not (co.worktree / "written-in-cwd.txt").exists()
    assert "E_SRC_MUTATED" not in outcome.codes


@needs_userns
def test_stage_copies_selected_files_without_entering_symlinked_dirs(world: World) -> None:
    """Data-root and checkout stages are copied under ``{out}/stage``; a wildcard never enters a symlinked dir."""
    elsewhere = world.tmp / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "hidden.txt").write_text("x")
    (world.data / "in" / "linked").symlink_to(elsewhere, target_is_directory=True)
    step = world.step(
        "prep",
        "stage",
        "{root:data}/in/a.txt",
        inputs=[{"root": "data", "patterns": ["in/**/*.txt"]}],
        stage=[
            {"root": "data", "patterns": ["in/**/*.txt"], "dest": "data"},
            {"root": "checkout", "patterns": ["*.py"], "dest": "code"},
        ],
    )
    manifest = world.manifest(step)
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.ok, outcome.codes
    assert outcome.out is not None
    stage = outcome.out / run.WORK / run.STAGE
    assert sorted(p.relative_to(stage).as_posix() for p in stage.rglob("*") if p.is_file()) == [
        "code/step.py",
        "data/in/a.txt",
    ]
    assert json.loads((outcome.out / run.WORK / "result.json").read_text())["value"] == 3.0


@needs_userns
def test_a_no_netns_run_is_never_reportable_or_done(world: World) -> None:
    """Without the network namespace a run carries ``E_NOT_REPORTABLE`` and a non-zero exit, so it never skips."""
    world.canary = REFUSED_CANARY
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    outcome = world.run(manifest, no_netns=True)["prep"]
    assert outcome.status == "executed" and outcome.exit != 0
    assert outcome.codes == ("E_NOT_REPORTABLE",)
    assert outcome.out is not None and not cache.is_done(outcome.out)
    assert world.run(manifest, no_netns=True)["prep"].status == "executed"


@needs_userns
@pytest.mark.network
def test_no_netns_with_a_reachable_canary_aborts_with_e_canary(world: World) -> None:
    """On a node with network and no namespace the canary connects, so the step never runs (spec C3 negative)."""
    world.canary = "pypi.org:443"
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    outcome = world.run(manifest, no_netns=True)["prep"]
    assert outcome.status == "failed" and "E_CANARY" in outcome.codes and outcome.exit != 0
    assert outcome.out is not None
    assert _summary(outcome)["canary_failed"] is False
    assert not (outcome.out / run.WORK / "result.json").exists()
    assert MARKER not in (outcome.out / run.LOGS / "step.stdout.log").read_text()


# --- refusals -----------------------------------------------------------------------------------


def test_a_step_without_a_prefetched_checkout_is_refused_before_any_key(world: World) -> None:
    """No worktree: ``E_SRC_MISSING`` with no key and nothing written."""
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    outcome = run_step(manifest.steps[0], "demo", world.config())
    assert (outcome.status, outcome.key, outcome.codes) == ("failed", None, ("E_SRC_MISSING",))
    assert not (world.tmp / "results").exists()


def test_an_input_pattern_that_matches_nothing_is_refused(world: World) -> None:
    """Declared inputs are digested with ``require_nonempty``."""
    manifest = world.manifest(
        world.step("prep", "prep", "{root:data}/in", inputs=[{"root": "data", "patterns": ["in/none*.txt"]}])
    )
    world.prefetch(manifest)
    assert run_step(manifest.steps[0], "demo", world.config()).codes == ("E_DIGEST_EMPTY",)


def test_unchecked_and_missing_venvs_and_no_userns_are_refused(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """A venv whose checks did not pass, a missing venv and a host without user namespaces never run."""
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    world.prefetch(manifest, checked=False)
    monkeypatch.setattr(sandbox, "sandbox_status", lambda: ("ok", ""))
    outcome = run_step(manifest.steps[0], "demo", world.config())
    assert (outcome.status, outcome.codes) == ("failed", ("E_ENV_UNCHECKED",))
    assert outcome.key is not None and not (world.tmp / "results" / outcome.key).exists()
    path = env.venv_dir(world.cache, "demo", manifest.steps[0].env)
    (path / env.MARKER).unlink()
    assert run_step(manifest.steps[0], "demo", world.config()).codes == ("E_ENV_MISSING",)
    monkeypatch.setattr(sandbox, "sandbox_status", lambda: ("no-userns", "unshare failed"))
    assert run_step(manifest.steps[0], "demo", world.config()).codes == ("E_NO_USERNS",)


def test_gpu_guard_and_driver_floor(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """A GPU step on a host without a GPU, or below ``min_driver``, is refused before it runs."""
    monkeypatch.setattr(sandbox, "sandbox_status", lambda: ("ok", ""))

    def runner(gpu: str) -> host.Runner:
        def probe(argv: Sequence[str]) -> tuple[int, str]:
            return (0, gpu) if argv[0] == "nvidia-smi" else (1, "")

        return probe

    step = world.step("prep", "prep", "{root:data}/in/a.txt", resources={"profile": "cpu", "min_driver": "560"})
    manifest = world.manifest(step)
    world.prefetch(manifest)
    no_gpu = run_step(manifest.steps[0], "demo", world.config(host_runner=runner("")))
    assert no_gpu.codes == ("E_GPU_MISSING",)
    old = run_step(manifest.steps[0], "demo", world.config(host_runner=runner("Test GPU, 550.54.15\n")))
    assert old.codes == ("E_DRIVER_OLD",)


# --- adapters, logs, host values ------------------------------------------------------------------


@needs_userns
@pytest.mark.parametrize(
    "line",
    [
        "import senselab.utils.file_lock",
        "from senselab.utils import fastio",
        "import importlib; importlib.import_module('senselab.utils.fastio')",
        "import senselab",
    ],
)
def test_an_adapter_may_import_only_the_reproduce_package(world: World, line: str) -> None:
    """Importing any senselab module outside ``senselab.utils.reproduce`` fails the step with ``E_ADAPTER_IMPORT``."""
    (world.base / "adapter.py").write_text(f"{line}\n" + ADAPTER)
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.codes == ("E_ADAPTER_IMPORT",) and outcome.exit != 0
    assert outcome.out is not None and not (outcome.out / run.METRICS).exists()


@needs_userns
def test_adapter_output_is_validated_and_driver_keys_refused(world: World) -> None:
    """An adapter writing a driver-only key fails the step with the metrics code."""
    (world.base / "adapter.py").write_text(ADAPTER.replace('"dendf": 10', '"dendf": 10, "pass": True'))
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.codes == ("E_METRICS_DRIVER_KEY",) and outcome.exit != 0


@needs_userns
def test_a_mutated_input_fails_the_step(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """A changed input digest after the step gives ``E_INPUT_MUTATED`` and no metrics."""
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    monkeypatch.setattr(digest, "digest_unchanged", lambda before, after: False)
    outcome = world.run(manifest)["prep"]
    assert outcome.codes == ("E_INPUT_MUTATED",) and outcome.exit != 0
    assert outcome.out is not None and not (outcome.out / run.METRICS).exists()


@needs_userns
def test_the_senselab_logger_goes_to_the_driver_log_only(
    world: World, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str], caplog: pytest.LogCaptureFixture
) -> None:
    """A warning logged while the claim is taken (the stale-takeover case) lands only in ``driver.log``."""
    real = cache.claim

    @contextlib.contextmanager
    def noisy(results_root: Path, key: str, **kwargs: Any) -> Iterator[tuple[Path, bool]]:  # noqa: ANN401
        logging.getLogger("senselab").warning("stale takeover LOCKPATH-5512")
        with real(results_root, key, **kwargs) as held:
            yield held

    monkeypatch.setattr(cache, "claim", noisy)
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    capfd.readouterr()
    outcome = world.run(manifest)["prep"]
    captured = capfd.readouterr()
    assert outcome.ok, outcome.codes
    assert "LOCKPATH-5512" not in captured.out + captured.err
    assert "LOCKPATH-5512" not in caplog.text
    assert outcome.out is not None
    assert "LOCKPATH-5512" in (outcome.out / run.LOGS / run.DRIVER_LOG).read_text()


@needs_userns
def test_off_pattern_host_values_are_dropped_with_a_warning(world: World) -> None:
    """A host value provenance would refuse is recorded as null, with ``W_HOST_VALUE_DROPPED``; the step passes."""

    def probe(argv: Sequence[str]) -> tuple[int, str]:
        return (0, "GPU/with/slashes, 550.1\n") if argv[0] == "nvidia-smi" else host._run(argv)

    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    outcome = world.run(manifest, host_runner=probe)["prep"]
    assert outcome.ok and outcome.codes == ("W_HOST_VALUE_DROPPED",)
    assert outcome.out is not None and cache.is_done(outcome.out)
    store = (outcome.out / "prov" / "store.jsonl").read_text()
    assert "slashes" not in store and '"gpu_name": null' in store


@needs_userns
def test_provenance_records_the_lock_shas_and_no_path(world: World) -> None:
    """The record carries the body lock sha and the input digest, and names no tmp path or data file."""
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.out is not None
    text = "".join(p.read_text() for p in (outcome.out / "prov").iterdir())
    assert SHA in text and input_set_digest(world.data, ["in/a.txt"])[0] in text
    assert str(world.tmp) not in text and "a.txt" not in text


# --- pure parts -----------------------------------------------------------------------------------


def _schema_step(**change: Any) -> StepSpec:  # noqa: ANN401
    data: dict[str, Any] = {
        "name": "s",
        "repo": "r",
        "commit": "0" * 40,
        "kind": "python",
        "env": {
            "python": "3.12",
            "python_exact": "3.12.11",
            "torch": None,
            "body": {"lock": "b.lock", "lock_sha256": SHA, "source": "uv-export", "export_cmd": ["uv"]},
            "exclude_newer": "2026-01-01",
        },
        "argv": ["x"],
        "resources": {"profile": "cpu"},
        "adapter": "a.py",
        "hf_weights": [],
        "restricted": False,
        "metrics_schema": {
            "kind": "classification",
            "headline": ["balanced_accuracy"],
            "secondary": ["top1"],
            "counts": ["n_clips"],
            "per_class": ["a", "b"],
        },
    }
    data.update(change)
    return StepSpec.model_validate(data)


def _metrics(ba: float, top1: float, unit: str = "fraction") -> dict[str, Any]:
    scale = 100.0 if unit == "pp" else 1.0

    def entry(name: str, value: float, floor: float) -> dict[str, Any]:
        return {"name": name, "value": value, "floor": floor * scale, "lift": value - floor * scale, "unit": unit}

    return {
        "headline": [entry("balanced_accuracy", ba, 0.5)],
        "secondary": [entry("top1", top1, 0.6)],
        "per_class": {
            "a": {"recall": 0.7, "precision": 0.7, "prevalence": 0.6},
            "b": {"recall": 0.5, "precision": 0.5, "prevalence": 0.4},
        },
        "counts": {"n_clips": 40},
    }


def test_evaluate_expect_abs_exact_and_units() -> None:
    """``abs`` uses the tolerance; ``exact`` equality; a ``pp`` expectation converts a fraction metric."""
    step = _schema_step(
        expect=[
            {"metric": "balanced_accuracy", "value": 62.0, "tolerance": 0.5, "unit": "pp", "kind": "abs"},
            {"metric": "top1", "value": 0.7, "kind": "exact"},
            {"metric": "counts.n_clips", "value": 40, "kind": "exact"},
        ],
        needs=[],
    )
    final, passed, codes = evaluate_expect(step, _metrics(0.623, 0.7), {})
    assert codes == [] and passed is True and final is not None
    ba = final["headline"][0]
    assert ba["target"] == pytest.approx(0.62) and ba["delta"] == pytest.approx(0.003) and ba["pass"] is True
    assert ba["tolerance"] == pytest.approx(0.005)
    assert final["secondary"][0]["pass"] is True and final["pass"] is True
    assert final["target"] == 40 and final["delta"] == 0 and "tolerance" not in final
    _, failed, _ = evaluate_expect(step, _metrics(0.63, 0.7), {})
    assert failed is False


def test_evaluate_expect_from_step_and_tolerance_from() -> None:
    """A target from an upstream metric and a tolerance of ``max(a, k * sd)`` over calibration steps."""
    step = _schema_step(
        needs=["t", "c1", "c2", "c3"],
        expect=[
            {
                "metric": "balanced_accuracy",
                "from_step": {"step": "t", "metric": "balanced_accuracy"},
                "tolerance_from": {"steps": ["c1", "c2", "c3"], "rule": "max(a, k*sd)", "a": 0.5, "k": 2},
                "unit": "pp",
                "kind": "abs",
            }
        ],
    )
    upstream = {
        "t": _metrics(0.60, 0.7),
        "c1": _metrics(58.0, 70.0, "pp"),
        "c2": _metrics(60.0, 70.0, "pp"),
        "c3": _metrics(62.0, 70.0, "pp"),
    }
    final, passed, codes = evaluate_expect(step, _metrics(0.635, 0.7), upstream)
    assert codes == [] and final is not None
    entry = final["headline"][0]
    assert entry["target"] == pytest.approx(0.60) and entry["tolerance"] == pytest.approx(0.04)
    assert entry["delta"] == pytest.approx(0.035) and passed is True
    _, failed, _ = evaluate_expect(step, _metrics(0.645, 0.7), upstream)
    assert failed is False


def test_evaluate_expect_shared_place_gets_pass_only_and_bad_address_is_a_code() -> None:
    """Two expectations on one class entry write only ``pass``; a suppressed source value is a fixed code."""
    step = _schema_step(
        expect=[
            {"metric": "per_class.a.recall", "value": 0.7, "tolerance": 0.01, "kind": "abs"},
            {"metric": "per_class.a.precision", "value": 0.1, "tolerance": 0.01, "kind": "abs"},
        ]
    )
    final, passed, _ = evaluate_expect(step, _metrics(0.6, 0.7), {})
    assert final is not None and passed is False
    assert final["per_class"]["a"]["pass"] is False and "target" not in final["per_class"]["a"]
    counted = _schema_step(expect=[{"metric": "counts.n_clips", "value": 40, "kind": "exact"}])
    suppressed = _metrics(0.6, 0.7) | {"counts": {"n_clips": "<5"}}
    assert evaluate_expect(counted, suppressed, {}) == (None, False, ["E_METRICS_SUPPRESSED"])


def test_preamble_parsing_is_strict() -> None:
    """Only one JSON object line with exactly the six keys and the right types is accepted."""
    good = {
        "sys_prefix": "/v",
        "python": "3.12.11",
        "torch": "2.5.1+cu128",
        "torch_cuda": "12.8",
        "canary_failed": True,
        "interpreter_ok": True,
    }
    assert run._parse_preamble((json.dumps(good) + "\n").encode()) == good
    for bad in (
        json.dumps(good) + "\n" + json.dumps(good) + "\n",
        json.dumps(good | {"extra": 1}),
        json.dumps({k: v for k, v in good.items() if k != "python"}),
        json.dumps(good | {"canary_failed": "true"}),
        json.dumps(good | {"torch": "2.5 /x"}),
        json.dumps(good | {"interpreter_ok": 1}),
        "not json",
        "",
    ):
        assert run._parse_preamble(bad.encode()) is None, bad
    assert run._parse_preamble(b"\xff\xfe") is None


def test_traceback_gives_class_and_checkout_relative_frame(tmp_path: Path) -> None:
    """The last frame inside the checkout and the exception class; frames outside it are not reported."""
    checkout = tmp_path / "wt"
    (checkout / "pkg").mkdir(parents=True)
    stderr = tmp_path / "err.log"
    stderr.write_text(
        "noise\nTraceback (most recent call last):\n"
        f'  File "{checkout}/pkg/run.py", line 12, in <module>\n    main()\n'
        '  File "/usr/lib/python3.12/json/__init__.py", line 346, in loads\n'
        "json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (subject X)\n"
    )
    assert run._traceback(stderr, checkout) == ("json.decoder.JSONDecodeError", "pkg/run.py:12")
    stderr.write_text("no traceback here\n")
    assert run._traceback(stderr, checkout) == (None, None)
    for last in ("subject_X42 recorded at clinic: boom", "notAClass: x", "Error: x"):
        stderr.write_text(f'Traceback (most recent call last):\n  File "{checkout}/pkg/run.py", line 3\n{last}\n')
        assert run._traceback(stderr, checkout) == (None, "pkg/run.py:3"), last
    stderr.write_text("Traceback (most recent call last):\nKeyboardInterrupt\n")
    assert run._traceback(stderr, checkout) == ("KeyboardInterrupt", None)


def test_exact_after_a_unit_conversion_compares_within_a_relative_tolerance() -> None:
    """0.7 as a fraction equals an exact 70 pp, and the result is written back as a fraction."""
    step = _schema_step(expect=[{"metric": "top1", "value": 70.0, "unit": "pp", "kind": "exact"}])
    final, passed, _ = evaluate_expect(step, _metrics(0.6, 0.7), {})
    assert passed is True and final is not None
    entry = final["secondary"][0]
    assert entry["target"] == pytest.approx(0.7) and entry["pass"] is True and "tolerance" not in entry
    same_unit = _schema_step(expect=[{"metric": "top1", "value": 0.7000000001, "kind": "exact"}])
    assert evaluate_expect(same_unit, _metrics(0.6, 0.7), {})[1] is False


# --- the read-only-by-default sandbox and the driver-owned layout ----------------------------------


def _py(world: World, code: str, *extra: str, **change: Any) -> dict[str, Any]:  # noqa: ANN401
    """A step running ``python -c code {out} {cache} <extra>``; ``code`` must write ``{out}/result.json``."""
    escaped = code.replace("{", "{{").replace("}", "}}")
    argv = [
        "{venv}/bin/python",
        "-c",
        escaped,
        "{out}",
        "{cache}",
        *(e.replace("{", "{{").replace("}", "}}") for e in extra),
    ]
    return world.step("prep", "prep", "{root:data}/in/a.txt", argv=argv, **change)


RESULT = "open(sys.argv[1] + '/result.json', 'w').write(json.dumps({'value': 2.0}))"


@needs_userns
@pytest.mark.parametrize("target", ["cache", "r_lib", "home", "senselab", "scratch"])
def test_writes_outside_the_writable_dirs_fail(world: World, target: str) -> None:
    """The cache, an R library, the real home, the senselab source and any other path are read-only."""
    r_lib = world.cache / "r" / "lib"
    r_lib.mkdir(parents=True)
    (world.tmp / "scratch").mkdir()
    places = {
        "cache": world.cache,
        "r_lib": r_lib,
        "home": Path.home(),
        "senselab": Path(senselab.__file__).resolve().parent,
        "scratch": world.tmp / "scratch",
    }
    probe = places[target] / f".reproduce-probe-{uuid.uuid4().hex}"
    code = f"import json, sys; open({str(probe)!r}, 'w').write('x'); {RESULT}"
    manifest = world.manifest(_py(world, code))
    world.prefetch(manifest)
    try:
        outcome = world.run(manifest)["prep"]
        assert outcome.status == "executed" and outcome.exit != 0 and "E_STEP_FAILED" in outcome.codes
        assert outcome.out is not None and _summary(outcome)["exc_class"] == "OSError"
        assert not probe.exists()
    finally:
        probe.unlink(missing_ok=True)


@needs_userns
def test_the_step_cannot_touch_driver_owned_files(world: World) -> None:
    """Logs, the result dir's own files and the dir itself are read-only: a symlink escape or forgery fails."""
    victim = world.tmp / "victim.txt"
    victim.write_text("ORIGINAL\n")
    code = (
        "import json, os, sys; r = os.path.dirname(sys.argv[1]); hits = []\n"
        f"for act in (lambda: os.symlink({str(victim)!r}, r + '/logs/adapter.stdout.log'),\n"
        "            lambda: os.unlink(r + '/logs/driver.log'),\n"
        "            lambda: open(r + '/summary.json', 'w').write('{\"exit\": 0}'),\n"
        "            lambda: open(r + '/metrics.json', 'w').write('{}'),\n"
        "            lambda: os.rmdir(r + '/tmp')):\n"
        "    try:\n"
        "        act(); hits.append(1)\n"
        "    except OSError:\n"
        "        hits.append(0)\n"
        "open(sys.argv[1] + '/hits.json', 'w').write(json.dumps(hits))\n"
        f"{RESULT}\n"
    )
    manifest = world.manifest(_py(world, code))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.ok, outcome.codes
    assert outcome.out is not None
    assert json.loads((outcome.out / run.WORK / "hits.json").read_text()) == [0, 0, 0, 0, 0]
    assert victim.read_text() == "ORIGINAL\n"
    assert (outcome.out / run.LOGS / run.DRIVER_LOG).is_file()
    assert not (outcome.out / run.LOGS / "adapter.stdout.log").is_symlink()


@needs_userns
def test_a_symlink_left_in_tmp_is_not_followed_by_the_cleanup(world: World) -> None:
    """The step's tmp is removed without following a link the step left there."""
    victim = world.tmp / "victim-dir"
    victim.mkdir()
    (victim / "keep.txt").write_text("x")
    code = f"import json, os, sys; os.symlink({str(victim)!r}, os.environ['TMPDIR'] + '/link'); {RESULT}"
    manifest = world.manifest(_py(world, code))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.ok, outcome.codes
    assert outcome.out is not None and not (outcome.out / run.TMP).exists()
    assert (victim / "keep.txt").read_text() == "x"


@needs_userns
def test_reserved_names_under_work_are_refused_and_deleted(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """A forged summary never makes the dir done, even right after the step; reserved names fail the step."""
    code = (
        "import json, os, sys; o = sys.argv[1]; os.makedirs(o + '/sub/run.json')\n"
        "open(o + '/summary.json', 'w').write(json.dumps({'exit': 0}))\n"
        "open(o + '/sub/metrics.json', 'w').write(json.dumps({'rows': ['SUBJECT-0042']}))\n"
        f"{RESULT}\n"
    )
    seen: dict[str, bool] = {}
    real = run._Execution._after

    def spy(self: run._Execution) -> None:
        seen["done"] = cache.is_done(self.result)
        real(self)

    monkeypatch.setattr(run._Execution, "_after", spy)
    manifest = world.manifest(_py(world, code))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert seen == {"done": False}
    assert outcome.codes == ("E_RESERVED_NAME",) and outcome.exit != 0
    assert outcome.out is not None and not cache.is_done(outcome.out)
    work = outcome.out / run.WORK
    assert not (work / "summary.json").exists() and not (work / "sub" / "metrics.json").exists()
    assert not (work / "sub" / "run.json").exists() and not (outcome.out / run.METRICS).exists()


@needs_userns
def test_a_step_cannot_write_its_upstream_result(world: World) -> None:
    """An upstream result dir is read-only for its dependents."""
    manifest = world.dag()
    world.prefetch(manifest)
    first = world.run(manifest)
    up = first["prep"].out
    assert up is not None
    before = (up / run.WORK / "result.json").read_text()
    code = "import sys; open(sys.argv[3] + '/result.json', 'w').write('tamper')"
    score = world.step(
        "score",
        "score",
        "{up:prep}/result.json",
        needs=["prep"],
        inputs=[{"root": "up:prep", "patterns": ["result.json"]}],
        argv=["{venv}/bin/python", "-c", code, "{out}", "{cache}", "{up:prep}"],
    )
    tampered = world.manifest(manifest.steps[0].model_dump(mode="json"), score)
    outcome = world.run(tampered)["score"]
    assert outcome.exit != 0 and "E_STEP_FAILED" in outcome.codes
    assert (up / run.WORK / "result.json").read_text() == before


@needs_userns
def test_offline_canary_with_a_local_listener(world: World) -> None:
    """A listener on the host's loopback: unreachable from the namespace, reached without it (spec C3)."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    try:
        world.canary = f"127.0.0.1:{listener.getsockname()[1]}"
        manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
        world.prefetch(manifest)
        exposed = world.run(manifest, no_netns=True)["prep"]
        assert set(exposed.codes) == {"E_CANARY", "E_NOT_REPORTABLE"} and exposed.status == "failed"
        assert world.run(manifest)["prep"].ok
    finally:
        listener.close()


@needs_userns
def test_summary_pass_is_none_without_expectations(world: World) -> None:
    """With no expectations nothing was evaluated, so ``pass`` is null and ``metrics.json`` has none."""
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.ok and outcome.out is not None
    assert _summary(outcome)["pass"] is None
    assert "pass" not in json.loads((outcome.out / run.METRICS).read_text())


# --- R, notebooks, timeouts, upstream metrics ----------------------------------------------------------


FAKE_RSCRIPT = """#!/bin/sh
# Rscript --vanilla <script> <r_lib> <out>
out="$4"
printf 'lme4\\t1.1-37\\n' > "$out/installed.tsv"
printf 'Matrix\\t@MATRIX@\\nlme4\\t1.1-37\\n' > "$out/namespaces.tsv"
printf '%s\\n' "$3" > "$out/lib_paths.txt"
printf 'R version 4.5.2 (2025-10-31)\\nother attached packages: lme4_1.1-37\\nloaded via a namespace: x\\n' \\
  > "$out/session_info.txt"
printf 'R version 4.5.2 (2025-10-31)\\n' > "$out/r_version.txt"
"""


def _r_step(world: World, session: dict[str, Any], matrix: str = "1.7-3") -> dict[str, Any]:
    r_lib = world.cache / "r" / "lme4"
    fake = world.tmp / "rbin"
    fake.mkdir(exist_ok=True)
    (fake / "Rscript").write_text(FAKE_RSCRIPT.replace("@MATRIX@", matrix))
    (fake / "Rscript").chmod(0o755)
    (fake / "rpy").write_text('#!/bin/sh\nexec /usr/bin/python3 "$@"\n')
    (fake / "rpy").chmod(0o755)
    r_lib.mkdir(parents=True, exist_ok=True)
    (r_lib / env.MODULE_ENV).write_text(json.dumps({"module": "R/4.5.2", "build_modules": [], "PATH": str(fake)}))
    (r_lib / env.SESSION).write_text(json.dumps({"key": {}, **session}))
    code = (
        "import json, os, sys\n"
        "keys = ('PATH', 'R_LIBS', 'R_LIBS_USER', 'R_LIBS_SITE')\n"
        "open(sys.argv[1] + '/env.json', 'w').write(json.dumps({k: os.environ.get(k) for k in keys}))\n"
        f"{RESULT}\n"
    )
    data = _py(world, code, kind="r")
    data["argv"][0] = "rpy"
    data["env"] |= {"r_packages": [{"name": "lme4", "version": "1.1-37"}], "r_snapshot": "2026-01-01"}
    data["env"]["r_lib"] = "{cache}/r/lme4"
    return data


def _session(world: World, r_version: str = "R version 4.5.2 (2025-10-31)") -> dict[str, Any]:
    return {
        "installed": {"lme4": "1.1-37"},
        "namespaces": {"Matrix": "1.7-3", "lme4": "1.1-37"},
        "lib_paths": [os.path.realpath(world.cache / "r" / "lme4")],
        "r_version": r_version,
        "session_info": ["R version 4.5.2 (2025-10-31)", "other attached packages: lme4_1.1-37"],
    }


@needs_userns
def test_an_r_step_checks_its_session_and_gets_the_module_env(world: World) -> None:
    """A matching session runs with the module's PATH and the R library; a different one is ``E_R_SESSION``."""
    manifest = world.manifest(_r_step(world, _session(world)))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.ok, outcome.codes
    assert outcome.out is not None
    seen = json.loads((outcome.out / run.WORK / "env.json").read_text())
    r_lib = str(world.cache / "r" / "lme4")
    assert seen["R_LIBS"] == seen["R_LIBS_USER"] == os.path.realpath(r_lib) and seen["R_LIBS_SITE"] == ""
    assert str(world.tmp / "rbin") in seen["PATH"].split(":")
    assert not (outcome.out / run.R_SESSION_DIR).exists()

    moved = world.manifest(_r_step(world, _session(world, "R version 4.4.0 (2024-04-24)")))
    refused = world.run(moved)["prep"]
    assert refused.codes == ("E_R_SESSION",) and refused.status == "failed" and refused.exit != 0
    assert refused.out is not None and not (refused.out / run.WORK / "result.json").exists()


@needs_userns
def test_a_notebook_step_runs_in_its_derived_venv(world: World) -> None:
    """A notebook kind runs with the ``notebook_tools`` derived venv as ``sys.prefix``."""
    data = world.step("prep", "prep", "{root:data}/in/a.txt", kind="notebook")
    data["env"]["notebook_tools"] = {"lock": "nb.lock", "lock_sha256": "ef" * 32}
    manifest = world.manifest(data)
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.ok, outcome.codes
    assert _summary(outcome)["sys_prefix_ok"] is True
    assert outcome.out is not None and (outcome.out / run.WORK / "notebooks").is_dir()


@needs_userns
def test_a_step_past_its_timeout_is_killed(world: World) -> None:
    """``E_TIMEOUT``, a non-zero exit and nothing executed past the kill."""
    manifest = world.manifest(_py(world, f"import json, sys, time; time.sleep(60); {RESULT}"))
    world.prefetch(manifest)
    outcome = world.run(manifest, timeout=5)["prep"]
    assert "E_TIMEOUT" in outcome.codes and outcome.exit != 0
    assert outcome.out is not None and not (outcome.out / run.WORK / "result.json").exists()


def _tolerance_dag(world: World) -> Manifest:
    calibration = [world.step(name, "prep", "{root:data}/in/a.txt", replicate=i) for i, name in ((1, "c1"), (2, "c2"))]
    score = world.step(
        "score",
        "score",
        "{up:prep}/result.json",
        needs=["prep", "c1", "c2"],
        inputs=[{"root": "up:prep", "patterns": ["result.json"]}],
        expect=[
            {
                "metric": "f_stat",
                "from_step": {"step": "prep", "metric": "f_stat"},
                "tolerance_from": {"steps": ["c1", "c2"], "rule": "max(a, k*sd)", "a": 2.5, "k": 2},
                "kind": "abs",
            }
        ],
    )
    return world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"), *calibration, score)


@needs_userns
def test_from_step_and_tolerance_from_read_upstream_results(world: World) -> None:
    """A target from an upstream step and a tolerance from calibration replicates, through their result dirs."""
    manifest = _tolerance_dag(world)
    world.prefetch(manifest)
    outcomes = world.run(manifest)
    assert all(o.ok for o in outcomes.values()), {n: o.codes for n, o in outcomes.items()}
    score = outcomes["score"].out
    assert score is not None
    entry = json.loads((score / run.METRICS).read_text())["headline"][0]
    assert (entry["target"], entry["tolerance"], entry["delta"], entry["pass"]) == (2.0, 2.5, 2.0, True)


def _band_dag(world: World, stored: float) -> Manifest:
    values = {"c1": 2.0, "c2": 3.0, "c3": 4.0, "gather": 3.0}
    for name, value in values.items():
        (world.data / "in" / f"{name}.txt").write_text(f"{value}\n")

    def step(name: str, **change: Any) -> dict[str, Any]:  # noqa: ANN401
        inputs = [{"root": "data", "patterns": [f"in/{name}.txt"]}]
        return world.step(name, "prep", f"{{root:data}}/in/{name}.txt", inputs=inputs, **change)

    band = {
        "stored": stored,
        "unit": "fraction",
        "floor": 0.0,
        "floor_source": "declared-zero",
        "chains": ["c1", "c2", "c3"],
    }
    gather = step("gather", needs=["c1", "c2", "c3"], expect=[{"metric": "f_stat", "kind": "seed_band", "band": band}])
    after = world.step(
        "after",
        "score",
        "{up:gather}/result.json",
        needs=["gather"],
        inputs=[{"root": "up:gather", "patterns": ["result.json"]}],
    )
    return world.manifest(step("c1"), step("c2"), step("c3"), gather, after)


@needs_userns
def test_a_band_gather_step_writes_verdict_and_exits_zero_unless_fail(world: World) -> None:
    """An ``uninformative`` band exits 0 with the verdict in the summary; a ``fail`` is ``E_EXPECT``."""
    manifest = _band_dag(world, 3.5)
    world.prefetch(manifest)
    outcomes = world.run(manifest)
    assert all(o.ok for o in outcomes.values()), {n: o.codes for n, o in outcomes.items()}
    gather = outcomes["gather"]
    summary = _summary(gather)
    assert summary["exit"] == 0 and summary["verdict"] == "uninformative" and summary["pass"] is False
    assert gather.out is not None
    entry = json.loads((gather.out / run.METRICS).read_text())["headline"][0]
    assert entry["verdict"] == "uninformative" and entry["tolerance"] == pytest.approx(2.0)
    assert _summary(outcomes["c1"])["verdict"] is None

    failing = _band_dag(world, 10.0)
    world.prefetch(failing)
    outcomes = world.run(failing)
    gather = outcomes["gather"]
    assert gather.status == "executed" and gather.exit != 0 and "E_EXPECT" in gather.codes
    assert _summary(gather)["verdict"] == "fail"
    assert outcomes["after"].status == "never-ran"


EXTRACT_ADAPTER = """\
import json, pathlib, sys
out, dest = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
value = json.loads((out / "result.json").read_text())["value"]
metrics = {"headline": [{"name": "f_stat", "value": value, "numdf": 1, "dendf": 10}]}
dest.write_text(json.dumps({"metrics": metrics, "extracted": EXTRACTED}))
"""
EXTRACT = [{"name": "matched", "kind": "yesno"}, {"name": "run_dir", "kind": "index", "allowed": ["run-00", "run-01"]}]


@needs_userns
def test_extracted_values_reach_summary_and_up_dir(world: World) -> None:
    """Clean values go to ``summary.json`` and ``work/extracted.json``; free text fails the step with ``E_EXTRACT``."""
    extracted = {"matched": "yes", "run_dir": 1}
    (world.base / "adapter_extract.py").write_text(EXTRACT_ADAPTER.replace("EXTRACTED", repr(extracted)))
    prep = world.step("prep", "prep", "{root:data}/in/a.txt", adapter="adapter_extract.py", extract=EXTRACT)
    after = world.step(
        "after",
        "score",
        "{up:prep}/result.json",
        needs=["prep"],
        inputs=[{"root": "up:prep", "patterns": ["extracted.json"]}],
    )
    manifest = world.manifest(prep, after)
    world.prefetch(manifest)
    outcomes = world.run(manifest)
    assert all(o.ok for o in outcomes.values()), {n: o.codes for n, o in outcomes.items()}
    out = outcomes["prep"].out
    assert out is not None
    assert _summary(outcomes["prep"])["extracted"] == extracted
    assert json.loads((out / run.WORK / run.EXTRACTED).read_text()) == extracted
    assert "extracted" not in json.loads((out / run.METRICS).read_text())
    assert _summary(outcomes["after"])["extracted"] is None

    secret = "see /x/y.csv"
    bad = {"matched": secret, "run_dir": 1}
    (world.base / "adapter_extract.py").write_text(EXTRACT_ADAPTER.replace("EXTRACTED", repr(bad)))
    outcomes = world.run(manifest)
    failed = outcomes["prep"]
    assert failed.status == "executed" and failed.exit != 0 and failed.codes == ("E_EXTRACT",)
    assert failed.out is not None and not (failed.out / run.WORK / run.EXTRACTED).exists()
    summary = _summary(failed)
    assert summary["extracted"] is None and secret not in json.dumps(summary)
    assert outcomes["after"].status == "never-ran"

    (world.base / "adapter_extract.py").write_text(EXTRACT_ADAPTER.replace("EXTRACTED", repr(extracted)))
    expect = [{"metric": "f_stat", "value": 100.0, "tolerance": 0.1, "kind": "abs"}]
    failing = world.manifest(
        world.step("prep", "prep", "{root:data}/in/a.txt", adapter="adapter_extract.py", extract=EXTRACT, expect=expect)
    )
    world.prefetch(failing)
    failed = world.run(failing)["prep"]
    assert failed.exit != 0 and "E_EXPECT" in failed.codes and failed.out is not None
    assert _summary(failed)["extracted"] is None and not (failed.out / run.WORK / run.EXTRACTED).exists()


@needs_userns
def test_a_step_cannot_write_extracted_json_itself(world: World) -> None:
    """A step that writes ``extracted.json`` under ``work/`` fails with ``E_RESERVED_NAME`` and the file is gone."""
    code = f"import json, sys; open(sys.argv[1] + '/extracted.json', 'w').write('{{}}'); {RESULT}"
    manifest = world.manifest(_py(world, code))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.codes == ("E_RESERVED_NAME",) and outcome.exit != 0
    assert outcome.out is not None and not (outcome.out / run.WORK / "extracted.json").exists()


@needs_userns
def test_an_unreadable_upstream_metrics_file_is_refused(world: World) -> None:
    """An expectation whose upstream ``metrics.json`` is gone gives ``E_UPSTREAM_METRICS``."""
    manifest = _tolerance_dag(world)
    world.prefetch(manifest)
    config = world.config()
    steps = {s.name: s for s in manifest.steps}
    ups = {n: run_step(steps[n], "demo", config) for n in ("prep", "c1", "c2")}
    assert all(o.ok for o in ups.values())
    prep_out = ups["prep"].out
    assert prep_out is not None
    (prep_out / run.METRICS).unlink()
    schemas = {n: steps[n].metrics_schema for n in ups}
    outcome = run_step(steps["score"], "demo", config, ups, schemas)
    assert outcome.codes == ("E_UPSTREAM_METRICS",) and outcome.exit != 0


@needs_userns
def test_a_missing_cwd_and_bad_stages_are_refused(world: World) -> None:
    """``E_CWD`` for a cwd that does not exist; ``E_STAGE_*`` for a stage matching nothing or leaving ``stage``."""
    cases: dict[str, dict[str, Any]] = {
        "E_CWD": {"cwd": "missing"},
        "E_STAGE_EMPTY": {"stage": [{"root": "checkout", "patterns": ["none*.py"], "dest": "x"}]},
        "E_STAGE_BAD_DEST": {"stage": [{"root": "checkout", "patterns": ["*.py"], "dest": "../x"}]},
    }
    for code, change in cases.items():
        manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt", **change))
        world.prefetch(manifest)
        outcome = world.run(manifest)["prep"]
        assert outcome.codes == (code,) and outcome.status == "failed" and outcome.exit != 0, code


@needs_userns
def test_a_venv_changed_during_the_step_fails_it(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """The venv's installed digest must be the same after the step as before (``E_ENV_CHANGED``)."""
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    real = env.installed_sha256
    calls: list[int] = []

    def drifting(path: Path) -> str | None:
        calls.append(1)
        return real(path) if len(calls) == 1 else "f" * 64

    monkeypatch.setattr(env, "installed_sha256", drifting)
    outcome = world.run(manifest)["prep"]
    assert outcome.codes == ("E_ENV_CHANGED",) and outcome.exit != 0


@needs_userns
def test_a_refused_provenance_record_fails_the_step(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """``E_PROV_REFUSED`` leaves the result uncached."""

    def refuse(*args: Any, **kwargs: Any) -> list[Path]:  # noqa: ANN401
        raise ValueError("record_step: host is not in its closed form")

    monkeypatch.setattr(provenance, "record_step", refuse)
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.codes == ("E_PROV_REFUSED",) and outcome.exit != 0
    assert outcome.out is not None and not cache.is_done(outcome.out)


@needs_userns
def test_a_driver_failure_is_a_code_with_a_summary_and_no_traceback(
    world: World, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    """An ``OSError`` while writing the summary is ``E_DRIVER``; a summary still lands and nothing is printed."""
    real = cache.write_summary
    calls: list[int] = []

    def flaky(out: Path, summary: dict[str, Any]) -> Path:
        calls.append(1)
        if len(calls) == 1:
            raise OSError(f"disk full at {out}")
        return real(out, summary)

    monkeypatch.setattr(cache, "write_summary", flaky)
    manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
    world.prefetch(manifest)
    capfd.readouterr()
    outcome = world.run(manifest)["prep"]
    captured = capfd.readouterr()
    assert "E_DRIVER" in outcome.codes and outcome.exit != 0
    assert "Traceback" not in captured.out + captured.err and str(world.tmp) not in captured.out + captured.err
    assert outcome.out is not None and "E_DRIVER" in _summary(outcome)["codes"]


@needs_userns
def test_stale_run_links_are_removed_for_steps_that_did_not_run(world: World) -> None:
    """A never-ran or keyless failed step leaves no same-day link to an older result."""
    manifest = world.dag()
    world.prefetch(manifest)
    world.run(manifest)
    run_dir = world.tmp / "runs" / world.config().date.strftime("%Y-%m-%d") / "demo"
    assert (run_dir / "score").is_symlink()
    failing = world.dag(prep_expect=[{"metric": "f_stat", "value": 99.0, "tolerance": 0.5, "kind": "abs"}])
    world.prefetch(failing)
    outcomes = world.run(failing)
    assert outcomes["score"].status == "never-ran" and not os.path.lexists(run_dir / "score")
    assert (run_dir / "prep").is_symlink()
    orphan = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt", commit="1" * 40))
    assert world.run(orphan)["prep"].codes == ("E_SRC_MISSING",)
    assert not os.path.lexists(run_dir / "prep")


@needs_userns
def test_a_namespace_updated_in_place_fails_the_r_session_check(world: World) -> None:
    """A namespace-only package (Matrix) at another version than at prefetch is ``E_R_SESSION``."""
    manifest = world.manifest(_r_step(world, _session(world), matrix="1.7-4"))
    world.prefetch(manifest)
    outcome = world.run(manifest)["prep"]
    assert outcome.codes == ("E_R_SESSION",) and outcome.status == "failed" and outcome.exit != 0


TERMINAL_PROBE = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from tests.utils.reproduce.run_test import World
world = World(Path(sys.argv[2]))
manifest = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt"))
world.prefetch(manifest)
day = world.tmp / "runs" / world.config().date.strftime("%Y-%m-%d")
day.mkdir(parents=True)
day.chmod(0o500)
try:
    outcome = world.run(manifest)["prep"]
finally:
    day.chmod(0o700)
log = (outcome.out / "logs" / "driver.log").read_text() if outcome.out else ""
print(json.dumps({"codes": list(outcome.codes), "logged": "Traceback" in log}))
"""


@needs_userns
@pytest.mark.skipif(os.geteuid() == 0, reason="root writes regardless of mode")
def test_a_driver_error_prints_nothing_on_the_terminal(tmp_path: Path) -> None:
    """With no logging plugin in the way, an unwritable runs dir gives ``E_DRIVER`` and an empty stderr.

    Run in a subprocess, because pytest's logging plugin installs handlers that hide
    `logging.lastResort`, which is what would print a stray traceback to stderr.
    """
    src = Path(__file__).resolve().parents[3]
    env_vars = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "LANG": "C.UTF-8"}
    if "LD_LIBRARY_PATH" in os.environ:
        env_vars["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
    done = subprocess.run(
        [sys.executable, "-c", TERMINAL_PROBE, str(src), str(tmp_path / "w")],
        env=env_vars,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert "E_DRIVER" in result["codes"] and result["logged"] is True
    assert "Traceback" not in done.stderr and str(tmp_path) not in done.stderr


def test_the_registry_gate_refuses_a_root_path_no_lock_input_covers(world: World) -> None:
    """A step whose ``{root:X}`` path only a ``lock_inputs`` tree could cover is checked against the registry."""
    world.registered["t"] = register_input(world.base, {"data": world.data}, "t", root="data", relpath="in")
    covered = world.manifest(world.step("prep", "prep", "{root:data}/in/a.txt", inputs=[], lock_inputs=["t"]))
    restricted.check_manifest_set([covered], world.base, world.profile())
    outside = world.manifest(world.step("prep", "prep", "{root:data}/other/a.txt", inputs=[], lock_inputs=["t"]))
    with pytest.raises(ValueError, match="E_SPEC_UNDECLARED_ROOT"):
        restricted.check_manifest_set([outside], world.base, world.profile())
    assert run_dry_plan_refuses(outside, world)


def run_dry_plan_refuses(manifest: Manifest, world: World) -> bool:
    """Whether `run.run_steps` refuses ``manifest`` with ``E_SPEC_UNDECLARED_ROOT`` before running anything."""
    try:
        run_steps([manifest], world.config())
    except ValueError as error:
        return "E_SPEC_UNDECLARED_ROOT" in str(error)
    return False


LIST_CODE = (
    "import json, os, sys; out = sys.argv[1]; "
    "open(os.path.join(out, 'names.txt'), 'w').write(' '.join(sorted(os.listdir({path!r})))); "
    "json.dump({{{{'value': 2.0}}}}, open(os.path.join(out, 'result.json'), 'w'))"
)


@needs_userns
def test_a_step_sees_only_the_roots_it_declares(world: World) -> None:
    """A root behind none of the step's inputs, stage, lock_inputs or hub cache is not bound, and is masked.

    Both steps list the other root through a literal path: the step that declares an input there sees its
    entries; the step that does not sees an empty dir.
    """
    other = world.tmp / "other"
    (other / "in").mkdir(parents=True)
    (other / "in" / "b.txt").write_text("3\n")
    world.extra_roots = {"other": str(other)}
    code = LIST_CODE.format(path=str(other))
    argv = ["{venv}/bin/python", "-c", code, "{out}"]
    blind = world.step("blind", "prep", "", argv=argv)
    seeing = world.step(
        "seeing",
        "prep",
        "",
        argv=argv,
        inputs=[{"root": "data", "patterns": ["in/a.txt"]}, {"root": "other", "patterns": ["in/b.txt"]}],
    )
    manifest = world.manifest(blind, seeing)
    world.prefetch(manifest)
    outcomes = world.run(manifest)
    assert all(o.ok for o in outcomes.values()), {n: o.codes for n, o in outcomes.items()}
    listed = {}
    for name, outcome in outcomes.items():
        assert outcome.out is not None
        listed[name] = (outcome.out / run.WORK / "names.txt").read_text()
    assert listed == {"blind": "", "seeing": "in"}
