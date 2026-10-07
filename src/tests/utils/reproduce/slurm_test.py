"""Tests for batch rendering, the preemption probe and ``afterok`` submission chains.

``sbatch`` is always faked here, except in the one ``network`` test, which runs a real ``--test-only``
probe and submits nothing. Profiles, manifests and run trees live in ``tmp_path``.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from senselab.utils.reproduce import run, slurm
from senselab.utils.reproduce.spec import Manifest, Profile, StepSpec

SHA = "ab" * 32
COMMIT = "c" * 40
DATE = datetime.date(2026, 3, 4)
EXPORTS = {
    "UV_CACHE_DIR": "/cache/uv",
    "UV_LINK_MODE": "hardlink",
    "UV_PYTHON_INSTALL_DIR": "/cache/py",
    "SENSELAB_REPRO_CACHE": "/cache/repro",
}


def _profile(tmp: Path, **change: Any) -> Profile:  # noqa: ANN401
    data: dict[str, Any] = {
        "roots": {"data": str(tmp / "data"), "extra": str(tmp / "extra")},
        "results_root": str(tmp / "results"),
        "runs_root": str(tmp / "runs"),
        "slurm_logs_root": str(tmp / "slurm"),
        "partitions": {
            "cpu": {"partition": "main", "cpus": 4, "mem": "8G", "time": "01:00:00", "fallback": "spare"},
            "gpu": {"partition": "accel", "gres": "gpu:a:1", "cpus": 8, "mem": "32G", "time": "02:00:00"},
            "build": {"partition": "quick", "cpus": 2, "mem": "4G", "time": "00:15:00"},
        },
        "build": "build",
        "deny_patterns_file": str(tmp / "deny.txt"),
        "canary_host": "192.0.2.1:9",
    }
    data.update(change)
    return Profile.model_validate(data)


def _step(name: str, **change: Any) -> dict[str, Any]:  # noqa: ANN401
    data: dict[str, Any] = {
        "name": name,
        "repo": "https://git.example/org/analysis.git",
        "commit": COMMIT,
        "kind": "python",
        "env": {
            "python": "3.12",
            "python_exact": "3.12.11",
            "torch": None,
            "body": {"lock": "b.lock", "lock_sha256": SHA, "source": "uv-export", "export_cmd": ["uv"]},
            "exclude_newer": "2026-01-01",
        },
        "argv": ["{venv}/bin/python", "{checkout}/step.py", "{root:data}/in", "{out}"],
        "inputs": [{"root": "data", "patterns": ["in/*.txt"]}],
        "resources": {"profile": "cpu"},
        "adapter": "adapter.py",
        "hf_weights": [],
        "metrics_schema": {"kind": "statistic", "headline": ["f_stat"]},
    }
    data.update(change)
    return data


def _spec(name: str, **change: Any) -> StepSpec:  # noqa: ANN401
    return StepSpec.model_validate(_step(name, **change))


def _manifest(name: str, *steps: dict[str, Any]) -> Manifest:
    return Manifest.model_validate({"name": name, "steps": list(steps)})


def _render(tmp: Path, step: StepSpec, **change: Any) -> str:  # noqa: ANN401
    kwargs: dict[str, Any] = {
        "partition": "main",
        "command": ["run", "/m/a.yaml", "--step", step.name],
        "date": DATE,
        "exports": EXPORTS,
        "driver_python": "/opt/py/bin/python",
    }
    kwargs.update(change)
    return slurm.render_step_job(step, _profile(tmp), **kwargs)


def _sbatch_lines(text: str) -> list[str]:
    return [line.removeprefix("#SBATCH ") for line in text.splitlines() if line.startswith("#SBATCH ")]


# --- rendering ---------------------------------------------------------------------------------------


def test_render_step_job_flags_and_log_paths(tmp_path: Path) -> None:
    """Partition limits, dated ``%j`` logs, an empty exported env, the afterok chain and kill-on-invalid-dep."""
    text = _render(tmp_path, _spec("prep"), dependencies=["11", "12"])
    flags = _sbatch_lines(text)
    logs = tmp_path / "slurm" / "2026-03-04"
    assert flags == [
        "--job-name=reproduce-prep",
        "--partition=main",
        "--cpus-per-task=4",
        "--mem=8G",
        "--time=01:00:00",
        f"--output={logs}/%j.out",
        f"--error={logs}/%j.err",
        "--export=NONE",
        "--dependency=afterok:11:12",
        "--kill-on-invalid-dep=yes",
    ]
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "set -euo pipefail" in lines and "umask 007" in lines
    assert lines[-1] == '"$REPRO_DRIVER_PYTHON" -m senselab.utils.reproduce run /m/a.yaml --step prep'


def test_render_step_job_gres_and_no_dependency(tmp_path: Path) -> None:
    """A GPU partition adds ``--gres``; a step with no needs has no ``--dependency`` line."""
    text = _render(tmp_path, _spec("train", resources={"profile": "gpu", "gpu_arch": ["sm_80"]}), partition="accel")
    flags = _sbatch_lines(text)
    assert "--gres=gpu:a:1" in flags and "--partition=accel" in flags and "--cpus-per-task=8" in flags
    assert not any(f.startswith("--dependency") for f in flags)
    assert "--kill-on-invalid-dep=yes" in flags


def test_render_step_job_exports_every_variable_explicitly(tmp_path: Path) -> None:
    """The uv, link-mode, Python-dir and cache variables are exported; nothing else is inherited."""
    text = _render(tmp_path, _spec("prep"))
    lines = text.splitlines()
    for key, value in EXPORTS.items():
        assert f"export {key}={value}" in lines
    assert "export PATH=/usr/bin:/bin" in lines
    assert not any(line.startswith("export LD_LIBRARY_PATH") for line in lines)
    assert "export REPRO_DRIVER_PYTHON=/opt/py/bin/python" in lines


def test_batch_exports_reads_uv_dirs_and_adds_native_libs(tmp_path: Path) -> None:
    """``batch_exports`` takes the uv dirs from the caller, pins hardlinks, names the cache and the native libs."""
    native = {"ffmpeg": {"bin_dirs": [str(tmp_path / "ff" / "bin")], "lib_dirs": [str(tmp_path / "ff" / "lib")]}}
    profile = _profile(tmp_path, native=native)
    environ = {"UV_CACHE_DIR": "/c/uv", "UV_PYTHON_INSTALL_DIR": "/c/py", "UV_LINK_MODE": "copy", "OTHER": "x"}
    exports = slurm.batch_exports(profile, cache=Path("/c/repro"), environ=environ)
    assert exports == {
        "UV_CACHE_DIR": "/c/uv",
        "UV_LINK_MODE": "hardlink",
        "UV_PYTHON_INSTALL_DIR": "/c/py",
        "SENSELAB_REPRO_CACHE": "/c/repro",
        "LD_LIBRARY_PATH": str(tmp_path / "ff" / "lib"),
    }
    with pytest.raises(ValueError, match="UV_PYTHON_INSTALL_DIR"):
        slurm.batch_exports(profile, cache=Path("/c/repro"), environ={"UV_CACHE_DIR": "/c/uv"})
    text = _render(tmp_path, _spec("prep"), exports=exports)
    assert f"export LD_LIBRARY_PATH={tmp_path / 'ff' / 'lib'}" in text.splitlines()


def test_render_step_job_refuses_unsafe_values(tmp_path: Path) -> None:
    """A partition, dependency or exported value that would need unsafe quoting is refused."""
    step = _spec("prep")
    with pytest.raises(ValueError, match="partition"):
        _render(tmp_path, step, partition="main; rm -rf /")
    with pytest.raises(ValueError, match="dependency"):
        _render(tmp_path, step, dependencies=["12:13"])
    with pytest.raises(ValueError, match="export"):
        _render(tmp_path, step, exports={**EXPORTS, "BAD NAME": "x"})


def test_rendered_script_runs_in_an_empty_environment(tmp_path: Path) -> None:
    """Under ``env -i`` with ``set -u`` the driver sees exactly the exported variables and its argv."""
    stub = tmp_path / "python"
    stub.write_text('#!/bin/sh\nenv | sort > "$OUT_FILE.env"\nprintf "%s\\n" "$@" > "$OUT_FILE.argv"\n')
    stub.chmod(0o750)
    out = tmp_path / "seen"
    exports = {**EXPORTS, "OUT_FILE": str(out)}
    text = _render(tmp_path, _spec("prep"), exports=exports, driver_python=str(stub))
    job = f"pytest-slurm-{os.getpid()}"
    result = subprocess.run(
        ["env", "-i", f"SLURM_JOB_ID={job}", "/bin/bash", "--noprofile", "--norc", "-c", text],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    seen = dict(line.split("=", 1) for line in (tmp_path / "seen.env").read_text().splitlines() if "=" in line)
    for key, value in EXPORTS.items():
        assert seen[key] == value
    assert seen["PATH"] == "/usr/bin:/bin" and seen["TMPDIR"] == f"/dev/shm/reproduce-{job}"
    argv = (tmp_path / "seen.argv").read_text().splitlines()
    assert argv == ["-m", "senselab.utils.reproduce", "run", "/m/a.yaml", "--step", "prep"]
    assert not Path(f"/dev/shm/reproduce-{job}").exists()


# --- the preemption probe ----------------------------------------------------------------------------


class FakeSbatch:
    """Records each call; ``--test-only`` answers from ``probe``, ``--parsable`` returns increasing job ids."""

    def __init__(self, probe: tuple[int, str] = (0, ""), fail_at: int | None = None) -> None:
        """Set the probe answer and, optionally, the submission that fails."""
        self.calls: list[tuple[list[str], str]] = []
        self.probe = probe
        self.fail_at = fail_at
        self.next_id = 100

    def __call__(self, argv: list[str], script: str) -> subprocess.CompletedProcess[str]:
        """Answer one sbatch call."""
        self.calls.append((argv, script))
        if "--test-only" in argv:
            code, text = self.probe
            return subprocess.CompletedProcess(argv, code, "", text)
        if self.fail_at is not None and len(self.submissions()) == self.fail_at:
            return subprocess.CompletedProcess(argv, 1, "", "sbatch: error: denied")
        self.next_id += 1
        return subprocess.CompletedProcess(argv, 0, f"{self.next_id};cluster\n", "")

    def submissions(self) -> list[str]:
        """The scripts passed to ``--parsable`` submissions."""
        return [s for a, s in self.calls if "--parsable" in a]


PROBE_OK = "sbatch: Job 4242 to start at 2026-03-04T10:00:00 using 4 processors on nodes n1 in partition main"
PROBE_PREEMPTS = PROBE_OK + "\nsbatch:   Preempts: 4100,4101"


@pytest.mark.parametrize(
    ("probe", "expected"),
    [((0, PROBE_OK), "main"), ((0, PROBE_PREEMPTS), "spare"), ((1, "sbatch: error: invalid partition"), "spare")],
)
def test_choose_partition_probes_and_falls_back(tmp_path: Path, probe: tuple[int, str], expected: str) -> None:
    """A probe that reports preemption, or fails, sends the job to the fallback partition."""
    fake = FakeSbatch(probe)
    chosen = slurm.choose_partition(_spec("prep"), _profile(tmp_path), lambda p: f"#!/bin/bash\n# {p}\n", fake)
    assert chosen == expected
    ((argv, script),) = fake.calls
    assert argv[-1] == "--test-only" and "# main" in script


def test_no_probe_without_a_fallback(tmp_path: Path) -> None:
    """A partition with no fallback is used as is, with no probe."""
    fake = FakeSbatch((0, PROBE_PREEMPTS))
    step = _spec("train", resources={"profile": "gpu"})
    assert slurm.choose_partition(step, _profile(tmp_path), lambda p: p, fake) == "accel"
    assert fake.calls == []


def test_probe_preempts_parses_the_will_run_answer() -> None:
    """Only a ``Preempts:`` line counts."""
    assert slurm.probe_preempts(PROBE_PREEMPTS) is True
    assert slurm.probe_preempts(PROBE_OK) is False


# --- resource and root cross-checks --------------------------------------------------------------


def test_resource_and_root_findings(tmp_path: Path) -> None:
    """Unknown partition keys, GPU steps without gres, unknown native entries and roots are each reported."""
    profile = _profile(tmp_path)
    assert slurm.step_findings(_spec("ok"), profile) == []
    bad = _spec(
        "bad",
        argv=["{venv}/bin/python", "{root:nowhere}/x", "{out}"],
        inputs=[{"root": "missing", "patterns": ["a"]}, {"root": "elsewhere", "patterns": ["b"]}],
        stage=[{"root": "elsewhere", "patterns": ["b"], "dest": "s"}],
        cwd="{root:cwdroot}",
        resources={"profile": "cpu", "gpu_arch": ["sm_80"], "native": ["ffmpeg"]},
    )
    codes = sorted({f.code for f in slurm.step_findings(bad, profile)})
    assert codes == ["gpu-no-gres", "native-unknown", "root-unknown"]
    details = " ".join(f.detail for f in slurm.step_findings(bad, profile) if f.code == "root-unknown")
    for name in ("nowhere", "missing", "elsewhere", "cwdroot"):
        assert name in details
    assert [f.code for f in slurm.step_findings(_spec("p", resources={"profile": "tpu"}), profile)] == [
        "partition-unknown"
    ]
    looping = _profile(
        tmp_path,
        partitions={
            "cpu": {"partition": "main", "cpus": 1, "mem": "1G", "time": "1:00", "fallback": "main"},
            "build": {"partition": "quick", "cpus": 1, "mem": "1G", "time": "1:00"},
        },
    )
    assert [f.code for f in slurm.step_findings(_spec("ok"), looping)] == ["fallback-invalid"]


# --- submission chains -----------------------------------------------------------------------------


def _config(tmp: Path, paths: Sequence[Path]) -> slurm.SubmitConfig:
    return slurm.SubmitConfig(
        profile=_profile(tmp),
        profile_path=tmp / "profile.yaml",
        manifest_paths=list(paths),
        base_dir=tmp / "manifests",
        cache=tmp / "cache",
        date=DATE,
        exports=EXPORTS,
        driver_python="/opt/py/bin/python",
    )


def _chain() -> list[Manifest]:
    first = _manifest("alpha", _step("prep"), _step("score", needs=["prep"]))
    second = _manifest("beta", _step("report", needs=["score", "prep"]), _step("lone"))
    return [first, second]


def _rows(tmp: Path, analysis: str) -> list[dict[str, Any]]:
    path = tmp / "runs" / DATE.isoformat() / analysis / run.RUN_JSON
    rows: list[dict[str, Any]] = json.loads(path.read_text())["steps"]
    return rows


def test_submit_chains_jobs_with_afterok_in_dag_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One job per step in DAG order, each depending on its needs' job ids, then a finalize job after all."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    fake = FakeSbatch((0, PROBE_OK))
    submission = slurm.submit(_chain(), _config(tmp_path, [Path("/m/a.yaml"), Path("/m/b.yaml")]), runner=fake)
    jobs = submission.jobs
    assert submission.finalize == "105"
    assert [(j.step, j.job_id) for j in jobs] == [("lone", "101"), ("prep", "102"), ("score", "103"), ("report", "104")]
    scripts = fake.submissions()
    assert len(scripts) == 5
    deps = [[f for f in _sbatch_lines(s) if f.startswith("--dependency")] for s in scripts]
    assert deps == [[], [], ["--dependency=afterok:102"], ["--dependency=afterok:103:102"], deps[4]]
    assert deps[4] == ["--dependency=afterany:101:102:103:104"]
    assert "--kill-on-invalid-dep=yes" not in _sbatch_lines(scripts[4])
    assert scripts[4].splitlines()[-1].endswith("--finalize")
    for job, script in zip(jobs, scripts):
        last = script.splitlines()[-1]
        assert f"--step {job.step}" in last and "--date 2026-03-04" in last
        assert "/m/a.yaml /m/b.yaml" in last and "--cache" in last and "--base-dir" in last
    assert os.listdir(cwd) == []
    assert (tmp_path / "slurm" / "2026-03-04").is_dir()
    alpha = _rows(tmp_path, "alpha")
    assert [(r["step"], r["status"], r["job"], r["key"]) for r in alpha] == [
        ("prep", "pending", "102", None),
        ("score", "pending", "103", None),
    ]
    assert [(r["step"], r["job"]) for r in _rows(tmp_path, "beta")] == [("lone", "101"), ("report", "104")]


def test_submit_does_not_overwrite_a_row_a_job_already_wrote(tmp_path: Path) -> None:
    """A job that finished before submit recorded its id keeps its own row."""
    config = _config(tmp_path, [Path("/m/a.yaml")])
    manifests = [_manifest("alpha", _step("prep"))]

    def racing(argv: list[str], script: str) -> subprocess.CompletedProcess[str]:
        if "--parsable" in argv and "--finalize" not in script:
            row = {"step": "prep", "key": "f" * 64, "status": "executed", "exit": 0, "codes": []}
            run.update_run_record(config.profile.runs_root, DATE, "alpha", [row], order=["prep"])
        return subprocess.CompletedProcess(argv, 0, "7\n", "")

    slurm.submit(manifests, config, runner=racing)
    (row,) = _rows(tmp_path, "alpha")
    assert row["status"] == "executed" and row["key"] == "f" * 64


def test_a_failed_submission_marks_the_rest_never_ran(tmp_path: Path) -> None:
    """When sbatch refuses a job, it and every later step are ``never-ran`` and the error is raised."""
    fake = FakeSbatch((0, PROBE_OK), fail_at=3)
    with pytest.raises(slurm.SlurmError, match="score"):
        slurm.submit(_chain(), _config(tmp_path, [Path("/m/a.yaml"), Path("/m/b.yaml")]), runner=fake)
    statuses = {r["step"]: r["status"] for a in ("alpha", "beta") for r in _rows(tmp_path, a)}
    assert statuses == {"lone": "pending", "prep": "pending", "score": "never-ran", "report": "never-ran"}


def test_submit_uses_the_fallback_partition_when_the_probe_reports_preemption(tmp_path: Path) -> None:
    """The probe runs per job whose partition has a fallback, and its answer picks the partition."""
    fake = FakeSbatch((0, PROBE_PREEMPTS))
    slurm.submit([_manifest("alpha", _step("prep"))], _config(tmp_path, [Path("/m/a.yaml")]), runner=fake)
    assert "--partition=spare" in _sbatch_lines(fake.submissions()[0])
    probes = [s for a, s in fake.calls if "--test-only" in a]
    assert len(probes) == 1 and not any(f.startswith("--dependency") for f in _sbatch_lines(probes[0]))


@pytest.mark.network
def test_a_real_test_only_probe_answers_with_a_start_time(tmp_path: Path) -> None:
    """``sbatch --test-only`` on the build partition submits nothing and its answer parses."""
    if shutil.which("sbatch") is None:
        pytest.skip("no sbatch on PATH")
    partition = (
        subprocess.run(["sinfo", "-h", "-o", "%P"], capture_output=True, text=True, check=True)
        .stdout.split()[0]
        .rstrip("*")
    )
    profile = _profile(
        tmp_path,
        partitions={
            "cpu": {"partition": partition, "cpus": 1, "mem": "1G", "time": "00:01:00", "fallback": partition},
            "build": {"partition": partition, "cpus": 1, "mem": "1G", "time": "00:01:00"},
        },
    )
    text = slurm.render_step_job(
        _spec("probe"),
        profile,
        partition=partition,
        command=["--help"],
        date=DATE,
        exports=EXPORTS,
        driver_python=sys.executable,
    )
    result = slurm.default_runner(["sbatch", "--test-only"], text)
    answer = result.stdout + result.stderr
    assert result.returncode == 0, answer
    assert "to start at" in answer
    assert slurm.probe_preempts(answer) in (True, False)
