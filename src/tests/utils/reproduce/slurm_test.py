"""Tests for batch rendering, the preemption probe and ``afterok`` submission chains.

``sbatch`` is always faked here, except in the one ``network`` test, which runs a real ``--test-only``
probe and submits nothing. Profiles, manifests and run trees live in ``tmp_path``.
"""

from __future__ import annotations

import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
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


_PARTS: dict[str, dict[str, Any]] = {
    "cpu": {"partition": "main", "cpus": 4, "mem": "8G", "time": "01:00:00", "fallback": "spare"},
    "gpu": {"partition": "accel", "gres": "gpu:a:1", "cpus": 8, "mem": "32G", "time": "02:00:00"},
    "build": {"partition": "quick", "cpus": 2, "mem": "4G", "time": "00:15:00"},
}


def _profile(tmp: Path, **change: Any) -> Profile:  # noqa: ANN401
    data: dict[str, Any] = {
        "roots": {"data": str(tmp / "data"), "extra": str(tmp / "extra")},
        "results_root": str(tmp / "results"),
        "runs_root": str(tmp / "runs"),
        "slurm_logs_root": str(tmp / "slurm"),
        "partitions": _PARTS,
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
        f"--chdir={logs}",
        "--export=NONE",
        "--dependency=afterok:11:12",
        "--kill-on-invalid-dep=yes",
    ]
    lines = text.splitlines()
    assert lines[0] == "#!/bin/bash"
    assert "set -euo pipefail" in lines and "umask 007" in lines
    assert lines[-1] == '"$REPRO_DRIVER_PYTHON" -P -m senselab.utils.reproduce run /m/a.yaml --step prep'


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
    with pytest.raises(ValueError, match="gres"):
        slurm.render_step_job(
            _spec("t", resources={"profile": "gpu"}),
            _profile(tmp_path, partitions={**_PARTS, "gpu": {**_PARTS["gpu"], "gres": "gpu:1 x"}}),
            partition="accel",
            command=["run"],
            date=DATE,
            exports=EXPORTS,
        )


def test_comma_lists_render(tmp_path: Path) -> None:
    """Slurm's comma forms (a partition list, several gres) are valid and render as given."""
    parts = {**_PARTS, "gpu": {**_PARTS["gpu"], "partition": "accel,spare", "gres": "gpu:a100:1,shard:1"}}
    profile = _profile(tmp_path, partitions=parts)
    step = _spec("t", resources={"profile": "gpu", "gpu_arch": ["sm_80"]})
    assert slurm.step_findings(step, profile) == []
    text = slurm.render_step_job(step, profile, partition="accel,spare", command=["run"], date=DATE, exports=EXPORTS)
    assert "--partition=accel,spare" in _sbatch_lines(text) and "--gres=gpu:a100:1,shard:1" in _sbatch_lines(text)


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
    assert argv == ["-P", "-m", "senselab.utils.reproduce", "run", "/m/a.yaml", "--step", "prep"]
    assert not Path(f"/dev/shm/reproduce-{job}").exists()


# --- a fake Slurm ------------------------------------------------------------------------------------


class FakeSbatch:
    """A fake sbatch/scontrol/scancel: records calls, the umask each ran under, and returns increasing job ids.

    ``probe`` answers ``--test-only`` (an exception instance is raised instead); ``fail_at`` makes the n-th
    submission fail, and ``raise_at`` makes it raise the given exception (or send SIGTERM for ``"SIGTERM"``).
    ``on_release`` runs when ``scontrol release`` is called.
    """

    def __init__(
        self,
        probe: tuple[int, str] | BaseException = (0, ""),
        fail_at: int | None = None,
        raise_at: tuple[int, BaseException | str] | None = None,
    ) -> None:
        """Set the probe answer and the failures."""
        self.calls: list[tuple[list[str], str]] = []
        self.umasks: list[int] = []
        self.probe = probe
        self.fail_at = fail_at
        self.raise_at = raise_at
        self.next_id = 100
        self.on_release: Any = None

    def __call__(self, argv: list[str], script: str) -> subprocess.CompletedProcess[str]:
        """Answer one call."""
        self.calls.append((argv, script))
        mask = os.umask(0)
        os.umask(mask)
        self.umasks.append(mask)
        if argv[0] in (slurm.SCANCEL, slurm.SCONTROL):
            if argv[:2] == [slurm.SCONTROL, "release"] and self.on_release is not None:
                self.on_release()
            return subprocess.CompletedProcess(argv, 0, "", "")
        if "--test-only" in argv:
            if isinstance(self.probe, BaseException):
                raise self.probe
            code, text = self.probe
            return subprocess.CompletedProcess(argv, code, "", text)
        count = len(self.submissions())
        if self.raise_at is not None and count == self.raise_at[0]:
            if self.raise_at[1] == "SIGTERM":
                os.kill(os.getpid(), signal.SIGTERM)
            else:
                assert isinstance(self.raise_at[1], BaseException)
                raise self.raise_at[1]
        if self.fail_at is not None and count == self.fail_at:
            return subprocess.CompletedProcess(argv, 1, "", "sbatch: error: denied")
        self.next_id += 1
        return subprocess.CompletedProcess(argv, 0, f"{self.next_id};cluster\n", "")

    def submissions(self) -> list[str]:
        """The scripts passed to ``--parsable`` submissions (the current call included)."""
        return [s for a, s in self.calls if "--parsable" in a]

    def commands(self, name: str) -> list[list[str]]:
        """Every call of one command."""
        return [a for a, _ in self.calls if a[0] == name]


# --- the feasibility probe ---------------------------------------------------------------------------


PROBE_OK = "sbatch: Job 4242 to start at 2026-03-04T10:00:00 using 4 processors on nodes n1 in partition main"
PROBE_PREEMPTS = PROBE_OK + "\nsbatch:   Preempts: 4100,4101"


@pytest.mark.parametrize(
    ("probe", "expected"),
    [
        ((0, PROBE_OK), "main"),
        ((0, PROBE_PREEMPTS), "main"),
        ((1, "sbatch: error: Requested node configuration is not available"), "spare"),
        (subprocess.TimeoutExpired(["sbatch"], 120), "spare"),
        (OSError("boom"), "spare"),
    ],
)
def test_choose_partition_is_a_feasibility_probe(tmp_path: Path, probe: Any, expected: str) -> None:  # noqa: ANN401
    """Only a rejected primary partition, or a probe that errors or times out, sends a job to the fallback."""
    fake = FakeSbatch(probe)
    chosen = slurm.choose_partition(_spec("prep"), _profile(tmp_path), lambda p: f"#!/bin/bash\n# {p}\n", fake)
    assert chosen == expected
    ((argv, script),) = fake.calls
    assert argv[-1] == "--test-only" and "# main" in script


def test_no_probe_without_a_fallback(tmp_path: Path) -> None:
    """A partition with no fallback is used as is, with no probe."""
    fake = FakeSbatch((1, "rejected"))
    step = _spec("train", resources={"profile": "gpu"})
    assert slurm.choose_partition(step, _profile(tmp_path), lambda p: p, fake) == "accel"
    assert fake.calls == []


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
    looping = _profile(tmp_path, partitions={**_PARTS, "cpu": {**_PARTS["cpu"], "fallback": "main"}})
    assert [f.code for f in slurm.step_findings(_spec("ok"), looping)] == ["fallback-invalid"]


@pytest.mark.parametrize("field", ["partition", "mem", "time", "gres", "fallback"])
def test_values_render_would_refuse_are_lint_findings(tmp_path: Path, field: str) -> None:
    """A partition, mem, time, gres or fallback that is not a plain word or comma list is ``resource-invalid``."""
    parts = {**_PARTS, "cpu": {**_PARTS["cpu"], "gres": "gpu:1", field: "x y; z"}}
    profile = _profile(tmp_path, partitions=parts)
    findings = slurm.step_findings(_spec("ok"), profile)
    assert ("resource-invalid", f"cpu.{field}") in [(f.code, f.detail) for f in findings]


def test_profile_findings(tmp_path: Path) -> None:
    """A log root or build partition value that cannot render is reported once, for the profile."""
    assert slurm.profile_findings(_profile(tmp_path)) == []
    bad = _profile(
        tmp_path,
        slurm_logs_root=str(tmp_path / "with space"),
        partitions={**_PARTS, "build": {**_PARTS["build"], "time": "1 h"}},
    )
    assert sorted(f.code for f in slurm.profile_findings(bad)) == ["logs-invalid", "resource-invalid"]


# --- held submission chains --------------------------------------------------------------------------


def _config(tmp: Path, paths: Sequence[Path], **change: Any) -> slurm.SubmitConfig:  # noqa: ANN401
    data: dict[str, Any] = {
        "profile": _profile(tmp),
        "profile_path": tmp / "profile.yaml",
        "manifest_paths": list(paths),
        "base_dir": tmp / "manifests",
        "cache": tmp / "cache",
        "date": DATE,
        "exports": EXPORTS,
        "driver_python": "/opt/py/bin/python",
        "submission": "sub1",
    }
    data.update(change)
    return slurm.SubmitConfig(**data)


def _chain() -> list[Manifest]:
    first = _manifest("alpha", _step("prep"), _step("score", needs=["prep"]))
    second = _manifest("beta", _step("report", needs=["score", "prep"]), _step("lone"))
    return [first, second]


def _record(tmp: Path, analysis: str) -> Path:
    return tmp / "runs" / DATE.isoformat() / analysis / run.RUN_JSON


def _rows(tmp: Path, analysis: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = json.loads(_record(tmp, analysis).read_text())["steps"]
    return rows


PATHS = [Path("/m/a.yaml"), Path("/m/b.yaml")]


def test_submit_holds_every_job_writes_rows_then_releases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Held jobs in DAG order with afterok on their needs, pending rows, a held finalize job, then one release."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    fake = FakeSbatch((0, PROBE_OK))
    seen: dict[str, Any] = {}

    def at_release() -> None:
        seen["rows"] = {r["step"]: r for a in ("alpha", "beta") for r in _rows(tmp_path, a)}

    fake.on_release = at_release
    submission = slurm.submit(_chain(), _config(tmp_path, PATHS, manifest_sha256s=["a" * 64, "b" * 64]), runner=fake)
    jobs = submission.jobs
    assert submission.finalize == "105" and submission.submission == "sub1"
    assert [(j.step, j.job_id) for j in jobs] == [("lone", "101"), ("prep", "102"), ("score", "103"), ("report", "104")]
    scripts = fake.submissions()
    assert len(scripts) == 5
    assert all(a[:3] == ["sbatch", "--parsable", "--hold"] for a, _ in fake.calls if "--parsable" in a)
    deps = [[f for f in _sbatch_lines(s) if f.startswith("--dependency")] for s in scripts]
    assert deps == [[], [], ["--dependency=afterok:102"], ["--dependency=afterok:103:102"], deps[4]]
    assert deps[4] == ["--dependency=afterany:101:102:103:104"]
    assert "--kill-on-invalid-dep=yes" not in _sbatch_lines(scripts[4])
    last = scripts[4].splitlines()[-1]
    assert last.endswith("--finalize") and "--submission sub1" in last
    for job, script in zip(jobs, scripts):
        line = script.splitlines()[-1]
        assert f"--step {job.step}" in line and "--date 2026-03-04" in line and "--submission sub1" in line
        assert f"--pin {'a' * 64} --pin {'b' * 64}" in line
        assert "/m/a.yaml /m/b.yaml" in line and "--cache" in line and "--base-dir" in line
    assert fake.commands("scontrol") == [["scontrol", "release", "101,102,103,104,105"]]
    assert fake.calls[-1][0][0] == "scontrol"
    assert {s: (r["status"], r["job"], r["submission"]) for s, r in seen["rows"].items()} == {
        "prep": ("pending", "102", "sub1"),
        "score": ("pending", "103", "sub1"),
        "report": ("pending", "104", "sub1"),
        "lone": ("pending", "101", "sub1"),
    }
    assert [r["step"] for r in _rows(tmp_path, "beta")] == ["lone", "report"]
    assert os.listdir(cwd) == []
    logs = tmp_path / "slurm" / "2026-03-04"
    assert logs.is_dir() and (logs.stat().st_mode & 0o7777) == slurm.LOG_DIR_MODE
    assert set(fake.umasks) == {slurm.SUBMIT_UMASK}


def test_a_render_error_has_no_side_effect(tmp_path: Path) -> None:
    """A script that cannot render fails before any probe, row, log dir or submission."""
    fake = FakeSbatch((0, PROBE_OK))
    with pytest.raises(ValueError, match="export"):
        slurm.submit(_chain(), _config(tmp_path, PATHS, exports={**EXPORTS, "BAD NAME": "x"}), runner=fake)
    assert fake.calls == []
    assert not (tmp_path / "runs").exists() and not (tmp_path / "slurm").exists()


@pytest.mark.parametrize(
    "failure",
    [("fail", 3), ("raise", RuntimeError("boom")), ("raise", KeyboardInterrupt()), ("raise", "SIGTERM")],
)
def test_an_aborted_submission_cancels_every_job_and_leaves_nothing_pending(tmp_path: Path, failure: Any) -> None:  # noqa: ANN401
    """A refused sbatch, an exception, Ctrl-C or SIGTERM: every queued job is cancelled, every row never-ran."""
    kind, what = failure
    fake = FakeSbatch(
        (0, PROBE_OK), fail_at=what if kind == "fail" else None, raise_at=None if kind == "fail" else (3, what)
    )
    with pytest.raises(slurm.SlurmError, match="E_SUBMIT_ABORTED"):
        slurm.submit(_chain(), _config(tmp_path, PATHS), runner=fake)
    assert fake.commands("scancel") == [["scancel", "101", "102"]]
    assert fake.commands("scontrol") == []
    rows = [r for a in ("alpha", "beta") for r in _rows(tmp_path, a)]
    assert {(r["status"], tuple(r["codes"]), r["submission"]) for r in rows} == {
        ("never-ran", ("E_SUBMIT_ABORTED",), "sub1")
    }
    assert len(rows) == 4


def test_a_failed_release_cancels_everything(tmp_path: Path) -> None:
    """If ``scontrol release`` fails, every job (finalize included) is cancelled."""

    class FailingRelease(FakeSbatch):
        def __call__(self, argv: list[str], script: str) -> subprocess.CompletedProcess[str]:
            if argv[:2] == [slurm.SCONTROL, "release"]:
                self.calls.append((argv, script))
                return subprocess.CompletedProcess(argv, 1, "", "")
            return super().__call__(argv, script)

    fake = FailingRelease((0, PROBE_OK))
    with pytest.raises(slurm.SlurmError, match="E_SUBMIT_ABORTED"):
        slurm.submit(_chain(), _config(tmp_path, PATHS), runner=fake)
    assert fake.commands("scancel") == [["scancel", "101", "102", "103", "104", "105"]]


def test_submit_refuses_while_another_submission_is_pending(tmp_path: Path) -> None:
    """Pending rows from another submission refuse with ``E_SUBMIT_OVERLAP`` and nothing is submitted."""
    row: dict[str, Any] = {"step": "prep", "key": None, "status": "pending", "exit": None, "codes": []}
    row["submission"] = "old"
    run.update_run_record(tmp_path / "runs", DATE, "alpha", [row], order=["prep"])
    fake = FakeSbatch((0, PROBE_OK))
    with pytest.raises(slurm.SlurmError, match="E_SUBMIT_OVERLAP"):
        slurm.submit(_chain(), _config(tmp_path, PATHS), runner=fake)
    assert fake.calls == []
    assert _rows(tmp_path, "alpha") == [row]


def test_submit_uses_the_fallback_only_when_the_probe_rejects(tmp_path: Path) -> None:
    """A rejected primary partition sends the job to the fallback; the probe script has no dependency."""
    fake = FakeSbatch(
        (1, "sbatch: error: Batch job submission failed: Requested partition configuration not available")
    )
    slurm.submit([_manifest("alpha", _step("prep"))], _config(tmp_path, [Path("/m/a.yaml")]), runner=fake)
    assert "--partition=spare" in _sbatch_lines(fake.submissions()[0])
    probes = [s for a, s in fake.calls if "--test-only" in a]
    assert len(probes) == 1 and not any(f.startswith("--dependency") for f in _sbatch_lines(probes[0]))


def test_two_concurrent_submissions_one_wins(tmp_path: Path) -> None:
    """Two submits racing for one run dir: exactly one reserves it; the other gets E_SUBMIT_OVERLAP, no jobs."""
    first_in = threading.Event()
    other_done = threading.Event()

    class Blocking(FakeSbatch):
        def __call__(self, argv: list[str], script: str) -> subprocess.CompletedProcess[str]:
            if "--parsable" in argv and not first_in.is_set():
                first_in.set()
                assert other_done.wait(30)
            return super().__call__(argv, script)

    results: dict[str, Any] = {}
    fakes = {"A": Blocking((0, PROBE_OK)), "B": Blocking((0, PROBE_OK))}

    def go(label: str) -> None:
        try:
            results[label] = slurm.submit(
                _chain(), _config(tmp_path, PATHS, submission=f"sub{label}"), runner=fakes[label]
            )
        except slurm.SlurmError as error:
            results[label] = error

    a = threading.Thread(target=go, args=("A",))
    a.start()
    assert first_in.wait(30)
    b = threading.Thread(target=go, args=("B",))
    b.start()
    b.join(30)
    other_done.set()
    a.join(30)
    assert isinstance(results["A"], slurm.Submission)
    assert isinstance(results["B"], slurm.SlurmError) and "E_SUBMIT_OVERLAP" in str(results["B"])
    assert fakes["B"].calls == []
    rows = [r for a_ in ("alpha", "beta") for r in _rows(tmp_path, a_)]
    assert {r["submission"] for r in rows} == {"subA"} and all(r["job"] for r in rows)


def test_a_submission_inside_another_is_refused(tmp_path: Path) -> None:
    """The reviewer's probe: a second submit run while the first is between sbatch calls gets the overlap."""
    seen: dict[str, Any] = {}

    class Nested(FakeSbatch):
        def __call__(self, argv: list[str], script: str) -> subprocess.CompletedProcess[str]:
            if "--parsable" in argv and "inner" not in seen:
                inner = FakeSbatch((0, PROBE_OK))
                try:
                    slurm.submit(_chain(), _config(tmp_path, PATHS, submission="subB"), runner=inner)
                    seen["inner"] = "submitted"
                except slurm.SlurmError as error:
                    seen["inner"] = str(error)
                seen["inner_calls"] = inner.calls
            return super().__call__(argv, script)

    slurm.submit(_chain(), _config(tmp_path, PATHS), runner=Nested((0, PROBE_OK)))
    assert "E_SUBMIT_OVERLAP" in seen["inner"] and seen["inner_calls"] == []


def test_an_abort_keeps_an_earlier_submissions_rows(tmp_path: Path) -> None:
    """Rows a placeholder replaced come back on abort; nothing else of another submission is touched."""
    old = [
        {"step": s, "key": "f" * 64, "status": "executed", "exit": 0, "codes": [], "submission": "old"}
        for s in ("prep", "score")
    ]
    run.update_run_record(tmp_path / "runs", DATE, "alpha", old, order=["prep", "score"])
    fake = FakeSbatch((0, PROBE_OK), fail_at=2)
    with pytest.raises(slurm.SlurmError, match="E_SUBMIT_ABORTED"):
        slurm.submit(_chain(), _config(tmp_path, PATHS), runner=fake)
    assert _rows(tmp_path, "alpha") == old
    assert {(r["status"], tuple(r["codes"])) for r in _rows(tmp_path, "beta")} == {("never-ran", ("E_SUBMIT_ABORTED",))}


def test_a_second_signal_cannot_interrupt_the_abort(tmp_path: Path) -> None:
    """SIGTERM at the release starts the abort; another SIGTERM during scancel is ignored and the abort ends."""

    class Signalling(FakeSbatch):
        def __call__(self, argv: list[str], script: str) -> subprocess.CompletedProcess[str]:
            if argv[:2] == [slurm.SCONTROL, "release"] or argv[0] == slurm.SCANCEL:
                os.kill(os.getpid(), signal.SIGTERM)
            return super().__call__(argv, script)

    fake = Signalling((0, PROBE_OK))
    with pytest.raises(slurm.SlurmError, match="E_SUBMIT_ABORTED: Terminated"):
        slurm.submit(_chain(), _config(tmp_path, PATHS), runner=fake)
    assert fake.commands("scancel") == [["scancel", "101", "102", "103", "104", "105"]]
    rows = [r for a in ("alpha", "beta") for r in _rows(tmp_path, a)]
    assert {r["status"] for r in rows} == {"never-ran"}
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL or callable(signal.getsignal(signal.SIGTERM))


def test_sighup_aborts_like_sigterm(tmp_path: Path) -> None:
    """SIGHUP during the submission aborts it and cancels the jobs already queued."""

    class Hangup(FakeSbatch):
        def __call__(self, argv: list[str], script: str) -> subprocess.CompletedProcess[str]:
            if "--parsable" in argv and len(self.submissions()) == 2:
                os.kill(os.getpid(), signal.SIGHUP)
            return super().__call__(argv, script)

    fake = Hangup((0, PROBE_OK))
    with pytest.raises(slurm.SlurmError, match="E_SUBMIT_ABORTED: Terminated"):
        slurm.submit(_chain(), _config(tmp_path, PATHS), runner=fake)
    assert fake.commands("scancel") == [["scancel", "101", "102"]]


def test_a_failing_scancel_is_reported_with_the_job_ids(tmp_path: Path) -> None:
    """When scancel fails, the error carries E_SCANCEL_FAILED and the leftover job ids, nothing else."""

    class NoCancel(FakeSbatch):
        def __call__(self, argv: list[str], script: str) -> subprocess.CompletedProcess[str]:
            if argv[0] == slurm.SCANCEL:
                self.calls.append((argv, script))
                return subprocess.CompletedProcess(argv, 1, "", "scancel: error")
            return super().__call__(argv, script)

    fake = NoCancel((0, PROBE_OK), fail_at=3)
    with pytest.raises(slurm.SlurmError) as info:
        slurm.submit(_chain(), _config(tmp_path, PATHS), runner=fake)
    assert "E_SCANCEL_FAILED: job(s) 101,102 may still be queued" in str(info.value)
    assert str(tmp_path) not in str(info.value)


def test_a_space_in_gres_is_refused_before_anything(tmp_path: Path) -> None:
    """A gres that cannot render is a finding: nothing probed, written or submitted."""
    parts = {**_PARTS, "gpu": {**_PARTS["gpu"], "gres": "gpu a100"}}
    manifests = [_manifest("alpha", _step("a"), _step("b", needs=["a"], resources={"profile": "gpu"}))]
    fake = FakeSbatch((0, PROBE_OK))
    with pytest.raises(slurm.SlurmError, match="cross-check"):
        slurm.submit(manifests, _config(tmp_path, PATHS, profile=_profile(tmp_path, partitions=parts)), runner=fake)
    assert fake.calls == [] and not (tmp_path / "runs").exists() and not (tmp_path / "slurm").exists()


def test_a_jobs_own_row_needs_its_submission(tmp_path: Path) -> None:
    """A job writes its row only over its own submission's row, an untagged row, or none."""
    runs = tmp_path / "runs"
    tagged: dict[str, Any] = {
        "step": "a",
        "key": None,
        "status": "pending",
        "exit": None,
        "codes": [],
        "submission": "s2",
    }
    untagged: dict[str, Any] = {"step": "b", "key": None, "status": "executed", "exit": 0, "codes": []}
    run.update_run_record(runs, DATE, "alpha", [tagged, untagged], order=["a", "b", "c"])
    mine = [{"step": x, "status": "executed", "exit": 0, "codes": []} for x in ("a", "b", "c")]
    run.update_run_record(runs, DATE, "alpha", mine, order=["a", "b", "c"], submission="s1", allow_untagged=True)
    got = {r["step"]: (r["status"], r.get("submission")) for r in _rows(tmp_path, "alpha")}
    assert got == {"a": ("pending", "s2"), "b": ("executed", None), "c": ("executed", None)}


def test_job_state_reads_sacct() -> None:
    """``sacct`` gives the first word of the state; anything else is None."""

    def sacct(answer: str, code: int = 0) -> Any:  # noqa: ANN401
        return lambda argv, script: subprocess.CompletedProcess(argv, code, answer, "")

    assert slurm.job_state("12", sacct("TIMEOUT\n")) == "TIMEOUT"
    assert slurm.job_state("12", sacct("CANCELLED by 77\n")) == "CANCELLED"
    assert slurm.job_state("12", sacct("", 1)) is None
    assert slurm.job_state("12; rm", sacct("TIMEOUT")) is None


# --- waited build jobs ----------------------------------------------------------------------------------


def test_submit_and_wait_has_no_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """The waiting sbatch runs with no timeout, under umask 007, named on the command line."""
    seen: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:  # noqa: ANN401
        mask = os.umask(0)
        os.umask(mask)
        seen.update(kwargs, argv=argv, umask=mask)
        return subprocess.CompletedProcess(argv, 0, "7\n", "")

    monkeypatch.setattr(slurm.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(slurm.subprocess, "run", fake_run)
    slurm.submit_and_wait("#!/bin/bash\n", name="reproduce-build-x-1")
    assert seen["timeout"] is None and seen["umask"] == slurm.SUBMIT_UMASK
    assert seen["argv"] == ["/usr/bin/sbatch", "--wait", "--parsable", "--job-name=reproduce-build-x-1"]


def test_submit_and_wait_failure_and_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed job raises; an interrupted wait cancels the job by name and re-raises."""
    cancels: list[list[str]] = []
    monkeypatch.setattr(slurm, "default_runner", lambda argv, script, timeout=None: cancels.append(argv))
    with pytest.raises(slurm.SlurmError, match="failed"):
        slurm.submit_and_wait("s", name="job-a", runner=lambda a, s: subprocess.CompletedProcess(a, 1, "", ""))
    assert cancels == []

    def interrupted(argv: list[str], script: str) -> subprocess.CompletedProcess[str]:
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        slurm.submit_and_wait("s", name="job-b", runner=interrupted)
    assert len(cancels) == 1 and cancels[0][:2] == ["scancel", "--name=job-b"]


@pytest.mark.network
def test_a_real_test_only_probe_answers_with_a_start_time(tmp_path: Path) -> None:
    """``sbatch --test-only`` on a real partition submits nothing and accepts the rendered script."""
    if shutil.which("sbatch") is None:
        pytest.skip("no sbatch on PATH")
    partition = subprocess.run(["sinfo", "-h", "-o", "%P"], capture_output=True, text=True, check=True).stdout.split()[
        0
    ]
    partition = partition.rstrip("*")
    parts = {
        "cpu": {"partition": partition, "cpus": 1, "mem": "1G", "time": "00:01:00", "fallback": "nonexistent-part"},
        "build": {"partition": partition, "cpus": 1, "mem": "1G", "time": "00:01:00"},
    }
    profile = _profile(tmp_path, partitions=parts)
    slurm.prepare_log_dir(profile, DATE)

    def script_for(p: str) -> str:
        return slurm.render_step_job(
            _spec("probe"),
            profile,
            partition=p,
            command=["--help"],
            date=DATE,
            exports=EXPORTS,
            driver_python=sys.executable,
        )

    result = slurm.default_runner(["sbatch", "--test-only"], script_for(partition))
    answer = result.stdout + result.stderr
    assert result.returncode == 0, answer
    assert "to start at" in answer
    assert slurm.choose_partition(_spec("probe"), profile, script_for, slurm.default_runner) == partition
