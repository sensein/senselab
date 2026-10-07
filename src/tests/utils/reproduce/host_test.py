"""Tests for the host record."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from senselab.utils.reproduce import host
from senselab.utils.reproduce.host import HOST_KEYS, host_record

KEYS = {
    "hostname",
    "cpu_model",
    "cpu_count",
    "cpus_allowed",
    "gpu_name",
    "gpu_driver",
    "gpu_count",
    "torch",
    "torch_cuda",
    "python",
    "slurm_job_id",
}


def _venv(tmp_path: Path) -> Path:
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "python").write_text("")
    return venv


class FakeRunner:
    """Answers nvidia-smi and the venv probe with canned output."""

    def __init__(self, smi: tuple[int, str] | None, probe: tuple[int, str]) -> None:
        """Store the canned answers."""
        self.smi, self.probe = smi, probe
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> tuple[int, str]:
        """Return the canned answer for ``argv``."""
        self.calls.append(list(argv))
        if argv[0] == "nvidia-smi":
            return self.smi if self.smi is not None else (127, "")
        return self.probe


def test_keys_are_fixed() -> None:
    """The record's key set is fixed."""
    assert set(HOST_KEYS) == KEYS


def test_cpu_node_without_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """On a CPU node the GPU fields are None and the count 0; the job id is read."""
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    runner = FakeRunner(smi=None, probe=(0, json.dumps({"python": "3.12.11", "torch": None, "torch_cuda": None})))
    record = host_record(_venv(tmp_path), runner=runner)
    assert set(record) == KEYS
    assert record["gpu_name"] is None and record["gpu_driver"] is None and record["gpu_count"] == 0
    assert record["torch"] is None and record["torch_cuda"] is None
    assert record["python"] == "3.12.11"
    assert record["slurm_job_id"] == "12345"
    assert isinstance(record["hostname"], str) and record["hostname"]
    assert isinstance(record["cpu_count"], int) and record["cpu_count"] >= 1
    assert isinstance(record["cpus_allowed"], int) and record["cpus_allowed"] >= 1
    assert runner.calls[0] == ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]
    assert runner.calls[1][0] == str(tmp_path / "venv" / "bin" / "python")


def test_gpu_node_with_torch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """nvidia-smi rows give the first GPU's name and driver and the count."""
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    smi = (0, "NVIDIA H100 80GB HBM3, 570.86.15\nNVIDIA H100 80GB HBM3, 570.86.15\n")
    probe = (0, "noise\n" + json.dumps({"python": "3.11.9", "torch": "2.9.0+cu128", "torch_cuda": "12.8"}) + "\n")
    record = host_record(_venv(tmp_path), runner=FakeRunner(smi=smi, probe=probe))
    assert record["gpu_name"] == "NVIDIA H100 80GB HBM3"
    assert record["gpu_driver"] == "570.86.15"
    assert record["gpu_count"] == 2
    assert record["torch"] == "2.9.0+cu128"
    assert record["torch_cuda"] == "12.8"
    assert record["slurm_job_id"] is None


def test_failing_probe_and_smi_give_none(tmp_path: Path) -> None:
    """Failing probe and smi give none."""
    record = host_record(_venv(tmp_path), runner=FakeRunner(smi=(9, "garbage"), probe=(1, "Traceback")))
    assert record["gpu_name"] is None and record["gpu_count"] == 0
    assert record["torch"] is None and record["torch_cuda"] is None and record["python"] is None


def test_garbled_probe_output_gives_none(tmp_path: Path) -> None:
    """Garbled probe output gives none."""
    record = host_record(_venv(tmp_path), runner=FakeRunner(smi=None, probe=(0, "{not json")))
    assert record["torch"] is None and record["python"] is None


def test_missing_venv_python_is_not_run(tmp_path: Path) -> None:
    """Missing venv python is not run."""
    runner = FakeRunner(smi=None, probe=(0, json.dumps({"python": "x", "torch": "y", "torch_cuda": None})))
    record = host_record(tmp_path / "absent", runner=runner)
    assert record["torch"] is None and record["python"] is None
    assert len(runner.calls) == 1


def test_default_runner_on_this_host(tmp_path: Path) -> None:
    """The subprocess runner works on this host, GPU or not."""
    record = host_record(tmp_path / "absent")
    assert set(record) == KEYS
    assert record["torch"] is None


def test_padded_cpu_and_gpu_names_are_single_spaced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs of spaces in cpuinfo or nvidia-smi names collapse, so provenance's single-space form accepts the host."""
    cpuinfo = tmp_path / "cpuinfo"
    cpuinfo.write_text("processor\t: 0\nmodel name\t: Intel(R) Xeon(R) CPU           X5650  @ 2.67GHz\n")
    monkeypatch.setattr(host, "CPUINFO", str(cpuinfo))
    smi = (0, "NVIDIA  A100-SXM4-80GB , 570.86.15\n")
    probe = (0, json.dumps({"python": "3.12.11", "torch": None, "torch_cuda": None}))
    record = host_record(_venv(tmp_path), runner=FakeRunner(smi=smi, probe=probe))
    assert record["cpu_model"] == "Intel(R) Xeon(R) CPU X5650 @ 2.67GHz"
    assert record["gpu_name"] == "NVIDIA A100-SXM4-80GB"


def test_no_venv_probes_only_the_host() -> None:
    """With no venv the interpreter fields are None and no interpreter is run."""
    calls: list[Sequence[str]] = []

    def runner(argv: Sequence[str]) -> tuple[int, str]:
        calls.append(argv)
        return 1, ""

    record = host_record(None, runner=runner)
    assert (record["python"], record["torch"], record["torch_cuda"]) == (None, None, None)
    assert [c[0] for c in calls] == ["nvidia-smi"]
