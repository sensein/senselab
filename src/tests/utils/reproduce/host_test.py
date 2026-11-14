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
    "host_cuda",
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
    assert host_record(None, runner=runner, hostname=False)["hostname"] is None
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


_TABLE = (
    0,
    "0, GPU-1111aaaa-0000, NVIDIA H100 80GB HBM3, 570.86.15\n"
    "1, GPU-2222bbbb-0000, NVIDIA  L40S , 570.86.15\n"
    "2, GPU-3333cccc-0000, NVIDIA A100-SXM4-80GB, 570.86.15\n",
)


def test_the_gpu_table_lists_every_gpu_from_one_query() -> None:
    """Index, UUID, single-spaced name and driver per GPU, in nvidia-smi's order; nothing without nvidia-smi."""
    runner = FakeRunner(smi=_TABLE, probe=(0, ""))
    table = host.gpu_table(runner=runner)
    assert [row["name"] for row in table] == ["NVIDIA H100 80GB HBM3", "NVIDIA L40S", "NVIDIA A100-SXM4-80GB"]
    assert table[1] == {"index": "1", "uuid": "GPU-2222bbbb-0000", "name": "NVIDIA L40S", "driver": "570.86.15"}
    assert runner.calls == [list(host.NVIDIA_SMI_DEVICES)]
    assert host.gpu_table(runner=FakeRunner(smi=None, probe=(0, ""))) == []
    assert host.gpu_table(runner=FakeRunner(smi=(9, "garbage"), probe=(0, ""))) == []


class TestTheVisibleMaskNamesTheDevices:
    """CUDA_VISIBLE_DEVICES, by index or by UUID, read against one nvidia-smi table without touching CUDA."""

    table = host.gpu_table(runner=FakeRunner(smi=_TABLE, probe=(0, "")))

    def test_an_index_list_names_those_gpus_in_its_order(self) -> None:
        """Indices are nvidia-smi's; the mask's order is the process's device order."""
        assert host.visible_gpu_names("2,0", self.table) == ["NVIDIA A100-SXM4-80GB", "NVIDIA H100 80GB HBM3"]

    def test_a_uuid_list_names_those_gpus_and_a_unique_prefix_will_do(self) -> None:
        """A mask may name GPUs by UUID; CUDA accepts a unique prefix, and so does this."""
        assert host.visible_gpu_names("GPU-2222bbbb-0000, GPU-1111", self.table) == [
            "NVIDIA L40S",
            "NVIDIA H100 80GB HBM3",
        ]

    def test_unset_means_every_gpu(self) -> None:
        """No mask: the process sees every GPU the node lists."""
        assert host.visible_gpu_names(None, self.table) == [
            "NVIDIA H100 80GB HBM3",
            "NVIDIA L40S",
            "NVIDIA A100-SXM4-80GB",
        ]
        assert host.visible_gpu_names(None, []) is None

    def test_an_empty_or_unmatched_mask_names_nothing_rather_than_guessing(self) -> None:
        """No device visible, or an entry naming no listed GPU (or an ambiguous prefix): None."""
        for mask in ("", "7", "GPU-9999", "GPU-", "0,7", "-1"):
            assert host.visible_gpu_names(mask, self.table) is None, mask


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


_FAKE_TORCH = """
__version__ = "{version}"


class cuda:
    @staticmethod
    def is_available():
        return {available}

    @staticmethod
    def device_count():
        return {count}

    @staticmethod
    def get_device_name(index):
        return ["NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB"][index]
"""


def _probe_in_a_worker(
    tmp_path: Path, torch_source: str | None, visible: str | None, *, call: str = "senselab_worker_device()"
) -> dict[str, object]:
    """Run the worker device probe in a fresh interpreter with no site-packages, seeing only the given ``torch``.

    Args:
        tmp_path: Where the stand-in ``torch`` package goes.
        torch_source: The stand-in's ``__init__.py``; None leaves torch unimportable.
        visible: The worker's ``CUDA_VISIBLE_DEVICES``; None leaves it unset.
        call: The probe call, after any imports it should find already made.

    Returns:
        What ``senselab_worker_device()`` returned there.
    """
    import os
    import subprocess
    import sys

    root = tmp_path / "site"
    if torch_source is not None:
        (root / "torch").mkdir(parents=True)
        (root / "torch" / "__init__.py").write_text(torch_source)
    env = {key: value for key, value in os.environ.items() if key not in ("PYTHONPATH", "CUDA_VISIBLE_DEVICES")}
    if visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible
    code = (
        f"import json, sys\nsys.path.insert(0, {str(root)!r})\n"
        + host.WORKER_DEVICE_PROBE
        + f"\nprint(json.dumps({call}))\n"
    )
    done = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, env=env, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


class TestTheWorkerNamesTheDevicesItSees:
    """Satra review 2026-11-14: the GPU on a reviewer's answer is the device its worker process saw."""

    def test_a_cuda_worker_names_every_visible_device_its_torch_and_its_mask(self, tmp_path: Path) -> None:
        """``torch.cuda.get_device_name`` per visible device, the process's own torch, and CUDA_VISIBLE_DEVICES."""
        source = _FAKE_TORCH.format(version="2.9.0+cu128", available=True, count=2)
        assert _probe_in_a_worker(tmp_path, source, "3,5") == {
            "gpus": ["NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB"],
            "torch": "2.9.0+cu128",
            "cuda_visible_devices": "3,5",
        }

    def test_a_cpu_worker_names_no_device(self, tmp_path: Path) -> None:
        """No CUDA: no devices, and torch still named."""
        source = _FAKE_TORCH.format(version="2.9.0+cpu", available=False, count=0)
        assert _probe_in_a_worker(tmp_path, source, None) == {
            "gpus": None,
            "torch": "2.9.0+cpu",
            "cuda_visible_devices": None,
        }

    def test_a_worker_without_torch_reports_none(self, tmp_path: Path) -> None:
        """Torch absent: every device field None, and the probe does not raise."""
        assert _probe_in_a_worker(tmp_path, None, "") == {"gpus": None, "torch": None, "cuda_visible_devices": ""}

    def test_a_cuda_query_that_raises_names_no_device(self, tmp_path: Path) -> None:
        """A CUDA runtime that fails on the query leaves the devices unnamed rather than failing the worker."""
        source = _FAKE_TORCH.format(version="2.9.0+cu128", available=True, count=3)
        assert _probe_in_a_worker(tmp_path, source, None)["gpus"] is None

    def test_beside_a_server_the_probe_touches_no_cuda(self, tmp_path: Path) -> None:
        """``cuda=False``: no CUDA query at all, and torch named only where the process had already imported it."""
        refusing = (
            '__version__ = "2.9.0+cu128"\n\n\ndef __getattr__(name):\n    raise AssertionError("touched " + name)\n'
        )
        imported = _probe_in_a_worker(
            tmp_path, refusing, "1", call="(__import__('torch'), senselab_worker_device(cuda=False))[1]"
        )
        assert imported == {"gpus": None, "torch": "2.9.0+cu128", "cuda_visible_devices": "1"}
        not_imported = _probe_in_a_worker(tmp_path / "again", refusing, None, call="senselab_worker_device(cuda=False)")
        assert not_imported == {"gpus": None, "torch": None, "cuda_visible_devices": None}

    def test_a_report_is_read_back_with_anything_malformed_as_none(self) -> None:
        """The host side takes the three keys and nothing else, and a missing or odd report is None throughout."""
        empty = dict.fromkeys(host.WORKER_DEVICE_KEYS)
        assert host.worker_device(None) == empty
        assert host.worker_device("garbage") == empty
        assert host.worker_device({"gpus": "H100", "torch": 2, "cuda_visible_devices": 0, "extra": 1}) == empty
        reported = {"gpus": ["NVIDIA L40S"], "torch": "2.8.0", "cuda_visible_devices": "0"}
        assert host.worker_device(reported) == reported
        assert set(host.worker_device(reported)) == set(host.WORKER_DEVICE_KEYS)
