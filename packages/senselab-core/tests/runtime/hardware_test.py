"""Unit tests for ``senselab_core.runtime.hardware``.

The probes take an injected runner; no test shells out to ``nvidia-smi``.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest
from senselab_core.runtime.hardware import (
    DeviceResolutionError,
    GpuInfo,
    HardwareInfo,
    device_memory_bytes,
    memory_bytes,
    numerics_class,
    probe_hardware,
    process_tree,
)

# ── probe_hardware ───────────────────────────────────────────────────

TWO_GPUS = (
    "0, GPU-aaaa-1111, NVIDIA H100 80GB HBM3, 81559, 550.54.15, 9.0\n"
    "1, GPU-bbbb-2222, NVIDIA A100-SXM4-40GB, 40960, 550.54.15, 8.0\n"
)


def _smi(output: str) -> Callable[[list[str]], str]:
    return lambda cmd: output


def _raising(exc: Exception) -> Callable[[list[str]], str]:
    def run(cmd: list[str]) -> str:
        raise exc

    return run


def _info(*gpus: GpuInfo, chip: str | None = None) -> HardwareInfo:
    return HardwareInfo(8, 1, gpus, chip is not None, "x", chip=chip)


H100 = GpuInfo("NVIDIA H100", 1, "550", "9.0", "GPU-aaaa-1111", 0)
A100 = GpuInfo("NVIDIA A100", 1, "550", "8.0", "GPU-bbbb-2222", 1)


@pytest.mark.logic
def test_parses_two_gpus_from_nvidia_smi_csv() -> None:
    """Two CSV lines give two GpuInfo with memory in bytes, uuid, index and compute capability."""
    info = probe_hardware(run=_smi(TWO_GPUS))
    assert info.gpus == (
        GpuInfo("NVIDIA H100 80GB HBM3", 81559 * 1024 * 1024, "550.54.15", "9.0", "GPU-aaaa-1111", 0),
        GpuInfo("NVIDIA A100-SXM4-40GB", 40960 * 1024 * 1024, "550.54.15", "8.0", "GPU-bbbb-2222", 1),
    )
    assert info.gpu_probe_error is None
    assert info.cpu_count >= 1


@pytest.mark.logic
def test_driver_rejecting_compute_cap_falls_back_and_records_it() -> None:
    """A non-zero exit on the compute_cap query retries without it; the note says so."""
    queries: list[str] = []

    def run(cmd: list[str]) -> str:
        if cmd[0] != "nvidia-smi":
            raise FileNotFoundError
        queries.append(cmd[1])
        if "compute_cap" in cmd[1]:
            raise subprocess.CalledProcessError(2, cmd)
        return "0, GPU-aaaa-1111, NVIDIA H100, 81559, 550.54.15\n"

    info = probe_hardware(run=run)
    assert len(queries) == 2 and info.gpus[0].compute_capability is None
    assert info.gpu_probe_error is not None and "compute_cap" in info.gpu_probe_error
    assert numerics_class(info, "cuda", "fp32", environ={})["gpu_architecture"] == "NVIDIA H100"


@pytest.mark.logic
def test_no_nvidia_smi_means_no_gpus() -> None:
    """A missing binary is an absent GPU, not an error."""
    info = probe_hardware(run=_raising(FileNotFoundError("nvidia-smi")))
    assert info.gpus == () and info.gpu_probe_error is None


@pytest.mark.logic
def test_nonzero_exit_means_no_gpus_with_error() -> None:
    """A failing nvidia-smi (both queries) gives no GPUs and says so in gpu_probe_error."""
    info = probe_hardware(run=_raising(subprocess.CalledProcessError(9, "nvidia-smi")))
    assert info.gpus == () and info.gpu_probe_error == "nvidia-smi exited 9"


@pytest.mark.logic
def test_timeout_means_no_gpus_with_error() -> None:
    """A timed-out nvidia-smi gives no GPUs and a typed error."""
    info = probe_hardware(run=_raising(subprocess.TimeoutExpired("nvidia-smi", 30)))
    assert info.gpus == () and info.gpu_probe_error == "nvidia-smi timed out"


@pytest.mark.logic
def test_permission_error_is_an_error_not_an_absence() -> None:
    """A present but unexecutable nvidia-smi is reported."""
    info = probe_hardware(run=_raising(PermissionError("denied")))
    assert info.gpus == () and info.gpu_probe_error is not None and "denied" in info.gpu_probe_error


@pytest.mark.logic
def test_unparseable_output_means_no_gpus_with_error() -> None:
    """Garbage output gives no GPUs and the typed gpu_probe_error field names the line."""
    info = probe_hardware(run=_smi("NVIDIA-SMI has failed\n"))
    assert info.gpus == ()
    assert info.gpu_probe_error is not None and "NVIDIA-SMI has failed" in info.gpu_probe_error


@pytest.mark.logic
def test_failed_memory_probe_sets_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing sysconf name yields memory 0 plus probe_error, not a silent 0."""

    def boom(name: str) -> int:
        raise ValueError(name)

    monkeypatch.setattr("os.sysconf", boom)
    info = probe_hardware(run=_raising(FileNotFoundError()))
    assert info.memory_bytes == 0 and info.probe_error is not None and "memory" in info.probe_error


@pytest.mark.logic
def test_apple_silicon_detected_from_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """Darwin on arm64 is Apple silicon with its chip from sysctl; the other combinations are not."""

    def run(cmd: list[str]) -> str:
        if cmd[0] == "sysctl":
            return "Apple M2 Max\n"
        raise FileNotFoundError

    for system, machine, expected in (
        ("Darwin", "arm64", True),
        ("Darwin", "x86_64", False),
        ("Linux", "arm64", False),
    ):
        monkeypatch.setattr("platform.system", lambda s=system: s)
        monkeypatch.setattr("platform.machine", lambda m=machine: m)
        info = probe_hardware(run=run)
        assert info.apple_silicon is expected
        assert info.chip == ("Apple M2 Max" if expected else None)
        assert info.platform == f"{system.lower()}-{machine}"


@pytest.mark.logic
def test_failed_chip_probe_sets_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing sysctl on Apple silicon leaves chip None and records probe_error."""
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "arm64")
    info = probe_hardware(run=_raising(FileNotFoundError()))
    assert info.chip is None and info.probe_error is not None and "chip" in info.probe_error


@pytest.mark.logic
def test_probe_imports_no_torch() -> None:
    """Importing the module and probing leaves torch out of sys.modules."""
    code = (
        "import sys\n"
        "from senselab_core.runtime.hardware import probe_hardware\n"
        "probe_hardware(run=lambda c: '')\n"
        "assert 'torch' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


# ── numerics_class ───────────────────────────────────────────────────


@pytest.mark.logic
def test_numerics_class_reports_architecture_without_cuda_visible_devices() -> None:
    """With the variable unset, cuda:N is the N-th GPU and the architecture is sm_<cc>."""
    info = _info(H100, A100)
    assert numerics_class(info, "cuda:0", "bf16", environ={}) == {
        "device_kind": "cuda",
        "precision": "bf16",
        "gpu_architecture": "sm_90",
    }
    assert numerics_class(info, "cuda:1", "fp16", environ={})["gpu_architecture"] == "sm_80"
    assert numerics_class(info, "cpu", "fp32") == {"device_kind": "cpu", "precision": "fp32"}


@pytest.mark.logic
def test_numerics_class_cuda_visible_devices_index_list() -> None:
    """cuda:0 under CUDA_VISIBLE_DEVICES=1 is physical GPU 1."""
    info = _info(H100, A100)
    assert numerics_class(info, "cuda:0", "fp32", environ={"CUDA_VISIBLE_DEVICES": "1"})["gpu_architecture"] == "sm_80"
    assert numerics_class(info, "cuda", "fp32", environ={"CUDA_VISIBLE_DEVICES": "1,0"})["gpu_architecture"] == "sm_80"
    assert (
        numerics_class(info, "cuda:1", "fp32", environ={"CUDA_VISIBLE_DEVICES": "1,0"})["gpu_architecture"] == "sm_90"
    )


@pytest.mark.logic
def test_numerics_class_cuda_visible_devices_uuid_list() -> None:
    """A GPU-<uuid> entry resolves to the matching GPU."""
    info = _info(H100, A100)
    env = {"CUDA_VISIBLE_DEVICES": "GPU-bbbb-2222"}
    assert numerics_class(info, "cuda:0", "fp32", environ=env)["gpu_architecture"] == "sm_80"


@pytest.mark.logic
def test_numerics_class_out_of_range_and_unmappable_raise() -> None:
    """Out-of-range N, MIG ids, unknown indices and no GPUs raise instead of using GPU 0."""
    info = _info(H100, A100)
    for device, env in (
        ("cuda:2", {}),
        ("cuda:1", {"CUDA_VISIBLE_DEVICES": "1"}),
        ("cuda:0", {"CUDA_VISIBLE_DEVICES": "MIG-abc"}),
        ("cuda:0", {"CUDA_VISIBLE_DEVICES": "7"}),
        ("cuda:0", {"CUDA_VISIBLE_DEVICES": ""}),
        ("cuda:x", {}),
    ):
        with pytest.raises(DeviceResolutionError):
            numerics_class(info, device, "fp32", environ=env)
    with pytest.raises(DeviceResolutionError):
        numerics_class(_info(), "cuda", "fp32", environ={})


@pytest.mark.logic
def test_numerics_class_mps_reports_chip() -> None:
    """On mps the architecture is the chip; unknown chip omits it."""
    assert numerics_class(_info(chip="Apple M2 Max"), "mps", "fp16")["gpu_architecture"] == "Apple M2 Max"
    assert "gpu_architecture" not in numerics_class(_info(), "mps", "fp16")


@pytest.mark.logic
def test_device_memory_counts_only_the_given_processes() -> None:
    """Device memory sums the compute apps of the given pids within the given timeout; a foreign pid never counts."""
    seen: list[float] = []

    def listing(cmd: list[str], timeout_s: float) -> str:
        seen.append(timeout_s)
        return "100, 1024\n200, 2048\n300, 4096\n"

    assert device_memory_bytes({100, 200}, timeout_s=7.0, run=listing) == 3072 * 2**20 and seen == [7.0]
    assert device_memory_bytes({999}, timeout_s=7.0, run=listing) == 0

    def absent(cmd: list[str], timeout_s: float) -> str:
        raise FileNotFoundError("nvidia-smi")

    assert device_memory_bytes({100}, timeout_s=7.0, run=absent) is None


@pytest.mark.logic
def test_process_tree_holds_this_process() -> None:
    """The tree of this process holds it (and, where `/proc` is read, a child it started)."""
    child = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    try:
        tree = process_tree()
        assert os.getpid() in tree and (child.pid in tree or not os.path.isdir("/proc/self"))
    finally:
        child.kill()
        child.wait()


@pytest.mark.logic
def test_memory_without_a_readable_cgroup_has_no_current_figure(tmp_path: Path) -> None:
    """With no cgroup v2 to read, the current memory is None and only the high-water is given."""
    current, peak = memory_bytes(tmp_path / "absent-cgroup")
    assert current is None and peak > 0
