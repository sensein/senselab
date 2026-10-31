"""Unit tests for ``senselab_core.runtime.hardware``.

All ``subprocess.run`` calls are mocked — these tests must run on CPU-only
CI without shelling out to ``nvidia-smi`` or ``nvcc``.
"""

import subprocess
import sys
from typing import Callable
from unittest.mock import patch

import pytest
from senselab_core.runtime.hardware import (
    DeviceResolutionError,
    GpuInfo,
    HardwareInfo,
    HostCuda,
    SenselabCudaCompatibilityError,
    TorchIndex,
    detect_host_cuda,
    numerics_class,
    pick_torch_index,
    probe_hardware,
)


def _fake_run(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ── detect_host_cuda ─────────────────────────────────────────────────


@pytest.mark.logic
def test_detect_host_cuda_parses_nvidia_smi_header() -> None:
    """nvidia-smi prints `CUDA Version: 12.9` in its default header."""
    nvidia_smi_stdout = (
        "+-----------------------------------------------------------------------------+\n"
        "| NVIDIA-SMI 535.86.05    Driver Version: 535.86.05    CUDA Version: 12.9     |\n"
        "+-----------------------------------------------------------------------------+\n"
    )
    with patch("subprocess.run", return_value=_fake_run(0, stdout=nvidia_smi_stdout)):
        host = detect_host_cuda()
    assert host == HostCuda(version=(12, 9), source="nvidia-smi", raw="CUDA Version: 12.9")


@pytest.mark.logic
def test_detect_host_cuda_falls_through_to_nvcc_when_nvidia_smi_missing() -> None:
    """When nvidia-smi raises FileNotFoundError, nvcc must be tried next."""
    nvcc_stdout = "nvcc: NVIDIA (R) Cuda compiler driver\nrelease 12.4, V12.4.131\n"

    def _runner(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        argv = args
        if argv and argv[0] == "nvidia-smi":
            raise FileNotFoundError("no nvidia-smi")
        if argv and argv[0] == "nvcc":
            return _fake_run(0, stdout=nvcc_stdout)
        raise AssertionError(f"unexpected invocation: {argv}")

    with patch("subprocess.run", side_effect=_runner):
        host = detect_host_cuda()
    assert host == HostCuda(version=(12, 4), source="nvcc", raw="release 12.4")


@pytest.mark.logic
def test_detect_host_cuda_returns_none_when_both_probes_missing() -> None:
    """No CUDA tooling at all → source="none" + version=None."""
    with patch("subprocess.run", side_effect=FileNotFoundError("missing")):
        host = detect_host_cuda()
    assert host == HostCuda(version=None, source="none", raw="")


@pytest.mark.logic
def test_detect_host_cuda_returns_none_when_nvidia_smi_succeeds_without_cuda_line() -> None:
    """Defensive: a future nvidia-smi that drops the header line falls through to nvcc, then none."""

    def _runner(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        argv = args
        if argv and argv[0] == "nvidia-smi":
            return _fake_run(0, stdout="some other output without the cuda line")
        if argv and argv[0] == "nvcc":
            raise FileNotFoundError("no nvcc either")
        raise AssertionError(f"unexpected invocation: {argv}")

    with patch("subprocess.run", side_effect=_runner):
        host = detect_host_cuda()
    assert host.version is None
    assert host.source == "none"


@pytest.mark.logic
def test_detect_host_cuda_handles_timeout() -> None:
    """A hung nvidia-smi (rare but possible) must not block forever."""
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)):
        host = detect_host_cuda()
    assert host.version is None
    assert host.source == "none"


# ── pick_torch_index ─────────────────────────────────────────────────


@pytest.mark.logic
def test_pick_torch_index_cuda_12_9_chooses_cu128() -> None:
    """The headline case: a CUDA-12.9 host must route through the cu128 index."""
    host = HostCuda(version=(12, 9), source="nvidia-smi", raw="CUDA Version: 12.9")
    idx = pick_torch_index(host)
    assert idx.tag == "cu128"
    assert idx.url == "https://download.pytorch.org/whl/cu128"
    assert idx.cuda_version == (12, 8)
    assert idx.source == "static-map"


@pytest.mark.logic
def test_pick_torch_index_cuda_12_8_chooses_cu128() -> None:
    """Exact match on a map entry returns that entry."""
    host = HostCuda(version=(12, 8), source="nvidia-smi", raw="CUDA Version: 12.8")
    idx = pick_torch_index(host)
    assert idx.tag == "cu128"


@pytest.mark.logic
def test_pick_torch_index_cuda_12_1_chooses_cu121() -> None:
    """Lower CUDA versions resolve to lower index entries."""
    host = HostCuda(version=(12, 1), source="nvcc", raw="release 12.1")
    idx = pick_torch_index(host)
    assert idx.tag == "cu121"
    assert idx.cuda_version == (12, 1)


@pytest.mark.logic
def test_pick_torch_index_cuda_11_8_below_map_falls_to_cpu() -> None:
    """Hosts older than every map entry get CPU wheels — better than crashing."""
    host = HostCuda(version=(11, 8), source="nvcc", raw="release 11.8")
    idx = pick_torch_index(host)
    assert idx.tag == "cpu"
    assert idx.cuda_version is None


@pytest.mark.logic
def test_pick_torch_index_no_cuda_chooses_cpu() -> None:
    """When the host has no CUDA at all, the picker returns the CPU index."""
    host = HostCuda(version=None, source="none", raw="")
    idx = pick_torch_index(host)
    assert idx.tag == "cpu"
    assert idx.url == "https://download.pytorch.org/whl/cpu"
    assert idx.source == "static-map"


@pytest.mark.logic
def test_pick_torch_index_ceiling_caps_modern_host_to_cu121() -> None:
    """The brouhaha case: a CUDA-12.8 host with a (12, 1) ceiling routes cu121, not cu128.

    brouhaha pins torch<2.3, which has no wheel on cu128; the ceiling forces the
    newest index that still carries it.
    """
    host = HostCuda(version=(12, 8), source="nvidia-smi", raw="CUDA Version: 12.8")
    idx = pick_torch_index(host, max_cuda_version=(12, 1))
    assert idx.tag == "cu121"
    assert idx.cuda_version == (12, 1)
    assert idx.source == "static-map"


@pytest.mark.logic
def test_pick_torch_index_ceiling_above_host_has_no_effect() -> None:
    """A ceiling higher than the host CUDA never raises the selected index."""
    host = HostCuda(version=(12, 1), source="nvcc", raw="release 12.1")
    idx = pick_torch_index(host, max_cuda_version=(12, 6))
    assert idx.tag == "cu121"


@pytest.mark.logic
def test_pick_torch_index_ceiling_ignored_when_host_has_no_cuda() -> None:
    """On a CPU-only host the ceiling is moot — the CPU index is returned."""
    host = HostCuda(version=None, source="none", raw="")
    idx = pick_torch_index(host, max_cuda_version=(12, 1))
    assert idx.tag == "cpu"


@pytest.mark.logic
def test_pick_torch_index_env_override_beats_ceiling() -> None:
    """An explicit operator override wins over both host detection and the ceiling."""
    host = HostCuda(version=(12, 8), source="nvidia-smi", raw="CUDA Version: 12.8")
    idx = pick_torch_index(host, env_override="https://mirror.example/cu999", max_cuda_version=(12, 1))
    assert idx.tag == "override"
    assert idx.url == "https://mirror.example/cu999"


@pytest.mark.logic
def test_pick_torch_index_env_override_wins_over_host_cuda() -> None:
    """SENSELAB_TORCH_INDEX_URL must short-circuit the static map even on CUDA hosts."""
    host = HostCuda(version=(12, 9), source="nvidia-smi", raw="CUDA Version: 12.9")
    idx = pick_torch_index(host, env_override="https://pypi.internal.example.com/pytorch/cu128")
    assert idx.tag == "override"
    assert idx.url == "https://pypi.internal.example.com/pytorch/cu128"
    assert idx.source == "env-override"
    assert idx.cuda_version is None


@pytest.mark.logic
def test_pick_torch_index_empty_string_override_treated_as_unset() -> None:
    """Bare ``SENSELAB_TORCH_INDEX_URL=`` (empty) must not be honored as an override."""
    host = HostCuda(version=(12, 9), source="nvidia-smi", raw="CUDA Version: 12.9")
    idx = pick_torch_index(host, env_override="")
    assert idx.tag == "cu128"
    assert idx.source == "static-map"


# ── SenselabCudaCompatibilityError ───────────────────────────────────


@pytest.mark.logic
def test_compatibility_error_message_includes_diagnostic_fields() -> None:
    """The error message names host CUDA, attempted index, failing packages, and the recommended action."""
    host = HostCuda(version=(12, 9), source="nvidia-smi", raw="CUDA Version: 12.9")
    idx = TorchIndex(
        url="https://download.pytorch.org/whl/cu128",
        tag="cu128",
        cuda_version=(12, 8),
        source="static-map",
    )
    err = SenselabCudaCompatibilityError(
        host_cuda=host,
        attempted_index=idx,
        failing_packages=["torch>=2.8,<2.9", "torchaudio>=2.8,<2.9"],
    )
    msg = str(err)
    assert "12.9" in msg
    assert "nvidia-smi" in msg
    assert "https://download.pytorch.org/whl/cu128" in msg
    assert "torch>=2.8,<2.9" in msg
    assert "SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu" in msg


@pytest.mark.logic
def test_compatibility_error_message_handles_no_cuda_host() -> None:
    """The error formatter doesn't crash when there is no host CUDA version to print."""
    host = HostCuda(version=None, source="none", raw="")
    idx = TorchIndex(url="https://download.pytorch.org/whl/cpu", tag="cpu", cuda_version=None, source="static-map")
    err = SenselabCudaCompatibilityError(host_cuda=host, attempted_index=idx, failing_packages=["torch>=2.8,<2.9"])
    msg = str(err)
    assert "none" in msg
    assert "cpu" in msg


# ── pytest collection sanity check ───────────────────────────────────


@pytest.mark.logic
def test_module_exposes_expected_names() -> None:
    """Guard against accidental rename of any public name from hardware."""
    from senselab_core.runtime import hardware

    for name in ("HostCuda", "TorchIndex", "SenselabCudaCompatibilityError", "detect_host_cuda", "pick_torch_index"):
        assert hasattr(hardware, name), f"hardware missing public name: {name}"


@pytest.mark.logic
def test_pick_torch_index_cuda_13_stays_on_cu128_without_a_declared_ceiling() -> None:
    """A CUDA-13 host routes every venv that declares no ceiling to cu128, the default ceiling."""
    host = HostCuda(version=(13, 1), source="nvidia-smi", raw="CUDA Version: 13.1")
    assert pick_torch_index(host).tag == "cu128"


@pytest.mark.logic
def test_pick_torch_index_a_declared_cuda_13_ceiling_routes_cu130() -> None:
    """A venv whose wheels need CUDA 13 (vLLM's torch) declares (13, 0) and gets cu130 on a CUDA-13 host."""
    host = HostCuda(version=(13, 1), source="nvidia-smi", raw="CUDA Version: 13.1")
    assert pick_torch_index(host, max_cuda_version=(13, 0)).tag == "cu130"
    older = HostCuda(version=(12, 9), source="nvidia-smi", raw="CUDA Version: 12.9")
    assert pick_torch_index(older, max_cuda_version=(13, 0)).tag == "cu128"


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
