"""Unit tests for ``senselab.utils.cuda_probe``.

All ``subprocess.run`` calls are mocked — these tests must run on CPU-only
CI without shelling out to ``nvidia-smi`` or ``nvcc``.
"""

import subprocess
from unittest.mock import patch

from senselab.utils.cuda_probe import (
    HostCuda,
    SenselabCudaCompatibilityError,
    TorchIndex,
    detect_host_cuda,
    pick_torch_index,
)


def _fake_run(returncode: int, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ── detect_host_cuda ─────────────────────────────────────────────────


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


def test_detect_host_cuda_falls_through_to_nvcc_when_nvidia_smi_missing() -> None:
    """When nvidia-smi raises FileNotFoundError, nvcc must be tried next."""
    nvcc_stdout = "nvcc: NVIDIA (R) Cuda compiler driver\nrelease 12.4, V12.4.131\n"

    def _runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        argv = args[0]
        if argv and argv[0] == "nvidia-smi":
            raise FileNotFoundError("no nvidia-smi")
        if argv and argv[0] == "nvcc":
            return _fake_run(0, stdout=nvcc_stdout)
        raise AssertionError(f"unexpected invocation: {argv}")

    with patch("subprocess.run", side_effect=_runner):
        host = detect_host_cuda()
    assert host == HostCuda(version=(12, 4), source="nvcc", raw="release 12.4")


def test_detect_host_cuda_returns_none_when_both_probes_missing() -> None:
    """No CUDA tooling at all → source="none" + version=None."""
    with patch("subprocess.run", side_effect=FileNotFoundError("missing")):
        host = detect_host_cuda()
    assert host == HostCuda(version=None, source="none", raw="")


def test_detect_host_cuda_returns_none_when_nvidia_smi_succeeds_without_cuda_line() -> None:
    """Defensive: a future nvidia-smi that drops the header line falls through to nvcc, then none."""

    def _runner(*args: object, **kwargs: object) -> subprocess.CompletedProcess:
        argv = args[0]
        if argv and argv[0] == "nvidia-smi":
            return _fake_run(0, stdout="some other output without the cuda line")
        if argv and argv[0] == "nvcc":
            raise FileNotFoundError("no nvcc either")
        raise AssertionError(f"unexpected invocation: {argv}")

    with patch("subprocess.run", side_effect=_runner):
        host = detect_host_cuda()
    assert host.version is None
    assert host.source == "none"


def test_detect_host_cuda_handles_timeout() -> None:
    """A hung nvidia-smi (rare but possible) must not block forever."""
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)):
        host = detect_host_cuda()
    assert host.version is None
    assert host.source == "none"


# ── pick_torch_index ─────────────────────────────────────────────────


def test_pick_torch_index_cuda_12_9_chooses_cu128() -> None:
    """The headline case: a CUDA-12.9 host must route through the cu128 index."""
    host = HostCuda(version=(12, 9), source="nvidia-smi", raw="CUDA Version: 12.9")
    idx = pick_torch_index(host)
    assert idx.tag == "cu128"
    assert idx.url == "https://download.pytorch.org/whl/cu128"
    assert idx.cuda_version == (12, 8)
    assert idx.source == "static-map"


def test_pick_torch_index_cuda_12_8_chooses_cu128() -> None:
    """Exact match on a map entry returns that entry."""
    host = HostCuda(version=(12, 8), source="nvidia-smi", raw="CUDA Version: 12.8")
    idx = pick_torch_index(host)
    assert idx.tag == "cu128"


def test_pick_torch_index_cuda_12_1_chooses_cu121() -> None:
    """Lower CUDA versions resolve to lower index entries."""
    host = HostCuda(version=(12, 1), source="nvcc", raw="release 12.1")
    idx = pick_torch_index(host)
    assert idx.tag == "cu121"
    assert idx.cuda_version == (12, 1)


def test_pick_torch_index_cuda_11_8_below_map_falls_to_cpu() -> None:
    """Hosts older than every map entry get CPU wheels — better than crashing."""
    host = HostCuda(version=(11, 8), source="nvcc", raw="release 11.8")
    idx = pick_torch_index(host)
    assert idx.tag == "cpu"
    assert idx.cuda_version is None


def test_pick_torch_index_no_cuda_chooses_cpu() -> None:
    """When the host has no CUDA at all, the picker returns the CPU index."""
    host = HostCuda(version=None, source="none", raw="")
    idx = pick_torch_index(host)
    assert idx.tag == "cpu"
    assert idx.url == "https://download.pytorch.org/whl/cpu"
    assert idx.source == "static-map"


def test_pick_torch_index_env_override_wins_over_host_cuda() -> None:
    """SENSELAB_TORCH_INDEX_URL must short-circuit the static map even on CUDA hosts."""
    host = HostCuda(version=(12, 9), source="nvidia-smi", raw="CUDA Version: 12.9")
    idx = pick_torch_index(host, env_override="https://pypi.internal.example.com/pytorch/cu128")
    assert idx.tag == "override"
    assert idx.url == "https://pypi.internal.example.com/pytorch/cu128"
    assert idx.source == "env-override"
    assert idx.cuda_version is None


def test_pick_torch_index_empty_string_override_treated_as_unset() -> None:
    """Bare ``SENSELAB_TORCH_INDEX_URL=`` (empty) must not be honored as an override."""
    host = HostCuda(version=(12, 9), source="nvidia-smi", raw="CUDA Version: 12.9")
    idx = pick_torch_index(host, env_override="")
    assert idx.tag == "cu128"
    assert idx.source == "static-map"


# ── SenselabCudaCompatibilityError ───────────────────────────────────


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


def test_compatibility_error_message_handles_no_cuda_host() -> None:
    """The error formatter doesn't crash when there is no host CUDA version to print."""
    host = HostCuda(version=None, source="none", raw="")
    idx = TorchIndex(url="https://download.pytorch.org/whl/cpu", tag="cpu", cuda_version=None, source="static-map")
    err = SenselabCudaCompatibilityError(host_cuda=host, attempted_index=idx, failing_packages=["torch>=2.8,<2.9"])
    msg = str(err)
    assert "none" in msg
    assert "cpu" in msg


# ── pytest collection sanity check ───────────────────────────────────


def test_module_exposes_expected_names() -> None:
    """Guard against accidental rename of any public name from cuda_probe."""
    from senselab.utils import cuda_probe

    for name in ("HostCuda", "TorchIndex", "SenselabCudaCompatibilityError", "detect_host_cuda", "pick_torch_index"):
        assert hasattr(cuda_probe, name), f"cuda_probe missing public name: {name}"
