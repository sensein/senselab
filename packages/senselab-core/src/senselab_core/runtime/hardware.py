"""Host hardware probe, CUDA probe and PyTorch wheel index picker.

``probe_hardware`` reports CPU count, memory, GPUs and platform without importing ``torch``;
``numerics_class`` reduces that to the portable facts an execution context records.

Host CUDA probe and PyTorch wheel index picker for subprocess venvs.

Used by ``ensure_venv`` to route ``torch`` and ``torchaudio`` installs through
the matching official PyTorch wheel index (``cu130`` / ``cu128`` / ``cu126`` /
``cu124`` / ``cu121`` / ``cpu``), avoiding the ABI mismatch where the two libraries end
up compiled for different CUDA toolchains.

The probe must work without ``torch`` already installed (it runs *before*
the target venv exists), so we parse ``nvidia-smi`` / ``nvcc`` output
directly. No driver→CUDA mapping table to maintain — both tools report the
CUDA version as a parseable string. Operators with internal PyPI mirrors
or unusual hosts can override the chosen index via the
``SENSELAB_TORCH_INDEX_URL`` environment variable.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import subprocess
from dataclasses import dataclass
from typing import Callable, Literal, Optional

logger = logging.getLogger("senselab")

# Static map of supported PyTorch wheel indexes. Update this list when
# PyTorch publishes a new ``cuXX`` index (e.g. ``cu129``). Sorted at
# module load by CUDA version descending — the picker iterates in that
# order and picks the first whose CUDA is ``<=`` the host's. The
# ``sorted`` call defends against future contributors appending an entry
# in the wrong position.
_PYTORCH_INDEX_MAP: list[tuple[str, tuple[int, int]]] = sorted(
    [
        ("cu130", (13, 0)),
        ("cu128", (12, 8)),
        ("cu126", (12, 6)),
        ("cu124", (12, 4)),
        ("cu121", (12, 1)),
    ],
    key=lambda entry: entry[1],
    reverse=True,
)
DEFAULT_MAX_CUDA: tuple[int, int] = (12, 8)
"""The CUDA index ceiling of a venv that declares none; a venv opts into a newer index by declaring it."""
_PYTORCH_INDEX_BASE = "https://download.pytorch.org/whl"
_CPU_INDEX_URL = f"{_PYTORCH_INDEX_BASE}/cpu"
_PROBE_TIMEOUT_S = 5.0

# Regex to extract ``CUDA Version: X.Y`` from nvidia-smi's default header.
_NVIDIA_SMI_CUDA_RE = re.compile(r"CUDA Version:\s*(\d+)\.(\d+)")
# Regex to extract ``release X.Y`` from ``nvcc --version`` output.
_NVCC_RELEASE_RE = re.compile(r"release\s+(\d+)\.(\d+)")


@dataclass(frozen=True)
class HostCuda:
    """Result of probing the host's runtime CUDA capability."""

    version: Optional[tuple[int, int]]
    source: Literal["nvidia-smi", "nvcc", "none"]
    raw: str


@dataclass(frozen=True)
class TorchIndex:
    """Chosen PyTorch wheel index for ``uv pip install``."""

    url: str
    tag: str
    cuda_version: Optional[tuple[int, int]]
    source: Literal["static-map", "env-override"]


class SenselabCudaCompatibilityError(RuntimeError):
    """Raised when no torch+torchaudio binary pair is installable on this host.

    Carries the diagnostic context the operator needs to recover: the host
    CUDA version, the index URL we tried, the failing packages parsed from
    uv's stderr, and a one-line action.
    """

    def __init__(
        self,
        host_cuda: HostCuda,
        attempted_index: TorchIndex,
        failing_packages: list[str],
    ) -> None:
        """Capture diagnostic context and format the user-facing message."""
        self.host_cuda = host_cuda
        self.attempted_index = attempted_index
        self.failing_packages = failing_packages
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        host_str = (
            f"{self.host_cuda.version[0]}.{self.host_cuda.version[1]} (source: {self.host_cuda.source})"
            if self.host_cuda.version is not None
            else f"none (source: {self.host_cuda.source})"
        )
        return (
            "No torch+torchaudio wheel available for this host.\n"
            f"Host CUDA: {host_str}\n"
            f"Attempted index: {self.attempted_index.url}\n"
            f"Failing packages: {self.failing_packages}\n"
            "Action: downgrade CUDA, set SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu "
            "for CPU fallback, or wait for upstream wheels."
        )


def detect_host_cuda() -> HostCuda:
    """Probe the host's runtime CUDA via ``nvidia-smi`` then ``nvcc``.

    Returns ``HostCuda(version=None, source="none", raw="")`` when no probe
    succeeds. Both probes parse a CUDA version directly out of stdout, so no
    driver-to-CUDA mapping is required.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result = None
    if result is not None and result.returncode == 0:
        match = _NVIDIA_SMI_CUDA_RE.search(result.stdout)
        if match:
            return HostCuda(
                version=(int(match.group(1)), int(match.group(2))),
                source="nvidia-smi",
                raw=match.group(0),
            )

    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        result = None
    if result is not None and result.returncode == 0:
        match = _NVCC_RELEASE_RE.search(result.stdout)
        if match:
            return HostCuda(
                version=(int(match.group(1)), int(match.group(2))),
                source="nvcc",
                raw=match.group(0),
            )

    return HostCuda(version=None, source="none", raw="")


def pick_torch_index(
    host_cuda: HostCuda,
    env_override: Optional[str] = None,
    max_cuda_version: Optional[tuple[int, int]] = None,
) -> TorchIndex:
    """Choose the PyTorch wheel index URL for this host.

    Order:

    1. If ``env_override`` is set + non-empty → return that URL verbatim with
       ``tag="override"``, ``source="env-override"``. Highest priority.
    2. If host has no CUDA → CPU index.
    3. Pick the highest entry in the static map whose CUDA version is ``<=``
       the *effective* CUDA version, where the effective version is the host's
       capped by ``max_cuda_version`` (see below). No match → CPU index.

    Args:
        host_cuda: Probed host CUDA capability.
        env_override: Operator override (``SENSELAB_TORCH_INDEX_URL``); wins outright.
        max_cuda_version: Per-venv ceiling on the CUDA index. A backend whose
            ``torch`` pin has no wheel on the newest index the host would
            otherwise select must declare the newest index that *does* carry
            its wheels. Example: brouhaha pins ``torch>=2.0,<2.3``; the
            ``cu128`` index (selected on a CUDA-12.8 host) publishes no
            ``torch<2.3`` linux-x86_64 wheel, so the Stage-1 install is
            unsatisfiable. Declaring ``(12, 1)`` caps the selection to
            ``cu121`` (which carries ``torch==2.2.2+cu121`` + matching
            ``torchaudio``), forward-compatible with 12.x/13.x drivers for
            inference. ``None`` (default) applies :data:`DEFAULT_MAX_CUDA`, so
            backends pinning modern torch (e.g. ``>=2.8``) keep ``cu128``; a
            backend whose wheels need a newer index (vLLM's ``cu130`` torch)
            declares it, e.g. ``(13, 0)``.
    """
    if env_override:
        return TorchIndex(
            url=env_override,
            tag="override",
            cuda_version=None,
            source="env-override",
        )
    if host_cuda.version is None:
        return TorchIndex(
            url=_CPU_INDEX_URL,
            tag="cpu",
            cuda_version=None,
            source="static-map",
        )
    # Cap the host CUDA version to the venv's ceiling so a backend pinning an
    # older torch is not routed through an index with no compatible wheel.
    effective = host_cuda.version
    ceiling = max_cuda_version if max_cuda_version is not None else DEFAULT_MAX_CUDA
    if ceiling < effective:
        effective = ceiling
    for tag, idx_version in _PYTORCH_INDEX_MAP:
        if idx_version <= effective:
            return TorchIndex(
                url=f"{_PYTORCH_INDEX_BASE}/{tag}",
                tag=tag,
                cuda_version=idx_version,
                source="static-map",
            )
    return TorchIndex(
        url=_CPU_INDEX_URL,
        tag="cpu",
        cuda_version=None,
        source="static-map",
    )


_MIB = 1024 * 1024
_NVIDIA_SMI_GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,memory.total,driver_version",
    "--format=csv,noheader,nounits",
]


@dataclass(frozen=True)
class GpuInfo:
    """One GPU as ``nvidia-smi`` reports it; ``memory_bytes`` is total device memory."""

    name: str
    memory_bytes: int
    driver: Optional[str]


@dataclass(frozen=True)
class HardwareInfo:
    """Host hardware facts, gathered without importing ``torch``.

    ``gpu_probe_error`` is None when the GPU probe found nothing wrong, including when ``nvidia-smi`` is
    absent. It carries a message when ``nvidia-smi`` exited non-zero or printed a line that could not be
    parsed; ``gpus`` is then empty.
    """

    cpu_count: int
    memory_bytes: int
    gpus: tuple[GpuInfo, ...]
    apple_silicon: bool
    platform: str
    gpu_probe_error: Optional[str] = None


def _run(cmd: list[str]) -> str:
    """Run ``cmd`` and return its stdout; raise on a missing binary, a timeout or a non-zero exit."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=_PROBE_TIMEOUT_S, check=True).stdout


def _parse_gpus(output: str) -> tuple[GpuInfo, ...]:
    """Parse ``name, memory MiB, driver`` CSV lines; raise ``ValueError`` on any malformed line."""
    gpus = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != 3 or not fields[0] or not fields[1].isdigit():
            raise ValueError(f"unparseable nvidia-smi line: {line!r}")
        gpus.append(GpuInfo(name=fields[0], memory_bytes=int(fields[1]) * _MIB, driver=fields[2] or None))
    return tuple(gpus)


def _probe_gpus(run: Callable[[list[str]], str]) -> tuple[tuple[GpuInfo, ...], Optional[str]]:
    try:
        output = run(_NVIDIA_SMI_GPU_QUERY)
    except (FileNotFoundError, PermissionError):
        return (), None
    except subprocess.CalledProcessError as exc:
        message = f"nvidia-smi exited {exc.returncode}"
    except subprocess.TimeoutExpired:
        message = "nvidia-smi timed out"
    else:
        try:
            return _parse_gpus(output), None
        except ValueError as exc:
            message = str(exc)
    logger.warning("GPU probe found no GPUs: %s", message)
    return (), message


def _physical_memory_bytes() -> int:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        return 0


def probe_hardware(*, run: Callable[[list[str]], str] = _run) -> HardwareInfo:
    """Probe CPU, memory, GPUs and platform; never imports ``torch``.

    ``run`` executes the ``nvidia-smi`` query and returns stdout; tests inject it.
    """
    system = platform.system()
    machine = platform.machine()
    gpus, error = _probe_gpus(run)
    return HardwareInfo(
        cpu_count=os.cpu_count() or 1,
        memory_bytes=_physical_memory_bytes(),
        gpus=gpus,
        apple_silicon=system == "Darwin" and machine == "arm64",
        platform=f"{system.lower()}-{machine}",
        gpu_probe_error=error,
    )


def numerics_class(info: HardwareInfo, device: str, precision: str) -> dict[str, str]:
    """Portable numerics facts for an execution context: device kind, precision, GPU architecture.

    ``device`` is a torch-style string (``cpu``, ``cuda``, ``cuda:1``, ``mps``). The GPU name stands in
    for the architecture on ``cuda`` devices; ``mps`` reports ``apple-silicon``. The result never feeds
    a logical key.
    """
    kind, _, index = device.partition(":")
    facts = {"device_kind": kind, "precision": precision}
    if kind == "cuda" and info.gpus:
        position = int(index) if index.isdigit() and int(index) < len(info.gpus) else 0
        facts["gpu_architecture"] = info.gpus[position].name
    elif kind == "mps" and info.apple_silicon:
        facts["gpu_architecture"] = "apple-silicon"
    return facts
