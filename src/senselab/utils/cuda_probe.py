"""Host CUDA probe and PyTorch wheel index picker for subprocess venvs.

Used by ``ensure_venv`` to route ``torch`` and ``torchaudio`` installs through
the matching official PyTorch wheel index (``cu128`` / ``cu126`` / ``cu124`` /
``cu121`` / ``cpu``), avoiding the ABI mismatch where the two libraries end
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
import re
import subprocess
from dataclasses import dataclass
from typing import Literal, Optional

logger = logging.getLogger("senselab")

# Static map of supported PyTorch wheel indexes. Update this list when
# PyTorch publishes a new ``cuXX`` index (e.g. ``cu129``). Sorted at
# module load by CUDA version descending — the picker iterates in that
# order and picks the first whose CUDA is ``<=`` the host's. The
# ``sorted`` call defends against future contributors appending an entry
# in the wrong position.
_PYTORCH_INDEX_MAP: list[tuple[str, tuple[int, int]]] = sorted(
    [
        ("cu128", (12, 8)),
        ("cu126", (12, 6)),
        ("cu124", (12, 4)),
        ("cu121", (12, 1)),
    ],
    key=lambda entry: entry[1],
    reverse=True,
)
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
) -> TorchIndex:
    """Choose the PyTorch wheel index URL for this host.

    Order:

    1. If ``env_override`` is set + non-empty → return that URL verbatim with
       ``tag="override"``, ``source="env-override"``. Highest priority.
    2. If host has no CUDA → CPU index.
    3. Pick the highest entry in the static map whose CUDA version is ``<=``
       the host's. No match → CPU index.
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
    for tag, idx_version in _PYTORCH_INDEX_MAP:
        if idx_version <= host_cuda.version:
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
