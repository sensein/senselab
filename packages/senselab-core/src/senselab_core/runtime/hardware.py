"""Host hardware probe, CUDA probe and PyTorch wheel index picker.

``probe_hardware`` reports CPU count, memory, GPUs and platform without importing ``torch``;
``numerics_class`` reduces that to the portable facts an execution context records.

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
from typing import Callable, Literal, Mapping, Optional

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
_GPU_QUERY_TIMEOUT_S = 30.0
_NVIDIA_SMI_FIELDS = "index,uuid,name,memory.total,driver_version"
_NVIDIA_SMI_FORMAT = "--format=csv,noheader,nounits"
_NVIDIA_SMI_GPU_QUERY = ["nvidia-smi", f"--query-gpu={_NVIDIA_SMI_FIELDS},compute_cap", _NVIDIA_SMI_FORMAT]
_NVIDIA_SMI_GPU_QUERY_NO_CAP = ["nvidia-smi", f"--query-gpu={_NVIDIA_SMI_FIELDS}", _NVIDIA_SMI_FORMAT]
_CHIP_QUERY = ["sysctl", "-n", "machdep.cpu.brand_string"]


class DeviceResolutionError(ValueError):
    """A ``cuda:N`` device that cannot be mapped to a GPU on this host."""


@dataclass(frozen=True)
class GpuInfo:
    """One GPU as ``nvidia-smi`` reports it; ``memory_bytes`` is total device memory.

    ``compute_capability`` is ``"<major>.<minor>"`` when the driver reports it; ``index`` and ``uuid`` are
    the identifiers ``CUDA_VISIBLE_DEVICES`` can name.
    """

    name: str
    memory_bytes: int
    driver: Optional[str]
    compute_capability: Optional[str] = None
    uuid: Optional[str] = None
    index: Optional[int] = None


@dataclass(frozen=True)
class HardwareInfo:
    """Host hardware facts, gathered without importing ``torch``.

    ``gpu_probe_error`` is None when the GPU probe found nothing wrong, including when ``nvidia-smi`` is
    absent. It carries a message when ``nvidia-smi`` could not run, exited non-zero, timed out or printed
    a line that could not be parsed (``gpus`` is then empty), or when the driver rejected ``compute_cap``
    (``gpus`` is then present without capabilities). ``probe_error`` carries a failed memory or chip
    probe; ``memory_bytes`` is 0 and ``chip`` None in that case. ``chip`` is the Apple silicon brand
    string, e.g. ``"Apple M2 Max"``.
    """

    cpu_count: int
    memory_bytes: int
    gpus: tuple[GpuInfo, ...]
    apple_silicon: bool
    platform: str
    gpu_probe_error: Optional[str] = None
    chip: Optional[str] = None
    probe_error: Optional[str] = None


def _run(cmd: list[str]) -> str:
    """Run ``cmd`` and return its stdout; raise on a missing binary, a timeout or a non-zero exit."""
    return subprocess.run(cmd, capture_output=True, text=True, timeout=_GPU_QUERY_TIMEOUT_S, check=True).stdout


def _parse_gpus(output: str, *, with_capability: bool) -> tuple[GpuInfo, ...]:
    """Parse the ``nvidia-smi`` CSV lines; raise ``ValueError`` on any malformed line."""
    width = 6 if with_capability else 5
    gpus = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [f.strip() for f in line.split(",")]
        if len(fields) != width or not fields[0].isdigit() or not fields[2] or not fields[3].isdigit():
            raise ValueError(f"unparseable nvidia-smi line: {line!r}")
        capability = fields[5] if with_capability and re.fullmatch(r"\d+\.\d+", fields[5]) else None
        gpus.append(
            GpuInfo(
                name=fields[2],
                memory_bytes=int(fields[3]) * _MIB,
                driver=fields[4] or None,
                compute_capability=capability,
                uuid=fields[1] or None,
                index=int(fields[0]),
            )
        )
    return tuple(gpus)


def _probe_gpus(run: Callable[[list[str]], str]) -> tuple[tuple[GpuInfo, ...], Optional[str]]:
    note: Optional[str] = None
    query, with_capability = _NVIDIA_SMI_GPU_QUERY, True
    try:
        try:
            output = run(query)
        except subprocess.CalledProcessError:
            query, with_capability = _NVIDIA_SMI_GPU_QUERY_NO_CAP, False
            output = run(query)
            note = "nvidia-smi rejected compute_cap; compute capability unknown"
            logger.debug(note)
    except FileNotFoundError:
        return (), None
    except PermissionError as exc:
        message = f"nvidia-smi could not be executed: {exc}"
    except subprocess.CalledProcessError as exc:
        message = f"nvidia-smi exited {exc.returncode}"
    except subprocess.TimeoutExpired:
        message = "nvidia-smi timed out"
    else:
        try:
            return _parse_gpus(output, with_capability=with_capability), note
        except ValueError as exc:
            message = str(exc)
    logger.warning("GPU probe found no GPUs: %s", message)
    return (), message


def _physical_memory_bytes() -> tuple[int, Optional[str]]:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES")), None
    except (ValueError, OSError, AttributeError) as exc:
        message = f"physical memory unavailable: {exc}"
        logger.warning(message)
        return 0, message


def _probe_chip(run: Callable[[list[str]], str]) -> tuple[Optional[str], Optional[str]]:
    try:
        chip = run(_CHIP_QUERY).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        message = f"Apple chip unavailable: {exc}"
        logger.warning(message)
        return None, message
    return chip or None, None if chip else "Apple chip unavailable: empty sysctl output"


def probe_hardware(*, run: Callable[[list[str]], str] = _run) -> HardwareInfo:
    """Probe CPU, memory, GPUs, Apple chip and platform; never imports ``torch``.

    ``run`` executes a command (the ``nvidia-smi`` query, ``sysctl`` on Apple silicon) and returns
    stdout; tests inject it.
    """
    system = platform.system()
    machine = platform.machine()
    gpus, gpu_error = _probe_gpus(run)
    memory, memory_error = _physical_memory_bytes()
    apple_silicon = system == "Darwin" and machine == "arm64"
    chip, chip_error = _probe_chip(run) if apple_silicon else (None, None)
    errors = [e for e in (memory_error, chip_error) if e]
    return HardwareInfo(
        cpu_count=os.cpu_count() or 1,
        memory_bytes=memory,
        gpus=gpus,
        apple_silicon=apple_silicon,
        platform=f"{system.lower()}-{machine}",
        gpu_probe_error=gpu_error,
        chip=chip,
        probe_error="; ".join(errors) or None,
    )


def _visible_gpus(info: HardwareInfo, environ: Mapping[str, str]) -> list[GpuInfo]:
    """The GPUs torch would enumerate: all of them, or those ``CUDA_VISIBLE_DEVICES`` names, in its order."""
    raw = environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return list(info.gpus)
    visible = []
    for entry in (e.strip() for e in raw.split(",") if e.strip()):
        if entry.isdigit():
            matches = [g for g in info.gpus if g.index == int(entry)]
        elif entry.upper().startswith("GPU-"):
            matches = [g for g in info.gpus if g.uuid and g.uuid.upper().startswith(entry.upper())]
        else:
            raise DeviceResolutionError(f"CUDA_VISIBLE_DEVICES entry {entry!r} is neither an index nor a GPU uuid")
        if len(matches) != 1:
            raise DeviceResolutionError(f"CUDA_VISIBLE_DEVICES entry {entry!r} matches {len(matches)} GPUs")
        visible.append(matches[0])
    return visible


def numerics_class(
    info: HardwareInfo, device: str, precision: str, *, environ: Optional[Mapping[str, str]] = None
) -> dict[str, str]:
    """Portable numerics facts for an execution context: device kind, precision, architecture.

    ``device`` is a torch-style string (``cpu``, ``cuda``, ``cuda:1``, ``mps``). ``cuda:N`` indexes the
    GPUs visible under ``CUDA_VISIBLE_DEVICES`` (read from ``environ``, default ``os.environ``);
    ``gpu_architecture`` is ``sm_<major><minor>``, or the product name when the compute capability is
    unknown. On ``mps`` it is the Apple chip, omitted when the chip is unknown. The result never feeds a
    logical key.

    Raises:
        DeviceResolutionError: ``cuda:N`` is out of range, malformed, or names a visible-device entry that
            cannot be mapped.
    """
    kind, _, index = device.partition(":")
    facts = {"device_kind": kind, "precision": precision}
    if kind == "cuda":
        if index and not index.isdigit():
            raise DeviceResolutionError(f"unparseable device {device!r}")
        visible = _visible_gpus(info, os.environ if environ is None else environ)
        position = int(index) if index else 0
        if position >= len(visible):
            raise DeviceResolutionError(f"{device!r} is out of range: {len(visible)} visible GPU(s)")
        gpu = visible[position]
        capability = gpu.compute_capability
        facts["gpu_architecture"] = f"sm_{capability.replace('.', '')}" if capability else gpu.name
    elif kind == "mps" and info.chip:
        facts["gpu_architecture"] = info.chip
    return facts
