"""Host hardware probe.

``probe_hardware`` reports CPU count, memory, GPUs and platform without importing ``torch``;
``numerics_class`` reduces that to the portable facts an execution context records.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import resource
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Mapping, Optional

logger = logging.getLogger("senselab")

_MIB = 1024 * 1024
_KIB: Final = 1024  # invariant: bytes in a kibibyte, the unit of Linux ru_maxrss
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


CGROUP_ROOT = Path("/sys/fs/cgroup")
"""Where cgroup v2 is mounted."""


def cgroup_memory_events(proc_cgroup: Path = Path("/proc/self/cgroup")) -> Optional[Path]:
    """Return the `memory.events` file of this process's cgroup v2, None where there is none."""
    try:
        lines = proc_cgroup.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in lines:
        if line.startswith("0::"):
            path = CGROUP_ROOT / line[3:].lstrip("/") / "memory.events"
            return path if path.is_file() else None
    return None


def oom_kills(memory_events: Optional[Path]) -> Optional[int]:
    """Return the `oom_kill` count a cgroup `memory.events` file records, None when it cannot be read."""
    if memory_events is None:
        return None
    try:
        lines = memory_events.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    counts = dict(line.split(maxsplit=1) for line in lines if len(line.split()) == 2)  # noqa: PLR2004 - name, count
    found = counts.get("oom_kill", "")
    return int(found) if found.isdigit() else None


def memory_bytes(proc_cgroup: Path = Path("/proc/self/cgroup")) -> tuple[int, int]:
    """Return this process's memory now and its high-water, in bytes.

    From its cgroup v2 (`memory.current`, `memory.peak`), which counts every process of the cgroup; elsewhere the
    largest resident set of this process and of its waited-for children, for both.
    """
    events = cgroup_memory_events(proc_cgroup)
    if events is not None:
        try:
            current = int((events.parent / "memory.current").read_text(encoding="utf-8"))
            peak = int((events.parent / "memory.peak").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        else:
            return current, max(current, peak)
    unit = 1 if platform.system() == "Darwin" else _KIB
    used = unit * max(resource.getrusage(who).ru_maxrss for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN))
    return used, used
