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
from typing import Callable, Collection, Final, Iterator, Mapping, Optional, Sequence

logger = logging.getLogger("senselab")

MIB: Final = 1024 * 1024  # invariant: bytes in a mebibyte
_KIB: Final = 1024  # invariant: bytes in a kibibyte, the unit of Linux ru_maxrss
_GPU_QUERY_TIMEOUT_S = 30.0
_NVIDIA_SMI_FIELDS = "index,uuid,name,memory.total,driver_version"
_NVIDIA_SMI_FORMAT = "--format=csv,noheader,nounits"
_NVIDIA_SMI_GPU_QUERY = ["nvidia-smi", f"--query-gpu={_NVIDIA_SMI_FIELDS},compute_cap", _NVIDIA_SMI_FORMAT]
_NVIDIA_SMI_GPU_QUERY_NO_CAP = ["nvidia-smi", f"--query-gpu={_NVIDIA_SMI_FIELDS}", _NVIDIA_SMI_FORMAT]
_CHIP_QUERY = ["sysctl", "-n", "machdep.cpu.brand_string"]
_LSCPU_QUERY = ["env", "LC_ALL=C", "lscpu"]
_APPS_FIELDS = ("pid", "used_memory")
_FREE_FIELDS = ("index", "uuid", "memory.free", "memory.total")
_NVIDIA_SMI_FREE_QUERY = ["nvidia-smi", f"--query-gpu={','.join(_FREE_FIELDS)}", _NVIDIA_SMI_FORMAT]
_NVIDIA_SMI_APPS_QUERY = ["nvidia-smi", f"--query-compute-apps={','.join(_APPS_FIELDS)}", _NVIDIA_SMI_FORMAT]
_STAT_PPID = 1
"""The field index of the parent pid in `/proc/<pid>/stat` after the parenthesised command name."""


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
    string, e.g. ``"Apple M2 Max"``; ``cpu_model`` the CPU's model string, None where the host does not name it.
    ``cpu_count`` is the CPUs the probing process may run on (``process_cpus``), not the node's.
    """

    cpu_count: int
    memory_bytes: int
    gpus: tuple[GpuInfo, ...]
    apple_silicon: bool
    platform: str
    gpu_probe_error: Optional[str] = None
    chip: Optional[str] = None
    probe_error: Optional[str] = None
    cpu_model: Optional[str] = None


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
                memory_bytes=int(fields[3]) * MIB,
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


def _read_cpuinfo() -> str:
    try:
        return Path("/proc/cpuinfo").read_text()
    except OSError:
        return ""


def _field(text: str, name: str) -> Optional[str]:
    for line in text.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == name and value.strip():
            return value.strip()
    return None


def _cpu_model(
    run: Callable[[list[str]], str], system: str, cpuinfo: Callable[[], str] = _read_cpuinfo
) -> Optional[str]:
    """Return the CPU's model string: `sysctl` on Darwin; on Linux `/proc/cpuinfo`, then `lscpu` in the C locale."""
    try:
        if system == "Darwin":
            return run(_CHIP_QUERY).strip() or None
        return _field(cpuinfo(), "model name") or _field(run(_LSCPU_QUERY), "Model name")
    except (OSError, subprocess.SubprocessError):
        return None


def process_cpus(
    affinity: Optional[Callable[[int], Collection[int]]] = getattr(os, "sched_getaffinity", None),
    count: Callable[[], Optional[int]] = os.cpu_count,
) -> int:
    """Return the CPUs this process may run on: its ``affinity`` where the platform has one, else ``count``."""
    if affinity is not None:
        return len(affinity(0))
    return count() or 1


def probe_hardware(
    *,
    run: Callable[[list[str]], str] = _run,
    cpuinfo: Callable[[], str] = _read_cpuinfo,
    cpus: Callable[[], int] = process_cpus,
) -> HardwareInfo:
    """Probe CPU, memory, GPUs, Apple chip and platform; never imports ``torch``.

    ``run`` executes a command (the ``nvidia-smi`` query, ``sysctl`` on Apple silicon) and returns
    stdout; ``cpuinfo`` reads ``/proc/cpuinfo``; ``cpus`` counts the process's CPUs; tests inject all three.
    """
    system = platform.system()
    machine = platform.machine()
    gpus, gpu_error = _probe_gpus(run)
    memory, memory_error = _physical_memory_bytes()
    apple_silicon = system == "Darwin" and machine == "arm64"
    chip, chip_error = _probe_chip(run) if apple_silicon else (None, None)
    errors = [e for e in (memory_error, chip_error) if e]
    return HardwareInfo(
        cpu_count=cpus(),
        memory_bytes=memory,
        gpus=gpus,
        apple_silicon=apple_silicon,
        platform=f"{system.lower()}-{machine}",
        gpu_probe_error=gpu_error,
        chip=chip,
        probe_error="; ".join(errors) or None,
        cpu_model=chip or _cpu_model(run, system, cpuinfo),
    )


def visible_gpus(info: HardwareInfo, environ: Optional[Mapping[str, str]] = None) -> list[GpuInfo]:
    """The GPUs torch would enumerate: all of them, or those ``CUDA_VISIBLE_DEVICES`` names, in its order.

    Raises:
        DeviceResolutionError: an entry of ``CUDA_VISIBLE_DEVICES`` names no GPU, or several.
    """
    environ = os.environ if environ is None else environ
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


def visible_devices(info: HardwareInfo, environ: Optional[Mapping[str, str]] = None) -> tuple[str, ...]:
    """Return the ``CUDA_VISIBLE_DEVICES`` entry naming each GPU this process sees: its uuid, else its index.

    Raises:
        DeviceResolutionError: an entry of ``CUDA_VISIBLE_DEVICES`` names no GPU, or several.
    """
    return tuple(g.uuid or str(g.index) for g in visible_gpus(info, environ) if g.uuid or g.index is not None)


def device_set(devices: Optional[Sequence[str]] = None, environ: Optional[Mapping[str, str]] = None) -> str:
    """Name devices: `devices`, else ``CUDA_VISIBLE_DEVICES``'s entries, joined by `-`; `all` unset, `none` empty."""
    if devices is None:
        raw = (os.environ if environ is None else environ).get("CUDA_VISIBLE_DEVICES")
        devices = ["all"] if raw is None else [e.strip() for e in raw.split(",") if e.strip()]
    return "-".join(devices) or "none"


@dataclass(frozen=True)
class DeviceMemory:
    """One GPU's free and total memory now, in bytes, named by its ``CUDA_VISIBLE_DEVICES`` entry."""

    device: str
    free_bytes: int
    total_bytes: int


def device_free_memory(
    devices: Optional[Sequence[str]] = None,
    environ: Optional[Mapping[str, str]] = None,
    *,
    run: Callable[[list[str]], str] = _run,
) -> Optional[tuple[DeviceMemory, ...]]:
    """Return the free and total memory of `devices` (``CUDA_VISIBLE_DEVICES`` entries; None: those this process sees).

    Read from ``nvidia-smi --query-gpu``; empty without ``nvidia-smi``; None, logged, where it fails or its output
    cannot be parsed.

    Raises:
        DeviceResolutionError: a device names no GPU, or several.
    """
    try:
        output = run(_NVIDIA_SMI_FREE_QUERY)
    except FileNotFoundError:
        return ()
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("device memory unreadable: %s", exc)
        return None
    rows = [[f.strip() for f in line.split(",")] for line in output.splitlines() if line.strip()]
    if any(len(r) != len(_FREE_FIELDS) or not (r[0].isdigit() and r[2].isdigit() and r[3].isdigit()) for r in rows):
        logger.warning("device memory unreadable: %r", output)
        return None
    gpus = [
        GpuInfo(name="", memory_bytes=int(r[3]) * MIB, driver=None, uuid=r[1] or None, index=int(r[0])) for r in rows
    ]
    free = {int(r[0]): int(r[2]) * MIB for r in rows}
    environ = os.environ if environ is None else environ
    named = environ if devices is None else {"CUDA_VISIBLE_DEVICES": ",".join(devices)}
    info = HardwareInfo(cpu_count=1, memory_bytes=0, gpus=tuple(gpus), apple_silicon=False, platform="")
    return tuple(
        DeviceMemory(g.uuid or str(g.index), free[g.index], g.memory_bytes)
        for g in visible_gpus(info, named)
        if g.index is not None
    )


def measured_budget(info: HardwareInfo, environ: Optional[Mapping[str, str]] = None) -> int:
    """Return the bytes a process's model cache may hold here: its smallest visible GPU's memory, else the host's."""
    gpus = visible_gpus(info, environ)
    return min(g.memory_bytes for g in gpus) if gpus else info.memory_bytes


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
        visible = visible_gpus(info, environ)
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


def memory_bytes(proc_cgroup: Path = Path("/proc/self/cgroup")) -> tuple[Optional[int], int]:
    """Return this process's memory now (None where unreadable) and its high-water, in bytes.

    From its cgroup v2 (`memory.current`, `memory.peak`), which counts every process of the cgroup; elsewhere no
    current figure and the largest resident set of this process and of its waited-for children.
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
    return None, unit * max(
        resource.getrusage(who).ru_maxrss for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)
    )


def proc_stats() -> Iterator[tuple[int, list[bytes]]]:
    """Yield each process's pid and the fields of its `/proc/<pid>/stat` after the command name; none off `/proc`."""
    for entry in os.listdir("/proc") if os.path.isdir("/proc/self") else ():
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as f:
                data = f.read()
        except OSError:
            continue
        yield int(entry), data[data.rfind(b")") + 2 :].split()


def process_tree(root: Optional[int] = None) -> set[int]:
    """Return the pid `root` (this process by default) and its descendants from `/proc`; `root` alone elsewhere."""
    root = os.getpid() if root is None else root
    parents: dict[int, list[int]] = {}
    for pid, fields in proc_stats():
        if len(fields) > _STAT_PPID:
            parents.setdefault(int(fields[_STAT_PPID]), []).append(pid)
    found, todo = {root}, [root]
    while todo:
        for child in parents.get(todo.pop(), ()):
            if child not in found:
                found.add(child)
                todo.append(child)
    return found


def _bounded(cmd: list[str], timeout_s: float) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=True).stdout


def device_memory_bytes(
    pids: Collection[int], *, timeout_s: float, run: Callable[[list[str], float], str] = _bounded
) -> Optional[int]:
    """Return the device memory the processes `pids` hold on their GPUs, None where `nvidia-smi` cannot answer.

    Read from `nvidia-smi --query-compute-apps` within `timeout_s`; other processes' memory is never counted.
    """
    try:
        output = run(_NVIDIA_SMI_APPS_QUERY, timeout_s)
    except (OSError, subprocess.SubprocessError):
        return None
    rows = [[p.strip() for p in line.split(",")] for line in output.splitlines()]
    return sum(
        int(r[1]) * MIB
        for r in rows
        if len(r) == len(_APPS_FIELDS) and r[0].isdigit() and r[1].isdigit() and int(r[0]) in pids
    )
