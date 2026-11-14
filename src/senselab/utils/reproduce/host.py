"""The host record: where a step ran, for provenance.

`host_record` returns a fixed set of keys (`HOST_KEYS`): hostname, CPU model and counts, the first
GPU's name and driver with the GPU count, the host's CUDA version (`cuda_probe.detect_host_cuda`), the venv's
Python, torch and torch CUDA versions, and the batch job id. A value that cannot be determined is None (GPU
count 0).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from senselab.utils import cuda_probe

Runner = Callable[[Sequence[str]], tuple[int, str]]

HOST_KEYS = (
    "hostname",
    "cpu_model",
    "cpu_count",
    "cpus_allowed",
    "gpu_name",
    "gpu_driver",
    "gpu_count",
    "host_cuda",
    "python",
    "torch",
    "torch_cuda",
    "slurm_job_id",
)
NVIDIA_SMI = ("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader")
PROBE_TIMEOUT_S = 300
_PROBE = (
    "import json, platform\n"
    "out = {'python': platform.python_version(), 'torch': None, 'torch_cuda': None}\n"
    "try:\n"
    "    import torch\n"
    "    out['torch'] = torch.__version__\n"
    "    out['torch_cuda'] = torch.version.cuda\n"
    "except Exception:\n"
    "    pass\n"
    "print(json.dumps(out))\n"
)
_PROBE_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def _run(argv: Sequence[str]) -> tuple[int, str]:
    try:
        done = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=PROBE_TIMEOUT_S,
            env=_PROBE_ENV,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 127, ""
    return done.returncode, done.stdout


CPUINFO = "/proc/cpuinfo"


def _one_line(value: str) -> str:
    # Older Intel parts pad /proc/cpuinfo model names with runs of spaces ("CPU           X5650  @ 2.67GHz");
    # provenance accepts only single-spaced words, so the record normalises rather than refusing the host.
    return " ".join(value.split())


def _cpu_model() -> str | None:
    try:
        with open(CPUINFO) as handle:
            for line in handle:
                key, _, value = line.partition(":")
                if key.strip() == "model name" and value.strip():
                    return _one_line(value)
    except OSError:
        return None
    return None


def _gpu_rows(runner: Runner) -> list[list[str]]:
    code, out = runner(NVIDIA_SMI)
    rows = [line.split(",") for line in out.splitlines() if line.strip()] if code == 0 else []
    return [[_one_line(c) for c in row] for row in rows if len(row) == 2]


def _gpus(runner: Runner) -> tuple[str | None, str | None, int]:
    rows = _gpu_rows(runner)
    if not rows:
        return None, None, 0
    return rows[0][0] or None, rows[0][1] or None, len(rows)


NVIDIA_SMI_DEVICES = ("nvidia-smi", "--query-gpu=index,uuid,name,driver_version", "--format=csv,noheader")
"""The query :func:`gpu_table` makes: one row per GPU, with what ``CUDA_VISIBLE_DEVICES`` names it by."""


def gpu_table(*, runner: Runner | None = None) -> list[dict[str, str]]:
    """Every GPU ``nvidia-smi`` lists: its index, UUID, name and driver version, from one query.

    Args:
        runner: As :func:`host_record`'s; defaults to a subprocess runner.

    Returns:
        One ``{index, uuid, name, driver}`` row per GPU in ``nvidia-smi``'s order; empty where ``nvidia-smi`` is
        absent or fails.
    """
    code, out = (runner or _run)(NVIDIA_SMI_DEVICES)
    rows = [line.split(",") for line in out.splitlines() if line.strip()] if code == 0 else []
    keys = ("index", "uuid", "name", "driver")
    return [dict(zip(keys, (_one_line(cell) for cell in row))) for row in rows if len(row) == len(keys)]


def visible_gpu_names(visible: str | None, table: Sequence[Mapping[str, str]]) -> list[str] | None:
    """The names of the GPUs a ``CUDA_VISIBLE_DEVICES`` value lets a process see, read off :func:`gpu_table`.

    Each entry is an ``nvidia-smi`` index or a GPU UUID (``GPU-...``, a unique prefix allowed, as CUDA allows).
    Unset, every listed GPU is visible. An index is ``nvidia-smi``'s, which is CUDA's own order only under
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID``; a UUID is unambiguous.

    Args:
        visible: The process's ``CUDA_VISIBLE_DEVICES``; None where it is unset.
        table: :func:`gpu_table`'s rows.

    Returns:
        One name per visible GPU, in the order the value lists them; None where no GPU is visible or any entry
        names no listed GPU.
    """
    if visible is None:
        return [row["name"] for row in table] or None
    names: list[str] = []
    for entry in (part.strip() for part in visible.split(",")):
        if entry.isdigit():
            found = [row for row in table if row["index"] == entry]
        else:
            found = [row for row in table if entry and row["uuid"].startswith(entry)]
        if len(found) != 1:
            return None
        names.append(found[0]["name"])
    return names or None


WORKER_DEVICE_KEYS = ("gpus", "torch", "cuda_visible_devices")
"""What :data:`WORKER_DEVICE_PROBE` reports about the process it runs in."""

WORKER_DEVICE_PROBE = r"""
def senselab_worker_device(cuda=True):
    import os
    import sys

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda:
        version = getattr(sys.modules.get("torch"), "__version__", None)
        return {"gpus": None, "torch": None if version is None else str(version), "cuda_visible_devices": visible}
    try:
        import torch
    except Exception:
        return {"gpus": None, "torch": None, "cuda_visible_devices": visible}
    gpus = None
    try:
        if torch.cuda.is_available():
            gpus = [str(torch.cuda.get_device_name(index)) for index in range(torch.cuda.device_count())]
    except Exception:
        gpus = None
    return {"gpus": gpus, "torch": str(torch.__version__), "cuda_visible_devices": visible}
"""
"""Source of ``senselab_worker_device()``, for a worker script to define and call in its own process.

It returns the keys in :data:`WORKER_DEVICE_KEYS`: the name of every CUDA device that process can see
(``torch.cuda.get_device_name``), the process's own ``torch.__version__`` and its ``CUDA_VISIBLE_DEVICES``.
``gpus`` is None where CUDA is unavailable and ``torch`` is None where torch does not import.

``senselab_worker_device(cuda=False)`` touches no CUDA, for a process beside a server that already holds the card:
``gpus`` is None (name them with :func:`visible_gpu_names`), and ``torch`` is named only where the process has already
imported it. The source holds no percent sign, so it can be pasted into a worker script that is itself
``%``-formatted.
"""


def worker_device(reported: object) -> dict[str, Any]:
    """A worker's :data:`WORKER_DEVICE_PROBE` report, with anything malformed or missing read as None.

    Args:
        reported: What the worker sent; None where it sent nothing.

    Returns:
        A dict with exactly the keys in :data:`WORKER_DEVICE_KEYS`.
    """
    given: dict[str, Any] = reported if isinstance(reported, dict) else {}
    gpus = given.get("gpus")
    torch_version, visible = given.get("torch"), given.get("cuda_visible_devices")
    return {
        "gpus": [str(name) for name in gpus] if isinstance(gpus, list) else None,
        "torch": torch_version if isinstance(torch_version, str) else None,
        "cuda_visible_devices": visible if isinstance(visible, str) else None,
    }


def _probe(venv: Path, runner: Runner) -> dict[str, Any]:
    empty: dict[str, Any] = {"python": None, "torch": None, "torch_cuda": None}
    python = venv / "bin" / "python"
    if not python.exists():
        return empty
    code, out = runner([str(python), "-I", "-c", _PROBE])
    if code != 0:
        return empty
    for line in reversed(out.splitlines()):
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            return {k: parsed[k] if isinstance(parsed.get(k), str) else None for k in empty}
    return empty


def _host_cuda() -> str | None:
    """The driver's CUDA version as ``nvidia-smi`` reports it (`cuda_probe.detect_host_cuda`); None otherwise."""
    try:
        found = cuda_probe.detect_host_cuda()
    except (OSError, subprocess.SubprocessError):
        return None
    if found.source != "nvidia-smi" or found.version is None:
        return None
    return f"{found.version[0]}.{found.version[1]}"


def host_record(
    venv: str | os.PathLike[str] | None, *, runner: Runner | None = None, hostname: bool = True
) -> dict[str, Any]:
    """Describe this host and a venv's interpreter.

    Args:
        venv: The step's venv; its ``bin/python`` is asked for its Python, torch and torch CUDA
            versions in isolated mode with a minimal environment. None leaves those three None, for a
            caller that learns them inside the sandbox instead (the run's preamble).
        runner: Runs an argv and returns ``(returncode, stdout)``; a missing program or timeout is
            ``(127, "")``. Defaults to a subprocess runner. The host CUDA version is always probed by
            `cuda_probe.detect_host_cuda` and kept only from ``nvidia-smi`` (an ``nvcc`` toolkit is not the driver's).
        hostname: Record this host's name; False records None (confidential provenance names no host).

    Returns:
        A dict with exactly the keys in `HOST_KEYS`.
    """
    run = runner or _run
    gpu_name, gpu_driver, gpu_count = _gpus(run)
    try:
        allowed: int | None = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        allowed = None
    return {
        "hostname": socket.gethostname() if hostname else None,
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpus_allowed": allowed,
        "gpu_name": gpu_name,
        "gpu_driver": gpu_driver,
        "gpu_count": gpu_count,
        "host_cuda": _host_cuda(),
        **(_probe(Path(venv), run) if venv is not None else {"python": None, "torch": None, "torch_cuda": None}),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
