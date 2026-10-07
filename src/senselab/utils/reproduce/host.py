"""The host record: where a step ran, for provenance.

`host_record` returns a fixed set of keys (`HOST_KEYS`): hostname, CPU model and counts, the first
GPU's name and driver with the GPU count, the venv's Python, torch and torch CUDA versions, and the
batch job id. A value that cannot be determined is None (GPU count 0).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

Runner = Callable[[Sequence[str]], tuple[int, str]]

HOST_KEYS = (
    "hostname",
    "cpu_model",
    "cpu_count",
    "cpus_allowed",
    "gpu_name",
    "gpu_driver",
    "gpu_count",
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


def _gpus(runner: Runner) -> tuple[str | None, str | None, int]:
    code, out = runner(NVIDIA_SMI)
    rows = [line.split(",") for line in out.splitlines() if line.strip()] if code == 0 else []
    rows = [[_one_line(c) for c in row] for row in rows if len(row) == 2]
    if not rows:
        return None, None, 0
    return rows[0][0] or None, rows[0][1] or None, len(rows)


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


def host_record(venv: str | os.PathLike[str] | None, *, runner: Runner | None = None) -> dict[str, Any]:
    """Describe this host and a venv's interpreter.

    Args:
        venv: The step's venv; its ``bin/python`` is asked for its Python, torch and torch CUDA
            versions in isolated mode with a minimal environment. None leaves those three None, for a
            caller that learns them inside the sandbox instead (the run's preamble).
        runner: Runs an argv and returns ``(returncode, stdout)``; a missing program or timeout is
            ``(127, "")``. Defaults to a subprocess runner.

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
        "hostname": socket.gethostname(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "cpus_allowed": allowed,
        "gpu_name": gpu_name,
        "gpu_driver": gpu_driver,
        "gpu_count": gpu_count,
        **(_probe(Path(venv), run) if venv is not None else {"python": None, "torch": None, "torch_cuda": None}),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
