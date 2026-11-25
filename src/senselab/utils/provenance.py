"""The execution environment behind a result, and a per-step record of how a data object was made.

`environment` describes the process a result is computed in: the library versions that affect
numerics, the CPU vendor and model, torch's CPU capability, the thread count, torch's determinism
settings, and the CUDA device when one is in use. `environment_fingerprint` hashes that record, and
`record_step` appends one processing step to a data object's ``metadata``, naming the environment
by its fingerprint and keeping the full environment once per fingerprint.

The record carries no timestamp and no hostname: both differ between two runs that produce the same
bytes, and a hostname identifies a machine.

The measurements behind recording the CPU vendor are in ``specs/20261125-resample-float64/design.md``.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.metadata
import json
import platform
from typing import Any, Dict, List, Mapping, Optional

__all__ = [
    "ENVIRONMENT_KEYS",
    "ENVIRONMENTS_KEY",
    "PROCESSING_STEPS_KEY",
    "cpu_info",
    "environment",
    "environment_fingerprint",
    "record_step",
]

CPUINFO = "/proc/cpuinfo"
NUMERIC_PACKAGES = ("senselab", "numpy", "scipy", "torch", "torchaudio")
"""Distributions whose version is recorded, in the order recorded."""

ENVIRONMENT_KEYS = (
    "python",
    *(f"{name}_version" for name in NUMERIC_PACKAGES),
    "machine",
    "cpu_vendor",
    "cpu_model",
    "torch_cpu_capability",
    "torch_num_threads",
    "torch_deterministic_algorithms",
    "torch_cuda",
    "cudnn_version",
    "cudnn_deterministic",
    "cudnn_benchmark",
    "gpu_name",
)
"""Every key `environment` returns. A value that cannot be determined is None."""

PROCESSING_STEPS_KEY = "processing_steps"
"""``metadata`` key holding the ordered list of steps `record_step` appended."""
ENVIRONMENTS_KEY = "environments"
"""``metadata`` key mapping each environment fingerprint a step names to its full record."""

FINGERPRINT_HEX_CHARS = 16


def _one_line(value: str) -> str:
    """Collapse runs of whitespace, so padded ``/proc/cpuinfo`` model names compare equal."""
    return " ".join(value.split())


@functools.lru_cache(maxsize=1)
def cpu_info() -> Dict[str, Optional[str]]:
    """Return the CPU vendor and model from ``/proc/cpuinfo``.

    The vendor is ``vendor_id`` on x86 (``GenuineIntel``, ``AuthenticAMD``) and ``CPU implementer``
    on ARM (a hex code such as ``0x41``). Off Linux, or when the file cannot be read, both are None.

    Returns:
        ``{"cpu_vendor": ..., "cpu_model": ...}``.
    """
    found: Dict[str, Optional[str]] = {"cpu_vendor": None, "cpu_model": None}
    wanted = {"vendor_id": "cpu_vendor", "CPU implementer": "cpu_vendor", "model name": "cpu_model"}
    try:
        with open(CPUINFO) as handle:
            for line in handle:
                key, _, value = line.partition(":")
                target = wanted.get(key.strip())
                if target is not None and found[target] is None and value.strip():
                    found[target] = _one_line(value)
                if all(v is not None for v in found.values()):
                    break
    except OSError:
        pass
    return found


def _version(distribution: str) -> Optional[str]:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


@functools.lru_cache(maxsize=1)
def _static_environment() -> Dict[str, Any]:
    """The part of the environment that cannot change within a process."""
    record: Dict[str, Any] = {"python": platform.python_version()}
    record.update({f"{name}_version": _version(name) for name in NUMERIC_PACKAGES})
    record["machine"] = platform.machine() or None
    record.update(cpu_info())
    return record


def environment() -> Dict[str, Any]:
    """Describe the process a result is computed in.

    Settings a caller can change at run time (thread count, determinism flags, the CUDA device) are
    read on every call; versions and the CPU are read once per process. CUDA is never initialized
    here: ``gpu_name`` is the current device's name only when CUDA is already initialized, and None
    otherwise.

    Returns:
        A new dict with exactly the keys in `ENVIRONMENT_KEYS`.
    """
    import torch

    record = dict(_static_environment())
    record["torch_cpu_capability"] = torch.backends.cpu.get_cpu_capability()
    record["torch_num_threads"] = torch.get_num_threads()
    record["torch_deterministic_algorithms"] = torch.are_deterministic_algorithms_enabled()
    record["torch_cuda"] = torch.version.cuda
    cuda_in_use = torch.cuda.is_available() and torch.cuda.is_initialized()
    record["cudnn_version"] = torch.backends.cudnn.version() if cuda_in_use else None
    record["cudnn_deterministic"] = bool(torch.backends.cudnn.deterministic) if cuda_in_use else None
    record["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark) if cuda_in_use else None
    record["gpu_name"] = torch.cuda.get_device_name(torch.cuda.current_device()) if cuda_in_use else None
    return {key: record.get(key) for key in ENVIRONMENT_KEYS}


def environment_fingerprint(record: Mapping[str, Any]) -> str:
    """Return a short sha256 of an environment record's canonical JSON.

    Args:
        record: A record from `environment`.

    Returns:
        The first 16 hex characters of the digest.
    """
    canonical = json.dumps(dict(record), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:FINGERPRINT_HEX_CHARS]


def record_step(
    metadata: Dict[str, Any],
    step: str,
    parameters: Mapping[str, Any],
    record: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Append one processing step to a ``metadata`` dict, with the environment it ran in.

    The step list and the environments mapping are replaced rather than mutated, so a ``metadata``
    that is a shallow copy of another object's never changes that object's record.

    Args:
        metadata: The ``metadata`` dict of the object the step produced. Updated in place.
        step: The step's name, e.g. ``"resample_audios"``.
        parameters: The values the step ran with, JSON-friendly.
        record: The environment, if the caller already has it; defaults to `environment`.

    Returns:
        The entry appended: ``{"step", "parameters", "environment"}``, where ``environment`` is the
        fingerprint under ``metadata["environments"]``.
    """
    env = dict(record) if record is not None else environment()
    fingerprint = environment_fingerprint(env)
    entry = {"step": step, "parameters": dict(parameters), "environment": fingerprint}
    steps: List[Dict[str, Any]] = list(metadata.get(PROCESSING_STEPS_KEY) or [])
    metadata[PROCESSING_STEPS_KEY] = [*steps, entry]
    environments = dict(metadata.get(ENVIRONMENTS_KEY) or {})
    environments.setdefault(fingerprint, env)
    metadata[ENVIRONMENTS_KEY] = environments
    return entry
