"""Opt-in network blocking for subprocess-venv workers: ``SENSELAB_BLOCK_NETWORK=1``.

When the variable is set in the parent, :func:`apply_network_guard` (called by
:func:`senselab.utils.subprocess_venv._clean_subprocess_env` and
:func:`senselab.utils.dependencies.hf_subprocess_env`) adds two things to a worker's environment:

* :data:`OFFLINE_ENV` — Hugging Face / transformers / datasets offline mode and telemetry,
  OpenTelemetry and pyannote metrics switched off;
* ``PYTHONPATH`` prefixed with :data:`GUARD_DIR`, whose ``sitecustomize.py`` the interpreter imports
  at startup, before ``-c`` code or any module. It refuses ``connect``/``connect_ex``/``sendto`` to a
  non-loopback address and name lookups of non-loopback hosts; ``AF_UNIX`` and loopback stay open.

The guard covers Python-level sockets. Native code that opens its own sockets (C libraries with
their own resolver) is not intercepted; the offline variables are what keep the Hugging Face stack
from trying. :func:`install_network_guard` installs the same guard in the current process.
"""

from __future__ import annotations

import importlib.util  # noqa: TID251
import os
from pathlib import Path
from typing import Dict, Mapping, MutableMapping, Optional

BLOCK_NETWORK_ENV = "SENSELAB_BLOCK_NETWORK"
"""The opt-in variable; any of ``1``, ``true``, ``yes``, ``on`` (case-insensitive) enables blocking."""

GUARD_DIR = Path(__file__).parent / "data" / "network_guard"
"""Directory holding the worker-side ``sitecustomize.py``."""

OFFLINE_ENV: Dict[str, str] = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    "DO_NOT_TRACK": "1",
    "OTEL_SDK_DISABLED": "true",
    "OTEL_TRACES_EXPORTER": "none",
    "OTEL_METRICS_EXPORTER": "none",
    "OTEL_LOGS_EXPORTER": "none",
    "PYANNOTE_METRICS_ENABLED": "false",
    "WANDB_MODE": "disabled",
    "GRADIO_ANALYTICS_ENABLED": "False",
    "MODELSCOPE_OFFLINE": "1",
}
"""Variables set in a blocked worker's environment."""

_TRUE = ("1", "true", "yes", "on")


def network_blocked(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether ``SENSELAB_BLOCK_NETWORK`` is set to a true value.

    Args:
        env: Environment to read; ``os.environ`` when omitted.

    Returns:
        True when blocking is requested.
    """
    value = (os.environ if env is None else env).get(BLOCK_NETWORK_ENV, "")
    return value.strip().lower() in _TRUE


def apply_network_guard(env: MutableMapping[str, str]) -> MutableMapping[str, str]:
    """Add the offline variables and the startup guard to a worker environment, when blocking is on.

    Blocking is on when either ``env`` or this process's environment requests it. The variables in
    :data:`OFFLINE_ENV` overwrite any existing values; :data:`GUARD_DIR` is prepended to
    ``PYTHONPATH`` once.

    Args:
        env: The worker's environment, modified in place.

    Returns:
        ``env``.
    """
    if not (network_blocked(env) or network_blocked()):
        return env
    env[BLOCK_NETWORK_ENV] = "1"
    env.update(OFFLINE_ENV)
    guard = str(GUARD_DIR)
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    if guard not in parts:
        env["PYTHONPATH"] = os.pathsep.join([guard, *parts])
    return env


def install_network_guard() -> None:
    """Install the worker guard in the current process (sets ``SENSELAB_BLOCK_NETWORK=1`` first).

    Idempotent. Also sets :data:`OFFLINE_ENV` in ``os.environ``; libraries that read those at import
    time must be imported after this call to see them.
    """
    import socket

    os.environ[BLOCK_NETWORK_ENV] = "1"
    os.environ.update(OFFLINE_ENV)
    if getattr(socket, "senselab_network_guard", False):
        return
    spec = importlib.util.spec_from_file_location("_senselab_network_guard", GUARD_DIR / "sitecustomize.py")  # noqa: TID251
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))  # noqa: TID251
