"""A step's inference engine: its identity, and the launcher that serves a ``vllm`` engine inside the sandbox.

`engine_identity` is what the step key, the step's env (``REPRO_ENGINE_IDENTITY``) and its provenance carry.
`LAUNCHER_CODE` runs with the driver's interpreter between the step's preamble and its argv: it opens
``servers_per_card`` `vllm_server.VllmServer` contexts one after the other over the step's hub view, reports the
outcome on a driver-held pipe that it closes before the step starts, runs the step with ``REPRO_ENGINE_URLS`` set,
and stops the servers when the step exits. `status_code` reads the report.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from senselab.utils import vllm_server
from senselab.utils.reproduce.spec import EngineSpec

EXIT_ENGINE = 93
"""The launcher's exit when its servers did not start; the step's argv did not run."""
READY = "ready"
STATUS_CODES = {
    "exited": "E_ENGINE_FIT",
    "timeout": "E_ENGINE_START_TIMEOUT",
    "version": "E_ENGINE_VERSION",
    "snapshot": "E_ENGINE_VERSION",
    "error": "E_ENGINE_LAUNCH",
}
"""The step code of each `vllm_server.START_ERROR_KINDS` kind, and of ``error`` (the launcher failed otherwise, e.g.
no ``vllm`` in the venv); a report that is absent or malformed is ``E_ENGINE_LAUNCH`` (`NO_REPORT`)."""
NO_REPORT = "E_ENGINE_LAUNCH"
"""The code of an absent or malformed report. It says nothing about the engine's fit, so it is not ``E_ENGINE_FIT``
(only ``exited`` is, and only that discards a resumable store): the launcher died or was not reached."""
CODES = frozenset({*STATUS_CODES.values(), NO_REPORT})
LOOPBACK = "127.0.0.1,localhost"
"""``no_proxy`` and ``NO_PROXY`` of a ``vllm`` step, so its clients reach the servers past the dead proxy."""
CACHE_TEMPLATES = {
    "VLLM_CACHE_ROOT": "{tmp}/vllm-cache",
    "TRITON_CACHE_DIR": "{tmp}/triton-cache",
    "TORCHINDUCTOR_CACHE_DIR": "{tmp}/inductor-cache",
    "FLASHINFER_WORKSPACE_BASE": "{tmp}/flashinfer",
}
"""Where a ``vllm`` step's compile caches go: under its ``{tmp}``, so no compiled kernel enters a result."""
LOG_DIR = "engine-logs"
LOCK_DIR = "engine-locks"


def server_args(spec: EngineSpec) -> list[str]:
    """The ``vllm serve`` arguments after the port: the declared settings, then ``server_args``.

    The settings go through `vllm_server.server_args`; ``--gpu-memory-utilization`` is `vllm_server.utilisation` of
    ``servers_per_card``.
    """
    settings = {
        "max_model_len": spec.max_model_len,
        "max_num_seqs": spec.max_num_seqs,
        "gpu_memory_utilization": vllm_server.utilisation(spec.servers_per_card),
        "kv_cache_dtype": spec.kv_cache_dtype,
        "generation_config": spec.generation_config,
        "limit_mm_per_prompt": dict(spec.limit_mm_per_prompt),
        "enable_prefix_caching": spec.enable_prefix_caching,
    }
    return [*vllm_server.server_args(settings), *spec.server_args]


def engine_identity(spec: EngineSpec) -> dict[str, Any]:
    """The engine identity: `vllm_server.engine_identity` for ``vllm``; else name, version, concurrency, batch size."""
    if spec.name == "vllm":
        return vllm_server.engine_identity(
            spec.version,
            server_args(spec),
            concurrency=spec.concurrency,
            batch_size=spec.batch_size,
            servers_per_card=spec.servers_per_card,
        )
    return {"name": spec.name, "version": spec.version, "concurrency": spec.concurrency, "batch_size": spec.batch_size}


def identity_json(spec: EngineSpec) -> str:
    """`engine_identity` as compact JSON with sorted keys (``REPRO_ENGINE_IDENTITY``)."""
    return json.dumps(engine_identity(spec), sort_keys=True, separators=(",", ":"))


def server_module_sha256() -> str:
    """The sha256 of the `vllm_server` source the launcher runs, keyed beside the identity."""
    return hashlib.sha256(Path(vllm_server.__file__).read_bytes()).hexdigest()


def launcher_config(spec: EngineSpec, *, executable: str, model_dir: str, tmp: str) -> dict[str, Any]:
    """The launcher's JSON argument.

    Args:
        spec: A ``vllm`` engine.
        executable: The step venv's ``vllm``, as the sandbox sees it.
        model_dir: The served snapshot in the step's hub view (``.../snapshots/<revision>``).
        tmp: The step's ``{tmp}`` as the sandbox sees it; the start locks and server logs go under it.

    Returns:
        ``servers``, ``executable``, ``model_dir``, ``server_args``, ``version``, ``lock_dir``, ``log_dir`` and
        ``startup_timeout_s``.
    """
    return {
        "servers": spec.servers_per_card,
        "executable": executable,
        "model_dir": model_dir,
        "server_args": server_args(spec),
        "version": spec.version,
        "lock_dir": f"{tmp}/{LOCK_DIR}",
        "log_dir": f"{tmp}/{LOG_DIR}",
        "startup_timeout_s": spec.startup_timeout_s,
    }


def status_code(report: bytes) -> str | None:
    """The step code of the launcher's report: None when the servers were ready, else an `CODES` member."""
    try:
        lines = report.decode().splitlines()
        parsed = json.loads(lines[0]) if len(lines) == 1 else None
    except (UnicodeDecodeError, ValueError, RecursionError):
        return NO_REPORT
    status = parsed.get("status") if isinstance(parsed, dict) else None
    if status == READY:
        return None
    return STATUS_CODES.get(status, NO_REPORT) if isinstance(status, str) else NO_REPORT


LAUNCHER_CODE = """\
import ctypes, json, os, signal, subprocess, sys, traceback
try:
    ctypes.CDLL(None).prctl(4, 0)
except Exception:
    pass
sys.path.insert(0, sys.argv[1])
from senselab.utils import vllm_server
_fd, _config, _argv = int(sys.argv[2]), json.loads(sys.argv[3]), sys.argv[4:]

def _report(status):
    with os.fdopen(_fd, "w") as handle:
        handle.write(json.dumps({"status": status}) + "\\n")

_servers = []

def _stop():
    for server in reversed(_servers):
        server.stop()

try:
    for _ in range(_config["servers"]):
        server = vllm_server.VllmServer(
            _config["executable"],
            _config["model_dir"],
            server_args=_config["server_args"],
            start_lock_dir=_config["lock_dir"],
            expected_version=_config["version"],
            log_dir=_config["log_dir"],
            startup_timeout_s=_config["startup_timeout_s"],
        )
        server.__enter__()
        _servers.append(server)
except vllm_server.VllmStartError as error:
    _stop()
    _report(error.kind)
    sys.exit(%(exit)d)
except BaseException:
    traceback.print_exc()
    _stop()
    _report("error")
    sys.exit(%(exit)d)
os.environ["REPRO_ENGINE_URLS"] = ",".join(server.url for server in _servers)
_report("ready")
_child = None

def _forward(signum, frame):
    if _child is not None:
        _child.send_signal(signum)

_forwarded = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
signal.pthread_sigmask(signal.SIG_BLOCK, _forwarded)
for _signal in _forwarded:
    signal.signal(_signal, _forward)
try:
    try:
        _child = subprocess.Popen(_argv, preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_UNBLOCK, _forwarded))
    finally:
        signal.pthread_sigmask(signal.SIG_UNBLOCK, _forwarded)
    _code = _child.wait()
finally:
    _stop()
sys.exit(_code if _code >= 0 else 128 - _code)
""" % {"exit": EXIT_ENGINE}
"""Runs as ``<driver python> -I -B -c LAUNCHER_CODE <senselab root> <report fd> <launcher_config JSON> <step argv…>``.

It makes itself non-dumpable, so the servers cannot open its fds through ``/proc``. It reports ``{"status": "ready"}``,
``{"status": <kind>}`` (a `vllm_server.VllmStartError` kind) or, on any other failure (printed to stderr), ``{"status":
"error"}`` on the report fd and closes it before the step starts. The fd is open while the preamble runs in the step's
venv, so the report is the launcher's only as far as that venv is trusted. Only ``ready`` runs the step, as a child that
inherits the env plus ``REPRO_ENGINE_URLS`` (the servers' loopback URLs, comma-separated); the launcher forwards
SIGTERM, SIGINT and SIGHUP to it (held back while the child starts), stops every server when it exits, and exits with
its code (``128 + n`` for signal ``n``). When no server starts it exits `EXIT_ENGINE`."""
