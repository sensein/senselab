"""The step runner: key, skip or claim, run in the sandbox, check, adapt, evaluate, record.

A step's result dir ``R = <results_root>/<step_key>`` is laid out by the driver:

- ``R/work/``: the only directory the step may write; the ``{out}`` placeholder (argv, ``REPRO_OUT``,
  ``outputs`` patterns, ``{up:<step>}`` of a dependent) names it. Staged files go to ``R/work/stage``.
- ``R/tmp/``: the step's scratch (``{tmp}``, ``TMPDIR``, ``/tmp`` when possible); removed after the
  run, and never read by the driver after the step.
- ``R/logs/``, ``R/prov/``, ``R/adapter/``, ``R/metrics.json`` and ``R/summary.json``: written by the
  driver only, never writable inside a sandbox.

`run_step` runs one step:

1. Resolve the prefetched worktree (`source.prefetched_checkout`), open every data root once so an
   automounted root is mounted before the namespace is entered, digest the declared inputs and the
   selected native dirs, and compute the step key (`cache.step_key`). Inputs are keyed, so they are
   hashed before the done check.
2. A done result dir (`cache.is_done`) is linked into the dated run and the step is ``skipped``.
   Otherwise the step's claim is taken and done is checked again.
3. Refuse, before the result dir is reset, unless unprivileged user namespaces work, the prefetched
   venv passed its checks (`env.venv_checked`), and the host meets the GPU and driver floor.
4. Reset the dir, stage files, build the env; for an R step, run ``Rscript`` in the same sandbox and
   compare the session with the prefetch ``SESSION.json`` (``E_R_SESSION``).
5. Run the step in the read-only-by-default sandbox (`sandbox.sandbox_argv`). A preamble in the same
   sandbox reports ``sys.prefix``, Python, torch and its CUDA version and the canary result on a
   driver-held pipe, and the step is exec'd only when the canary's connect failed.
6. Digest the inputs again (``E_INPUT_MUTATED``), verify the checkout (``E_SRC_MUTATED``) and the venv
   (``E_ENV_CHANGED``), and refuse and delete any ``summary.json``, ``metrics.json`` or ``run.json``
   under ``work/`` (``E_RESERVED_NAME``).
7. Run the adapter in the sandbox with the driver's interpreter; it may import nothing from senselab
   outside ``senselab.utils.reproduce`` (``E_ADAPTER_IMPORT``) and returns its metrics on a pipe.
   Validate, evaluate ``expect`` (`evaluate_expect`), validate again as the driver with the same
   ``min_cell`` and write ``R/metrics.json``.
8. Remove ``R/tmp``, fix the exit, record provenance with that exit (`provenance.record_step`) and
   write ``R/summary.json`` last.

Only ``E_*`` codes fail a step; ``W_*`` codes are recorded. Driver errors go to ``R/logs/driver.log``;
callers get fixed codes only. `run_steps` runs a set of manifests in DAG order; a step whose need did
not succeed is ``never-ran``.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import logging
import math
import os
import re
import shutil
import stat
import statistics
import subprocess
import sys
import threading
import traceback
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, NamedTuple

from senselab.utils.reproduce import cache, digest, env, host, metrics, provenance, sandbox, source
from senselab.utils.reproduce.dag import toposort
from senselab.utils.reproduce.spec import Expect, LockRef, Manifest, MetricsSchema, Profile, StepSpec, render

Status = Literal["executed", "skipped", "failed", "never-ran"]

WORK = "work"
TMP = "tmp"
LOGS = "logs"
STAGE = "stage"
ADAPTER_DIR = "adapter"
ADAPTER_TMP = "adapter-tmp"
R_SESSION_DIR = "rsession"
METRICS = "metrics.json"
DRIVER_LOG = "driver.log"
RUN_JSON = "run.json"
RESERVED_NAMES = frozenset({cache.SUMMARY, METRICS, RUN_JSON})
NOTEBOOK_KINDS = frozenset({"notebook", "ir_notebook"})
EXIT_DRIVER = 1
EXIT_CWD = 96
EXIT_PREAMBLE = 97
EXIT_ADAPTER_IMPORT = 86
TRACEBACK_TAIL = 65536
PIPE_LIMIT = 1 << 20
EXACT_RELATIVE = 1e-9

CODES = frozenset(
    {
        "E_SRC_MISSING",
        "E_SRC_MISMATCH",
        "E_SRC_MUTATED",
        "E_ROOT_MISSING",
        "E_ADAPTER_MISSING",
        "E_ADAPTER_CHANGED",
        "E_ADAPTER_FAILED",
        "E_ADAPTER_IMPORT",
        "E_ADAPTER_NO_METRICS",
        "E_R_MISSING",
        "E_R_SESSION",
        "E_KEY_INVALID",
        "E_NO_USERNS",
        "E_NOT_REPORTABLE",
        "E_ENV_MISSING",
        "E_ENV_UNCHECKED",
        "E_ENV_CHANGED",
        "E_ENV_BUILD",
        "E_RENDER_PLACEHOLDER",
        "E_RENDER_BAD",
        "E_CWD",
        "E_STAGE_BAD_DEST",
        "E_STAGE_CONFLICT",
        "E_GPU_MISSING",
        "E_DRIVER_OLD",
        "E_GPU_TORCH_CPU",
        "E_MAX_CUDA",
        "E_CANARY",
        "E_SYS_PREFIX",
        "E_TIMEOUT",
        "E_STEP_FAILED",
        "E_INPUT_MUTATED",
        "E_RESERVED_NAME",
        "E_OUTPUT_MISSING",
        "E_UPSTREAM_METRICS",
        "E_EXPECT",
        "E_PROV_REFUSED",
        "E_DRIVER",
        "W_HOST_VALUE_DROPPED",
    }
)
"""The run's own codes. Digest, cache, metrics and stage codes (``E_DIGEST_*``, ``E_KEY_*``,
``E_RESULT_*``, ``E_CLAIM_HELD``, ``E_METRICS_*``, ``E_SPLIT_*``, ``E_STAGE_*``) pass through as their
modules name them."""

_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.+!_-]{0,63}$")
_EXC_CLASS = re.compile(
    r"^((?:[A-Za-z_][A-Za-z0-9_]{0,63}\.){0,8}[A-Z][A-Za-z0-9_]{0,63}"
    r"(?:Error|Exception|Interrupt|Exit|Warning|Iteration|Timeout))(?::|$)"
)
_FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+)')
_CODE_PATH = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./+-]{0,255}$")
_PREAMBLE_KEYS = frozenset({"sys_prefix", "python", "torch", "torch_cuda", "canary_failed"})
_RESERVED_ROOTS = ("checkout",)

WRAPPER = 'cd -- "$1" || exit 96\nshift\nexec "$@"\n'
"""The script the sandbox execs after its mounts: ``cd`` into the cwd (inside the namespace, so the cwd is on
the read-only bind), then exec its arguments. For the step those are the preamble, which writes to the
driver's pipe, closes it and execs the step itself."""

_PREAMBLE_CODE = """\
import contextlib, io, json, os, platform, sys
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    exec(compile({canary!r}, "<canary>", "exec"), {{}})
try:
    _canary = json.loads(_buf.getvalue())
except ValueError:
    _canary = None
_failed = _canary.get("canary_failed") if isinstance(_canary, dict) else None
_out = {{"sys_prefix": sys.prefix, "python": platform.python_version(), "torch": None, "torch_cuda": None,
         "canary_failed": _failed}}
if _failed is True:
    try:
        import torch
        _out["torch"] = str(torch.__version__)
        _cuda = torch.version.cuda
        _out["torch_cuda"] = None if _cuda is None else str(_cuda)
    except Exception:
        pass
with os.fdopen(int(sys.argv[1]), "w") as _f:
    _f.write(json.dumps(_out) + "\\n")
if _failed is not True:
    sys.exit(97)
os.execvp(sys.argv[2], sys.argv[2:])
"""
"""Run as ``<venv>/bin/python -I -B -c PREAMBLE <pipe fd> <step argv…>``: the canary, then torch, a JSON line
on the driver's pipe (closed before the step starts), then ``execvp`` of the step, only when the canary's
connect failed."""

ADAPTER_RUNNER = """\
import builtins, importlib, os, pkgutil, runpy, sys, traceback
sys.path.insert(0, sys.argv[1])
_fd = int(sys.argv[2])
ALLOWED = "senselab.utils.reproduce"
import senselab.utils.reproduce as _pkg
for _m in pkgutil.iter_modules(_pkg.__path__):
    importlib.import_module(ALLOWED + "." + _m.name)

def _senselab(name):
    return name == "senselab" or name.startswith("senselab.")

def _allowed(name):
    return name == ALLOWED or name.startswith(ALLOWED + ".")

_before = {n for n in sys.modules if _senselab(n)}
_violations = []
_import = builtins.__import__
_import_module = importlib.import_module

def _refuse(name, fromlist, importer):
    if _senselab(importer) or not _senselab(name) or _allowed(name):
        return False
    if name in ("senselab", "senselab.utils") and fromlist:
        return not all(_allowed(name + "." + f) for f in fromlist)
    return True

def _guarded(name, globals=None, locals=None, fromlist=(), level=0):
    importer = (globals or {}).get("__name__") or ""
    if level == 0 and _refuse(name, fromlist or (), importer):
        _violations.append(name)
        raise ImportError("E_ADAPTER_IMPORT")
    return _import(name, globals, locals, fromlist, level)

def _guarded_module(name, package=None):
    importer = sys._getframe(1).f_globals.get("__name__") or ""
    if not name.startswith(".") and _refuse(name, (), importer):
        _violations.append(name)
        raise ImportError("E_ADAPTER_IMPORT")
    return _import_module(name, package)

builtins.__import__ = _guarded
importlib.import_module = _guarded_module
sys.argv = [sys.argv[3], sys.argv[4], "/proc/self/fd/%d" % _fd]
try:
    runpy.run_path(sys.argv[0], run_name="__main__")
    _code = 0
except SystemExit as _exit:
    _code = _exit.code if isinstance(_exit.code, int) else (0 if _exit.code is None else 1)
except BaseException:
    traceback.print_exc()
    _code = 1
try:
    os.close(_fd)
except OSError:
    pass
_new = sorted(n for n in sys.modules if _senselab(n) and n not in _before and not _allowed(n))
if _violations or _new:
    print("E_ADAPTER_IMPORT", file=sys.stderr)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(86)
sys.exit(_code)
"""
"""Runs an adapter as ``__main__`` with ``sys.argv = [adapter, work dir, metrics path]``, where the metrics
path is the driver's pipe. It first imports every ``senselab.utils.reproduce`` module, then refuses any
import, by an adapter, of a senselab module outside that package, and exits 86 if one was attempted or one
was loaded anyway."""


class RunError(Exception):
    """A step refused or failed for a fixed reason; the message is the code only."""

    def __init__(self, code: str) -> None:
        """Store the code as both the message and ``code``.

        Args:
            code: A fixed ``E_*`` code.
        """
        super().__init__(code)
        self.code = code


class StepOutcome(NamedTuple):
    """What became of one step.

    ``status`` is ``executed`` (the step's own process ran; ``exit`` says how it went), ``skipped`` (a done
    result was reused), ``failed`` (the driver refused the step, or the preamble did not hand over to it:
    a missing cwd or a canary that connected) or ``never-ran`` (a need did not succeed). ``out`` is the
    result dir ``R`` once the key is known.
    """

    step: str
    key: str | None
    status: Status
    exit: int | None
    codes: tuple[str, ...]
    out: Path | None

    @property
    def ok(self) -> bool:
        """Whether the step's result can be used downstream: executed or skipped with exit 0."""
        return self.status in ("executed", "skipped") and self.exit == 0


@dataclass
class RunConfig:
    """What every step of a run shares.

    Attributes:
        profile: The caller's site profile.
        cache: The reproduce cache (worktrees, venvs, R libraries, Hugging Face and torch homes).
        base_dir: The manifest repository; adapters and patches are relative to it.
        date: The run date, naming the dated run dir.
        min_cell: The minimum cell size for metrics suppression, used for both validations.
        no_netns: Test-only: run without the network namespace. Such a run is never reportable: it
            always carries ``E_NOT_REPORTABLE`` and a non-zero exit, so it is never served as done.
        timeout: Seconds before a step (and separately its adapter) is killed (``E_TIMEOUT``); None waits.
        parent_env: Where the recorded-only job variables are read from; defaults to ``os.environ``.
        host_runner: Runs host probes (`host.host_record`); defaults to subprocesses.
    """

    profile: Profile
    cache: Path
    base_dir: Path
    date: datetime.date = field(default_factory=lambda: datetime.datetime.now(datetime.UTC).date())
    min_cell: int = 5
    no_netns: bool = False
    timeout: float | None = None
    parent_env: Mapping[str, str] | None = None
    host_runner: host.Runner | None = None


class _Keyed(NamedTuple):
    key: str
    result: Path
    checkout: source.Checkout
    input_digests: list[tuple[str, int]]
    adapter: Path
    adapter_sha256: str
    paths: dict[str, Path]


# --- driver-side file operations under the result dir --------------------------------------------


def _refuse_link(path: Path) -> None:
    if path.is_symlink():
        raise RunError("E_DRIVER")


def _mkdir(path: Path) -> Path:
    """Create a directory the driver owns; an existing entry or a symlink in its place is refused."""
    os.mkdir(path, 0o770)
    st = os.lstat(path)
    if not stat.S_ISDIR(st.st_mode):
        raise RunError("E_DRIVER")
    return path


def _create(path: Path) -> int:
    """Open a new file the driver owns for writing: ``O_CREAT | O_EXCL | O_NOFOLLOW``."""
    return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o660)


def _write_json(path: Path, obj: object) -> None:
    """Write JSON atomically through a new ``O_EXCL | O_NOFOLLOW`` file renamed over ``path``."""
    _refuse_link(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    fd = _create(tmp)
    try:
        try:
            os.write(fd, (json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def _read_nofollow(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as handle:
        return handle.read()


def _remove_tree(path: Path) -> None:
    """Remove a directory without following any symlink in it; a symlink in its place is unlinked."""
    if path.is_symlink():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


# --- keys ------------------------------------------------------------------------------------------


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _step_venv(step: StepSpec, analysis: str, cache_dir: Path) -> tuple[Path, LockRef | None]:
    overlay = step.env.notebook_tools if step.kind in NOTEBOOK_KINDS else None
    if overlay is not None:
        return env.derived_venv_dir(cache_dir, analysis, step.env, overlay), overlay
    return env.venv_dir(cache_dir, analysis, step.env), None


def _up_work(upstream: Mapping[str, StepOutcome], name: str) -> Path:
    out = upstream[name].out
    if out is None:
        raise RunError("E_UPSTREAM_METRICS")
    return out / WORK


def _root_dir(root: str, config: RunConfig, upstream: Mapping[str, StepOutcome], checkout: Path) -> Path:
    if root.startswith("up:"):
        return _up_work(upstream, root.split(":", 1)[1])
    if root in _RESERVED_ROOTS:
        return checkout
    if root not in config.profile.roots:
        raise RunError("E_ROOT_MISSING")
    return config.profile.roots[root]


def _touch_roots(profile: Profile) -> None:
    """Open every data root once, so an automounted root is mounted before the namespace is entered."""
    for root in profile.roots.values():
        try:
            with os.scandir(root) as entries:
                next(entries, None)
        except OSError:
            raise RunError("E_ROOT_MISSING") from None


def _r_lib(step: StepSpec, cache_dir: Path) -> Path:
    assert step.env.r_lib is not None
    return Path(render(step.env.r_lib, {"cache": cache_dir}))


def _r_digests(step: StepSpec, cache_dir: Path) -> tuple[str | None, str | None]:
    if step.kind not in cache.R_KINDS:
        return None, None
    r_lib = _r_lib(step, cache_dir)
    try:
        return _sha256_file(r_lib / env.SESSION), _sha256_file(r_lib / env.MODULE_ENV)
    except OSError:
        raise RunError("E_R_MISSING") from None


def _key(step: StepSpec, analysis: str, config: RunConfig, upstream: Mapping[str, StepOutcome]) -> _Keyed:
    try:
        co = source.prefetched_checkout(step, cache=config.cache)
    except source.ReproduceSourceError as error:
        raise RunError("E_SRC_MISSING" if error.code == source.WORKTREE_MISSING else "E_SRC_MISMATCH") from None
    _touch_roots(config.profile)
    input_digests = [
        digest.input_set_digest(_root_dir(i.root, config, upstream, co.worktree), i.patterns, require_nonempty=True)
        for i in step.inputs
    ]
    native = cache.native_digests(step, config.profile)
    adapter = config.base_dir / step.adapter
    try:
        adapter_sha = _sha256_file(adapter)
    except OSError:
        raise RunError("E_ADAPTER_MISSING") from None
    r_session, r_module = _r_digests(step, config.cache)
    try:
        key = cache.step_key(
            step,
            config.profile,
            upstream={name: str(upstream[name].key) for name in step.needs},
            submodules=co.submodules if step.submodules else {},
            adapter_sha256=adapter_sha,
            adapter_env_sha256=cache.driver_env_sha256(),
            input_digests=[d for d, _ in input_digests],
            native=native,
            r_session_sha256=r_session,
            r_module_env_sha256=r_module,
        )
    except ValueError:
        raise RunError("E_KEY_INVALID") from None
    venv_path, _ = _step_venv(step, analysis, config.cache)
    result = cache.result_dir(config.profile.results_root, key)
    paths = {
        "out": result / WORK,
        "tmp": result / TMP,
        "checkout": co.worktree,
        "venv": venv_path,
        "cache": config.cache,
    }
    paths |= {f"up:{name}": _up_work(upstream, name) for name in step.needs}
    return _Keyed(key, result, co, input_digests, adapter, adapter_sha, paths)


# --- logging -----------------------------------------------------------------------------------------


class _Buffer(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.lines: list[str] = []
        self.target: Path | None = None
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        line = self.format(record) + "\n"
        if self.target is None:
            self.lines.append(line)
            return
        try:
            fd = os.open(self.target, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
            try:
                os.write(fd, line.encode())
            finally:
                os.close(fd)
        except OSError:
            self.lines.append(line)

    def open(self, target: Path) -> None:
        fd = _create(target)
        try:
            os.write(fd, "".join(self.lines).encode())
        finally:
            os.close(fd)
        self.lines.clear()
        self.target = target


@contextlib.contextmanager
def _driver_log() -> Iterator[_Buffer]:
    """Route the ``senselab`` logger to one buffer (later ``R/logs/driver.log``) and nowhere else.

    The claim's stale-takeover warning names the lock path under the results tree, so while a step is
    claimed, run and recorded the logger neither propagates nor keeps its own handlers.
    """
    logger = logging.getLogger("senselab")
    saved = (logger.handlers[:], logger.propagate, logger.level)
    buffer = _Buffer()
    logger.handlers[:] = [buffer]
    logger.propagate = False
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    try:
        yield buffer
    finally:
        logger.handlers[:], logger.propagate = saved[0], saved[1]
        logger.setLevel(saved[2])


def _log() -> logging.Logger:
    return logging.getLogger("senselab")


# --- run_step ------------------------------------------------------------------------------------------


def _failed(step: StepSpec, code: str, key: str | None = None, out: Path | None = None) -> StepOutcome:
    return StepOutcome(step.name, key, "failed", EXIT_DRIVER, (code,), out)


def _read_summary(out: Path) -> tuple[str, ...]:
    try:
        data = json.loads((out / cache.SUMMARY).read_text())
    except (OSError, ValueError):
        return ()
    codes = data.get("codes") if isinstance(data, dict) else None
    return tuple(c for c in codes if isinstance(c, str)) if isinstance(codes, list) else ()


def _unlink_run(config: RunConfig, analysis: str, step: str) -> None:
    """Remove a stale same-day ``<runs_root>/<date>/<analysis>/<step>`` link (a symlink only)."""
    with contextlib.suppress(OSError, cache.CacheError):
        link = cache.run_dir(config.profile.runs_root, config.date, analysis) / step
        if link.is_symlink():
            link.unlink()


def _skipped(step: StepSpec, key: str, run: Path, out: Path) -> StepOutcome:
    try:
        cache.link_run(run, step.name, out)
    except cache.CacheError as error:
        return _failed(step, error.code, key, out)
    return StepOutcome(step.name, key, "skipped", 0, _read_summary(out), out)


def run_step(
    step: StepSpec,
    analysis: str,
    config: RunConfig,
    upstream: Mapping[str, StepOutcome] | None = None,
    upstream_schemas: Mapping[str, MetricsSchema] | None = None,
) -> StepOutcome:
    """Run one step, or reuse its done result.

    Args:
        step: The step.
        analysis: The name of the manifest the step belongs to (it names the venv and the run dir).
        config: The run configuration.
        upstream: The outcome of each of ``step.needs``; each must be `StepOutcome.ok`.
        upstream_schemas: The ``metrics_schema`` of each step an expectation reads (``from_step``,
            ``tolerance_from``), to validate its ``metrics.json`` before a value is taken from it.

    Returns:
        The outcome; ``codes`` are fixed codes only, and no exception, message or path reaches the
        caller (any ``OSError``, ``TypeError`` or ``ValueError`` is ``E_DRIVER``). A run that reached its
        result dir has written ``summary.json`` there when it could, and the dated run links ``<step>``
        to it.
    """
    upstream = upstream or {}
    if any(name not in upstream or not upstream[name].ok for name in step.needs):
        _unlink_run(config, analysis, step.name)
        return StepOutcome(step.name, None, "never-ran", None, (), None)
    try:
        return _run_step(step, analysis, config, upstream, upstream_schemas or {})
    except (OSError, TypeError, ValueError):
        _log().error("driver error\n%s", traceback.format_exc())
        return _failed(step, "E_DRIVER")


def _run_step(
    step: StepSpec,
    analysis: str,
    config: RunConfig,
    upstream: Mapping[str, StepOutcome],
    schemas: Mapping[str, MetricsSchema],
) -> StepOutcome:
    try:
        keyed = _key(step, analysis, config, upstream)
    except (RunError, digest.DigestError, cache.CacheError) as error:
        _unlink_run(config, analysis, step.name)
        return _failed(step, error.code)
    result = keyed.result
    run = cache.run_dir(config.profile.runs_root, config.date, analysis)
    if cache.is_done(result):
        return _skipped(step, keyed.key, run, result)
    with _driver_log() as log:
        try:
            with cache.claim(config.profile.results_root, keyed.key):
                if cache.is_done(result):
                    return _skipped(step, keyed.key, run, result)
                execution = _Execution(step, analysis, config, upstream, schemas, keyed)
                try:
                    execution.refusals()
                except RunError as error:
                    return _failed(step, error.code, keyed.key)
                cache.reset_result_dir(config.profile.results_root, keyed.key)
                summary = _summary_or_driver_error(execution, log)
        except cache.CacheError as error:
            return _failed(step, error.code, keyed.key)
    status: Status = "executed" if summary["executed"] else "failed"
    codes, exit_code = tuple(summary["codes"]), summary["exit"]
    try:
        cache.link_run(run, step.name, result)
    except cache.CacheError as error:
        codes, exit_code = tuple(sorted({*codes, error.code})), exit_code or EXIT_DRIVER
    return StepOutcome(step.name, keyed.key, status, exit_code, codes, result)


def _summary_or_driver_error(execution: _Execution, log: _Buffer) -> dict[str, Any]:
    """Run a reset step and write its summary; a driver failure still leaves an ``E_DRIVER`` summary."""
    result = execution.keyed.result
    try:
        _mkdir(result / LOGS)
        log.open(result / LOGS / DRIVER_LOG)
        summary = execution.run()
        cache.write_summary(result, summary)
        return summary
    except (OSError, TypeError, ValueError, RunError):
        _log().error("driver error\n%s", traceback.format_exc())
        summary = execution.driver_error_summary()
        with contextlib.suppress(OSError, TypeError, ValueError):
            cache.write_summary(result, summary)
        return summary


class _Execution:
    """One claimed run of a step: `refusals` before the reset, then `run`, which returns the summary."""

    def __init__(
        self,
        step: StepSpec,
        analysis: str,
        config: RunConfig,
        upstream: Mapping[str, StepOutcome],
        schemas: Mapping[str, MetricsSchema],
        keyed: _Keyed,
    ) -> None:
        self.step, self.analysis, self.config, self.keyed = step, analysis, config, keyed
        self.upstream, self.schemas = upstream, schemas
        self.result = keyed.result
        self.work = keyed.paths["out"]
        self.tmp = keyed.paths["tmp"]
        self.codes: set[str] = set()
        self.executed = False
        self.rc: int | None = None
        self.preamble: dict[str, Any] | None = None
        self.passed: bool | None = None
        self.frame: tuple[str | None, str | None] = (None, None)
        self.venv: env.VenvResult | None = None
        self.overlay: LockRef | None = None
        self.installed: str | None = None
        self.host: dict[str, Any] | None = None
        self.env: dict[str, str] = {}
        self.recorded_sha: str | None = None
        self.argv: list[str] = []
        self.started = _now()

    # --- before the reset ----------------------------------------------------------------------

    def refusals(self) -> None:
        """Refuse without touching the result dir, so a previous run's diagnostics survive.

        Raises:
            RunError: ``E_NO_USERNS``, ``E_ENV_MISSING``, ``E_ENV_UNCHECKED``, ``E_GPU_MISSING`` or
                ``E_DRIVER_OLD``.
        """
        step, config = self.step, self.config
        if not sandbox.userns_available():
            raise RunError("E_NO_USERNS")
        venv_path, self.overlay = _step_venv(step, self.analysis, config.cache)
        venv = env.load_venv(venv_path, step.env, self.overlay)
        if venv is None:
            raise RunError("E_ENV_MISSING")
        if not venv.checked:
            raise RunError("E_ENV_UNCHECKED")
        self.venv, self.installed = venv, env.installed_sha256(venv_path)
        self.host = host.host_record(None, runner=config.host_runner)
        resources, record = step.resources, self.host
        if (resources.gpu_arch or resources.min_driver) and (record.get("gpu_count") or 0) < 1:
            raise RunError("E_GPU_MISSING")
        if resources.min_driver is not None:
            driver = record.get("gpu_driver")
            if not isinstance(driver, str) or _version(driver) < _version(resources.min_driver):
                raise RunError("E_DRIVER_OLD")

    # --- the run -------------------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Run the step and everything after it.

        Returns:
            The summary; provenance carries the same exit.
        """
        try:
            self._prepare()
            self._exec()
            self._after()
            if self.executed and self.rc == 0 and not self._failing():
                self._adapt()
        except RunError as error:
            self.codes.add(error.code)
        except (digest.DigestError, cache.CacheError, metrics.MetricsError) as error:
            self.codes.add(error.code)
        except Exception:
            _log().error("driver error\n%s", traceback.format_exc())
            self.codes.add("E_DRIVER")
        if self.config.no_netns:
            self.codes.add("E_NOT_REPORTABLE")
        outputs = self._outputs()
        for scratch in (self.tmp, self.result / ADAPTER_TMP, self.result / R_SESSION_DIR):
            try:
                _remove_tree(scratch)
            except OSError:
                _log().error("could not remove a scratch dir\n%s", traceback.format_exc())
                self.codes.add("E_DRIVER")
        self._record(outputs)
        return self._summary()

    def driver_error_summary(self) -> dict[str, Any]:
        """The summary of a run the driver could not finish: ``E_DRIVER`` and a non-zero exit."""
        self.codes.add("E_DRIVER")
        return self._summary()

    def _failing(self) -> bool:
        return any(c.startswith("E_") for c in self.codes)

    def _exit(self) -> int:
        if self.rc is not None and self.rc != 0:
            return self.rc
        return EXIT_DRIVER if self._failing() else 0

    def _summary(self) -> dict[str, Any]:
        pre = self.preamble or {}
        return {
            "exit": self._exit(),
            "exc_class": self.frame[0],
            "frame": self.frame[1],
            "codes": sorted(self.codes),
            "pass": self.passed,
            "sys_prefix_ok": None if self.preamble is None else self._prefix_ok(),
            "torch": pre.get("torch"),
            "torch_cuda": pre.get("torch_cuda"),
            "canary_failed": pre.get("canary_failed"),
            "executed": self.executed,
        }

    # --- before the step ----------------------------------------------------------------------

    def _prepare(self) -> None:
        step, config = self.step, self.config
        _mkdir(self.work)
        _mkdir(self.tmp)
        for sub in ("home", "ipython", "jupyter", "jupyter/runtime", "jupyter/data", "jupyter/config", "cache"):
            _mkdir(self.tmp / sub)
        _mkdir(self.tmp / "cuda-cache")
        if step.kind in NOTEBOOK_KINDS:
            _mkdir(self.work / "notebooks")
        self._stage()
        try:
            self.env, _, self.recorded_sha = sandbox.build_env(
                step, config.profile, self.keyed.paths, parent=config.parent_env
            )
        except ValueError:
            raise RunError("E_ENV_BUILD") from None
        if step.kind in cache.R_KINDS:
            self._r_env()
            self._r_session()

    def _values(self) -> dict[str, str | os.PathLike[str]]:
        values: dict[str, str | os.PathLike[str]] = {k: os.path.realpath(v) for k, v in self.keyed.paths.items()}
        values |= {f"root:{n}": os.path.realpath(r) for n, r in self.config.profile.roots.items()}
        return values

    def _render(self, template: str) -> str:
        try:
            return render(template, self._values())
        except KeyError:
            raise RunError("E_RENDER_PLACEHOLDER") from None
        except ValueError:
            raise RunError("E_RENDER_BAD") from None

    def _r_env(self) -> None:
        assert self.step.env.r_lib is not None
        r_lib = self._render(self.step.env.r_lib)
        try:
            module_env = json.loads((Path(r_lib) / env.MODULE_ENV).read_text())
        except (OSError, ValueError):
            raise RunError("E_R_MISSING") from None
        if not isinstance(module_env, dict):
            raise RunError("E_R_MISSING")
        native = [self.config.profile.native[n] for n in self.step.resources.native]
        self.env |= env.r_env_vars(
            {k: v for k, v in module_env.items() if k in ("PATH", "LD_LIBRARY_PATH") and isinstance(v, str)},
            r_lib,
            path_prefix=[os.path.join(self._render("{venv}"), "bin")],
            path_suffix=[*sandbox.SYSTEM_PATH, *(os.fspath(d) for n in native for d in n.bin_dirs)],
            native_lib_dirs=[os.fspath(d) for n in native for d in n.lib_dirs],
        )
        text = json.dumps(self.env, sort_keys=True, separators=(",", ":"))
        self.recorded_sha = hashlib.sha256(text.encode()).hexdigest()

    def _r_session(self) -> None:
        """Record the R session in the step's own sandbox and env, and compare it with ``SESSION.json``."""
        assert self.step.env.r_lib is not None
        r_lib = Path(self._render(self.step.env.r_lib))
        try:
            recorded = json.loads((r_lib / env.SESSION).read_text())
        except (OSError, ValueError):
            raise RunError("E_R_MISSING") from None
        out = _mkdir(self.result / R_SESSION_DIR)
        script = _mkdir(self.result / ADAPTER_DIR) / "session.R"
        fd = _create(script)
        try:
            os.write(fd, env.r_session_script(self.step.env.r_packages).encode())
        finally:
            os.close(fd)
        argv = [sandbox.SH, "-c", WRAPPER, "sh", str(out), "Rscript", "--vanilla", str(script), str(r_lib)]
        argv.append(str(out))
        rc, _ = self._spawn(self._sandbox(argv, [out], tmp=self.tmp), self.env, "rsession")
        if rc != 0:
            raise RunError("E_R_SESSION")
        try:
            for name in ("installed.tsv", "lib_paths.txt", "session_info.txt", "r_version.txt"):
                _refuse_link(out / name)
            current = env.read_r_session(out)
        except (OSError, RunError):
            raise RunError("E_R_SESSION") from None
        if not isinstance(recorded, dict) or any(recorded.get(k) != current[k] for k in env.R_SESSION_FIELDS):
            _log().error(
                "R session differs from SESSION.json in: %s",
                [k for k in env.R_SESSION_FIELDS if not isinstance(recorded, dict) or recorded.get(k) != current[k]],
            )
            raise RunError("E_R_SESSION")

    def _stage(self) -> None:
        target_root = _mkdir(self.work / STAGE)
        taken: set[Path] = set()
        for entry in self.step.stage:
            dest = PurePosixPath(self._render(entry.dest)) if entry.dest else PurePosixPath(".")
            if dest.is_absolute() or ".." in dest.parts or "\\" in str(dest):
                raise RunError("E_STAGE_BAD_DEST")
            root = _root_dir(entry.root, self.config, self.upstream, self.keyed.checkout.worktree)
            try:
                files = digest.matching_files(root, entry.patterns, require_nonempty=True)
            except digest.DigestError as error:
                raise RunError(error.code.replace("E_DIGEST_", "E_STAGE_")) from None
            for rel in files:
                target = target_root / dest / rel
                if target in taken:
                    raise RunError("E_STAGE_CONFLICT")
                taken.add(target)
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with open(root / rel, "rb") as source_file:
                        fd = _create(target)
                        with os.fdopen(fd, "wb") as handle:
                            shutil.copyfileobj(source_file, handle)
                except OSError:
                    raise RunError("E_STAGE_UNREADABLE") from None

    # --- exec ----------------------------------------------------------------------------------

    def _sandbox(self, argv: list[str], rw: list[Path], *, tmp: Path | None) -> list[str]:
        ups = [self.keyed.paths[f"up:{n}"] for n in self.step.needs]
        return sandbox.sandbox_argv(
            argv,
            self.config.profile.roots.values(),
            rw,
            [self.keyed.checkout.worktree, self.keyed.paths["venv"], *ups],
            netns=not self.config.no_netns,
            tmp=tmp,
            visible=[self.result, self.config.cache],
        )

    def _exec(self) -> None:
        step, config = self.step, self.config
        argv = [self._render(a) for a in step.argv]
        cwd = Path(self._render(step.cwd))
        cwd = cwd if cwd.is_absolute() else Path(os.path.realpath(self.keyed.checkout.worktree)) / cwd
        cwd = Path(os.path.realpath(cwd))
        allowed = [os.path.realpath(self.keyed.checkout.worktree), os.path.realpath(self.work)]
        if not cwd.is_dir() or not any(cwd == Path(a) or cwd.is_relative_to(a) for a in allowed):
            raise RunError("E_CWD")
        self.argv = argv
        python = os.path.join(self._render("{venv}"), "bin", "python")
        preamble = _PREAMBLE_CODE.format(canary=sandbox.canary(config.profile.canary_host))
        read, write = os.pipe()
        wrapped = [sandbox.SH, "-c", WRAPPER, "sh", str(cwd), python, "-I", "-B", "-c", preamble, str(write), *argv]
        self.rc, data = self._spawn(self._sandbox(wrapped, [self.work], tmp=self.tmp), self.env, "step", (read, write))
        self.preamble = _parse_preamble(data)
        self.executed = self.preamble is not None and self.preamble["canary_failed"] is True
        if self.preamble is not None and self.host is not None:
            self.host |= {k: self.preamble[k] for k in ("python", "torch", "torch_cuda")}
        if self.executed and self.rc != 0:
            self.codes.add("E_STEP_FAILED")
            self.frame = _traceback(self.result / LOGS / "step.stderr.log", self.keyed.checkout.worktree)

    def _spawn(
        self, argv: list[str], run_env: Mapping[str, str], name: str, pipe: tuple[int, int] | None = None
    ) -> tuple[int, bytes]:
        """Run a sandbox with stdout and stderr on driver-held log files and, given one, a pipe to read."""
        logs = self.result / LOGS
        out, err = _create(logs / f"{name}.stdout.log"), _create(logs / f"{name}.stderr.log")
        chunks: list[bytes] = []
        reader: threading.Thread | None = None
        try:
            proc = subprocess.Popen(
                argv,
                env=dict(run_env),
                cwd=self.result,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                pass_fds=() if pipe is None else (pipe[1],),
            )
            if pipe is not None:
                os.close(pipe[1])
                reader = threading.Thread(target=_drain, args=(pipe[0], chunks), daemon=True)
                reader.start()
            try:
                rc = proc.wait(timeout=self.config.timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise RunError("E_TIMEOUT") from None
        finally:
            os.close(out)
            os.close(err)
            if pipe is not None:
                with contextlib.suppress(OSError):
                    os.close(pipe[1])
                if reader is not None:
                    reader.join()
                with contextlib.suppress(OSError):
                    os.close(pipe[0])
        return rc, b"".join(chunks)

    # --- after the step -----------------------------------------------------------------------

    def _prefix_ok(self) -> bool:
        prefix = (self.preamble or {}).get("sys_prefix")
        return isinstance(prefix, str) and os.path.realpath(prefix) == os.path.realpath(self.keyed.paths["venv"])

    def _after(self) -> None:
        pre = self.preamble
        if pre is None and self.rc == EXIT_CWD:
            self.codes.add("E_CWD")
        elif pre is None or pre["canary_failed"] is not True:
            self.codes.add("E_CANARY")
        elif not self._prefix_ok():
            self.codes.add("E_SYS_PREFIX")
        if pre is not None and pre["canary_failed"] is True:
            self._torch_guard(pre)
        self._reserved_names()
        for i, (spec, before) in enumerate(zip(self.step.inputs, self.keyed.input_digests)):
            root = _root_dir(spec.root, self.config, self.upstream, self.keyed.checkout.worktree)
            if not digest.digest_unchanged(before, digest.input_set_digest(root, spec.patterns)):
                _log().error("input %d changed during the step", i)
                self.codes.add("E_INPUT_MUTATED")
        try:
            source.verify_checkout(self.keyed.checkout)
        except source.ReproduceSourceError as error:
            _log().error("checkout changed during the step: %s\n%s", error.code, error.detail)
            self.codes.add("E_SRC_MUTATED")
        assert self.venv is not None
        again = env.load_venv(self.venv.path, self.step.env, self.overlay)
        if (
            again is None
            or (again.env_key, again.overlay_sha256) != (self.venv.env_key, self.venv.overlay_sha256)
            or env.installed_sha256(self.venv.path) != self.installed
        ):
            self.codes.add("E_ENV_CHANGED")

    def _reserved_names(self) -> None:
        """Refuse and delete any file or dir under ``work/`` named like a driver-owned file."""
        found = False
        for top, dirs, files in os.walk(self.work, followlinks=False):
            for name in [*dirs, *files]:
                if name not in RESERVED_NAMES:
                    continue
                found = True
                path = Path(top) / name
                if path.is_dir() and not path.is_symlink():
                    shutil.rmtree(path)
                    dirs.remove(name)
                else:
                    path.unlink()
        if found:
            self.codes.add("E_RESERVED_NAME")

    def _torch_guard(self, pre: Mapping[str, Any]) -> None:
        torch_cuda = pre.get("torch_cuda")
        if self.step.resources.gpu_arch and pre.get("torch") is not None and torch_cuda is None:
            self.codes.add("E_GPU_TORCH_CPU")
        max_cuda = self.step.env.max_cuda
        if max_cuda is not None and isinstance(torch_cuda, str) and _version(torch_cuda) > _version(max_cuda):
            self.codes.add("E_MAX_CUDA")

    # --- adapter, expectations -----------------------------------------------------------------

    def _adapt(self) -> None:
        step, config = self.step, self.config
        adapter_dir = self.result / ADAPTER_DIR
        if not adapter_dir.is_dir():
            _mkdir(adapter_dir)
        _refuse_link(adapter_dir)
        copy = adapter_dir / "adapter.py"
        data = self.keyed.adapter.read_bytes()
        if hashlib.sha256(data).hexdigest() != self.keyed.adapter_sha256:
            raise RunError("E_ADAPTER_CHANGED")
        fd = _create(copy)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        scratch = _mkdir(self.result / ADAPTER_TMP)
        _mkdir(scratch / "home")
        senselab_root = str(Path(sys.modules["senselab"].__file__ or "").resolve().parent.parent)
        read, write = os.pipe()
        argv = [sys.executable, "-I", "-B", "-c", ADAPTER_RUNNER, senselab_root, str(write), str(copy), str(self.work)]
        adapter_env = {
            "PATH": ":".join(sandbox.SYSTEM_PATH),
            "HOME": str(scratch / "home"),
            "TMPDIR": str(scratch),
            **sandbox.FIXED_KEYED,
            **sandbox.OFFLINE,
            **{v: sandbox.PROXY for v in sandbox.PROXY_VARS},
            **{v.upper(): sandbox.PROXY for v in sandbox.PROXY_VARS},
            **{v: "" for v in sandbox.NO_PROXY_VARS},
        }
        wrapped = [sandbox.SH, "-c", WRAPPER, "sh", str(self.work), *argv]
        rc, raw = self._spawn(self._sandbox(wrapped, [scratch], tmp=None), adapter_env, "adapter", (read, write))
        if rc == EXIT_ADAPTER_IMPORT:
            raise RunError("E_ADAPTER_IMPORT")
        if rc != 0:
            raise RunError("E_ADAPTER_FAILED")
        try:
            obj = json.loads(raw.decode())
        except (UnicodeDecodeError, ValueError):
            raise RunError("E_ADAPTER_NO_METRICS") from None
        clean, found = metrics.validate_metrics(obj, step.metrics_schema, writer="adapter", min_cell=config.min_cell)
        self.codes.update(found)
        if clean is None:
            return
        upstream_metrics = {name: self._upstream_metrics(name) for name in _expect_sources(step)}
        final, passed, codes = evaluate_expect(step, clean, upstream_metrics)
        self.codes.update(codes)
        if final is None:
            return
        final_clean, found = metrics.validate_metrics(
            final, step.metrics_schema, writer="driver", min_cell=config.min_cell
        )
        self.codes.update(found)
        if final_clean is None:
            return
        _write_json(self.result / METRICS, final_clean)
        if step.expect:
            self.passed = passed
        if not passed:
            self.codes.add("E_EXPECT")

    def _upstream_metrics(self, name: str) -> dict[str, Any]:
        out = self.upstream[name].out
        schema = self.schemas.get(name)
        if out is None or schema is None:
            raise RunError("E_UPSTREAM_METRICS")
        try:
            obj = json.loads(_read_nofollow(out / METRICS))
        except (OSError, ValueError):
            raise RunError("E_UPSTREAM_METRICS") from None
        clean, _ = metrics.validate_metrics(obj, schema, writer="driver", min_cell=self.config.min_cell)
        if clean is None:
            raise RunError("E_UPSTREAM_METRICS")
        return clean

    # --- provenance ----------------------------------------------------------------------------

    def _outputs(self) -> list[provenance.Artifact]:
        artifacts = []
        for pattern in self.step.outputs:
            succeeded = self._exit() == 0
            try:
                found = digest.input_set_digest(self.work, [pattern], require_nonempty=succeeded)
            except digest.DigestError as error:
                if succeeded:
                    self.codes.add("E_OUTPUT_MISSING" if error.code == "E_DIGEST_EMPTY" else error.code)
                found = digest.input_set_digest(self.result, [])
            artifacts.append(provenance.Artifact(digest=found[0], count=found[1]))
        return artifacts

    def _record(self, outputs: list[provenance.Artifact]) -> None:
        if self.venv is None or self.host is None:
            return
        record, dropped = provenance.drop_unrecordable_host_values(self.host)
        if dropped:
            _log().warning("host values not recorded: %s", ", ".join(dropped))
            self.codes.add("W_HOST_VALUE_DROPPED")
        inputs = [
            provenance.Artifact(digest=d, count=n, root=spec.root)
            for spec, (d, n) in zip(self.step.inputs, self.keyed.input_digests)
        ]
        argv_sha = hashlib.sha256(json.dumps(self.argv).encode()).hexdigest() if self.argv else None
        try:
            provenance.record_step(
                self.result / "prov",
                step_key=self.keyed.key,
                step=self.step.name,
                commit_sha=self.step.commit,
                venv=self.venv,
                torch_lock_sha256=None if self.step.env.torch is None else self.step.env.torch.lock_sha256,
                body_lock_sha256=self.step.env.body.lock_sha256,
                host=record,
                inputs=inputs,
                outputs=outputs,
                started=self.started,
                ended=_now(),
                exit_code=self._exit(),
                codes=sorted(self.codes),
                env_keys=sorted(self.env),
                env_sha256=self.recorded_sha,
                argv_sha256=argv_sha,
            )
        except ValueError:
            _log().error("provenance refused\n%s", traceback.format_exc())
            self.codes.add("E_PROV_REFUSED")
        except (OSError, TypeError):
            _log().error("provenance failed\n%s", traceback.format_exc())
            self.codes.add("E_DRIVER")


def _drain(fd: int, chunks: list[bytes]) -> None:
    total = 0
    while chunk := os.read(fd, 65536):
        total += len(chunk)
        if total <= PIPE_LIMIT:
            chunks.append(chunk)


def _expect_sources(step: StepSpec) -> list[str]:
    names = [e.from_step.step for e in step.expect if e.from_step is not None]
    names += [s for e in step.expect if e.tolerance_from is not None for s in e.tolerance_from.steps]
    return sorted(set(names))


def _version(text: str) -> tuple[int, ...]:
    parts = []
    for part in text.split("."):
        digits = re.match(r"\d+", part)
        parts.append(int(digits.group(0)) if digits else 0)
    return tuple(parts)


def _parse_preamble(data: bytes) -> dict[str, Any] | None:
    """The preamble's JSON when it is exactly one object of the expected shape, else None (``E_CANARY``)."""
    try:
        lines = data.decode().splitlines()
        parsed = json.loads(lines[0]) if len(lines) == 1 else None
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict) or set(parsed) != _PREAMBLE_KEYS:
        return None
    if not isinstance(parsed["canary_failed"], bool) or not isinstance(parsed["sys_prefix"], str):
        return None
    for key in ("python", "torch", "torch_cuda"):
        value = parsed[key]
        if value is not None and not (isinstance(value, str) and _VERSION.match(value)):
            return None
    return parsed


def _traceback(stderr: Path, checkout: Path) -> tuple[str | None, str | None]:
    """The exception class and the last checkout-relative ``file:line`` of the last Python traceback."""
    try:
        fd = os.open(stderr, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as handle:
            handle.seek(max(0, os.fstat(handle.fileno()).st_size - TRACEBACK_TAIL))
            text = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None, None
    start = text.rfind("Traceback (most recent call last):")
    if start < 0:
        return None, None
    lines = text[start:].splitlines()[1:]
    root = os.path.realpath(checkout)
    frame = None
    exc = None
    for line in lines:
        match = _FRAME.match(line)
        if match:
            path = os.path.realpath(match.group(1))
            rel = os.path.relpath(path, root)
            if path.startswith(root + os.sep) and _CODE_PATH.match(rel) and ".." not in rel.split("/"):
                frame = f"{rel}:{match.group(2)}"
        elif line and not line[0].isspace():
            found = _EXC_CLASS.match(line)
            exc = found.group(1) if found else None
            break
    return exc, frame


# --- expectations ------------------------------------------------------------------------------


def _metric(obj: Mapping[str, Any], address: str) -> tuple[float, str | None]:
    """A metric's value and unit.

    The unit is an entry's ``unit`` (default ``fraction``), ``fraction`` for a per-class rate, None for a
    count or statistic.
    """
    value = metrics.metric_value(obj, address)
    quantity = metrics.metric_quantity(address)
    if quantity not in metrics.SCALED_QUANTITIES:
        return value, None
    if address.startswith("per_class."):
        return value, "fraction"
    name, _, label = address.partition("@")
    entries = [
        e
        for e in (*obj.get("headline", []), *obj.get("secondary", []))
        if e.get("name") == name and (e.get("label") == label if label else True)
    ]
    return value, str(entries[0].get("unit", "fraction")) if entries else "fraction"


def _convert(value: float, unit: str | None, to: str | None) -> float:
    if unit is None or to is None or unit == to:
        return value
    return value * 100.0 if to == "pp" else value / 100.0


def _location(obj: dict[str, Any], address: str) -> dict[str, Any]:
    """The mapping an expectation's results are written into.

    That is its headline or secondary entry, its per-class entry, or the top level for a count or
    confusion cell.
    """
    if address.startswith("per_class."):
        entry: dict[str, Any] = obj["per_class"][address.split(".")[1]]
        return entry
    if address.startswith(("counts.", "confusion.")):
        return obj
    name, _, label = address.partition("@")
    for e in (*obj.get("headline", []), *obj.get("secondary", [])):
        if e.get("name") == name and (e.get("label") == label if label else True):
            found: dict[str, Any] = e
            return found
    return obj


def evaluate_expect(
    step: StepSpec, clean: Mapping[str, Any], upstream: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any] | None, bool, list[str]]:
    """Evaluate a step's expectations against its validated metrics; only the driver writes the results.

    Each expectation is compared in its ``unit`` when it gives one, otherwise in the unit of the step's
    own metric entry; values in the other unit are converted (``pp`` = 100 × ``fraction``). A target
    is ``value`` or the ``from_step`` metric of an upstream step; a tolerance is ``tolerance`` or
    ``max(a, k * sd)`` over the expectation's own metric in the ``tolerance_from`` steps (sample
    standard deviation). ``abs`` passes when ``|value - target| <= tolerance``; ``exact`` when the two
    are equal, within a relative ``1e-9`` when a unit was converted; ``none`` always passes and records the
    delta when it has a target.

    The results go into the metric's own entry (``target``, ``tolerance``, ``delta``, ``pass``), in that
    entry's own unit (converted back from the comparison unit): its headline or secondary entry, its
    per-class entry, or the top level for a count or confusion cell. A place addressed by several
    expectations gets only ``pass``, true when all of them pass. The top level gets ``pass`` for the
    whole step when there are expectations, so a count or confusion expectation writes only ``target``,
    ``tolerance`` and ``delta`` there, and only when it is the only expectation on the top level.

    Args:
        step: The step.
        clean: Its adapter's output after `metrics.validate_metrics`.
        upstream: For each step named by ``from_step`` or ``tolerance_from``, its driver-validated
            ``metrics.json``.

    Returns:
        ``(metrics, passed, codes)``: a copy of ``clean`` with the results written in, whether every
        expectation passed, and fixed codes (`metrics.ADDRESS_CODES`) for a value that cannot be read;
        ``metrics`` is None when there is such a code.
    """
    final = json.loads(json.dumps(clean))
    results: list[tuple[Expect, dict[str, float | bool]]] = []
    try:
        for expect in step.expect:
            results.append((expect, _evaluate(expect, final, upstream)))
    except metrics.MetricsError as error:
        return None, False, [error.code]
    places: dict[int, list[dict[str, float | bool]]] = {}
    objects: dict[int, dict[str, Any]] = {}
    for expect, result in results:
        place = _location(final, expect.metric)
        places.setdefault(id(place), []).append(result)
        objects[id(place)] = place
    for ident, found in places.items():
        place = objects[ident]
        if place is final:
            if len(found) == 1:
                place.update({k: v for k, v in found[0].items() if k != "pass"})
        elif len(found) == 1:
            place.update(found[0])
        else:
            place["pass"] = all(bool(r["pass"]) for r in found)
    passed = all(bool(r["pass"]) for _, r in results)
    if results:
        final["pass"] = passed
    return final, passed, []


def _evaluate(
    expect: Expect, own: Mapping[str, Any], upstream: Mapping[str, Mapping[str, Any]]
) -> dict[str, float | bool]:
    raw, unit = _metric(own, expect.metric)
    work = expect.unit or unit
    converted = unit != work
    value = _convert(raw, unit, work)
    target: float | None = None
    if expect.value is not None:
        target = expect.value
    elif expect.from_step is not None:
        source_value, source_unit = _metric(upstream[expect.from_step.step], expect.from_step.metric)
        converted = converted or source_unit != work
        target = _convert(source_value, source_unit, work)
    tolerance: float | None = expect.tolerance
    if expect.tolerance_from is not None:
        rule = expect.tolerance_from
        spread = []
        for name in rule.steps:
            calibration, calibration_unit = _metric(upstream[name], expect.metric)
            converted = converted or calibration_unit != work
            spread.append(_convert(calibration, calibration_unit, work))
        tolerance = max(rule.a, rule.k * statistics.stdev(spread))
    result: dict[str, float | bool] = {}
    if target is None:
        result["pass"] = True
        return result
    delta = value - target
    if expect.kind == "abs":
        assert tolerance is not None
        passed = abs(delta) <= tolerance + 1e-12 * max(1.0, abs(target))
    elif expect.kind == "exact":
        passed = math.isclose(value, target, rel_tol=EXACT_RELATIVE, abs_tol=0.0) if converted else value == target
    else:
        passed = True
    result["target"] = _convert(target, work, unit)
    result["delta"] = _convert(delta, work, unit)
    if tolerance is not None:
        result["tolerance"] = _convert(tolerance, work, unit)
    result["pass"] = bool(passed) and math.isfinite(delta)
    return result


# --- a set of manifests ------------------------------------------------------------------------


def run_steps(manifests: Sequence[Manifest], config: RunConfig) -> list[StepOutcome]:
    """Run every step of a set of manifests in DAG order (`dag.toposort`).

    A step whose need did not succeed (`StepOutcome.ok`) is ``never-ran``; the others run or skip as
    `run_step` decides. Each analysis gets ``<runs_root>/<date>/<analysis>/run.json`` listing its steps'
    keys, statuses, exits and codes.

    Args:
        manifests: The manifests; step names are unique across them.
        config: The run configuration.

    Returns:
        One outcome per step, in run order.
    """
    analysis = {s.name: m.name for m in manifests for s in m.steps}
    specs = {s.name: s for m in manifests for s in m.steps}
    outcomes: dict[str, StepOutcome] = {}
    ordered = toposort(list(specs.values()))
    for step in ordered:
        upstream = {name: outcomes[name] for name in step.needs}
        schemas = {name: specs[name].metrics_schema for name in _expect_sources(step)}
        outcomes[step.name] = run_step(step, analysis[step.name], config, upstream, schemas)
    for name in sorted(set(analysis.values())):
        rows = [outcomes[s.name] for s in ordered if analysis[s.name] == name]
        run = cache.run_dir(config.profile.runs_root, config.date, name)
        run.mkdir(parents=True, exist_ok=True)
        _write_json(
            run / RUN_JSON,
            {
                "steps": [
                    {"step": o.step, "key": o.key, "status": o.status, "exit": o.exit, "codes": list(o.codes)}
                    for o in rows
                ]
            },
        )
    return [outcomes[s.name] for s in ordered]
