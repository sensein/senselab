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
not succeed is ``never-ran``. With ``only`` it runs just the named steps, as one batch job does: their
upstream steps are keyed and must already be done (`resolve_step`), never run, and ``run.json`` is
merged under a lock rather than rewritten, so jobs of one analysis can finish in any order.
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
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Literal, NamedTuple

from senselab.utils.reproduce import build, cache, digest, env, host, metrics, provenance, sandbox, source
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
RUN_LOCK = ".run.json.lock"
PENDING = "pending"
RESERVED_NAMES = frozenset({cache.SUMMARY, METRICS, RUN_JSON})
NOTEBOOK_KINDS = frozenset({"notebook", "ir_notebook"})
EXIT_DRIVER = 1
EXIT_CWD = 96
EXIT_PREAMBLE = 97
EXIT_INTERPRETER = 95
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
        "E_UPSTREAM_NOT_DONE",
        "E_STEP_INTERPRETER",
        "E_OUTPUT_SYMLINK",
        "E_JOB_ENDED_WITHOUT_RECORD",
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
_PREAMBLE_KEYS = frozenset({"sys_prefix", "python", "torch", "torch_cuda", "canary_failed", "interpreter_ok"})
_RESERVED_ROOTS = ("checkout",)

WRAPPER = 'cd -- "$1" || exit 96\nshift\nexec "$@"\n'
"""The script the sandbox execs after its mounts: ``cd`` into the cwd (inside the namespace, so the cwd is on
the read-only bind), then exec its arguments. For the step those are the preamble, which writes to the
driver's pipe, closes it and execs the step itself."""

_PREAMBLE_CODE = """\
import contextlib, io, json, os, platform, shutil, sys
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    exec(compile({canary!r}, "<canary>", "exec"), {{}})
try:
    _canary = json.loads(_buf.getvalue())
except ValueError:
    _canary = None
_failed = _canary.get("canary_failed") if isinstance(_canary, dict) else None
_allowed = json.loads(sys.argv[2])
_target = sys.argv[3]
_found = _target if "/" in _target else shutil.which(_target, path=os.environ.get("PATH", ""))
_ok = False
if _found:
    _places = (os.path.realpath(_found), os.path.realpath(os.path.dirname(os.path.abspath(_found))))
    _ok = any(p == a or p.startswith(a.rstrip("/") + "/") for a in _allowed for p in _places)
_out = {{"sys_prefix": sys.prefix, "python": platform.python_version(), "torch": None, "torch_cuda": None,
         "canary_failed": _failed, "interpreter_ok": _ok}}
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
if not _ok:
    sys.exit(95)
os.execvp(sys.argv[3], sys.argv[3:])
"""
"""Run as ``<venv>/bin/python -I -B -c PREAMBLE <pipe fd> <allowed dirs JSON> <step argv…>``: the canary, then
torch, a JSON line on the driver's pipe (closed before the step starts), then ``execvp`` of the step, only when
the canary's connect failed and the step's program, found on its ``PATH``, lies in an allowed dir (by its real
path, or by the real path of the dir it was found in)."""

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


def _empty_file(path: Path) -> None:
    """Make ``path`` an empty regular file the driver owns, replacing anything else there."""
    try:
        st = os.lstat(path)
        if stat.S_ISREG(st.st_mode) and st.st_size == 0:
            return
        if stat.S_ISDIR(st.st_mode):
            raise RunError("E_R_MISSING")
        os.unlink(path)
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o440))


def _under(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip("/") + "/")


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
        return StepOutcome(step.name, None, "never-ran", None, upstream_codes(step, upstream), None)
    seen: dict[str, Path] = {}
    with _quiet_log():
        try:
            return _run_step(step, analysis, config, upstream, upstream_schemas or {}, seen)
        except (OSError, TypeError, ValueError):
            _append_driver_log(seen.get("result"), "driver error\n" + traceback.format_exc())
            return _failed(step, "E_DRIVER", out=seen.get("result"))


@contextlib.contextmanager
def _quiet_log() -> Iterator[None]:
    """Give the ``senselab`` logger a `logging.NullHandler` and no propagation for the whole step.

    Outside `_driver_log` nothing the driver or a library logs may reach a terminal; with no handler at
    all, `logging.lastResort` would print it to stderr.
    """
    logger = logging.getLogger("senselab")
    saved = (logger.handlers[:], logger.propagate)
    logger.handlers[:] = [logging.NullHandler()]
    logger.propagate = False
    try:
        yield
    finally:
        logger.handlers[:], logger.propagate = saved


def _append_driver_log(result: Path | None, text: str) -> None:
    """Append to ``R/logs/driver.log`` when that file exists as a regular file; otherwise drop the text."""
    if result is None:
        return
    with contextlib.suppress(OSError):
        fd = os.open(result / LOGS / DRIVER_LOG, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW)
        try:
            if stat.S_ISREG(os.fstat(fd).st_mode):
                os.write(fd, (text + "\n").encode())
        finally:
            os.close(fd)


def _run_step(
    step: StepSpec,
    analysis: str,
    config: RunConfig,
    upstream: Mapping[str, StepOutcome],
    schemas: Mapping[str, MetricsSchema],
    seen: dict[str, Path],
) -> StepOutcome:
    try:
        keyed = _key(step, analysis, config, upstream)
    except (RunError, digest.DigestError, cache.CacheError) as error:
        _unlink_run(config, analysis, step.name)
        return _failed(step, error.code)
    result = keyed.result
    seen["result"] = result
    run = cache.run_dir(config.profile.runs_root, config.date, analysis)
    if cache.is_done(result):
        return _skipped(step, keyed.key, run, result)
    with _driver_log() as log:
        try:
            with cache.claim(config.profile.results_root, keyed.key):
                if cache.is_done(result):
                    return _skipped(step, keyed.key, run, result)
                execution = _Execution(step, analysis, config, upstream, schemas, keyed)
                with _env_locks(step, analysis, config):
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


@contextlib.contextmanager
def _env_locks(step: StepSpec, analysis: str, config: RunConfig) -> Iterator[None]:
    """Hold shared locks on the step's venv and R library, so a rebuild (exclusive) waits for the step."""
    venv_path, _ = _step_venv(step, analysis, config.cache)
    locks = [venv_path.with_name(venv_path.name + ".lock")]
    if step.kind in cache.R_KINDS:
        r_lib = _r_lib(step, config.cache)
        locks.append(r_lib.with_name(r_lib.name + ".lock"))
    with contextlib.ExitStack() as stack:
        for lock in locks:
            stack.enter_context(build.file_lock(lock, shared=True))
        yield


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
        self.r_bins: list[str] = []
        self.recorded_sha: str | None = None
        self.argv: list[str] = []
        self.started = _now()

    # --- before the reset ----------------------------------------------------------------------

    def refusals(self) -> None:
        """Refuse without touching the result dir, so a previous run's diagnostics survive.

        Raises:
            RunError: ``E_NO_USERNS``, ``E_ENV_MISSING``, ``E_ENV_UNCHECKED``, ``E_ENV_CHANGED`` (the dist-info
                METADATA in the venv no longer name the installed set), ``E_GPU_MISSING`` or ``E_DRIVER_OLD``.
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
        if env.installed_from_metadata(venv_path) != venv.packages:
            raise RunError("E_ENV_CHANGED")
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
            _empty_file(Path(self._render(sandbox.R_STARTUP_TEMPLATE)))
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
        system = {os.path.realpath(p) for p in sandbox.SYSTEM_PATH} | set(sandbox.SYSTEM_PATH)
        raw_path = module_env.get("PATH")
        module_path = raw_path if isinstance(raw_path, str) else ""
        self.r_bins = [p for p in module_path.split(":") if p and p not in system and os.path.realpath(p) not in system]
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
        argv = [sandbox.SH, "-c", WRAPPER, "sh", str(self._cwd()), "Rscript", "--vanilla", str(script), str(r_lib)]
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
            if entry.root.startswith("up:"):
                real_root = os.path.realpath(root)
                if any(not _under(os.path.realpath(root / rel), real_root) for rel in files):
                    raise RunError("E_OUTPUT_SYMLINK")
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
            masked=[] if self.step.hf_weights else [self.config.cache / "huggingface"],
        )

    def _cwd(self) -> Path:
        cwd = Path(self._render(self.step.cwd))
        cwd = cwd if cwd.is_absolute() else Path(os.path.realpath(self.keyed.checkout.worktree)) / cwd
        cwd = Path(os.path.realpath(cwd))
        allowed = [os.path.realpath(self.keyed.checkout.worktree), os.path.realpath(self.work)]
        if not cwd.is_dir() or not any(cwd == Path(a) or cwd.is_relative_to(a) for a in allowed):
            raise RunError("E_CWD")
        return cwd

    def _interpreter_dirs(self) -> list[str]:
        """Where the step's program may live: the venv (Python kinds), the R module's and native bin dirs (R)."""
        step = self.step
        native = [os.fspath(d) for n in step.resources.native for d in self.config.profile.native[n].bin_dirs]
        venv = [self._render("{venv}")]
        if step.kind in ("python", "notebook"):
            places = venv
        elif step.kind == "r":
            places = [*self.r_bins, *native]
        else:
            places = [*venv, *self.r_bins, *native]
        return [os.path.realpath(p) for p in places]

    def _exec(self) -> None:
        step, config = self.step, self.config
        argv = [self._render(a) for a in step.argv]
        cwd = self._cwd()
        self.argv = argv
        python = os.path.join(self._render("{venv}"), "bin", "python")
        preamble = _PREAMBLE_CODE.format(canary=sandbox.canary(config.profile.canary_host))
        allowed = json.dumps(self._interpreter_dirs())
        read, write = os.pipe()
        wrapped = [sandbox.SH, "-c", WRAPPER, "sh", str(cwd), python, "-I", "-B", "-c", preamble, str(write), allowed]
        wrapped += argv
        self.rc, data = self._spawn(self._sandbox(wrapped, [self.work], tmp=self.tmp), self.env, "step", (read, write))
        self.preamble = _parse_preamble(data)
        self.executed = (
            self.preamble is not None
            and self.preamble["canary_failed"] is True
            and self.preamble["interpreter_ok"] is True
        )
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
        elif pre["interpreter_ok"] is not True:
            self.codes.add("E_STEP_INTERPRETER")
        if pre is not None and pre["canary_failed"] is True:
            self._torch_guard(pre)
        self._reserved_names()
        self._symlinks_out()
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

    def _symlinks_out(self) -> None:
        """Refuse any symlink under ``work/`` that resolves outside ``work/``; none is followed."""
        work = os.path.realpath(self.work)
        for top, dirs, files in os.walk(self.work, followlinks=False):
            for name in [*dirs, *files]:
                path = os.path.join(top, name)
                if os.path.islink(path) and not _under(os.path.realpath(path), work):
                    self.codes.add("E_OUTPUT_SYMLINK")
                    return

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
        if "E_OUTPUT_SYMLINK" in self.codes:
            empty = digest.input_set_digest(self.result, [])
            return [provenance.Artifact(digest=empty[0], count=empty[1]) for _ in self.step.outputs]
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
    if not isinstance(parsed["interpreter_ok"], bool):
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


def resolve_step(
    step: StepSpec,
    analysis: str,
    config: RunConfig,
    upstream: Mapping[str, StepOutcome] | None = None,
) -> StepOutcome:
    """Key a step and report its done result, without running, claiming or linking anything.

    A batch job for a downstream step uses this for each upstream step: the upstream job finished first
    (``afterok``), and its key is computed again here from the same inputs, because a key is never
    remembered.

    Args:
        step: The step.
        analysis: The name of the manifest the step belongs to.
        config: The run configuration.
        upstream: The outcome of each of ``step.needs``.

    Returns:
        ``skipped`` with exit 0 and the summary's codes when the result dir is done; ``never-ran`` when a
        need is not `StepOutcome.ok`; otherwise ``failed`` with ``E_UPSTREAM_NOT_DONE`` (the key is known
        but its result is not done) or the keying code.
    """
    upstream = upstream or {}
    if any(name not in upstream or not upstream[name].ok for name in step.needs):
        return StepOutcome(step.name, None, "never-ran", None, upstream_codes(step, upstream), None)
    with _quiet_log():
        try:
            keyed = _key(step, analysis, config, upstream)
        except (RunError, digest.DigestError, cache.CacheError) as error:
            return _failed(step, error.code)
        except (OSError, TypeError, ValueError):
            return _failed(step, "E_DRIVER")
    if not cache.is_done(keyed.result):
        return _failed(step, "E_UPSTREAM_NOT_DONE", keyed.key, keyed.result)
    return StepOutcome(step.name, keyed.key, "skipped", 0, _read_summary(keyed.result), keyed.result)


def _row(outcome: StepOutcome) -> dict[str, Any]:
    return {
        "step": outcome.step,
        "key": outcome.key,
        "status": outcome.status,
        "exit": outcome.exit,
        "codes": list(outcome.codes),
    }


def upstream_codes(step: StepSpec, upstream: Mapping[str, StepOutcome]) -> tuple[str, ...]:
    """Why a step never ran: the codes of its needs that did not succeed, else ``E_UPSTREAM_NOT_DONE``."""
    codes = {c for name in step.needs if name in upstream and not upstream[name].ok for c in upstream[name].codes}
    return tuple(sorted(codes)) or ("E_UPSTREAM_NOT_DONE",)


def never_ran_row(step: str, codes: Sequence[str] = ()) -> dict[str, Any]:
    """The ``run.json`` row of a step that did not run and will not: no key, no exit."""
    return {"step": step, "key": None, "status": "never-ran", "exit": None, "codes": list(codes)}


def read_run_record(path: Path) -> list[dict[str, Any]]:
    """Read ``run.json``'s rows; an absent file has none.

    Args:
        path: The ``run.json`` file (its last component is not followed if it is a symlink).

    Returns:
        The rows, each a mapping with a string ``step``.

    Raises:
        ValueError: If the file is not ``{"steps": [{"step": <str>, ...}, ...]}``.
        OSError: If it exists and cannot be read.
    """
    try:
        data = json.loads(_read_nofollow(path))
    except FileNotFoundError:
        return []
    rows = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(rows, list) or not all(isinstance(r, dict) and isinstance(r.get("step"), str) for r in rows):
        raise ValueError("run.json is not a run record")
    return rows


def update_run_record(
    runs_root: Path,
    date: datetime.date,
    analysis: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    order: Sequence[str],
    pending_only: bool = False,
    submission: str | None = None,
    allow_untagged: bool = False,
    untagged_only: bool = False,
) -> None:
    """Merge rows into ``<runs_root>/<date>/<analysis>/run.json`` under an exclusive lock.

    Each row replaces the fields of the same step's row and keeps the others (a submitted job's ``job``
    id survives its result). Rows are kept in ``order``, then by name.

    Args:
        runs_root: The profile's runs root.
        date: The run date.
        analysis: The manifest name.
        rows: Rows with at least ``step``.
        order: Step names in DAG order.
        pending_only: Change only rows that exist and are still ``pending``; others are left as they are.
        submission: Change only rows tagged with this submission id.
        allow_untagged: With ``submission``, also change rows with no submission tag (or no row at all), as a
            job's own row may be one a local run wrote.
        untagged_only: Change only rows with no submission tag (or no row): a local run never rewrites a
            submission's rows.

    Raises:
        OSError: If the run dir, the lock or the file cannot be written.
        ValueError: If the existing ``run.json`` is not a run record.
        cache.CacheError: If the analysis name or date cannot name a run dir.
    """
    run = cache.run_dir(runs_root, date, analysis)
    run.mkdir(parents=True, exist_ok=True)
    with build.file_lock(run / RUN_LOCK):
        current = {r["step"]: r for r in read_run_record(run / RUN_JSON)}
        for row in rows:
            old = current.get(row["step"])
            if pending_only and (old is None or old.get("status") != PENDING):
                continue
            if submission is not None and not _tag_matches(old, submission, allow_untagged):
                continue
            if untagged_only and old is not None and old.get("submission") is not None:
                continue
            current[row["step"]] = {**(old or {}), **row}
        _write_ordered(run, current, order)


def _tag_matches(old: Mapping[str, Any] | None, submission: str, allow_untagged: bool) -> bool:
    if old is None or old.get("submission") is None:
        return allow_untagged
    return bool(old.get("submission") == submission)


class SubmissionOverlap(Exception):
    """A run dir holds ``pending`` rows of another submission; ``others`` names those submissions."""

    code = "E_SUBMIT_OVERLAP"

    def __init__(self, analysis: str, others: Sequence[str]) -> None:
        """Store the analysis and the other submission ids."""
        self.analysis = analysis
        self.others = tuple(others)
        super().__init__(f"E_SUBMIT_OVERLAP: {analysis}: {', '.join(self.others)}")


def reserve_run_record(
    runs_root: Path, date: datetime.date, analysis: str, order: Sequence[str], submission: str
) -> dict[str, dict[str, Any] | None]:
    """Under the record's lock: refuse another submission's ``pending`` rows, then write placeholder rows.

    Every step of ``order`` gets ``{status: pending, job: null, submission}``. The check and the write share
    one lock, so of two submissions racing for one run dir exactly one reserves it.

    Args:
        runs_root: The profile's runs root.
        date: The run date.
        analysis: The manifest name.
        order: The analysis's step names in DAG order.
        submission: This submission's id.

    Returns:
        The rows the placeholders replaced (None where there was none), for `release_reservation`.

    Raises:
        SubmissionOverlap: If another submission still has a ``pending`` row there.
    """
    run = cache.run_dir(runs_root, date, analysis)
    run.mkdir(parents=True, exist_ok=True)
    with build.file_lock(run / RUN_LOCK):
        current = {r["step"]: r for r in read_run_record(run / RUN_JSON)}
        others = sorted(
            {
                str(r.get("submission"))
                for r in current.values()
                if r.get("status") == PENDING and r.get("submission") != submission
            }
        )
        if others:
            raise SubmissionOverlap(analysis, others)
        snapshot = {name: current.get(name) for name in order}
        for name in order:
            current[name] = {**never_ran_row(name), "status": PENDING, "job": None, "submission": submission}
        _write_ordered(run, current, order)
    return snapshot


def release_reservation(
    runs_root: Path,
    date: datetime.date,
    analysis: str,
    order: Sequence[str],
    submission: str,
    snapshot: Mapping[str, Mapping[str, Any] | None],
    codes: Sequence[str] | None,
) -> None:
    """Undo this submission's still-``pending`` rows: restore the row each replaced, or settle it.

    A row another writer changed since (a job that ran, another submission) is left alone. Where the
    placeholder replaced a row, that row comes back; otherwise the row is removed (``codes`` None) or becomes
    ``never-ran`` with ``codes``, tagged with this submission.
    """
    run = cache.run_dir(runs_root, date, analysis)
    with build.file_lock(run / RUN_LOCK):
        current = {r["step"]: r for r in read_run_record(run / RUN_JSON)}
        for name in order:
            row = current.get(name)
            if row is None or row.get("submission") != submission or row.get("status") != PENDING:
                continue
            before = snapshot.get(name)
            if before is not None:
                current[name] = dict(before)
            elif codes is None:
                del current[name]
            else:
                current[name] = {**row, **never_ran_row(name, codes)}
        _write_ordered(run, current, order)


def _write_ordered(run: Path, current: Mapping[str, Mapping[str, Any]], order: Sequence[str]) -> None:
    position = {name: index for index, name in enumerate(order)}
    steps = sorted(current.values(), key=lambda r: (position.get(r["step"], len(position)), r["step"]))
    _write_json(run / RUN_JSON, {"steps": steps})


def check_no_live_submission(runs_root: Path, date: datetime.date, analyses: Sequence[str]) -> None:
    """Refuse a local run while any submission still has ``pending`` rows in one of the run dirs.

    Raises:
        SubmissionOverlap: Naming the analysis and the submissions.
    """
    for name in analyses:
        rows = read_run_record(cache.run_dir(runs_root, date, name) / RUN_JSON)
        others = sorted({str(r.get("submission")) for r in rows if r.get("status") == PENDING and r.get("submission")})
        if others:
            raise SubmissionOverlap(name, others)


def pending_jobs(manifests: Sequence[Manifest], config: RunConfig, submission: str) -> list[str]:
    """The job ids recorded in a submission's rows that are still ``pending``."""
    jobs = []
    for manifest in manifests:
        rows = read_run_record(cache.run_dir(config.profile.runs_root, config.date, manifest.name) / RUN_JSON)
        jobs += [
            str(r["job"])
            for r in rows
            if r.get("status") == PENDING and r.get("submission") == submission and r.get("job")
        ]
    return jobs


def descendants(specs: Mapping[str, StepSpec], roots: set[str]) -> set[str]:
    """Every step that depends, directly or not, on one of ``roots``."""
    return _descendants(specs, roots)


def _descendants(specs: Mapping[str, StepSpec], roots: set[str]) -> set[str]:
    found: set[str] = set()
    changed = True
    while changed:
        changed = False
        for spec in specs.values():
            if spec.name not in found and any(n in roots or n in found for n in spec.needs):
                found.add(spec.name)
                changed = True
    return found


def finalize_run_records(
    manifests: Sequence[Manifest],
    config: RunConfig,
    submission: str | None = None,
    state_of: Callable[[str], str | None] | None = None,
) -> None:
    """Settle every row still ``pending`` once every job of a submission has ended.

    Steps are taken in DAG order. A placeholder row (a submission's, with no job id) was never submitted:
    ``never-ran`` with ``E_SUBMIT_ABORTED``. A pending step with a need that did not succeed was cancelled with its
    upstream (``--kill-on-invalid-dep``): ``never-ran`` with ``E_UPSTREAM_NOT_DONE``. Otherwise its own job
    ended without writing its row (killed, timed out, node failure): ``never-ran`` with
    ``E_JOB_ENDED_WITHOUT_RECORD`` and, when ``state_of`` knows it, the job's final state as ``slurm_state``.
    A requeued job's later row replaces either.

    Args:
        manifests: The manifests of the submission.
        config: The run configuration (its ``date`` is the submission's run date).
        submission: Only rows tagged with this submission id are touched; None touches every pending row.
        state_of: Maps a job id to its final state word (``TIMEOUT``, ``NODE_FAIL``, ``CANCELLED``), or None.
    """
    ordered = toposort([s for m in manifests for s in m.steps])
    analysis = {s.name: m.name for m in manifests for s in m.steps}
    rows: dict[str, dict[str, Any]] = {}
    for manifest in manifests:
        record = cache.run_dir(config.profile.runs_root, config.date, manifest.name) / RUN_JSON
        rows.update({r["step"]: r for r in read_run_record(record) if r["step"] in analysis})
    settled: dict[str, list[dict[str, Any]]] = {}
    for step in ordered:
        row = rows.get(step.name)
        if row is None or row.get("status") != PENDING:
            continue
        if submission is not None and row.get("submission") != submission:
            continue
        upstream_bad = any(not _row_ok(rows.get(n)) for n in step.needs)
        if row.get("job") is None and row.get("submission") is not None:
            new = never_ran_row(step.name, ("E_SUBMIT_ABORTED",))
        elif upstream_bad:
            new = never_ran_row(step.name, ("E_UPSTREAM_NOT_DONE",))
        else:
            new = never_ran_row(step.name, ("E_JOB_ENDED_WITHOUT_RECORD",))
            job = row.get("job")
            state = state_of(str(job)) if state_of is not None and job else None
            if state is not None:
                new["slurm_state"] = state
        rows[step.name] = {**row, **new}
        settled.setdefault(analysis[step.name], []).append(new)
    for name, new_rows in settled.items():
        order = [s.name for s in ordered if analysis[s.name] == name]
        update_run_record(
            config.profile.runs_root,
            config.date,
            name,
            new_rows,
            order=order,
            pending_only=True,
            submission=submission,
        )


def _row_ok(row: Mapping[str, Any] | None) -> bool:
    return row is not None and row.get("status") in ("executed", "skipped") and row.get("exit") == 0


def run_steps(
    manifests: Sequence[Manifest],
    config: RunConfig,
    only: Collection[str] | None = None,
    submission: str | None = None,
) -> list[StepOutcome]:
    """Run every step of a set of manifests in DAG order (`dag.toposort`), or only the named ones.

    A step whose need did not succeed (`StepOutcome.ok`) is ``never-ran``; the others run or skip as
    `run_step` decides. Each analysis gets ``<runs_root>/<date>/<analysis>/run.json`` listing its steps'
    keys, statuses, exits and codes; when it cannot be written, every step of that analysis also carries
    ``E_DRIVER`` and a non-zero exit (its result dirs are unchanged).

    With ``only``, the named steps run and every step they depend on, directly or not, is resolved with
    `resolve_step` (keyed, never run). Only the named steps' rows are written, merged into ``run.json``
    (`update_run_record`); a named step that did not succeed also turns its dependents' ``pending`` rows
    into ``never-ran``.

    Args:
        manifests: The manifests; step names are unique across them.
        config: The run configuration.
        only: Step names to run; None runs every step.
        submission: With ``only``, the submission id whose ``pending`` dependent rows a failure may settle.

    Returns:
        One outcome per step run, in run order (with ``only``, the named steps only).

    A local run (``submission`` None) refuses while any submission has ``pending`` rows in one of the run
    dirs, and never rewrites a row a submission tagged; rows of steps it did not run are kept.

    Raises:
        ValueError: If ``only`` names a step outside the manifests.
        SubmissionOverlap: For a local run while a submission is live in one of the run dirs.
    """
    analysis = {s.name: m.name for m in manifests for s in m.steps}
    specs = {s.name: s for m in manifests for s in m.steps}
    ordered = toposort(list(specs.values()))
    if submission is None:
        check_no_live_submission(config.profile.runs_root, config.date, sorted(set(analysis.values())))
    if only is not None:
        return _run_only(config, analysis, specs, ordered, set(only), submission)
    outcomes: dict[str, StepOutcome] = {}
    for step in ordered:
        upstream = {name: outcomes[name] for name in step.needs}
        schemas = {name: specs[name].metrics_schema for name in _expect_sources(step)}
        outcomes[step.name] = run_step(step, analysis[step.name], config, upstream, schemas)
    for name in sorted(set(analysis.values())):
        rows = [outcomes[s.name] for s in ordered if analysis[s.name] == name]
        order = [s.name for s in ordered if analysis[s.name] == name]
        try:
            update_run_record(
                config.profile.runs_root, config.date, name, [_row(o) for o in rows], order=order, untagged_only=True
            )
        except (OSError, TypeError, ValueError, RunError, cache.CacheError):
            for o in rows:
                outcomes[o.step] = o._replace(codes=tuple(sorted({*o.codes, "E_DRIVER"})), exit=o.exit or EXIT_DRIVER)
    return [outcomes[s.name] for s in ordered]


def _run_only(
    config: RunConfig,
    analysis: Mapping[str, str],
    specs: Mapping[str, StepSpec],
    ordered: Sequence[StepSpec],
    only: set[str],
    submission: str | None,
) -> list[StepOutcome]:
    unknown = sorted(only - set(specs))
    if unknown:
        raise ValueError(f"steps not in the manifests: {unknown}")
    wanted, stack = set(only), list(only)
    while stack:
        for need in specs[stack.pop()].needs:
            if need not in wanted:
                wanted.add(need)
                stack.append(need)
    outcomes: dict[str, StepOutcome] = {}
    for step in ordered:
        if step.name not in wanted:
            continue
        upstream = {name: outcomes[name] for name in step.needs}
        if step.name in only:
            schemas = {name: specs[name].metrics_schema for name in _expect_sources(step)}
            outcomes[step.name] = run_step(step, analysis[step.name], config, upstream, schemas)
        else:
            outcomes[step.name] = resolve_step(step, analysis[step.name], config, upstream)
    selected = [outcomes[s.name] for s in ordered if s.name in only]
    failed = [o for o in selected if not o.ok]
    dependents = _descendants(specs, {o.step for o in failed}) - only
    why = tuple(sorted({c for o in failed for c in o.codes})) or ("E_UPSTREAM_NOT_DONE",)
    for name in sorted({analysis[o.step] for o in selected} | {analysis[d] for d in dependents}):
        order = [s.name for s in ordered if analysis[s.name] == name]
        own = [_row(o) for o in selected if analysis[o.step] == name]
        gone = [never_ran_row(d, why) for d in order if d in dependents]
        try:
            if own:
                update_run_record(
                    config.profile.runs_root,
                    config.date,
                    name,
                    own,
                    order=order,
                    submission=submission,
                    allow_untagged=True,
                    untagged_only=submission is None,
                )
            if gone:
                update_run_record(
                    config.profile.runs_root,
                    config.date,
                    name,
                    gone,
                    order=order,
                    pending_only=True,
                    submission=submission,
                )
        except (OSError, TypeError, ValueError, RunError, cache.CacheError):
            for o in selected:
                if analysis[o.step] == name:
                    outcomes[o.step] = o._replace(
                        codes=tuple(sorted({*o.codes, "E_DRIVER"})), exit=o.exit or EXIT_DRIVER
                    )
    return [outcomes[s.name] for s in ordered if s.name in only]
