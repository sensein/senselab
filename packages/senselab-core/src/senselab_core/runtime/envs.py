"""Locked envs: `ensure_env` builds an `Env.isolated` or `Env.server` env from its lock and describes `Env.host`.

An isolated env is a uv venv on a uv-managed interpreter (`--python-preference only-managed`, the lock's named
version, `t` for free-threaded), installed from its lock: a lock with `torch:` pins installs them from the PyTorch
wheel index routed to the host's CUDA (`pick_torch_index`), then the lock body with `--no-deps`. Under `root`, env
`<name>` lives in `<name>[-<tag>]-<uuid>/`, and `<name>[-<tag>].json` (`env-record@1.0`) names the current build;
both are written under a `Lease` on `<name>[-<tag>]`. A build is never moved or removed here, so a broken lease
costs a duplicate build, never a mixed env, and a superseded build stays until an explicit clean-up. The worker
runtime, and the worker modules a backend declares, are copied into every env as an overlay that leaves its
wheel set unchanged.
"""

import errno
import importlib.metadata
import logging
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import sysconfig
import tempfile
import tomllib
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import senselab_worker
from packaging.requirements import InvalidRequirement, Requirement
from pydantic import ValidationError
from senselab_worker.digest import file_digest, identity_digest

from senselab_core.kernel.declare import Env
from senselab_core.kernel.formats import FormatTag, Formatted
from senselab_core.runtime.config import load_yaml
from senselab_core.runtime.hardware import _run
from senselab_core.runtime.locks import TIMEOUT_S, Lease
from senselab_core.runtime.network import Network, require_online
from senselab_core.runtime.profiles import Envs
from senselab_core.runtime.storage import write_atomically

logger = logging.getLogger("senselab")

PYTORCH_INDEX_BASE = "https://download.pytorch.org/whl"
CPU_INDEX = f"{PYTORCH_INDEX_BASE}/cpu"
CUDA_INDEXES: dict[str, tuple[int, int]] = {
    "cu130": (13, 0),
    "cu128": (12, 8),
    "cu126": (12, 6),
    "cu124": (12, 4),
    "cu121": (12, 1),
}
"""The PyTorch CUDA wheel indexes, newest first."""
DEFAULT_MAX_CUDA = CUDA_INDEXES["cu128"]
"""The CUDA ceiling of a lock that names no `torch-index:`."""
PYTHON_PREFERENCE = "only-managed"
ENV_RECORD_FORMAT = "env-record@1.0"
WHEEL_SET_TAG = "wheel-set@1"
_PYTHON_VERSION = re.compile(r"^\d+\.\d+t?$")
_SMI_CUDA = re.compile(r"CUDA Version:\s*(\d+)\.(\d+)")
_NVCC_CUDA = re.compile(r"release\s+(\d+)\.(\d+)")
_NO_WHEEL = re.compile(
    r"(?:no matching distribution(?: found)?(?: for)?|could not find a (?:version|distribution)"
    r"(?: that satisfies)?(?: the requirement)?)\s+`([^`]+)`",
    re.IGNORECASE,
)
_NO_INTERPRETER = re.compile(r"no interpreter found|no download found", re.IGNORECASE)
OVERLAY_DIGEST_TAG = "env-overlay@1"
"""The `identity_digest` tag of an env's overlay: the worker runtime identity and each worker module's digest."""
LOCK_DIGEST_TAG = "env-lock@1"
"""The `identity_digest` tag of a lock's inputs, recorded in its `# identity-digest:` header line."""
LOCK_INPUTS_FORMAT = "env-lock-inputs@1.0"
TORCH_PACKAGES = frozenset({"torch", "torchaudio"})
_TORCH_VARIANTS = ("nvidia-", "triton", "pytorch-triton", "cuda-")
_LOCK_LINE = re.compile(r"^([A-Za-z0-9._-]+)(\[[^\]]*\])?\s*(==|@)\s*([^\s;#\\]+)")
_VIA = re.compile(r"#\s*via\s+(.*)$")
_LINUX_X86_64 = ("linux", "x86_64")
_MACOS_ARM64 = ("macosx", "arm64")


class EnvLockError(ValueError):
    """A lock file without the header fields an env is built from."""


class HostLockLacksBackend(EnvLockError):
    """The senselab lock does not cover a backend declared `Env.host()`; the message names the requirements."""


class ManagedPythonMissing(RuntimeError):
    """No uv-managed interpreter of the version an env needs; the message names `uv python install <version>`.

    Attributes:
        version: The interpreter version, `t` included for free-threaded.
    """

    def __init__(self, version: str, detail: str) -> None:
        """Record `version`; `detail` says what was tried."""
        super().__init__(f"{detail}; install one with `uv python install {version}`")
        self.version = version


class CudaCompatibilityError(RuntimeError):
    """The routed PyTorch index serves no wheel for the lock's torch pins."""

    def __init__(self, host: "HostCuda", index: "TorchIndex", packages: list[str]) -> None:
        """Name the host CUDA, the index tried and the packages it lacks."""
        cuda = ".".join(map(str, host.version)) if host.version else "none"
        super().__init__(
            f"no wheel for {packages} on {index.url} (host CUDA {cuda}, from {host.source}); set the execution "
            f"profile's envs.torch_index to another index, e.g. {CPU_INDEX}"
        )


@dataclass(frozen=True)
class HostCuda:
    """The host's CUDA version, `None` when no probe reports one, and the probe that reported it."""

    version: tuple[int, int] | None
    source: Literal["nvidia-smi", "nvcc", "none"]


@dataclass(frozen=True)
class TorchIndex:
    """The PyTorch wheel index torch is installed from: its URL, tag (`cu128`, `cpu`, `profile`) and source."""

    url: str
    tag: str
    source: Literal["probe", "profile"]


@dataclass(frozen=True)
class EnvLock:
    """A lock file's header: the interpreter version, the torch pins and the newest CUDA index they are served on."""

    path: Path
    python: str
    torch_pins: tuple[str, ...]
    max_cuda: tuple[int, int]

    @classmethod
    def read(cls, path: Path) -> "EnvLock":
        """Parse the `# key: value` header of the lock at `path`.

        Raises:
            EnvLockError: `python:` or `torch:` is missing, or a value is not one the builder knows.
        """
        header: dict[str, str] = {}
        for line in Path(path).read_text("utf-8").splitlines():
            if not line.startswith("#"):
                break
            key, sep, value = line[1:].partition(":")
            if sep:
                header.setdefault(key.strip(), value.strip())
        missing = [k for k in ("python", "torch") if k not in header]
        if missing:
            raise EnvLockError(f"lock {path} has no {missing} header line")
        if not _PYTHON_VERSION.fullmatch(header["python"]):
            raise EnvLockError(f"lock {path} names python {header['python']!r}, not <major>.<minor>[t]")
        ceiling = header.get("torch-index", "none")
        if ceiling not in (*CUDA_INDEXES, "cpu", "none"):
            raise EnvLockError(f"lock {path} names torch-index {ceiling!r}; known: {[*CUDA_INDEXES, 'cpu']}")
        torch = header["torch"]
        return cls(
            path=Path(path),
            python=header["python"],
            torch_pins=tuple(torch.split()) if torch != "none" else (),
            max_cuda=CUDA_INDEXES.get(ceiling, (0, 0) if ceiling == "cpu" else DEFAULT_MAX_CUDA),
        )


@dataclass(frozen=True)
class EnvHandle:
    """A ready env: its interpreter and the identities an execution context records.

    `lock_sha` is the lock file's sha256, `wheel_set` the digest of the installed distributions, `torch_index` the
    URL torch came from (None for a torch-free env), `python_version` the interpreter's (`t` for free-threaded),
    `lock` the lock file it was built from (None for a host env).
    """

    python: str
    python_version: str
    lock_sha: str
    wheel_set: str
    torch_index: str | None
    lock: Path | None = None

    def context_fields(self) -> dict[str, Any]:
        """Return the `ExecutionContext` fields this env decides: `env_lock`, `wheel_set` and `driver`."""
        driver = {"torch_index": self.torch_index} if self.torch_index else {}
        return {"env_lock": self.lock_sha, "wheel_set": self.wheel_set, "driver": driver}


class EnvRecord(Formatted):
    """The current build of one env under a root, at `<root>/<name>[-<tag>].json`."""

    format: FormatTag = ENV_RECORD_FORMAT
    directory: str
    lock_sha: str
    torch_index: str | None
    python_version: str
    wheel_set: str


@dataclass(frozen=True)
class Uv:
    """The `uv` command line every build runs through."""

    command: tuple[str, ...]

    @classmethod
    def find(cls) -> "Uv":
        """Return `uv` from `PATH`, else `~/.local/bin/uv` or `~/.cargo/bin/uv`.

        Raises:
            FileNotFoundError: none of them is an executable.
        """
        home = Path.home()
        for candidate in (shutil.which("uv"), home / ".local" / "bin" / "uv", home / ".cargo" / "bin" / "uv"):
            if candidate and os.access(candidate, os.X_OK):
                return cls((str(candidate),))
        raise FileNotFoundError("uv not found on PATH, in ~/.local/bin or in ~/.cargo/bin")

    def __call__(self, *args: str) -> subprocess.CompletedProcess[str]:
        """Run `uv <args>`, capturing its output; the caller reads `returncode`."""
        return subprocess.run([*self.command, *args], capture_output=True, text=True, check=False)


def detect_host_cuda(*, run: Callable[[list[str]], str] = _run) -> HostCuda:
    """Return the CUDA version `nvidia-smi` reports, else the one `nvcc --version` reports, else none.

    `run` executes a command and returns its stdout, raising when it cannot; tests inject it.
    """
    for command, pattern, source in (
        (["nvidia-smi"], _SMI_CUDA, "nvidia-smi"),
        (["nvcc", "--version"], _NVCC_CUDA, "nvcc"),
    ):
        try:
            match = pattern.search(run(command))
        except (OSError, subprocess.SubprocessError):
            continue
        if match:
            return HostCuda((int(match[1]), int(match[2])), source)  # type: ignore[arg-type]
    return HostCuda(None, "none")


def pick_torch_index(
    host: HostCuda, *, override: str | None = None, max_cuda: tuple[int, int] | None = None
) -> TorchIndex:
    """Return `override` when set, else the newest CUDA index at or below the host's CUDA capped at `max_cuda`.

    `max_cuda` defaults to `DEFAULT_MAX_CUDA`; a host without CUDA, or older than every index, gets the CPU index.
    """
    if override:
        return TorchIndex(override, "profile", "profile")
    if host.version is not None:
        effective = min(host.version, max_cuda or DEFAULT_MAX_CUDA)
        for tag, version in CUDA_INDEXES.items():
            if version <= effective:
                return TorchIndex(f"{PYTORCH_INDEX_BASE}/{tag}", tag, "probe")
    return TorchIndex(CPU_INDEX, "cpu", "probe")


def ensure_env(  # noqa: PLR0913 - the env, its lock, root and profile settings, and two injectable boundaries
    env: Env,
    lock: Path | None,
    root: Path,
    *,
    settings: Envs,
    requirements: Sequence[str] = (),
    workers: Mapping[str, Path] | None = None,
    uv: Uv | None = None,
    host_cuda: HostCuda | None = None,
    network: Network = "online",
) -> EnvHandle:
    """Return the env `env` built from `lock` and overlaid with `workers` under `root`, built once across processes.

    A published env is never changed: an env of another lock, torch index, worker runtime or worker module source is
    a separate build, named by its overlay digest, and the build it replaces stays for the workers running in it.

    For `Env.host()`, `lock` is the senselab lock, or None when none ships with the install, in which case the installed
    distributions are checked against `requirements`; the env is this interpreter's, which must be uv-managed.

    Args:
        env: The env a backend declares.
        lock: The lock file the env is installed from.
        root: The directory the profile's `env_roots` gives for `env.kind`; unused for a host env.
        settings: The profile's `envs`: `torch_index` replaces the probed index when set, and a build's lease is
            broken once older than `lease_stale_after_s`; a waiter waits that long plus `locks.TIMEOUT_S`.
        requirements: The backend's requirements; required for a host env, which is refused unless `lock` covers each.
        workers: The backend's worker modules, dotted name to source file; each is copied into the env's
            site-packages as that module, beside the worker runtime, when the env is built.
        uv: The `uv` command; `Uv.find()` when None.
        host_cuda: The host's CUDA; probed with `detect_host_cuda` when None and the lock pins torch.
        network: The profile's `network`; an `offline` profile uses a built env and builds none.

    Raises:
        ManagedPythonMissing: no uv-managed interpreter of the needed version, or the host's is not uv-managed.
        CudaCompatibilityError: the routed index has no wheel for the lock's torch pins.
        EnvLockError: the lock's header lacks what a build needs, or a host env got no `requirements`.
        HostLockLacksBackend: a host env whose `lock` does not pin every one of `requirements`.
        RuntimeError: another `uv` step failed; the message carries uv's stderr.
        LeaseTimeout, LeaseUnavailable: as `Lease`.
        Offline: the env is not built and `network` is `offline`.
    """
    uv = uv or Uv.find()
    if env.kind == "host":
        if not requirements:
            raise EnvLockError("an Env.host() backend must pass its requirements, so the senselab lock can be checked")
        if lock is None:
            lacking = installed_gaps(requirements)
            if lacking:
                raise HostLockLacksBackend(f"the installed packages do not satisfy {lacking}")
            return _host_env(None, uv, requirements)
        lacking = host_lock_gaps(lock, requirements)
        if lacking:
            raise HostLockLacksBackend(f"{lock} does not cover {lacking}; declare the backend Env.isolated instead")
        return _host_env(lock, uv, requirements)
    if lock is None:
        raise EnvLockError(f"a {env.kind} env is built from its lock, none was given")
    parsed = EnvLock.read(lock)
    index = None
    if parsed.torch_pins:
        host_cuda = host_cuda or detect_host_cuda()
        index = pick_torch_index(host_cuda, override=settings.torch_index, max_cuda=parsed.max_cuda)
    overlay = _overlay_digest(workers or {})
    stem = f"{env.name}-{index.tag}-{overlay[:16]}" if index else f"{env.name}-{overlay[:16]}"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_sha = file_digest("file@1", parsed.path)
    wanted = (lock_sha, index.url if index else None)
    current = _read_record(root, stem)
    if current is not None and (current.lock_sha, current.torch_index) == wanted:
        return _handle(root, current, parsed.path)
    stale_after = settings.lease_stale_after_s
    with Lease(root / stem, stale_after=stale_after, timeout=stale_after + TIMEOUT_S):
        current = _read_record(root, stem)
        if current is None or (current.lock_sha, current.torch_index) != wanted:
            require_online(network, f"building env {stem}")
            current = _build(root, stem, parsed, lock_sha, index, uv, host_cuda, workers or {})
            write_atomically(root / f"{stem}.json", current.model_dump_json().encode())
    return _handle(root, current, parsed.path)


def _handle(root: Path, record: EnvRecord, lock: Path) -> EnvHandle:
    return EnvHandle(
        python=str(_python(root / record.directory)),
        python_version=record.python_version,
        lock_sha=record.lock_sha,
        wheel_set=record.wheel_set,
        torch_index=record.torch_index,
        lock=lock,
    )


def _read_record(root: Path, stem: str) -> EnvRecord | None:
    try:
        record = EnvRecord.model_validate_json((root / f"{stem}.json").read_bytes())
    except FileNotFoundError:
        return None
    except (OSError, ValidationError) as err:
        logger.warning("env record %s.json is unreadable and is rebuilt: %s", stem, err)
        return None
    if not re.fullmatch(rf"{re.escape(stem)}-[0-9a-f]{{32}}", record.directory):
        logger.warning("env record %s.json names %r, not a build of %s; it is rebuilt", stem, record.directory, stem)
        return None
    return record if (root / record.directory).is_dir() else None


def _host_env(lock: Path | None, uv: Uv, requirements: Sequence[str]) -> EnvHandle:
    suffix = "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""
    managed = _check(uv("python", "dir"), "find the uv-managed interpreters").stdout.strip()
    if not managed or not Path(sys.base_prefix).resolve().is_relative_to(Path(managed).resolve()):
        raise ManagedPythonMissing(
            f"{sys.version_info.major}.{sys.version_info.minor}{suffix}",
            f"the host interpreter {sys.executable} is not uv-managed",
        )
    return EnvHandle(
        python=sys.executable,
        python_version=platform.python_version() + suffix,
        lock_sha=file_digest("file@1", lock) if lock else _requirements_digest(requirements),
        wheel_set=_wheel_set(Path(sysconfig.get_paths()["purelib"])),
        torch_index=None,
    )


def _build(  # noqa: PLR0913 - one build's inputs
    root: Path,
    stem: str,
    lock: EnvLock,
    lock_sha: str,
    index: TorchIndex | None,
    uv: Uv,
    host_cuda: HostCuda | None,
    workers: Mapping[str, Path],
) -> EnvRecord:
    directory = root / f"{stem}-{uuid.uuid4().hex}"
    logger.info("building env %s on Python %s (torch index %s)", stem, lock.python, index.tag if index else "none")
    try:
        created = uv("venv", "--python", lock.python, "--python-preference", PYTHON_PREFERENCE, str(directory))
        if created.returncode and _NO_INTERPRETER.search(created.stderr):
            raise ManagedPythonMissing(
                lock.python,
                f"uv has no managed Python {lock.python} "
                f"(--python-preference {PYTHON_PREFERENCE}): {created.stderr.strip()}",
            )
        _check(created, f"create a venv on Python {lock.python}")
        python = str(_python(directory))
        if index is not None:
            torch = uv("pip", "install", "--index-url", index.url, "--python", python, *lock.torch_pins)
            missing = _NO_WHEEL.findall(torch.stderr)
            if torch.returncode and missing:
                raise CudaCompatibilityError(host_cuda or HostCuda(None, "none"), index, list(dict.fromkeys(missing)))
            _check(torch, "install torch")
        body = ("pip", "install", "--python", python, "--no-deps", "--no-sources", "--requirement", str(lock.path))
        _check(uv(*body), "install the lock")
        _overlay(_site(directory), workers)
        _group_readable(directory)
        version = re.search(r"^version(?:_info)?\s*=\s*(\S+)", (directory / "pyvenv.cfg").read_text(), re.MULTILINE)
    except BaseException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return EnvRecord(
        directory=directory.name,
        lock_sha=lock_sha,
        torch_index=index.url if index else None,
        python_version=(version[1] if version else lock.python.rstrip("t"))
        + ("t" if lock.python.endswith("t") else ""),
        wheel_set=_wheel_set(_site(directory)),
    )


def _check(result: subprocess.CompletedProcess[str], what: str) -> subprocess.CompletedProcess[str]:
    if result.returncode:
        raise RuntimeError(f"uv could not {what} (exit {result.returncode}): {result.stderr.strip()}")
    return result


def _python(directory: Path) -> Path:
    return directory / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _site(directory: Path) -> Path:
    found = (
        [directory / "Lib" / "site-packages"]
        if sys.platform == "win32"
        else list(directory.glob("lib/python*/site-packages"))
    )
    if len(found) != 1:
        raise RuntimeError(f"env {directory.name} has {len(found)} site-packages directories, not one")
    return found[0]


def _wheel_set(site: Path) -> str:
    """Return `identity_digest("wheel-set@1", [[name, version], ...])` over the `*.dist-info` names in `site`."""
    installed = sorted(p.name.removesuffix(".dist-info").rpartition("-")[::2] for p in site.glob("*.dist-info"))
    return identity_digest(WHEEL_SET_TAG, [list(pair) for pair in installed])


def _overlay_digest(workers: Mapping[str, Path]) -> str:
    modules = [[module, file_digest("file@1", source)] for module, source in sorted(workers.items())]
    return identity_digest(OVERLAY_DIGEST_TAG, [senselab_worker.runtime_identity(), modules])


def _overlay(site: Path, workers: Mapping[str, Path]) -> None:
    """Copy this worker runtime and the backend's worker modules into a new env's site-packages."""
    shutil.copytree(
        Path(senselab_worker.__file__).parent, site / "senselab_worker", ignore=shutil.ignore_patterns("__pycache__")
    )
    for module, source in workers.items():
        *packages, name = module.split(".")
        target = site.joinpath(*packages, f"{name}.py")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _group_readable(directory: Path) -> None:
    """Give the group read on every file and traverse on every directory, execute where the owner has it."""
    for here, _, files in os.walk(directory):
        for path, is_dir in ((Path(here), True), *((Path(here) / f, False) for f in files)):
            try:
                mode = path.lstat().st_mode
                if stat.S_ISLNK(mode):
                    continue
                exec_bit = stat.S_IXGRP if is_dir or mode & stat.S_IXUSR else 0
                path.chmod(mode | stat.S_IRGRP | exec_bit)
            except OSError as err:
                if err.errno not in (errno.EPERM, errno.EACCES):
                    raise


@dataclass(frozen=True)
class LockInputs:
    """Everything a lock is compiled from; `lock_digest` covers every field but `name`.

    `exclude_newer` is the RFC 3339 cutoff for uploads, `quarantined` the constraints applied to the resolution,
    `base` the requirements resolved into every lock beside `requirements`, `excluded` the distributions left out of
    the body, `max_cuda` the CUDA index ceiling (None: `DEFAULT_MAX_CUDA`).
    """

    name: str
    requirements: tuple[str, ...]
    python: str
    exclude_newer: str
    max_cuda: tuple[int, int] | None = None
    excluded: tuple[str, ...] = ()
    quarantined: tuple[str, ...] = ()
    base: tuple[str, ...] = ()


def read_lock_inputs(path: Path) -> dict[str, LockInputs]:
    """Read the `env-lock-inputs@1.0` file at `path`: its `quarantine` and `base_requirements`, and each env's inputs.

    Raises:
        EnvLockError: the format is not `LOCK_INPUTS_FORMAT`, or an env lacks `python`, `requirements` or
            `exclude_newer`.
    """
    raw = load_yaml(Path(path), "env lock inputs")
    if raw.get("format") != LOCK_INPUTS_FORMAT:
        raise EnvLockError(f"{path}: format is {raw.get('format')!r}, not {LOCK_INPUTS_FORMAT!r}")
    quarantined = tuple(raw["quarantine"]["value"])
    base = tuple(raw["base_requirements"]["value"])
    out = {}
    for name, spec in raw["envs"].items():
        missing = [k for k in ("python", "requirements", "exclude_newer") if k not in spec]
        if missing:
            raise EnvLockError(f"{path}: env {name!r} lacks {missing}")
        cuda = spec.get("max_cuda")
        out[name] = LockInputs(
            name=name,
            requirements=tuple(spec["requirements"]),
            python=str(spec["python"]),
            exclude_newer=str(spec["exclude_newer"]),
            max_cuda=(cuda[0], cuda[1]) if cuda else None,
            excluded=tuple(spec.get("excluded", ())),
            quarantined=quarantined,
            base=base,
        )
    return out


def lock_digest(inputs: LockInputs) -> str:
    """Return the digest of `inputs`, tagged `LOCK_DIGEST_TAG`; the env's name is not part of it."""
    return identity_digest(
        LOCK_DIGEST_TAG,
        {
            "requirements": list(inputs.requirements),
            "base": list(inputs.base),
            "quarantined": list(inputs.quarantined),
            "python": inputs.python,
            "exclude_newer": inputs.exclude_newer,
            "max_cuda": list(inputs.max_cuda) if inputs.max_cuda else None,
            "exclude": sorted(_normalize(n) for n in inputs.excluded),
        },
    )


def lock_identity_digest(path: Path) -> tuple[str, str]:
    """Return `(tag, digest)` from the `# identity-digest:` header of the lock at `path`.

    Raises:
        EnvLockError: the lock has no such line, or it is not `<tag> <64 hex>`.
    """
    for line in Path(path).read_text("utf-8").splitlines():
        if not line.startswith("#"):
            break
        if line.startswith("# identity-digest:"):
            found = re.fullmatch(r"\s*(\S+) ([0-9a-f]{64})\s*", line.partition(":")[2])
            if found:
                return found[1], found[2]
            break
    raise EnvLockError(f"lock {path} has no `# identity-digest: <tag> <sha256>` header line")


def host_lock_gaps(lock: Path, requirements: Sequence[str]) -> list[str]:
    """Return the `requirements` the `uv.lock` at `lock` does not cover.

    A requirement whose marker is false on this host is skipped. Any other is covered when its distribution is a
    locked package at a version its specifier admits (pre-releases included) and, for each extra it names, the
    package's locked optional dependencies for that extra are locked too. A requirement naming a URL is never covered.

    Raises:
        EnvLockError: a requirement is not a valid requirement string.
    """
    packages: dict[str, dict[str, Any]] = {
        _normalize(str(p["name"])): p for p in tomllib.loads(Path(lock).read_text("utf-8")).get("package", [])
    }
    gaps = []
    for text in requirements:
        try:
            req = Requirement(text)
        except InvalidRequirement as err:
            raise EnvLockError(f"invalid requirement {text!r}: {err}") from err
        if req.marker is not None and not req.marker.evaluate():
            continue
        package = packages.get(_normalize(req.name))
        covered = (
            package is not None and req.url is None and req.specifier.contains(package["version"], prereleases=True)
        )
        if covered and package is not None:
            extras = package.get("optional-dependencies", {})
            covered = all(
                extra in extras and all(_normalize(d["name"]) in packages for d in extras[extra])
                for extra in req.extras
            )
        if not covered:
            gaps.append(text)
    return gaps


def _requirements_digest(requirements: Sequence[str]) -> str:
    """Return the `wheel-set@1` digest of each requirement's installed `[name, version]`, for a lockless host env."""
    pairs = sorted(
        [_normalize(Requirement(t).name), importlib.metadata.version(Requirement(t).name)]
        for t in requirements
        if not (Requirement(t).marker and not Requirement(t).marker.evaluate())  # type: ignore[union-attr]
    )
    return identity_digest(WHEEL_SET_TAG, pairs)


def installed_gaps(requirements: Sequence[str]) -> list[str]:
    """Return the `requirements` the installed distributions do not satisfy.

    A requirement whose marker is false on this host is skipped; any other needs its distribution installed at a version
    its specifier admits (pre-releases included). A requirement naming a URL is never satisfied.

    Raises:
        EnvLockError: a requirement is not a valid requirement string.
    """
    gaps = []
    for text in requirements:
        try:
            req = Requirement(text)
        except InvalidRequirement as err:
            raise EnvLockError(f"invalid requirement {text!r}: {err}") from err
        if req.marker is not None and not req.marker.evaluate():
            continue
        try:
            version = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            gaps.append(text)
            continue
        if req.url is not None or not req.specifier.contains(version, prereleases=True):
            gaps.append(text)
    return gaps


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _torch_family(name: str) -> bool:
    return name in TORCH_PACKAGES or name.startswith(_TORCH_VARIANTS)


def _pip_compile(uv: Uv, in_file: Path, python: str, flags: Sequence[str]) -> str:
    args = [
        "pip", "compile", str(in_file), "--universal", "--python-version", python,
        "--python-preference", PYTHON_PREFERENCE, "--no-header", "--quiet", "--color", "never", "--no-sources",
        *flags,
    ]  # fmt: skip
    return _check(uv(*args), f"compile {in_file.name}").stdout


def _resolved(annotated: str) -> dict[str, tuple[str, set[str]]]:
    out: dict[str, tuple[str, set[str]]] = {}
    for line in annotated.splitlines():
        match = _LOCK_LINE.match(line)
        if not match:
            continue
        name = _normalize(match[1])
        via = _VIA.search(line)
        parents = {_normalize(p.strip()) for p in via[1].split(",")} if via else set()
        out[name] = (out[name][0], out[name][1] | parents) if name in out else (match[4], parents)
    return out


def lock_pins(text: str) -> dict[str, str]:
    """Return the ``normalised name -> version`` pins of lock text; markers, hashes and comments are not read."""
    return {_normalize(m[1]): m[4] for line in text.splitlines() if (m := _LOCK_LINE.match(line))}


def _torch_only(packages: dict[str, tuple[str, set[str]]]) -> list[str]:
    """Names left out of a lock body: torch, torchaudio, what only they pull in, and triton under torch."""
    family = {n for n in packages if n in TORCH_PACKAGES}
    grew = True
    while grew:
        grew = False
        for name, (_, parents) in packages.items():
            if name not in family and parents and parents <= family:
                family.add(name)
                grew = True
    triton = {
        n
        for n, (_, parents) in packages.items()
        if n.startswith(("triton", "pytorch-triton")) and parents & TORCH_PACKAGES
    }
    return sorted({n for n in family if _torch_family(n)} | triton)


def _top_cuda_tag(max_cuda: tuple[int, int] | None) -> str:
    ceiling = max_cuda or DEFAULT_MAX_CUDA
    return next((tag for tag, version in CUDA_INDEXES.items() if version <= ceiling), "cpu")


def _wheel_on_index(package: str, version: str, tag: str, python: str, platform_parts: tuple[str, ...]) -> bool:
    url = f"{PYTORCH_INDEX_BASE}/{tag}/{package}/"
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - fixed https URL
        wheels = re.findall(r">([^<>]+\.whl)<", response.read().decode())
    cp = "cp" + python.replace(".", "")
    prefixes = (f"{package}-{version}-", f"{package}-{version}%2B", f"{package}-{version}+")
    return any(w.startswith(prefixes) and f"-{cp}-" in w and all(p in w for p in platform_parts) for w in wheels)


def compile_lock(
    inputs: LockInputs, *, check_torch_index: bool = True, max_steps: int = 12, uv: Uv | None = None
) -> str:
    """Return a lock's full text: `uv pip compile --universal --generate-hashes` under a header.

    Needs network. When the resolution pins `torch`, its and `torchaudio`'s versions must be served by the top CUDA
    index and the CPU index on Linux x86_64 and by the CPU index on macOS arm64; otherwise `torch` is bounded below
    the rejected version and the resolution repeated, up to `max_steps` times. `torch`, `torchaudio`, the CUDA
    packages only they pull in and `inputs.excluded` are left out of the body; the builder installs the header's pins
    from the PyTorch index. uv runs with `--python-preference only-managed`.

    Args:
        inputs: What the lock is compiled from.
        check_torch_index: Whether to check the PyTorch indexes (needs network).
        max_steps: How many times `torch` may be bounded down.
        uv: The `uv` command; `Uv.find()` when None.

    Raises:
        EnvLockError: no torch within the requirements is served on every index.
        RuntimeError: `uv pip compile` failed; the message carries uv's stderr.
    """
    uv = uv or Uv.find()
    python, name = inputs.python, inputs.name
    tag = _top_cuda_tag(inputs.max_cuda)
    targets = [(tag, _LINUX_X86_64), ("cpu", _LINUX_X86_64), ("cpu", _MACOS_ARM64)]
    with tempfile.TemporaryDirectory(prefix=f"env-lock-{name}-") as tmp:
        work = Path(tmp)
        in_file, constraints = work / "requirements.in", work / "constraints.txt"
        in_file.write_text("\n".join([*inputs.requirements, *inputs.base]) + "\n")
        extra: list[str] = []
        notes: list[str] = []
        flags = ["--constraint", str(constraints), "--exclude-newer", inputs.exclude_newer]
        for _ in range(max_steps):
            constraints.write_text("\n".join([*inputs.quarantined, *extra]) + "\n")
            packages = _resolved(_pip_compile(uv, in_file, python, [*flags, "--annotation-style", "line"]))
            pins = tuple(f"{p}=={packages[p][0]}" for p in sorted(TORCH_PACKAGES) if p in packages)
            if not pins or not check_torch_index:
                break
            missing = [
                f"{pin} on {t}/{plat}"
                for pin in pins
                for t, plat in targets
                if not _wheel_on_index(pin.partition("==")[0], pin.partition("==")[2], t, python, plat)
            ]
            if not missing:
                break
            if "torch" not in packages:
                raise EnvLockError(f"{name!r}: torchaudio without torch is not served: {missing}")
            notes.append(f"torch {packages['torch'][0]} skipped: not served as " + "; ".join(missing))
            extra.append(f"torch<{packages['torch'][0]}")
        else:
            raise EnvLockError(f"{name!r}: no torch within the requirements is served on every index: {notes}")
        no_emit = [*(_torch_only(packages) if pins else ()), *sorted(_normalize(n) for n in inputs.excluded)]
        skip = [a for n in no_emit for a in ("--no-emit-package", n)]
        body = _pip_compile(uv, in_file, python, [*flags, "--generate-hashes", *skip])
    header = [
        f"# senselab env lock: {name}",
        f"# identity-digest: {LOCK_DIGEST_TAG} {lock_digest(inputs)}",
        f"# python: {python}",
        f"# torch: {' '.join(pins) if pins else 'none'}",
        f"# torch-index: {tag if pins else 'none'}",
        f"# exclude-newer: {inputs.exclude_newer}",
        f"# uv: {_check(uv('--version'), 'report its version').stdout.strip()}",
        *([f"# exclude: {' '.join(sorted(_normalize(n) for n in inputs.excluded))}"] if inputs.excluded else []),
        *(f"# note: {note}" for note in notes),
    ]
    return "\n".join(header) + "\n" + body


__all__ = [
    "CPU_INDEX",
    "CUDA_INDEXES",
    "DEFAULT_MAX_CUDA",
    "ENV_RECORD_FORMAT",
    "LOCK_DIGEST_TAG",
    "LOCK_INPUTS_FORMAT",
    "PYTHON_PREFERENCE",
    "CudaCompatibilityError",
    "EnvHandle",
    "EnvLock",
    "EnvLockError",
    "EnvRecord",
    "HostCuda",
    "HostLockLacksBackend",
    "LockInputs",
    "ManagedPythonMissing",
    "TorchIndex",
    "Uv",
    "compile_lock",
    "detect_host_cuda",
    "ensure_env",
    "host_lock_gaps",
    "installed_gaps",
    "lock_digest",
    "lock_identity_digest",
    "pick_torch_index",
    "read_lock_inputs",
]
