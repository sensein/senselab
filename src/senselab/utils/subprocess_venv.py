"""Runtime subprocess venv manager for isolated backend dependencies.

Uses uv to create and manage isolated virtual environments for backends
that conflict with the core senselab installation. Each venv is installed
from its committed lock (:mod:`senselab.utils.venv_lock`). IPC uses a temp
directory with:
- manifest.json: call spec + JSON-serializable args + file metadata
- *.safetensors: tensor data (via safetensors, already a dep)
- *.wav: audio data (float, written through the range policy)
- *.npy: numpy arrays

File references include optional integrity metadata:
- checksum (SHA-256) for verifying data integrity
- readonly flag to prevent in-place modification
- a shared file lock (``SharedFileLock``) with a heartbeat that stale-detection
  actually reads: a holder that dies mid-install or mid-transfer is detected and
  taken over on the next uncontended acquire, rather than blocking every waiter
  for the full lock timeout; a holder that is still alive and legitimately slow
  (a multi-GB torch install on a congested mirror) is waited out in an unbounded
  retry loop instead of turning into a hard failure for every other process

Safety features are configurable via ``safe_mode`` to minimize
overhead for simple single-process workflows.
"""

import contextvars
import glob
import hashlib
import json
import logging
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

from senselab.utils import venv_lock
from senselab.utils.cuda_probe import (
    HostCuda,
    SenselabCudaCompatibilityError,
    TorchIndex,
    detect_host_cuda,
    pick_torch_index,
)
from senselab.utils.file_lock import SharedFileLock
from senselab.utils.venv_lock import VenvLock, load_lock

logger = logging.getLogger("senselab")

_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "senselab" / "venvs"


# ── File reference with integrity metadata ────────────────────────────


@dataclass
class FileRef:
    """A file reference with optional integrity and concurrency metadata.

    Use this to wrap file paths passed to ``call_in_venv`` when you need
    checksum verification, read-only enforcement, or file locking.

    For simple workflows, pass raw ``Path`` objects instead — no overhead.

    Args:
        path: Path to the file.
        readonly: If True, the subprocess receives a read-only copy or
            is instructed not to modify the file in-place. Default True.
        checksum: If True, compute SHA-256 before sending and verify
            after receiving. Catches corruption or unintended mutation.
        lock: If True, acquire a file lock (with heartbeat) for the
            duration of the subprocess call. Prevents parallel processes
            from mutating the file.
        lock_timeout: Max seconds to wait for the lock. Default 300.
    """

    path: Path
    readonly: bool = True
    checksum: bool = False
    lock: bool = False
    lock_timeout: int = 300
    _computed_hash: Optional[str] = field(default=None, repr=False)

    def compute_checksum(self) -> str:
        """Compute SHA-256 of the file."""
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        self._computed_hash = h.hexdigest()
        return self._computed_hash

    def verify_checksum(self) -> bool:
        """Verify the file matches the previously computed checksum."""
        if self._computed_hash is None:
            raise ValueError("No checksum computed yet — call compute_checksum() first")
        current = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                current.update(chunk)
        return current.hexdigest() == self._computed_hash

    def to_manifest(self) -> dict:
        """Serialize metadata for the IPC manifest."""
        entry: dict = {
            "type": "fileref",
            "path": str(self.path),
            "readonly": self.readonly,
        }
        if self.checksum and self._computed_hash:
            entry["checksum"] = self._computed_hash
        return entry


def _cache_dir_path() -> Path:
    """Return the cache directory path for cached subprocess venvs, without creating it.

    Side-effect-free so callers that only need to *check* a venv's location
    (e.g. a test's existence-based skip gate) don't risk failing at import time
    on a read-only/sandboxed HOME — creating the directory is ``_cache_dir()``'s job.
    """
    return Path(os.environ.get("SENSELAB_VENV_CACHE", str(_DEFAULT_CACHE_DIR)))


def provisioned_venv_dirs(name: str) -> list[Path]:
    """Every completed venv for one backend, across whatever device keys exist.

    A backend whose install depends on the device lives at ``<name>-<tag>``, so a caller asking
    whether it is provisioned cannot name the directory in advance. Only directories carrying the
    completion marker count; a half-built tree is not provisioned.

    Args:
        name: The backend's venv name, as passed to :func:`ensure_venv`.

    Returns:
        The matching directories, sorted, empty when the backend has never been built here.
    """
    cache = _cache_dir_path()
    if not cache.is_dir():
        return []
    candidates = [cache / name, *sorted(cache.glob(f"{name}-*"))]
    return [directory for directory in candidates if (directory / ".senselab-installed").is_file()]


_VENV_USE_RECORDER: "contextvars.ContextVar[Optional[dict[str, Path]]]" = contextvars.ContextVar(
    "_VENV_USE_RECORDER", default=None
)


@contextmanager
def record_venv_use() -> Iterator[dict[str, Path]]:
    """Record which subprocess venvs :func:`ensure_venv` resolves to within this context.

    Nesting is not supported: an inner call replaces the outer recorder for its duration.

    Yields:
        The dict, updated in place as :func:`ensure_venv` calls occur inside the block.
    """
    used: dict[str, Path] = {}
    token = _VENV_USE_RECORDER.set(used)
    try:
        yield used
    finally:
        _VENV_USE_RECORDER.reset(token)


def _note_venv_use(name: str, venv_dir: Path) -> None:
    """Record a resolved venv directory, when :func:`record_venv_use` is active."""
    recorder = _VENV_USE_RECORDER.get()
    if recorder is not None:
        recorder[name] = venv_dir


_DECLARED_ENV_PACKAGES = frozenset(
    {
        "torch",
        "torchaudio",
        "torchcodec",
        "transformers",
        "tensorflow",
        "tensorflow-hub",
        "keras",
        "numpy",
        "crisperwhisper",
        "qwen-asr",
        "clearvoice",
    }
)
"""The packages a venv's environment record names in full: the ones that decide its numerical
results, plus each backend's own library."""


def _normalize_package_name(name: str) -> str:
    """A dist-info package name, folded to compare across ``-``/``_`` spelling variants."""
    return name.lower().replace("_", "-")


def _venv_python_version(venv_dir: Path) -> str:
    """The interpreter version a venv was built with, from ``pyvenv.cfg``.

    Returns:
        The value of ``pyvenv.cfg``'s ``version_info`` (falling back to ``version``) key, or
        ``"unknown"`` when neither is present.
    """
    cfg = venv_dir / "pyvenv.cfg"
    try:
        text = cfg.read_text()
    except OSError:
        return "unknown"
    match = re.search(r"^version(?:_info)?\s*=\s*(\S+)", text, re.MULTILINE)
    return match.group(1) if match else "unknown"


def _venv_dist_info(venv_dir: Path) -> dict[str, str]:
    """Every installed distribution's name and version, read from ``*.dist-info`` directory names.

    Pure filesystem work: no interpreter start, no ``uv pip freeze``.

    Args:
        venv_dir: The venv's directory.

    Returns:
        Package name to version, for every ``*.dist-info`` directory found.
    """
    pattern = str(venv_dir / "lib" / "python*" / "site-packages" / "*.dist-info")
    if sys.platform == "win32":
        pattern = str(venv_dir / "Lib" / "site-packages" / "*.dist-info")
    out: dict[str, str] = {}
    for entry in glob.glob(pattern):
        base = os.path.basename(entry)[: -len(".dist-info")]
        name, _, version = base.rpartition("-")
        if name:
            out[name] = version
    return out


def venv_environment(name: str, venv_dir: Path) -> dict[str, Any]:
    """The environment record for one resolved subprocess venv.

    Args:
        name: The venv's backend name, as passed to :func:`ensure_venv`.
        venv_dir: Its resolved directory, from :func:`ensure_venv` or :func:`record_venv_use`.

    Returns:
        Keyword arguments for :meth:`~senselab.utils.prov_store.ProvStore.environment`: ``label``
        (the resolved directory's own name, which encodes its device key), ``python_version``,
        ``dependencies`` (the declared subset actually installed — see
        :data:`_DECLARED_ENV_PACKAGES`) and ``dependencies_digest`` (a SHA-256 over the full
        listing, so a mismatch against a fresh scan is detectable without storing every package).
    """
    full = _venv_dist_info(venv_dir)
    declared = {pkg: version for pkg, version in full.items() if _normalize_package_name(pkg) in _DECLARED_ENV_PACKAGES}
    digest = hashlib.sha256(json.dumps(sorted(full.items()), separators=(",", ":")).encode()).hexdigest()
    return {
        "label": venv_dir.name,
        "python_version": _venv_python_version(venv_dir),
        "dependencies": declared,
        "dependencies_digest": digest,
    }


def _cache_dir() -> Path:
    """Return the directory for cached subprocess venvs, creating it if missing."""
    cache = _cache_dir_path()
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _find_uv() -> str:
    """Find the uv binary, auto-installing if not present.

    Checks PATH and common install locations. If uv is not found,
    installs it automatically (needed for environments like Google Colab
    where uv is not pre-installed).
    """
    uv = shutil.which("uv")
    if uv:
        return uv
    for candidate in [
        Path.home() / ".local" / "bin" / "uv",
        Path.home() / ".cargo" / "bin" / "uv",
    ]:
        if candidate.is_file():
            return str(candidate)

    # Auto-install uv (e.g., on Google Colab or fresh environments)
    logger.info("uv not found — installing automatically...")
    result = subprocess.run(
        ["pip", "install", "uv"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode == 0:
        uv = shutil.which("uv")
        if uv:
            return uv
    raise FileNotFoundError("uv not found and auto-install failed. Install with: pip install uv")


# ── Venv management ──────────────────────────────────────────────────

# Overrides the build-lock timeout below; see
# specs/20260907-shared-lock-heartbeat-inf/stampede-timeout-and-identity-file.md for the
# derivation of the packaged default from measured cold-build times.
_VENV_LOCK_TIMEOUT_ENV = "SENSELAB_VENV_LOCK_TIMEOUT"
_DEFAULT_VENV_LOCK_TIMEOUT = 1200.0

# Bounded retries for the rare case where a completed build finds it no longer owns the lock
# (see `_VenvLockLost`) -- not a threshold fitted to data, just a small ceiling so a genuine
# takeover gets a few chances to reuse whoever won before giving up.
_MAX_LOCK_LOST_RETRIES = 3


def _venv_lock_timeout() -> float:
    """Return the configured venv-build lock timeout, in seconds."""
    return float(os.environ.get(_VENV_LOCK_TIMEOUT_ENV, str(_DEFAULT_VENV_LOCK_TIMEOUT)))


class _VenvLockLost(RuntimeError):
    """A just-completed build found it no longer owns the lock it built under.

    Raised by :func:`_ensure_venv_once` and retried a bounded number of times by
    :func:`ensure_venv`. See
    specs/20260907-shared-lock-heartbeat-inf/stampede-timeout-and-identity-file.md.
    """


def _resolve_lock(
    name: str,
    requirements: list[str],
    python_version: Optional[str],
    max_cuda_version: Optional[tuple[int, int]],
    compile_lock: bool,
) -> VenvLock:
    """The lock a venv is installed from: the committed one, or one compiled now for a probe."""
    if not compile_lock:
        return load_lock(name, requirements, python_version, max_cuda_version)
    python = python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
    text = venv_lock.compile_lock(name, requirements, python, max_cuda_version, check_torch_index=False)
    path = _cache_dir() / ".locks" / f"{name}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return venv_lock.parse_lock(path, name)


def ensure_venv(
    name: str,
    requirements: list[str],
    python_version: Optional[str] = None,
    max_cuda_version: Optional[tuple[int, int]] = None,
    *,
    compile_lock: bool = False,
) -> Path:
    """Create or reuse an isolated virtual environment from its pinned lock.

    The venv is installed from ``data/venv_locks/<name>.txt`` (see :mod:`senselab.utils.venv_lock`),
    which must have been compiled from exactly these ``requirements``, ``python_version`` and
    ``max_cuda_version``. Delegates to :func:`_ensure_venv_once` for one attempt; a build that finds
    another process took over the lock (:class:`_VenvLockLost`) is retried up to
    ``_MAX_LOCK_LOST_RETRIES`` times, each retry re-checking the completion marker first.

    Args:
        name: Unique identifier for this venv (e.g., "coqui", "ppgs"); also the lock's name.
        requirements: The requirement specs the lock was compiled from.
        python_version: Python version (e.g., "3.11"). None takes the lock's.
        max_cuda_version: Optional ceiling on the CUDA wheel index for this venv, forwarded to
            ``pick_torch_index``. ``None`` applies no cap.
        compile_lock: Compile a lock now instead of reading the committed one. For compatibility
            probes over version ranges only; such a venv is not reproducible.

    Returns:
        Path to the venv directory.

    Raises:
        VenvLockError: When the committed lock is missing or was compiled from other inputs.
    """
    lock = _resolve_lock(name, requirements, python_version, max_cuda_version, compile_lock)
    last_error: Optional[_VenvLockLost] = None
    for attempt in range(1, _MAX_LOCK_LOST_RETRIES + 1):
        try:
            venv_dir = _ensure_venv_once(name, lock, max_cuda_version)
            _note_venv_use(name, venv_dir)
            return venv_dir
        except _VenvLockLost as exc:
            last_error = exc
            logger.warning(
                "ensure_venv('%s'): lost the lock during build (attempt %d/%d); re-acquiring and "
                "checking for a completed venv before rebuilding: %s",
                name,
                attempt,
                _MAX_LOCK_LOST_RETRIES,
                exc,
            )
    assert last_error is not None  # the loop above always sets this before falling through
    raise last_error


def _ensure_venv_once(
    name: str,
    lock: VenvLock,
    max_cuda_version: Optional[tuple[int, int]] = None,
) -> Path:
    """Create or reuse an isolated virtual environment from ``lock``.

    A lock with ``torch:`` pins is installed in two stages: the exact pins from the host's
    CUDA-matched PyTorch index alone, then the lock body with ``--no-deps``. A torch-free lock is
    one ``--no-deps`` install of its body. The venv directory is ``name`` for a torch-free venv and
    ``f"{name}-{tag}"`` (e.g. ``"crisperwhisper-cu128"``) otherwise, ``tag`` being the resolved
    ``TorchIndex.tag``. The completion marker records the lock's digest, so a changed lock rebuilds.

    Args:
        name: The venv name.
        lock: The lock to install.
        max_cuda_version: Optional ceiling on the CUDA wheel index, forwarded to
            ``pick_torch_index``.

    Returns:
        Path to the venv directory.
    """
    torch_specs = list(lock.torch_pins)
    host_cuda: Optional[HostCuda] = None
    torch_index: Optional[TorchIndex] = None
    if torch_specs:
        env_override = os.getenv("SENSELAB_TORCH_INDEX_URL") or None
        probed = detect_host_cuda()
        host_cuda = probed
        torch_index = pick_torch_index(probed, env_override=env_override, max_cuda_version=max_cuda_version)

    dir_name = f"{name}-{torch_index.tag}" if torch_index is not None else name
    venv_dir = _cache_dir() / dir_name
    marker = venv_dir / ".senselab-installed"

    # A lock timeout only proves a live holder, so waiting retries unboundedly; see
    # specs/20260907-shared-lock-heartbeat-inf/stampede-timeout-and-identity-file.md.
    lock_timeout = _venv_lock_timeout()
    build_lock = SharedFileLock(venv_dir, timeout=lock_timeout)
    while True:
        try:
            build_lock.__enter__()
            break
        except TimeoutError:
            logger.info(
                "Still waiting for another process to build venv '%s' (lock held for the last %.0fs)",
                name,
                lock_timeout,
            )
            continue
    try:
        expected_index_url = torch_index.url if torch_index is not None else None
        if marker.is_file():
            stored = json.loads(marker.read_text())
            stored_index_url = (stored.get("torch_index") or {}).get("url")
            if stored.get("lock_sha256") == lock.sha256 and stored_index_url == expected_index_url:
                logger.debug("Reusing existing venv: %s", venv_dir)
                return venv_dir

        uv = _find_uv()
        py_ver = lock.python
        index_label = torch_index.tag if torch_index is not None else "n/a (torch-free)"
        logger.info(
            "Creating isolated venv '%s' with Python %s from %s (torch index: %s)",
            name,
            py_ver,
            lock.path.name,
            index_label,
        )

        if venv_dir.exists():
            shutil.rmtree(venv_dir)

        try:
            subprocess.run(
                [uv, "venv", "--python", py_ver, str(venv_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            logger.error("Failed to create venv '%s': %s", name, exc.stderr)
            shutil.rmtree(venv_dir, ignore_errors=True)
            raise

        if torch_index is not None:
            assert host_cuda is not None  # narrows the Optional for type-checkers
            # Stage 1 names only the CUDA index: uv ranks --extra-index-url above --index-url, so a
            # PyPI fallback here would let PyPI's differently-tagged torch win. See
            # specs/20260512-204619-fix-canary-cuda-conflict/.
            try:
                subprocess.run(
                    [
                        uv,
                        "pip",
                        "install",
                        "--index-url",
                        torch_index.url,
                        "--python",
                        venv_python(venv_dir),
                        *torch_specs,
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as exc:
                shutil.rmtree(venv_dir, ignore_errors=True)
                failing = _classify_uv_failure(exc.stderr or "")
                if failing is not None:
                    logger.debug("Wheel not found installing torch in venv '%s': %s", name, exc.stderr)
                    raise SenselabCudaCompatibilityError(
                        host_cuda=host_cuda,
                        attempted_index=torch_index,
                        failing_packages=failing,
                    ) from exc
                logger.error("Failed to install torch in venv '%s': %s", name, exc.stderr)
                raise

        try:
            subprocess.run(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    venv_python(venv_dir),
                    "--no-deps",
                    "--no-sources",
                    "--requirement",
                    str(lock.path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            shutil.rmtree(venv_dir, ignore_errors=True)
            logger.error("Failed to install in venv '%s': %s", name, exc.stderr)
            raise

        marker_data: dict[str, object] = {
            "lock": lock.path.name,
            "lock_sha256": lock.sha256,
            "python_version": py_ver,
        }
        if torch_index is not None:
            marker_data["torch_index"] = {
                "tag": torch_index.tag,
                "url": torch_index.url,
                "source": torch_index.source,
            }
        # A takeover elsewhere may already be mutating venv_dir; certifying it would mark a
        # half-built venv complete. Refuse and rebuild instead.
        if not build_lock.owns():
            logger.error(
                "Lock for venv '%s' was taken over by another process during this build; "
                "declining to mark %s complete and removing it for a clean rebuild.",
                name,
                venv_dir,
            )
            shutil.rmtree(venv_dir, ignore_errors=True)
            raise _VenvLockLost(f"Venv '{name}' lost its lock to a concurrent process during build; retry.")

        # Before the marker: a kill between the two must leave no marker, or the reuse fast path
        # would keep a half-permissioned venv forever.
        _make_group_readable(venv_dir)
        marker.write_text(json.dumps(marker_data))
        logger.info("Venv '%s' ready at %s", name, venv_dir)
        return venv_dir
    finally:
        build_lock.__exit__(None, None, None)


def _make_group_readable(venv_dir: Path) -> None:
    """Add group read (and execute, where already owner-executable) across a completed venv tree.

    Building the venv under a group-writable cache directory only lets a second user take over a
    stale build (``SharedFileLock``'s ``LOCK_DIR_MODE``) -- it says nothing about whether that user
    can *run* the interpreter the first user produced. ``uv venv`` and the subsequent installs create
    every file under the process umask, so unless the group happens to already have read (and, for
    executables, execute) access from some other setting, a second user can see the venv but not use
    it: a shared cache that is buildable but not usable, which is the gap this closes.

    Directories always gain group execute -- without it the directory can't be traversed by the
    group regardless of what's inside. Files gain group read always, and group execute only when
    already owner-executable: mirroring the owner's bit rather than escalating it, so a data file
    does not become runnable just because it lives in the same tree as ``bin/python``.

    A failed ``chmod`` is ignored: a shared cache directory can hold entries left by a different
    user's earlier, unrelated build, which this process cannot re-permission -- and raising here
    would abort the whole walk on that one leftover instead of still fixing every entry this
    process does own.

    Args:
        venv_dir: Root of a just-completed venv tree. Call this **before** writing the
            ``.senselab-installed`` marker (see ``ensure_venv``): the marker is what later
            calls trust to skip straight to the reuse fast path without re-running this
            pass, so if a kill lands mid-walk, the marker must not exist yet -- otherwise
            the next call would reuse a half-permissioned venv forever instead of rebuilding.
    """
    for root, _dirs, files in os.walk(venv_dir):
        root_path = Path(root)
        try:
            mode = root_path.stat().st_mode
            os.chmod(root_path, mode | stat.S_IRGRP | stat.S_IXGRP)
        except OSError:
            pass
        for name in files:
            file_path = root_path / name
            try:
                mode = file_path.stat().st_mode
            except OSError:
                continue
            new_mode = mode | stat.S_IRGRP
            if mode & stat.S_IXUSR:
                new_mode |= stat.S_IXGRP
            try:
                os.chmod(file_path, new_mode)
            except OSError:
                pass


# uv emits these phrases when it can't find a compatible wheel. The
# package spec is captured from the same phrase — anchoring prevents
# unrelated backticked hints (``uv cache clean``, ``--reinstall``, ...)
# from leaking into the user-facing error as "failing packages".
_FAILING_REQ_PATTERNS = [
    re.compile(
        r"no matching distribution(?: found)?(?: for)?\s+`([^`]+)`",
        re.IGNORECASE,
    ),
    re.compile(
        r"could not find a (?:version|distribution) (?:that satisfies)?(?:\s+the requirement)?\s+`([^`]+)`",
        re.IGNORECASE,
    ),
]


def _classify_uv_failure(stderr: str) -> Optional[list[str]]:
    """Return the failing package specs if stderr is a wheel-not-found error.

    Returns ``None`` for any other failure (network, permission, syntax) so
    the caller can re-raise the original ``CalledProcessError`` unchanged.
    """
    if not stderr:
        return None
    matches: list[str] = []
    for pattern in _FAILING_REQ_PATTERNS:
        matches.extend(pattern.findall(stderr))
    if not matches:
        return None
    # De-duplicate preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def venv_python(venv_dir: Path) -> str:
    """Return the path to the Python interpreter inside a venv.

    Uses ``Scripts/python.exe`` on Windows, ``bin/python`` elsewhere.
    """
    if sys.platform == "win32":
        return str(venv_dir / "Scripts" / "python.exe")
    return str(venv_dir / "bin" / "python")


def stage_portable_audio_io(directory: "str | Path") -> str:
    """Copy the portable audio I/O module into ``directory`` and return that directory.

    A worker runs in a venv where senselab is absent, so it cannot import the range policy --
    it gets the file handed to it instead. The worker adds the returned directory to
    ``sys.path`` and does ``from portable_audio_io import read_audio, write_audio``.

    Copying the file rather than inlining its source into the worker string keeps the worker
    readable and keeps one copy of the policy on disk: an inlined prelude would be a second
    rendering of the same module, and a reader of the worker could not tell which one ran.

    Args:
        directory: A directory the worker can read, normally the parent's ``TemporaryDirectory``.

    Returns:
        ``directory`` as a string, for the worker payload.
    """
    from senselab.utils import portable_audio_io

    source = Path(portable_audio_io.__file__)
    destination = Path(directory) / source.name
    shutil.copyfile(source, destination)
    return str(directory)


def _clean_subprocess_env() -> dict:
    """Return a copy of os.environ fit for a subprocess venv.

    Strips MPLBACKEND (matplotlib_inline's backend is not available in subprocesses) and points TLS
    verification at a CA bundle that exists.

    **Why the CA bundle needs saying explicitly.** These venvs run on the uv-managed interpreter, which
    is python-build-standalone: statically linked, and with no usable system CA path compiled in. So
    ``ssl.create_default_context()`` finds no trust store and every ``urlopen`` inside a worker fails
    with ``CERTIFICATE_VERIFY_FAILED`` — on a host whose network is fine and where ``curl`` to the same
    URL succeeds, because curl uses the system bundle and Python does not. Measured on MIT ORCD:

        coqui venv python 3.11.15
        urlopen as-is                        URLError CERTIFICATE_VERIFY_FAILED
        urlopen with SSL_CERT_FILE=certifi   OK

    The bundle certifi ships was already installed in that venv as a transitive dependency; nothing
    told Python to use it. Passing the *parent's* bundle is enough — a CA bundle is a file, not a
    per-interpreter object — which fixes all sixteen call sites from one place.

    An operator's existing ``SSL_CERT_FILE`` / ``REQUESTS_CA_BUNDLE`` is left alone: a host behind a
    corporate CA has already answered this question, and overriding it would break exactly the setup
    that took the trouble to configure it.
    """
    env = {k: v for k, v in os.environ.items() if k not in ("MPLBACKEND",)}
    if not env.get("SSL_CERT_FILE") or not env.get("REQUESTS_CA_BUNDLE"):
        try:
            import certifi

            bundle = certifi.where()
        except Exception:  # noqa: BLE001 — certifi absent is not a reason to fail the call
            return env
        env.setdefault("SSL_CERT_FILE", bundle)
        env.setdefault("REQUESTS_CA_BUNDLE", bundle)
    return env


# ── Subprocess result parsing with error propagation ──────────────────


def parse_subprocess_result(result: "subprocess.CompletedProcess[str]", venv_label: str = "subprocess") -> dict:
    """Parse a subprocess result, raising the original exception type if it failed.

    Worker scripts should print JSON to stdout. If the JSON contains an
    ``"error"`` key with ``"type"`` and ``"message"``, the original exception
    is reconstructed and raised.

    Args:
        result: The completed subprocess result.
        venv_label: Label for error messages (e.g., "Coqui", "SPARC").

    Returns:
        Parsed JSON dict from the last line of stdout.

    Raises:
        ValueError, RuntimeError, etc.: Reconstructed from worker error JSON.
        RuntimeError: If the subprocess failed without structured error output.
    """
    if result.returncode != 0:
        # Try to extract structured error from stdout
        stdout_lines = (result.stdout or "").strip().splitlines()
        if stdout_lines:
            try:
                output = json.loads(stdout_lines[-1])
                if "error" in output:
                    err = output["error"]
                    exc_type = err.get("type", "RuntimeError")
                    exc_msg = err.get("message", "Unknown error")
                    # Reconstruct common exception types
                    exc_class = {"ValueError": ValueError, "TypeError": TypeError}.get(exc_type, RuntimeError)
                    raise exc_class(exc_msg)
            except json.JSONDecodeError:
                pass
        raise RuntimeError(f"{venv_label} venv failed:\n{result.stderr}")

    stdout_lines = (result.stdout or "").strip().splitlines()
    if not stdout_lines:
        raise RuntimeError(f"{venv_label} venv produced no output")
    return json.loads(stdout_lines[-1])


# ── Container pack/unpack (host side) ─────────────────────────────────


def _pack_value(key: str, value: object, data_dir: Path) -> dict:
    """Pack a single value into the container, returning its manifest entry.

    Codec selection by type:
    - FileRef → path reference with integrity metadata
    - torch.Tensor → safetensors (fast, safe, HF standard)
    - numpy.ndarray → .npy (native numpy)
    - senselab Audio → .wav (float, via its own writer)
    - senselab Video / Path to video → path reference (no copy)
    - PIL.Image → .png (lossless)
    - bytes/bytearray → .bin (raw binary)
    - Pydantic BaseModel → .json (via model_dump_json)
    - everything else → JSON
    """
    import numpy as np
    import torch
    from safetensors.torch import save_file

    # FileRef → path reference with integrity metadata
    if isinstance(value, FileRef):
        if value.checksum:
            value.compute_checksum()
        return value.to_manifest()

    # torch.Tensor → safetensors
    if isinstance(value, torch.Tensor):
        path = data_dir / f"{key}.safetensors"
        save_file({"data": value.detach().cpu()}, str(path))
        return {"type": "tensor", "file": f"{key}.safetensors"}

    # numpy.ndarray → .npy
    if isinstance(value, np.ndarray):
        path = data_dir / f"{key}.npy"
        np.save(str(path), value)
        return {"type": "ndarray", "file": f"{key}.npy"}

    # senselab Audio (has waveform + sampling_rate) → WAV/FLOAT via its own writer.
    if hasattr(value, "waveform") and hasattr(value, "sampling_rate") and hasattr(value, "save_to_file"):
        path = data_dir / f"{key}.wav"
        value.save_to_file(str(path))
        return {"type": "audio", "file": f"{key}.wav", "sr": value.sampling_rate}

    # senselab Video or file path → pass path reference (no copy)
    if hasattr(value, "_file_path") and getattr(value, "_file_path", None) is not None:
        return {"type": "path", "value": str(value._file_path)}
    if isinstance(value, Path):
        return {"type": "path", "value": str(value)}

    # PIL Image → PNG (lossless)
    if type(value).__module__.startswith("PIL") or type(value).__name__ == "Image":
        path = data_dir / f"{key}.png"
        getattr(value, "save")(str(path), format="PNG")
        return {"type": "image", "file": f"{key}.png"}

    # bytes/bytearray → raw binary
    if isinstance(value, (bytes, bytearray)):
        path = data_dir / f"{key}.bin"
        path.write_bytes(value)
        return {"type": "binary", "file": f"{key}.bin"}

    # Pydantic BaseModel → JSON via model_dump
    if hasattr(value, "model_dump_json"):
        return {
            "type": "pydantic",
            "model_class": f"{type(value).__module__}.{type(value).__name__}",
            "value": json.loads(value.model_dump_json()),
        }  # type: ignore[union-attr]

    # JSON-serializable fallback
    return {"type": "json", "value": value}


def _unpack_value(entry: dict, data_dir: Path) -> object:
    """Unpack a single value from its manifest entry."""
    import numpy as np
    import torch
    from safetensors.torch import load_file

    btype = entry["type"]
    if btype == "tensor":
        return load_file(str(data_dir / entry["file"]))["data"]
    if btype == "ndarray":
        return np.load(str(data_dir / entry["file"]), allow_pickle=False)
    if btype == "audio":
        import torchaudio

        waveform, sr = torchaudio.load(str(data_dir / entry["file"]))
        return {"waveform": waveform, "sampling_rate": sr}
    if btype == "path":
        return Path(entry["value"])
    if btype == "fileref":
        ref_path = Path(entry["path"])
        if entry.get("checksum"):
            ref = FileRef(path=ref_path, checksum=True)
            ref._computed_hash = entry["checksum"]
            if not ref.verify_checksum():
                raise ValueError(f"Checksum mismatch for {ref_path} — file was modified during transfer")
        return ref_path
    if btype == "image":
        from PIL import Image

        return Image.open(str(data_dir / entry["file"]))
    if btype == "binary":
        return (data_dir / entry["file"]).read_bytes()
    if btype == "pydantic":
        # Caller is responsible for reconstructing the model
        return entry.get("value")
    return entry.get("value")


# ── Subprocess shim (embedded, runs in the isolated venv) ─────────────

_SHIM = r"""
import json, sys
from pathlib import Path
import numpy as np

container = Path(sys.stdin.read().strip())
manifest = json.loads((container / "manifest.json").read_text())
data_dir = container / "data"

# senselab is not installed in this venv; the parent stages the audio I/O policy alongside the
# payload so a result audio is written under the same range policy the host applies.
sys.path.insert(0, str(container))
from portable_audio_io import write_audio

try:
    from safetensors.torch import load_file as _st_load, save_file as _st_save
except ImportError:
    _st_load = _st_save = None

# ── Unpack args ──
args = {}
for key, entry in manifest.get("entries", {}).items():
    t = entry["type"]
    if t == "tensor" and _st_load:
        args[key] = _st_load(str(data_dir / entry["file"]))["data"]
    elif t == "ndarray":
        args[key] = np.load(str(data_dir / entry["file"]), allow_pickle=False)
    elif t == "audio":
        import torchaudio
        wf, sr = torchaudio.load(str(data_dir / entry["file"]))
        args[key] = {"waveform": wf, "sampling_rate": sr}
    elif t == "path":
        args[key] = Path(entry["value"])
    elif t == "fileref":
        args[key] = Path(entry["path"])
    elif t == "image":
        from PIL import Image
        args[key] = Image.open(str(data_dir / entry["file"]))
    elif t == "binary":
        args[key] = (data_dir / entry["file"]).read_bytes()
    elif t == "pydantic":
        args[key] = entry.get("value")  # passed as dict; callee reconstructs if needed
    else:
        args[key] = entry.get("value")

# ── Call function ──
call = manifest["call"]
mod = __import__(call["module"], fromlist=[call["function"]])
result = getattr(mod, call["function"])(**args)

# ── Pack result ──
ret = container / "return"
ret.mkdir(exist_ok=True)
rd = ret / "data"
rd.mkdir(exist_ok=True)
ret_entries = {}

def pack(name, obj):
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            if _st_save:
                _st_save({"data": obj.detach().cpu()}, str(rd / f"{name}.safetensors"))
                return {"type": "tensor", "file": f"{name}.safetensors"}
    except ImportError:
        pass
    if isinstance(obj, np.ndarray):
        np.save(str(rd / f"{name}.npy"), obj)
        return {"type": "ndarray", "file": f"{name}.npy"}
    if hasattr(obj, "waveform") and hasattr(obj, "sampling_rate"):
        samples = obj.waveform
        samples = samples.detach().cpu().numpy() if hasattr(samples, "detach") else np.asarray(samples)
        write_audio(str(rd / f"{name}.wav"), samples, int(obj.sampling_rate))
        return {"type": "audio", "file": f"{name}.wav"}
    if isinstance(obj, Path):
        return {"type": "path", "value": str(obj)}
    if isinstance(obj, (bytes, bytearray)):
        (rd / f"{name}.bin").write_bytes(obj)
        return {"type": "binary", "file": f"{name}.bin"}
    if hasattr(obj, "save") and hasattr(obj, "mode"):  # PIL Image
        (rd / f"{name}.png").parent.mkdir(exist_ok=True)
        obj.save(str(rd / f"{name}.png"), format="PNG")
        return {"type": "image", "file": f"{name}.png"}
    return {"type": "json", "value": obj}

if isinstance(result, dict):
    for k, v in result.items():
        ret_entries[k] = pack(k, v)
elif isinstance(result, (list, tuple)):
    for i, v in enumerate(result):
        ret_entries[f"__item_{i}__"] = pack(f"item_{i}", v)
    ret_entries["__is_sequence__"] = {"type": "json", "value": True}
    ret_entries["__sequence_len__"] = {"type": "json", "value": len(result)}
else:
    ret_entries["__result__"] = pack("result", result)

(ret / "manifest.json").write_text(json.dumps({"entries": ret_entries}, default=str))
print("OK")
"""


# ── Public API ────────────────────────────────────────────────────────


def call_in_venv(
    name: str,
    requirements: list[str],
    module: str,
    function: str,
    args: Optional[dict[str, object]] = None,
    python_version: Optional[str] = None,
    timeout: int = 600,
    safe_mode: bool = False,
) -> object:
    """Call a function in an isolated venv using container-based IPC.

    Data is serialized using efficient codecs:
    - torch.Tensor → safetensors (fast, safe, HF standard)
    - numpy.ndarray → .npy
    - senselab Audio → .wav (FLOAT, exactly preserving values beyond ±1)
    - FileRef → path with checksum/lock metadata
    - PIL Image → .png, bytes → .bin, Pydantic → JSON
    - everything else → JSON

    Args:
        name: Venv identifier.
        requirements: Pip install specs.
        module: Python module path (e.g., "TTS.api").
        function: Function name.
        args: Keyword arguments. Tensors, arrays, Audio objects, and
            FileRef objects are handled automatically. Use FileRef to
            wrap paths that need checksum or lock protection.
        python_version: Python version for the venv.
        timeout: Max execution time in seconds.
        safe_mode: If True, automatically wrap all Path args as FileRef
            with checksum=True and readonly=True. Default False for
            minimal overhead in simple workflows.

    Returns:
        The function's return value with blobs loaded back to native types.
    """
    venv_dir = ensure_venv(name, requirements, python_version)
    python = venv_python(venv_dir)

    # In safe_mode, auto-wrap Path args as FileRef with checksum + readonly
    effective_args = dict(args or {})
    if safe_mode:
        for key, value in effective_args.items():
            if isinstance(value, Path) and value.is_file():
                effective_args[key] = FileRef(path=value, readonly=True, checksum=True)

    # Collect FileRef locks to hold during execution. SharedFileLock derives its own
    # ".lock" / ".heartbeat" paths from value.path by appending, never Path.with_suffix
    # (see file_lock.py) -- a FileRef path can legitimately contain a dot (e.g. a
    # revisioned filename), and with_suffix would silently collide two such paths onto
    # one lock file.
    file_locks: list[SharedFileLock] = []
    for value in effective_args.values():
        if isinstance(value, FileRef) and value.lock:
            # manage_dir_mode=False: value.path is caller-supplied, so the directory we are about
            # to drop a .lock into is the *invoking user's own* — a stranger's would be left alone
            # anyway, since chmod(2) returns EPERM unless the effective UID owns it and
            # `_ensure_dir` swallows that. The user's own directory is the one that actually
            # changes, and the change widens it: measured, a 0o700 input directory comes back
            # 0o2775 — setgid + group-write *and* other-read + other-execute, i.e. world traversal
            # of a directory its owner made private, as a side effect of taking a lock. So the mode
            # management senselab's own cache dirs want is exactly wrong here.
            fl = SharedFileLock(value.path, timeout=value.lock_timeout, manage_dir_mode=False)
            fl.__enter__()
            file_locks.append(fl)

    try:
        with tempfile.TemporaryDirectory(prefix="senselab-ipc-") as tmpdir:
            container = Path(tmpdir)
            data_dir = container / "data"
            data_dir.mkdir()
            # The worker writes audio results, so it needs the range policy staged next to them.
            stage_portable_audio_io(container)

            # Pack args
            entries: dict[str, object] = {}
            for key, value in effective_args.items():
                entries[key] = _pack_value(key, value, data_dir)

            manifest = {
                "call": {"module": module, "function": function},
                "entries": entries,
            }
            (container / "manifest.json").write_text(json.dumps(manifest, default=str))

            # Execute in subprocess
            try:
                result = subprocess.run(
                    [python, "-c", _SHIM],
                    input=str(container),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=_clean_subprocess_env(),
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Venv '{name}' timed out after {timeout}s") from exc

            if result.returncode != 0:
                raise RuntimeError(f"Venv '{name}' failed:\n{result.stderr}")

            # Verify checksums on FileRef args after subprocess completes
            for value in effective_args.values():
                if isinstance(value, FileRef) and value.checksum and value.readonly:
                    if not value.verify_checksum():
                        raise ValueError(
                            f"File {value.path} was modified during subprocess execution (readonly=True was specified)"
                        )

            # Unpack result
            ret_dir = container / "return"
            if not ret_dir.exists():
                return None

            ret_manifest = json.loads((ret_dir / "manifest.json").read_text())
            ret_data = ret_dir / "data"

            unpacked: dict[str, object] = {}
            for key, entry in ret_manifest.get("entries", {}).items():
                unpacked[key] = _unpack_value(entry, ret_data)

            # Unwrap single result
            if len(unpacked) == 1 and "__result__" in unpacked:
                return unpacked["__result__"]

            # Reconstruct sequences
            if unpacked.get("__is_sequence__"):
                seq_len = int(str(unpacked.get("__sequence_len__", 0)))
                return [unpacked.get(f"__item_{i}__") for i in range(seq_len)]

            # Filter out internal keys
            return {k: v for k, v in unpacked.items() if not k.startswith("__")}
    finally:
        # Release all file locks
        for fl in file_locks:
            fl.__exit__(None, None, None)
