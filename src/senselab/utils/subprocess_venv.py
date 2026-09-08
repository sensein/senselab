"""Runtime subprocess venv manager for isolated backend dependencies.

Uses uv to create and manage isolated virtual environments for backends
that conflict with the core senselab installation. IPC uses a temp
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

from senselab.utils.cuda_probe import (
    HostCuda,
    SenselabCudaCompatibilityError,
    TorchIndex,
    detect_host_cuda,
    pick_torch_index,
)
from senselab.utils.file_lock import SharedFileLock

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


def ensure_venv(
    name: str,
    requirements: list[str],
    python_version: Optional[str] = None,
    max_cuda_version: Optional[tuple[int, int]] = None,
) -> Path:
    """Create or reuse an isolated virtual environment, retrying a lost-lock build a few times.

    Delegates to :func:`_ensure_venv_once` for one attempt. If that attempt's own build
    completes but then finds another process took over the lock in the meantime
    (:class:`_VenvLockLost`), the attempt is retried up to ``_MAX_LOCK_LOST_RETRIES`` times: a
    fresh attempt re-acquires the lock and re-checks the completion marker first, so it reuses
    whatever the other process finished instead of rebuilding, unless no marker is present yet.

    Args:
        name: Unique identifier for this venv (e.g., "coqui", "ppgs").
        requirements: List of pip install specs (e.g., ["coqui-tts~=0.27"]).
        python_version: Python version (e.g., "3.11"). Defaults to current.
        max_cuda_version: Optional ceiling on the CUDA wheel index for this venv, forwarded to
            ``pick_torch_index``. ``None`` applies no cap.

    Returns:
        Path to the venv directory.
    """
    last_error: Optional[_VenvLockLost] = None
    for attempt in range(1, _MAX_LOCK_LOST_RETRIES + 1):
        try:
            venv_dir = _ensure_venv_once(name, requirements, python_version, max_cuda_version)
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
    requirements: list[str],
    python_version: Optional[str] = None,
    max_cuda_version: Optional[tuple[int, int]] = None,
) -> Path:
    """Create or reuse an isolated virtual environment.

    Whether the CUDA-aware two-stage install fires is decided by the
    contents of ``requirements`` itself: any ``torch`` or ``torchaudio``
    spec triggers the probe + Stage-1 install via the chosen PyTorch
    wheel index. Backends that declare neither — including the genuinely
    torch-free case (e.g. a venv that only consumes a pure-Python GitHub
    repo) — skip the probe and the CUDA index entirely and do a single
    install pass against default PyPI. Backends that need ``torch`` /
    ``torchaudio`` (including via a transitive dep) MUST pin them in
    their own ``_REQUIREMENTS`` so the CUDA routing applies — otherwise
    Stage 2's transitive resolution against PyPI can split them across
    mismatched local-version tags.

    The venv directory name reflects the resolved install: ``name`` unchanged for a
    torch-free backend, ``f"{name}-{tag}"`` (e.g. ``"crisperwhisper-cu128"``) for a
    torch-bearing one, where ``tag`` is the resolved ``TorchIndex.tag``.

    Args:
        name: Unique identifier for this venv (e.g., "coqui", "ppgs").
        requirements: List of pip install specs (e.g., ["coqui-tts~=0.27"]).
        python_version: Python version (e.g., "3.11"). Defaults to current.
        max_cuda_version: Optional ceiling on the CUDA wheel index for this
            venv, forwarded to ``pick_torch_index``. Declare it when the venv
            pins a ``torch`` version that has no wheel on the newest index the
            host would otherwise select (e.g. brouhaha's ``torch<2.3`` needs
            ``(12, 1)`` → ``cu121``). ``None`` applies no cap.

    Returns:
        Path to the venv directory.
    """
    # Resolved before the directory is named -- see specs/20260907-venv-dir-keyed-by-index/
    # for why. A torch-free backend resolves no index and keeps the bare ``name``; a
    # torch-bearing backend's directory carries the resolved ``TorchIndex.tag``.
    torch_specs = _torch_install_specs(requirements)
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

    # SharedFileLock derives its own ".lock" / ".heartbeat" / ".holder" paths from venv_dir by
    # appending (never Path.with_suffix, which would collide two venv names differing only
    # after a dot -- see file_lock.py's class docstring). See
    # specs/20260907-shared-lock-heartbeat-inf/stampede-timeout-and-identity-file.md for the
    # timeout's derivation and for why `except TimeoutError` below retries unboundedly rather
    # than raising: a timeout only proves a live holder (SharedFileLock's contract -- a crashed
    # process's flock is kernel-released the instant it exits, so it cannot hold continuously
    # through a whole timeout window), and this mirrors ensure_hf_model's identical pattern in
    # dependencies.py.
    lock_timeout = _venv_lock_timeout()
    lock = SharedFileLock(venv_dir, timeout=lock_timeout)
    while True:
        try:
            lock.__enter__()
            break
        except TimeoutError:
            logger.info(
                "Still waiting for another process to build venv '%s' (lock held for the last %.0fs)",
                name,
                lock_timeout,
            )
            continue
    try:
        # Defensive check, not load-bearing: `dir_name` already carries the resolved
        # `torch_index.tag`, so two different indexes can no longer collide on one
        # `venv_dir`. A mismatch here means `dir_name` itself is wrong -- see the
        # `logger.error` below and specs/20260907-venv-dir-keyed-by-index/.
        expected_index_url = torch_index.url if torch_index is not None else None
        if marker.is_file():
            stored = json.loads(marker.read_text())
            stored_index_url = (stored.get("torch_index") or {}).get("url")
            if stored.get("requirements") == sorted(requirements) and stored_index_url == expected_index_url:
                logger.debug("Reusing existing venv: %s", venv_dir)
                return venv_dir
            if stored.get("requirements") == sorted(requirements) and stored_index_url != expected_index_url:
                logger.error(
                    "Venv '%s' at %s has a stale torch_index (%r != %r) despite a device-keyed "
                    "directory name -- this means the directory key ('%s') no longer reflects "
                    "the resolved index, not that a legitimate device change occurred.",
                    name,
                    venv_dir,
                    stored_index_url,
                    expected_index_url,
                    dir_name,
                )

        uv = _find_uv()
        py_ver = python_version or f"{sys.version_info.major}.{sys.version_info.minor}"
        index_label = torch_index.tag if torch_index is not None else "n/a (torch-free)"
        logger.info(
            "Creating isolated venv '%s' with Python %s (torch index: %s)",
            name,
            py_ver,
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
            # uv venv may have partially populated the directory; wipe it
            # so the next run starts from a clean baseline (mirrors the
            # install-failure cleanup below).
            shutil.rmtree(venv_dir, ignore_errors=True)
            raise

        if torch_index is not None:
            assert host_cuda is not None  # narrows the Optional for type-checkers
            # Two-stage install — works around uv's flag-precedence quirk.
            #
            # uv treats ``--extra-index-url`` as having higher priority
            # than ``--index-url`` (opposite of pip). So the obvious one-
            # shot form ``--index-url <cuda> --extra-index-url pypi`` lets
            # PyPI win for every package, including ``torch`` and
            # ``torchaudio``. On hosts where PyPI ships those two with
            # mismatched ``+cu`` local-version tags (currently
            # ``torch==X+cu129`` vs ``torchaudio==X`` with no tag), the
            # resulting venv hits the ABI mismatch this routing was meant
            # to prevent (``RuntimeError: PyTorch has CUDA version 12.9
            # whereas TorchAudio has CUDA version 12.8``).
            #
            # Stage 1: install caller-declared torch / torchaudio specs
            # with ONLY the chosen CUDA index named (no ``--extra-index-
            # url``), so the index is unambiguously primary and both
            # wheels — plus their ``nvidia-cuda-runtime-cu12`` transitives
            # — come from it with matched toolchains.
            #
            # Stage 2: install the remaining requirements (with torch +
            # torchaudio specs filtered out so uv can't re-resolve them
            # against PyPI) + the IPC serialization deps via default
            # PyPI. uv sees torch / torchaudio already installed and
            # satisfying any pin in ``requirements``, so it doesn't
            # re-resolve them. The PyTorch index governs only the two
            # packages it's designed for, and stale wheels on the CUDA
            # index for utilities like setuptools or pyarrow stay out
            # of the picture.
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
                # Wipe the half-built venv before raising so the next run starts clean.
                shutil.rmtree(venv_dir, ignore_errors=True)
                failing = _classify_uv_failure(exc.stderr or "")
                if failing is not None:
                    # Compat-error path: the wrapped exception's message already
                    # carries the diagnostic fields (host CUDA, attempted index,
                    # failing packages, recommended action). Logging the full uv
                    # stderr would just duplicate that.
                    logger.debug("Wheel not found installing torch in venv '%s': %s", name, exc.stderr)
                    raise SenselabCudaCompatibilityError(
                        host_cuda=host_cuda,
                        attempted_index=torch_index,
                        failing_packages=failing,
                    ) from exc
                # Pass-through path: log the stderr so the user can see what
                # really went wrong (network, permission, syntax, ...).
                logger.error("Failed to install torch in venv '%s': %s", name, exc.stderr)
                raise

            # Stage 2: backend requirements (minus any torch / torchaudio
            # specs — those are already installed and listing them here
            # without an index flag could let uv consider replacing the
            # matched wheels with PyPI's tagless versions) plus the IPC
            # serialization deps (safetensors + numpy). ``torchaudio``
            # was installed in Stage 1; both safetensors and numpy are
            # pure-Python and pull cleanly from PyPI everywhere.
            stage_two_reqs = [r for r in requirements if _spec_pkg_name(r) not in _TORCH_PKG_NAMES] + [
                "safetensors",
                "numpy",
            ]
            # Filtering the torch specs out of the install list is not enough on its own: it also
            # removes the only thing constraining torch during Stage 2's resolution. Measured on an
            # H100 -- unasdiff pins torch==2.6.0 and the cu124 index was correctly selected, yet the
            # finished venv held 2.13.0+cu130, because timm depends on torch with no upper bound and
            # uv was free to upgrade the CUDA-matched wheel to PyPI's newest. The comment above
            # assumed uv would leave an already-satisfying install alone; it does not.
            #
            # Pass the pins as *constraints* instead. They bound the resolution without becoming
            # install targets, so the Stage-1 wheels stay exactly as the PyTorch index built them
            # while a transitive dependent can no longer drag torch forward.
            torch_constraints = [r for r in requirements if _spec_pkg_name(r) in _TORCH_PKG_NAMES]
        else:
            # Torch-free path: single install pass straight from default
            # PyPI. No probe, no CUDA-index routing, no ``torchaudio``
            # forced into the install — backends that don't declare
            # torch / torchaudio don't pay for them.
            stage_two_reqs = [*requirements, "safetensors", "numpy"]
            torch_constraints = []

        constraint_file: Optional[Path] = None
        if torch_constraints:
            constraint_file = venv_dir / ".torch-constraints.txt"
            constraint_file.write_text("\n".join(torch_constraints) + "\n")

        try:
            subprocess.run(
                [
                    uv,
                    "pip",
                    "install",
                    "--python",
                    venv_python(venv_dir),
                    *(["--constraint", str(constraint_file)] if constraint_file else []),
                    *stage_two_reqs,
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
            "requirements": sorted(requirements),
            "python_version": py_ver,
        }
        if torch_index is not None:
            # Only stamp the index field on torch-using venvs; the marker
            # comparison treats its absence as "no torch routing in use",
            # so a future call against the same name whose ``requirements``
            # have grown a torch / torchaudio spec correctly invalidates
            # and rebuilds.
            marker_data["torch_index"] = {
                "tag": torch_index.tag,
                "url": torch_index.url,
                "source": torch_index.source,
            }
        # A takeover elsewhere overwrote this lock's identity without contacting this
        # process, so `venv_dir` may already have a second builder treating it as its own --
        # certifying completion here would be exactly the "importable-looking venv missing
        # a shared object" corruption from a concurrent build. Refuse and rebuild instead.
        if not lock.owns():
            logger.error(
                "Lock for venv '%s' was taken over by another process during this build; "
                "declining to mark %s complete and removing it for a clean rebuild.",
                name,
                venv_dir,
            )
            shutil.rmtree(venv_dir, ignore_errors=True)
            raise _VenvLockLost(f"Venv '{name}' lost its lock to a concurrent process during build; retry.")

        # Strictly before the marker write: a hard kill (OOM, CI timeout) between chmod and
        # the marker would otherwise leave `.senselab-installed` present with the chmod pass
        # incomplete. Every later ensure_venv call takes the reuse fast path on seeing the
        # marker and returns immediately -- but that path never calls _make_group_readable,
        # so a half-permissioned venv would never be repaired. Marker-after means an
        # interrupted chmod instead leaves no marker: the next call's marker check fails,
        # `shutil.rmtree` fires, and the rebuild reruns chmod to completion.
        _make_group_readable(venv_dir)
        marker.write_text(json.dumps(marker_data))
        logger.info("Venv '%s' ready at %s", name, venv_dir)
        return venv_dir
    finally:
        # __exit__ ignores its exc_info arguments (it never suppresses an exception), so
        # passing None here is equivalent to the (exc_type, exc, tb) a `with` block would
        # supply -- see ensure_hf_model's identical finally in dependencies.py.
        lock.__exit__(None, None, None)


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


# Packages whose install must be routed through the chosen CUDA-tagged
# PyTorch wheel index. We don't include downstream torch-ecosystem names
# like ``torchvision``/``torchtext`` because senselab's subprocess venvs
# don't currently use them; add here if that changes.
_TORCH_PKG_NAMES = frozenset({"torch", "torchaudio"})

# Capture the package name at the start of a uv pip install spec, stopping
# at the first character that isn't part of a PEP 508 distribution name —
# extras start with ``[``, versions with one of ``<>=!~``, URL/git refs
# with whitespace or ``@``. Matches ``torch``, ``torch>=2.8,<2.9``,
# ``torch[gpu]==2.8``, and ``nemo_toolkit[asr,tts] @ git+https://...``.
_PKG_NAME_RE = re.compile(r"^\s*([A-Za-z0-9._-]+)")


def _spec_pkg_name(spec: str) -> str:
    """Return the lowercased package name from a uv pip install spec.

    Returns ``""`` for specs that don't begin with a recognizable package
    name (e.g. a stray empty string). The match is loose by design — we
    only need it to identify torch + torchaudio entries inside
    ``requirements`` lists.
    """
    m = _PKG_NAME_RE.match(spec)
    return m.group(1).lower() if m else ""


def _torch_install_specs(requirements: list[str]) -> list[str]:
    """Return the ``torch`` / ``torchaudio`` specs explicitly named in ``requirements``.

    Forwards EVERY torch / torchaudio entry verbatim — so a backend
    pinning ``["torch>=2.8", "torch<2.9"]`` as two separate constraints
    gets both passed to uv, which combines them at resolve time. A single
    combined spec like ``"torch>=2.8,<2.9"`` still flows through
    unchanged.

    Returns an empty list when neither package is in ``requirements``,
    which ``ensure_venv`` treats as "skip Stage 1, skip the probe". The
    earlier version of this helper padded the return with bare ``torch``
    / ``torchaudio`` names so every venv routed both packages through
    the CUDA index regardless of declared needs — that meant
    ``torchaudio`` got force-installed in venvs (``yamnet``,
    ``continuous-ser``) that never imported it, adding ~200 MB of wheels
    for no reason. Backends that genuinely need ``torch`` or
    ``torchaudio`` — including via a transitive dep like ``qwen-asr`` —
    MUST pin them in their own ``_REQUIREMENTS`` so this helper picks
    them up and Stage 1 routes them through the matched CUDA index.
    Otherwise Stage 2's transitive resolution against PyPI can split
    them across mismatched local-version tags.
    """
    return [spec for spec in requirements if _spec_pkg_name(spec) in _TORCH_PKG_NAMES]


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
