"""Committed, fully pinned locks for the subprocess venvs that :func:`ensure_venv` builds.

Every venv name has one lock under ``data/venv_locks/<name>.txt``, compiled by
``scripts/lock_subprocess_venvs.py`` from the backend's requirement list with
``uv pip compile --universal --generate-hashes``. A lock opens with a header::

    # senselab subprocess-venv lock: ppgs
    # requirements-sha256: <sha256 of the backend's requirements, Python version and CUDA cap>
    # python: 3.11
    # torch: torch==2.8.0 torchaudio==2.8.0
    # exclude-newer: 2026-10-02T00:00:00Z
    # uv: uv 0.11.3

followed by uv's pinned, hashed requirements. ``torch``, ``torchaudio`` and the CUDA-variant packages
only they pull in (``nvidia-*``, ``triton``) are left out of the pinned body: ``ensure_venv``
installs the exact ``torch:`` pins from the host's CUDA-matched PyTorch index, then the body with
``--no-deps``.

The design is in ``specs/20261002-subprocess-venv-locks/design.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

LOCK_DIR = Path(__file__).parent / "data" / "venv_locks"
"""Where the committed locks live."""

IPC_REQUIREMENTS = ("safetensors", "numpy")
"""Installed into every venv for :func:`call_in_venv`'s serialization."""

QUARANTINED = ("lightning!=2.6.2,!=2.6.3", "pytorch-lightning!=2.6.2,!=2.6.3")
"""Constraints applied to every compile: releases withdrawn from PyPI as malicious."""

TORCH_PACKAGES = frozenset({"torch", "torchaudio"})
"""Installed from the host's PyTorch index, never from the lock body."""

_CUDA_VARIANT_PREFIXES = ("nvidia-", "triton", "pytorch-triton", "cuda-")
_TRITON_PREFIXES = ("triton", "pytorch-triton")
"""Installed with ``torch`` from its index wherever ``torch`` depends on them, whatever else does."""

PYTORCH_INDEX_BASE = "https://download.pytorch.org/whl"

_HEADER_RE = re.compile(r"^#\s*([a-z0-9-]+):\s*(.*?)\s*$")
_LINE_RE = re.compile(r"^([A-Za-z0-9._-]+)(\[[^\]]*\])?\s*(==|@)\s*([^\s;#\\]+)")
_VIA_RE = re.compile(r"#\s*via\s+(.*)$")

LOCK_MISSING_HINT = "Run `uv run python scripts/lock_subprocess_venvs.py {name}` and commit the lock."


class VenvLockError(RuntimeError):
    """A venv has no lock, or its lock was compiled from different requirements."""


@dataclass(frozen=True)
class VenvLock:
    """A parsed lock file.

    Attributes:
        name: The venv name.
        path: The lock file.
        requirements_sha256: Digest of the inputs the lock was compiled from.
        python: The Python version the venv is built with.
        torch_pins: Exact ``torch`` / ``torchaudio`` pins, empty for a torch-free venv.
        sha256: Digest of the whole lock file; the venv's identity.
    """

    name: str
    path: Path
    requirements_sha256: str
    python: str
    torch_pins: tuple[str, ...]
    sha256: str
    header: dict[str, str] = field(default_factory=dict)


def normalize_name(name: str) -> str:
    """Return a distribution name normalized per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def requirements_sha256(
    requirements: list[str], python_version: str, max_cuda_version: Optional[tuple[int, int]] = None
) -> str:
    """Digest of everything a lock is compiled from.

    Args:
        requirements: The backend's requirement specs.
        python_version: The venv's ``major.minor`` Python version.
        max_cuda_version: The backend's CUDA index ceiling, if any.

    Returns:
        A hex SHA-256.
    """
    payload = {
        "requirements": sorted(requirements),
        "ipc": list(IPC_REQUIREMENTS),
        "python": python_version,
        "max_cuda": list(max_cuda_version) if max_cuda_version else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def lock_path(name: str, lock_dir: Path = LOCK_DIR) -> Path:
    """Return the lock file path for a venv name."""
    return lock_dir / f"{name}.txt"


def parse_lock(path: Path, name: Optional[str] = None) -> VenvLock:
    """Read a lock file's header.

    Args:
        path: The lock file.
        name: The venv name; defaults to the file stem.

    Returns:
        The parsed lock.

    Raises:
        VenvLockError: When a required header field is missing.
    """
    text = path.read_text()
    header: dict[str, str] = {}
    for line in text.splitlines():
        if not line.startswith("#"):
            break
        match = _HEADER_RE.match(line)
        if match:
            header[match.group(1)] = match.group(2)
    for key in ("requirements-sha256", "python", "torch"):
        if key not in header:
            raise VenvLockError(f"lock {path} has no '{key}:' header line")
    torch = header["torch"]
    pins = tuple(torch.split()) if torch and torch != "none" else ()
    return VenvLock(
        name=name or path.stem,
        path=path,
        requirements_sha256=header["requirements-sha256"],
        python=header["python"],
        torch_pins=pins,
        sha256=hashlib.sha256(text.encode()).hexdigest(),
        header=header,
    )


def load_lock(
    name: str,
    requirements: list[str],
    python_version: Optional[str] = None,
    max_cuda_version: Optional[tuple[int, int]] = None,
    lock_dir: Path = LOCK_DIR,
) -> VenvLock:
    """Load the committed lock for a venv and check it was compiled from these requirements.

    Args:
        name: The venv name.
        requirements: The backend's requirement specs.
        python_version: The requested Python version; None takes the lock's.
        max_cuda_version: The backend's CUDA index ceiling.
        lock_dir: Where locks live.

    Returns:
        The lock.

    Raises:
        VenvLockError: When the lock is missing or stale.
    """
    path = lock_path(name, lock_dir)
    if not path.is_file():
        raise VenvLockError(f"No lock for venv '{name}' at {path}. " + LOCK_MISSING_HINT.format(name=name))
    lock = parse_lock(path, name)
    python = python_version or lock.python
    if python != lock.python:
        raise VenvLockError(
            f"Lock for venv '{name}' is for Python {lock.python}, not {python}. " + LOCK_MISSING_HINT.format(name=name)
        )
    expected = requirements_sha256(requirements, python, max_cuda_version)
    if lock.requirements_sha256 != expected:
        raise VenvLockError(
            f"Lock for venv '{name}' was compiled from different requirements. " + LOCK_MISSING_HINT.format(name=name)
        )
    return lock


@dataclass
class Backend:
    """One venv name's inputs, as the code passes them to ``ensure_venv``.

    Attributes:
        name: The venv name.
        requirements: The requirement specs.
        python: ``major.minor``.
        max_cuda_version: The CUDA index ceiling, or None.
        modules: The modules whose calls name this venv.
    """

    name: str
    requirements: list[str]
    python: str
    max_cuda_version: Optional[tuple[int, int]]
    modules: list[str] = field(default_factory=list)


_NOT_BACKENDS = frozenset({"senselab.utils.subprocess_venv", "senselab.utils.compatibility_test_runner"})


def discover_backends(package_root: Optional[Path] = None) -> dict[str, Backend]:
    """Find every ``ensure_venv`` call in senselab and resolve its arguments.

    Each call's arguments are read from the calling module's attributes: a name, a call of a
    zero-argument function, or a literal.

    Args:
        package_root: The ``senselab`` package directory; defaults to the installed one.

    Returns:
        Backends by venv name.

    Raises:
        VenvLockError: When two calls name one venv with different inputs, or an argument cannot
            be resolved.
    """
    import ast
    import importlib

    import senselab

    root = package_root or Path(senselab.__file__).parent
    found: dict[str, Backend] = {}
    for path in sorted(root.rglob("*.py")):
        source = path.read_text()
        if "ensure_venv(" not in source:
            continue
        module_name = ".".join(("senselab", *path.relative_to(root).with_suffix("").parts))
        if module_name in _NOT_BACKENDS:
            continue
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call)
            and (getattr(node.func, "id", None) or getattr(node.func, "attr", None)) == "ensure_venv"
        ]
        if not calls:
            continue
        module = importlib.import_module(module_name)

        def value(node: Optional[ast.expr], module: object = module) -> object:
            if node is None:
                return None
            if isinstance(node, ast.Constant):
                return node.value
            if isinstance(node, ast.Name):
                return getattr(module, node.id)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.args:
                return getattr(module, node.func.id)()
            raise VenvLockError(f"{module_name}: cannot resolve ensure_venv argument {ast.unparse(node)}")

        for call in calls:
            keywords = {k.arg: k.value for k in call.keywords}
            name = value(call.args[0] if call.args else keywords.get("name"))
            requirements = value(call.args[1] if len(call.args) > 1 else keywords.get("requirements"))
            python = value(call.args[2] if len(call.args) > 2 else keywords.get("python_version"))
            cuda = value(keywords.get("max_cuda_version"))
            if not isinstance(name, str) or not isinstance(requirements, list):
                raise VenvLockError(f"{module_name}: ensure_venv needs a str name and a list of requirements")
            if not isinstance(python, str):
                raise VenvLockError(f"{module_name}: ensure_venv('{name}') must declare python_version")
            backend = Backend(
                name=name,
                requirements=list(requirements),
                python=python,
                max_cuda_version=tuple(cuda) if cuda else None,  # type: ignore[arg-type]
                modules=[module_name],
            )
            if name in found:
                prior = found[name]
                if (prior.requirements, prior.python, prior.max_cuda_version) != (
                    backend.requirements,
                    backend.python,
                    backend.max_cuda_version,
                ):
                    raise VenvLockError(f"venv '{name}' has different inputs in {prior.modules} and {module_name}")
                if module_name not in prior.modules:
                    prior.modules.append(module_name)
            else:
                found[name] = backend
    return found


def _uv() -> str:
    found = shutil.which("uv")
    if not found:
        raise VenvLockError("uv is not on PATH; it is needed to compile a lock")
    return found


def _compile(
    in_file: Path,
    python_version: str,
    constraints: Path,
    exclude_newer: Optional[str],
    *,
    hashes: bool,
    no_emit: tuple[str, ...] = (),
) -> str:
    cmd = [
        _uv(),
        "pip",
        "compile",
        str(in_file),
        "--universal",
        "--python-version",
        python_version,
        "--constraint",
        str(constraints),
        "--no-header",
        "--quiet",
        "--color",
        "never",
        "--no-sources",
    ]
    if exclude_newer:
        cmd += ["--exclude-newer", exclude_newer]
    if hashes:
        cmd.append("--generate-hashes")
    else:
        cmd += ["--annotation-style", "line"]
    for package in no_emit:
        cmd += ["--no-emit-package", package]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=in_file.parent)
    if result.returncode != 0:
        raise VenvLockError(f"uv pip compile failed for {in_file.name}:\n{result.stderr.strip()}")
    return result.stdout


def resolved_packages(line_annotated: str) -> dict[str, tuple[str, set[str]]]:
    """Parse ``--annotation-style line`` output into ``{name: (version_or_url, parents)}``.

    A name resolved more than once under different markers keeps its first entry and the union of
    parents.
    """
    out: dict[str, tuple[str, set[str]]] = {}
    for line in line_annotated.splitlines():
        match = _LINE_RE.match(line)
        if not match:
            continue
        name = normalize_name(match.group(1))
        via = _VIA_RE.search(line)
        parents = {normalize_name(p.strip()) for p in via.group(1).split(",")} if via else set()
        if name in out:
            out[name] = (out[name][0], out[name][1] | parents)
        else:
            out[name] = (match.group(4), parents)
    return out


def torch_only_cuda_packages(packages: dict[str, tuple[str, set[str]]]) -> list[str]:
    """Names to leave out of the lock body.

    These are ``torch``, ``torchaudio``, the CUDA-variant packages that only they pull in, and ``triton``
    wherever ``torch`` pulls it in.
    """
    family = {name for name in packages if name in TORCH_PACKAGES}
    changed = True
    while changed:
        changed = False
        for name, (_, parents) in packages.items():
            if name not in family and parents and parents <= family:
                family.add(name)
                changed = True
    triton = {
        name
        for name, (_, parents) in packages.items()
        if name.startswith(_TRITON_PREFIXES) and parents & TORCH_PACKAGES
    }
    return sorted(
        {name for name in family if name in TORCH_PACKAGES or name.startswith(_CUDA_VARIANT_PREFIXES)} | triton
    )


def top_cuda_tag(max_cuda_version: Optional[tuple[int, int]]) -> str:
    """The newest PyTorch CUDA index this venv can be routed to."""
    from senselab.utils.cuda_probe import _PYTORCH_INDEX_MAP, DEFAULT_MAX_CUDA

    ceiling = max_cuda_version if max_cuda_version is not None else DEFAULT_MAX_CUDA
    for tag, version in _PYTORCH_INDEX_MAP:
        if version <= ceiling:
            return tag
    return "cpu"


LINUX_X86_64 = ("linux", "x86_64")
MACOS_ARM64 = ("macosx", "arm64")

_INDEX_PAGES: dict[str, list[str]] = {}


def _index_wheels(tag: str, package: str) -> list[str]:
    url = f"{PYTORCH_INDEX_BASE}/{tag}/{package}/"
    if url not in _INDEX_PAGES:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 -- fixed https URL
            page = response.read().decode()
        _INDEX_PAGES[url] = re.findall(r">([^<>]+\.whl)<", page)
    return _INDEX_PAGES[url]


def wheel_on_index(package: str, version: str, tag: str, python_version: str, platform: tuple[str, ...]) -> bool:
    """Whether the PyTorch index ``tag`` serves ``package==version`` for this Python and platform.

    Args:
        package: ``torch`` or ``torchaudio``.
        version: The version, without a local tag.
        tag: ``cu128``, ``cpu``, ...
        python_version: ``major.minor``.
        platform: Substrings the wheel's platform tag must all contain, e.g. :data:`LINUX_X86_64`.

    Returns:
        True when a matching wheel is listed.
    """
    cp = "cp" + python_version.replace(".", "")
    for wheel in _index_wheels(tag, package):
        if not (
            wheel.startswith(f"{package}-{version}-")
            or wheel.startswith(f"{package}-{version}%2B")
            or wheel.startswith(f"{package}-{version}+")
        ):
            continue
        if f"-{cp}-" in wheel and all(part in wheel for part in platform):
            return True
    return False


def torch_targets(max_cuda_version: Optional[tuple[int, int]]) -> list[tuple[str, tuple[str, ...]]]:
    """The ``(index tag, platform)`` pairs a venv's torch pins must be served on."""
    return [(top_cuda_tag(max_cuda_version), LINUX_X86_64), ("cpu", LINUX_X86_64), ("cpu", MACOS_ARM64)]


def compile_lock(
    name: str,
    requirements: list[str],
    python_version: str,
    max_cuda_version: Optional[tuple[int, int]] = None,
    *,
    exclude_newer: Optional[str] = None,
    check_torch_index: bool = True,
    max_steps: int = 12,
) -> str:
    """Compile a lock's full text.

    When the resolution includes ``torch``, the chosen ``torch`` / ``torchaudio`` versions must be
    served by every index in :func:`torch_targets`; otherwise ``torch`` is bounded below the
    rejected version and the resolution repeated.

    Args:
        name: The venv name.
        requirements: The backend's requirement specs.
        python_version: ``major.minor``.
        max_cuda_version: The backend's CUDA index ceiling.
        exclude_newer: An RFC 3339 timestamp; distributions uploaded later are ignored.
        check_torch_index: Whether to check the PyTorch indexes (needs network).
        max_steps: How many times ``torch`` may be bounded down.

    Returns:
        The lock text, header included.

    Raises:
        VenvLockError: When no ``torch`` within the requirements is served on every target.
    """
    with tempfile.TemporaryDirectory(prefix=f"venv-lock-{name}-") as tmp:
        work = Path(tmp)
        in_file = work / "requirements.in"
        in_file.write_text("\n".join([*requirements, *IPC_REQUIREMENTS]) + "\n")
        extra: list[str] = []
        notes: list[str] = []
        for _ in range(max_steps):
            constraints = work / "constraints.txt"
            constraints.write_text("\n".join([*QUARANTINED, *extra]) + "\n")
            annotated = _compile(in_file, python_version, constraints, exclude_newer, hashes=False)
            packages = resolved_packages(annotated)
            pins = tuple(f"{p}=={packages[p][0]}" for p in sorted(TORCH_PACKAGES) if p in packages)
            if not pins or not check_torch_index:
                break
            missing = [
                f"{pin} on {tag}/{platform}"
                for pin in pins
                for tag, platform in torch_targets(max_cuda_version)
                if not wheel_on_index(pin.split("==")[0], pin.split("==")[1], tag, python_version, platform)
            ]
            if not missing:
                break
            torch_version = packages["torch"][0] if "torch" in packages else None
            if torch_version is None:
                raise VenvLockError(f"'{name}': torchaudio without torch is not served: {missing}")
            notes.append(f"torch {torch_version} skipped: not served as " + "; ".join(missing))
            extra.append(f"torch<{torch_version}")
        else:
            raise VenvLockError(f"'{name}': no torch within the requirements is served on every index: {notes}")
        no_emit = tuple(torch_only_cuda_packages(packages)) if pins else ()
        body = _compile(in_file, python_version, constraints, exclude_newer, hashes=True, no_emit=no_emit)
    uv_version = subprocess.run([_uv(), "--version"], capture_output=True, text=True).stdout.strip()
    lines = [
        f"# senselab subprocess-venv lock: {name}",
        f"# requirements-sha256: {requirements_sha256(requirements, python_version, max_cuda_version)}",
        f"# python: {python_version}",
        f"# torch: {' '.join(pins) if pins else 'none'}",
        f"# torch-index: {top_cuda_tag(max_cuda_version) if pins else 'none'}",
        f"# exclude-newer: {exclude_newer or 'none'}",
        f"# uv: {uv_version}",
        *(f"# note: {note}" for note in notes),
        f"# Regenerate: uv run python scripts/lock_subprocess_venvs.py {name}",
    ]
    return "\n".join(lines) + "\n" + body
