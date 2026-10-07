"""Stage 3 of a reproduce env: sdists built into wheels, recorded in ``wheels.lock`` and reused.

Each :class:`SdistSpec` is built in its own build venv, made from its hashed ``build_env`` lock (and
the env's torch lock when given), with ``uv build --wheel --no-build-isolation``, an allowlisted
environment (:func:`build_environment`), a fixed ``SOURCE_DATE_EPOCH`` and umask, inside the step
sandbox's network namespace (:func:`wheel_build_argv`), so no index or host is reachable. Wheels
are stored content-addressed, ``<wheelhouse>/<wheel sha256>/<wheel file>``, and never overwritten.
The first build records the wheel in ``wheels.lock``; later calls reuse it after checking its
sha256, and only ``rebuild`` builds again, adding a new wheel beside the old one.

``wheels.lock`` has one whitespace-separated line per sdist under a column header::

    # name version sdist_url sdist_sha256 build_env_sha256 patch_sha256s build_config_sha256 wheel_path wheel_sha256

``patch_sha256s`` is a comma-joined list in manifest order, or ``-``. ``build_config_sha256`` is
:func:`build_config_sha256`. ``wheel_path`` is relative to the wheelhouse. Reads and writes hold a
``flock`` on the sidecar ``<wheels.lock>.lock``; writes replace the file atomically. Paths in a spec
are relative to the manifest repository, passed as ``base_dir``.
"""

from __future__ import annotations

import contextlib
import datetime
import fcntl
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import ssl
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from senselab.utils.reproduce import sandbox
from senselab.utils.reproduce.lockgen import find_uv
from senselab.utils.reproduce.locks import ReproduceLockError, file_sha256, lock_entries, normalize_name, verify_files
from senselab.utils.reproduce.spec import FileRef, LockRef, Profile, SdistSpec, TorchSpec

SOURCE_DATE_EPOCH = "315532800"
BUILD_UMASK = 0o007
SYSTEM_PATH = "/usr/bin:/bin"
MODULES_INIT = "/etc/profile.d/modules.sh"
DEAD_PROXY = "http://127.0.0.1:9"
FETCH_TIMEOUT = 60
WHEELS_HEADER = (
    "# name version sdist_url sdist_sha256 build_env_sha256 patch_sha256s build_config_sha256 wheel_path wheel_sha256"
)
CUDA_HOME_FROM_NVCC = 'export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"'
BATCH_EXPORTS = ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "SENSELAB_REPRO_CACHE")
OFFLINE_ENV = {
    "UV_OFFLINE": "1",
    "PIP_NO_INDEX": "1",
    "http_proxy": DEAD_PROXY,
    "https_proxy": DEAD_PROXY,
    "HTTP_PROXY": DEAD_PROXY,
    "HTTPS_PROXY": DEAD_PROXY,
}
_TAIL = 4000
_SAFE_WORD = re.compile(r"^[A-Za-z0-9_.+/@:=-]+$")
_DOUBLE_QUOTE_UNSAFE = re.compile(r'["$`\\\n\r]')


class WheelLine(NamedTuple):
    """One ``wheels.lock`` line: the build inputs of an sdist and the wheel they produced."""

    name: str
    version: str
    sdist_url: str
    sdist_sha256: str
    build_env_sha256: str
    patch_sha256s: tuple[str, ...]
    build_config_sha256: str
    wheel_path: str
    wheel_sha256: str


class WheelResult(NamedTuple):
    """What :func:`ensure_wheel` returns.

    ``wheels_lock_sha256`` is the sha256 of ``wheels.lock`` afterwards; when ``built`` it changed, and
    the caller writes it into the manifest's ``wheels.lock_sha256``.
    """

    wheel: Path
    wheel_sha256: str
    wheels_lock_sha256: str
    built: bool


def _run(argv: list[str], env: dict[str, str] | None = None) -> None:
    """Run one command under :data:`BUILD_UMASK`; raise with the tail of its output if it fails."""
    result = subprocess.run(argv, env=env, capture_output=True, text=True, check=False, umask=BUILD_UMASK)
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()[-_TAIL:]
        raise ReproduceLockError(f"{shlex.join(argv[:3])} failed ({result.returncode}): {output}")


def _capture(argv: list[str], env: dict[str, str] | None = None) -> str:
    """Run a version query; return its output, or ``unavailable`` if it cannot run or fails."""
    try:
        result = subprocess.run(argv, env=env, capture_output=True, text=True, check=False, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _sdist_filename(url: str) -> str:
    name = urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1])
    if not name or name in (".", "..") or "/" in name:
        raise ReproduceLockError(f"sdist URL has no file name: {url}")
    return name


def ca_bundle() -> str | None:
    """The CA bundle for TLS: ``SSL_CERT_FILE`` when set, else certifi's, else none."""
    if os.environ.get("SSL_CERT_FILE"):
        return os.environ["SSL_CERT_FILE"]
    try:
        import certifi
    except ImportError:
        return None
    return certifi.where()


def fetch_sdist(url: str, sha256: str, dest: Path) -> Path:
    """Download an sdist into ``dest`` and check its sha256; reuse a present copy that matches.

    Args:
        url: The sdist URL (``https://`` or ``file://``).
        sha256: The manifest sha256.
        dest: The download directory; created if absent.

    Returns:
        ``dest / <file name from the URL>``.

    Raises:
        ReproduceLockError: If the downloaded bytes do not match ``sha256``; nothing is left behind.
    """
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / _sdist_filename(url)
    if path.is_file() and file_sha256(path) == sha256:
        return path
    digest = hashlib.sha256()
    https = urllib.parse.urlsplit(url).scheme == "https"
    context = ssl.create_default_context(cafile=ca_bundle()) if https else None
    fd, part = tempfile.mkstemp(dir=dest, prefix=f".{path.name}.", suffix=".part")
    try:
        with (
            os.fdopen(fd, "wb") as out,
            urllib.request.urlopen(url, context=context, timeout=FETCH_TIMEOUT) as response,  # noqa: S310
        ):
            while chunk := response.read(1 << 20):
                digest.update(chunk)
                out.write(chunk)
        if digest.hexdigest() != sha256:
            raise ReproduceLockError(f"sdist {url}: sha256 {digest.hexdigest()} != manifest {sha256}")
        os.replace(part, path)
    finally:
        if os.path.exists(part):
            os.unlink(part)
    return path


def _copy_verified(refs: Iterable[FileRef], base_dir: Path, inputs: Path) -> None:
    """Copy hashed files from ``base_dir`` into ``inputs`` and check the copies' sha256s."""
    refs = list(refs)
    for ref in refs:
        source = base_dir / ref.file
        if not source.resolve().is_relative_to(base_dir.resolve()):
            raise ReproduceLockError(f"{ref.file} resolves outside {base_dir}")
        target = inputs / ref.file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    verify_files(refs, inputs)


def _file_mode(archive_mode: int) -> int:
    """A regular file's mode: its archive mode, owner read/write added, under :data:`BUILD_UMASK`."""
    return ((archive_mode & 0o777) | 0o600) & ~BUILD_UMASK


def _tar_member(member: tarfile.TarInfo, dest: str) -> tarfile.TarInfo | None:
    """The ``data`` extraction filter, with a regular file's mode from :func:`_file_mode`."""
    filtered = tarfile.data_filter(member, dest)
    if filtered is None or not member.isreg():
        return filtered
    return filtered.replace(mode=_file_mode(member.mode), deep=False)


def _unpack(data: bytes, filename: str, dest: Path) -> Path:
    """Unpack sdist bytes into ``dest`` and return the single top-level directory.

    File modes end up inside the wheel, so a regular file's mode is :func:`_file_mode` of its
    archive mode, whatever the caller's umask.

    Args:
        data: The verified sdist bytes.
        filename: The sdist file name; ``.zip`` selects zip, anything else tar.
        dest: The unpack directory; must not exist.
    """
    dest.mkdir(parents=True)
    root = dest.resolve()
    if filename.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            for member in members:
                if not (dest / member.filename).resolve().is_relative_to(root):
                    raise ReproduceLockError(f"{filename}: member {member.filename!r} leaves the unpack directory")
            archive.extractall(dest)
            for member in members:
                if not member.is_dir():
                    (dest / member.filename).chmod(_file_mode((member.external_attr >> 16) or 0o644))
    else:
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            archive.extractall(dest, filter=_tar_member)
    tops = list(dest.iterdir())
    if len(tops) != 1 or not tops[0].is_dir():
        raise ReproduceLockError(f"{filename} does not unpack to one top-level directory")
    return tops[0]


def _apply_patch(patch: Path, src: Path, env: dict[str, str]) -> None:
    """Apply a ``-p1`` unified diff to the unpacked sdist root, with no fuzz."""
    argv = ["patch", "-p1", "--batch", "--forward", "--fuzz=0", "--no-backup-if-mismatch", "-d", str(src)]
    argv += ["-i", str(patch)]
    result = subprocess.run(argv, env=env, capture_output=True, text=True, check=False, umask=BUILD_UMASK)
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()[-_TAIL:]
        raise ReproduceLockError(f"patch {patch.name} does not apply to {src.name}: {output}")


def _install_argv(uv: str, python: Path, lock: Path, index_url: str | None, binary_only: bool) -> list[str]:
    argv = [uv, "pip", "install", "--no-config", "--python", str(python), "--no-deps", "--require-hashes"]
    if binary_only:
        argv += ["--only-binary", ":all:"]
    if index_url is not None:
        argv += ["--index-url", index_url]
    return [*argv, "-r", str(lock)]


def torch_index_url(torch: TorchSpec, text: str) -> str | None:
    """The index a torch lock installs from: its ``index_url`` if any entry is an ``==`` pin, else none.

    Args:
        torch: The env's torch spec.
        text: The verified torch lock contents.

    Returns:
        ``torch.index_url`` for an index-compiled lock; ``None`` for a URL-pinned (uv-export) lock.
    """
    return torch.index_url if any(not e.is_url for e in lock_entries(text)) else None


def needs_nvcc(sdist: SdistSpec) -> bool:
    """Whether the build needs ``nvcc``: an arch list is declared or a ``cuda/*`` module is loaded."""
    return sdist.torch_cuda_arch_list is not None or any(m.startswith("cuda/") for m in sdist.build_modules)


def in_modules(sdist: SdistSpec, argv: list[str]) -> list[str]:
    """Wrap ``argv`` so it runs with the sdist's modules loaded, without reading any dotfile.

    Without ``build_modules`` ``argv`` is returned unchanged. With them it becomes ``bash --noprofile
    --norc -c '. <MODULES_INIT> && module load <m…> && exec "$@"'``; when :func:`needs_nvcc`, the
    script also fails unless ``nvcc`` is on ``PATH`` and exports ``CUDA_HOME`` from it.

    Args:
        sdist: The sdist spec.
        argv: The command.

    Returns:
        The argv to run.
    """
    if not sdist.build_modules:
        return argv
    loaded = " ".join(_safe_word(m, "build module") for m in sdist.build_modules)
    script = f". {MODULES_INIT} && module load {loaded} && "
    if needs_nvcc(sdist):
        script += (
            f'if ! command -v nvcc >/dev/null; then echo "nvcc not on PATH after module load {loaded}" >&2; exit 1; fi'
            f" && {CUDA_HOME_FROM_NVCC} && "
        )
    script += 'exec "$@"'
    return ["bash", "--noprofile", "--norc", "-c", script, "bash", *argv]


def wheel_build_argv(
    uv: str, build_python: Path, src: Path, out_dir: Path, *, venv: Path, writable: Iterable[Path]
) -> list[str]:
    """The exact argv that builds the wheel: ``uv build --wheel --no-build-isolation`` inside the sandbox.

    The build runs in :func:`sandbox.sandbox_argv`'s namespaces: no network, the build venv mounted
    read-only, every capability dropped. Modules are not loaded inside; their variables reach the
    build through :func:`build_environment`.

    Args:
        uv: The uv executable.
        build_python: The build venv's interpreter.
        src: The unpacked, patched sdist.
        out_dir: Where uv writes the wheel.
        venv: The build venv, mounted read-only.
        writable: Directories the build writes (the source, the output, ``HOME``, ``TMPDIR``).

    Returns:
        The argv.
    """
    argv = [uv, "build", "--no-config", "--wheel", "--no-build-isolation", "--python", str(build_python)]
    return sandbox.sandbox_argv([*argv, "--out-dir", str(out_dir), str(src)], [], writable, [venv])


_UNRECORDED_VARS = frozenset({"_", "SHLVL", "PWD", "OLDPWD"})


def _capture_env(argv: list[str], env: dict[str, str]) -> str:
    """Run a command that prints ``env -0``; raise with the tail of its output if it fails."""
    result = subprocess.run(argv, env=env, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()[-_TAIL:]
        raise ReproduceLockError(f"loading build modules failed ({result.returncode}): {output}")
    return result.stdout


def module_environment(sdist: SdistSpec, base: Mapping[str, str]) -> dict[str, str]:
    """The variables the sdist's ``build_modules`` set, captured outside the sandbox.

    Runs :func:`in_modules` around ``env -0`` with ``base`` (so the nvcc guard and ``CUDA_HOME`` apply)
    and returns every variable that is new or changed, except shell bookkeeping (``_``, ``SHLVL``,
    ``PWD``, ``OLDPWD``).

    Args:
        sdist: The sdist spec; without ``build_modules`` nothing runs.
        base: :func:`base_environment`.

    Returns:
        Name to value; empty without modules.

    Raises:
        ReproduceLockError: If loading the modules fails.
    """
    if not sdist.build_modules:
        return {}
    text = _capture_env(in_modules(sdist, ["env", "-0"]), dict(base))
    pairs = dict(item.split("=", 1) for item in text.split("\0") if "=" in item)
    return {k: v for k, v in pairs.items() if k not in _UNRECORDED_VARS and base.get(k) != v}


def _uv_location(uv: str, variable: str, args: list[str]) -> str:
    """The caller's ``variable`` if set, else what ``uv <args>`` prints for it."""
    if os.environ.get(variable):
        return os.environ[variable]
    result = subprocess.run([uv, *args], capture_output=True, text=True, check=False)
    if result.returncode != 0 or not result.stdout.strip():
        raise ReproduceLockError(f"uv {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def base_environment(uv: str, work_dir: Path) -> dict[str, str]:
    """The allowlisted environment for every subprocess of a build.

    ``PATH`` is the system path; ``HOME`` and ``TMPDIR`` are under ``work_dir``; the locale is
    ``C.UTF-8``; ``SSL_CERT_FILE`` is :func:`ca_bundle`; ``UV_CACHE_DIR`` and
    ``UV_PYTHON_INSTALL_DIR`` are the caller's, resolved through uv when unset. Nothing else of the
    caller's environment is passed.

    Args:
        uv: The uv executable.
        work_dir: The build work directory; ``home`` and ``tmp`` are created under it.

    Returns:
        A new mapping.
    """
    home, tmp = work_dir / "home", work_dir / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    env = {
        "PATH": SYSTEM_PATH,
        "HOME": str(home),
        "TMPDIR": str(tmp),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "UV_CACHE_DIR": _uv_location(uv, "UV_CACHE_DIR", ["cache", "dir"]),
        "UV_PYTHON_INSTALL_DIR": _uv_location(uv, "UV_PYTHON_INSTALL_DIR", ["python", "dir"]),
    }
    bundle = ca_bundle()
    if bundle is not None:
        env["SSL_CERT_FILE"] = bundle
    return env


def build_environment(
    sdist: SdistSpec, base: Mapping[str, str], venv: Path, module_env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The wheel build's environment: ``base``, module variables, the venv first on ``PATH``, declared and fixed ones.

    The fixed variables are ``SOURCE_DATE_EPOCH``, ``TORCH_CUDA_ARCH_LIST`` (when declared) and
    :data:`OFFLINE_ENV`, so the build reaches no index or proxy.

    Args:
        sdist: The sdist spec.
        base: :func:`base_environment`.
        venv: The build venv.
        module_env: :func:`module_environment`; its ``PATH`` follows the venv's ``bin``.

    Returns:
        A new mapping.

    Raises:
        ReproduceLockError: If a ``build_env_vars`` key would replace an allowlisted or fixed variable.
    """
    fixed = {"SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH, **OFFLINE_ENV}
    if sdist.torch_cuda_arch_list is not None:
        fixed["TORCH_CUDA_ARCH_LIST"] = sdist.torch_cuda_arch_list
    modules = dict(module_env or {})
    path = modules.pop("PATH", SYSTEM_PATH)
    env = {**base, **modules, "PATH": f"{venv / 'bin'}:{path}"}
    clash = sorted((set(env) | set(fixed) | {"TORCH_CUDA_ARCH_LIST"}) & set(sdist.build_env_vars))
    if clash:
        raise ReproduceLockError(f"build_env_vars of {sdist.name} may not set {clash}")
    return {**env, **sdist.build_env_vars, **fixed}


def _torch_ref(torch_lock: TorchSpec | None) -> list[FileRef]:
    return [] if torch_lock is None else [FileRef(file=torch_lock.lock, sha256=torch_lock.lock_sha256)]


def _first_line(text: str) -> str:
    return text.splitlines()[0] if text and text != "unavailable" else text


def _last_line(text: str) -> str:
    return text.splitlines()[-1] if text and text != "unavailable" else text


def _build_info(uv: str, sdist: SdistSpec, python: str, env: dict[str, str]) -> dict[str, object]:
    info: dict[str, object] = {
        "python_exact": python,
        "uv": _capture([uv, "--version"], env),
        "gcc": _first_line(_capture(["gcc", "--version"], env)),
        "build_modules": list(sdist.build_modules),
        "torch_cuda_arch_list": sdist.torch_cuda_arch_list,
        "build_env_vars": dict(sorted(sdist.build_env_vars.items())),
        "source_date_epoch": SOURCE_DATE_EPOCH,
    }
    if needs_nvcc(sdist):
        info["nvcc"] = _last_line(_capture(["nvcc", "--version"], env))
    return info


def _store(wheel: Path, wheelhouse: Path, info: Mapping[str, object]) -> tuple[Path, str]:
    """Store a built wheel at ``<wheelhouse>/<sha256>/<name>``; an existing copy must hold the same bytes."""
    data = wheel.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    target = wheelhouse / sha / wheel.name
    if target.exists():
        if file_sha256(target) != sha:
            raise ReproduceLockError(f"{target} exists with other bytes than its directory name says")
        return target, sha
    target.parent.mkdir(parents=True, exist_ok=True)
    _place(data, target, replace=False)
    _place(
        (json.dumps(info, indent=2, sort_keys=True) + "\n").encode(), target.with_name(target.name + ".build-info.json")
    )
    return target, sha


def _place(data: bytes, target: Path, replace: bool = True) -> None:
    """Write ``data`` to a temp file beside ``target``, then move it into place in one step.

    With ``replace=False`` the move is a hard link, which fails if ``target`` exists; nothing is overwritten.
    """
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.chmod(tmp, 0o666 & ~_current_umask())
        if replace:
            os.replace(tmp, target)
        else:
            try:
                os.link(tmp, target)
            except FileExistsError as error:
                raise ReproduceLockError(f"{target} appeared while storing it; not overwritten") from error
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def build_wheel(
    sdist: SdistSpec,
    base_dir: Path,
    out_dir: Path,
    torch_lock: TorchSpec | None = None,
    *,
    python: str,
    work_dir: Path | None = None,
    sdist_cache: Path | None = None,
) -> tuple[Path, str]:
    """Build one sdist into a wheel in a fresh build venv.

    In order: the build lock, torch lock and patches are copied from ``base_dir`` and the copies
    checked against their manifest sha256s; the sdist is fetched, its bytes read once, checked,
    unpacked and patched; the build venv is created and the verified torch lock, then the build
    lock, are installed ``--no-deps --require-hashes``; the modules' variables are captured
    (:func:`module_environment`); the wheel is built in the sandbox (:func:`wheel_build_argv`,
    :func:`build_environment`) and stored content-addressed in ``out_dir`` with a
    ``<wheel>.build-info.json`` beside it (uv, gcc and nvcc versions, modules, ``python``).

    Args:
        sdist: The sdist spec.
        base_dir: The manifest repository.
        out_dir: The wheelhouse; the wheel goes to ``<out_dir>/<sha256>/<wheel file>``.
        torch_lock: The env's torch lock, installed into the build venv first when the build needs torch.
        python: The interpreter for the build venv (the env's ``python_exact``).
        work_dir: Where the build venv, inputs and source go; must not exist yet. Kept when given;
            otherwise a temporary directory is used and removed.
        sdist_cache: Where the sdist is downloaded; defaults to ``work_dir / "sdist"``.

    Returns:
        ``(wheel path, wheel sha256)``.

    Raises:
        ReproduceLockError: If an input's sha256 differs, a patch does not apply, a command fails, the
            build yields other than one wheel, or the wheelhouse holds other bytes at the wheel's path.
    """
    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="reproduce-build-") as tmp:
            return build_wheel(
                sdist, base_dir, out_dir, torch_lock, python=python, work_dir=Path(tmp) / "w", sdist_cache=sdist_cache
            )
    if work_dir.exists():
        raise ReproduceLockError(f"build work dir already exists: {work_dir}")
    inputs = work_dir / "inputs"
    build_lock = FileRef(file=sdist.build_env.lock, sha256=sdist.build_env.lock_sha256)
    _copy_verified([*_torch_ref(torch_lock), build_lock, *sdist.patches], base_dir, inputs)
    archive = fetch_sdist(sdist.url, sdist.sha256, sdist_cache or work_dir / "sdist")
    data = archive.read_bytes()
    if hashlib.sha256(data).hexdigest() != sdist.sha256:
        raise ReproduceLockError(f"sdist {archive}: sha256 changed after download")

    uv = find_uv()
    base = base_environment(uv, work_dir)
    src = _unpack(data, archive.name, work_dir / "src")
    for patch in sdist.patches:
        _apply_patch(inputs / patch.file, src, base)

    venv = work_dir / "venv"
    build_python = venv / "bin" / "python"
    _run([uv, "venv", "--no-config", "--python", python, str(venv)], env=base)
    if torch_lock is not None:
        torch_file = inputs / torch_lock.lock
        index = torch_index_url(torch_lock, torch_file.read_text())
        _run(_install_argv(uv, build_python, torch_file, index, binary_only=False), env=base)
    _run(_install_argv(uv, build_python, inputs / sdist.build_env.lock, None, binary_only=True), env=base)

    dist = work_dir / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    env = build_environment(sdist, base, venv, module_environment(sdist, base))
    env["UV_CACHE_DIR"] = str(work_dir / "tmp" / "uv-cache")
    writable = [src.parent, dist, work_dir / "home", work_dir / "tmp"]
    _run(wheel_build_argv(uv, build_python, src, dist, venv=venv, writable=writable), env=env)
    wheels = sorted(dist.glob("*.whl")) if dist.is_dir() else []
    if len(wheels) != 1:
        raise ReproduceLockError(f"building {sdist.name} produced {len(wheels)} wheels, expected 1")
    return _store(wheels[0], out_dir, _build_info(uv, sdist, python, env))


def build_config_sha256(sdist: SdistSpec, torch_lock: LockRef | None, python: str) -> str:
    """Digest of the build inputs beyond the sdist, build lock and patches.

    Args:
        sdist: The sdist spec.
        torch_lock: The torch lock installed into the build venv, if any.
        python: The build venv's interpreter version (``python_exact``).

    Returns:
        A hex sha256 over ``python``, the torch lock sha256, ``build_env_vars``, ``build_modules`` (as a list) and
        ``torch_cuda_arch_list``.
    """
    payload = {
        "python_exact": python,
        "torch_lock_sha256": None if torch_lock is None else torch_lock.lock_sha256,
        "build_env_vars": dict(sorted(sdist.build_env_vars.items())),
        "build_modules": list(sdist.build_modules),
        "torch_cuda_arch_list": sdist.torch_cuda_arch_list,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _format_line(line: WheelLine) -> str:
    patches = ",".join(line.patch_sha256s) or "-"
    fields = [line.name, line.version, line.sdist_url, line.sdist_sha256, line.build_env_sha256, patches]
    fields += [line.build_config_sha256, line.wheel_path, line.wheel_sha256]
    for field in fields:
        if not field or any(c.isspace() for c in field):
            raise ReproduceLockError(f"wheels.lock field is empty or has whitespace: {field!r}")
    return " ".join(fields)


def parse_wheels_lock(text: str, source: str = "wheels.lock") -> dict[str, WheelLine]:
    """Parse ``wheels.lock`` text.

    Args:
        text: The file's contents.
        source: A name for error messages.

    Returns:
        Lines keyed by normalised name.

    Raises:
        ReproduceLockError: If a line does not have the nine columns, or a name repeats.
    """
    out: dict[str, WheelLine] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = raw.split()
        if len(fields) != len(WheelLine._fields):
            raise ReproduceLockError(f"{source}:{number}: {len(fields)} columns, expected {len(WheelLine._fields)}")
        name, version, url, sdist_sha, build_sha, patch_field, config, wheel_path, wheel_sha = fields
        patches = () if patch_field == "-" else tuple(patch_field.split(","))
        line = WheelLine(name, version, url, sdist_sha, build_sha, patches, config, wheel_path, wheel_sha)
        key = normalize_name(line.name)
        if key in out:
            raise ReproduceLockError(f"{source}:{number}: {line.name} appears twice")
        out[key] = line
    return out


def format_wheels_lock(lines: Iterable[WheelLine]) -> str:
    """The text of ``wheels.lock``: the column header, then one line per sdist sorted by normalised name."""
    body = [_format_line(line) for line in sorted(lines, key=lambda line: normalize_name(line.name))]
    return "\n".join([WHEELS_HEADER, *body]) + "\n"


@contextmanager
def file_lock(path: Path, *, shared: bool = False) -> Iterator[None]:
    """Hold a ``flock`` on ``path`` (created if absent, with its directory), shared or exclusive.

    The file is opened read-write: on NFS ``flock`` becomes a POSIX lock, and a shared lock on a
    write-only descriptor fails with ``EBADF``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _locked(wheels_lock: Path, shared: bool) -> contextlib.AbstractContextManager[None]:
    """Hold a :func:`file_lock` on the sidecar ``<wheels.lock>.lock``."""
    return file_lock(wheels_lock.with_name(wheels_lock.name + ".lock"), shared=shared)


def _read_verified(wheels_lock: Path, expected_sha256: str | None) -> bytes | None:
    """Read ``wheels.lock`` once and check it against the manifest sha256 when one is given; ``None`` if absent."""
    if not wheels_lock.exists():
        if expected_sha256 is not None:
            raise ReproduceLockError(f"{wheels_lock} is missing; the manifest records sha256 {expected_sha256}")
        return None
    data = wheels_lock.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise ReproduceLockError(f"{wheels_lock}: sha256 {actual} != manifest {expected_sha256}")
    return data


def _write_atomic(path: Path, text: str) -> str:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        os.chmod(tmp, 0o666 & ~_current_umask())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return hashlib.sha256(text.encode()).hexdigest()


def _current_umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def read_wheels_lock(path: Path, expected_sha256: str | None = None) -> dict[str, WheelLine]:
    """Read ``wheels.lock`` under a shared lock.

    Args:
        path: The file.
        expected_sha256: The manifest's ``wheels.lock_sha256``; checked when given.

    Returns:
        Lines keyed by normalised name; empty if the file does not exist and no sha256 is expected.

    Raises:
        ReproduceLockError: If the sha256 differs, or a line is malformed.
    """
    with _locked(path, shared=True):
        data = _read_verified(path, expected_sha256)
    return {} if data is None else parse_wheels_lock(data.decode(), str(path))


def write_wheels_lock(path: Path, lines: Iterable[WheelLine]) -> str:
    """Replace ``wheels.lock`` atomically under an exclusive lock.

    Args:
        path: The file; its directory is created if absent.
        lines: The lines.

    Returns:
        The new file's sha256.
    """
    with _locked(path, shared=False):
        return _write_atomic(path, format_wheels_lock(lines))


def _expected(sdist: SdistSpec, torch_lock: LockRef | None, python: str) -> tuple[str, str, str, tuple[str, ...], str]:
    patches = tuple(p.sha256 for p in sdist.patches)
    config = build_config_sha256(sdist, torch_lock, python)
    return sdist.url, sdist.sha256, sdist.build_env.lock_sha256, patches, config


def record_wheel(
    wheels_lock: Path,
    sdist: SdistSpec,
    wheelhouse: Path,
    wheel: Path,
    wheel_sha256: str,
    torch_lock: LockRef | None = None,
    *,
    python: str,
    expected_sha256: str | None = None,
) -> tuple[WheelLine, str]:
    """Point the sdist's ``wheels.lock`` line at a stored wheel, replacing any earlier line for it.

    The earlier wheel stays in the wheelhouse. The caller writes the returned sha256 into the
    manifest's ``wheels.lock_sha256``.

    Args:
        wheels_lock: The file; created if absent.
        sdist: The sdist spec the wheel was built from.
        wheelhouse: The wheelhouse; ``wheel`` must lie under it.
        wheel: The stored wheel, ``<wheelhouse>/<sha256>/<file>``.
        wheel_sha256: Its sha256.
        torch_lock: The torch lock the build venv had, if any.
        python: The build venv's interpreter version.
        expected_sha256: The manifest's current ``wheels.lock_sha256``; the existing file is checked against it.

    Returns:
        ``(line written, new wheels.lock sha256)``.

    Raises:
        ReproduceLockError: If the wheel is not a wheel of ``sdist.name`` stored under its sha256, or the
            existing file does not match ``expected_sha256``.
    """
    parts = wheel.name.split("-")
    if not wheel.name.endswith(".whl") or len(parts) < 5 or normalize_name(parts[0]) != normalize_name(sdist.name):
        raise ReproduceLockError(f"{wheel.name} is not a wheel of {sdist.name}")
    relative = wheel.relative_to(wheelhouse).as_posix()
    if relative != f"{wheel_sha256}/{wheel.name}":
        raise ReproduceLockError(f"{wheel} is not stored at <wheelhouse>/<sha256>/<file>")
    url, sha, build_sha, patches, config = _expected(sdist, torch_lock, python)
    line = WheelLine(sdist.name, parts[1], url, sha, build_sha, patches, config, relative, wheel_sha256)
    with _locked(wheels_lock, shared=False):
        data = _read_verified(wheels_lock, expected_sha256)
        lines = {} if data is None else parse_wheels_lock(data.decode(), str(wheels_lock))
        lines[normalize_name(sdist.name)] = line
        new_sha = _write_atomic(wheels_lock, format_wheels_lock(lines.values()))
    return line, new_sha


def _wheel_location(wheelhouse: Path, line: WheelLine) -> Path:
    wheel = wheelhouse / line.wheel_path
    if line.wheel_path.split("/")[0] != line.wheel_sha256 or not wheel.resolve().is_relative_to(wheelhouse.resolve()):
        raise ReproduceLockError(f"wheels.lock path {line.wheel_path} is not <sha256>/<file> under the wheelhouse")
    return wheel


def reuse_wheel(
    wheels_lock: Path,
    sdist: SdistSpec,
    wheelhouse: Path,
    torch_lock: LockRef | None = None,
    *,
    python: str,
    expected_sha256: str | None = None,
) -> tuple[Path, str] | None:
    """Return the recorded wheel for an sdist, after checking ``wheels.lock``, the line and the wheel's sha256.

    Args:
        wheels_lock: The file.
        sdist: The sdist spec.
        wheelhouse: Where recorded wheels live.
        torch_lock: The torch lock the build venv would have, if any.
        python: The build venv's interpreter version.
        expected_sha256: The manifest's ``wheels.lock_sha256``; checked when given.

    Returns:
        ``(wheel path, sha256)``, or ``None`` if ``wheels.lock`` has no line for ``sdist.name``.

    Raises:
        ReproduceLockError: If ``wheels.lock`` does not match ``expected_sha256``, the line was recorded
            for other build inputs, the wheel is missing, or its sha256 differs from the line.
    """
    found = _reuse(wheels_lock, sdist, wheelhouse, torch_lock, python, expected_sha256)
    return None if found is None else found[:2]


def _reuse(
    wheels_lock: Path,
    sdist: SdistSpec,
    wheelhouse: Path,
    torch_lock: LockRef | None,
    python: str,
    expected_sha256: str | None,
) -> tuple[Path, str, str] | None:
    """:func:`reuse_wheel`, also returning the sha256 of the ``wheels.lock`` bytes it checked."""
    with _locked(wheels_lock, shared=True):
        data = _read_verified(wheels_lock, expected_sha256)
    if data is None:
        return None
    line = parse_wheels_lock(data.decode(), str(wheels_lock)).get(normalize_name(sdist.name))
    if line is None:
        return None
    recorded = (line.sdist_url, line.sdist_sha256, line.build_env_sha256, line.patch_sha256s, line.build_config_sha256)
    if recorded != _expected(sdist, torch_lock, python):
        raise ReproduceLockError(
            f"{wheels_lock}: the {sdist.name} line was recorded for other build inputs; build again with --rebuild"
        )
    wheel = _wheel_location(wheelhouse, line)
    if not wheel.is_file():
        raise ReproduceLockError(f"recorded wheel {line.wheel_path} is missing from {wheelhouse}; use --rebuild")
    actual = file_sha256(wheel)
    if actual != line.wheel_sha256:
        raise ReproduceLockError(f"{wheel}: sha256 {actual} != wheels.lock {line.wheel_sha256}")
    return wheel, actual, hashlib.sha256(data).hexdigest()


def stage_wheel(wheel: Path, sha256: str, dest_dir: Path) -> Path:
    """Copy a recorded wheel to ``dest_dir`` and check the copy's sha256, so the install reads checked bytes.

    Args:
        wheel: The wheelhouse wheel.
        sha256: Its recorded sha256.
        dest_dir: A private directory, e.g. under ``{tmp}``; created if absent.

    Returns:
        ``dest_dir / <wheel file name>``.

    Raises:
        ReproduceLockError: If the copied bytes do not match ``sha256``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    data = wheel.read_bytes()
    actual = hashlib.sha256(data).hexdigest()
    if actual != sha256:
        raise ReproduceLockError(f"{wheel}: sha256 {actual} != recorded {sha256}")
    target = dest_dir / wheel.name
    target.write_bytes(data)
    return target


def ensure_wheel(
    sdist: SdistSpec,
    base_dir: Path,
    wheelhouse: Path,
    wheels_lock: Path,
    torch_lock: TorchSpec | None = None,
    *,
    python: str,
    rebuild: bool = False,
    expected_sha256: str | None = None,
    work_dir: Path | None = None,
    sdist_cache: Path | None = None,
) -> WheelResult:
    """Reuse the recorded wheel for an sdist, or build and record it; the entry point of ``build-one``.

    Without ``work_dir`` the build runs in a temporary directory that is removed afterwards.

    Args:
        sdist: The sdist spec.
        base_dir: The manifest repository.
        wheelhouse: Where wheels live.
        wheels_lock: The env's ``wheels.lock``.
        torch_lock: The env's torch lock, when the build needs torch.
        python: The build venv interpreter (the env's ``python_exact``).
        rebuild: Build even if a line exists; the line is repointed at the new wheel.
        expected_sha256: The manifest's ``wheels.lock_sha256``, if it has one.
        work_dir: Passed to :func:`build_wheel`.
        sdist_cache: Passed to :func:`build_wheel`.

    Returns:
        A :class:`WheelResult`; when ``built``, the caller records ``wheels_lock_sha256`` in the manifest.

    Raises:
        ReproduceLockError: From :func:`reuse_wheel` (without ``rebuild``), :func:`build_wheel` or :func:`record_wheel`.
    """
    if not rebuild:
        reused = _reuse(wheels_lock, sdist, wheelhouse, torch_lock, python, expected_sha256)
        if reused is not None:
            return WheelResult(*reused, built=False)
    wheel, sha = build_wheel(
        sdist, base_dir, wheelhouse, torch_lock, python=python, work_dir=work_dir, sdist_cache=sdist_cache
    )
    _, lock_sha = record_wheel(
        wheels_lock, sdist, wheelhouse, wheel, sha, torch_lock, python=python, expected_sha256=expected_sha256
    )
    return WheelResult(wheel, sha, lock_sha, built=True)


def _safe_word(value: str, what: str) -> str:
    if not _SAFE_WORD.match(value):
        raise ValueError(f"{what} {value!r} is not a plain word")
    return value


def _double_quoted(value: str, what: str) -> str:
    if _DOUBLE_QUOTE_UNSAFE.search(value):
        raise ValueError(f"{what} {value!r} cannot be double-quoted safely")
    return f'"{value}"'


def render_build_job(
    profile: Profile,
    sdist: SdistSpec,
    *,
    manifest: Path,
    profile_path: Path,
    step: str,
    rebuild: bool = False,
    base_dir: Path | None = None,
    date: datetime.date | None = None,
    driver_python: str = sys.executable,
    uv: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """Render the batch script that runs ``build-one`` for one sdist under the profile's build partition.

    The job starts from an empty environment (``--export=NONE``) and exports only ``PATH``, the
    :data:`BATCH_EXPORTS` variables set in ``environ``, the driver interpreter, ``REPRO_UV`` (the
    absolute uv, read by :func:`find_uv`) and a per-job ``TMPDIR`` under ``/dev/shm`` that is
    removed on exit. Modules are not loaded here: ``build-one`` loads ``build_modules`` itself
    (:func:`in_modules`). The log directory ``<slurm_logs_root>/<date>/`` must exist before submission.

    Args:
        profile: The site profile; ``partitions[build]`` gives partition, CPUs, memory, time and gres.
        sdist: The sdist spec; its ``torch_cuda_arch_list`` is exported.
        manifest: The manifest holding the step.
        profile_path: The profile file passed to ``build-one``.
        step: The step whose env lists the sdist.
        rebuild: Pass ``--rebuild``.
        base_dir: Passed to ``build-one`` as ``--base-dir`` (the manifest repository prefetch used).
        date: The log date; defaults to today. The job runs in the log dir (``--chdir``).
        driver_python: The interpreter with senselab installed.
        uv: The uv executable; defaults to :func:`find_uv`, resolved to an absolute path now.
        environ: Where the exported variables' values come from; defaults to ``os.environ``.

    Returns:
        The script text.

    Raises:
        ValueError: If a value would need quoting the script cannot express safely.
    """
    environ = os.environ if environ is None else environ
    uv = find_uv() if uv is None else uv
    if not os.path.isabs(uv):
        raise ValueError(f"uv must be an absolute path: {uv!r}")
    part = profile.partitions[profile.build]
    logs = _safe_word(str(profile.slurm_logs_root / (date or datetime.date.today()).isoformat()), "log dir")
    name = _safe_word(sdist.name, "sdist name")
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=reproduce-build-{name}",
        f"#SBATCH --partition={_safe_word(part.partition, 'partition')}",
        f"#SBATCH --cpus-per-task={part.cpus}",
        f"#SBATCH --mem={_safe_word(part.mem, 'mem')}",
        f"#SBATCH --time={_safe_word(part.time, 'time')}",
    ]
    if part.gres is not None:
        lines.append(f"#SBATCH --gres={_safe_word(part.gres, 'gres')}")
    lines += [
        f"#SBATCH --output={logs}/build-{name}-%j.out",
        f"#SBATCH --error={logs}/build-{name}-%j.err",
        f"#SBATCH --chdir={logs}",
        "#SBATCH --export=NONE",
        "set -euo pipefail",
        f"export PATH={SYSTEM_PATH}",
    ]
    lines += [f"export {key}={shlex.quote(environ[key])}" for key in BATCH_EXPORTS if environ.get(key)]
    lines += [
        f"export REPRO_DRIVER_PYTHON={shlex.quote(driver_python)}",
        f"export REPRO_UV={shlex.quote(uv)}",
        'export TMPDIR="/dev/shm/reproduce-build-${SLURM_JOB_ID}"',
        'mkdir -p "$TMPDIR"',
        "trap 'rm -rf \"$TMPDIR\"' EXIT",
    ]
    if sdist.torch_cuda_arch_list is not None:
        arch = _double_quoted(sdist.torch_cuda_arch_list, "torch_cuda_arch_list")
        lines.append(f"export TORCH_CUDA_ARCH_LIST={arch}")
    argv = ["-P", "-m", "senselab.utils.reproduce", "build-one", str(manifest)]
    argv += ["--profile", str(profile_path), "--step", step, "--sdist", sdist.name]
    if base_dir is not None:
        argv += ["--base-dir", str(base_dir)]
    if rebuild:
        argv.append("--rebuild")
    lines.append('"$REPRO_DRIVER_PYTHON" ' + shlex.join(argv))
    return "\n".join(lines) + "\n"
