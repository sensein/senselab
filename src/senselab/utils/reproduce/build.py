"""Stage 3 of a reproduce env: sdists built into wheels, recorded in ``wheels.lock`` and reused.

Each :class:`SdistSpec` is built in its own build venv, made from its hashed ``build_env`` lock (and
the env's torch lock when given), with ``uv build --wheel --no-build-isolation`` and a fixed
``SOURCE_DATE_EPOCH`` and umask. The first build records the wheel's sha256 in ``wheels.lock``; later calls
reuse the wheel from the wheelhouse after checking that sha256, and only ``rebuild`` builds again.

``wheels.lock`` has one whitespace-separated line per sdist under a column header::

    # name version sdist_url sdist_sha256 build_env_sha256 patch_sha256s build_config_sha256 wheel_file wheel_sha256

``patch_sha256s`` is a comma-joined list in manifest order, or ``-``. ``build_config_sha256`` is
:func:`build_config_sha256`: the torch lock sha256, build variables, CUDA module and arch list.
Paths in a spec are relative to the manifest repository, passed as ``base_dir``.
"""

from __future__ import annotations

import datetime
import fcntl
import hashlib
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

from senselab.utils.reproduce.lockgen import find_uv
from senselab.utils.reproduce.locks import ReproduceLockError, file_sha256, lock_entries, normalize_name, verify_files
from senselab.utils.reproduce.spec import FileRef, LockRef, Profile, SdistSpec, TorchSpec

SOURCE_DATE_EPOCH = "315532800"
BUILD_UMASK = 0o007
WHEELS_HEADER = (
    "# name version sdist_url sdist_sha256 build_env_sha256 patch_sha256s build_config_sha256 wheel_file wheel_sha256"
)
CUDA_HOME_FROM_NVCC = 'export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"'
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
    wheel_file: str
    wheel_sha256: str


def _run(argv: list[str], env: dict[str, str] | None = None) -> None:
    """Run one command under :data:`BUILD_UMASK`; raise with the tail of its output if it fails."""
    result = subprocess.run(argv, env=env, capture_output=True, text=True, check=False, umask=BUILD_UMASK)
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()[-_TAIL:]
        raise ReproduceLockError(f"{shlex.join(argv[:3])} failed ({result.returncode}): {output}")


def _sdist_filename(url: str) -> str:
    name = urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1])
    if not name or name in (".", "..") or "/" in name:
        raise ReproduceLockError(f"sdist URL has no file name: {url}")
    return name


def _tls_context() -> ssl.SSLContext:
    """A TLS context: ``SSL_CERT_FILE`` when set, else certifi's bundle, else the interpreter default."""
    if os.environ.get("SSL_CERT_FILE"):
        return ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


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
    fd, part = tempfile.mkstemp(dir=dest, prefix=f".{path.name}.", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(url, context=_tls_context()) as response:  # noqa: S310
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


def _tar_member(member: tarfile.TarInfo, dest: str) -> tarfile.TarInfo | None:
    """The ``data`` extraction filter, with a regular file's mode set to its archive mode under :data:`BUILD_UMASK`."""
    filtered = tarfile.data_filter(member, dest)
    if filtered is None or not member.isreg():
        return filtered
    return filtered.replace(mode=((member.mode & 0o777) | 0o600) & ~BUILD_UMASK, deep=False)


def _unpack(sdist: Path, dest: Path) -> Path:
    """Unpack an sdist into ``dest`` and return its single top-level directory.

    File modes end up inside the wheel, so a tar member's mode is its archive mode under
    :data:`BUILD_UMASK`, whatever the caller's umask.
    """
    dest.mkdir(parents=True)
    if sdist.name.endswith(".zip"):
        with zipfile.ZipFile(sdist) as archive:
            for member in archive.namelist():
                target = (dest / member).resolve()
                if not target.is_relative_to(dest.resolve()):
                    raise ReproduceLockError(f"{sdist.name}: member {member!r} leaves the unpack directory")
            archive.extractall(dest)
    else:
        with tarfile.open(sdist) as archive:
            archive.extractall(dest, filter=_tar_member)
    tops = list(dest.iterdir())
    if len(tops) != 1 or not tops[0].is_dir():
        raise ReproduceLockError(f"{sdist.name} does not unpack to one top-level directory")
    return tops[0]


def _apply_patch(patch: Path, src: Path) -> None:
    """Apply a ``-p1`` unified diff to the unpacked sdist root, with no fuzz."""
    argv = ["patch", "-p1", "--batch", "--forward", "--fuzz=0", "--no-backup-if-mismatch", "-d", str(src)]
    result = subprocess.run([*argv, "-i", str(patch)], capture_output=True, text=True, check=False, umask=BUILD_UMASK)
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


def wheel_build_argv(uv: str, sdist: SdistSpec, build_python: Path, src: Path, out_dir: Path) -> list[str]:
    """The exact argv that builds the wheel.

    Without a CUDA module this is ``uv build --wheel --no-build-isolation``. With one, the same argv
    runs inside ``bash -lc 'module load <cuda_module> && export CUDA_HOME=… && exec "$@"'``, because
    ``module`` is a login-shell function.

    Args:
        uv: The uv executable.
        sdist: The sdist spec.
        build_python: The build venv's interpreter.
        src: The unpacked, patched sdist.
        out_dir: Where uv writes the wheel.

    Returns:
        The argv.

    Raises:
        ValueError: If ``cuda_module`` is not a plain module name.
    """
    argv = [uv, "build", "--no-config", "--wheel", "--no-build-isolation", "--python", str(build_python)]
    argv += ["--out-dir", str(out_dir), str(src)]
    if sdist.cuda_module is None:
        return argv
    module = _safe_word(sdist.cuda_module, "cuda_module")
    script = f'module load {module} && {CUDA_HOME_FROM_NVCC} && exec "$@"'
    return ["bash", "-lc", script, "bash", *argv]


def build_environment(sdist: SdistSpec, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The build's environment: ``base`` (default ``os.environ``) plus the fixed and declared variables.

    Args:
        sdist: The sdist spec; its ``build_env_vars`` and ``torch_cuda_arch_list`` are added.
        base: The environment to extend.

    Returns:
        A new mapping with ``SOURCE_DATE_EPOCH`` fixed.
    """
    env = dict(os.environ if base is None else base)
    env.update(sdist.build_env_vars)
    env["SOURCE_DATE_EPOCH"] = SOURCE_DATE_EPOCH
    if sdist.torch_cuda_arch_list is not None:
        env["TORCH_CUDA_ARCH_LIST"] = sdist.torch_cuda_arch_list
    return env


def _torch_ref(torch_lock: TorchSpec | None) -> list[FileRef]:
    return [] if torch_lock is None else [FileRef(file=torch_lock.lock, sha256=torch_lock.lock_sha256)]


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

    In order: the build lock, torch lock and patches are copied from ``base_dir`` and checked
    against their manifest sha256s; the sdist is fetched, checked, unpacked and patched; the build
    venv is created and the verified torch lock, then the build lock, are installed ``--no-deps
    --require-hashes``; the wheel is built (:func:`wheel_build_argv`,
    :func:`build_environment`) and moved into ``out_dir``.

    Args:
        sdist: The sdist spec.
        base_dir: The manifest repository.
        out_dir: The wheelhouse; created if absent. A wheel of the same name is replaced.
        torch_lock: The env's torch lock, installed into the build venv first when the build needs torch.
        python: The interpreter for the build venv (the env's ``python_exact``).
        work_dir: Where the build venv, inputs and source go; must not exist yet. Kept when given;
            otherwise a temporary directory is used and removed.
        sdist_cache: Where the sdist is downloaded; defaults to ``work_dir / "sdist"``.

    Returns:
        ``(wheel path, wheel sha256)``.

    Raises:
        ReproduceLockError: If an input's sha256 differs, a patch does not apply, a command fails, or
            the build yields other than one wheel.
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

    src = _unpack(archive, work_dir / "src")
    for patch in sdist.patches:
        _apply_patch(inputs / patch.file, src)

    uv = find_uv()
    venv = work_dir / "venv"
    build_python = venv / "bin" / "python"
    _run([uv, "venv", "--no-config", "--python", python, str(venv)])
    if torch_lock is not None:
        torch_file = inputs / torch_lock.lock
        index = torch_index_url(torch_lock, torch_file.read_text())
        _run(_install_argv(uv, build_python, torch_file, index, binary_only=False))
    _run(_install_argv(uv, build_python, inputs / sdist.build_env.lock, None, binary_only=True))

    dist = work_dir / "dist"
    _run(wheel_build_argv(uv, sdist, build_python, src, dist), env=build_environment(sdist))
    wheels = sorted(dist.glob("*.whl")) if dist.is_dir() else []
    if len(wheels) != 1:
        raise ReproduceLockError(f"building {sdist.name} produced {len(wheels)} wheels, expected 1")
    out_dir.mkdir(parents=True, exist_ok=True)
    wheel = out_dir / wheels[0].name
    shutil.move(wheels[0], wheel)
    return wheel, file_sha256(wheel)


def build_config_sha256(sdist: SdistSpec, torch_lock: LockRef | None) -> str:
    """Digest of the build inputs beyond the sdist, build lock and patches.

    Args:
        sdist: The sdist spec.
        torch_lock: The torch lock installed into the build venv, if any.

    Returns:
        A hex sha256 over the torch lock sha256, ``build_env_vars``, ``cuda_module`` and ``torch_cuda_arch_list``.
    """
    payload = {
        "torch_lock_sha256": None if torch_lock is None else torch_lock.lock_sha256,
        "build_env_vars": dict(sorted(sdist.build_env_vars.items())),
        "cuda_module": sdist.cuda_module,
        "torch_cuda_arch_list": sdist.torch_cuda_arch_list,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _format_line(line: WheelLine) -> str:
    patches = ",".join(line.patch_sha256s) or "-"
    fields = [line.name, line.version, line.sdist_url, line.sdist_sha256, line.build_env_sha256, patches]
    fields += [line.build_config_sha256, line.wheel_file, line.wheel_sha256]
    for field in fields:
        if not field or any(c.isspace() for c in field):
            raise ReproduceLockError(f"wheels.lock field is empty or has whitespace: {field!r}")
    return " ".join(fields)


def _parse_lines(text: str, source: str) -> dict[str, WheelLine]:
    out: dict[str, WheelLine] = {}
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip() or raw.startswith("#"):
            continue
        fields = raw.split()
        if len(fields) != len(WheelLine._fields):
            raise ReproduceLockError(f"{source}:{number}: {len(fields)} columns, expected {len(WheelLine._fields)}")
        patches = () if fields[5] == "-" else tuple(fields[5].split(","))
        name, version, url, sdist_sha, build_sha, _, config, wheel_file, wheel_sha = fields
        line = WheelLine(name, version, url, sdist_sha, build_sha, patches, config, wheel_file, wheel_sha)
        key = normalize_name(line.name)
        if key in out:
            raise ReproduceLockError(f"{source}:{number}: {line.name} appears twice")
        out[key] = line
    return out


def read_wheels_lock(path: Path) -> dict[str, WheelLine]:
    """Read ``wheels.lock``.

    Args:
        path: The file.

    Returns:
        Lines keyed by normalised name; empty if the file does not exist.

    Raises:
        ReproduceLockError: If a line does not have the nine columns, or a name repeats.
    """
    if not path.exists():
        return {}
    return _parse_lines(path.read_text(), str(path))


def write_wheels_lock(path: Path, lines: Iterable[WheelLine]) -> None:
    """Write ``wheels.lock``: the column header, then one line per sdist sorted by normalised name.

    Args:
        path: The file; its directory must exist.
        lines: The lines.
    """
    body = [_format_line(line) for line in sorted(lines, key=lambda line: normalize_name(line.name))]
    path.write_text("\n".join([WHEELS_HEADER, *body]) + "\n")


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    """Hold an exclusive lock on ``path`` (created if absent) so concurrent build jobs serialise their updates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _expected(sdist: SdistSpec, torch_lock: LockRef | None) -> tuple[str, str, str, tuple[str, ...], str]:
    patches = tuple(p.sha256 for p in sdist.patches)
    return sdist.url, sdist.sha256, sdist.build_env.lock_sha256, patches, build_config_sha256(sdist, torch_lock)


def record_wheel(
    wheels_lock: Path, sdist: SdistSpec, wheel: Path, wheel_sha256: str, torch_lock: LockRef | None = None
) -> WheelLine:
    """Record a built wheel in ``wheels.lock``, replacing any line for the same sdist name.

    Args:
        wheels_lock: The file; created if absent.
        sdist: The sdist spec the wheel was built from.
        wheel: The wheel in the wheelhouse.
        wheel_sha256: Its sha256.
        torch_lock: The torch lock the build venv had, if any.

    Returns:
        The line written.

    Raises:
        ReproduceLockError: If the wheel file name is not a wheel of ``sdist.name``.
    """
    parts = wheel.name.split("-")
    if not wheel.name.endswith(".whl") or len(parts) < 5 or normalize_name(parts[0]) != normalize_name(sdist.name):
        raise ReproduceLockError(f"{wheel.name} is not a wheel of {sdist.name}")
    url, sha, build_sha, patches, config = _expected(sdist, torch_lock)
    line = WheelLine(sdist.name, parts[1], url, sha, build_sha, patches, config, wheel.name, wheel_sha256)
    with _locked(wheels_lock):
        lines = _parse_lines(wheels_lock.read_text(), str(wheels_lock))
        lines[normalize_name(sdist.name)] = line
        write_wheels_lock(wheels_lock, lines.values())
    return line


def reuse_wheel(
    wheels_lock: Path, sdist: SdistSpec, wheelhouse: Path, torch_lock: LockRef | None = None
) -> tuple[Path, str] | None:
    """Return the recorded wheel for an sdist, after checking its line and its sha256.

    Args:
        wheels_lock: The file.
        sdist: The sdist spec.
        wheelhouse: Where recorded wheels live.
        torch_lock: The torch lock the build venv would have, if any.

    Returns:
        ``(wheel path, sha256)``, or ``None`` if ``wheels.lock`` has no line for ``sdist.name``.

    Raises:
        ReproduceLockError: If the line was recorded for other build inputs, the wheel is missing, or its
            sha256 differs from the line. Each needs an explicit rebuild.
    """
    line = read_wheels_lock(wheels_lock).get(normalize_name(sdist.name))
    if line is None:
        return None
    recorded = (line.sdist_url, line.sdist_sha256, line.build_env_sha256, line.patch_sha256s, line.build_config_sha256)
    if recorded != _expected(sdist, torch_lock):
        raise ReproduceLockError(
            f"{wheels_lock}: the {sdist.name} line was recorded for other build inputs; build again with --rebuild"
        )
    wheel = wheelhouse / line.wheel_file
    if not wheel.is_file():
        raise ReproduceLockError(f"recorded wheel {line.wheel_file} is missing from {wheelhouse}; use --rebuild")
    actual = file_sha256(wheel)
    if actual != line.wheel_sha256:
        raise ReproduceLockError(f"{wheel}: sha256 {actual} != wheels.lock {line.wheel_sha256}")
    return wheel, actual


def ensure_wheel(
    sdist: SdistSpec,
    base_dir: Path,
    wheelhouse: Path,
    wheels_lock: Path,
    torch_lock: TorchSpec | None = None,
    *,
    python: str,
    rebuild: bool = False,
    work_dir: Path | None = None,
    sdist_cache: Path | None = None,
) -> tuple[Path, str]:
    """Reuse the recorded wheel for an sdist, or build and record it; the entry point of ``build-one``.

    Args:
        sdist: The sdist spec.
        base_dir: The manifest repository.
        wheelhouse: Where wheels live.
        wheels_lock: The env's ``wheels.lock``.
        torch_lock: The env's torch lock, when the build needs torch.
        python: The build venv interpreter (the env's ``python_exact``).
        rebuild: Build even if a line exists, and replace that line.
        work_dir: Passed to :func:`build_wheel`.
        sdist_cache: Passed to :func:`build_wheel`.

    Returns:
        ``(wheel path, sha256)``.

    Raises:
        ReproduceLockError: From :func:`reuse_wheel` (without ``rebuild``) or :func:`build_wheel`.
    """
    if not rebuild:
        reused = reuse_wheel(wheels_lock, sdist, wheelhouse, torch_lock)
        if reused is not None:
            return reused
    wheel, sha = build_wheel(
        sdist, base_dir, wheelhouse, torch_lock, python=python, work_dir=work_dir, sdist_cache=sdist_cache
    )
    record_wheel(wheels_lock, sdist, wheel, sha, torch_lock)
    return wheel, sha


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
    date: datetime.date | None = None,
    driver_python: str = sys.executable,
) -> str:
    """Render the batch script that runs ``build-one`` for one sdist under the profile's build partition.

    The log directory ``<slurm_logs_root>/<date>/`` must exist before submission.

    Args:
        profile: The site profile; ``partitions[build]`` gives partition, CPUs, memory, time and gres.
        sdist: The sdist spec; ``cuda_module`` and ``torch_cuda_arch_list`` are loaded and exported.
        manifest: The manifest holding the step.
        profile_path: The profile file passed to ``build-one``.
        step: The step whose env lists the sdist.
        rebuild: Pass ``--rebuild``.
        date: The log date; defaults to today.
        driver_python: The interpreter with senselab installed.

    Returns:
        The script text.

    Raises:
        ValueError: If a value would need quoting the script cannot express safely.
    """
    part = profile.partitions[profile.build]
    logs = profile.slurm_logs_root / (date or datetime.date.today()).isoformat()
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
        f"#SBATCH --output={_safe_word(str(logs), 'log dir')}/build-{name}-%j.out",
        f"#SBATCH --error={_safe_word(str(logs), 'log dir')}/build-{name}-%j.err",
        "set -euo pipefail",
    ]
    if sdist.cuda_module is not None:
        lines.append(f"module load {_safe_word(sdist.cuda_module, 'cuda_module')}")
    if sdist.torch_cuda_arch_list is not None:
        arch = _double_quoted(sdist.torch_cuda_arch_list, "torch_cuda_arch_list")
        lines.append(f"export TORCH_CUDA_ARCH_LIST={arch}")
    argv = [driver_python, "-m", "senselab.utils.reproduce", "build-one", str(manifest)]
    argv += ["--profile", str(profile_path), "--step", step, "--sdist", sdist.name]
    if rebuild:
        argv.append("--rebuild")
    lines.append("exec " + shlex.join(argv))
    return "\n".join(lines) + "\n"
