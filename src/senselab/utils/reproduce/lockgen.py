"""Lock generation for reproduce envs: the uv commands that produce each lock of an :class:`EnvSpec`.

Every builder takes a (possibly draft) env and the manifest repository ``base_dir``, runs one uv
command, writes the lock to the path the env names under ``base_dir`` with a ``# reproduce lock``
header, and returns the argv it ran. Paths in that argv are relative to the command's working
directory (``base_dir``, or the project for ``uv export``), so the recorded command is
location-independent. Lock sha256s in the env are not read; ``reproduce lock`` fills them in afterwards.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import shlex
import subprocess
import tomllib
import urllib.parse
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from senselab.utils import uv_tools
from senselab.utils.reproduce.locks import (
    TORCH_NAMES,
    ReproduceLockError,
    is_torch_family,
    lock_entries,
    normalize_name,
    requirements_digest_for,
    uv_lock_torch_family,
    write_header,
)
from senselab.utils.reproduce.spec import EnvSpec

TORCH_PLATFORM = "x86_64-manylinux_2_28"
UPLOAD_TIMELESS_PREFIX = "nvidia-"


def uv_executable() -> str:
    """Return the absolute path of the ``uv`` executable: ``REPRO_UV`` when set, else `uv_tools.find_uv` on ``PATH``.

    Only ``PATH`` is searched (``home=False``), never a home install the job's environment did not name.

    Raises:
        ReproduceLockError: If ``REPRO_UV`` is not an absolute path to an executable file, or it is unset
            and ``PATH`` has no ``uv``.
    """
    pinned = os.environ.get("REPRO_UV")
    if pinned:
        if not os.path.isabs(pinned) or not os.path.isfile(pinned) or not os.access(pinned, os.X_OK):
            raise ReproduceLockError(f"REPRO_UV={pinned!r} is not an absolute path to an executable")
        return pinned
    try:
        uv = uv_tools.find_uv(home=False)
    except FileNotFoundError:
        raise ReproduceLockError("uv not found on PATH; lock generation needs uv") from None
    return os.path.abspath(uv)


def exclude_newer_timestamp(date: datetime.date) -> str:
    """The ``--exclude-newer`` value for a manifest date: the end of that day in UTC."""
    return f"{date.isoformat()}T23:59:59Z"


def _run(argv: list[str], cwd: Path) -> str:
    result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise ReproduceLockError(f"{shlex.join(argv[1:3])} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


def _write(base_dir: Path, rel: str, text: str) -> None:
    path = base_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _emit(base_dir: Path, rel: str, argv: list[str], cwd: Path, fields: Mapping[str, str]) -> list[str]:
    body = _run(argv, cwd)
    command = shlex.join(["uv", *argv[1:]])
    _write(base_dir, rel, write_header(body, {**fields, "command": command}))
    return argv


def _no_emit(names: Iterable[str]) -> list[str]:
    out: list[str] = []
    for name in dict.fromkeys(normalize_name(n) for n in names):
        out += ["--no-emit-package", name]
    return out


def _lock_names(base_dir: Path, rel: str) -> list[str]:
    return sorted({e.name for e in lock_entries((base_dir / rel).read_text())})


def _held_out(env: EnvSpec) -> list[str]:
    return [s.name for s in env.sdist_allow] + list(env.extra_no_emit)


def _resolve_family(env: EnvSpec, base_dir: Path) -> dict[str, str]:
    """The torch-family pins of the first resolve, ``{name: version}``, in name order."""
    if env.torch is None:
        return {}
    if env.first_resolve is None:
        raise ReproduceLockError("a torch lock with a requirements.in body needs first_resolve")
    pins: dict[str, str] = {}
    for entry in lock_entries((base_dir / env.first_resolve.file).read_text()):
        if not is_torch_family(entry.name):
            continue
        if entry.is_url or pins.get(entry.name, entry.pin) != entry.pin:
            raise ReproduceLockError(f"first_resolve pins {entry.name} more than one way or by URL")
        pins[entry.name] = entry.pin
    return dict(sorted(pins.items()))


def _compile_inputs(env: EnvSpec) -> list[str]:
    body = env.body
    argv = ["--python-version", env.python, "--exclude-newer", exclude_newer_timestamp(env.exclude_newer)]
    argv += ["--index-url", body.index_url]
    if body.config_file is not None:
        argv += ["--config-file", body.config_file]
    if body.overrides is not None:
        argv += ["--overrides", body.overrides]
    return argv


def _requirements_in(env: EnvSpec) -> str:
    if env.body.source != "requirements.in" or env.body.requirements_in is None:
        raise ReproduceLockError("this lock is compiled from a requirements.in body")
    return env.body.requirements_in


def first_resolve(env: EnvSpec, base_dir: Path) -> list[str]:
    """Write ``env.first_resolve``: the unhashed universal resolve of the body inputs.

    Args:
        env: A requirements-derived env with ``first_resolve`` set.
        base_dir: The manifest repository.

    Returns:
        The argv run, with ``base_dir`` as its working directory.

    Raises:
        ReproduceLockError: If the env has no ``first_resolve``, uv is missing or the resolve fails.
    """
    if env.first_resolve is None:
        raise ReproduceLockError("env has no first_resolve file")
    argv = [uv_executable(), "pip", "compile", _requirements_in(env), "--universal", "--no-annotate", "--no-header"]
    argv += _compile_inputs(env)
    return _emit(base_dir, env.first_resolve.file, argv, base_dir, {"source": "first-resolve", "python": env.python})


def torch_lock_from_index(env: EnvSpec, base_dir: Path) -> list[str]:
    """Write ``env.torch.lock`` compiled against ``env.torch.index_url`` from the first resolve's torch pins.

    The pins are written to a ``.in`` file beside the lock. Every ``nvidia-*`` name gets
    ``--exclude-newer-package <name>=false``, because the PyTorch index carries no upload times for them.

    Args:
        env: A requirements-derived env with a torch lock and a written ``first_resolve``.
        base_dir: The manifest repository.

    Returns:
        The argv run, with ``base_dir`` as its working directory.

    Raises:
        ReproduceLockError: If the env has no torch lock, the first resolve has no torch family or pins a
            name two ways, uv is missing or the compile fails.
    """
    if env.torch is None:
        raise ReproduceLockError("env has no torch lock")
    pins = _resolve_family(env, base_dir)
    if not pins:
        raise ReproduceLockError("first_resolve has no torch-family pins")
    torch_in = str(PurePosixPath(env.torch.lock).with_suffix(".in"))
    _write(base_dir, torch_in, "".join(f"{name}=={version}\n" for name, version in pins.items()))
    argv = [
        uv_executable(),
        "pip",
        "compile",
        torch_in,
        "--no-deps",
        "--generate-hashes",
        "--emit-index-url",
        "--no-header",
    ]
    argv += ["--python-version", env.python, "--exclude-newer", exclude_newer_timestamp(env.exclude_newer)]
    for name in pins:
        if name.startswith(UPLOAD_TIMELESS_PREFIX):
            argv += ["--exclude-newer-package", f"{name}=false"]
    argv += ["--python-platform", TORCH_PLATFORM, "--index-url", env.torch.index_url]
    return _emit(base_dir, env.torch.lock, argv, base_dir, {"source": "torch-index", "python": env.python})


def _wheel_tags(url: str) -> tuple[set[str], str, set[str]]:
    filename = urllib.parse.unquote(urllib.parse.urlsplit(url).path.rsplit("/", 1)[-1])
    python, abi, platform = filename.removesuffix(".whl").split("-")[-3:]
    return set(python.split(".")), abi, set(platform.split("."))


def _linux_wheel(name: str, packages: list[dict[str, Any]], cp: str) -> dict[str, Any]:
    matches = []
    for package in packages:
        for wheel in package.get("wheels", []):
            python, abi, platform = _wheel_tags(wheel["url"])
            on_linux = any("manylinux" in p and p.endswith("x86_64") for p in platform)
            fits = (cp in python and abi in (cp, "abi3", "none")) or ("py3" in python and abi == "none")
            if on_linux and fits:
                matches.append(wheel)
    if len(matches) != 1:
        raise ReproduceLockError(f"{name}: {len(matches)} linux x86_64 wheels for {cp} in uv.lock, need exactly 1")
    return matches[0]


def torch_lock_from_uvlock(env: EnvSpec, base_dir: Path, uv_lock: Path, roots: Sequence[str]) -> list[str]:
    """Write ``env.torch.lock`` as URL-pinned, hashed lines taken from a repository ``uv.lock``.

    The names are :func:`~senselab.utils.reproduce.locks.uv_lock_torch_family` of ``roots``, the set the
    integrity check compares against; each gets the one linux x86_64 wheel for the env's Python.

    Args:
        env: A uv-export env with a torch lock.
        base_dir: The manifest repository.
        uv_lock: The checkout's ``uv.lock``.
        roots: The torch, torchvision and torchaudio names the env installs.

    Returns:
        An empty argv: nothing is run.

    Raises:
        ReproduceLockError: If the env has no torch lock, a name is missing from ``uv.lock``, or a name has
            no single matching wheel.
    """
    if env.torch is None:
        raise ReproduceLockError("env has no torch lock")
    raw = uv_lock.read_bytes()
    packages: list[dict[str, Any]] = tomllib.loads(raw.decode()).get("package", [])
    by_name: dict[str, list[dict[str, Any]]] = {}
    for package in packages:
        by_name.setdefault(normalize_name(package["name"]), []).append(package)
    cp = "cp" + env.python.replace(".", "")
    lines = []
    for name in sorted(uv_lock_torch_family(packages, roots)):
        if name not in by_name:
            raise ReproduceLockError(f"{name}: not in uv.lock")
        wheel = _linux_wheel(name, by_name[name], cp)
        lines.append(f"{name} @ {wheel['url']} \\\n    --hash={wheel['hash']}\n")
    fields = {
        "source": "uv.lock",
        "python": env.python,
        "roots": " ".join(sorted({normalize_name(r) for r in roots})),
        "uv-lock-sha256": hashlib.sha256(raw).hexdigest(),
    }
    _write(base_dir, env.torch.lock, write_header("".join(lines), fields))
    return []


def body_lock_from_requirements(env: EnvSpec, base_dir: Path) -> list[str]:
    """Write ``env.body.lock`` compiled from the sidecar ``requirements.in``, with the digest header.

    The torch family (from the first resolve), the ``sdist_allow`` names and ``extra_no_emit`` are each
    left out with ``--no-emit-package``.

    Args:
        env: A requirements-derived env; with a torch lock, its ``first_resolve`` must be written.
        base_dir: The manifest repository.

    Returns:
        The argv run, with ``base_dir`` as its working directory.

    Raises:
        ReproduceLockError: If the body is not requirements-derived, uv is missing or the compile fails.
    """
    argv = [uv_executable(), "pip", "compile", _requirements_in(env), "--universal", "--generate-hashes", "--no-header"]
    argv += _compile_inputs(env) + _no_emit([*_resolve_family(env, base_dir), *_held_out(env)])
    fields = {"requirements-sha256": requirements_digest_for(env, base_dir), "python": env.python}
    return _emit(base_dir, env.body.lock, argv, base_dir, fields)


def body_lock_from_uvlock_export(
    env: EnvSpec,
    base_dir: Path,
    project_dir: Path,
    *,
    extras: Sequence[str] = (),
    groups: Sequence[str] = (),
) -> list[str]:
    """Write ``env.body.lock`` as a hashed ``uv export`` of the project's ``uv.lock``.

    The ``uv.lock`` torch closure of the torch, torchvision and torchaudio names in ``env.torch.lock``
    (written first), the ``sdist_allow`` names and ``extra_no_emit`` are each left out with
    ``--no-emit-package``.

    Args:
        env: A uv-export env.
        base_dir: The manifest repository.
        project_dir: The checkout holding ``pyproject.toml`` and ``uv.lock``; the command's working directory.
        extras: Project extras to include.
        groups: Dependency groups to include.

    Returns:
        The argv run; record it as ``body.export_cmd``.

    Raises:
        ReproduceLockError: If ``torch.lock`` holds no torch root, uv is missing or the export fails.
    """
    family: set[str] = set()
    if env.torch is not None:
        roots = sorted(TORCH_NAMES & set(_lock_names(base_dir, env.torch.lock)))
        if not roots:
            raise ReproduceLockError(f"{env.torch.lock} holds no torch, torchvision or torchaudio")
        packages = tomllib.loads((project_dir / "uv.lock").read_text()).get("package", [])
        family = uv_lock_torch_family(packages, roots)
    argv = [uv_executable(), "export", "--frozen", "--no-emit-project", "--no-dev", "--no-header"]
    for extra in extras:
        argv += ["--extra", extra]
    for group in groups:
        argv += ["--group", group]
    argv += _no_emit([*sorted(family), *_held_out(env)])
    return _emit(base_dir, env.body.lock, argv, project_dir, {"source": "uv-export", "python": env.python})


def overlay_lock(
    kind: Literal["test", "notebook_tools"],
    env: EnvSpec,
    base_dir: Path,
    *,
    project_dir: Path | None = None,
    requirements: str | None = None,
    group: str = "dev",
) -> list[str]:
    """Write an overlay lock: the project's test group, or notebook tools constrained by the base locks.

    Every name in ``torch.lock`` and ``body.lock`` and every name held out of the base (``sdist_allow``,
    ``extra_no_emit``) is left out, so the overlay holds only names the base lacks. The base venv
    provides or deliberately lacks the others; for notebook tools the constraints still bound them.

    Args:
        kind: ``test`` exports ``group`` from the project's ``uv.lock``; ``notebook_tools`` compiles
            ``requirements`` with ``torch.lock`` and ``body.lock`` as ``--constraints``.
        env: The env; its ``test`` or ``notebook_tools`` lock is written, and its base locks must exist.
        base_dir: The manifest repository.
        project_dir: The checkout, for ``test``.
        requirements: The notebook tools' requirements file relative to ``base_dir``, for ``notebook_tools``.
        group: The dependency group holding the test tools.

    Returns:
        The argv run.

    Raises:
        ReproduceLockError: If the env has no such overlay, its input is missing, uv is missing or uv fails.
    """
    ref = env.test if kind == "test" else env.notebook_tools
    if ref is None:
        raise ReproduceLockError(f"env has no {kind} lock")
    if kind == "test":
        if project_dir is None:
            raise ReproduceLockError("a test overlay needs project_dir")
        cwd = project_dir
        argv = [uv_executable(), "export", "--frozen", "--no-emit-project", "--only-group", group, "--no-header"]
    else:
        if requirements is None:
            raise ReproduceLockError("a notebook_tools overlay needs its requirements file")
        cwd = base_dir
        argv = [uv_executable(), "pip", "compile", requirements, "--universal", "--generate-hashes", "--no-header"]
        argv += ["--python-version", env.python, "--exclude-newer", exclude_newer_timestamp(env.exclude_newer)]
        argv += ["--index-url", env.body.index_url]
        for base in ([env.torch.lock] if env.torch else []) + [env.body.lock]:
            argv += ["--constraints", base]
    torch_names = _lock_names(base_dir, env.torch.lock) if env.torch else []
    argv += _no_emit([*torch_names, *_lock_names(base_dir, env.body.lock), *_held_out(env)])
    return _emit(base_dir, ref.lock, argv, cwd, {"source": kind, "python": env.python})
