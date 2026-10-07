"""The install engine: build a step's venv from its verified locks, check it, and prepare R libraries.

`prefetch_env` runs, in order: the integrity check (`locks.check_integrity`); a copy of every
verified file into a private temp dir, each copy re-checked against the verified sha256; ``uv venv
--python <python_exact>``; stage 1, the torch lock (``--no-deps --require-hashes``, plus
``--index-url`` for an index-compiled lock); stage 2, the body (``--no-deps --require-hashes
--only-binary :all: --index-url <body.index_url>``); stage 3, each ``sdist_allow`` wheel, reused or
built through `build.ensure_wheel`, copied with `build.stage_wheel` and installed ``--no-deps``;
then each overlay (``test``, ``notebook_tools``) into its own derived venv holding the base locks
plus the overlay; and finally ``installed.json``, the parsed ``uv pip freeze``, which must equal
the locks. Every install reads a copy, never a file in the manifest repository or the wheelhouse.
Every uv call passes ``--no-config`` and runs with `build.base_environment`; nothing else of the
caller's environment reaches uv.

Venvs live at ``<cache>/venvs/repro-<analysis>-<env_key8>`` (derived:
``…+<overlay sha256[:8]>``). Each has a ``.reproduce-venv.json`` marker naming its key, overlay, wheel
sha256s, the sha256 of ``installed.json`` and the result of the checks. An installed venv whose marker
matches is reused; anything else is removed and rebuilt, under a ``flock`` on the sidecar
``<venv>.lock``. A venv is complete (`venv_checked`) only once ``uv pip check`` and the smoke imports
have passed, which `prefetch_env` runs only when given a profile; a run must refuse an unchecked venv.

R envs (kinds ``r`` and ``ir_notebook``) are prepared by `prefetch_r` under the profile's R module,
outside the network namespace, from the Posit Package Manager snapshot of ``env.r_snapshot``. It
writes ``<r_lib>/MODULE_ENV.json`` (the modules' ``PATH`` and ``LD_LIBRARY_PATH``) and
``<r_lib>/SESSION.json`` (``sessionInfo()``, ``.libPaths()`` and installed versions), and fails
unless every declared ``r_packages`` version is the installed one. `r_env_vars` gives a step's R
variables in the merge order: module ``LD_LIBRARY_PATH`` entries first, then native lib dirs.
"""

from __future__ import annotations

import contextlib
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

from packaging.markers import Marker, default_environment

from senselab.utils.reproduce import sandbox
from senselab.utils.reproduce.build import (
    MODULES_INIT,
    SYSTEM_PATH,
    base_environment,
    ca_bundle,
    ensure_wheel,
    file_lock,
    reuse_wheel,
    stage_wheel,
    torch_index_url,
)
from senselab.utils.reproduce.lockgen import find_uv
from senselab.utils.reproduce.locks import (
    ReproduceLockError,
    check_integrity,
    entry_version,
    lock_entries,
    normalize_name,
    parse_freeze,
)
from senselab.utils.reproduce.spec import EnvSpec, LockRef, PipCheckExpected, Profile, RPackage, StepSpec, render

ENV_KEY_SCHEMA = "reproduce-env-1"
VENVS = "venvs"
MARKER = ".reproduce-venv.json"
INSTALLED = "installed.json"
OVERLAYS = ("test", "notebook_tools")
PPM_CRAN = "https://packagemanager.posit.co/cran"
SESSION = "SESSION.json"
MODULE_ENV = "MODULE_ENV.json"
SMOKE_TIMEOUT = 600
R_TIMEOUT = 4 * 3600
_TAIL = 4000
_ANALYSIS = re.compile(r"^[A-Za-z0-9_.-]+$")
_MODULE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_ENV_MODULE = re.compile(r"^[A-Za-z0-9_.+-]+(/[A-Za-z0-9_.+-]+)*$")
_R_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9.]*$")
_R_VERSION = re.compile(r"^[0-9][0-9.-]*$")
_CUDA_LOCAL = re.compile(r"\+cu\d+$")
_PIP_CHECK_LINE = re.compile(r"^The package `([^`]+)` (.*)$")

Runner = Callable[[list[str], dict[str, str], Path], "subprocess.CompletedProcess[str]"]


class VenvResult(NamedTuple):
    """One built or reused venv: path, key, overlay sha256 (derived venvs), reuse flag, ``{name: version}``.

    ``checked`` is `venv_checked`: both checks passed and are recorded in the marker.
    """

    path: Path
    env_key: str
    overlay_sha256: str | None
    reused: bool
    packages: dict[str, str]
    checked: bool


class PrefetchResult(NamedTuple):
    """What `prefetch_env` returns.

    ``wheels_lock_sha256`` is the sha256 of ``wheels.lock`` after the prefetch (``None`` for an env with
    no sdists). ``record`` is true when it differs from the manifest's ``env.wheels.lock_sha256`` (or the
    manifest has none): the caller must then write it into the manifest, or the next integrity check
    refuses the env. ``built`` names the sdists built in this call.
    """

    base: VenvResult
    derived: dict[str, VenvResult]
    wheels_lock_sha256: str | None
    record: bool
    built: tuple[str, ...]


class Violation(NamedTuple):
    """One normalised ``uv pip check`` incompatibility.

    ``kind`` is ``requirement`` (a ``requires …`` line), ``platform`` (a line about the platform)
    or ``other``; ``detail`` is the line after the package name, backticks removed and whitespace
    collapsed.
    """

    package: str
    kind: str
    detail: str


class RResult(NamedTuple):
    """What `prefetch_r` returns: the library, its two JSON files and their sha256s, and the reuse flag."""

    r_lib: Path
    session: Path
    session_sha256: str
    module_env: Path
    module_env_sha256: str
    reused: bool


def local_free(version: str) -> str:
    """Return ``version`` without a ``+cuXXX`` local tag, as `locks.parse_freeze` reports it."""
    return _CUDA_LOCAL.sub("", version)


def env_key(env: EnvSpec) -> str:
    """Return the identity of an env's base venv, from its inputs only.

    Args:
        env: The step's env.

    Returns:
        A hex sha256 over the torch and body lock sha256s, each sdist's name, URL, sha256, patch
        sha256s, build lock sha256, build variables, build modules and arch list, ``python_exact``
        and ``max_cuda``. Wheel sha256s, overlays and file locations are not part of it.
    """
    payload = {
        "schema": ENV_KEY_SCHEMA,
        "torch_lock_sha256": None if env.torch is None else env.torch.lock_sha256,
        "body_lock_sha256": env.body.lock_sha256,
        "sdist": [
            {
                "name": s.name,
                "url": s.url,
                "sha256": s.sha256,
                "patch_sha256s": [p.sha256 for p in s.patches],
                "build_env_lock_sha256": s.build_env.lock_sha256,
                "build_env_vars": dict(sorted(s.build_env_vars.items())),
                "build_modules": list(s.build_modules),
                "torch_cuda_arch_list": s.torch_cuda_arch_list,
            }
            for s in env.sdist_allow
        ],
        "python_exact": env.python_exact,
        "max_cuda": env.max_cuda,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _analysis(analysis: str) -> str:
    if not _ANALYSIS.match(analysis) or analysis in (".", ".."):
        raise ValueError(f"analysis name must be a plain word: {analysis!r}")
    return analysis


def venv_dir(cache: Path, analysis: str, env: EnvSpec) -> Path:
    """Return ``<cache>/venvs/repro-<analysis>-<env_key[:8]>``."""
    return cache / VENVS / f"repro-{_analysis(analysis)}-{env_key(env)[:8]}"


def derived_venv_dir(cache: Path, analysis: str, env: EnvSpec, overlay: LockRef) -> Path:
    """Return ``<cache>/venvs/repro-<analysis>-<env_key[:8]>+<overlay lock sha256[:8]>``."""
    base = venv_dir(cache, analysis, env)
    return base.with_name(f"{base.name}+{overlay.lock_sha256[:8]}")


def install_argv(uv: str, python: Path, lock: Path, index_url: str | None, binary_only: bool) -> list[str]:
    """The argv of a hashed lock install: ``uv pip install --no-config --python <py> --no-deps --require-hashes …``.

    Args:
        uv: The uv executable.
        python: The venv interpreter.
        lock: The verified copy of the lock.
        index_url: Passed as ``--index-url`` when given.
        binary_only: Add ``--only-binary :all:``.

    Returns:
        The argv, ending ``-r <lock>``.
    """
    argv = [uv, "pip", "install", "--no-config", "--python", str(python), "--no-deps", "--require-hashes"]
    if binary_only:
        argv += ["--only-binary", ":all:"]
    if index_url is not None:
        argv += ["--index-url", index_url]
    return [*argv, "-r", str(lock)]


def _run(argv: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run one command and capture its output."""
    return subprocess.run(argv, env=env, capture_output=True, text=True, check=False)


def _run_ok(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    result = _run(argv, env)
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()[-_TAIL:]
        raise ReproduceLockError(f"{' '.join(argv[:3])} failed ({result.returncode}): {output}")
    return result


def _flock(path: Path) -> contextlib.AbstractContextManager[None]:
    return file_lock(path, shared=False)


def _write_json(path: Path, data: object) -> str:
    text = json.dumps(data, indent=2, sort_keys=True) + "\n"
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
        mask = os.umask(0)
        os.umask(mask)
        os.chmod(tmp, 0o666 & ~mask)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return hashlib.sha256(text.encode()).hexdigest()


def copy_verified(base_dir: Path, verified: Mapping[str, str], dest: Path) -> None:
    """Copy each verified file into ``dest`` and check the copy against the verified sha256.

    Args:
        base_dir: The manifest repository.
        verified: `locks.check_integrity`'s ``{relative path: sha256}``.
        dest: A private directory; the copies keep their relative paths.

    Raises:
        ReproduceLockError: If a copy's bytes differ from what the integrity check hashed.
    """
    for rel, sha in verified.items():
        data = (base_dir / rel).read_bytes()
        actual = hashlib.sha256(data).hexdigest()
        if actual != sha:
            raise ReproduceLockError(f"copy of {rel}: sha256 {actual} != verified {sha}; it changed after the check")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _marker_environment(env: EnvSpec) -> dict[str, str]:
    markers = {k: str(v) for k, v in default_environment().items()}
    markers.update(python_version=env.python, python_full_version=env.python_exact, extra="")
    return markers


def declared_packages(env: EnvSpec, locks: Iterable[str], wheels: Iterable[Path]) -> dict[str, str]:
    """Return ``{name: version}`` that a venv built from these locks and wheels must hold.

    Args:
        env: The step's env; its Python version decides the markers.
        locks: Verified lock texts, in install order; entries whose marker is false are skipped.
        wheels: The stage 3 wheel files.

    Returns:
        Normalised names to versions, ``+cuXXX`` dropped, as `locks.parse_freeze` reports them.

    Raises:
        ReproduceLockError: If two applicable entries pin one name to different versions.
    """
    markers = _marker_environment(env)
    out: dict[str, str] = {}

    def add(name: str, version: str) -> None:
        if out.setdefault(name, version) != version:
            raise ReproduceLockError(f"{name} is pinned to {out[name]} and {version}")

    for text in locks:
        for entry in lock_entries(text):
            if not entry.marker or Marker(entry.marker).evaluate(markers):
                add(entry.name, local_free(entry_version(entry)))
    for wheel in wheels:
        parts = wheel.name.split("-")
        add(normalize_name(parts[0]), local_free(parts[1]))
    return out


def _diff(installed: Mapping[str, str], declared: Mapping[str, str]) -> list[str]:
    names = sorted(set(installed) | set(declared))
    differ = [n for n in names if installed.get(n) != declared.get(n)]
    return [f"{n}: installed {installed.get(n)}, locked {declared.get(n)}" for n in differ]


def _wheels_location(env: EnvSpec, base_dir: Path, wheels_lock: Path | None) -> tuple[Path | None, str | None]:
    if not env.sdist_allow:
        return None, None
    if env.wheels is not None:
        if wheels_lock is not None and wheels_lock != base_dir / env.wheels.lock:
            raise ValueError("wheels_lock is given and env.wheels names another file")
        return base_dir / env.wheels.lock, env.wheels.lock_sha256
    if wheels_lock is None:
        raise ValueError("an env with sdist_allow needs env.wheels or a wheels_lock path to record wheels in")
    if wheels_lock.exists():
        raise ReproduceLockError(
            f"{wheels_lock} exists but the manifest records no wheels lock; record its sha256 in env.wheels first"
        )
    return wheels_lock, None


class _Wheels:
    """Stage 3 inputs: the recorded wheel sha256s, and the wheels reused or built, copied and checked on first use."""

    def __init__(
        self,
        env: EnvSpec,
        base_dir: Path,
        wheelhouse: Path,
        lock: Path | None,
        expected: str | None,
        rebuild: bool,
        dest: Path,
    ) -> None:
        self.env, self.base_dir, self.wheelhouse, self.lock = env, base_dir, wheelhouse, lock
        self.lock_sha256, self.rebuild, self.dest = expected, rebuild, dest
        self.built: list[str] = []
        self.shas: dict[str, str] = {}
        self._copies: list[Path] | None = None

    def recorded(self) -> dict[str, str] | None:
        """``{sdist name: wheel sha256}`` from the verified ``wheels.lock`` lines; ``None`` if a build is due."""
        if self.rebuild:
            return None
        if self._copies is not None:
            return dict(self.shas)
        out = {}
        for sdist in self.env.sdist_allow:
            assert self.lock is not None
            found = reuse_wheel(
                self.lock,
                sdist,
                self.wheelhouse,
                self.env.torch,
                python=self.env.python_exact,
                expected_sha256=self.lock_sha256,
            )
            if found is None:
                return None
            out[sdist.name] = found[1]
        return out

    def copies(self) -> list[Path]:
        if self._copies is None:
            self._copies = []
            for sdist in self.env.sdist_allow:
                assert self.lock is not None
                result = ensure_wheel(
                    sdist,
                    self.base_dir,
                    self.wheelhouse,
                    self.lock,
                    self.env.torch,
                    python=self.env.python_exact,
                    rebuild=self.rebuild,
                    expected_sha256=self.lock_sha256,
                )
                self.lock_sha256 = result.wheels_lock_sha256
                if result.built:
                    self.built.append(sdist.name)
                self.shas[sdist.name] = result.wheel_sha256
                dest = self.dest / result.wheel_sha256
                self._copies.append(stage_wheel(result.wheel, result.wheel_sha256, dest))
        return self._copies


def checks_digest(env: EnvSpec) -> str:
    """Digest of what the venv checks compare against: ``pip_check_expected`` and ``smoke_imports``."""
    payload = {
        "pip_check_expected": [e.model_dump() for e in env.pip_check_expected],
        "smoke_imports": list(env.smoke_imports),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _read_marker(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((path / MARKER).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _installed_ok(path: Path, marker: Mapping[str, Any]) -> dict[str, Any] | None:
    try:
        data = (path / INSTALLED).read_bytes()
    except OSError:
        return None
    if hashlib.sha256(data).hexdigest() != marker.get("installed_sha256"):
        return None
    installed: dict[str, Any] = json.loads(data)
    return installed


def venv_checked(path: Path, env: EnvSpec, overlay: LockRef | None = None) -> bool:
    """Whether a venv is complete for ``env``: installed for its key and overlay, and both checks passed.

    A run (B12) must refuse a venv for which this is false. `prefetch_env` marks the checks only when
    given a profile.

    Args:
        path: The venv.
        env: The step's env.
        overlay: The overlay lock of a derived venv.

    Returns:
        True when the marker names ``env_key(env)`` and the overlay sha256, ``installed.json`` matches the
        marker, and ``pip_check_ok`` and ``smoke_ok`` were recorded for the current `checks_digest`.
    """
    marker = _read_marker(path)
    if marker is None or _installed_ok(path, marker) is None:
        return False
    checks = marker.get("checks") or {}
    return (
        marker.get("env_key") == env_key(env)
        and marker.get("overlay_sha256") == (None if overlay is None else overlay.lock_sha256)
        and checks.get("digest") == checks_digest(env)
        and checks.get("pip_check_ok") is True
        and checks.get("smoke_ok") is True
    )


def _build_or_reuse(
    uv: str,
    env: EnvSpec,
    path: Path,
    inputs: Path,
    wheels: _Wheels,
    overlay: LockRef | None,
    run_env: dict[str, str],
) -> VenvResult:
    key = env_key(env)
    overlay_sha = None if overlay is None else overlay.lock_sha256
    with _flock(path.with_name(path.name + ".lock")):
        marker = _read_marker(path)
        if marker is not None and (marker.get("env_key"), marker.get("overlay_sha256")) == (key, overlay_sha):
            installed = _installed_ok(path, marker)
            if installed is not None and marker.get("wheels") == wheels.recorded():
                return VenvResult(path, key, overlay_sha, True, installed["packages"], venv_checked(path, env, overlay))
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
        _run_ok([uv, "venv", "--no-config", "--python", env.python_exact, str(path)], run_env)
        python = path / "bin" / "python"
        texts = []
        if env.torch is not None:
            torch_lock = inputs / env.torch.lock
            text = torch_lock.read_text()
            _run_ok(install_argv(uv, python, torch_lock, torch_index_url(env.torch, text), False), run_env)
            texts.append(text)
        body = inputs / env.body.lock
        _run_ok(install_argv(uv, python, body, env.body.index_url, True), run_env)
        texts.append(body.read_text())
        copies = wheels.copies()
        for wheel in copies:
            _run_ok([uv, "pip", "install", "--no-config", "--python", str(python), "--no-deps", str(wheel)], run_env)
        if overlay is not None:
            overlay_lock = inputs / overlay.lock
            _run_ok(install_argv(uv, python, overlay_lock, env.body.index_url, True), run_env)
            texts.append(overlay_lock.read_text())
        freeze = _run_ok([uv, "pip", "freeze", "--no-config", "--python", str(python)], run_env).stdout
        packages = parse_freeze(freeze)
        problems = _diff(packages, declared_packages(env, texts, copies))
        if problems:
            raise ReproduceLockError(f"{path.name}: the installed packages differ from the locks: {problems}")
        lines = sorted(line.strip() for line in freeze.splitlines() if line.strip())
        installed = {
            "env_key": key,
            "overlay_sha256": overlay_sha,
            "python_exact": env.python_exact,
            "wheels": dict(wheels.shas),
            "freeze": lines,
            "packages": packages,
        }
        installed_sha = _write_json(path / INSTALLED, installed)
        _write_json(
            path / MARKER,
            {
                "env_key": key,
                "overlay_sha256": overlay_sha,
                "wheels": dict(wheels.shas),
                "installed_sha256": installed_sha,
                "checks": {"digest": checks_digest(env), "pip_check_ok": False, "smoke_ok": False},
            },
        )
        return VenvResult(path, key, overlay_sha, False, packages, False)


def _check_venv(venv: VenvResult, env: EnvSpec, profile: Profile, overlay: LockRef | None) -> VenvResult:
    """Run `check_pip` and `smoke` on a venv unless its marker already records both for the current digest."""
    with _flock(venv.path.with_name(venv.path.name + ".lock")):
        if venv_checked(venv.path, env, overlay):
            return venv._replace(checked=True)
        marker = _read_marker(venv.path)
        if marker is None or marker.get("env_key") != venv.env_key:
            raise ReproduceLockError(f"{venv.path.name} changed while it was being checked")
        check_pip(venv.path, env)
        failures = smoke(venv.path, env.smoke_imports, profile)
        if failures:
            raise ReproduceLockError(f"smoke imports failed in {venv.path.name}: {failures}")
        marker["checks"] = {"digest": checks_digest(env), "pip_check_ok": True, "smoke_ok": True}
        _write_json(venv.path / MARKER, marker)
    return venv._replace(checked=True)


def prefetch_env(
    env: EnvSpec,
    base_dir: Path,
    cache: Path,
    *,
    analysis: str,
    wheelhouse: Path,
    uv_lock: Path | None = None,
    profile: Profile | None = None,
    wheels_lock: Path | None = None,
    rebuild: bool = False,
) -> PrefetchResult:
    """Build (or reuse) an env's base venv and its derived venvs from verified copies of its locks.

    A venv is reused when its marker names its key and overlay, its ``installed.json`` matches the
    marker's sha256, and its recorded wheel sha256s equal the verified ``wheels.lock`` lines. Anything
    else, or ``rebuild``, rebuilds it.

    Args:
        env: The step's env.
        base_dir: The manifest repository; every path in ``env`` is relative to it.
        cache: The reproduce cache; venvs go under ``<cache>/venvs``.
        analysis: The manifest name, used in venv names.
        wheelhouse: Where stage 3 wheels are stored.
        uv_lock: The checkout's ``uv.lock``, for a uv-export body with a torch lock.
        profile: When given, each venv not yet checked is checked: ``uv pip check`` must equal
            ``pip_check_expected`` (`check_pip`) and ``smoke_imports`` must import in the sandbox (`smoke`);
            the marker then records both. Without it no venv is marked checked (`venv_checked` stays false).
        wheels_lock: Where to record wheels when ``env.wheels`` is unset (the first prefetch); the file must
            not exist yet.
        rebuild: Build every sdist again (`build.ensure_wheel`'s ``rebuild``) and rebuild every venv.

    Returns:
        A `PrefetchResult`.

    Raises:
        ReproduceLockError: If the integrity check, a copy check, an install, the freeze comparison or a
            check fails, or ``wheels_lock`` exists while ``env.wheels`` is unset.
        ValueError: If an env with sdists has nowhere to record wheels, or ``wheels_lock`` contradicts ``env.wheels``.
    """
    lock_path, expected = _wheels_location(env, base_dir, wheels_lock)
    verified = check_integrity(env, base_dir, uv_lock)
    uv = find_uv()
    with tempfile.TemporaryDirectory(prefix="reproduce-env-") as tmp:
        work = Path(tmp)
        inputs = work / "inputs"
        copy_verified(base_dir, verified, inputs)
        run_env = base_environment(uv, work)
        wheels = _Wheels(env, base_dir, wheelhouse, lock_path, expected, rebuild, work / "wheels")
        base = _build_or_reuse(uv, env, venv_dir(cache, analysis, env), inputs, wheels, None, run_env)
        derived = {}
        for name in OVERLAYS:
            ref: LockRef | None = getattr(env, name)
            if ref is not None:
                path = derived_venv_dir(cache, analysis, env, ref)
                derived[name] = _build_or_reuse(uv, env, path, inputs, wheels, ref, run_env)
    if profile is not None:
        base = _check_venv(base, env, profile, None)
        derived = {name: _check_venv(v, env, profile, getattr(env, name)) for name, v in derived.items()}
    lock_sha = None
    if lock_path is not None and lock_path.is_file():
        lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    record = lock_sha is not None and lock_sha != expected
    return PrefetchResult(base, derived, lock_sha, record, tuple(wheels.built))


def parse_pip_check(text: str) -> list[Violation]:
    """Parse ``uv pip check`` output into sorted, normalised violations.

    Args:
        text: The command's output; lines other than ``The package `…` …`` are ignored.

    Returns:
        Violations sorted by package, kind and detail.
    """
    found = []
    for line in text.splitlines():
        match = _PIP_CHECK_LINE.match(line.strip())
        if match:
            found.append(_violation(match.group(1), None, match.group(2)))
    return sorted(found)


def _violation(package: str, kind: str | None, detail: str) -> Violation:
    detail = " ".join(detail.replace("`", "").split())
    if kind is None:
        kind = "requirement" if detail.startswith("requires ") else "platform" if "platform" in detail else "other"
    return Violation(normalize_name(package), kind, detail)


def compare_pip_check(
    found: Iterable[Violation], expected: Iterable[PipCheckExpected]
) -> tuple[list[Violation], list[Violation]]:
    """Compare violations with the declared deviations.

    Args:
        found: `pip_check`'s violations.
        expected: The env's ``pip_check_expected``; package and detail are normalised as `parse_pip_check` does.

    Returns:
        ``(undeclared, declared but absent)``; both empty means the venv passes.
    """
    found = list(found)
    wanted = [_violation(e.package, e.kind, e.detail) for e in expected]
    return [v for v in found if v not in wanted], [w for w in wanted if w not in found]


def pip_check(venv: Path) -> list[Violation]:
    """Run ``uv pip check`` on a venv.

    Args:
        venv: The venv.

    Returns:
        The violations, as `parse_pip_check` gives them.

    Raises:
        ReproduceLockError: If uv fails for another reason than incompatibilities.
    """
    uv = find_uv()
    with tempfile.TemporaryDirectory(prefix="reproduce-check-") as tmp:
        argv = [uv, "pip", "check", "--no-config", "--python", str(venv / "bin" / "python")]
        result = _run(argv, base_environment(uv, Path(tmp)))
    found = parse_pip_check(result.stdout + "\n" + result.stderr)
    if result.returncode not in (0, 1) or (result.returncode == 1) != bool(found):
        output = (result.stdout + result.stderr).strip()[-_TAIL:]
        raise ReproduceLockError(f"uv pip check on {venv} failed ({result.returncode}): {output}")
    return found


def check_pip(venv: Path, env: EnvSpec) -> None:
    """Fail unless the venv's ``uv pip check`` violations equal ``env.pip_check_expected`` exactly.

    Raises:
        ReproduceLockError: Naming the undeclared violations and the declared ones that did not occur.
    """
    unexpected, missing = compare_pip_check(pip_check(venv), env.pip_check_expected)
    if unexpected or missing:
        raise ReproduceLockError(f"uv pip check on {venv.name}: undeclared {unexpected}; declared but absent {missing}")


def _smoke_runner(argv: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True, check=False, timeout=SMOKE_TIMEOUT)


def _r_runner(argv: list[str], env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, env=env, cwd=cwd, capture_output=True, text=True, check=False, timeout=R_TIMEOUT)


def _smoke_env(venv: Path, tmp: Path, profile: Profile | None) -> dict[str, str]:
    user = dict(profile.env) if profile is not None else {}
    reserved = sorted(k for k in user if k in sandbox.RESERVED or k.startswith(sandbox.RESERVED_PREFIXES))
    if reserved:
        raise ValueError(f"profile env may not set driver-owned variables: {reserved}")
    env = {**user, **sandbox.FIXED_KEYED, **sandbox.OFFLINE}
    env |= {v: sandbox.PROXY for v in sandbox.PROXY_VARS} | {v.upper(): sandbox.PROXY for v in sandbox.PROXY_VARS}
    env |= {v: "" for v in sandbox.NO_PROXY_VARS}
    for sub in ("home", "cache"):
        (tmp / sub).mkdir(exist_ok=True)
    env |= {
        "PATH": ":".join([f"{venv}/bin", *sandbox.SYSTEM_PATH]),
        "HOME": str(tmp / "home"),
        "TMPDIR": str(tmp),
        "XDG_CACHE_HOME": str(tmp / "cache"),
    }
    return env


def smoke(
    venv: Path,
    modules: Sequence[str],
    profile: Profile | None = None,
    *,
    step: StepSpec | None = None,
    runner: Runner | None = None,
) -> list[str]:
    """Import each module with the venv's interpreter inside the sandbox, offline, one process per module.

    Args:
        venv: The venv; mounted read-only.
        modules: Dotted module names.
        profile: Its ``env`` map is passed to the import; with ``step``, the step's whole env
            (`sandbox.build_env`) is used instead.
        step: The step whose env to use, with ``profile``.
        runner: Runs ``(argv, env, cwd)``; defaults to ``subprocess.run`` with captured output and a
            :data:`SMOKE_TIMEOUT` second timeout.

    Returns:
        ``"<module>: <output tail>"`` for each import that failed; empty when all succeeded. Each import
        runs with an empty private working directory, so nothing in the caller's directory is importable.

    Raises:
        ValueError: If a module name is not a dotted identifier.
        ReproduceLockError: ``E_SMOKE_TIMEOUT`` if an import outlives the timeout.
    """
    bad = [m for m in modules if not _MODULE_NAME.match(m)]
    if bad:
        raise ValueError(f"not module names: {bad}")
    run = runner or _smoke_runner
    venv = venv.absolute()
    failures = []
    with tempfile.TemporaryDirectory(prefix="reproduce-smoke-") as name:
        tmp = Path(name).resolve()
        cwd = tmp / "cwd"
        cwd.mkdir()
        if step is not None and profile is not None:
            paths = {"out": tmp / "out", "tmp": tmp, "checkout": tmp / "checkout", "venv": venv, "cache": tmp / "cache"}
            env = sandbox.build_env(step, profile, paths)[0]
        else:
            env = _smoke_env(venv, tmp, profile)
        for module in modules:
            argv = sandbox.sandbox_argv([str(venv / "bin" / "python"), "-c", f"import {module}"], [], [tmp], [venv])
            try:
                result = run(argv, env, cwd)
            except subprocess.TimeoutExpired as error:
                raise ReproduceLockError(
                    f"E_SMOKE_TIMEOUT: import {module} in {venv.name} ran past {error.timeout}s"
                ) from error
            if result.returncode != 0:
                failures.append(f"{module}: {(result.stdout + result.stderr).strip()[-_TAIL:]}")
    return failures


def r_snapshot_url(date: datetime.date) -> str:
    """The Posit Package Manager CRAN snapshot of a date."""
    return f"{PPM_CRAN}/{date.isoformat()}"


def r_script(packages: Sequence[RPackage]) -> str:
    """The R script prefetch runs with ``Rscript --vanilla <script> <r_lib> <out dir> <repo> <kernel prefix>``.

    It installs the packages and their ``Depends``/``Imports``/``LinkingTo`` from ``<repo>`` into
    ``<r_lib>``, loads each, runs ``IRkernel::installspec(prefix=…)`` when a kernel prefix is given,
    and writes ``installed.tsv``, ``lib_paths.txt``, ``session_info.txt`` and ``r_version.txt`` to
    the out dir.

    Raises:
        ValueError: If a package name or version is not plain.
    """
    bad = [p for p in packages if not _R_NAME.match(p.name) or not _R_VERSION.match(p.version)]
    if bad:
        raise ValueError(f"R package names and versions must be plain: {bad}")
    names = ", ".join(f'"{p.name}"' for p in packages)
    return "\n".join(
        [
            "args <- commandArgs(trailingOnly = TRUE)",
            "lib <- args[[1]]",
            "out <- args[[2]]",
            "repo <- args[[3]]",
            'prefix <- if (length(args) >= 4) args[[4]] else ""',
            f"pkgs <- c({names})",
            "options(repos = c(CRAN = repo))",
            "dir.create(lib, recursive = TRUE, showWarnings = FALSE)",
            'install.packages(pkgs, lib = lib, dependencies = c("Depends", "Imports", "LinkingTo"))',
            "for (p in pkgs) suppressPackageStartupMessages(library(p, character.only = TRUE))",
            "if (nzchar(prefix)) IRkernel::installspec(user = FALSE, prefix = prefix)",
            'ip <- installed.packages(lib.loc = lib)[, c("Package", "Version"), drop = FALSE]',
            'write.table(ip, file.path(out, "installed.tsv"), sep = "\\t", quote = FALSE, row.names = FALSE,'
            " col.names = FALSE)",
            'writeLines(.libPaths(), file.path(out, "lib_paths.txt"))',
            'writeLines(capture.output(sessionInfo()), file.path(out, "session_info.txt"))',
            'writeLines(R.version.string, file.path(out, "r_version.txt"))',
            "",
        ]
    )


def _r_environment(work: Path) -> dict[str, str]:
    home, tmp = work / "home", work / "tmp"
    home.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(parents=True, exist_ok=True)
    env = {"PATH": SYSTEM_PATH, "HOME": str(home), "TMPDIR": str(tmp), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
    bundle = ca_bundle()
    if bundle is not None:
        env["SSL_CERT_FILE"] = bundle
    return env


def _in_bash(script: str, *args: str) -> list[str]:
    return ["bash", "--noprofile", "--norc", "-c", script, "bash", *args]


def _checked(result: subprocess.CompletedProcess[str], what: str) -> str:
    if result.returncode != 0:
        output = (result.stdout + result.stderr).strip()[-_TAIL:]
        raise ReproduceLockError(f"{what} failed ({result.returncode}): {output}")
    return result.stdout


def _kernel(venv: Path) -> Path:
    return venv / "share" / "jupyter" / "kernels" / "ir" / "kernel.json"


def r_module_env_script(build_modules: Sequence[str] = ()) -> str:
    """The ``bash -c`` script that loads ``build_modules`` then the R module (``$1``) and prints ``env -0``."""
    return f'. {MODULES_INIT} && module load {_module_words(build_modules)}"$1" && exec env -0'


def r_install_script(build_modules: Sequence[str] = ()) -> str:
    """The ``bash -c`` script that loads the modules and runs the generated R script.

    Positional arguments: ``$1`` R module, ``$2`` r_lib, ``$3`` script, ``$4`` out dir, ``$5`` repository,
    ``$6`` kernel prefix (may be empty). ``R_LIBS`` and ``R_LIBS_USER`` are the library and
    ``R_LIBS_SITE`` is empty, so ``.libPaths()`` is the library then R's own.
    """
    return (
        f'. {MODULES_INIT} && module load {_module_words(build_modules)}"$1" && '
        'export R_LIBS="$2" R_LIBS_USER="$2" R_LIBS_SITE= && '
        'if [ -n "$6" ]; then export PATH="$6/bin:$PATH"; fi && '
        'exec Rscript --vanilla "$3" "$2" "$4" "$5" "$6"'
    )


def _module_words(modules: Sequence[str]) -> str:
    bad = [m for m in modules if not _ENV_MODULE.match(m)]
    if bad:
        raise ValueError(f"module names must be plain: {bad}")
    return "".join(f"{m} " for m in modules)


def _r_key(
    module: str, build_modules: Sequence[str], r_version: str, repo: str, declared: dict[str, str]
) -> dict[str, Any]:
    return {
        "module": module,
        "build_modules": list(build_modules),
        "r_version": r_version,
        "snapshot": repo,
        "declared": declared,
    }


def _r_reusable(session_path: Path, module_path: Path, key: Mapping[str, Any], kernel_ok: bool) -> bool:
    if not (session_path.is_file() and module_path.is_file() and kernel_ok):
        return False
    try:
        recorded = json.loads(session_path.read_text())
    except ValueError:
        return False
    return isinstance(recorded, dict) and recorded.get("key") == key


def prefetch_r(
    env: EnvSpec,
    cache: Path,
    profile: Profile,
    *,
    kind: str = "r",
    venv: Path | None = None,
    runner: Runner | None = None,
) -> RResult:
    """Install an env's R packages into its private library under the profile's R module, with network.

    Runs outside the network namespace. ``profile.r.build_modules`` and then ``profile.r.module`` are
    loaded with ``bash --noprofile --norc -c '. <MODULES_INIT> && module load …'``, so no dotfile is
    read, and with an allowlisted environment. The packages come from `r_snapshot_url` of
    ``env.r_snapshot``; ``R_LIBS`` and ``R_LIBS_USER`` are ``<r_lib>`` and ``R_LIBS_SITE`` is empty, so
    neither the module's site library nor a user library under ``HOME`` is seen. A library whose
    ``SESSION.json`` records the same key (modules, R version, snapshot, declared packages) is reused;
    otherwise ``SESSION.json`` is removed and the library rebuilt from empty.

    Args:
        env: The step's env; ``r_packages``, ``r_snapshot`` and ``r_lib`` are required.
        cache: The reproduce cache; fills ``{cache}`` in ``r_lib``.
        profile: Its ``r`` entry gives the modules and the R version to expect.
        kind: ``r`` or ``ir_notebook``; the latter installs the ``ir`` kernel spec into ``venv``.
        venv: The derived (notebook tools) venv, for ``ir_notebook``; its ``bin`` leads ``PATH``.
        runner: Runs ``(argv, env, cwd)``; defaults to ``subprocess.run`` with captured output and a
            :data:`R_TIMEOUT` second timeout.

    Returns:
        An `RResult`.

    Raises:
        ValueError: If the profile has no R module, the env lacks its R fields, or an ``ir_notebook``
            env has no venv or does not declare ``IRkernel``.
        ReproduceLockError: If a command fails, R is not ``profile.r.version``, the library is not first
            on ``.libPaths()``, the kernel spec is missing, or an installed version differs from the declared one.
    """
    if profile.r is None:
        raise ValueError("an R env needs profile.r")
    if not env.r_packages or env.r_snapshot is None or env.r_lib is None:
        raise ValueError("an R env needs r_packages, r_snapshot and r_lib")
    if kind == "ir_notebook":
        if "IRkernel" not in {p.name for p in env.r_packages}:
            raise ValueError("an ir_notebook env must declare IRkernel in r_packages")
        if venv is None:
            raise ValueError("an ir_notebook env needs the derived venv for its kernel spec")
    module, build_modules, r_version = profile.r.module, profile.r.build_modules, profile.r.version
    if not _ENV_MODULE.match(module):
        raise ValueError(f"R module {module!r} is not a plain module name")
    run = runner or _r_runner
    r_lib = Path(render(env.r_lib, {"cache": cache}))
    repo = r_snapshot_url(env.r_snapshot)
    declared = {p.name: p.version for p in env.r_packages}
    key = _r_key(module, build_modules, r_version, repo, declared)
    session_path, module_path = r_lib / SESSION, r_lib / MODULE_ENV
    kernel_ok = venv is None or kind != "ir_notebook" or _kernel(venv).is_file()
    with _flock(r_lib.with_name(r_lib.name + ".lock")):
        if _r_reusable(session_path, module_path, key, kernel_ok):
            return RResult(
                r_lib,
                session_path,
                hashlib.sha256(session_path.read_bytes()).hexdigest(),
                module_path,
                hashlib.sha256(module_path.read_bytes()).hexdigest(),
                True,
            )
        session_path.unlink(missing_ok=True)
        if r_lib.is_symlink() or r_lib.is_file():
            r_lib.unlink()
        elif r_lib.exists():
            shutil.rmtree(r_lib)
        r_lib.mkdir(parents=True)
        with tempfile.TemporaryDirectory(prefix="reproduce-r-") as tmp:
            work = Path(tmp)
            run_env = _r_environment(work)
            captured = _checked(
                run(_in_bash(r_module_env_script(build_modules), module), run_env, work), f"module load {module}"
            )
            pairs = dict(item.split("=", 1) for item in captured.split("\0") if "=" in item)
            module_env: dict[str, Any] = {"module": module, "build_modules": list(build_modules)}
            module_env["PATH"] = pairs.get("PATH", "")
            if "LD_LIBRARY_PATH" in pairs:
                module_env["LD_LIBRARY_PATH"] = pairs["LD_LIBRARY_PATH"]
            script, out = work / "install.R", work / "out"
            script.write_text(r_script(env.r_packages))
            out.mkdir()
            prefix = str(venv) if kind == "ir_notebook" and venv is not None else ""
            argv = _in_bash(r_install_script(build_modules), module, str(r_lib), str(script), str(out), repo, prefix)
            _checked(run(argv, run_env, work), "Rscript install")
            rows = [line.split("\t") for line in (out / "installed.tsv").read_text().splitlines() if line.strip()]
            installed = {row[0]: row[1] for row in rows}
            lib_paths = (out / "lib_paths.txt").read_text().splitlines()
            reported_version = (out / "r_version.txt").read_text().strip()
            session = {
                "key": key,
                "installed": installed,
                "lib_paths": lib_paths,
                "r_version": reported_version,
                "session_info": (out / "session_info.txt").read_text().splitlines(),
                "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            }
        if reported_version != f"R version {r_version}" and not reported_version.startswith(f"R version {r_version} "):
            raise ReproduceLockError(f"module {module} gave {reported_version!r}, the profile expects R {r_version}")
        differ = {n: v for n, v in declared.items() if installed.get(n) != v}
        mismatches = [f"{n}: declared {v}, installed {installed.get(n)}" for n, v in differ.items()]
        if mismatches:
            raise ReproduceLockError(f"R library {r_lib}: installed versions differ from r_packages: {mismatches}")
        if not lib_paths or os.path.realpath(lib_paths[0]) != os.path.realpath(r_lib):
            raise ReproduceLockError(f"R library {r_lib} is not first on .libPaths(): {lib_paths}")
        if kind == "ir_notebook" and venv is not None and not _kernel(venv).is_file():
            raise ReproduceLockError(f"IRkernel installspec wrote no kernel spec under {venv}")
        module_sha = _write_json(module_path, module_env)
        session_sha = _write_json(session_path, session)
    return RResult(r_lib, session_path, session_sha, module_path, module_sha, False)


def _dedup(entries: Iterable[str]) -> list[str]:
    out: list[str] = []
    for entry in entries:
        if entry and entry not in out:
            out.append(entry)
    return out


def r_env_vars(
    module_env: Mapping[str, str],
    r_lib: str,
    *,
    path_prefix: Sequence[str] = (),
    path_suffix: Sequence[str] = (),
    native_lib_dirs: Sequence[str] = (),
) -> dict[str, str]:
    """A step's R variables from ``MODULE_ENV.json``.

    Args:
        module_env: The captured module env (``PATH`` and optionally ``LD_LIBRARY_PATH``).
        r_lib: The library, as a value or a template such as ``{cache}/r/x``.
        path_prefix: ``PATH`` entries before the module's, e.g. ``{venv}/bin``.
        path_suffix: ``PATH`` entries after the module's, e.g. native bin dirs.
        native_lib_dirs: The step's selected native lib dirs.

    Returns:
        ``PATH`` (prefix, module, suffix), ``LD_LIBRARY_PATH`` (module entries first, then native lib
        dirs; omitted when empty), ``R_LIBS`` and ``R_LIBS_USER`` set to ``r_lib``, and ``R_LIBS_SITE=""``.
        Repeated entries keep their first position.
    """
    path = _dedup([*path_prefix, *module_env.get("PATH", "").split(":"), *path_suffix])
    libs = _dedup([*module_env.get("LD_LIBRARY_PATH", "").split(":"), *native_lib_dirs])
    out = {"PATH": ":".join(path)} if path else {}
    if libs:
        out["LD_LIBRARY_PATH"] = ":".join(libs)
    return {**out, "R_LIBS": r_lib, "R_LIBS_USER": r_lib, "R_LIBS_SITE": ""}
