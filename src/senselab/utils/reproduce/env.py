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
``…+<overlay sha256[:8]>``). A venv is complete when its ``.reproduce-venv.json`` marker names its
key; a complete venv is reused, an incomplete one is removed and rebuilt, under a ``flock`` on the
sidecar ``<venv>.lock``.

R envs (kinds ``r`` and ``ir_notebook``) are prepared by `prefetch_r` under the profile's R module,
outside the network namespace, from the Posit Package Manager snapshot of ``env.r_snapshot``. It
writes ``<r_lib>/MODULE_ENV.json`` (the module's ``PATH`` and ``LD_LIBRARY_PATH``) and
``<r_lib>/SESSION.json`` (``sessionInfo()``, ``.libPaths()`` and installed versions), and fails
unless every declared ``r_packages`` version is the installed one. `r_env_vars` gives a step's R
variables in the merge order: module ``LD_LIBRARY_PATH`` entries first, then native lib dirs.
"""

from __future__ import annotations

import datetime
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import NamedTuple

from packaging.markers import Marker, default_environment

from senselab.utils.reproduce import sandbox
from senselab.utils.reproduce.build import (
    MODULES_INIT,
    SYSTEM_PATH,
    base_environment,
    ca_bundle,
    ensure_wheel,
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
R_MODULE_ENV_SCRIPT = f'. {MODULES_INIT} && module load "$1" && exec env -0'
R_INSTALL_SCRIPT = (
    f'. {MODULES_INIT} && module load "$1" && export R_LIBS="$2" R_LIBS_SITE= && '
    'if [ -n "$6" ]; then export PATH="$6/bin:$PATH"; fi && '
    'exec Rscript --vanilla "$3" "$2" "$4" "$5" "$6"'
)
SMOKE_TIMEOUT = 600
_TAIL = 4000
_ANALYSIS = re.compile(r"^[A-Za-z0-9_.-]+$")
_MODULE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")
_ENV_MODULE = re.compile(r"^[A-Za-z0-9_.+-]+(/[A-Za-z0-9_.+-]+)*$")
_R_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9.]*$")
_R_VERSION = re.compile(r"^[0-9][0-9.-]*$")
_CUDA_LOCAL = re.compile(r"\+cu\d+$")
_PIP_CHECK_LINE = re.compile(r"^The package `([^`]+)` (.*)$")

Runner = Callable[[list[str], dict[str, str]], "subprocess.CompletedProcess[str]"]


class VenvResult(NamedTuple):
    """One built or reused venv: its path, key, overlay sha256 (derived venvs), reuse flag and ``{name: version}``."""

    path: Path
    env_key: str
    overlay_sha256: str | None
    reused: bool
    packages: dict[str, str]


class PrefetchResult(NamedTuple):
    """What `prefetch_env` returns.

    ``wheels_lock_sha256`` is the sha256 of ``wheels.lock`` after stage 3 (``None`` when no wheel was
    needed); when ``built`` is non-empty it changed, and the caller writes it into the manifest.
    """

    base: VenvResult
    derived: dict[str, VenvResult]
    wheels_lock_sha256: str | None
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


@contextmanager
def _flock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


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
    if env.wheels is not None:
        if wheels_lock is not None and wheels_lock != base_dir / env.wheels.lock:
            raise ValueError("wheels_lock is given and env.wheels names another file")
        return base_dir / env.wheels.lock, env.wheels.lock_sha256
    if env.sdist_allow and wheels_lock is None:
        raise ValueError("an env with sdist_allow needs env.wheels or a wheels_lock path to record wheels in")
    return wheels_lock, None


class _Wheels:
    """Stage 3 inputs, prepared on first use: each sdist's wheel reused or built, then copied and checked."""

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
        self._copies: list[Path] | None = None

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
                dest = self.dest / result.wheel_sha256
                self._copies.append(stage_wheel(result.wheel, result.wheel_sha256, dest))
        return self._copies


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
        marker = path / MARKER
        if marker.is_file() and (path / INSTALLED).is_file():
            recorded = json.loads(marker.read_text())
            if recorded.get("env_key") == key and recorded.get("overlay_sha256") == overlay_sha:
                packages = json.loads((path / INSTALLED).read_text())["packages"]
                return VenvResult(path, key, overlay_sha, True, packages)
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
            "freeze": lines,
            "packages": packages,
        }
        installed_sha = _write_json(path / INSTALLED, installed)
        _write_json(marker, {"env_key": key, "overlay_sha256": overlay_sha, "installed_sha256": installed_sha})
        return VenvResult(path, key, overlay_sha, False, packages)


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

    Args:
        env: The step's env.
        base_dir: The manifest repository; every path in ``env`` is relative to it.
        cache: The reproduce cache; venvs go under ``<cache>/venvs``.
        analysis: The manifest name, used in venv names.
        wheelhouse: Where stage 3 wheels are stored.
        uv_lock: The checkout's ``uv.lock``, for a uv-export body with a torch lock.
        profile: When given, each venv is then checked: ``uv pip check`` must equal
            ``pip_check_expected`` (`check_pip`) and ``smoke_imports`` must import in the sandbox (`smoke`).
        wheels_lock: Where to record wheels when ``env.wheels`` is unset (the first prefetch).
        rebuild: Build every sdist again (`build.ensure_wheel`'s ``rebuild``).

    Returns:
        A `PrefetchResult`.

    Raises:
        ReproduceLockError: If the integrity check, a copy check, an install, the freeze comparison or
            a profile check fails.
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
        for venv in (base, *derived.values()):
            check_pip(venv.path, env)
            failures = smoke(venv.path, env.smoke_imports, profile)
            if failures:
                raise ReproduceLockError(f"smoke imports failed in {venv.path.name}: {failures}")
    lock_sha = wheels.lock_sha256
    if lock_sha is None and lock_path is not None and lock_path.is_file():
        lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    return PrefetchResult(base, derived, lock_sha, tuple(wheels.built))


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


def _default_runner(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, env=env, capture_output=True, text=True, check=False, timeout=SMOKE_TIMEOUT)


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
        runner: Runs ``(argv, env)``; defaults to ``subprocess.run`` with captured output.

    Returns:
        ``"<module>: <output tail>"`` for each import that failed; empty when all succeeded.

    Raises:
        ValueError: If a module name is not a dotted identifier.
    """
    bad = [m for m in modules if not _MODULE_NAME.match(m)]
    if bad:
        raise ValueError(f"not module names: {bad}")
    run = runner or _default_runner
    venv = venv.absolute()
    failures = []
    with tempfile.TemporaryDirectory(prefix="reproduce-smoke-") as name:
        tmp = Path(name).resolve()
        if step is not None and profile is not None:
            paths = {"out": tmp / "out", "tmp": tmp, "checkout": tmp / "checkout", "venv": venv, "cache": tmp / "cache"}
            env = sandbox.build_env(step, profile, paths)[0]
        else:
            env = _smoke_env(venv, tmp, profile)
        for module in modules:
            argv = sandbox.sandbox_argv([str(venv / "bin" / "python"), "-c", f"import {module}"], [], [tmp], [venv])
            result = run(argv, env)
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

    Runs outside the network namespace. The module is loaded with ``bash --noprofile --norc -c '.
    <MODULES_INIT> && module load …'``, so no dotfile is read, and with an allowlisted environment.
    The packages come from `r_snapshot_url` of ``env.r_snapshot``; ``R_LIBS=<r_lib>`` and an empty
    ``R_LIBS_SITE`` keep the module's site library out. A library whose ``SESSION.json`` records the
    same module, snapshot and declared packages is reused.

    Args:
        env: The step's env; ``r_packages``, ``r_snapshot`` and ``r_lib`` are required.
        cache: The reproduce cache; fills ``{cache}`` in ``r_lib``.
        profile: Its ``r.module`` is loaded.
        kind: ``r`` or ``ir_notebook``; the latter installs the ``ir`` kernel spec into ``venv``.
        venv: The derived (notebook tools) venv, for ``ir_notebook``; its ``bin`` leads ``PATH``.
        runner: Runs ``(argv, env)``; defaults to ``subprocess.run`` with captured output.

    Returns:
        An `RResult`.

    Raises:
        ValueError: If the profile has no R module, the env lacks its R fields, or an ``ir_notebook``
            env has no venv or does not declare ``IRkernel``.
        ReproduceLockError: If a command fails, the library is not first on ``.libPaths()``, the kernel
            spec is missing, or an installed version differs from the declared one.
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
    module = profile.r.module
    if not _ENV_MODULE.match(module):
        raise ValueError(f"R module {module!r} is not a plain module name")
    run = runner or _default_runner
    r_lib = Path(render(env.r_lib, {"cache": cache}))
    repo = r_snapshot_url(env.r_snapshot)
    declared = {p.name: p.version for p in env.r_packages}
    session_path, module_path = r_lib / SESSION, r_lib / MODULE_ENV
    kernel_ok = venv is None or kind != "ir_notebook" or _kernel(venv).is_file()
    with _flock(r_lib.with_name(r_lib.name + ".lock")):
        if session_path.is_file() and module_path.is_file() and kernel_ok:
            recorded = json.loads(session_path.read_text())
            if (recorded.get("module"), recorded.get("snapshot"), recorded.get("declared")) == (module, repo, declared):
                return RResult(
                    r_lib,
                    session_path,
                    hashlib.sha256(session_path.read_bytes()).hexdigest(),
                    module_path,
                    hashlib.sha256(module_path.read_bytes()).hexdigest(),
                    True,
                )
        r_lib.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="reproduce-r-") as tmp:
            work = Path(tmp)
            run_env = _r_environment(work)
            captured = _checked(run(_in_bash(R_MODULE_ENV_SCRIPT, module), run_env), f"module load {module}")
            pairs = dict(item.split("=", 1) for item in captured.split("\0") if "=" in item)
            module_env = {"module": module, "PATH": pairs.get("PATH", "")}
            if "LD_LIBRARY_PATH" in pairs:
                module_env["LD_LIBRARY_PATH"] = pairs["LD_LIBRARY_PATH"]
            script, out = work / "install.R", work / "out"
            script.write_text(r_script(env.r_packages))
            out.mkdir()
            prefix = str(venv) if kind == "ir_notebook" and venv is not None else ""
            argv = _in_bash(R_INSTALL_SCRIPT, module, str(r_lib), str(script), str(out), repo, prefix)
            _checked(run(argv, run_env), "Rscript install")
            rows = [line.split("\t") for line in (out / "installed.tsv").read_text().splitlines() if line.strip()]
            installed = {row[0]: row[1] for row in rows}
            lib_paths = (out / "lib_paths.txt").read_text().splitlines()
            session = {
                "module": module,
                "snapshot": repo,
                "declared": declared,
                "installed": installed,
                "lib_paths": lib_paths,
                "r_version": (out / "r_version.txt").read_text().strip(),
                "session_info": (out / "session_info.txt").read_text().splitlines(),
                "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
            }
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
        dirs; omitted when empty), ``R_LIBS=<r_lib>`` and ``R_LIBS_SITE=""``. Repeated entries keep
        their first position.
    """
    path = _dedup([*path_prefix, *module_env.get("PATH", "").split(":"), *path_suffix])
    libs = _dedup([*module_env.get("LD_LIBRARY_PATH", "").split(":"), *native_lib_dirs])
    out = {"PATH": ":".join(path)} if path else {}
    if libs:
        out["LD_LIBRARY_PATH"] = ":".join(libs)
    return {**out, "R_LIBS": r_lib, "R_LIBS_SITE": ""}
