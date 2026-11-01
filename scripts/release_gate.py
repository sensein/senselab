"""Refuse a ``dist/`` that cannot be released.

Usage::

    uv build --all-packages
    uv run python scripts/release_gate.py dist [--tag TAG --prerelease true|false] [--floor VERSION]

Exit status 0 means three checks passed over one listing of ``dist/``:

- the version is ``X.Y.Z`` or ``X.Y.Z(a|b|rc)N``, shared by every file, at or above the floor, equal to the tag
  and matching the prerelease flag;
- each distribution pins its siblings with ``==`` to the version in ``dist/`` and pins nothing else of the set;
- each distribution's wheel installs, with its siblings' wheels, into a fresh venv and its packages import.

The floor defaults to ``[tool.senselab.release] RELEASE_FLOOR`` in the root pyproject.
"""

from __future__ import annotations

import argparse
import email.parser
import re
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import tomllib
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
DEPENDS_ON: dict[str, tuple[str, ...]] = {
    "senselab": ("senselab-core", "senselab-worker"),
    "senselab-core": ("senselab-worker",),
    "senselab-worker": (),
}
DISTRIBUTIONS = tuple(DEPENDS_ON)
_NUMBER = r"(?:0|[1-9][0-9]*)"
VERSION_GRAMMAR = re.compile(rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}(?:(?:a|b|rc){_NUMBER})?")
TAG_GRAMMAR = re.compile(rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}-(?:alpha|beta|rc)\.{_NUMBER}")


class GateError(Exception):
    """A reason the contents of ``dist/`` cannot be released."""


def release_floor() -> Version:
    """Return ``RELEASE_FLOOR`` from the root pyproject."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return Version(tomllib.load(handle)["tool"]["senselab"]["release"]["RELEASE_FLOOR"])


def version_problem(text: str) -> str | None:
    """Return why ``text`` is not a releasable version string, or ``None`` when it is one."""
    try:
        version = Version(text)
    except InvalidVersion:
        return "is not a version"
    if version.epoch or version.local is not None or version.is_postrelease:
        return "has an epoch, local or post segment"
    if version.is_devrelease:
        return "is a dev build: HEAD is not on a release tag, or the tag is not X.Y.Z"
    if not VERSION_GRAMMAR.fullmatch(text) or str(version) != text:
        return "is not written X.Y.Z or X.Y.Z(a|b|rc)N"
    return None


def tag_problem(tag: str) -> str | None:
    """Return why ``tag`` is not a release tag, or ``None``: a releasable version or ``X.Y.Z-(alpha|beta|rc).N``."""
    if TAG_GRAMMAR.fullmatch(tag):
        return None
    return version_problem(tag)


def parse_name(path: Path) -> tuple[str, str, str]:
    """Return ``(kind, canonical name, version text)`` of a wheel or sdist file name.

    Raises:
        GateError: If the name is neither.
    """
    try:
        if path.name.endswith(".whl"):
            name, version, _, _ = parse_wheel_filename(path.name)
            return "wheel", canonicalize_name(name), str(version)
        name, version = parse_sdist_filename(path.name)
    except ValueError as exc:
        raise GateError(f"{path.name}: not a wheel or sdist file name ({exc})") from exc
    return "sdist", canonicalize_name(name), str(version)


def dist_files(dist: Path) -> list[Path]:
    """Return every wheel and sdist in ``dist``, one of each per distribution.

    Raises:
        GateError: If ``dist`` holds another file, or lacks or repeats a wheel or sdist of a distribution.
    """
    files = sorted(p for p in dist.iterdir() if not p.name.startswith("."))
    foreign = [p.name for p in files if not (p.is_file() and p.name.endswith((".whl", ".tar.gz")))]
    if foreign:
        raise GateError(f"dist/ holds files that are neither wheel nor sdist: {', '.join(foreign)}")
    found = [parse_name(p)[:2] for p in files]
    for name in DISTRIBUTIONS:
        for kind in ("wheel", "sdist"):
            if found.count((kind, name)) != 1:
                raise GateError(f"dist/ must hold exactly one {kind} of {name}, found {found.count((kind, name))}")
    extra = {name for _, name in found} - set(DISTRIBUTIONS)
    if extra:
        raise GateError(f"dist/ holds unknown distributions: {', '.join(sorted(extra))}")
    return files


def requirements(path: Path) -> list[Requirement]:
    """Return the ``Requires-Dist`` entries of a wheel's METADATA or an sdist's PKG-INFO.

    Raises:
        GateError: If the artifact has no metadata file.
    """
    _, name, version = parse_name(path)
    stem = f"{name.replace('-', '_')}-{version}"
    try:
        if path.suffix == ".whl":
            with zipfile.ZipFile(path) as archive:
                raw = archive.read(f"{stem}.dist-info/METADATA")
        else:
            with tarfile.open(path) as tar:
                handle = tar.extractfile(f"{stem}/PKG-INFO")
                raw = b"" if handle is None else handle.read()
    except KeyError as exc:
        raise GateError(f"{path.name}: no core metadata") from exc
    message = email.parser.BytesParser().parsebytes(raw, headersonly=True)
    return [Requirement(r) for r in message.get_all("Requires-Dist", [])]


def check_versions(
    files: list[Path], floor: Version, *, tag: str | None = None, prerelease: bool | None = None
) -> None:
    """Require one shared releasable version at or above ``floor``, equal to ``tag`` when given.

    Raises:
        GateError: If the files disagree, the version is not releasable, is below ``floor``, is not the tag's, or
            ``prerelease`` differs from the version's own status.
    """
    found = {path.name: parse_name(path)[2] for path in files}
    if len(set(found.values())) != 1:
        raise GateError("files disagree on the version: " + ", ".join(f"{n} is {v}" for n, v in sorted(found.items())))
    (text,) = set(found.values())
    problem = version_problem(text)
    if problem:
        raise GateError(f"version {text} {problem}")
    version = Version(text)
    if version < floor:
        raise GateError(f"version {version} is below the release floor {floor}")
    if tag is not None:
        problem = tag_problem(tag)
        if problem:
            raise GateError(f"tag {tag!r} {problem}")
        if Version(tag) != version:
            raise GateError(f"version {text} was built for another tag than {tag!r}")
    if prerelease is not None and prerelease != version.is_prerelease:
        raise GateError(
            f"release marked prerelease={prerelease}, version {version} is_prerelease={version.is_prerelease}"
        )


def check_pins(files: list[Path]) -> None:
    """Require each file to pin exactly the siblings ``DEPENDS_ON`` names, each ``==`` the version in ``dist/``.

    Raises:
        GateError: If a pin is missing, extra, or not a single ``==`` on the built version.
    """
    built = {parse_name(p)[1]: parse_name(p)[2] for p in files}
    missing = sorted(set(DISTRIBUTIONS) - set(built))
    if missing:
        raise GateError(f"dist/ lacks {', '.join(missing)}: the release would publish an uninstallable senselab")
    for path in files:
        _, name, _ = parse_name(path)
        pins = [r for r in requirements(path) if canonicalize_name(r.name) in DISTRIBUTIONS]
        got = sorted(canonicalize_name(r.name) for r in pins)
        if got != sorted(DEPENDS_ON[name]):
            raise GateError(f"{name} requires siblings {got or 'none'}, the graph requires {sorted(DEPENDS_ON[name])}")
        for pin in pins:
            version = built[canonicalize_name(pin.name)]
            specs = list(pin.specifier)
            if len(specs) != 1 or specs[0].operator != "==" or specs[0].version != version or pin.marker or pin.extras:
                raise GateError(f"{name} requires {pin}, not exactly {pin.name}=={version}")


def _run(*args: str, cwd: str) -> str:
    done = subprocess.run(args, capture_output=True, text=True, cwd=cwd)
    if done.returncode != 0:
        raise GateError(f"{' '.join(args)} failed ({done.returncode}):\n{done.stdout}{done.stderr}")
    return done.stdout


MANAGED_PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}" + (
    "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""
)
"""The running interpreter's version, which the gate's venvs request from uv-managed interpreters only."""


def check_installs(files: list[Path]) -> None:
    """Install each distribution's wheel with its siblings' wheels into a fresh venv and import its packages.

    Raises:
        GateError: If creating the venv, installing or importing fails.
    """
    wheels = {parse_name(p)[1]: p.resolve() for p in files if p.suffix == ".whl"}
    with tempfile.TemporaryDirectory(prefix="release-gate-") as tmp:
        for name in DISTRIBUTIONS:
            venv = Path(tmp) / name
            _run("uv", "venv", "--python-preference", "only-managed", "--python", MANAGED_PYTHON, str(venv), cwd=tmp)
            python = str(venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
            reached = [name, *(n for n in DISTRIBUTIONS if n != name and n in _reach(name))]
            _run("uv", "pip", "install", "--python", python, *(str(wheels[n]) for n in reached), cwd=tmp)
            with zipfile.ZipFile(wheels[name]) as archive:
                tops = sorted(
                    {m.split("/")[0] for m in archive.namelist() if not m.split("/")[0].endswith(".dist-info")}
                )
            code = "import importlib, sys\nfor m in sys.argv[1:]:\n    importlib.import_module(m)\n"
            _run(python, "-I", "-c", code, *(t.removesuffix(".py") for t in tops), cwd=tmp)


def _reach(name: str) -> set[str]:
    out: set[str] = set()
    pending = list(DEPENDS_ON[name])
    while pending:
        sibling = pending.pop()
        if sibling not in out:
            out.add(sibling)
            pending += DEPENDS_ON[sibling]
    return out


def main(dist: Path, *, floor: str | None = None, tag: str | None = None, prerelease: bool | None = None) -> int:
    """Run the three checks over one listing of ``dist``; print the reason and return 1 on failure, else 0."""
    try:
        files = dist_files(dist)
        check_versions(files, Version(floor) if floor else release_floor(), tag=tag, prerelease=prerelease)
        check_pins(files)
        check_installs(files)
    except (GateError, ValueError) as exc:
        print(f"release gate refused: {exc}", file=sys.stderr)
        return 1
    print(f"release gate passed: {len(files)} files checked; each distribution installed and imported")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dist", type=Path)
    parser.add_argument("--floor")
    parser.add_argument("--tag")
    parser.add_argument("--prerelease", choices=("true", "false"))
    arguments = parser.parse_args()
    sys.exit(
        main(
            arguments.dist,
            floor=arguments.floor,
            tag=arguments.tag,
            prerelease=None if arguments.prerelease is None else arguments.prerelease == "true",
        )
    )
