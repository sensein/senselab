"""Refuse a ``dist/`` that cannot be released.

Usage::

    uv build --all-packages
    uv run python scripts/release_gate.py dist [--tag TAG --prerelease true|false] [--floor VERSION]

Exit status 0 means every file in ``dist/`` is a wheel or sdist of one of the three distributions at one
releasable version, the sibling pins are exact, no untracked file sits under a package root, and each
distribution installs on its own, with only its own siblings, into a fresh virtual environment at its floor
Python and imports. The floor defaults to ``[tool.senselab.release] RELEASE_FLOOR`` in the root pyproject.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections.abc import Mapping
from email.message import Message
from email.parser import Parser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
DEPENDS_ON = {"senselab": "senselab-core", "senselab-core": "senselab-worker", "senselab-worker": None}
"""The one declared dependency graph: each distribution and the sibling it pins exactly."""
DISTRIBUTIONS = tuple(DEPENDS_ON)
FLOOR_PYTHONS = {"senselab-worker": "3.10", "senselab-core": "3.11", "senselab": "3.12"}
IMPORTS = {
    "senselab-worker": "senselab_worker",
    "senselab-core": "senselab_worker, senselab_core",
    "senselab": "senselab_worker, senselab_core, senselab",
}
PACKAGE_ROOTS = ("packages", "src/senselab")


class GateError(Exception):
    """A reason the contents of ``dist/`` cannot be released."""


def release_floor() -> Version:
    """Return ``RELEASE_FLOOR`` from the root pyproject."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        return Version(tomllib.load(handle)["tool"]["senselab"]["release"]["RELEASE_FLOOR"])


def _message(path: Path) -> Message:
    """Return the core metadata of a wheel (METADATA) or an sdist (PKG-INFO)."""
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = [n for n in archive.namelist() if re.fullmatch(r"[^/]+\.dist-info/METADATA", n)]
            if len(names) != 1:
                raise GateError(f"{path.name}: expected one dist-info/METADATA, found {len(names)}")
            return Parser().parsestr(archive.read(names[0]).decode())
    with tarfile.open(path) as tar:
        names = [m.name for m in tar.getmembers() if re.fullmatch(r"[^/]+/PKG-INFO", m.name)]
        if len(names) != 1:
            raise GateError(f"{path.name}: expected one PKG-INFO, found {len(names)}")
        member = tar.extractfile(names[0])
        assert member is not None
        return Parser().parsestr(member.read().decode())


def _metadata(path: Path) -> tuple[str, Version, list[Requirement]]:
    """Return the canonical name, version and requirements recorded in a wheel or sdist."""
    message = _message(path)
    try:
        version = Version(message["Version"])
    except (InvalidVersion, TypeError) as exc:
        raise GateError(f"{path.name}: unreadable version {message['Version']!r}") from exc
    requirements = [Requirement(line) for line in message.get_all("Requires-Dist") or []]
    return canonicalize_name(message["Name"]), version, requirements


def dist_files(dist: Path) -> list[Path]:
    """Return every file ``dist/*`` matches, refusing anything that is not a wheel or an sdist.

    Dotfiles, such as the ``.gitignore`` uv writes into ``dist/``, are not matched by the upload glob.

    Raises:
        GateError: If ``dist`` holds a directory or a file of another kind.
    """
    files = sorted(p for p in dist.iterdir() if not p.name.startswith("."))
    foreign = [p.name for p in files if not (p.is_file() and p.name.endswith((".whl", ".tar.gz")))]
    if foreign:
        raise GateError(f"dist/ holds files that are neither wheel nor sdist: {', '.join(foreign)}")
    return files


def check_files(files: list[Path]) -> None:
    """Require exactly one wheel and one sdist of each distribution.

    Raises:
        GateError: If a distribution lacks either file or has two of one kind.
    """
    seen: dict[tuple[str, str], int] = {}
    for path in files:
        key = (_metadata(path)[0], "wheel" if path.suffix == ".whl" else "sdist")
        seen[key] = seen.get(key, 0) + 1
    for name in DISTRIBUTIONS:
        for kind in ("wheel", "sdist"):
            if seen.get((name, kind), 0) != 1:
                raise GateError(f"dist/ must hold exactly one {kind} of {name}, found {seen.get((name, kind), 0)}")
    extra = {name for name, _ in seen} - set(DISTRIBUTIONS)
    if extra:
        raise GateError(f"dist/ holds unknown distributions: {', '.join(sorted(extra))}")


def check_versions(files: list[Path], floor: Version, *, tag: str | None = None, prerelease: bool | None = None) -> None:
    """Require one shared final or pre-release version at or above ``floor``, bound to the tag when given.

    Raises:
        GateError: If the files disagree, or the version is dev, post, local or below ``floor``, or differs from
            ``tag``, or ``prerelease`` differs from the version's own pre-release status.
    """
    found = {path.name: _metadata(path)[1] for path in files}
    if not found:
        raise GateError("no file to check")
    versions = set(found.values())
    if len(versions) != 1:
        raise GateError("files disagree on the version: " + ", ".join(f"{n} is {v}" for n, v in sorted(found.items())))
    (version,) = versions
    if version.local is not None:
        raise GateError(f"version {version} has a local segment")
    if version.is_devrelease:
        raise GateError(f"version {version} is a dev build: HEAD is not on a release tag, or the tag is not X.Y.Z")
    if version.is_postrelease:
        raise GateError(f"version {version} is a post release")
    if version < floor:
        raise GateError(f"version {version} is below the release floor {floor}")
    if tag is not None:
        try:
            tagged = Version(tag)
        except InvalidVersion as exc:
            raise GateError(f"tag {tag!r} is not a version") from exc
        if tagged != version:
            raise GateError(f"version {version} was built for tag {tag!r}, which is {tagged}")
    if prerelease is not None and prerelease != version.is_prerelease:
        raise GateError(f"release marked prerelease={prerelease} but version {version} is_prerelease={version.is_prerelease}")


def _equals(text: str, version: Version) -> bool:
    """Whether ``text`` is a plain version, not a wildcard, equal to ``version``."""
    try:
        return Version(text) == version
    except InvalidVersion:
        return False


def check_closure(wheels: list[Path]) -> None:
    """Require each wheel to pin exactly the sibling ``DEPENDS_ON`` names, and no other sibling.

    Raises:
        GateError: If a distribution is absent, a required pin is missing or is not ``==`` that wheel's version,
            or a wheel requires a sibling the graph does not give it.
    """
    built = {name: version for name, version, _ in map(_metadata, wheels)}
    missing = sorted(set(DISTRIBUTIONS) - set(built))
    if missing:
        raise GateError(f"dist/ lacks {', '.join(missing)}: the release would publish an uninstallable senselab")
    for wheel in wheels:
        name, _, requirements = _metadata(wheel)
        pins = [r for r in requirements if canonicalize_name(r.name) in DISTRIBUTIONS]
        expected = DEPENDS_ON[name]
        wanted = [] if expected is None else [expected]
        got = sorted(canonicalize_name(r.name) for r in pins)
        if got != wanted:
            raise GateError(f"{name} requires siblings {got or 'none'}, the graph requires {wanted or 'none'}")
        for pin in pins:
            specs = list(pin.specifier)
            if not (len(specs) == 1 and specs[0].operator == "==" and _equals(specs[0].version, built[expected])):
                raise GateError(f"{name} requires {pin}, not exactly {expected}=={built[expected]} from dist/")


def check_untracked(dist: Path) -> None:
    """Refuse the build when git lists an untracked, unignored file under a package root.

    A checkout that is not a git repository whose top level is ``dist``'s parent is not checked.

    Raises:
        GateError: If such a file exists.
    """
    repo = dist.resolve().parent
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=repo, capture_output=True, text=True)
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != repo:
        return
    listing = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", *PACKAGE_ROOTS],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    untracked = [p for p in listing.stdout.split("\0") if p]
    if untracked:
        raise GateError("untracked files under a package root would ship in a wheel: " + ", ".join(untracked[:5]))


def _run(*args: str) -> None:
    """Run a command; raise ``GateError`` carrying its output on a non-zero status."""
    done = subprocess.run(args, capture_output=True, text=True)
    if done.returncode != 0:
        raise GateError(f"{' '.join(args)} failed ({done.returncode}):\n{done.stdout}{done.stderr}")


def check_installs(dist: Path, pythons: Mapping[str, str] = FLOOR_PYTHONS) -> None:
    """Install each distribution alone into its own fresh venv at ``pythons[name]`` and import it.

    Each venv receives that distribution's wheel and the wheels of the siblings it pins, as paths, with
    dependency resolution on; third-party requirements come from the index.

    Raises:
        GateError: If creating an environment, installing or importing fails.
    """
    wheels = {_metadata(p)[0]: p.resolve() for p in dist.glob("*.whl")}
    with tempfile.TemporaryDirectory(prefix="release-gate-") as tmp:
        for name in DISTRIBUTIONS:
            chain, link = [], name
            while link is not None:
                chain.append(str(wheels[link]))
                link = DEPENDS_ON[link]
            venv = Path(tmp) / name
            _run("uv", "venv", "--python", pythons[name], str(venv))
            interpreter = str(venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
            _run("uv", "pip", "install", "--python", interpreter, *chain)
            _run(interpreter, "-c", f"import {IMPORTS[name]}")


def main(
    dist: Path,
    *,
    floor: str | None = None,
    tag: str | None = None,
    prerelease: bool | None = None,
    pythons: Mapping[str, str] = FLOOR_PYTHONS,
) -> int:
    """Run every check over ``dist``; print the reason and return 1 on the first failure, else return 0."""
    try:
        files = dist_files(dist)
        check_files(files)
        check_versions(files, Version(floor) if floor else release_floor(), tag=tag, prerelease=prerelease)
        check_closure([p for p in files if p.suffix == ".whl"])
        check_untracked(dist)
        check_installs(dist, pythons)
    except GateError as exc:
        print(f"release gate refused: {exc}", file=sys.stderr)
        return 1
    print(f"release gate passed: {len(files)} files checked; each distribution installed alone and imported")
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
