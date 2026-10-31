"""Refuse a ``dist/`` that cannot be released: wrong version, missing sibling or an install that fails.

Usage::

    uv build --all-packages
    uv run python scripts/release_gate.py dist --floor 2.0.0

Exit status 0 means the three distributions in ``dist/`` share one releasable version, every sibling
requirement is satisfied by a wheel in ``dist/``, and ``senselab`` installs from ``dist/`` into a fresh
virtual environment and imports.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

DISTRIBUTIONS = ("senselab", "senselab-core", "senselab-worker")
"""The distributions the repository builds; a release carries every one of them."""
_IMPORTS = "import senselab_worker, senselab_core, senselab"


class GateError(Exception):
    """A reason the contents of ``dist/`` cannot be released."""


def _metadata(wheel: Path) -> tuple[str, Version, list[Requirement]]:
    """Return the canonical name, version and requirements recorded in a wheel's METADATA."""
    with zipfile.ZipFile(wheel) as archive:
        names = [n for n in archive.namelist() if re.fullmatch(r"[^/]+\.dist-info/METADATA", n)]
        if len(names) != 1:
            raise GateError(f"{wheel.name}: expected one dist-info/METADATA, found {len(names)}")
        message = Parser().parsestr(archive.read(names[0]).decode())
    try:
        version = Version(message["Version"])
    except (InvalidVersion, TypeError) as exc:
        raise GateError(f"{wheel.name}: unreadable version {message['Version']!r}") from exc
    requirements = [Requirement(line) for line in message.get_all("Requires-Dist") or []]
    return canonicalize_name(message["Name"]), version, requirements


def check_versions(wheels: list[Path], floor: Version) -> None:
    """Require one shared version, at or above ``floor``, that is a final or pre-release but not a dev build.

    A version with a ``.dev`` segment is what hatch-vcs writes when HEAD is not on a release tag, so the
    tag-derived pre-releases (``a``, ``b``, ``rc``) pass and everything else is refused.

    Raises:
        GateError: If the wheels disagree, or the version is a dev build, has a local segment or is below floor.
    """
    found = {wheel.name: _metadata(wheel)[1] for wheel in wheels}
    if not found:
        raise GateError("no wheel to check")
    versions = set(found.values())
    if len(versions) != 1:
        raise GateError("wheels disagree on the version: " + ", ".join(f"{n} is {v}" for n, v in sorted(found.items())))
    (version,) = versions
    if version.local is not None:
        raise GateError(f"version {version} has a local segment")
    if version.is_devrelease:
        raise GateError(f"version {version} is a dev build: HEAD is not on a release tag, or the tag is not X.Y.Z")
    if version < floor:
        raise GateError(f"version {version} is below the release floor {floor}")


def _equals(text: str, version: Version) -> bool:
    """Whether ``text`` is a plain version, not a wildcard, equal to ``version``."""
    try:
        return Version(text) == version
    except InvalidVersion:
        return False


def check_closure(wheels: list[Path]) -> None:
    """Require every distribution, and every sibling requirement satisfied exactly by a wheel in the set.

    Raises:
        GateError: If a distribution is absent, or a sibling requirement is not ``==`` the version of a wheel here.
    """
    built = {name: version for name, version, _ in map(_metadata, wheels)}
    missing = sorted(set(DISTRIBUTIONS) - set(built))
    if missing:
        raise GateError(f"dist/ lacks {', '.join(missing)}: the release would publish an uninstallable senselab")
    for wheel in wheels:
        name, _, requirements = _metadata(wheel)
        for requirement in requirements:
            sibling = canonicalize_name(requirement.name)
            if sibling not in DISTRIBUTIONS:
                continue
            specs = list(requirement.specifier)
            if not (len(specs) == 1 and specs[0].operator == "==" and _equals(specs[0].version, built[sibling])):
                raise GateError(f"{name} requires {requirement}, not exactly {sibling}=={built[sibling]} from dist/")


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a command, returning its output; raise ``GateError`` with that output on a non-zero status."""
    done = subprocess.run(args, capture_output=True, text=True)
    if done.returncode != 0:
        raise GateError(f"{' '.join(args)} failed ({done.returncode}):\n{done.stdout}{done.stderr}")
    return done


def check_installs(dist: Path, python: str) -> None:
    """Install ``senselab`` from ``dist/`` into a fresh venv and import all three packages.

    The siblings are installed first with ``--no-index`` so they can only come from ``dist/``; the root
    distribution's third-party requirements then resolve from the index, with the siblings already satisfied.

    Raises:
        GateError: If creating the environment, installing or importing fails.
    """
    dist = dist.resolve()
    wheels = sorted(dist.glob("*.whl"))
    version = _metadata(next(w for w in wheels if _metadata(w)[0] == "senselab"))[1]
    third_party = any(
        canonicalize_name(r.name) not in DISTRIBUTIONS
        for wheel in wheels
        for r in _metadata(wheel)[2]
        if r.marker is None
    )
    with tempfile.TemporaryDirectory(prefix="release-gate-") as tmp:
        venv = Path(tmp) / "venv"
        _run("uv", "venv", "--python", python, str(venv))
        interpreter = str(venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
        pip = ["uv", "pip", "install", "--python", interpreter, "--find-links", str(dist)]
        _run(*pip, "--no-index", "--no-deps", *DISTRIBUTIONS)
        if third_party:
            _run(*pip, f"senselab=={version}")
        _run(interpreter, "-c", _IMPORTS)


def main(dist: Path, *, floor: str) -> int:
    """Run every check over ``dist``; print the reason and return 1 on the first failure, else return 0."""
    try:
        wheels = sorted(dist.glob("*.whl"))
        check_versions(wheels, Version(floor))
        check_closure(wheels)
        check_installs(dist, sys.executable)
    except GateError as exc:
        print(f"release gate refused: {exc}", file=sys.stderr)
        return 1
    print(f"release gate passed: {len(wheels)} wheels installed from {dist} and imported")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dist", type=Path)
    parser.add_argument("--floor", required=True)
    arguments = parser.parse_args()
    sys.exit(main(arguments.dist, floor=arguments.floor))
