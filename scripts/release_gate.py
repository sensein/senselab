"""Refuse a ``dist/`` that cannot be released.

Usage::

    uv build --all-packages
    uv run python scripts/release_gate.py dist [--tag TAG --prerelease true|false] [--floor VERSION]

Exit status 0 means every file in ``dist/`` is a wheel or sdist of one of the three distributions at one
releasable version, the sibling pins are exact, every file inside every artifact is a git-tracked file of its package, and each
distribution installs on its own, with only its own siblings, into a fresh virtual environment at its floor
Python and imports. The floor defaults to ``[tool.senselab.release] RELEASE_FLOOR`` in the root pyproject.
"""

from __future__ import annotations

import argparse
import json
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
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
DEPENDS_ON = {"senselab": "senselab-core", "senselab-core": "senselab-worker", "senselab-worker": None}
"""The one declared dependency graph: each distribution and the sibling it pins exactly."""
DISTRIBUTIONS = tuple(DEPENDS_ON)
DIST_ROOTS = {"senselab": ".", "senselab-core": "packages/senselab-core", "senselab-worker": "packages/senselab-worker"}
"""Each distribution's root, relative to the repository."""
GENERATED = frozenset({"METADATA", "WHEEL", "RECORD", "entry_points.txt", "PKG-INFO"})
"""File names a build writes into an artifact that no tracked file corresponds to."""
SDIST_ALIASES = {"_shared_hatch_build_pins.py": "hatch_build_pins.py", ".gitignore": ".gitignore"}
"""Sdist members that come from the repository root: the shared hook, and the ignore file hatchling adds."""
IMPORTS = {
    "senselab-worker": "senselab_worker",
    "senselab-core": "senselab_worker, senselab_core",
    "senselab": "senselab_worker, senselab_core, senselab",
}


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
        raise GateError(f"release marked prerelease={prerelease}, version {version} is_prerelease={version.is_prerelease}")


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
            if not _is_exact_pin(pin, expected, built[expected]):
                raise GateError(f"{name} requires {pin}, not exactly {expected}=={built[expected]} from dist/")


def _is_exact_pin(pin: Requirement, expected: str, version: Version) -> bool:
    """Whether ``pin`` is ``expected==version`` and nothing more: no marker, no extras, one ``==`` clause."""
    specs = list(pin.specifier)
    return (
        canonicalize_name(pin.name) == expected
        and pin.marker is None
        and not pin.extras
        and pin.url is None
        and len(specs) == 1
        and specs[0].operator == "=="
        and _equals(specs[0].version, version)
    )


def _tracked(repo: Path) -> set[str]:
    """Return every path git tracks in ``repo``.

    Raises:
        GateError: If ``repo`` is not a git work tree.
    """
    done = subprocess.run(["git", "ls-files", "-z"], cwd=repo, capture_output=True)
    if done.returncode != 0:
        raise GateError(f"{repo} is not a git work tree: the artifact contents cannot be checked")
    return {p for p in done.stdout.decode().split("\0") if p}


def _wheel_sources(repo: Path, root: Path) -> dict[str, str]:
    """Map each top-level name a wheel of the distribution at ``root`` ships to its repo-relative source directory."""
    with (repo / root / "pyproject.toml").open("rb") as handle:
        config = tomllib.load(handle)
    packages = config["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    return {Path(p).name: (root / p).as_posix() for p in packages}


def artifact_sources(path: Path, repo: Path) -> tuple[str, list[tuple[str, str | None]]]:
    """Return the distribution name and, for each file in the artifact, the repo path it must come from.

    The path is ``None`` for a build-generated file in ``GENERATED``.
    """
    name = _metadata(path)[0]
    root = Path(DIST_ROOTS[name]) if name in DIST_ROOTS else None
    out: list[tuple[str, str | None]] = []
    if root is None:
        return name, out
    if path.suffix == ".whl":
        sources = _wheel_sources(repo, root)
        with zipfile.ZipFile(path) as archive:
            members = [n for n in archive.namelist() if not n.endswith("/")]
        for member in members:
            top, _, rest = member.partition("/")
            if top.endswith(".dist-info"):
                if rest in GENERATED:
                    out.append((member, None))
                elif rest.startswith("licenses/"):
                    out.append((member, (root / rest.removeprefix("licenses/")).as_posix()))
                else:
                    out.append((member, f"{top}/{rest}"))
            else:
                src = sources.get(top)
                out.append((member, f"{src}/{rest}" if src else f"{root.as_posix()}/{member}"))
        return name, out
    with tarfile.open(path) as tar:
        members = [m.name for m in tar.getmembers() if m.isfile()]
    for member in members:
        rest = member.partition("/")[2]
        if rest == "PKG-INFO":
            out.append((member, None))
        elif rest in SDIST_ALIASES:
            out.append((member, SDIST_ALIASES[rest]))
        else:
            out.append((member, (root / rest).as_posix()))
    return name, out


def check_contents(files: list[Path], repo: Path) -> None:
    """Require every file inside every artifact to be a git-tracked file of its package, or a generated one.

    Raises:
        GateError: If an artifact holds a file with no tracked source, or ``repo`` is not a git work tree.
    """
    tracked = _tracked(repo)
    stray = []
    for path in files:
        _, entries = artifact_sources(path, repo)
        stray += [f"{path.name}:{member}" for member, source in entries if source is not None and source not in tracked]
    if stray:
        raise GateError(f"artifacts hold files that git does not track: {', '.join(stray[:5])}")


def _run(*args: str) -> None:
    """Run a command; raise ``GateError`` carrying its output on a non-zero status."""
    done = subprocess.run(args, capture_output=True, text=True)
    if done.returncode != 0:
        raise GateError(f"{' '.join(args)} failed ({done.returncode}):\n{done.stdout}{done.stderr}")


def floor_pythons(repo: Path = ROOT) -> dict[str, str]:
    """Return each distribution's lowest supported Python, read from its pyproject ``requires-python``."""
    floors = {}
    for name, root in DIST_ROOTS.items():
        with (repo / root / "pyproject.toml").open("rb") as handle:
            requires = SpecifierSet(tomllib.load(handle)["project"]["requires-python"])
        lowest = [Version(s.version) for s in requires if s.operator == ">="]
        if not lowest:
            raise GateError(f"{name}: requires-python {requires} has no lower bound")
        floors[name] = f"{lowest[0].major}.{lowest[0].minor}"
    return floors


_VERIFY = (
    "import json, sys\n"
    "from importlib import metadata\n"
    "bad = {n: metadata.version(n) for n, v in json.loads(sys.argv[1]).items() if metadata.version(n) != v}\n"
    "assert not bad, f'siblings not at the built version: {bad}'\n"
)


def check_installs(files: list[Path], pythons: Mapping[str, str]) -> None:
    """Install each distribution alone into its own fresh venv at ``pythons[name]``, check its siblings, import it.

    Only the distribution's own wheel is passed by path. Its siblings must be reached by the resolver through
    ``--find-links`` on the directory of ``files``; third-party requirements come from the index. Each sibling
    must then be installed at the built version.

    Raises:
        GateError: If creating an environment, installing, a sibling's version or importing fails.
    """
    wheels = {_metadata(p)[0]: p.resolve() for p in files if p.suffix == ".whl"}
    versions = {_metadata(p)[0]: str(_metadata(p)[1]) for p in wheels.values()}
    links = next(iter(wheels.values())).parent
    with tempfile.TemporaryDirectory(prefix="release-gate-") as tmp:
        for name in DISTRIBUTIONS:
            siblings, link = {}, DEPENDS_ON[name]
            while link is not None:
                siblings[link] = versions[link]
                link = DEPENDS_ON[link]
            venv = Path(tmp) / name
            _run("uv", "venv", "--python", pythons[name], str(venv))
            interpreter = str(venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
            _run("uv", "pip", "install", "--python", interpreter, "--find-links", str(links), str(wheels[name]))
            _run(interpreter, "-c", _VERIFY + f"import {IMPORTS[name]}", json.dumps(siblings))


def main(
    dist: Path,
    *,
    floor: str | None = None,
    tag: str | None = None,
    prerelease: bool | None = None,
    pythons: Mapping[str, str] | None = None,
    repo: Path | None = None,
) -> int:
    """Run every check over one listing of ``dist``; print the reason and return 1 on the first failure, else 0.

    ``repo`` is the git work tree the artifacts were built from; ``None`` skips the contents check.
    """
    try:
        files = dist_files(dist)
        check_files(files)
        check_versions(files, Version(floor) if floor else release_floor(), tag=tag, prerelease=prerelease)
        check_closure([p for p in files if p.suffix == ".whl"])
        if repo is not None:
            check_contents(files, repo)
        check_installs(files, pythons or floor_pythons())
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
    parser.add_argument("--repo", type=Path, help="git work tree the dist was built from; default: dist's parent")
    arguments = parser.parse_args()
    sys.exit(
        main(
            arguments.dist,
            floor=arguments.floor,
            tag=arguments.tag,
            prerelease=None if arguments.prerelease is None else arguments.prerelease == "true",
            repo=(arguments.repo or arguments.dist.resolve().parent),
        )
    )
