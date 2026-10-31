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
import base64
import hashlib
import json
import re
import stat
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
SDIST_ALIASES = {"_shared_hatch_build_pins.py": "hatch_build_pins.py"}
"""Sdist members force-included from the repository root under another name."""
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


def _hash(data: bytes) -> str:
    """Return ``data``'s digest in the RECORD notation, ``sha256=<urlsafe base64 without padding>``."""
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def artifact_members(path: Path) -> list[tuple[str, bytes | None, str | None]]:
    """Return ``(member, content, problem)`` for every entry of a wheel or sdist that is not a plain directory.

    ``content`` is ``None`` and ``problem`` names the reason for an entry that is not a regular file: a symlink,
    hardlink, device or FIFO in an sdist, or a symlink in a wheel.
    """
    out: list[tuple[str, bytes | None, str | None]] = []
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                mode = info.external_attr >> 16
                if info.is_dir():
                    continue
                if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
                    out.append((info.filename, None, "not a regular file"))
                else:
                    out.append((info.filename, archive.read(info), None))
        return out
    with tarfile.open(path) as tar:
        for member in tar.getmembers():
            if member.isdir():
                continue
            if not member.isreg():
                out.append((member.name, None, "not a regular file"))
            else:
                handle = tar.extractfile(member)
                assert handle is not None
                out.append((member.name, handle.read(), None))
    return out


def _candidates(path: Path, member: str, root: Path, license_files: set[str], sources: dict[str, str]) -> list[str] | None:
    """Return the repo paths ``member`` may be a copy of, ``[]`` when none is allowed, ``None`` when generated."""
    posix = root.as_posix()
    if path.suffix == ".whl":
        stem = "-".join(path.name.split("-")[:2])
        top, _, rest = member.partition("/")
        if top == f"{stem}.dist-info":
            if rest in GENERATED:
                return None
            if rest.startswith("licenses/") and rest.removeprefix("licenses/") in license_files:
                return [(root / rest.removeprefix("licenses/")).as_posix()]
            return []
        if top.endswith((".dist-info", ".data")) or top not in sources:
            return []
        return [f"{sources[top]}/{rest}"]
    rest = member.partition("/")[2]
    if rest == "PKG-INFO":
        return None
    if rest == ".gitignore":
        return [(root / ".gitignore").as_posix(), ".gitignore"]
    if rest in SDIST_ALIASES:
        return [SDIST_ALIASES[rest]]
    return [f"{posix}/{rest}" if posix != "." else rest]


def check_contents(files: list[Path], repo: Path) -> None:
    """Require every file in every artifact to be a byte-identical copy of a git-tracked file, or a generated one.

    Aliased members (the shared hook, the ignore file, license files) are bound to their content, not only
    their name. Entries that are not regular files, and extra dist-info or data directories, are refused.

    Raises:
        GateError: If an artifact holds such an entry, or ``repo`` is not a git work tree.
    """
    tracked = _tracked(repo)
    stray = []
    for path in files:
        name = _metadata(path)[0]
        root = Path(DIST_ROOTS[name])
        sources = _wheel_sources(repo, root) if path.suffix == ".whl" else {}
        license_files = set(_message(path).get_all("License-File") or [])
        for member, content, problem in artifact_members(path):
            label = f"{path.name}:{member}"
            if problem:
                stray.append(f"{label} ({problem})")
                continue
            candidates = _candidates(path, member, root, license_files, sources)
            if candidates is None:
                continue
            if not any(c in tracked and (repo / c).read_bytes() == content for c in candidates):
                stray.append(f"{label} (no identical tracked file)")
    if stray:
        raise GateError(f"artifacts hold files that are not copies of tracked files: {', '.join(stray[:5])}")


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
    "import csv, io, json, sys\n"
    "from importlib import metadata\n"
    "bad = []\n"
    "for n, want in json.loads(sys.argv[1]).items():\n"
    "    dist = metadata.distribution(n)\n"
    "    record = {r[0]: r[1] for r in csv.reader(io.StringIO(dist.read_text('RECORD'))) if len(r) > 1 and r[1]}\n"
    "    wrong = [f for f, h in want['files'].items() if record.get(f) != h]\n"
    "    if dist.version != want['version'] or wrong:\n"
    "        bad.append(f'{n} {dist.version}: {wrong[:3]}')\n"
    "assert not bad, f'siblings not installed from dist: {bad}'\n"
)


def wheel_file_hashes(wheel: Path) -> dict[str, str]:
    """Return the RECORD-notation hash of every file in ``wheel`` except its RECORD."""
    return {
        member: _hash(content)
        for member, content, _ in artifact_members(wheel)
        if content is not None and not member.endswith(".dist-info/RECORD")
    }


def check_installs(files: list[Path], pythons: Mapping[str, str]) -> None:
    """Install each distribution alone into its own fresh venv at ``pythons[name]``, check its siblings, import it.

    Only the distribution's own wheel is passed by path. Its siblings must be reached by the resolver through
    ``--find-links`` on the directory of ``files``, in an install that neither reads nor writes any uv cache
    (``--no-cache``), so a cached wheel of the same name and version cannot stand in for the one in ``dist/``.
    The third-party requirements of the distribution and its siblings are installed first, by name, through the
    host's cache, which no first-party wheel enters. Each sibling must then be installed with the version and
    file hashes of the wheel in ``dist/``, not merely the version.

    Raises:
        GateError: If creating an environment, installing, a sibling's version or importing fails.
    """
    wheels = {_metadata(p)[0]: p.resolve() for p in files if p.suffix == ".whl"}
    third_party = {
        n: [str(r) for r in _metadata(p)[2] if canonicalize_name(r.name) not in DISTRIBUTIONS]
        for n, p in wheels.items()
    }
    built = {
        n: {"version": str(_metadata(p)[1]), "files": wheel_file_hashes(p)} for n, p in wheels.items()
    }
    links = next(iter(wheels.values())).parent
    with tempfile.TemporaryDirectory(prefix="release-gate-") as tmp:
        for name in DISTRIBUTIONS:
            siblings, link = {}, DEPENDS_ON[name]
            while link is not None:
                siblings[link] = built[link]
                link = DEPENDS_ON[link]
            venv = Path(tmp) / name
            _run("uv", "venv", "--python", pythons[name], str(venv))
            interpreter = str(venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
            requirements = [r for n in (name, *siblings) for r in third_party[n]]
            if requirements:
                _run("uv", "pip", "install", "--python", interpreter, *requirements)
            install = ("uv", "pip", "install", "--no-cache", "--python", interpreter, "--find-links", str(links))
            _run(*install, str(wheels[name]))
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
