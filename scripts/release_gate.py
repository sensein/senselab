"""Refuse a ``dist/`` that cannot be released.

Usage::

    uv build --all-packages
    uv run python scripts/release_gate.py dist [--repo DIR] [--tag TAG --prerelease true|false] [--floor VERSION]
        [--index-url URL]

Exit status 0 means every check below passed over one listing of ``dist/``, in this order, before anything is
uploaded:

- every file is a wheel or sdist named exactly ``<name>-<version>-py3-none-any.whl`` / ``<name>-<version>.tar.gz``,
  one of each per distribution, its members normalised (no absolute, ``..``, empty or case-duplicate name; an
  sdist's all under its one top directory) and its core metadata parsed strictly (no repeated single field);
- the version is ``X.Y.Z`` or ``X.Y.Z(a|b|rc)N`` exactly (no epoch, local, post or dev segment), shared by every
  file and filename, at or above the floor, equal to the tag and matching the prerelease flag;
- each wheel's and sdist's metadata pins its sibling exactly and states the name, ``Requires-Python``,
  requirements, extras and license files its pyproject declares, nothing more;
- each artifact's members equal the set derived from the tracked files its hatch config selects plus the named
  generated files, each a byte copy of its source: a missing member and an extra one are both refused;
- each wheel's ``WHEEL`` states the filename's tag, its RECORD lists its members with their hashes and its entry
  points are the declared ones;
- each distribution installs alone, with only its own siblings, into a fresh venv at its floor Python, and
  imports there from the installed files: every module of core and the worker, and every top-level package and
  subpackage of the root distribution;
- each sdist builds a wheel whose members are byte-identical to the wheel in ``dist/``.

Every subprocess runs in ``gate_environment``: only the variables it names, whatever the caller exports.
The floor defaults to ``[tool.senselab.release] RELEASE_FLOOR`` in the root pyproject.
"""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import email.parser
import fnmatch
import hashlib
import io
import json
import os
import posixpath
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from packaging.markers import Marker
from packaging.metadata import Metadata, parse_email
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import Tag
from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

ROOT = Path(__file__).resolve().parents[1]
DEPENDS_ON = {"senselab": "senselab-core", "senselab-core": "senselab-worker", "senselab-worker": None}
"""The one declared dependency graph: each distribution and the sibling it pins exactly."""
DISTRIBUTIONS = tuple(DEPENDS_ON)
DIST_ROOTS = {"senselab": ".", "senselab-core": "packages/senselab-core", "senselab-worker": "packages/senselab-worker"}
"""Each distribution's root, relative to the repository."""
WHEEL_TAG = Tag("py3", "none", "any")
"""The one tag every wheel is built for: pure Python 3."""
_NUMBER = r"(?:0|[1-9][0-9]*)"
VERSION_GRAMMAR = re.compile(rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}(?:(?:a|b|rc){_NUMBER})?")
"""The releasable version strings: a final release or a numbered alpha, beta or release candidate."""
TAG_GRAMMAR = re.compile(rf"{_NUMBER}\.{_NUMBER}\.{_NUMBER}-(?:alpha|beta|rc)\.{_NUMBER}")
"""The repository's release-tag spelling of a pre-release, ``X.Y.Z-(alpha|beta|rc).N``, accepted beside the canonical one."""
IMPORT_EVERY_MODULE = frozenset({"senselab-core", "senselab-worker"})
"""Distributions whose every module the install check imports; the others import their packages and subpackages."""
WHEEL_GENERATED = ("METADATA", "WHEEL", "RECORD")
"""The ``dist-info`` files every wheel build writes; ``entry_points.txt`` and ``licenses/`` are derived per wheel."""
DEFAULT_LICENSE_FILES = ("LICEN[CS]E*", "COPYING*", "NOTICE*", "AUTHORS*")
"""The license-file globs hatchling applies at a project root when the pyproject names none."""
SUPPORTED_BUILD_KEYS = {
    "build": {"artifacts", "targets"},
    "wheel": {"packages", "shared-scripts", "force-include"},
    "sdist": {"only-include", "force-include"},
}
"""The hatch build settings the member derivation understands; any other refuses the release."""
REGULAR_MODES = frozenset({"100644", "100755"})
PYPI = "https://pypi.org/simple"
"""The index third-party requirements come from unless ``--index-url`` names another."""


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
    if version.epoch:
        return "has an epoch"
    if version.local is not None:
        return "has a local segment"
    if version.is_devrelease:
        return "is a dev build: HEAD is not on a release tag, or the tag is not X.Y.Z"
    if version.is_postrelease:
        return "is a post release"
    if not VERSION_GRAMMAR.fullmatch(text) or str(version) != text:
        return "is not written X.Y.Z or X.Y.Z(a|b|rc)N"
    return None


def tag_problem(tag: str) -> str | None:
    """Return why ``tag`` is not a release tag, or ``None``: a releasable version string or ``X.Y.Z-(alpha|beta|rc).N``."""
    if TAG_GRAMMAR.fullmatch(tag):
        return None
    return version_problem(tag)


def _escaped(name: str) -> str:
    """The distribution name as it appears in file and directory names: canonical, ``-`` written ``_``."""
    return canonicalize_name(name).replace("-", "_")


def filename_parts(path: Path) -> tuple[str, str, str]:
    """Return ``(kind, canonical name, version text)`` of a wheel or sdist named exactly as the build names it.

    Raises:
        GateError: If the name is not ``<name>-<version>-py3-none-any.whl`` or ``<name>-<version>.tar.gz`` with the
            name escaped and the version normalised, or carries a build tag or another tag.
    """
    try:
        if path.name.endswith(".whl"):
            name, version, build, tags = parse_wheel_filename(path.name)
            kind = "wheel"
            if build or tags != {WHEEL_TAG}:
                raise GateError(f"{path.name}: a wheel must be tagged py3-none-any with no build number")
            expected = f"{_escaped(name)}-{version}-py3-none-any.whl"
        else:
            name, version = parse_sdist_filename(path.name)
            kind = "sdist"
            expected = f"{_escaped(name)}-{version}.tar.gz"
    except ValueError as exc:
        raise GateError(f"{path.name}: not a wheel or sdist file name ({exc})") from exc
    if path.name != expected:
        raise GateError(f"{path.name}: must be named {expected}")
    return kind, canonicalize_name(name), str(version)


def stem(path: Path) -> str:
    """``<escaped name>-<version>``: the wheel's dist-info prefix, the sdist's top directory."""
    _, name, version = filename_parts(path)
    return f"{_escaped(name)}-{version}"


def _metadata_member(path: Path) -> str:
    return f"{stem(path)}.dist-info/METADATA" if path.suffix == ".whl" else f"{stem(path)}/PKG-INFO"


def _read_member(path: Path, member: str) -> bytes:
    """Return the bytes of the regular file ``member`` in the wheel or sdist ``path``.

    Raises:
        GateError: If the artifact has no such regular file.
    """
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            try:
                return archive.read(member)
            except KeyError as exc:
                raise GateError(f"{path.name}: no {member}") from exc
    with tarfile.open(path) as tar:
        try:
            info = tar.getmember(member)
        except KeyError as exc:
            raise GateError(f"{path.name}: no {member}") from exc
        handle = tar.extractfile(info) if info.isreg() else None
        if handle is None:
            raise GateError(f"{path.name}: {member} is not a regular file")
        return handle.read()


def core_metadata(path: Path) -> tuple[Metadata, str]:
    """Return the strictly parsed core metadata of a wheel (METADATA) or sdist (PKG-INFO) and its raw version.

    Raises:
        GateError: If a single-use field repeats or does not parse, the metadata is invalid, or its name or
            version differs from the file name's.
    """
    _, name, version = filename_parts(path)
    raw, unparsed = parse_email(_read_member(path, _metadata_member(path)))
    if unparsed:
        raise GateError(f"{path.name}: core metadata fields repeated or unparseable: {', '.join(sorted(unparsed))}")
    try:
        metadata = Metadata.from_raw(raw, validate=True)
    except (ExceptionGroup, ValueError) as exc:
        raise GateError(f"{path.name}: invalid core metadata ({exc})") from exc
    if canonicalize_name(metadata.name) != name or raw.get("version") != version:
        raise GateError(f"{path.name}: metadata names {metadata.name} {raw.get('version')}, the file {name} {version}")
    return metadata, version


def _metadata(path: Path) -> tuple[str, Version, list[Requirement]]:
    """Return the canonical name, version and requirements recorded in a wheel or sdist."""
    metadata, _ = core_metadata(path)
    return canonicalize_name(metadata.name), metadata.version, list(metadata.requires_dist or [])


def dist_files(dist: Path) -> list[Path]:
    """Return every file ``dist/*`` matches, dotfiles excluded as the shell glob excludes them.

    Raises:
        GateError: If ``dist`` holds a directory or a file of another kind.
    """
    files = sorted(p for p in dist.iterdir() if not p.name.startswith("."))
    foreign = [p.name for p in files if not (p.is_file() and p.name.endswith((".whl", ".tar.gz")))]
    if foreign:
        raise GateError(f"dist/ holds files that are neither wheel nor sdist: {', '.join(foreign)}")
    return files


def member_problems(path: Path) -> list[str]:
    """Return each entry name of the wheel or sdist ``path`` that is not normalised.

    A name is refused when it is absolute, holds a backslash, NUL, empty, ``.`` or ``..`` component, repeats
    another name up to case, or (in an sdist) lies outside the one top directory ``<name>-<version>``.
    """
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = [i.filename.rstrip("/") for i in archive.infolist()]
        top = None
    else:
        with tarfile.open(path) as tar:
            names = [m.name.rstrip("/") for m in tar.getmembers()]
        top = stem(path)
    problems = []
    for name in names:
        parts = name.split("/")
        if name.startswith("/") or "\\" in name or "\0" in name or any(p in ("", ".", "..") for p in parts):
            problems.append(f"{name!r} is not a normalised relative name")
        elif top is not None and parts[0] != top:
            problems.append(f"{name!r} is outside {top}/")
    folded = Counter(n.casefold() for n in names)
    problems += [f"{n!r} appears {c} times up to case" for n, c in sorted(folded.items()) if c > 1]
    return problems


def check_names(files: list[Path]) -> None:
    """Require every file name, member name and core metadata of ``files`` to parse strictly.

    Raises:
        GateError: On the first file whose name, members or metadata are not as the build writes them.
    """
    for path in files:
        filename_parts(path)
        problems = member_problems(path)
        if problems:
            raise GateError(f"{path.name}: {'; '.join(problems[:5])}")
        core_metadata(path)


def check_files(files: list[Path]) -> None:
    """Require exactly one wheel and one sdist of each distribution.

    Raises:
        GateError: If a distribution lacks either file or has two of one kind.
    """
    seen: Counter[tuple[str, str]] = Counter()
    for path in files:
        kind, name, _ = filename_parts(path)
        seen[(_metadata(path)[0], kind)] += 1
        if name != _metadata(path)[0]:
            raise GateError(f"{path.name}: file name and metadata name differ")
    for name in DISTRIBUTIONS:
        for kind in ("wheel", "sdist"):
            if seen[(name, kind)] != 1:
                raise GateError(f"dist/ must hold exactly one {kind} of {name}, found {seen[(name, kind)]}")
    extra = {name for name, _ in seen} - set(DISTRIBUTIONS)
    if extra:
        raise GateError(f"dist/ holds unknown distributions: {', '.join(sorted(extra))}")


def check_versions(
    files: list[Path], floor: Version, *, tag: str | None = None, prerelease: bool | None = None
) -> None:
    """Require one shared releasable version string at or above ``floor``, equal to the tag when given.

    Raises:
        GateError: If the files disagree, the version is not ``X.Y.Z(a|b|rc)N``-shaped (epoch, local, post and dev
            segments included), is below ``floor``, is not the version of ``tag`` (a releasable version string or
            ``X.Y.Z-(alpha|beta|rc).N``), or
            ``prerelease`` differs from the version's own pre-release status.
    """
    found = {path.name: core_metadata(path)[1] for path in files}
    if not found:
        raise GateError("no file to check")
    texts = set(found.values())
    if len(texts) != 1:
        raise GateError("files disagree on the version: " + ", ".join(f"{n} is {v}" for n, v in sorted(found.items())))
    (text,) = texts
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
        try:
            tagged = Version(tag)
        except InvalidVersion as exc:
            raise GateError(f"tag {tag!r} is not a version") from exc
        if tagged != version:
            raise GateError(f"version {text} was built for another tag than {tag!r}")
    if prerelease is not None and prerelease != version.is_prerelease:
        raise GateError(
            f"release marked prerelease={prerelease}, version {version} is_prerelease={version.is_prerelease}"
        )


def _equals(text: str, version: Version) -> bool:
    """Whether ``text`` is a plain version, not a wildcard, equal to ``version``."""
    try:
        return Version(text) == version
    except InvalidVersion:
        return False


def check_closure(files: list[Path]) -> None:
    """Require each wheel and sdist to pin exactly the sibling ``DEPENDS_ON`` names, and no other sibling.

    Raises:
        GateError: If a distribution is absent, a required pin is missing or is not ``==`` that distribution's
            version, or a file requires a sibling the graph does not give it.
    """
    built = {name: version for name, version, _ in map(_metadata, files)}
    missing = sorted(set(DISTRIBUTIONS) - set(built))
    if missing:
        raise GateError(f"dist/ lacks {', '.join(missing)}: the release would publish an uninstallable senselab")
    for path in files:
        name, _, requirements = _metadata(path)
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


def gate_git(repo: Path, *args: str) -> bytes:
    """Run ``git -C repo args`` in ``gate_environment`` with no system or global configuration; return stdout.

    Raises:
        GateError: If git fails, as it does when ``repo`` is not a git work tree.
    """
    env = {**gate_environment(os.environ), "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull}
    command = ["git", "-c", f"safe.directory={Path(repo).resolve()}", "-C", str(repo), *args]
    done = subprocess.run(command, capture_output=True, env=env)
    if done.returncode != 0:
        raise GateError(f"{repo} is not a git work tree: the artifact contents cannot be checked")
    return done.stdout


def tracked_modes(repo: Path) -> dict[str, str]:
    """Return ``{path: mode}`` for every path in the index of ``repo``, read with the gate's environment.

    Raises:
        GateError: If ``repo`` is not a git work tree.
    """
    out = {}
    for line in gate_git(repo, "ls-files", "-s", "-z").decode().split("\0"):
        if line:
            meta, _, path = line.partition("\t")
            out[path] = meta.split()[0]
    return out


def _tracked(repo: Path) -> set[str]:
    """Return every path git tracks in ``repo``."""
    return set(tracked_modes(repo))


def _pyproject(repo: Path, root: Path) -> dict[str, Any]:
    with (repo / root / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _hatch_build(pyproject: dict[str, Any], name: str) -> dict[str, Any]:
    """Return ``[tool.hatch.build]`` after refusing any setting the member derivation does not understand.

    Raises:
        GateError: Naming the unsupported setting, so a config change cannot ship members the gate did not derive.
    """
    build = pyproject.get("tool", {}).get("hatch", {}).get("build", {})
    targets = build.get("targets", {})
    unknown = [f"build.{k}" for k in sorted(set(build) - SUPPORTED_BUILD_KEYS["build"])]
    unknown += [f"targets.{k}" for k in sorted(set(targets) - {"wheel", "sdist"})]
    for target in ("wheel", "sdist"):
        unknown += [f"{target}.{k}" for k in sorted(set(targets.get(target, {})) - SUPPORTED_BUILD_KEYS[target])]
    if unknown:
        raise GateError(f"{name}: the gate cannot derive the members of hatch settings {', '.join(unknown)}")
    return build


def _join(root: Path, relative: str) -> str:
    """The repo-relative POSIX path of ``relative`` taken from the distribution root ``root``."""
    return posixpath.normpath((root / relative).as_posix())


def _under(paths: Iterable[str], source: str) -> dict[str, str]:
    """Map each of ``paths`` that is ``source`` or lies under it to its path relative to ``source`` (``""`` for itself)."""
    if source in (".", ""):
        return {p: p for p in paths}
    return {p: "" if p == source else p[len(source) + 1 :] for p in paths if p == source or p.startswith(source + "/")}


def declared_license_files(repo: Path, root: Path, regular: set[str]) -> set[str]:
    """Return the license files, relative to ``root``, that hatchling records for the distribution at ``root``.

    These are the tracked files under ``root`` matching ``[project] license-files``, or the default globs, one
    path component per glob component.
    """
    globs = _pyproject(repo, root)["project"].get("license-files", DEFAULT_LICENSE_FILES)

    def matches(rel: str, glob: str) -> bool:
        parts, pattern = rel.split("/"), glob.split("/")
        return len(parts) == len(pattern) and all(
            fnmatch.fnmatchcase(a, b) for a, b in zip(parts, pattern, strict=True)
        )

    return {rel for rel in _under(regular, _join(root, ".")).values() if any(matches(rel, g) for g in globs)}


def located_gitignore(repo: Path, root: Path, regular: set[str]) -> str:
    """Return the repo-relative ``.gitignore`` hatchling copies into an sdist: the first one up from ``root``.

    The search is over the work tree, as hatchling's is; the file it finds must be in ``regular``, the tracked
    regular files.

    Raises:
        GateError: If there is none between ``root`` and the repository root, or the one found is not tracked.
    """
    directory = _join(root, ".")
    while True:
        candidate = posixpath.normpath(posixpath.join(directory, ".gitignore"))
        if (repo / candidate).is_file():
            if candidate not in regular:
                raise GateError(f"{candidate}: the .gitignore hatchling copies into the sdist is not a tracked file")
            return candidate
        if directory in (".", ""):
            raise GateError(f"no .gitignore between {root} and the repository root")
        directory = posixpath.dirname(directory) or "."


def expected_members(path: Path, repo: Path, modes: Mapping[str, str]) -> dict[str, str | None]:
    """Return every member the wheel or sdist ``path`` must hold, each mapped to its tracked source.

    A member mapped to ``None`` is generated by the build and checked by its own rule (metadata, RECORD, entry
    points). The tracked sources are the regular files the hatch config selects: a wheel's ``packages``,
    ``shared-scripts`` and ``force-include``; an sdist's root (or its ``only-include`` entries), its
    ``force-include`` and the ``.gitignore`` hatchling copies in.

    Raises:
        GateError: If the config uses a setting the derivation does not know, or a selected path is a symlink or
            submodule.
    """
    kind, name, _ = filename_parts(path)
    root = Path(DIST_ROOTS[name])
    pyproject = _pyproject(repo, root)
    targets = _hatch_build(pyproject, name).get("targets", {})
    regular = {p for p, m in modes.items() if m in REGULAR_MODES}
    irregular = {p for p, m in modes.items() if m not in REGULAR_MODES}
    top = stem(path)
    out: dict[str, str | None] = {}

    def add(dest: str, source: str) -> None:
        if _under(irregular, source):
            raise GateError(f"{name}: {source} holds a tracked symlink or submodule, which cannot ship")
        for repo_path, rel in _under(regular, source).items():
            out[posixpath.join(dest, rel) if rel else dest] = repo_path

    if kind == "wheel":
        wheel = targets.get("wheel", {})
        for package in wheel["packages"]:
            add(posixpath.basename(package), _join(root, package))
        for source, dest in wheel.get("shared-scripts", {}).items():
            out[f"{top}.data/scripts/{dest}"] = _join(root, source)
        for source, dest in wheel.get("force-include", {}).items():
            add(dest, _join(root, source))
        out.update({f"{top}.dist-info/{n}": None for n in WHEEL_GENERATED})
        if declared_entry_points(repo, root):
            out[f"{top}.dist-info/entry_points.txt"] = None
        for license_file in declared_license_files(repo, root, regular):
            out[f"{top}.dist-info/licenses/{license_file}"] = _join(root, license_file)
        return out
    sdist = targets.get("sdist", {})
    for entry in sdist.get("only-include", ["."]):
        add(posixpath.normpath(posixpath.join(top, entry)), _join(root, entry))
    for source, dest in sdist.get("force-include", {}).items():
        add(posixpath.join(top, dest), _join(root, source))
    out[f"{top}/PKG-INFO"] = None
    out[f"{top}/.gitignore"] = located_gitignore(repo, root, regular)
    return out


def check_members(files: list[Path], repo: Path) -> None:
    """Require each artifact's members to be exactly its derived members, each a byte copy of its tracked source.

    Raises:
        GateError: Naming members that are missing, extra, not regular files or not copies of their source.
    """
    modes = tracked_modes(repo)
    problems = []
    for path in files:
        expected = expected_members(path, repo, modes)
        actual = {}
        for member, content, problem in artifact_members(path):
            if problem:
                problems.append(f"{path.name}:{member} ({problem})")
            actual[member] = content
        problems += [f"{path.name}:{m} (missing)" for m in sorted(set(expected) - set(actual))]
        problems += [
            f"{path.name}:{m} (not a tracked file the config selects)" for m in sorted(set(actual) - set(expected))
        ]
        for member in sorted(set(expected) & set(actual)):
            source = expected[member]
            if source is not None and (repo / source).read_bytes() != actual[member]:
                problems.append(f"{path.name}:{member} (differs from {source})")
    if problems:
        raise GateError(f"artifact members differ from the tracked sources: {', '.join(problems[:8])}")


def _requirement_key(requirement: Requirement, extra: str | None = None) -> tuple[Any, ...]:
    """A comparable form of ``requirement``: name, extras, specifiers, URL and marker, ``extra`` folded in."""
    marker = requirement.marker
    if extra is not None:
        clause = f'extra == "{canonicalize_name(extra)}"'
        marker = Marker(f"({marker}) and {clause}" if marker else clause)
    return (
        canonicalize_name(requirement.name),
        tuple(sorted(requirement.extras)),
        str(requirement.specifier),
        requirement.url,
        str(marker) if marker else None,
    )


def declared_requirements(repo: Path, name: str, version: str) -> set[tuple[Any, ...]]:
    """Return the requirement keys the pyproject of ``name`` declares at ``version``, extras expanded as hatch does.

    Base requirements come from ``[project] dependencies``, or from the pin hook's ``dependencies`` when they are
    dynamic, plus the exact sibling pin. Each extra's requirements carry ``extra == "<name>"``; an extra naming the
    distribution itself contributes the requirements of the extras it names.
    """
    pyproject = _pyproject(repo, Path(DIST_ROOTS[name]))
    project = pyproject["project"]
    if "dependencies" in project.get("dynamic", []):
        base = list(pyproject["tool"]["hatch"]["metadata"]["hooks"]["custom"]["dependencies"])
    else:
        base = list(project.get("dependencies", []))
    if DEPENDS_ON[name] is not None:
        base.append(f"{DEPENDS_ON[name]}=={version}")
    extras = {canonicalize_name(k): v for k, v in project.get("optional-dependencies", {}).items()}

    def expand(extra: str, seen: frozenset[str]) -> list[Requirement]:
        out = []
        for line in extras[extra]:
            requirement = Requirement(line)
            if canonicalize_name(requirement.name) == name:
                for nested in sorted(requirement.extras):
                    if canonicalize_name(nested) not in seen:
                        out += expand(canonicalize_name(nested), seen | {canonicalize_name(nested)})
            else:
                out.append(requirement)
        return out

    keys = {_requirement_key(Requirement(line)) for line in base}
    for extra in extras:
        keys |= {_requirement_key(r, extra) for r in expand(extra, frozenset({extra}))}
    return keys


def check_metadata(files: list[Path], repo: Path) -> None:
    """Require each artifact's core metadata to state exactly what its pyproject declares.

    Name, ``Requires-Python``, ``Requires-Dist`` (each line once), ``Provides-Extra`` and ``License-File`` must
    equal the declaration; any other dependency, extra or Python range is refused.

    Raises:
        GateError: Naming the artifact and the first field that differs.
    """
    modes = tracked_modes(repo)
    regular = {p for p, m in modes.items() if m in REGULAR_MODES}
    for path in files:
        metadata, version = core_metadata(path)
        name = canonicalize_name(metadata.name)
        root = Path(DIST_ROOTS[name])
        project = _pyproject(repo, root)["project"]
        shipped = Counter(_requirement_key(r) for r in metadata.requires_dist or [])
        declared = declared_requirements(repo, name, version)
        extras = {canonicalize_name(e) for e in project.get("optional-dependencies", {})}
        problems = []
        if metadata.name != project["name"]:
            problems.append(f"Name {metadata.name!r} is not {project['name']!r}")
        if metadata.requires_python != SpecifierSet(project["requires-python"]):
            problems.append(f"Requires-Python {metadata.requires_python} is not {project['requires-python']}")
        problems += [f"Requires-Dist {k} appears {c} times" for k, c in shipped.items() if c > 1]
        problems += [f"Requires-Dist {k} is not declared" for k in sorted(set(shipped) - declared, key=str)]
        problems += [f"declared {k} is not in Requires-Dist" for k in sorted(declared - set(shipped), key=str)]
        if {canonicalize_name(e) for e in metadata.provides_extra or []} != extras:
            problems.append(f"Provides-Extra {metadata.provides_extra} is not {sorted(extras)}")
        if set(metadata.license_files or []) != declared_license_files(repo, root, regular):
            problems.append(f"License-File {metadata.license_files} is not the declared license files")
        if problems:
            raise GateError(f"{path.name}: {'; '.join(problems[:5])}")


def wheel_problems(path: Path) -> list[str]:
    """Return how the wheel's ``WHEEL`` file differs from a pure-Python wheel tagged as its file name says."""
    text = _read_member(path, f"{stem(path)}.dist-info/WHEEL").decode()
    message = email.parser.Parser().parsestr(text)
    keys = Counter(message.keys())
    problems = [f"{k} appears {c} times" for k, c in keys.items() if c > 1 and k != "Tag"]
    problems += [
        f"unexpected header {k}" for k in keys if k not in ("Wheel-Version", "Generator", "Root-Is-Purelib", "Tag")
    ]
    if message["Wheel-Version"] != "1.0":
        problems.append(f"Wheel-Version {message['Wheel-Version']} is not 1.0")
    if message["Root-Is-Purelib"] != "true":
        problems.append(f"Root-Is-Purelib {message['Root-Is-Purelib']} is not true")
    tags = message.get_all("Tag") or []
    if tags != [str(WHEEL_TAG)]:
        problems.append(f"Tag {tags} is not the file name's [{WHEEL_TAG}]")
    return problems


def declared_entry_points(repo: Path, root: Path) -> dict[str, dict[str, str]]:
    """Return the entry points the pyproject at ``root`` declares, by group, as ``entry_points.txt`` holds them."""
    project = _pyproject(repo, root)["project"]
    groups = {"console_scripts": project.get("scripts", {}), "gui_scripts": project.get("gui-scripts", {})}
    groups.update(project.get("entry-points", {}))
    return {group: dict(points) for group, points in groups.items() if points}


def _entry_points(text: str) -> dict[str, dict[str, str]]:
    parser = configparser.ConfigParser(delimiters=("=",), interpolation=None)
    parser.optionxform = str  # type: ignore[assignment,method-assign]
    parser.read_string(text)
    return {group: dict(parser[group]) for group in parser.sections()}


def check_wheel_metadata(files: list[Path], repo: Path) -> None:
    """Require each wheel's ``WHEEL`` to match its file name, RECORD to list its members, entry points as declared.

    Raises:
        GateError: If ``WHEEL`` states another tag, purelib flag or version, or repeats a header; a member is
            missing from RECORD or RECORD names an absent one, a row is repeated or its hash or size differs from
            the member's bytes; or ``entry_points.txt`` differs from the declaration.
    """
    for path in files:
        if path.suffix != ".whl":
            continue
        name = _metadata(path)[0]
        problems = wheel_problems(path) + record_problems(path)
        members = {m: c for m, c, _ in artifact_members(path)}
        shipped = members.get(f"{stem(path)}.dist-info/entry_points.txt")
        found = _entry_points(shipped.decode()) if shipped is not None else {}
        declared = declared_entry_points(repo, Path(DIST_ROOTS[name]))
        if found != declared:
            problems.append(f"entry_points.txt holds {found}, the pyproject declares {declared}")
        if problems:
            raise GateError(f"{path.name}: {'; '.join(problems[:5])}")


def record_problems(wheel: Path) -> list[str]:
    """Return how the wheel's RECORD differs from its members: names, sha256 hashes and sizes, one entry each."""
    with zipfile.ZipFile(wheel) as archive:
        names = [i.filename for i in archive.infolist() if not i.is_dir()]
        records = [n for n in names if re.fullmatch(r"[^/]+\.dist-info/RECORD", n)]
        if len(records) != 1:
            return [f"expected one dist-info/RECORD, found {len(records)}"]
        (record,) = records
        rows = list(csv.reader(io.StringIO(archive.read(record).decode("utf-8"))))
        problems = [f"{n} appears {names.count(n)} times" for n in sorted(set(names)) if names.count(n) > 1]
        listed: dict[str, list[str]] = {}
        for row in rows:
            if not row:
                continue
            if len(row) != 3:
                problems.append(f"RECORD row {row} is not path,hash,size")
            elif row[0] in listed:
                problems.append(f"RECORD lists {row[0]} twice")
            else:
                listed[row[0]] = row
        problems += [f"{n} is not in RECORD" for n in sorted(set(names) - set(listed))]
        problems += [f"RECORD lists {n}, which the wheel lacks" for n in sorted(set(listed) - set(names))]
        for member in sorted(set(names) & set(listed)):
            _, digest, size = listed[member]
            if member == record:
                if digest or size:
                    problems.append("RECORD records a hash or size for itself")
                continue
            data = archive.read(member)
            if digest != _hash(data) or size != str(len(data)):
                problems.append(f"{member} is recorded as {digest},{size}, holds {_hash(data)},{len(data)}")
    return problems


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


UV_KEPT = frozenset(
    {
        "UV_CACHE_DIR",
        "UV_PYTHON_INSTALL_DIR",
        "UV_PYTHON_BIN_DIR",
        "UV_HTTP_TIMEOUT",
        "UV_CONCURRENT_DOWNLOADS",
        "UV_NATIVE_TLS",
        "UV_NO_PROGRESS",
    }
)
"""The ``UV_*`` variables the gate passes on: where uv keeps its files and how it talks to the network."""
ENV_KEPT = UV_KEPT | {
    "PATH",
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "SYSTEMROOT",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "all_proxy",
}
"""Every variable a gate subprocess sees: where tools and temporaries are, the locale, and how to reach the network."""


def gate_environment(environ: Mapping[str, str]) -> dict[str, str]:
    """Return the variables of ``environ`` named in ``ENV_KEPT``, and no other."""
    return {k: v for k, v in environ.items() if k in ENV_KEPT}


def _run(*args: str, cwd: str | None = None) -> str:
    """Run a command in ``gate_environment``; a ``uv`` command also gets ``--no-config``; return its stdout.

    Raises:
        GateError: carrying the command's output on a non-zero status.
    """
    if args[0] == "uv":
        args = (*args[: 3 if args[1] == "pip" else 2], "--no-config", *args[3 if args[1] == "pip" else 2 :])
    done = subprocess.run(args, capture_output=True, text=True, env=gate_environment(os.environ), cwd=cwd)
    if done.returncode != 0:
        raise GateError(f"{' '.join(args)} failed ({done.returncode}):\n{done.stdout}{done.stderr}")
    return done.stdout


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


_MARKER_ENVIRONMENT = (
    "import json, os, platform, sys\n"
    "i = sys.implementation.version\n"
    "v = f'{i.major}.{i.minor}.{i.micro}' + ('' if i.releaselevel == 'final' else i.releaselevel[0] + str(i.serial))\n"
    "print(json.dumps({'implementation_name': sys.implementation.name, 'implementation_version': v,\n"
    "    'os_name': os.name, 'platform_machine': platform.machine(), 'platform_release': platform.release(),\n"
    "    'platform_system': platform.system(), 'platform_version': platform.version(),\n"
    "    'python_full_version': platform.python_version(),\n"
    "    'platform_python_implementation': platform.python_implementation(),\n"
    "    'python_version': '.'.join(platform.python_version_tuple()[:2]), 'sys_platform': sys.platform}))\n"
)
"""Prints the PEP 508 marker environment of the interpreter running it."""

_VERIFY = (
    "import csv, importlib, importlib.util, io, json, os, sys\n"
    "from importlib import metadata\n"
    "bad = []\n"
    "for n, want in json.loads(sys.argv[1]).items():\n"
    "    dist = metadata.distribution(n)\n"
    "    record = {r[0]: r[1] for r in csv.reader(io.StringIO(dist.read_text('RECORD'))) if len(r) > 1 and r[1]}\n"
    "    wrong = [f for f, h in want['files'].items() if record.get(f) != h]\n"
    "    if dist.version != want['version'] or wrong:\n"
    "        bad.append(f'{n} {dist.version}: {wrong[:3]}')\n"
    "assert not bad, f'siblings not installed from dist: {bad}'\n"
    "for m in json.loads(sys.argv[2]):\n"
    "    if m.rpartition('.')[2] == '__main__':\n"
    "        spec = importlib.util.find_spec(m)\n"
    "        compile(open(spec.origin, 'rb').read(), spec.origin, 'exec')\n"
    "    else:\n"
    "        importlib.import_module(m)\n"
    "prefix = os.path.realpath(sys.prefix) + os.sep\n"
    "tops = set(json.loads(sys.argv[3]))\n"
    "stray = sorted(f'{k} from {v.__file__}' for k, v in list(sys.modules.items())\n"
    "    if k.partition('.')[0] in tops and not os.path.realpath(getattr(v, '__file__', None) or '').startswith(prefix))\n"
    "assert not stray, f'first-party modules imported from outside the venv: {stray[:3]}'\n"
)
"""Checks the siblings' files, imports every listed module and requires every first-party module to be installed."""


def wheel_modules(wheel: Path) -> list[str]:
    """Return every module of the wheel's packages: each ``.py`` whose directories all hold an ``__init__.py``.

    A ``.py`` file in a directory without ``__init__.py`` is package data, not a module, and is not listed.
    """
    names = {m for m, _, _ in artifact_members(wheel)}
    modules = []
    for member in sorted(names):
        parts = member[: -len(".py")].split("/") if member.endswith(".py") else []
        if not parts or parts[0].endswith((".dist-info", ".data")):
            continue
        if all("/".join(parts[:i]) + "/__init__.py" in names for i in range(1, len(parts))):
            modules.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return modules


def import_targets(wheel: Path) -> list[str]:
    """Return what the install check imports from ``wheel``.

    Every module for the distributions in ``IMPORT_EVERY_MODULE``; for the others each top-level package and each
    of its direct subpackages, whose members the member check already proves complete.
    """
    modules = wheel_modules(wheel)
    if _metadata(wheel)[0] in IMPORT_EVERY_MODULE:
        return modules
    names = {m for m, _, _ in artifact_members(wheel)}
    return [m for m in modules if m.count(".") <= 1 and m.replace(".", "/") + "/__init__.py" in names]


def wheel_file_hashes(wheel: Path) -> dict[str, str]:
    """Return the RECORD-notation hash of every file in ``wheel`` except its RECORD."""
    return {
        member: _hash(content)
        for member, content, _ in artifact_members(wheel)
        if content is not None and not member.endswith(".dist-info/RECORD")
    }


def selected_requirements(
    requirements: Iterable[Requirement], environment: dict[str, str], extras: Iterable[str]
) -> list[str]:
    """Return the requirements that apply to ``environment`` with ``extras`` requested, each without its marker.

    A requirement applies when it has no marker, or its marker holds in ``environment`` with ``extra`` empty or
    set to one of ``extras``.
    """
    out = []
    for requirement in requirements:
        if requirement.marker is None or any(
            requirement.marker.evaluate({**environment, "extra": e}) for e in ("", *extras)
        ):
            plain = Requirement(str(requirement))
            plain.marker = None
            out.append(str(plain))
    return out


def check_installs(files: list[Path], pythons: Mapping[str, str], *, index_url: str = PYPI) -> None:
    """Install each distribution alone into its own fresh venv at ``pythons[name]``; import its ``import_targets``.

    Only the distribution's own wheel is passed by path. Its siblings are reached by the resolver through
    ``--find-links`` on the directory of ``files``, with ``--no-index`` and ``--no-cache``. Every ``uv`` call runs
    with ``--no-config`` in ``gate_environment``. The third-party requirements of the distribution and of its siblings, no extra requested, are
    installed first, by name, from ``index_url`` only, through the host's cache, which no first-party wheel enters.
    Each sibling must then be installed with the version and file hashes of the wheel in ``dist/``. The import
    targets are imported by the venv's interpreter in isolated mode (``-I``) from a scratch directory,
    and every first-party module loaded must come from the venv.

    Raises:
        GateError: If creating an environment, installing, a sibling's files or importing fails.
    """
    wheels = {_metadata(p)[0]: p.resolve() for p in files if p.suffix == ".whl"}
    meta = {n: core_metadata(p)[0] for n, p in wheels.items()}
    built = {n: {"version": str(meta[n].version), "files": wheel_file_hashes(p)} for n, p in wheels.items()}
    tops = sorted({m.partition(".")[0] for p in wheels.values() for m in wheel_modules(p)})
    links = next(iter(wheels.values())).parent
    with tempfile.TemporaryDirectory(prefix="release-gate-") as tmp:
        for name in DISTRIBUTIONS:
            siblings, link = {}, DEPENDS_ON[name]
            while link is not None:
                siblings[link] = built[link]
                link = DEPENDS_ON[link]
            venv = Path(tmp) / name
            _run("uv", "venv", "--python", pythons[name], str(venv), cwd=tmp)
            interpreter = str(venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"))
            environment = json.loads(_run(interpreter, "-I", "-c", _MARKER_ENVIRONMENT, cwd=tmp))
            requirements = []
            for n in (name, *siblings):
                third_party = [r for r in meta[n].requires_dist or [] if canonicalize_name(r.name) not in DISTRIBUTIONS]
                requirements += selected_requirements(third_party, environment, ())
            if requirements:
                _run("uv", "pip", "install", "--python", interpreter, "--index-url", index_url, *requirements, cwd=tmp)
            install = ("uv", "pip", "install", "--no-cache", "--no-index", "--python", interpreter)
            install += ("--find-links", str(links))
            _run(*install, str(wheels[name]), cwd=tmp)
            modules = json.dumps(import_targets(wheels[name]))
            _run(interpreter, "-I", "-c", _VERIFY, json.dumps(siblings), modules, json.dumps(tops), cwd=tmp)


def check_sdist_builds(files: list[Path], *, index_url: str = PYPI) -> None:
    """Build a wheel from each sdist alone and require its members to be byte-identical to the wheel in ``dist/``.

    Raises:
        GateError: If an sdist does not build, or its wheel's name or any member differs from the direct wheel's.
    """
    wheels = {_metadata(p)[0]: p for p in files if p.suffix == ".whl"}
    with tempfile.TemporaryDirectory(prefix="release-gate-sdist-") as tmp:
        for sdist in (p for p in files if p.suffix != ".whl"):
            name = _metadata(sdist)[0]
            unpacked, out = Path(tmp) / name / "src", Path(tmp) / name / "out"
            unpacked.mkdir(parents=True)
            with tarfile.open(sdist) as tar:
                tar.extractall(unpacked, filter="data")
            _run("uv", "build", "--wheel", "--no-cache", "--index-url", index_url, "--out-dir", str(out),
                 str(unpacked / stem(sdist)), cwd=tmp)  # fmt: skip
            rebuilt = [p for p in out.iterdir() if p.suffix == ".whl"]
            if [p.name for p in rebuilt] != [wheels[name].name]:
                raise GateError(f"{sdist.name} builds {[p.name for p in rebuilt]}, not {wheels[name].name}")
            direct = {m: c for m, c, _ in artifact_members(wheels[name])}
            again = {m: c for m, c, _ in artifact_members(rebuilt[0])}
            differ = sorted(m for m in set(direct) | set(again) if direct.get(m) != again.get(m))
            if differ:
                raise GateError(f"{sdist.name} builds a wheel that differs from {wheels[name].name} in {differ[:5]}")


def static_checks(
    files: list[Path], repo: Path, floor: Version, *, tag: str | None = None, prerelease: bool | None = None
) -> None:
    """Run every check that needs no environment: names, set, versions, pins, metadata, members, RECORD.

    Raises:
        GateError: On the first failing check.
    """
    check_names(files)
    check_files(files)
    check_versions(files, floor, tag=tag, prerelease=prerelease)
    check_closure(files)
    check_metadata(files, repo)
    check_members(files, repo)
    check_wheel_metadata(files, repo)


def main(
    dist: Path,
    *,
    repo: Path,
    floor: str | None = None,
    tag: str | None = None,
    prerelease: bool | None = None,
    pythons: Mapping[str, str] | None = None,
    index_url: str = PYPI,
) -> int:
    """Run every check over one listing of ``dist``; print the reason and return 1 on the first failure, else 0.

    ``repo`` is the git work tree the artifacts were built from.
    """
    try:
        files = dist_files(dist)
        try:
            lowest = Version(floor) if floor else release_floor()
        except InvalidVersion as exc:
            raise GateError(f"floor {floor!r} is not a version") from exc
        static_checks(files, repo, lowest, tag=tag, prerelease=prerelease)
        check_installs(files, pythons or floor_pythons(), index_url=index_url)
        check_sdist_builds(files, index_url=index_url)
    except (GateError, ValueError) as exc:
        print(f"release gate refused: {exc}", file=sys.stderr)
        return 1
    print(
        f"release gate passed: {len(files)} files checked; each installed alone, imported, and rebuilt from its sdist"
    )
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dist", type=Path)
    parser.add_argument("--floor")
    parser.add_argument("--tag")
    parser.add_argument("--prerelease", choices=("true", "false"))
    parser.add_argument("--repo", type=Path, help="git work tree the dist was built from; default: dist's parent")
    parser.add_argument("--index-url", default=PYPI, help=f"index for third-party requirements; default: {PYPI}")
    arguments = parser.parse_args()
    sys.exit(
        main(
            arguments.dist,
            floor=arguments.floor,
            tag=arguments.tag,
            prerelease=None if arguments.prerelease is None else arguments.prerelease == "true",
            repo=(arguments.repo or arguments.dist.resolve().parent),
            index_url=arguments.index_url,
        )
    )
