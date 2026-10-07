"""Lock headers, the pre-install integrity check and freeze parsing for reproduce envs.

A requirements-derived body lock opens with a header::

    # reproduce lock
    # requirements-sha256: <requirements_sha256(...) of the compile inputs>
    # python: 3.11

followed by uv's pinned, hashed lines. Paths in an :class:`EnvSpec` are relative to the manifest
repository, passed as ``base_dir``.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
import tomllib
import urllib.parse
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import NamedTuple

from senselab.utils.reproduce.spec import EnvSpec

HEADER_MARKER = "# reproduce lock"

_HEADER_RE = re.compile(r"^#\s*([a-z0-9-]+):\s*(.*?)\s*$")
_KEY_RE = re.compile(r"^[a-z0-9-]+$")
_LINE_RE = re.compile(r"^([A-Za-z0-9._-]+)(?:\[[^\]]*\])?\s*(==|@)\s*([^\s;\\]+)")
_HASH_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})")
_CUDA_LOCAL_RE = re.compile(r"\+cu\d+$")
_COMMENT_RE = re.compile(r"(^|\s)#.*$")


class ReproduceLockError(RuntimeError):
    """A lock, overlay or resolve does not match its manifest or the locks it must agree with."""


class LockEntry(NamedTuple):
    """One pinned requirement: normalised name, ``==`` version or ``@`` URL, and its sha256 hashes."""

    name: str
    pin: str
    is_url: bool
    hashes: frozenset[str]


def normalize_name(name: str) -> str:
    """Return a distribution name normalised per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def file_sha256(path: Path) -> str:
    """Return the hex sha256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def requirements_sha256(
    requirements: Iterable[str],
    python: str,
    max_cuda: str | None,
    exclude_newer: datetime.date,
    config_sha: str | None,
    overrides_sha: str | None,
    extra_no_emit: Iterable[str],
) -> str:
    """Digest of everything a requirements-derived body lock is compiled from.

    Args:
        requirements: The requirement lines of ``requirements.in``; order does not matter.
        python: The env's ``major.minor`` Python version.
        max_cuda: The env's CUDA ceiling, if any.
        exclude_newer: The compile's ``--exclude-newer`` date.
        config_sha: sha256 of the uv ``--config-file``, if any.
        overrides_sha: sha256 of the ``--overrides`` file, if any.
        extra_no_emit: Names left out beyond the torch family and the sdists; order does not matter.

    Returns:
        A hex sha256.
    """
    payload = {
        "requirements": sorted(requirements),
        "python": python,
        "max_cuda": max_cuda,
        "exclude_newer": exclude_newer.isoformat(),
        "config_file_sha256": config_sha,
        "overrides_sha256": overrides_sha,
        "extra_no_emit": sorted(extra_no_emit),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_header(body: str, fields: Mapping[str, str]) -> str:
    """Prefix a lock body with the reproduce header.

    Args:
        body: uv's pinned, hashed lines.
        fields: Header keys (lowercase, digits and ``-``) and single-line values, in order.

    Returns:
        The header followed by ``body``.

    Raises:
        ValueError: If a key is malformed or a value spans lines.
    """
    lines = [HEADER_MARKER]
    for key, value in fields.items():
        if not _KEY_RE.match(key) or "\n" in value or "\r" in value:
            raise ValueError(f"bad header field {key!r}: {value!r}")
        lines.append(f"# {key}: {value}")
    return "\n".join(lines) + "\n" + body


def parse_header(text: str) -> dict[str, str]:
    """Read the reproduce header of a lock.

    Args:
        text: The lock file's contents.

    Returns:
        The header fields, keyed as written.

    Raises:
        ReproduceLockError: If the first line is not the reproduce marker.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != HEADER_MARKER:
        raise ReproduceLockError(f"lock does not start with {HEADER_MARKER!r}")
    header: dict[str, str] = {}
    for line in lines[1:]:
        if not line.startswith("#"):
            break
        match = _HEADER_RE.match(line)
        if match:
            header[match.group(1)] = match.group(2)
    return header


def _logical_lines(text: str) -> list[str]:
    joined = re.sub(r"\\\r?\n\s*", " ", text)
    return [line.strip() for line in joined.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def lock_entries(text: str) -> list[LockEntry]:
    """Parse a pinned requirements file: a uv lock, an export, a resolve or a freeze.

    Args:
        text: The file's contents; comments and backslash continuations are allowed.

    Returns:
        One entry per requirement line, in file order.

    Raises:
        ReproduceLockError: If a line is not an ``==`` or ``@`` pin.
    """
    entries = []
    for line in _logical_lines(text):
        match = _LINE_RE.match(line)
        if not match:
            raise ReproduceLockError(f"not a pinned requirement: {line!r}")
        name, op, pin = match.groups()
        entries.append(LockEntry(normalize_name(name), pin, op == "@", frozenset(_HASH_RE.findall(line))))
    return entries


def _normalise_version(version: str) -> str:
    return _CUDA_LOCAL_RE.sub("", version)


def entry_version(entry: LockEntry) -> str:
    """Return an entry's version with any ``+cuXXX`` local tag dropped.

    Args:
        entry: A parsed requirement.

    Returns:
        The ``==`` version, or the version in a ``.whl`` URL's file name.

    Raises:
        ReproduceLockError: If a URL pin is not a wheel over ``file://``, ``http://`` or ``https://``.
    """
    if not entry.is_url:
        return _normalise_version(entry.pin)
    url = urllib.parse.urlsplit(entry.pin)
    filename = urllib.parse.unquote(url.path.rsplit("/", 1)[-1])
    parts = filename.split("-")
    if url.scheme not in ("file", "http", "https") or not filename.endswith(".whl") or len(parts) < 5:
        raise ReproduceLockError(f"{entry.name} is not pinned to a wheel: {entry.pin}")
    return _normalise_version(parts[1])


def parse_freeze(text: str) -> dict[str, str]:
    """Parse ``uv pip freeze`` output into ``{name: version}``.

    Args:
        text: Freeze lines: ``name==version``, ``name @ file://…whl`` or ``name @ https://…whl``.

    Returns:
        Normalised names mapped to versions, ``+cuXXX`` dropped.

    Raises:
        ReproduceLockError: For editable installs, source trees, VCS URLs or unpinned lines.
    """
    return {entry.name: entry_version(entry) for entry in lock_entries(text)}


def _by_name(entries: Iterable[LockEntry]) -> dict[str, set[tuple[str, frozenset[str]]]]:
    out: dict[str, set[tuple[str, frozenset[str]]]] = {}
    for entry in entries:
        out.setdefault(entry.name, set()).add((entry.pin, entry.hashes))
    return out


def _check_sha(base_dir: Path, rel: str, expected: str, what: str) -> None:
    actual = file_sha256(base_dir / rel)
    if actual != expected:
        raise ReproduceLockError(f"{what} {rel}: sha256 {actual} != manifest {expected}")


def _requirement_lines(text: str) -> list[str]:
    lines = (_COMMENT_RE.sub("", line).strip() for line in text.splitlines())
    return [line for line in lines if line]


def _torch_against_resolve(torch: list[LockEntry], resolve: list[LockEntry]) -> list[str]:
    versions: dict[str, set[str]] = {}
    for entry in resolve:
        versions.setdefault(entry.name, set()).add(entry_version(entry))
    return [
        f"{e.name}: torch.lock {entry_version(e)}, first_resolve {sorted(versions.get(e.name, set()))}"
        for e in torch
        if entry_version(e) not in versions.get(e.name, set())
    ]


def _torch_against_uv_lock(torch: list[LockEntry], uv_lock: Path) -> list[str]:
    packages: dict[str, list[tuple[str, set[str]]]] = {}
    for package in tomllib.loads(uv_lock.read_text()).get("package", []):
        hashes = {w["hash"].removeprefix("sha256:") for w in package.get("wheels", []) if "hash" in w}
        packages.setdefault(normalize_name(package["name"]), []).append(
            (_normalise_version(package["version"]), hashes)
        )
    bad = []
    for entry in torch:
        candidates = packages.get(entry.name, [])
        version = entry_version(entry)
        if not any(v == version and entry.hashes <= h for v, h in candidates):
            bad.append(
                f"{entry.name}: torch.lock {version}, uv.lock {sorted(v for v, _ in candidates)} or hash differs"
            )
    return bad


def check_integrity(env: EnvSpec, base_dir: Path, uv_lock: Path | None = None) -> None:
    """Check an env's hashed files and pin agreement before anything is installed.

    The checks run in order: lock file sha256s (torch, body, wheels, overlays, build envs); the
    ``requirements.in`` sidecar, config and overrides sha256s and the recomputed header digest; the
    ``first_resolve`` sha256; overlay pins against the base locks; torch-family pins against
    ``first_resolve`` (``requirements.in`` bodies) or the repository ``uv.lock`` (uv-export bodies).

    Args:
        env: The step's env.
        base_dir: The manifest repository; every path in ``env`` is relative to it.
        uv_lock: The checkout's ``uv.lock``; needed for a uv-export body with a torch lock.

    Raises:
        ReproduceLockError: On the first check that fails.
    """
    locks = [("torch lock", env.torch), ("body lock", env.body), ("wheels lock", env.wheels)]
    locks += [("test lock", env.test), ("notebook_tools lock", env.notebook_tools)]
    locks += [(f"build_env lock of {s.name}", s.build_env) for s in env.sdist_allow]
    for what, ref in locks:
        if ref is not None:
            _check_sha(base_dir, ref.lock, ref.lock_sha256, what)

    body = env.body
    if body.source == "requirements.in":
        if body.requirements_in is None or body.requirements_in_sha256 is None:
            raise ReproduceLockError("a requirements.in body needs requirements_in and its sha256")
        _check_sha(base_dir, body.requirements_in, body.requirements_in_sha256, "requirements.in")
        for what, rel, sha in (
            ("config file", body.config_file, body.config_file_sha256),
            ("overrides", body.overrides, body.overrides_sha256),
        ):
            if rel is not None and sha is not None:
                _check_sha(base_dir, rel, sha, what)
        header = parse_header((base_dir / body.lock).read_text())
        expected = requirements_sha256(
            _requirement_lines((base_dir / body.requirements_in).read_text()),
            env.python,
            env.max_cuda,
            env.exclude_newer,
            body.config_file_sha256,
            body.overrides_sha256,
            env.extra_no_emit,
        )
        if header.get("requirements-sha256") != expected:
            raise ReproduceLockError(
                f"body lock {body.lock}: header requirements-sha256 {header.get('requirements-sha256')} "
                f"!= recomputed {expected}"
            )

    if env.first_resolve is not None:
        _check_sha(base_dir, env.first_resolve.file, env.first_resolve.sha256, "first_resolve")

    base = lock_entries((base_dir / body.lock).read_text())
    if env.torch is not None:
        base += lock_entries((base_dir / env.torch.lock).read_text())
    pinned = _by_name(base)
    for what, ref in (("test", env.test), ("notebook_tools", env.notebook_tools)):
        if ref is None:
            continue
        bad = sorted(
            e.name
            for e in lock_entries((base_dir / ref.lock).read_text())
            if e.name in pinned and (e.pin, e.hashes) not in pinned[e.name]
        )
        if bad:
            raise ReproduceLockError(f"{what} overlay pins differ from the base locks: {bad}")

    if env.torch is None:
        return
    torch = lock_entries((base_dir / env.torch.lock).read_text())
    if body.source == "requirements.in":
        if env.first_resolve is None:
            raise ReproduceLockError("a torch lock with a requirements.in body needs first_resolve")
        mismatches = _torch_against_resolve(torch, lock_entries((base_dir / env.first_resolve.file).read_text()))
    else:
        if uv_lock is None:
            raise ReproduceLockError("a uv-export body with a torch lock needs the repository uv.lock to check against")
        mismatches = _torch_against_uv_lock(torch, uv_lock)
    if mismatches:
        raise ReproduceLockError(f"torch lock {env.torch.lock} disagrees: {mismatches}")
