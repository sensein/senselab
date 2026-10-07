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
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, NamedTuple

from senselab.utils.reproduce.spec import EnvSpec, FileRef

HEADER_MARKER = "# reproduce lock"
INDEX_OPTIONS = frozenset({"--index-url", "-i", "--extra-index-url", "--find-links", "-f"})
TORCH_NAMES = frozenset({"torch", "torchvision", "torchaudio"})
TORCH_VARIANT_PREFIXES = ("nvidia-", "triton", "pytorch-triton", "cuda-")

_HEADER_RE = re.compile(r"^#\s*([a-z0-9-]+):\s*(.*?)\s*$")
_KEY_RE = re.compile(r"^[a-z0-9-]+$")
_LINE_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*(==(?!=)|@)\s*([^\s;\\]+)\s*(?:;\s*(.*))?$")
_HASH_RE = re.compile(r"--hash=sha256:([0-9a-f]{64})")
_CUDA_LOCAL_RE = re.compile(r"\+cu\d+$")
_COMMENT_RE = re.compile(r"(^|\s)#.*$")


class ReproduceLockError(RuntimeError):
    """A lock, overlay or resolve does not match its manifest or the locks it must agree with."""


class LockEntry(NamedTuple):
    """One pinned requirement: normalised name, ``==`` version or ``@`` URL, marker text and sha256 hashes."""

    name: str
    pin: str
    is_url: bool
    marker: str
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
    lines = (_COMMENT_RE.sub("", line).strip() for line in joined.splitlines())
    return [line for line in lines if line]


def _is_index_option(line: str) -> bool:
    """Whether ``line`` is an index or find-links option; any other option line is refused."""
    if not line.startswith("-"):
        return False
    if re.split(r"[\s=]", line, maxsplit=1)[0] in INDEX_OPTIONS:
        return True
    raise ReproduceLockError(f"option line refused: {line!r}")


def lock_entries(text: str) -> list[LockEntry]:
    """Parse a pinned requirements file: a uv lock, an export, a resolve or a freeze.

    Index and find-links option lines are skipped; every other option line is refused.

    Args:
        text: The file's contents; comments and backslash continuations are allowed.

    Returns:
        One entry per requirement line, in file order.

    Raises:
        ReproduceLockError: If a line is an ``-e``, ``-r``, ``-c`` or other option, or not an ``==``/``@`` pin.
    """
    entries = []
    for line in _logical_lines(text):
        if _is_index_option(line):
            continue
        match = _LINE_RE.match(_HASH_RE.sub("", line).strip())
        if not match:
            raise ReproduceLockError(f"not a pinned requirement: {line!r}")
        name, op, pin, marker = match.groups()
        hashes = frozenset(_HASH_RE.findall(line))
        entries.append(LockEntry(normalize_name(name), pin, op == "@", " ".join((marker or "").split()), hashes))
    return entries


def entry_version(entry: LockEntry) -> str:
    """Return an entry's full version, local tag included.

    Args:
        entry: A parsed requirement.

    Returns:
        The ``==`` version, or the version in a ``.whl`` URL's file name.

    Raises:
        ReproduceLockError: If a URL pin is not a wheel over ``file://``, ``http://`` or ``https://``, or the
            wheel's project name is not the entry's name.
    """
    if not entry.is_url:
        return entry.pin
    url = urllib.parse.urlsplit(entry.pin)
    filename = urllib.parse.unquote(url.path.rsplit("/", 1)[-1])
    parts = filename.split("-")
    if url.scheme not in ("file", "http", "https") or not filename.endswith(".whl") or len(parts) < 5:
        raise ReproduceLockError(f"{entry.name} is not pinned to a wheel: {entry.pin}")
    if normalize_name(parts[0]) != entry.name:
        raise ReproduceLockError(f"{entry.name} is pinned to a wheel of {parts[0]}: {entry.pin}")
    return parts[1]


def parse_freeze(text: str) -> dict[str, str]:
    """Parse ``uv pip freeze`` output into ``{name: version}``.

    Args:
        text: Freeze lines: ``name==version``, ``name @ file://…whl`` or ``name @ https://…whl``.

    Returns:
        Normalised names mapped to versions, ``+cuXXX`` dropped.

    Raises:
        ReproduceLockError: For editable installs, source trees, VCS URLs or unpinned lines.
    """
    return {entry.name: _CUDA_LOCAL_RE.sub("", entry_version(entry)) for entry in lock_entries(text)}


def is_torch_family(name: str) -> bool:
    """Whether a distribution belongs in ``torch.lock``: torch, torchvision, torchaudio or a CUDA-variant name."""
    name = normalize_name(name)
    return name in TORCH_NAMES or name.startswith(TORCH_VARIANT_PREFIXES)


def versions_agree(a: str, b: str) -> bool:
    """Compare versions, ignoring a ``+local`` tag only when exactly one side has one.

    Args:
        a: A version, e.g. ``2.8.0+cu128``.
        b: Another version.

    Returns:
        Whether they name the same release.
    """
    if ("+" in a) != ("+" in b):
        return a.split("+", 1)[0] == b.split("+", 1)[0]
    return a == b


def _path(base_dir: Path, rel: str) -> Path:
    path = base_dir / rel
    if not path.resolve().is_relative_to(base_dir.resolve()):
        raise ReproduceLockError(f"{rel} resolves outside {base_dir}")
    return path


def _verify(
    base_dir: Path, rel: str, expected: str, what: str, verified: dict[str, str], contents: dict[str, bytes]
) -> None:
    if rel not in contents:
        contents[rel] = _path(base_dir, rel).read_bytes()
    actual = hashlib.sha256(contents[rel]).hexdigest()
    if actual != expected:
        raise ReproduceLockError(f"{what} {rel}: sha256 {actual} != manifest {expected}")
    verified[rel] = actual


def verify_files(refs: Iterable[FileRef], base_dir: Path) -> dict[str, str]:
    """Check hashed files, such as patches, against their manifest sha256s.

    Args:
        refs: The files, relative to ``base_dir``.
        base_dir: The manifest repository.

    Returns:
        ``{relative path: sha256}`` of the bytes checked.

    Raises:
        ReproduceLockError: If a file resolves outside ``base_dir`` or its sha256 differs.
    """
    verified: dict[str, str] = {}
    contents: dict[str, bytes] = {}
    for ref in refs:
        _verify(base_dir, ref.file, ref.sha256, "file", verified, contents)
    return verified


def requirements_digest_for(env: EnvSpec, base_dir: Path) -> str:
    """Return the ``requirements-sha256`` header value for a requirements-derived body.

    ``requirements.in`` lines are normalised first: continuations joined, comments stripped, inner
    whitespace collapsed. Index options are digested with the requirements; ``-r``, ``-c``, ``-e`` and
    every other option are refused, because the files they name are not hashed.

    Args:
        env: The step's env; its body source must be ``requirements.in``.
        base_dir: The manifest repository.

    Returns:
        :func:`requirements_sha256` over the requirement lines and the env's compile inputs.

    Raises:
        ReproduceLockError: If the body is not requirements-derived, a line is a non-index option, or a
            requirement line repeats.
    """
    body = env.body
    if body.source != "requirements.in" or body.requirements_in is None:
        raise ReproduceLockError("only a requirements.in body has a requirements digest")
    return _requirements_digest(env, _path(base_dir, body.requirements_in).read_text())


def _requirements_digest(env: EnvSpec, text: str) -> str:
    body = env.body
    lines = [" ".join(line.split()) for line in _logical_lines(text)]
    for line in lines:
        _is_index_option(line)
    dupes = sorted({line for line in lines if lines.count(line) > 1})
    if dupes:
        raise ReproduceLockError(f"{body.requirements_in} repeats requirement lines: {dupes}")
    return requirements_sha256(
        lines,
        env.python,
        env.max_cuda,
        env.exclude_newer,
        body.config_file_sha256,
        body.overrides_sha256,
        env.extra_no_emit,
    )


def _group(entries: Iterable[LockEntry]) -> dict[str, list[LockEntry]]:
    out: dict[str, list[LockEntry]] = {}
    for entry in entries:
        out.setdefault(entry.name, []).append(entry)
    return out


def _overlay_problems(
    env: EnvSpec,
    overlay: list[LockEntry],
    base: list[LockEntry],
    torch_names: set[str],
    in_family: Callable[[str], bool],
) -> list[str]:
    pinned = _group(base)
    held_out = {normalize_name(s.name) for s in env.sdist_allow} | {normalize_name(n) for n in env.extra_no_emit}
    problems = []
    for name, entries in sorted(_group(overlay).items()):
        if name in held_out:
            problems.append(f"{name}: held out of the base locks (sdist_allow or extra_no_emit)")
        elif in_family(name) and name not in torch_names:
            problems.append(f"{name}: torch-family name absent from torch.lock")
        elif name not in pinned:
            continue
        elif len(pinned[name]) > 1 or len(entries) > 1:
            problems.append(f"{name}: pinned in more than one entry; markers are not evaluated")
        else:
            mine, theirs = entries[0], pinned[name][0]
            if (mine.pin, mine.marker, mine.hashes) != (theirs.pin, theirs.marker, theirs.hashes):
                problems.append(f"{name}: {mine.pin} differs from base {theirs.pin} or its marker or hashes differ")
    return problems


def _torch_problems(torch: list[LockEntry], reference: dict[str, list[tuple[str, set[str] | None]]]) -> list[str]:
    names = {e.name for e in torch}
    problems = [f"{e.name}: no --hash in torch.lock" for e in torch if not e.hashes]
    for entry in torch:
        version = entry_version(entry)
        candidates = reference.get(entry.name, [])
        if not any(versions_agree(version, v) and (h is None or entry.hashes <= h) for v, h in candidates):
            problems.append(f"{entry.name}: torch.lock {version}, reference {sorted(v for v, _ in candidates)}")
    problems += [f"{n}: in the reference, absent from torch.lock" for n in sorted(reference) if n not in names]
    return problems


def _resolve_reference(resolve: list[LockEntry]) -> dict[str, list[tuple[str, set[str] | None]]]:
    out: dict[str, list[tuple[str, set[str] | None]]] = {}
    for entry in resolve:
        if is_torch_family(entry.name):
            out.setdefault(entry.name, []).append((entry_version(entry), None))
    return out


def uv_lock_torch_family(packages: Iterable[Mapping[str, Any]], roots: Iterable[str]) -> set[str]:
    """Return the torch family a repository ``uv.lock`` implies.

    The family is the roots plus the torch-family-named packages (``nvidia-*``, ``triton``, …) in the
    transitive closure of the roots' ``dependencies``, over every marker. A CUDA-named package that only
    another project dependency pulls in is not part of it.

    Args:
        packages: The ``[[package]]`` tables of a parsed ``uv.lock``.
        roots: ``torch``, ``torchvision`` and ``torchaudio`` names present in ``torch.lock``.

    Returns:
        Normalised names.
    """
    deps: dict[str, set[str]] = {}
    for package in packages:
        names = {normalize_name(d["name"]) for d in package.get("dependencies", [])}
        deps.setdefault(normalize_name(package["name"]), set()).update(names)
    seen = {normalize_name(r) for r in roots}
    stack = list(seen)
    while stack:
        for dep in deps.get(stack.pop(), set()) - seen:
            seen.add(dep)
            stack.append(dep)
    return {n for n in seen if is_torch_family(n)}


def _uv_lock_reference(
    packages: list[dict[str, Any]], family: set[str]
) -> dict[str, list[tuple[str, set[str] | None]]]:
    out: dict[str, list[tuple[str, set[str] | None]]] = {}
    for package in packages:
        name = normalize_name(package["name"])
        if name in family:
            hashes = {w["hash"].removeprefix("sha256:") for w in package.get("wheels", []) if "hash" in w}
            out.setdefault(name, []).append((package["version"], hashes))
    for name in family - set(out):
        out[name] = []
    return out


def check_integrity(env: EnvSpec, base_dir: Path, uv_lock: Path | None = None) -> dict[str, str]:
    """Check an env's hashed files and pin agreement before anything is installed.

    The checks run in order: lock and sdist-patch sha256s (torch, body, wheels, overlays, build envs);
    the ``requirements.in``, config and overrides sha256s and the recomputed header digest; the
    ``first_resolve`` sha256; overlay pins against the base locks; torch-family pins against
    ``first_resolve`` (``requirements.in`` bodies) or the repository ``uv.lock`` (uv-export bodies).
    Against ``uv.lock`` the torch family is :func:`uv_lock_torch_family`, which also decides which
    names an overlay may not pin; otherwise it is the name rule of :func:`is_torch_family`.

    Each file is read once; the header, pin and torch checks run on the bytes that were hashed. The
    returned sha256s describe those bytes. The installer copies each file into ``{tmp}``,
    checks the copy against this mapping and installs from the copy, never from ``base_dir``.

    Args:
        env: The step's env.
        base_dir: The manifest repository; every path in ``env`` is relative to it and must resolve inside it.
        uv_lock: The checkout's ``uv.lock``; needed for a uv-export body with a torch lock.

    Returns:
        ``{relative path: sha256}`` of every file verified.

    Raises:
        ReproduceLockError: On the first check that fails.
    """
    verified: dict[str, str] = {}
    contents: dict[str, bytes] = {}

    def text(rel: str) -> str:
        return contents[rel].decode()

    locks = [("torch lock", env.torch), ("body lock", env.body), ("wheels lock", env.wheels)]
    locks += [("test lock", env.test), ("notebook_tools lock", env.notebook_tools)]
    locks += [(f"build_env lock of {s.name}", s.build_env) for s in env.sdist_allow]
    for what, ref in locks:
        if ref is not None:
            _verify(base_dir, ref.lock, ref.lock_sha256, what, verified, contents)
    for sdist in env.sdist_allow:
        for patch in sdist.patches:
            _verify(base_dir, patch.file, patch.sha256, f"patch of {sdist.name}", verified, contents)

    body = env.body
    if body.source == "requirements.in":
        if body.requirements_in is None or body.requirements_in_sha256 is None:
            raise ReproduceLockError("a requirements.in body needs requirements_in and its sha256")
        _verify(base_dir, body.requirements_in, body.requirements_in_sha256, "requirements.in", verified, contents)
        for what, rel, sha in (
            ("config file", body.config_file, body.config_file_sha256),
            ("overrides", body.overrides, body.overrides_sha256),
        ):
            if rel is not None and sha is not None:
                _verify(base_dir, rel, sha, what, verified, contents)
        recorded = parse_header(text(body.lock)).get("requirements-sha256")
        expected = _requirements_digest(env, text(body.requirements_in))
        if recorded != expected:
            raise ReproduceLockError(
                f"body lock {body.lock}: header requirements-sha256 {recorded} != recomputed {expected}"
            )

    if env.first_resolve is not None:
        _verify(base_dir, env.first_resolve.file, env.first_resolve.sha256, "first_resolve", verified, contents)

    torch = lock_entries(text(env.torch.lock)) if env.torch is not None else []
    torch_names = {e.name for e in torch}
    base = lock_entries(text(body.lock)) + torch
    in_family: Callable[[str], bool] = is_torch_family
    uv_packages: list[dict[str, Any]] = []
    family: set[str] = set()
    if env.torch is not None and body.source == "uv-export" and uv_lock is not None:
        uv_packages = tomllib.loads(uv_lock.read_text()).get("package", [])
        family = uv_lock_torch_family(uv_packages, torch_names & TORCH_NAMES)
        in_family = family.__contains__
    for what, ref in (("test", env.test), ("notebook_tools", env.notebook_tools)):
        if ref is not None:
            problems = _overlay_problems(env, lock_entries(text(ref.lock)), base, torch_names, in_family)
            if problems:
                raise ReproduceLockError(f"{what} overlay refused: {problems}")

    if env.torch is None:
        return verified
    if body.source == "requirements.in":
        if env.first_resolve is None:
            raise ReproduceLockError("a torch lock with a requirements.in body needs first_resolve")
        reference = _resolve_reference(lock_entries(text(env.first_resolve.file)))
    else:
        if uv_lock is None:
            raise ReproduceLockError("a uv-export body with a torch lock needs the repository uv.lock to check against")
        reference = _uv_lock_reference(uv_packages, family)
    problems = _torch_problems(torch, reference)
    if problems:
        raise ReproduceLockError(f"torch lock {env.torch.lock} disagrees: {problems}")
    return verified
