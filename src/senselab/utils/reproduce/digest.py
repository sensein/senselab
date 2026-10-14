"""Input-set digests: a content hash over the files a step reads, revealing no file names.

The digest is sha256 over the sorted ``(sha256(relative path), sha256(content), size)`` triples of the
files under a root that match any of a list of patterns. Patterns are relative POSIX globs: ``*``,
``?`` and ``[...]`` match within one path component, and a whole ``**`` component matches any number
of components (including none). Errors carry a fixed code and no path.

Symlinks are asymmetric. A pattern's wildcard-free leading components are resolved by path, so a
symlinked directory named there (or anywhere in a pattern with no wildcard) is followed. Below that
prefix the walk never descends into a symlinked directory, so ``*`` or ``**`` never reaches one.
Symlinked files that match are hashed through the link.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from pathlib import PurePosixPath
from typing import TypeVar

from senselab.utils import fastio

_HEADER = b"senselab-reproduce-input-digest-v1\n"
_CHUNK = 1 << 20
_WILD = re.compile(r"[*?\[]")
T = TypeVar("T")


class DigestError(Exception):
    """A digest could not be taken; the message is the fixed code only."""

    def __init__(self, code: str) -> None:
        """Store the code as both the message and ``code``.

        Args:
            code: ``E_DIGEST_NO_ROOT``, ``E_DIGEST_BAD_PATTERN``, ``E_DIGEST_UNREADABLE`` or
                ``E_DIGEST_EMPTY``.
        """
        super().__init__(code)
        self.code = code


def _component(part: str) -> str:
    out, i = [], 0
    while i < len(part):
        char = part[i]
        if char == "*":
            out.append("[^/]*")
        elif char == "?":
            out.append("[^/]")
        elif char == "[":
            end = part.find("]", i + 2 if part[i + 1 : i + 2] in ("!", "]") else i + 1)
            if end < 0:
                out.append(re.escape(char))
            else:
                body = part[i + 1 : end]
                body = "^" + body[1:] if body.startswith("!") else body
                out.append(f"[{body.replace(chr(92), chr(92) * 2)}]")
                i = end
        else:
            out.append(re.escape(char))
        i += 1
    return "".join(out)


@lru_cache(maxsize=256)
def _compile(pattern: str) -> re.Pattern[str]:
    parts = pattern.split("/")
    out = ""
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        if part == "**":
            out += ".+" if last else "(?:[^/]+/)*"
        else:
            out += _component(part) + ("" if last else "/")
    return re.compile(out)


def _check_pattern(pattern: str) -> None:
    path = PurePosixPath(pattern)
    parts = pattern.split("/")
    if not pattern or path.is_absolute() or "\\" in pattern or "" in parts or "." in parts or ".." in parts:
        raise DigestError("E_DIGEST_BAD_PATTERN")


def pattern_matches(rel: str, pattern: str) -> bool:
    """Say whether a relative POSIX path matches a pattern.

    Args:
        rel: A path relative to the root, with ``/`` separators.
        pattern: A relative glob (see the module docstring).

    Returns:
        True when the whole path matches.
    """
    return _compile(pattern).fullmatch(rel) is not None


def reaches_below(parts: Sequence[str], rel: Sequence[str]) -> bool:
    """Whether pattern components ``parts`` can match a path that runs through every component of ``rel``.

    With ``rel`` empty this is whether the pattern names anything at all; it is not the same as ``rel`` itself
    matching (`pattern_matches`): ``in/*.txt`` reaches below ``in`` but does not match it.
    """
    if not rel:
        return bool(parts)
    if not parts:
        return False
    if parts[0] == "**":
        return True
    return pattern_matches(rel[0], parts[0]) and reaches_below(parts[1:], rel[1:])


def _base(pattern: str) -> str:
    parts = pattern.split("/")
    fixed = []
    for part in parts[:-1]:
        if _WILD.search(part) or part == "**":
            break
        fixed.append(part)
    return "/".join(fixed)


def _excluded(rel: str, exclude: tuple[str, ...]) -> bool:
    return any(rel == e or rel.startswith(e + "/") for e in exclude)


def _hash_file(path: str) -> tuple[str, int]:
    if not stat.S_ISREG(os.stat(path).st_mode):
        raise DigestError("E_DIGEST_UNREADABLE")
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        size = 0
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _select(
    root: str | os.PathLike[str],
    patterns: Iterable[str],
    exclude: Iterable[str],
    exclude_patterns: Iterable[str],
    fn: Callable[[str, str], T],
) -> dict[str, T]:
    """``fn(path, rel)`` for every file under ``root`` matching a pattern, keyed by ``rel``; the shared walk."""
    patterns = list(patterns)
    skip = tuple(e.strip("/") for e in exclude)
    skip_patterns = list(exclude_patterns)
    for pattern in (*patterns, *skip, *skip_patterns):
        _check_pattern(pattern)
    top = os.fspath(root)
    if not os.path.isdir(top):
        raise DigestError("E_DIGEST_NO_ROOT")
    literal = [p for p in patterns if not _WILD.search(p)]
    bases = sorted({_base(p) for p in patterns if _WILD.search(p)})
    bases = [b for b in bases if not any(o != b and (o == "" or b.startswith(o + "/")) for o in bases)]

    def visit(path: str) -> tuple[str, T] | None:
        rel = os.path.relpath(path, top).replace(os.sep, "/")
        if (
            _excluded(rel, skip)
            or any(pattern_matches(rel, p) for p in skip_patterns)
            or not any(pattern_matches(rel, p) for p in patterns)
        ):
            return None
        return rel, fn(path, rel)

    found: dict[str, T] = {}
    try:
        for pattern in literal:
            path = os.path.join(top, pattern)
            if os.path.lexists(path) and not os.path.isdir(path) and (item := visit(path)) is not None:
                found[item[0]] = item[1]
        for base in bases:
            start = os.path.join(top, base) if base else top
            if not os.path.isdir(start):
                continue
            for item in fastio.map_files(start, visit):
                if item is not None:
                    found[item[0]] = item[1]
    except (OSError, fastio.WalkError):
        raise DigestError("E_DIGEST_UNREADABLE") from None
    return found


def matching_files(
    root: str | os.PathLike[str],
    patterns: Iterable[str],
    *,
    exclude: Iterable[str] = (),
    require_nonempty: bool = False,
) -> list[str]:
    """The files under a root that match any pattern, chosen exactly as `input_set_digest` chooses them.

    Args:
        root: The directory the patterns are relative to.
        patterns: Relative globs.
        exclude: Relative subpaths left out, matched by whole path components.
        require_nonempty: Raise ``E_DIGEST_EMPTY`` when no file matches.

    Returns:
        The sorted relative POSIX paths.

    Raises:
        DigestError: As `input_set_digest`; a matching file that is not a regular file (or a dangling
            symlink) gives ``E_DIGEST_UNREADABLE``.
    """

    def regular(path: str, rel: str) -> None:
        if not stat.S_ISREG(os.stat(path).st_mode):
            raise DigestError("E_DIGEST_UNREADABLE")

    found = _select(root, patterns, exclude, (), regular)
    if require_nonempty and not found:
        raise DigestError("E_DIGEST_EMPTY")
    return sorted(found)


def input_set_digest(
    root: str | os.PathLike[str],
    patterns: Iterable[str],
    *,
    exclude: Iterable[str] = (),
    exclude_patterns: Iterable[str] = (),
    require_nonempty: bool = False,
) -> tuple[str, int]:
    """Digest the files under a root that match any pattern.

    Args:
        root: The directory the patterns are relative to.
        patterns: Relative globs; a file matching several is counted once.
        exclude: Relative subpaths whose files are left out (e.g. ``cuda-cache`` under ``{out}``),
            matched by whole path components.
        exclude_patterns: Relative globs, in the pattern grammar, whose matching files are left out (e.g.
            ``**/__pycache__/**``, ``**/*.pyc``).
        require_nonempty: Raise ``E_DIGEST_EMPTY`` when no file matches.

    Returns:
        ``(hex digest, file count)``.

    Raises:
        DigestError: ``E_DIGEST_NO_ROOT`` when the root is not a directory, ``E_DIGEST_BAD_PATTERN`` for
            an absolute, empty or ``..`` pattern or exclusion, ``E_DIGEST_UNREADABLE`` when a matching
            file cannot be read, is not a regular file, or a directory cannot be listed, and
            ``E_DIGEST_EMPTY`` when ``require_nonempty`` and nothing matches. Patterns and exclusions
            with an empty, ``.`` or ``..`` component are refused.
    """

    def entry(path: str, rel: str) -> tuple[str, str, int]:
        content, size = _hash_file(path)
        return hashlib.sha256(rel.encode()).hexdigest(), content, size

    found = _select(root, patterns, exclude, exclude_patterns, entry)
    if require_nonempty and not found:
        raise DigestError("E_DIGEST_EMPTY")
    digest = hashlib.sha256(_HEADER)
    for rel_hash, content, size in sorted(found.values()):
        digest.update(f"{rel_hash}\0{content}\0{size}\n".encode())
    return digest.hexdigest(), len(found)


def digest_unchanged(before: tuple[str, int], after: tuple[str, int]) -> bool:
    """Say whether two digests of the same input set agree.

    Args:
        before: ``input_set_digest``'s result before the step.
        after: Its result after the step.

    Returns:
        True when both the digest and the count are equal.
    """
    return tuple(before) == tuple(after)
