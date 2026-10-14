"""Registered inputs: registry entries, the per-file manifest format and the manifest writer.

A **registered input** is a frozen tree under a profile root, described by a registry entry
``<base_dir>/inputs/<id>.yaml`` (`RegisteredInput`) that names the tree, its per-file manifest and that
manifest's sha256. A **manifest** is a TSV: one ``# root=…`` comment line, the header `MANIFEST_COLUMNS`
(optionally followed by ``sha256``), then one row per file or symlink, sorted by relative path. A
symlink is recorded by its target and never followed; a file by its size, ``mtime_ns`` and xxh3-128.

Errors are an `InputError` whose message is a fixed code only.
"""

from __future__ import annotations

import datetime
import hashlib
import os
import re
import socket
import stat
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Annotated, Literal, NamedTuple

import pydantic
import xxhash
import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints, field_validator

from senselab.utils import fastio
from senselab.utils.reproduce import digest
from senselab.utils.reproduce.spec import Name, RelPath, Sha256

REGISTRY_DIR = "inputs"
MANIFEST_COLUMNS = ("relpath", "size", "mtime_ns", "xxh3_128", "link_target")
SHA256_COLUMN = "sha256"
CODE = "E_INPUT_REGISTRY"
_CHUNK = 8 << 20
_XXH = re.compile(r"^[0-9a-f]{32}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ROW = re.compile(r"^([A-Z])([0-9]{1,2})(?:-([A-Z])([0-9]{1,2}))?$")

Licence = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.+-]{0,63}$")]


def _row_ref(value: str) -> str:
    match = _ROW.match(value)
    if match is None:
        raise ValueError(f"not a row reference: {value!r}")
    letter, start, end_letter, end = match.groups()
    if end_letter is not None and (end_letter != letter or int(end) <= int(start)):
        raise ValueError(f"a row range keeps its letter and ascends: {value!r}")
    return value


RowRef = Annotated[str, AfterValidator(_row_ref)]


class InputError(Exception):
    """A registry entry or manifest could not be used; the message is the fixed code only."""

    def __init__(self, code: str) -> None:
        """Store the code as both the message and ``code``.

        Args:
            code: ``E_INPUT_REGISTRY``.
        """
        super().__init__(code)
        self.code = code


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ManifestRef(_Model):
    """Where a registered input's manifest lives, and its sha256."""

    root: Name
    relpath: RelPath
    sha256: Sha256


class OriginalRef(_Model):
    """The tree a hardlinked derivative mirrors: the same relpaths under another root."""

    root: Name
    relpath: RelPath


class RegisteredInput(_Model):
    """One registry entry: a frozen tree under a profile root, its manifest and how it may be used.

    ``rows`` are the result rows the input feeds (``C1``, ranges such as ``C1-C5`` that keep their letter
    and ascend); the YAML scalar ``none`` means no rows. ``include``/``exclude`` select files in the
    `digest` pattern grammar; ``extras: check`` refuses files under the tree that the manifest lacks.
    """

    id: Name
    root: Name
    relpath: RelPath
    manifest: ManifestRef
    include: list[str] = []
    exclude: list[str] = []
    extras: Literal["check", "ignore"] = "check"
    rung: Literal["a", "a-prime", "b", "c", "d", "e"]
    rows: list[RowRef]
    restricted: bool
    licence: Licence
    category: Literal[
        "derivative",
        "restricted-derivative",
        "sourcedata",
        "code",
        "model",
        "external-pretraining",
        "external-validation",
    ]
    original: OriginalRef | None = None

    @field_validator("rows", mode="before")
    @classmethod
    def _none_is_empty(cls, value: object) -> object:
        return [] if value == "none" else value


def registry_path(base_dir: Path, input_id: str) -> Path:
    """``<base_dir>/inputs/<id>.yaml``."""
    return Path(base_dir) / REGISTRY_DIR / f"{input_id}.yaml"


def read_registry_entry(base_dir: Path, input_id: str) -> tuple[RegisteredInput, str]:
    """Read one registry entry and the sha256 of its file.

    Args:
        base_dir: The manifest repository.
        input_id: The input id; the file is ``inputs/<id>.yaml``.

    Returns:
        ``(entry, sha256 of the file's bytes)``.

    Raises:
        InputError: ``E_INPUT_REGISTRY`` for a missing or unreadable file, an entry that does not
            validate, or an ``id`` other than the file stem.
    """
    try:
        data = registry_path(base_dir, input_id).read_bytes()
        entry = RegisteredInput.model_validate(yaml.safe_load(data))
    except (OSError, ValueError, yaml.YAMLError, pydantic.ValidationError):
        raise InputError(CODE) from None
    if entry.id != input_id:
        raise InputError(CODE)
    return entry, hashlib.sha256(data).hexdigest()


def load_registry(base_dir: Path, ids: Iterable[str]) -> dict[str, RegisteredInput]:
    """Read the registry entries of ``ids`` from ``<base_dir>/inputs/``.

    Raises:
        InputError: As `read_registry_entry`.
    """
    return {i: read_registry_entry(base_dir, i)[0] for i in ids}


class Row(NamedTuple):
    """One manifest row: a file (size, mtime, hash) or a symlink (target); ``sha256`` when the column is present."""

    relpath: str
    size: int | None
    mtime_ns: int | None
    xxh3_128: str | None
    link_target: str | None
    sha256: str | None


def _int(text: str) -> int:
    if not text.isdigit():
        raise ValueError(text)
    return int(text)


def _parse_row(fields: Sequence[str], with_sha: bool) -> Row:
    rel, size, mtime, xxh, link = fields[:5]
    sha = fields[5] if with_sha else ""
    if not rel:
        raise ValueError("empty relpath")
    if link:
        if size or mtime or xxh or sha:
            raise ValueError("a link row carries only its target")
        return Row(rel, None, None, None, link, None)
    if not _XXH.match(xxh) or (with_sha and not _HEX64.match(sha)):
        raise ValueError("bad hash")
    return Row(rel, _int(size), _int(mtime), xxh, None, sha or None)


def read_manifest(path: Path, expected_sha256: str) -> list[Row]:
    """Read a manifest whose bytes must have the registered sha256.

    Args:
        path: The manifest file.
        expected_sha256: The registry entry's ``manifest.sha256``.

    Returns:
        The rows, in file order.

    Raises:
        InputError: ``E_INPUT_REGISTRY`` for a missing or unreadable file, a sha256 other than
            ``expected_sha256``, a missing ``# root=`` line or header, a malformed row, or an ``ERR:`` hash.
    """
    try:
        data = Path(path).read_bytes()
    except OSError:
        raise InputError(CODE) from None
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise InputError(CODE)
    try:
        lines = data.decode().split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        if len(lines) < 2 or not lines[0].startswith("# root="):
            raise ValueError("no root line")
        header = tuple(lines[1].split("\t"))
        with_sha = header == (*MANIFEST_COLUMNS, SHA256_COLUMN)
        if header != MANIFEST_COLUMNS and not with_sha:
            raise ValueError("bad header")
        rows = []
        for line in lines[2:]:
            fields = line.split("\t")
            if len(fields) != len(header):
                raise ValueError("bad row")
            rows.append(_parse_row(fields, with_sha))
    except (UnicodeDecodeError, ValueError):
        raise InputError(CODE) from None
    return rows


def _hashes(path: str, sha256: bool) -> tuple[str, str]:
    xxh = xxhash.xxh3_128()
    sha = hashlib.sha256() if sha256 else None
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK):
            xxh.update(chunk)
            if sha is not None:
                sha.update(chunk)
    return xxh.hexdigest(), "" if sha is None else sha.hexdigest()


def _row(path: str, rel: str, sha256: bool) -> tuple[str, ...]:
    tail = ("",) if sha256 else ()
    try:
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode):
            return (rel, "", "", "", os.readlink(path), *tail)
        xxh, sha = _hashes(path, sha256)
        return (rel, str(st.st_size), str(st.st_mtime_ns), xxh, "", *((sha,) if sha256 else ()))
    except OSError as error:
        return (rel, "", "", f"ERR:{type(error).__name__}", "", *tail)


def _selected(rel: str, include: Sequence[str], exclude: Sequence[str]) -> bool:
    if include and not any(digest.pattern_matches(rel, p) for p in include):
        return False
    return not any(digest.pattern_matches(rel, p) for p in exclude)


def _starts(root: str, include: Sequence[str]) -> tuple[list[str], list[str]]:
    """The literal includes to ``lstat`` and the directories to walk, pruned to the includes' fixed prefixes."""
    if not include:
        return [], [""]
    literal = [p for p in include if not digest._WILD.search(p)]
    files = [p for p in literal if not os.path.isdir(os.path.join(root, p)) or os.path.islink(os.path.join(root, p))]
    bases = {p for p in literal if p not in files} | {digest._base(p) for p in include if digest._WILD.search(p)}
    walks = sorted(b for b in bases if not any(o != b and (o == "" or b.startswith(o + "/")) for o in bases))
    return files, walks


def write_manifest(
    root: str | os.PathLike[str],
    out: str | os.PathLike[str],
    *,
    include: Sequence[str] = (),
    exclude: Sequence[str] = (),
    sha256: bool = False,
) -> dict[str, int]:
    """Write the manifest of a tree, atomically; nothing is printed.

    Args:
        root: The tree's top; relpaths are relative to it.
        out: The manifest file to write.
        include: Patterns (the `digest` grammar) selecting what is recorded; empty records everything. A
            pattern with no wildcard is ``lstat``-ed directly (a literal directory is walked whole), and a
            wildcard pattern walks only from its wildcard-free leading directories.
        exclude: Patterns whose matches are left out.
        sha256: Also record each file's sha256, as a sixth column.

    Returns:
        ``{"entries", "links", "unreadable", "walk_errors", "bytes"}`` counts.

    Raises:
        InputError: ``E_INPUT_REGISTRY`` for a pattern outside the grammar or a root that is not a directory.
    """
    top = os.fspath(root)
    try:
        for pattern in (*include, *exclude):
            digest._check_pattern(pattern)
    except digest.DigestError:
        raise InputError(CODE) from None
    if not os.path.isdir(top):
        raise InputError(CODE)
    patterns = [
        f"{p}/**" if p and not digest._WILD.search(p) and os.path.isdir(os.path.join(top, p)) else p for p in include
    ]
    files, walks = _starts(top, include)
    errors: list[str] = []
    found: dict[str, tuple[str, ...]] = {}

    def visit(path: str) -> tuple[str, ...] | None:
        rel = os.path.relpath(path, top).replace(os.sep, "/")
        return _row(path, rel, sha256) if _selected(rel, patterns, exclude) else None

    for literal in files:
        path = os.path.join(top, literal)
        if os.path.lexists(path) and (row := visit(path)) is not None:
            found[row[0]] = row
    for base in walks:
        start = os.path.join(top, base) if base else top
        if not os.path.isdir(start):
            continue
        for row in fastio.map_files(start, visit, links_as_files=True, on_error=lambda p, e: errors.append(p)):
            if row is not None:
                found[row[0]] = row
    rows = [found[k] for k in sorted(found)]
    header = (*MANIFEST_COLUMNS, SHA256_COLUMN) if sha256 else MANIFEST_COLUMNS
    when = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    text = f"# root={top} date={when} host={socket.gethostname()} xxhash={xxhash.VERSION}\n"
    text += "\t".join(header) + "\n" + "".join("\t".join(r) + "\n" for r in rows)
    _atomic_write(Path(out), text.encode())
    return {
        "entries": len(rows),
        "links": sum(1 for r in rows if r[4]),
        "unreadable": sum(1 for r in rows if r[3].startswith("ERR:")),
        "walk_errors": len(errors),
        "bytes": sum(int(r[1]) for r in rows if r[1]),
    }


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
