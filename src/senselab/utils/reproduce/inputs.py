"""Registered inputs: registry entries, the per-file manifest format and the manifest writer.

A **registered input** is a frozen tree under a profile root, described by a registry entry
``<base_dir>/inputs/<id>.yaml`` (`RegisteredInput`) that names the tree, its per-file manifest and that
manifest's sha256. A **manifest** is a TSV: one ``# root=…`` comment line, the header `MANIFEST_COLUMNS`
(optionally followed by ``sha256``), then one row per file or symlink, sorted by relative path. A
symlink is recorded by its target and never followed; a file by its size, ``mtime_ns`` and xxh3-128.

`precheck` is the gate's cheap check of a registered input against its manifest: ``lstat`` only, no file
content. It also applies the link rule: a link row's target, resolved against the link's own directory,
must lie under the input itself or under another registered input the step declares.

`verify` re-hashes a registered input against its manifest. `append_record` stores the outcome in
``<verify dir>/<id>.json`` (`verify_dir`) under a lock, with an accepted-mtime overlay
``<id>.mtime-overlay.tsv`` when only mtimes drifted. `verified_codes` gives ``E_INPUT_UNVERIFIED`` unless
the latest record is current, ``ok`` and for the registered manifest.

Errors are an `InputError` whose message is a fixed code only.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import socket
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Annotated, Literal, NamedTuple, TypeVar

import pydantic
import xxhash
import yaml
from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, StringConstraints, field_validator

from senselab.utils import fastio
from senselab.utils.reproduce import build, digest
from senselab.utils.reproduce.spec import Name, Profile, RelPath, Sha256

REGISTRY_DIR = "inputs"
MANIFEST_COLUMNS = ("relpath", "size", "mtime_ns", "xxh3_128", "link_target")
SHA256_COLUMN = "sha256"
CODE = "E_INPUT_REGISTRY"
MISSING = "E_INPUT_MISSING"
MISMATCH = "E_INPUT_MISMATCH"
LINK_UNREGISTERED = "E_INPUT_LINK_UNREGISTERED"
UNVERIFIED = "E_INPUT_UNVERIFIED"
DESCRIPTION = "dataset_description.json"
VERIFY_DIR = "input-verify"
OVERLAY_SUFFIX = ".mtime-overlay.tsv"
OVERLAY_HEADER = "relpath\tmtime_ns"
RECORD_MODE = 0o660
MAX_AGE = datetime.timedelta(days=31)
FUTURE_SKEW = datetime.timedelta(minutes=10)
RECORD = "E_INPUT_RECORD"
VERIFY_THREADS = 32
COUNT_NAMES = ("files", "missing", "size", "hash", "mtime_only", "links", "extra", "links_out", "unlinked", "paired")
"""The counts of a verification, in the order ``verify-inputs`` prints them."""
_CHUNK = 8 << 20
_XXH = re.compile(r"^[0-9a-f]{32}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ROW = re.compile(r"^([A-Z])([0-9]{1,2})(?:-([A-Z])([0-9]{1,2}))?$")

T = TypeVar("T")

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


def check_patterns(patterns: list[str]) -> list[str]:
    """Refuse a pattern outside the `digest` grammar (absolute, empty, or with an empty, ``.`` or ``..`` part).

    Raises:
        ValueError: For such a pattern.
    """
    try:
        for pattern in patterns:
            digest._check_pattern(pattern)
    except digest.DigestError:
        raise ValueError("a pattern is outside the digest grammar") from None
    return patterns


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

    @field_validator("include", "exclude")
    @classmethod
    def _patterns(cls, value: list[str]) -> list[str]:
        return check_patterns(value)

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
            ``expected_sha256``, a missing ``# root=`` line or header, a malformed row, an ``ERR:`` hash, or a
            relpath that is absolute, has an empty, ``.`` or ``..`` component, or is not strictly ascending.
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
        rows: list[Row] = []
        for line in lines[2:]:
            fields = line.split("\t")
            if len(fields) != len(header):
                raise ValueError("bad row")
            row = _parse_row(fields, with_sha)
            parts = row.relpath.split("/")
            if row.relpath.startswith("/") or any(p in ("", ".", "..") for p in parts):
                raise ValueError("bad relpath")
            if rows and row.relpath <= rows[-1].relpath:
                raise ValueError("relpaths not strictly ascending")
            rows.append(row)
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


def _lstat_path(top: str, rel: str) -> tuple[str, str]:
    """``lstat`` each component of ``rel`` under ``top``, never following a symlink.

    Returns:
        ``("link", prefix)`` at the first symlinked component, ``("missing", prefix)`` where the path
        stops existing (or a non-final component is not a directory), else ``("dir" | "file", rel)``.

    Raises:
        OSError: When a component cannot be ``lstat``-ed for another reason.
    """
    parts = rel.split("/") if rel else []
    for index in range(len(parts)):
        prefix = "/".join(parts[: index + 1])
        try:
            st = os.lstat(os.path.join(top, prefix))
        except (FileNotFoundError, NotADirectoryError):
            return "missing", prefix
        if stat.S_ISLNK(st.st_mode):
            return "link", prefix
        if index < len(parts) - 1 and not stat.S_ISDIR(st.st_mode):
            return "missing", prefix
        if index == len(parts) - 1:
            return ("dir" if stat.S_ISDIR(st.st_mode) else "file"), rel
    return "dir", rel


def _collect(
    top: str,
    include: Sequence[str],
    exclude: Sequence[str],
    visit: Callable[[str, str], T],
    *,
    strict: bool = True,
) -> tuple[dict[str, T], list[str]]:
    """Select the files and links of a tree as `write_manifest` does, and apply ``visit(path, rel)`` to each.

    Args:
        top: The tree's top.
        include: Selecting patterns (empty: everything).
        exclude: Patterns whose matches are left out.
        visit: Applied to each selected entry, inside the walk's threads.
        strict: Refuse a literal include that is missing or cannot be ``lstat``-ed; otherwise skip it.

    Returns:
        ``({rel: visit result}, the paths whose listing or ``lstat`` failed)``.

    Raises:
        InputError: ``E_INPUT_REGISTRY`` for a pattern outside the grammar, a top that is not a directory,
            or (``strict``) a literal include that is missing or cannot be ``lstat``-ed.
    """
    try:
        for pattern in (*include, *exclude):
            digest._check_pattern(pattern)
    except digest.DigestError:
        raise InputError(CODE) from None
    if not os.path.isdir(top):
        raise InputError(CODE)
    patterns: list[str] = []
    files: list[str] = []
    bases: set[str] = set() if include else {""}
    links: set[str] = set()
    errors: list[str] = []
    for pattern in include:
        literal = not digest._WILD.search(pattern)
        try:
            kind, at = _lstat_path(top, pattern if literal else digest._base(pattern))
        except OSError:
            if literal and strict:
                raise InputError(CODE) from None
            errors.append(pattern)
            continue
        if kind == "missing" and literal:
            if strict:
                raise InputError(CODE)
            continue
        if kind == "link":
            links.add(at)
        elif kind == "dir":
            bases.add(at)
            patterns.append(f"{pattern}/**" if literal else pattern)
        elif kind == "file":
            files.append(at)
            patterns.append(pattern)
    walks = sorted(b for b in bases if not any(o != b and (o == "" or b.startswith(o + "/")) for o in bases))
    found: dict[str, T] = {}

    def selected(path: str) -> tuple[str, T] | None:
        rel = os.path.relpath(path, top).replace(os.sep, "/")
        return (rel, visit(path, rel)) if _selected(rel, patterns, exclude) else None

    for rel in sorted(links):
        if _selected(rel, (), exclude):
            found[rel] = visit(os.path.join(top, rel), rel)
    for rel_file in files:
        if (item := selected(os.path.join(top, rel_file))) is not None:
            found[item[0]] = item[1]
    for base in walks:
        start = os.path.join(top, base) if base else top
        for item in fastio.map_files(start, selected, links_as_files=True, on_error=lambda p, e: errors.append(p)):
            if item is not None:
                found[item[0]] = item[1]
    return found, errors


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
            wildcard pattern walks only from its wildcard-free leading directories. Each component of a
            literal path or a walk's base is ``lstat``-ed; a symlinked component is recorded as a link row
            and not followed.
        exclude: Patterns whose matches are left out.
        sha256: Also record each file's sha256, as a sixth column.

    Returns:
        ``{"entries", "links", "unreadable", "walk_errors", "bytes"}`` counts.

    Raises:
        InputError: ``E_INPUT_REGISTRY`` for a pattern outside the grammar, a root that is not a directory,
            or a literal include that is missing or cannot be ``lstat``-ed.
    """
    top = os.fspath(root)
    found, errors = _collect(top, include, exclude, lambda path, rel: _row(path, rel, sha256))
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


def _atomic_write(path: Path, data: bytes, mode: int | None = None) -> None:
    """Write ``data`` through a temporary file and a rename; ``mode`` defaults to ``0o666`` less the umask."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is None:
            mask = os.umask(0)
            os.umask(mask)
            mode = 0o666 & ~mask
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# --- the pre-check ---------------------------------------------------------------------------------------


def input_top(inp: RegisteredInput, profile: Profile) -> Path:
    """``<profile root>/<relpath>`` of a registered input.

    Raises:
        InputError: ``E_INPUT_REGISTRY`` when the profile has no such root.
    """
    if inp.root not in profile.roots:
        raise InputError(CODE)
    return Path(os.path.normpath(profile.roots[inp.root] / inp.relpath))


def original_top(inp: RegisteredInput, profile: Profile) -> Path | None:
    """The tree a hardlinked derivative mirrors, or None.

    Raises:
        InputError: ``E_INPUT_REGISTRY`` when the profile has no such root.
    """
    if inp.original is None:
        return None
    if inp.original.root not in profile.roots:
        raise InputError(CODE)
    return Path(os.path.normpath(profile.roots[inp.original.root] / inp.original.relpath))


def manifest_rows(inp: RegisteredInput, profile: Profile) -> list[Row]:
    """The rows of a registered input's manifest, checked against the registered sha256.

    Raises:
        InputError: As `read_manifest`, or when the profile has no such root.
    """
    if inp.manifest.root not in profile.roots:
        raise InputError(CODE)
    return read_manifest(profile.roots[inp.manifest.root] / inp.manifest.relpath, inp.manifest.sha256)


def _under(path: str, parent: str) -> bool:
    return path == parent or path.startswith(parent.rstrip("/") + "/")


def link_resolves_inside(link: str, target: str, tops: Sequence[str]) -> bool:
    """Whether a link's target, resolved lexically against the link's own directory, lies under one of ``tops``."""
    resolved = os.path.normpath(os.path.join(os.path.dirname(link), target))
    return any(_under(resolved, top) for top in tops)


def _tops(entries: Iterable[RegisteredInput], profile: Profile) -> list[str]:
    return [os.fspath(input_top(e, profile)) for e in entries if e.root in profile.roots]


class Scan(NamedTuple):
    """What an ``lstat`` pass over a manifest found; ``present`` lists file rows whose size still matches."""

    files: int
    links: int
    missing: int
    size_diff: int
    mtime_diff: int
    link_diff: int
    links_out: int
    unlinked: int
    paired: int
    present: list[tuple[Row, str, int]]


def _parent_state(top: str, rel: str, seen: dict[str, str]) -> str:
    """``ok``, ``missing`` or ``changed`` (a component is a symlink or not a directory) for ``rel``'s parent."""
    parts = rel.split("/")[:-1]
    for index in range(len(parts)):
        prefix = "/".join(parts[: index + 1])
        state = seen.get(prefix)
        if state is None:
            try:
                st = os.lstat(os.path.join(top, prefix))
                state = "ok" if stat.S_ISDIR(st.st_mode) else "changed"
            except (FileNotFoundError, NotADirectoryError):
                state = "missing"
            except OSError:
                state = "changed"
            seen[prefix] = state
        if state != "ok":
            return state
    return "ok"


def scan(
    inp: RegisteredInput,
    profile: Profile,
    rows: Sequence[Row],
    tops: Sequence[str],
    overlay: Mapping[str, int] | None = None,
) -> Scan:
    """``lstat`` every manifest row under the input (and its original), reading no file content.

    Args:
        inp: The registered input.
        profile: The site profile.
        rows: Its manifest rows (`manifest_rows`).
        tops: The trees a link may point into, besides the input itself.
        overlay: Accepted ``mtime_ns`` per relpath, which an mtime difference may equal.

    Returns:
        The counts: ``missing`` (no entry, or a parent dir gone), ``size_diff`` (a file row whose entry is no
        longer a regular file under real directories, or has another size), ``mtime_diff`` (same size,
        another mtime, not the accepted one), ``link_diff`` (a link row whose entry is no longer that link),
        ``links_out`` (a link row whose recorded target lies outside the input and ``tops``), and, for an
        input with ``original``, ``paired`` (file rows other than the top-level ``dataset_description.json``)
        and ``unlinked`` (paired rows whose original is missing or another inode).

    Raises:
        InputError: ``E_INPUT_REGISTRY`` when a root is not in the profile.
    """
    top = os.fspath(input_top(inp, profile))
    original = original_top(inp, profile)
    allowed = [top, *tops]
    accepted = overlay or {}
    seen: dict[str, str] = {}
    files = links = missing = size_diff = mtime_diff = link_diff = links_out = unlinked = paired = 0
    present: list[tuple[Row, str, int]] = []
    for row in rows:
        path = os.path.join(top, row.relpath)
        if row.link_target is not None:
            links += 1
            if not link_resolves_inside(path, row.link_target, allowed):
                links_out += 1
        else:
            files += 1
        state = _parent_state(top, row.relpath, seen)
        if state == "missing":
            missing += 1
            continue
        try:
            st = None if state != "ok" else os.lstat(path)
        except (FileNotFoundError, NotADirectoryError):
            missing += 1
            continue
        except OSError:
            st = None
        if row.link_target is not None:
            try:
                same = st is not None and stat.S_ISLNK(st.st_mode) and os.readlink(path) == row.link_target
            except OSError:
                same = False
            link_diff += not same
            continue
        if st is None or not stat.S_ISREG(st.st_mode) or st.st_size != row.size:
            size_diff += 1
            continue
        if st.st_mtime_ns != row.mtime_ns and accepted.get(row.relpath) != st.st_mtime_ns:
            mtime_diff += 1
        present.append((row, path, st.st_mtime_ns))
        if original is not None and row.relpath != DESCRIPTION:
            paired += 1
            try:
                other = os.lstat(os.path.join(original, row.relpath))
                unlinked += (other.st_dev, other.st_ino) != (st.st_dev, st.st_ino)
            except OSError:
                unlinked += 1
    return Scan(files, links, missing, size_diff, mtime_diff, link_diff, links_out, unlinked, paired, present)


def precheck(
    inp: RegisteredInput,
    profile: Profile,
    step_inputs: Mapping[str, RegisteredInput],
    overlay: Mapping[str, int] | None = None,
) -> list[str]:
    """The gate's pre-check of one registered input: ``lstat`` of every manifest row, no file content.

    Args:
        inp: The registered input.
        profile: The site profile.
        step_inputs: The registered inputs the step declares (``lock_inputs``); a link may point into any of
            them.
        overlay: Accepted ``mtime_ns`` per relpath (from the latest verification record).

    Returns:
        Sorted codes: ``E_INPUT_MISSING`` for a missing row, ``E_INPUT_MISMATCH`` for a size, mtime, link or
        hardlink difference (`scan`), ``E_INPUT_LINK_UNREGISTERED`` for a link whose target lies in no
        declared input, ``E_INPUT_REGISTRY`` when the manifest cannot be read; empty when it passes.
    """
    try:
        rows = manifest_rows(inp, profile)
        found = scan(inp, profile, rows, _tops(step_inputs.values(), profile), overlay)
    except InputError as error:
        return [error.code]
    codes = set()
    if found.missing:
        codes.add(MISSING)
    if found.size_diff or found.mtime_diff or found.link_diff or found.unlinked:
        codes.add(MISMATCH)
    if found.links_out:
        codes.add(LINK_UNREGISTERED)
    return sorted(codes)


def unregistered_links(inp: RegisteredInput, profile: Profile, others: Iterable[RegisteredInput]) -> int:
    """How many link rows of the manifest point outside the input and ``others``; nothing under the input is read.

    Raises:
        InputError: As `manifest_rows`.
    """
    top = os.fspath(input_top(inp, profile))
    tops = [top, *_tops(others, profile)]
    rows = manifest_rows(inp, profile)
    return sum(
        1
        for r in rows
        if r.link_target is not None and not link_resolves_inside(os.path.join(top, r.relpath), r.link_target, tops)
    )


# --- full verification and its records --------------------------------------------------------------------


class VerifyRecord(_Model):
    """One completed verification of a registered input.

    ``counts`` are `COUNT_NAMES`. ``overlay_sha256`` is the sha256 of the accepted-mtime overlay written with
    an ``ok`` record whose ``mtime_only`` is not 0, else None. ``restricted`` is the registry flag at the time.
    """

    manifest_sha256: Sha256
    status: Literal["ok", "mismatch"]
    date: AwareDatetime
    counts: dict[str, int]
    restricted: bool
    overlay_sha256: Sha256 | None = None


class Verification(NamedTuple):
    """A record and the overlay bytes it names (None without an overlay)."""

    record: VerifyRecord
    overlay: bytes | None


def restricted_dir_paths(profile: Profile) -> list[str]:
    """Every restricted dir of the profile, as written and resolved."""
    return sorted({s for d in profile.restricted_dirs for s in (os.path.normpath(d), os.path.realpath(d))})


def in_restricted_dir(inp: RegisteredInput, profile: Profile) -> bool:
    """Whether the input's tree, manifest or original lies in (or holds) a restricted dir, as written or resolved."""
    places = [(inp.root, inp.relpath), (inp.manifest.root, inp.manifest.relpath)]
    if inp.original is not None:
        places.append((inp.original.root, inp.original.relpath))
    dirs = restricted_dir_paths(profile)
    for root, rel in places:
        if root in profile.roots:
            path = profile.roots[root] / rel
            for spelled in (os.path.normpath(path), os.path.realpath(path)):
                if any(_under(spelled, d) or _under(d, spelled) for d in dirs):
                    return True
    return False


def check_flag(inp: RegisteredInput, profile: Profile) -> None:
    """Refuse an input in a restricted dir that is not flagged ``restricted: true``.

    Raises:
        InputError: ``E_INPUT_REGISTRY``.
    """
    if not inp.restricted and in_restricted_dir(inp, profile):
        raise InputError(CODE)


def remove_public_records(input_id: str, profile: Profile) -> None:
    """Remove an input's records and overlay from the public ``input-verify/``, under the records lock."""
    directory = profile.results_root.parent / VERIFY_DIR
    path = _records_path(directory, input_id)
    if not directory.is_dir():
        return
    with build.file_lock(path.with_name(path.name + ".lock")):
        for target in (path, directory / f"{input_id}{OVERLAY_SUFFIX}"):
            target.unlink(missing_ok=True)


def verify_dir(inp: RegisteredInput, profile: Profile) -> Path:
    """Where an input's records live: ``input-verify/`` beside the results root (the restricted one if restricted).

    Raises:
        InputError: ``E_INPUT_REGISTRY`` for a restricted input on a profile with no restricted results root, or
            an unflagged input in a restricted dir (`check_flag`).
    """
    check_flag(inp, profile)
    if not inp.restricted:
        return profile.results_root.parent / VERIFY_DIR
    if profile.restricted_results_root is None:
        raise InputError(CODE)
    return profile.restricted_results_root.parent / VERIFY_DIR


def _xxh(path: str) -> str | None:
    try:
        return _hashes(path, False)[0]
    except OSError:
        return None


def _extras(inp: RegisteredInput, top: str, listed: set[str]) -> int:
    """Selected entries under the input that the manifest lacks; a failed listing counts as one each."""
    found, errors = _collect(top, inp.include, inp.exclude, lambda path, rel: None, strict=False)
    return len(set(found) - listed) + len(errors)


def _overlay_bytes(overlay: Mapping[str, int]) -> bytes:
    lines = [OVERLAY_HEADER, *(f"{rel}\t{overlay[rel]}" for rel in sorted(overlay))]
    return ("\n".join(lines) + "\n").encode()


def verify(
    inp: RegisteredInput,
    profile: Profile,
    *,
    now: datetime.datetime,
    others: Iterable[RegisteredInput] = (),
) -> Verification:
    """Re-hash every file row of a registered input (xxh3-128) and count the differences; nothing is written.

    Args:
        inp: The registered input.
        profile: The site profile.
        now: The record's date.
        others: The other registry entries; a link into one of them is not counted in ``links_out``.

    Returns:
        The verification. ``status`` is ``ok`` iff ``missing``, ``size``, ``hash``, ``links``, ``extra`` and
        ``unlinked`` are 0. ``hash`` counts files whose hash differs or that cannot be read; ``mtime_only``
        files with the recorded hash and another mtime; ``links`` changed link rows; ``extra`` selected
        entries absent from the manifest (0 with ``extras: ignore``); ``links_out`` links outside the input
        and ``others`` (it does not change the status). An ``ok`` record with ``mtime_only`` above 0 carries
        the overlay of the current mtimes.

    Raises:
        InputError: ``E_INPUT_REGISTRY`` when the manifest cannot be read (`manifest_rows`).
    """
    rows = manifest_rows(inp, profile)
    found = scan(inp, profile, rows, _tops(others, profile))
    hashes = list(fastio.ordered_map(lambda item: _xxh(item[1]), found.present, threads=VERIFY_THREADS))
    hash_diff = mtime_only = 0
    overlay: dict[str, int] = {}
    for (row, _, mtime), xxh in zip(found.present, hashes):
        if xxh != row.xxh3_128:
            hash_diff += 1
        elif mtime != row.mtime_ns:
            mtime_only += 1
            overlay[row.relpath] = mtime
    top = os.fspath(input_top(inp, profile))
    extra = 0 if inp.extras == "ignore" or not os.path.isdir(top) else _extras(inp, top, {r.relpath for r in rows})
    counts = dict(
        zip(
            COUNT_NAMES,
            (
                found.files,
                found.missing,
                found.size_diff,
                hash_diff,
                mtime_only,
                found.link_diff,
                extra,
                found.links_out,
                found.unlinked,
                found.paired,
            ),
        )
    )
    ok = not (found.missing or found.size_diff or hash_diff or found.link_diff or extra or found.unlinked)
    data = _overlay_bytes(overlay) if ok and overlay else None
    record = VerifyRecord(
        manifest_sha256=inp.manifest.sha256,
        status="ok" if ok else "mismatch",
        date=now,
        counts=counts,
        restricted=inp.restricted,
        overlay_sha256=None if data is None else hashlib.sha256(data).hexdigest(),
    )
    return Verification(record, data)


def _records_path(directory: Path, input_id: str) -> Path:
    return directory / f"{input_id}.json"


def _load_records(path: Path) -> list[VerifyRecord]:
    """Every record of a records file; none when the file is absent.

    Raises:
        ValueError: When the file exists but cannot be read, or a record does not validate.
    """
    try:
        data = json.loads(path.read_bytes())
    except FileNotFoundError:
        return []
    except OSError as error:
        raise ValueError("unreadable records") from error
    records = data.get("records") if isinstance(data, dict) else None
    if not isinstance(records, list):
        raise ValueError("not a records file")
    try:
        return [VerifyRecord.model_validate(r) for r in records]
    except pydantic.ValidationError as error:
        raise ValueError("a record does not validate") from error


def _order(record: VerifyRecord) -> tuple[datetime.datetime, bool]:
    """Records are ordered by the date their verification started; on a tie a mismatch comes last."""
    return record.date, record.status == "mismatch"


def append_record(directory: Path, input_id: str, verification: Verification) -> None:
    """Append a completed verification to ``<directory>/<id>.json`` under its lock.

    Its overlay is written first, and only when the record is the latest (`latest_record`), so a verification
    that started earlier but finished later never replaces a newer overlay. Both files are written
    atomically with mode ``0660``; the manifest is never touched.

    Raises:
        InputError: ``E_INPUT_RECORD`` when the records file exists but does not parse (it is left as it is).
    """
    path = _records_path(directory, input_id)
    with build.file_lock(path.with_name(path.name + ".lock")):
        try:
            records = _load_records(path)
        except ValueError:
            raise InputError(RECORD) from None
        latest = all(_order(verification.record) >= _order(r) for r in records)
        if verification.overlay is not None and latest:
            _atomic_write(directory / f"{input_id}{OVERLAY_SUFFIX}", verification.overlay, RECORD_MODE)
        data = [*(r.model_dump(mode="json") for r in records), verification.record.model_dump(mode="json")]
        _atomic_write(path, (json.dumps({"records": data}) + "\n").encode(), RECORD_MODE)


def latest_record(directory: Path, input_id: str) -> VerifyRecord | None:
    """The record whose verification started last (a mismatch wins a tie), whatever the order of appending.

    Returns:
        None when there is no record, or the file or any record in it does not validate.
    """
    try:
        records = _load_records(_records_path(directory, input_id))
    except ValueError:
        return None
    return max(records, key=_order) if records else None


def _read_overlay(data: bytes) -> dict[str, int]:
    lines = data.decode().split("\n")
    if lines[-1] == "":
        lines.pop()
    if not lines or lines[0] != OVERLAY_HEADER:
        raise ValueError("bad overlay header")
    overlay: dict[str, int] = {}
    for line in lines[1:]:
        rel, mtime = line.split("\t")
        overlay[rel] = _int(mtime)
    return overlay


def accepted(
    inp: RegisteredInput, profile: Profile, *, now: datetime.datetime
) -> tuple[list[str], dict[str, int] | None]:
    """The submit rule for one input, and the overlay the pre-check may use.

    Returns:
        ``([], overlay or None)`` when the latest record has the registered manifest sha256, ``status: ok``
        and is at most 31 days old (and not dated more than `FUTURE_SKEW` ahead), and its overlay (if any)
        has the recorded sha256; otherwise
        ``(["E_INPUT_UNVERIFIED"], None)``, or ``(["E_INPUT_REGISTRY"], None)`` when `verify_dir` refuses.
    """
    try:
        directory = verify_dir(inp, profile)
    except InputError as error:
        return [error.code], None
    record = latest_record(directory, inp.id)
    if (
        record is None
        or record.manifest_sha256 != inp.manifest.sha256
        or record.status != "ok"
        or now - record.date > MAX_AGE
        or record.date - now > FUTURE_SKEW
    ):
        return [UNVERIFIED], None
    if record.overlay_sha256 is None:
        return [], None
    try:
        data = (directory / f"{inp.id}{OVERLAY_SUFFIX}").read_bytes()
        if hashlib.sha256(data).hexdigest() != record.overlay_sha256:
            return [UNVERIFIED], None
        return [], _read_overlay(data)
    except (OSError, ValueError):
        return [UNVERIFIED], None


def verified_codes(inp: RegisteredInput, profile: Profile, *, now: datetime.datetime) -> list[str]:
    """``E_INPUT_UNVERIFIED`` unless the submit rule holds for the input (`accepted`); empty when it does."""
    return accepted(inp, profile, now=now)[0]


def registry_codes(base_dir: Path, input_id: str, profile: Profile) -> list[str]:
    """Check one registry file without hashing any input file: its types, manifest sha256 and root line.

    Returns:
        ``["E_INPUT_REGISTRY"]`` when the entry does not validate, lies in a restricted dir unflagged
        (`check_flag`), its manifest cannot be read with the registered sha256, or the manifest's ``# root=``
        names neither the input's tree nor its original;
        empty otherwise.
    """
    try:
        entry, _ = read_registry_entry(base_dir, input_id)
        check_flag(entry, profile)
        manifest_rows(entry, profile)
        path = profile.roots[entry.manifest.root] / entry.manifest.relpath
        first = path.read_bytes().split(b"\n", 1)[0].decode()
        if not first.startswith("# root="):
            return [CODE]
        named = os.path.normpath(first.removeprefix("# root=").partition(" date=")[0])
        trees = [os.fspath(input_top(entry, profile))]
        if (other := original_top(entry, profile)) is not None:
            trees.append(os.fspath(other))
    except (InputError, OSError, UnicodeDecodeError):
        return [CODE]
    return [] if named in trees else [CODE]
