"""A content-addressed result store keyed by logical key, with a recorded execution context (ADR-002, ADR-003).

Layout under `root`:

- `store@4/<d[:2]>/<d>/<f>/`, where `d` is `LogicalKey.digest()` and `f` is
  `ExecutionContext.fingerprint()`, holding:
  - `result.json`: JCS JSON with `format: "result@2.0"`, the key, its digest, the context fingerprint,
    the encoded outputs and their output identity;
  - `context.json`: JCS JSON with `format: "execution-context@1.0"` and the context's fields;
  - `<sha256>.npy`: each array output in canonical `.npy`, named by the sha256 of its bytes.
- `tmp/<uuid>/`: a writer's staging directory, committed by renaming it onto the entry.
- `quarantine/<uuid>`: entries that failed verification, and objects of the wrong kind found at a path
  the store owns, moved aside by rename.

An entry is bound to its address and its context: a reader recomputes the key digest from the stored key
and the fingerprint from the stored context, and requires both, and the digest and fingerprint recorded in
`result.json`, to equal the path it read. A committed entry is read-only (`committed_modes`);
directories that hold entries are `shared_dir_mode`, whatever the umask. World bits are set only when
the `StorePolicy` allows them. An existing `root` is left as it is.

Every directory below `root` is reached through `StorageDir`, so no symlink below the root is followed.
A symlink or a non-directory where a directory of the store belongs is moved into the quarantine and the
directory is made again; one that cannot be moved raises `WrongKindError`, naming it and its owner.

Only a completed read that fails verification moves an entry. An `OSError` while reading or
writing is raised as `StoreUnavailable` (`StorageFull` for `ENOSPC` and `EDQUOT`) and changes nothing.

Outputs are a mapping of JSON values (what `canonical_scalar` accepts, in exact `list`, `tuple` and
`str`-keyed `dict` containers) and plain numpy arrays, nested to any depth. In the encoded form each
array is replaced by `{"$array": <sha256 of its canonical .npy>}`, so `$array` is a reserved key.
The output identity is `"output:" + identity_digest("output@3", encoded form)`.
"""

import errno
import io
import logging
import os
import re
import stat
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import numpy
from pydantic import ValidationError
from senselab_worker import _raw_digest
from senselab_worker.canonical import (
    ARRAY_TOKEN,
    Walk,
    canonical_npy_bytes,
    canonical_scalar,
    jcs_dumps,
    jcs_loads,
)
from senselab_worker.digest import identity_digest
from senselab_worker.ownership import UUID_HEX

from senselab_core.kernel.identity import ExecutionContext, LogicalKey, OutputIdentity, ReusePolicy, acceptable
from senselab_core.runtime.provenance import CHECKSUM_KEY, ProvStore
from senselab_core.runtime.storage import (
    ASIDE_NAME,
    STAGING_DIR_MODE,
    StorageDir,
    StorePolicy,
    StoreUnavailable,
    WrongKindError,
    committed_modes,
    owner_remedy,
    shared_dir_mode,
    storage_fault,
)

logger = logging.getLogger("senselab")

STORE_DIR = "store@4"
STAGING_DIR = "tmp"
QUARANTINE_DIR = "quarantine"
RESULT_FORMAT = "result@2.0"
CONTEXT_FORMAT = "execution-context@1.0"
ARRAY_TAG = ARRAY_TOKEN
COMMIT_ATTEMPTS = 3
"""How many times `put` stages and renames before giving up on a target that keeps turning out corrupt."""

_POLICY_ORDER = (ReusePolicy.EXACT, ReusePolicy.EQUIVALENT, ReusePolicy.TOLERANT)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PREFIX = re.compile(r"[0-9a-f]{2}")
MEMBER_NAME = re.compile(r"result\.json|context\.json|[0-9a-f]{64}\.npy")
"""A file of an entry, written in its staging directory."""
FINGERPRINT_NAME = _SHA256
"""An entry, renamed from its staging directory onto `<key digest>/<context fingerprint>`."""
WRITTEN_NAMES = frozenset({MEMBER_NAME, FINGERPRINT_NAME, ASIDE_NAME})
"""The grammars of the names this module writes (besides `storage.WRITTEN_NAMES`)."""
T = TypeVar("T")


class CorruptEntryError(ValueError):
    """A committed store entry that a completed read found not to be what was written."""


class MisfiledEntryError(CorruptEntryError):
    """A store entry whose stored key or context does not match the address it was read from."""


@dataclass(frozen=True)
class StoredResult:
    """A result read from the store.

    Attributes:
        outputs: The decoded outputs, in their canonical forms (ADR-006): arrays come back as
            little-endian numpy arrays of their shape with `-0.0` as `0.0`, one NaN and bools as 0/1,
            tuples as lists, integral floats as ints, strings in canonical text.
        output_identity: The `output:` identity of the encoded outputs.
        context: The execution context the result was computed in.
        policy_used: The narrowest policy, no wider than the one requested, under which `context`
            may serve the current context: `exact` for an equal fingerprint, `equivalent` for a
            measured equivalence class, `tolerant` otherwise.
    """

    outputs: dict[str, object]
    output_identity: OutputIdentity
    context: ExecutionContext
    policy_used: ReusePolicy


def _encode(value: object, arrays: dict[str, bytes], walk: Walk) -> object:
    if type(value) is numpy.ndarray:
        a = value
        return walk.enter(a, lambda: _array(a, arrays))
    if type(value) is dict:
        d = cast("dict[str, object]", value)
        if ARRAY_TAG in d:
            raise ValueError(f"{ARRAY_TAG!r} is a reserved key in stored outputs")
        return walk.enter(d, lambda: {k: _encode(v, arrays, walk) for k, v in d.items()})
    if type(value) is list or type(value) is tuple:
        items = cast("list[object] | tuple[object, ...]", value)
        return walk.enter(items, lambda: [_encode(v, arrays, walk) for v in items])
    return canonical_scalar(value)


def _array(a: numpy.ndarray, arrays: dict[str, bytes]) -> object:
    data = canonical_npy_bytes(a)
    sha = _raw_digest.bytes_digest("npy@1", data)
    arrays[sha] = data
    return {ARRAY_TAG: sha}


def _tree_identity(tree: object) -> OutputIdentity:
    return "output:" + identity_digest("output@3", tree)


def _serialise(outputs: Mapping[str, object]) -> tuple[object, dict[str, bytes], OutputIdentity]:
    arrays: dict[str, bytes] = {}
    tree = _encode(dict(outputs), arrays, Walk())
    return tree, arrays, _tree_identity(tree)


def output_identity(outputs: Mapping[str, object]) -> OutputIdentity:
    """Return the output identity of `outputs`: `"output:" + identity_digest("output@3", encoded form)`.

    Raises:
        TypeError: on a value that is neither a canonical JSON value nor a plain numpy array.
        ValueError: on a container that contains itself, a reserved `$array` key, a value `jcs_dumps`
            refuses, or an array dtype `canonical_npy_bytes` refuses.
    """
    return _serialise(outputs)[2]


def _require_directory(entry: Path) -> None:
    try:
        mode = os.lstat(entry).st_mode
    except FileNotFoundError as err:
        raise CorruptEntryError(f"store entry {entry} is gone") from err
    if not stat.S_ISDIR(mode):
        raise CorruptEntryError(f"corrupt store entry {entry}: not a directory")


def _read_member(entry: Path, name: str) -> bytes:
    """Return the bytes of the regular file `entry/name`, read through the entry's descriptor.

    The member is neither followed nor waited on; anything but a regular file there, or a gone entry,
    makes the entry corrupt.
    """
    try:
        directory = StorageDir.open_unfollowed(entry)
    except OSError as err:
        if err.errno not in (errno.ENOENT, errno.ENOTDIR, errno.ELOOP):
            raise
        raise CorruptEntryError(f"corrupt store entry {entry}: it is gone or not a directory") from err
    with directory:
        try:
            data = directory.read_regular(name)
        except WrongKindError as err:
            raise CorruptEntryError(f"corrupt store entry {entry}: {name} is not a regular file") from err
    if data is None:
        raise CorruptEntryError(f"corrupt store entry {entry}: {name} is missing")
    return data


def _decode(value: object, entry: Path) -> object:
    if isinstance(value, dict):
        if set(value) == {ARRAY_TAG}:
            sha = value[ARRAY_TAG]
            if not isinstance(sha, str) or not _SHA256.fullmatch(sha):
                raise CorruptEntryError(f"corrupt store entry {entry}: array reference {sha!r} is not a sha256")
            data = _read_member(entry, f"{sha}.npy")
            if _raw_digest.bytes_digest("npy@1", data) != sha:
                raise CorruptEntryError(f"corrupt store entry {entry}: {sha}.npy does not hash to its name")
            try:
                return numpy.load(io.BytesIO(data), allow_pickle=False)
            except OSError as err:
                raise CorruptEntryError(f"corrupt store entry {entry}: {sha}.npy is not an array: {err}") from err
        return {k: _decode(v, entry) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v, entry) for v in value]
    return value


def _write_synced(directory: StorageDir, name: str, data: bytes, mode: int) -> None:
    directory.write_new(name, data, mode, grammar=MEMBER_NAME)


def _read_object(entry: Path, name: str, fmt: str) -> dict[str, Any]:
    path = entry / name
    record = jcs_loads(_read_member(entry, name))
    if not isinstance(record, dict):
        raise CorruptEntryError(f"corrupt store entry: {path} is not a JSON object")
    if record.pop("format", None) != fmt:
        raise CorruptEntryError(f"corrupt store entry: {path} is not {fmt}")
    return record


def _guarded(entry: Path, read: Callable[[], T]) -> T:
    """Run `read`; a failed verification is `CorruptEntryError`, an `OSError` is a storage fault."""
    try:
        return read()
    except CorruptEntryError:
        raise
    except OSError as err:
        raise storage_fault(err, entry) from err
    except (ValueError, KeyError, TypeError, ValidationError) as err:
        raise CorruptEntryError(f"corrupt store entry {entry}: {type(err).__name__}: {err}") from err


def _stored_context(entry: Path) -> ExecutionContext:
    _require_directory(entry)
    return ExecutionContext.model_validate(_read_object(entry, "context.json", CONTEXT_FORMAT))


def _stored_address(entry: Path) -> tuple[str, str, dict[str, Any], ExecutionContext]:
    """Return `(key digest, context fingerprint, result record, context)` as the entry's files state them.

    The key in `result.json` must digest to its recorded `key_digest`, and the context in `context.json`
    must fingerprint to the `context_fingerprint` recorded in `result.json`.
    """
    context = _stored_context(entry)
    record = _read_object(entry, "result.json", RESULT_FORMAT)
    digest = LogicalKey.model_validate(record["key"]).digest()
    if record["key_digest"] != digest:
        raise MisfiledEntryError(f"stored key digest {record['key_digest']!r} is not the digest of the stored key")
    fingerprint = context.fingerprint()
    if record["context_fingerprint"] != fingerprint:
        raise MisfiledEntryError(
            f"misfiled store entry {entry}: result.json was written under context "
            f"{str(record['context_fingerprint'])[:12]}..., context.json holds {fingerprint[:12]}..."
        )
    return digest, fingerprint, record, context


def _bound(entry: Path, digest: str, fingerprint: str, stored_digest: str, stored_fingerprint: str) -> None:
    if (stored_digest, stored_fingerprint) != (digest, fingerprint):
        raise MisfiledEntryError(
            f"misfiled store entry {entry}: holds key {stored_digest[:12]}... context {stored_fingerprint[:12]}..., "
            f"read as key {digest[:12]}... context {fingerprint[:12]}..."
        )


def _read_context(entry: Path, fingerprint: str) -> ExecutionContext:
    """Return the context recorded in `entry`, whose fingerprint must be `fingerprint`.

    Raises:
        CorruptEntryError: the entry is gone, corrupt, or (`MisfiledEntryError`) holds another context.
        StoreUnavailable: the entry could not be read.
    """

    def read() -> ExecutionContext:
        context = _stored_context(entry)
        if context.fingerprint() != fingerprint:
            raise MisfiledEntryError(f"misfiled store entry {entry}: its context is not {fingerprint[:12]}...")
        return context

    return _guarded(entry, read)


def _read_result(entry: Path, *, digest: str, fingerprint: str) -> tuple[str, dict[str, object], ExecutionContext]:
    """Return the output identity, decoded outputs and context of the entry at `entry`.

    The entry must be bound (`_stored_address`) to key digest `digest` and context fingerprint
    `fingerprint`, its outputs must hash to their output identity, and every array file must hash to its name.

    Raises:
        CorruptEntryError: the entry is gone or fails verification; `MisfiledEntryError` when it
            belongs to another address or context.
        StoreUnavailable: the entry could not be read.
    """

    def read() -> tuple[str, dict[str, object], ExecutionContext]:
        stored_digest, stored_fingerprint, record, context = _stored_address(entry)
        _bound(entry, digest, fingerprint, stored_digest, stored_fingerprint)
        tree, identity = record["outputs"], record["output_identity"]
        if not isinstance(identity, str) or _tree_identity(tree) != identity:
            raise CorruptEntryError(f"corrupt store entry {entry}: outputs do not hash to their output identity")
        outputs = _decode(tree, entry)
        if not isinstance(outputs, dict):
            raise CorruptEntryError(f"corrupt store entry {entry}: outputs are not an object")
        return identity, outputs, context

    return _guarded(entry, read)


def _is_fingerprint(name: str) -> bool:
    return _SHA256.fullmatch(name) is not None


class _StoreEntryFormat:
    """The store entry as an addressed format: `<key digest>/<context fingerprint>`."""

    name = "store-entry"

    def address_of(self, path: Path) -> str:
        def read() -> str:
            digest, fingerprint, _, _ = _stored_address(path)
            return f"{digest}/{fingerprint}"

        return _guarded(path, read)

    def address_at(self, path: Path) -> str:
        return f"{path.parent.name}/{path.name}"

    def read(self, path: Path) -> object:
        return _read_result(path, digest=path.parent.name, fingerprint=path.name)[1]


STORE_ENTRY = _StoreEntryFormat()


def _committed_identity(entry: Path, *, digest: str, fingerprint: str) -> str | None:
    """Return the output identity of the entry at `entry`, None if nothing is there; every file is verified.

    Raises:
        CorruptEntryError: something is at `entry` that is not a valid entry for this address.
        StoreUnavailable: the entry could not be read.
    """
    try:
        os.lstat(entry)
    except FileNotFoundError:
        return None
    except OSError as err:
        raise storage_fault(err, entry) from err
    return _read_result(entry, digest=digest, fingerprint=fingerprint)[0]


def _commit(
    key: LogicalKey,
    context: ExecutionContext,
    parent: StorageDir,
    name: str,
    staging_root: StorageDir,
    payload: tuple[object, dict[str, bytes], str],
    policy: StorePolicy,
) -> bool:
    """Stage an entry, make its files read-only, and rename it onto `parent/name`; return False if another was first.

    The entry directory is made read-only after the rename.

    Raises:
        StoreUnavailable: staging or renaming failed for any reason but an occupied target.
    """
    tree, arrays, identity = payload
    file_mode, dir_mode = committed_modes(policy)
    staging_name = uuid.uuid4().hex
    try:
        with staging_root.child(staging_name, create=True, mode=STAGING_DIR_MODE) as staging:
            for sha, data in arrays.items():
                _write_synced(staging, f"{sha}.npy", data, file_mode)
            _write_synced(
                staging,
                "context.json",
                jcs_dumps({"format": CONTEXT_FORMAT, **context.model_dump(mode="json")}),
                file_mode,
            )
            result = {
                "format": RESULT_FORMAT,
                "key": jcs_loads(jcs_dumps(key.model_dump())),
                "key_digest": key.digest(),
                "context_fingerprint": context.fingerprint(),
                "output_identity": identity,
                "outputs": tree,
            }
            _write_synced(staging, "result.json", jcs_dumps(result), file_mode)
            staging.fsync()
        try:
            staging_root.rename(staging_name, parent, name, grammar=FINGERPRINT_NAME)
        except OSError:
            if parent.kind(name) is not None:
                return False
            raise
        _seal(parent, name, dir_mode)
        parent.fsync()
        return True
    except OSError as err:
        raise storage_fault(err, parent.path / name) from err
    finally:
        _discard_staging(staging_root, staging_name)


def _seal(parent: StorageDir, name: str, dir_mode: int) -> None:
    try:
        with parent.child(name) as entry:
            entry.set_mode(dir_mode)
    except (OSError, WrongKindError) as err:
        logger.warning("cannot make store entry %s read-only: %s", parent.path / name, err)


def _discard_staging(staging_root: StorageDir, name: str) -> None:
    try:
        staging_root.remove_owned(name, pattern=UUID_HEX)
    except OSError as err:
        logger.warning("cannot remove staging directory %s: %s", staging_root.path / name, err)


def _listed(directory: StorageDir) -> list[str]:
    """Return the names in `directory`; none if it cannot be listed, which is logged."""
    try:
        return directory.listdir()
    except OSError as err:
        logger.warning("cannot list %s to collect it: %s", directory.path, err)
        return []


def _opened(directory: StorageDir, *parts: str) -> StorageDir | None:
    """Return `directory/parts` opened, None if a part is missing or is not a directory (which is logged)."""
    try:
        return directory.walk(*parts)
    except FileNotFoundError:
        return None
    except WrongKindError as err:
        logger.warning("not collecting %s: %s", err.path, err)
        return None
    except OSError as err:
        logger.warning("cannot open %s to collect it: %s", directory.path.joinpath(*parts), err)
        return None


class ResultStore:
    """Results addressed by logical key, one entry per execution context, safe for concurrent writers.

    Each entry is staged in `root/tmp/<uuid>/` and committed by `os.rename`; the first writer to
    rename wins and later writers discard their staging and return the committed identity. A
    staging directory is never read by `get`. An entry that a completed read finds corrupt or
    misfiled is moved into `root/quarantine/<uuid>` (see `_quarantine`) and never served. `collect`,
    which `put` runs at most once per `policy.collect_interval_s`, removes quarantined entries and
    staging directories older than the policy's ages. An entry found valid but writable is re-sealed.

    Args:
        root: The store root.
        provenance: Where `put`, quarantines and relaxed-policy `get` hits record activities; the
            store owns a fresh one when None.
        policy: Modes and retention ages; `StorePolicy.load_default()` when None.
    """

    def __init__(self, root: Path, provenance: ProvStore | None = None, *, policy: StorePolicy | None = None) -> None:
        """Open a store at `root`; nothing is created until the first `put`."""
        self.root = Path(root)
        self.provenance = provenance if provenance is not None else ProvStore(run_id=f"store-{uuid.uuid4().hex}")
        self.policy = policy if policy is not None else StorePolicy.load_default()
        self._last_collect: float | None = None

    def _entry(self, key: LogicalKey) -> Path:
        d = key.digest()
        return self.root / STORE_DIR / d[:2] / d

    def _open_root(self, *, create: bool) -> StorageDir:
        return StorageDir.open_root(self.root, create=create, mode=shared_dir_mode(self.policy))

    def _quarantine_dir(self, root: StorageDir) -> StorageDir | None:
        """Return `root/quarantine` opened, made if missing; None (logged) if it cannot be."""
        try:
            return root.walk(QUARANTINE_DIR, create=True, mode=shared_dir_mode(self.policy), aside=lambda: None)
        except (OSError, WrongKindError) as err:
            logger.warning("cannot open the quarantine of store %s: %s", self.root, err)
            return None

    def _ensure(self, root: StorageDir, *parts: str) -> StorageDir:
        """Open `root/<parts...>`, making missing directories `shared_dir_mode` and quarantining wrong kinds.

        Raises:
            StoreUnavailable: a directory could not be created or opened.
            WrongKindError: an object of the wrong kind on the way cannot be moved aside.
        """
        try:
            return root.walk(
                *parts, create=True, mode=shared_dir_mode(self.policy), aside=lambda: self._quarantine_dir(root)
            )
        except OSError as err:
            raise storage_fault(err, root.path.joinpath(*parts)) from err

    def _parent(self, root: StorageDir, entry: Path) -> StorageDir | None:
        """Return the directory holding `entry` opened below `root`, None if it is missing.

        Raises:
            WrongKindError: a directory on the way is of the wrong kind.
            StoreUnavailable: it could not be opened.
        """
        try:
            return root.walk(*entry.parent.relative_to(self.root).parts)
        except FileNotFoundError:
            return None
        except OSError as err:
            raise storage_fault(err, entry.parent) from err

    def _quarantine(self, entry: Path, *, digest: str, fingerprint: str) -> None:
        """Move `entry` aside, re-verify the moved copy, and put it back if it verifies.

        The entry goes to `root/quarantine/<uuid>`. If it cannot be made writable for that move (it
        belongs to another user) or the quarantine cannot be created, it is renamed to `<uuid>`
        beside itself instead, which needs write permission only on its parent; that name is never
        a fingerprint, so `get` ignores it, and its owner must remove it. The re-verification uses
        the address the entry was read at (`digest`, `fingerprint`). A copy that verifies, or cannot
        be read, is renamed back without replacing anything: if a new entry took the address
        meanwhile the copy stays aside. Nothing is done if `entry` is already gone. A provenance
        activity records where it went, the outcome and the reason.

        Raises:
            StoreUnavailable: `entry` cannot be moved at all; the message names its owner and the remedy.
            WrongKindError: a directory above `entry` is of the wrong kind.
        """
        try:
            root = self._open_root(create=False)
        except FileNotFoundError:
            return
        except OSError as err:
            raise storage_fault(err, self.root) from err
        with root:
            parent = self._parent(root, entry)
            if parent is None:
                return
            with parent:
                self._quarantine_in(root, parent, entry.name, digest=digest, fingerprint=fingerprint)

    def _quarantine_in(self, root: StorageDir, parent: StorageDir, name: str, *, digest: str, fingerprint: str) -> None:
        entry = parent.path / name
        try:
            info = parent.lstat(name)
        except OSError as err:
            raise storage_fault(err, entry) from err
        if info is None:
            return
        is_dir = stat.S_ISDIR(info.st_mode)
        into = self._aside(root, parent, name, is_dir)
        moved_name = uuid.uuid4().hex
        target = into if into is not None else parent
        try:
            failure: OSError | None = None
            try:
                parent.rename(name, target, moved_name, grammar=ASIDE_NAME)
            except FileNotFoundError:
                return
            except OSError as err:
                failure = err
            if failure is not None:
                if is_dir and into is not None:
                    _seal(parent, name, committed_modes(self.policy)[1])
                raise storage_fault(failure, entry, hint=owner_remedy(entry)) from failure
            moved = target.path / moved_name
            try:
                target.stamp(moved_name)
            except OSError as err:
                logger.warning("cannot stamp quarantined entry %s: %s", moved, err)
            verdict, reason = self._verdict(moved, is_dir, digest=digest, fingerprint=fingerprint)
            outcome = "quarantined" if into is not None else "set aside in place"
            if verdict != "corrupt":
                outcome = self._restore(target, moved_name, parent, name)
            elif into is None:
                logger.warning("corrupt store entry set aside at %s; %s", moved, owner_remedy(moved))
        finally:
            if into is not None:
                into.close()
        logger.warning("store entry %s %s (%s): %s", entry, outcome, verdict, reason)
        self.provenance.activity(
            node="store.quarantine",
            step=None,
            parameters={
                "key": digest,
                "context": fingerprint,
                "moved_to": str(moved.relative_to(self.root)),
                "verdict": verdict,
                "outcome": outcome,
                "reason": reason,
            },
        )

    def _aside(self, root: StorageDir, parent: StorageDir, name: str, is_dir: bool) -> StorageDir | None:
        """Return the quarantine if `name` can be moved there, None to set it aside beside itself."""
        quarantine = self._quarantine_dir(root)
        if quarantine is None or not is_dir:
            return quarantine
        try:
            with parent.child(name) as entry:
                entry.set_mode(STAGING_DIR_MODE)
        except (OSError, WrongKindError) as err:
            logger.warning("cannot move %s into the quarantine, setting it aside in place: %s", parent.path / name, err)
            quarantine.close()
            return None
        return quarantine

    @staticmethod
    def _verdict(moved: Path, is_dir: bool, *, digest: str, fingerprint: str) -> tuple[str, str]:
        """Return `("corrupt" | "verified" | "unreadable", reason)` for a moved entry."""
        if not is_dir:
            return "corrupt", "not a directory"
        try:
            _read_result(moved, digest=digest, fingerprint=fingerprint)
        except CorruptEntryError as err:
            return "corrupt", str(err)
        except StoreUnavailable as err:
            return "unreadable", str(err)
        return "verified", "the moved copy verifies"

    def _restore(self, holder: StorageDir, moved_name: str, parent: StorageDir, name: str) -> str:
        """Rename `holder/moved_name` back onto `parent/name` unless something now occupies it; return the outcome."""
        failure: OSError | None = None
        try:
            holder.rename(moved_name, parent, name, grammar=FINGERPRINT_NAME)
        except OSError as err:
            failure = err
        if failure is not None:
            if parent.kind(name) is not None:
                return "superseded"
            raise storage_fault(failure, parent.path / name) from failure
        _seal(parent, name, committed_modes(self.policy)[1])
        return "restored"

    def _set_aside(self, entry: Path, *, digest: str, fingerprint: str) -> None:
        """Quarantine `entry`; if the store cannot be written, log it and leave the entry unserved."""
        try:
            self._quarantine(entry, digest=digest, fingerprint=fingerprint)
        except (StoreUnavailable, WrongKindError) as err:
            logger.warning("cannot quarantine store entry %s, skipping it: %s", entry, err)

    def _collect_due(self) -> None:
        """Run `collect` if this store has not collected within `policy.collect_interval_s` (monotonic)."""
        now = time.monotonic()
        if self._last_collect is None or now - self._last_collect >= self.policy.collect_interval_s:
            self._last_collect = now
            self.collect()

    def _reseal(self, parent: StorageDir, name: str) -> None:
        """Make a verified entry's directory read-only again if a writer died between its rename and seal."""
        try:
            info = parent.lstat(name)
        except OSError as err:
            logger.warning("cannot examine store entry %s: %s", parent.path / name, err)
            return
        if info is not None and info.st_mode & 0o222:
            _seal(parent, name, committed_modes(self.policy)[1])

    def collect(self, now: float | None = None) -> int:
        """Remove quarantined entries and staging directories older than the policy's ages; return how many.

        Objects set aside in place (`<uuid>` under the root, `store@4/`, a prefix or a key, beside what
        they displaced) count as quarantined. Ages are measured from each object's mtime to `now`, which
        defaults to the storage's own clock (`StorageDir.fs_now` on `root`). Only uuid-named children are
        removed, through `remove_owned`; no directory below `root` is reached through a symlink. A failure
        to read the clock, list or remove is logged and skips that item.
        """
        try:
            root = self._open_root(create=False)
        except OSError as err:
            if not isinstance(err, FileNotFoundError):
                logger.warning("cannot open store %s; not collecting: %s", self.root, err)
            return 0
        with root:
            if now is None:
                try:
                    now = root.fs_now()
                except (OSError, WrongKindError) as err:
                    logger.warning("cannot read the clock of store %s; not collecting: %s", self.root, err)
                    return 0
            aside = now - self.policy.quarantine_retention_s
            removed = root.remove_older(aside)
            for parts, cutoff in (((QUARANTINE_DIR,), aside), ((STAGING_DIR,), now - self.policy.staging_stale_s)):
                directory = _opened(root, *parts)
                if directory is not None:
                    with directory:
                        removed += directory.remove_older(cutoff)
            store = _opened(root, STORE_DIR)
            if store is None:
                return removed
            with store:
                removed += store.remove_older(aside)
                for prefix in _listed(store):
                    prefix_dir = _opened(store, prefix) if _PREFIX.fullmatch(prefix) else None
                    if prefix_dir is None:
                        continue
                    with prefix_dir:
                        removed += prefix_dir.remove_older(aside)
                        for digest in _listed(prefix_dir):
                            key_dir = _opened(prefix_dir, digest) if _SHA256.fullmatch(digest) else None
                            if key_dir is not None:
                                with key_dir:
                                    removed += key_dir.remove_older(aside)
            return removed

    def put(self, key: LogicalKey, outputs: dict[str, object], context: ExecutionContext) -> OutputIdentity:
        """Store `outputs` under `key` and `context`, and return the committed output identity.

        Outputs are stored as JCS canonicalises them: `1.0` is stored as `1`, strings are
        NFC-normalised and tuples become lists, and the output identity is taken over that form.
        If a valid entry for `key` and `context` is already committed, nothing is written and that
        entry's identity is returned, which may differ from the identity of `outputs`; every file of
        it, arrays included, is verified first. Anything else at the entry's path is quarantined and
        replaced, as is anything of the wrong kind at a directory of the store on the way to it.
        Either way one provenance activity is appended.

        Raises:
            TypeError: on an output that is neither a JSON value nor a numpy array.
            ValueError: on a container that contains itself, a reserved `$array` key, a value
                `jcs_dumps` refuses, or an array dtype `canonical_npy_bytes` refuses.
            StoreUnavailable: the store could not be read or written (`StorageFull` when out of space).
            WrongKindError: an object of the wrong kind in the store could not be moved aside.
            RuntimeError: if the target is still corrupt after `COMMIT_ATTEMPTS` commits.
        """
        payload = _serialise(outputs)
        d = key.digest()
        try:
            root = self._open_root(create=True)
        except OSError as err:
            raise storage_fault(err, self.root) from err
        with root:
            with self._ensure(root, STAGING_DIR) as staging_root:
                self._collect_due()
                with self._ensure(root, STORE_DIR, d[:2], d) as parent:
                    return self._put_in(key, context, parent, staging_root, payload)

    def _put_in(
        self,
        key: LogicalKey,
        context: ExecutionContext,
        parent: StorageDir,
        staging_root: StorageDir,
        payload: tuple[object, dict[str, bytes], str],
    ) -> OutputIdentity:
        d, f = key.digest(), context.fingerprint()
        entry = parent.path / f
        for _ in range(COMMIT_ATTEMPTS):
            try:
                existing = _committed_identity(entry, digest=d, fingerprint=f)
            except CorruptEntryError:
                self._quarantine(entry, digest=d, fingerprint=f)
                existing = None
            if existing is not None:
                self._reseal(parent, f)
                self._record_put(key, context, existing, outcome="existing")
                return existing
            if _commit(key, context, parent, f, staging_root, payload, self.policy):
                self._record_put(key, context, payload[2], outcome="committed")
                return payload[2]
        raise RuntimeError(f"{entry} was still corrupt after {COMMIT_ATTEMPTS} commits")

    def _record_put(self, key: LogicalKey, context: ExecutionContext, identity: str, *, outcome: str) -> None:
        digest = key.digest()
        prov = self.provenance
        activity = prov.activity(
            node=key.task,
            step=key.backend,
            parameters={
                "key": digest,
                "logical_key": jcs_loads(jcs_dumps(key.model_dump())),
                "context": {**context.model_dump(mode="json"), "fingerprint": context.fingerprint()},
                "outcome": outcome,
            },
        )
        for name, input_identity in sorted(key.inputs.items()):
            prov.used(
                activity,
                prov.entity(prov_type="artifact", extent=None, attributes={"input": name, "identity": input_identity}),
            )
        prov.was_generated_by(self._result_entity(identity, digest), activity)

    def _result_entity(self, identity: OutputIdentity, digest: str) -> str:
        """Record a result: `checksum_sha256` is the hex of its tagged `output@3` digest, not a file hash."""
        return self.provenance.entity(
            prov_type="artifact",
            extent=None,
            attributes={CHECKSUM_KEY: identity.removeprefix("output:"), "output_identity": identity, "key": digest},
        )

    def _record_reuse(
        self, key: LogicalKey, stored: ExecutionContext, current: ExecutionContext, hit: StoredResult
    ) -> None:
        digest = key.digest()
        activity = self.provenance.activity(
            node="store.reuse",
            step=key.task,
            parameters={
                "key": digest,
                "stored_context": stored.fingerprint(),
                "current_context": current.fingerprint(),
                "policy_used": hit.policy_used.value,
            },
        )
        self.provenance.used(activity, self._result_entity(hit.output_identity, digest))

    def get(self, key: LogicalKey, context: ExecutionContext, policy: ReusePolicy | str) -> StoredResult | None:
        """Return a committed result for `key` whose context `policy` admits for `context`, or None.

        Candidates are tried narrowest admitting policy first, the current fingerprint first among
        `exact`, then by fingerprint. A candidate that fails verification is quarantined (or, if the
        store cannot be written, skipped) and the next one tried. A hit admitted under a policy other
        than `exact` appends a reuse activity to provenance.

        Raises:
            ValueError: if `policy` is not a `ReusePolicy` value.
            StoreUnavailable: a candidate could not be read; nothing was quarantined.

        An object of the wrong kind at a directory of the store on the way to `key` is quarantined and the
        key reads as absent.
        """
        policy = ReusePolicy(policy)
        d = key.digest()
        try:
            root = self._open_root(create=False)
        except FileNotFoundError:
            return None
        except OSError as err:
            raise storage_fault(err, self.root) from err
        with root:
            try:
                base = root.walk(STORE_DIR, d[:2], d, aside=lambda: self._quarantine_dir(root))
            except FileNotFoundError:
                return None
            except WrongKindError as err:
                logger.warning("cannot set aside %s, so key %s... is not served: %s", err.path, d[:12], err)
                return None
            except OSError as err:
                raise storage_fault(err, self._entry(key)) from err
            with base:
                return self._get_in(key, context, policy, base)

    def _get_in(
        self, key: LogicalKey, context: ExecutionContext, policy: ReusePolicy, base_dir: StorageDir
    ) -> StoredResult | None:
        d, base = key.digest(), base_dir.path
        try:
            names = sorted(n for n in base_dir.listdir() if _is_fingerprint(n))
        except OSError as err:
            raise storage_fault(err, base) from err
        allowed = _POLICY_ORDER[: _POLICY_ORDER.index(policy) + 1]
        current = context.fingerprint()
        candidates: list[tuple[int, bool, str, ExecutionContext]] = []
        for name in names:
            try:
                stored = _read_context(base / name, name)
            except CorruptEntryError:
                self._set_aside(base / name, digest=d, fingerprint=name)
                continue
            rank = next(
                (i for i, p in enumerate(allowed) if acceptable(stored, context, p, backend=key.backend)),
                None,
            )
            if rank is not None:
                candidates.append((rank, name != current, name, stored))
        for rank, _, name, _stored in sorted(candidates, key=lambda c: c[:3]):
            try:
                identity, outputs, stored = _read_result(base / name, digest=d, fingerprint=name)
            except CorruptEntryError:
                self._set_aside(base / name, digest=d, fingerprint=name)
                continue
            self._reseal(base_dir, name)
            hit = StoredResult(outputs=outputs, output_identity=identity, context=stored, policy_used=allowed[rank])
            if hit.policy_used is not ReusePolicy.EXACT:
                self._record_reuse(key, stored, context, hit)
            return hit
        return None
