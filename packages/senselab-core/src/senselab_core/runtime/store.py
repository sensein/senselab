"""A content-addressed result store keyed by logical key, with a recorded execution context (ADR-002, ADR-003).

Layout under `root`:

- `store@1/<d[:2]>/<d>/<f>/`, where `d` is `LogicalKey.digest()` and `f` is
  `ExecutionContext.fingerprint()`, holding:
  - `result.json`: JCS JSON with `format: "result@1.0"`, the key, its digest, the encoded outputs
    and their output identity;
  - `context.json`: JCS JSON with `format: "execution-context@1.0"` and the context's fields;
  - `<sha256>.npy`: each array output in canonical `.npy`, named by the sha256 of its bytes.
- `tmp/<uuid>/`: a writer's staging directory, committed by renaming it onto the entry.
- `quarantine/<uuid>/`: entries found corrupt, moved aside by rename.

Directories the store creates are mode `LOCK_DIR_MODE` (0o2775) and files `LOCK_FILE_MODE` (0o664),
whatever the umask; an existing `root` is left as it is.

Outputs are a mapping of JSON values (None, bool, int, float, str, lists, tuples, str-keyed dicts,
numpy scalars) and numpy arrays, nested to any depth. In the encoded form each array is replaced by
`{"$array": <sha256 of its canonical .npy>}`, so `$array` is a reserved key. The output identity is
the sha256 of the JCS of the encoded form.
"""

import errno
import io
import json
import os
import shutil
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import numpy
from pydantic import ValidationError
from senselab_worker.canonical import canonical_npy_bytes, jcs_dumps
from senselab_worker.hashing import sha256_bytes

from senselab_core.kernel.identity import ExecutionContext, LogicalKey, ReusePolicy, acceptable
from senselab_core.runtime.locks import LOCK_FILE_MODE, ensure_dirs
from senselab_core.runtime.provenance import CHECKSUM_KEY, ProvStore

STORE_DIR = "store@1"
STAGING_DIR = "tmp"
QUARANTINE_DIR = "quarantine"
RESULT_FORMAT = "result@1.0"
CONTEXT_FORMAT = "execution-context@1.0"
ARRAY_TAG = "$array"
STAGING_STALE_SECONDS = 24 * 60 * 60
"""A staging directory whose mtime is older than this is a crashed writer's and is removed by `put`."""
COMMIT_ATTEMPTS = 3
"""How many times `put` stages and renames before giving up on a target that keeps turning out corrupt."""

_POLICY_ORDER = (ReusePolicy.EXACT, ReusePolicy.EQUIVALENT, ReusePolicy.TOLERANT)
_HEX = frozenset("0123456789abcdef")
T = TypeVar("T")


class CorruptEntryError(ValueError):
    """A committed store entry that cannot be read back as written."""


@dataclass(frozen=True)
class StoredResult:
    """A result read from the store.

    Attributes:
        outputs: The decoded outputs, as JCS canonicalised them: arrays come back as little-endian
            numpy arrays, tuples as lists, integral floats as ints, strings NFC-normalised.
        output_identity: The sha256 of the JCS of the encoded outputs.
        context: The execution context the result was computed in.
        policy_used: The narrowest policy, no wider than the one requested, under which `context`
            may serve the current context: `exact` for an equal fingerprint, `equivalent` for a
            measured equivalence class, `tolerant` otherwise.
    """

    outputs: dict[str, object]
    output_identity: str
    context: ExecutionContext
    policy_used: ReusePolicy


def _encode(value: object, arrays: dict[str, bytes]) -> object:
    if isinstance(value, numpy.ndarray):
        data = canonical_npy_bytes(value)
        sha = sha256_bytes(data)
        arrays[sha] = data
        return {ARRAY_TAG: sha}
    if isinstance(value, numpy.generic):
        return _encode(value.item(), arrays)
    if isinstance(value, Mapping):
        if ARRAY_TAG in value:
            raise ValueError(f"{ARRAY_TAG!r} is a reserved key in stored outputs")
        return {k: _encode(v, arrays) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(v, arrays) for v in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"cannot store an output of type {type(value).__name__}")


def _serialise(outputs: Mapping[str, object]) -> tuple[object, dict[str, bytes], str]:
    arrays: dict[str, bytes] = {}
    tree = _encode(dict(outputs), arrays)
    return tree, arrays, sha256_bytes(jcs_dumps(tree))


def output_identity(outputs: Mapping[str, object]) -> str:
    """Return the output identity of `outputs`: the sha256 of the JCS of their encoded form.

    Raises:
        TypeError: on a value that is neither a JSON value nor a numpy array.
        ValueError: on a reserved `$array` key, a value `jcs_dumps` refuses, or an array dtype
            `canonical_npy_bytes` refuses.
    """
    return _serialise(outputs)[2]


def _decode(value: object, entry: Path) -> object:
    if isinstance(value, dict):
        if set(value) == {ARRAY_TAG}:
            sha = value[ARRAY_TAG]
            data = (entry / f"{sha}.npy").read_bytes()
            if sha256_bytes(data) != sha:
                raise CorruptEntryError(f"corrupt store entry {entry}: {sha}.npy does not hash to its name")
            return numpy.load(io.BytesIO(data), allow_pickle=False)
        return {k: _decode(v, entry) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode(v, entry) for v in value]
    return value


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_synced(path: Path, data: bytes) -> None:
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fchmod(handle.fileno(), LOCK_FILE_MODE)
        os.fsync(handle.fileno())


def _read_object(path: Path, fmt: str) -> dict[str, Any]:
    record = json.loads(path.read_bytes())
    if not isinstance(record, dict):
        raise CorruptEntryError(f"corrupt store entry: {path} is not a JSON object")
    if record.pop("format", None) != fmt:
        raise CorruptEntryError(f"corrupt store entry: {path} is not {fmt}")
    return record


def _guarded(entry: Path, read: Callable[[], T]) -> T:
    try:
        return read()
    except CorruptEntryError:
        raise
    except (OSError, ValueError, KeyError, TypeError, ValidationError) as err:
        raise CorruptEntryError(f"corrupt store entry {entry}: {type(err).__name__}: {err}") from err


def _read_context(entry: Path) -> ExecutionContext:
    """Return the context recorded in `entry`; raise `CorruptEntryError` if it cannot be read."""
    return _guarded(
        entry, lambda: ExecutionContext.model_validate(_read_object(entry / "context.json", CONTEXT_FORMAT))
    )


def _read_result(entry: Path) -> tuple[str, dict[str, object]]:
    """Return the output identity and decoded outputs of `entry`; raise `CorruptEntryError` if it is corrupt."""

    def read() -> tuple[str, dict[str, object]]:
        record = _read_object(entry / "result.json", RESULT_FORMAT)
        tree, identity = record["outputs"], record["output_identity"]
        if not isinstance(identity, str) or sha256_bytes(jcs_dumps(tree)) != identity:
            raise CorruptEntryError(f"corrupt store entry {entry}: outputs do not hash to their output identity")
        outputs = _decode(tree, entry)
        if not isinstance(outputs, dict):
            raise CorruptEntryError(f"corrupt store entry {entry}: outputs are not an object")
        return identity, outputs

    return _guarded(entry, read)


def _committed_identity(entry: Path) -> str | None:
    """Return the output identity of the entry at `entry`, None if there is none.

    Raises:
        CorruptEntryError: if an entry exists but cannot be read back whole.
    """
    if not entry.is_dir():
        return None
    _read_context(entry)
    return _read_result(entry)[0]


def _is_fingerprint(name: str) -> bool:
    return len(name) == 64 and set(name) <= _HEX


def _commit(
    key: LogicalKey,
    context: ExecutionContext,
    entry: Path,
    staging_root: Path,
    tree: object,
    arrays: dict[str, bytes],
    identity: str,
) -> bool:
    """Stage an entry and rename it onto `entry`; return False if another entry was there first."""
    staging = staging_root / uuid.uuid4().hex
    staging.mkdir()
    try:
        ensure_dirs(staging)
        for sha, data in arrays.items():
            _write_synced(staging / f"{sha}.npy", data)
        _write_synced(
            staging / "context.json",
            jcs_dumps({"format": CONTEXT_FORMAT, **context.model_dump(mode="json")}),
        )
        result = {
            "format": RESULT_FORMAT,
            "key": json.loads(jcs_dumps(key.model_dump())),
            "key_digest": key.digest(),
            "output_identity": identity,
            "outputs": tree,
        }
        _write_synced(staging / "result.json", jcs_dumps(result))
        _fsync_dir(staging)
        try:
            os.rename(staging, entry)
        except OSError as err:
            if err.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            return False
        _fsync_dir(entry.parent)
        return True
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _remove_stale_staging(staging_root: Path) -> None:
    cutoff = time.time() - STAGING_STALE_SECONDS
    try:
        children = list(staging_root.iterdir())
    except FileNotFoundError:
        return
    for child in children:
        try:
            stale = child.stat().st_mtime < cutoff
        except FileNotFoundError:
            continue
        if stale:
            shutil.rmtree(child, ignore_errors=True)


class ResultStore:
    """Results addressed by logical key, one entry per execution context, safe for concurrent writers.

    Each entry is staged in `root/tmp/<uuid>/` and committed by `os.rename`; the first writer to
    rename wins and later writers discard their staging and return the committed identity. A
    staging directory is never read by `get`; `put` removes those older than
    `STAGING_STALE_SECONDS` (24 hours, by directory mtime). An entry found corrupt by `get` or
    `put` is renamed into `root/quarantine/<uuid>/` and never served.

    Args:
        root: The store root.
        provenance: Where `put` and relaxed-policy `get` hits record activities; the store owns a
            fresh one when None.
    """

    def __init__(self, root: Path, provenance: ProvStore | None = None) -> None:
        """Open a store at `root`; nothing is created until the first `put`."""
        self.root = Path(root)
        self.provenance = provenance if provenance is not None else ProvStore(run_id=f"store-{uuid.uuid4().hex}")

    def _entry(self, key: LogicalKey) -> Path:
        d = key.digest()
        return self.root / STORE_DIR / d[:2] / d

    def _ensure(self, *parts: str) -> Path:
        if not self.root.exists():
            ensure_dirs(self.root)
        path = self.root
        for part in parts:
            path = path / part
            ensure_dirs(path)
        return path

    def _quarantine(self, entry: Path) -> None:
        """Rename `entry` into `root/quarantine/<uuid>`; a concurrent quarantine of it is not an error."""
        target = self._ensure(QUARANTINE_DIR) / uuid.uuid4().hex
        try:
            os.rename(entry, target)
        except FileNotFoundError:
            pass

    def _existing(self, entry: Path) -> str | None:
        try:
            return _committed_identity(entry)
        except CorruptEntryError:
            self._quarantine(entry)
            return None

    def put(self, key: LogicalKey, outputs: dict[str, object], context: ExecutionContext) -> str:
        """Store `outputs` under `key` and `context`, and return the committed output identity.

        Outputs are stored as JCS canonicalises them: `1.0` is stored as `1`, strings are
        NFC-normalised and tuples become lists, and the output identity is taken over that form.
        If a readable entry for `key` and `context` is already committed, nothing is written and
        that entry's identity is returned, which may differ from the identity of `outputs`; a
        corrupt one is quarantined and replaced. Either way one provenance activity is appended.

        Raises:
            TypeError: on an output that is neither a JSON value nor a numpy array.
            ValueError: on a reserved `$array` key, a value `jcs_dumps` refuses, or an array dtype
                `canonical_npy_bytes` refuses.
            RuntimeError: if the target is still corrupt after `COMMIT_ATTEMPTS` commits.
        """
        tree, arrays, identity = _serialise(outputs)
        d = key.digest()
        staging_root = self._ensure(STAGING_DIR)
        _remove_stale_staging(staging_root)
        entry = self._ensure(STORE_DIR, d[:2], d) / context.fingerprint()
        for _ in range(COMMIT_ATTEMPTS):
            existing = self._existing(entry)
            if existing is not None:
                self._record_put(key, context, existing, outcome="existing")
                return existing
            if _commit(key, context, entry, staging_root, tree, arrays, identity):
                self._record_put(key, context, identity, outcome="committed")
                return identity
        raise RuntimeError(f"{entry} was still corrupt after {COMMIT_ATTEMPTS} commits")

    def _record_put(self, key: LogicalKey, context: ExecutionContext, identity: str, *, outcome: str) -> None:
        digest = key.digest()
        prov = self.provenance
        activity = prov.activity(
            node=key.task,
            step=key.backend,
            parameters={
                "key": digest,
                "logical_key": json.loads(jcs_dumps(key.model_dump())),
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

    def _result_entity(self, identity: str, digest: str) -> str:
        return self.provenance.entity(
            prov_type="artifact", extent=None, attributes={CHECKSUM_KEY: identity, "key": digest}
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

    def get(self, key: LogicalKey, context: ExecutionContext, policy: ReusePolicy) -> StoredResult | None:
        """Return a committed result for `key` whose context `policy` admits for `context`, or None.

        Candidates are tried narrowest admitting policy first, the current fingerprint first among
        `exact`, then by fingerprint. A corrupt candidate is quarantined and the next one tried. A
        hit admitted under a policy other than `exact` appends a reuse activity to provenance.
        """
        base = self._entry(key)
        try:
            names = sorted(n for n in os.listdir(base) if _is_fingerprint(n))
        except FileNotFoundError:
            return None
        allowed = _POLICY_ORDER[: _POLICY_ORDER.index(policy) + 1]
        current = context.fingerprint()
        candidates: list[tuple[int, bool, str, ExecutionContext]] = []
        for name in names:
            try:
                stored = _read_context(base / name)
            except CorruptEntryError:
                self._quarantine(base / name)
                continue
            rank = next(
                (i for i, p in enumerate(allowed) if acceptable(stored, context, p, backend=key.backend)),
                None,
            )
            if rank is not None:
                candidates.append((rank, name != current, name, stored))
        for rank, _, name, stored in sorted(candidates, key=lambda c: c[:3]):
            try:
                identity, outputs = _read_result(base / name)
            except CorruptEntryError:
                self._quarantine(base / name)
                continue
            hit = StoredResult(outputs=outputs, output_identity=identity, context=stored, policy_used=allowed[rank])
            if hit.policy_used is not ReusePolicy.EXACT:
                self._record_reuse(key, stored, context, hit)
            return hit
        return None
