"""A content-addressed result store keyed by logical key, with a recorded execution context (ADR-002, ADR-003).

Layout under `root`:

- `store@1/<d[:2]>/<d>/<f>/`, where `d` is `LogicalKey.digest()` and `f` is
  `ExecutionContext.fingerprint()`, holding:
  - `result.json`: JCS JSON with `format: "result@1.0"`, the key, its digest, the encoded outputs
    and their output identity;
  - `context.json`: JCS JSON with `format: "execution-context@1.0"` and the context's fields;
  - `<sha256>.npy`: each array output in canonical `.npy`, named by the sha256 of its bytes.
- `tmp/<uuid>/`: a writer's staging directory, committed by renaming it onto the entry.

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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy
from senselab_worker.canonical import canonical_npy_bytes, jcs_dumps
from senselab_worker.hashing import sha256_bytes

from senselab_core.kernel.identity import ExecutionContext, LogicalKey, ReusePolicy, acceptable
from senselab_core.runtime.provenance import CHECKSUM_KEY, ProvStore

STORE_DIR = "store@1"
STAGING_DIR = "tmp"
RESULT_FORMAT = "result@1.0"
CONTEXT_FORMAT = "execution-context@1.0"
ARRAY_TAG = "$array"
STAGING_STALE_SECONDS = 24 * 60 * 60
"""A staging directory whose mtime is older than this is a crashed writer's and is removed by `put`."""

_POLICY_ORDER = (ReusePolicy.EXACT, ReusePolicy.EQUIVALENT, ReusePolicy.TOLERANT)
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class StoredResult:
    """A result read from the store.

    Attributes:
        outputs: The decoded outputs; arrays come back as little-endian numpy arrays, tuples as lists.
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
        return value.item()
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
                raise ValueError(f"corrupt store entry {entry}: {sha}.npy does not hash to its name")
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
        os.fsync(handle.fileno())


def _read_json(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_bytes())
    if not isinstance(record, dict):
        raise ValueError(f"{path} is not a JSON object")
    return record


def _committed_identity(entry: Path) -> str | None:
    """Return the output identity of the committed entry at `entry`, or None if there is none."""
    try:
        record = _read_json(entry / "result.json")
    except FileNotFoundError:
        return None
    return str(record["output_identity"])


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
) -> tuple[str, bool]:
    """Stage and rename an entry; return the committed identity and whether this writer's rename won."""
    staging = staging_root / uuid.uuid4().hex
    staging.mkdir()
    try:
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
        entry.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(staging, entry)
        except OSError as err:
            if err.errno not in (errno.EEXIST, errno.ENOTEMPTY):
                raise
            winner = _committed_identity(entry)
            if winner is None:
                raise
            return winner, False
        _fsync_dir(entry.parent)
        return identity, True
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _remove_stale_staging(staging_root: Path) -> None:
    cutoff = time.time() - STAGING_STALE_SECONDS
    for child in staging_root.iterdir():
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
    `STAGING_STALE_SECONDS` (24 hours, by directory mtime).

    Args:
        root: The store root.
        provenance: Where `put` records its activity; the store owns a fresh one when None.
    """

    def __init__(self, root: Path, provenance: ProvStore | None = None) -> None:
        """Open a store at `root`; nothing is created until the first `put`."""
        self.root = Path(root)
        self.provenance = provenance if provenance is not None else ProvStore(run_id=f"store-{uuid.uuid4().hex}")

    def _entry(self, key: LogicalKey) -> Path:
        d = key.digest()
        return self.root / STORE_DIR / d[:2] / d

    def put(self, key: LogicalKey, outputs: dict[str, object], context: ExecutionContext) -> str:
        """Store `outputs` under `key` and `context`, and return the committed output identity.

        If an entry for `key` and `context` is already committed, nothing is written and that
        entry's identity is returned, which may differ from the identity of `outputs`. Either way
        one provenance activity is appended.

        Raises:
            TypeError: on an output that is neither a JSON value nor a numpy array.
            ValueError: on a reserved `$array` key, a value `jcs_dumps` refuses, or an array dtype
                `canonical_npy_bytes` refuses.
        """
        tree, arrays, identity = _serialise(outputs)
        staging_root = self.root / STAGING_DIR
        staging_root.mkdir(parents=True, exist_ok=True)
        _remove_stale_staging(staging_root)
        entry = self._entry(key) / context.fingerprint()
        existing = _committed_identity(entry)
        if existing is None:
            committed, won = _commit(key, context, entry, staging_root, tree, arrays, identity)
        else:
            committed, won = existing, False
        self._record(key, context, committed, outcome="committed" if won else "existing")
        return committed

    def _record(self, key: LogicalKey, context: ExecutionContext, identity: str, *, outcome: str) -> None:
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
        generated = prov.entity(prov_type="artifact", extent=None, attributes={CHECKSUM_KEY: identity, "key": digest})
        prov.was_generated_by(generated, activity)

    def get(self, key: LogicalKey, context: ExecutionContext, policy: ReusePolicy) -> StoredResult | None:
        """Return a committed result for `key` whose context `policy` admits for `context`, or None.

        An entry with the current fingerprint is preferred; otherwise the entry admitted under the
        narrowest policy, ties broken by fingerprint order.

        Raises:
            ValueError: if the chosen entry is corrupt (an array file or the output identity does
                not match its recorded hash, or a format tag is not one this store reads).
        """
        base = self._entry(key)
        try:
            names = sorted(n for n in os.listdir(base) if _is_fingerprint(n))
        except FileNotFoundError:
            return None
        allowed = _POLICY_ORDER[: _POLICY_ORDER.index(policy) + 1]
        current = context.fingerprint()
        best: tuple[int, str, ExecutionContext] | None = None
        for name in sorted(names, key=lambda n: n != current):
            stored = self._read_context(base / name)
            rank = next(
                (i for i, p in enumerate(allowed) if acceptable(stored, context, p, backend=key.backend)),
                None,
            )
            if rank is not None and (best is None or rank < best[0]):
                best = (rank, name, stored)
            if best is not None and best[0] == 0:
                break
        if best is None:
            return None
        rank, name, stored = best
        return self._read_result(base / name, stored, allowed[rank])

    @staticmethod
    def _read_context(entry: Path) -> ExecutionContext:
        record = _read_json(entry / "context.json")
        if record.pop("format", None) != CONTEXT_FORMAT:
            raise ValueError(f"corrupt store entry {entry}: context.json is not {CONTEXT_FORMAT}")
        return ExecutionContext.model_validate(record)

    @staticmethod
    def _read_result(entry: Path, context: ExecutionContext, policy_used: ReusePolicy) -> StoredResult:
        record = _read_json(entry / "result.json")
        if record.get("format") != RESULT_FORMAT:
            raise ValueError(f"corrupt store entry {entry}: result.json is not {RESULT_FORMAT}")
        tree = record["outputs"]
        identity = record["output_identity"]
        if sha256_bytes(jcs_dumps(tree)) != identity:
            raise ValueError(f"corrupt store entry {entry}: outputs do not hash to their output identity")
        outputs = _decode(tree, entry)
        if not isinstance(outputs, dict):
            raise ValueError(f"corrupt store entry {entry}: outputs are not an object")
        return StoredResult(outputs=outputs, output_identity=identity, context=context, policy_used=policy_used)
