"""A backend whose outputs are hashes of their inputs, with a declared fault model.

Run as `python -m senselab_worker.launch senselab_worker.fake`. Faults come from `init["faults"]` and are replaced, per
request, by `params["faults"]`; both take the fields of `FakeFaults`.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy

from senselab_worker.canonical import canonical_npy_bytes, jcs_dumps
from senselab_worker.codec import MAX_LINE_BYTES
from senselab_worker.faults import ContentFault, ResourceFault
from senselab_worker.hashing import sha256_bytes

EXIT_CRASH = 70
_NONCE_BYTES = 16


@dataclass(frozen=True)
class FakeFaults:
    """Faults the fake injects; the default injects none.

    Attributes:
        fail_items: item ids that fail with a `ContentFault`.
        oom_items: item ids that fail with `OutOfMemoryError`, a `ResourceFault`.
        crash_after: the process exits abruptly before its next item once it has handled this many.
        batch_coupled: each value also depends on its batch-mates' inputs.
        nondeterministic: each value also depends on fresh random bytes.
        latency_s: seconds spent per item.
        oversize_items: item ids whose value is a string longer than the wire line budget.
        unprintable_items: item ids that fail with `UnprintableFault`, a `ContentFault` that cannot be printed.
    """

    fail_items: frozenset[str] = frozenset()
    oom_items: frozenset[str] = frozenset()
    crash_after: int | None = None
    batch_coupled: bool = False
    nondeterministic: bool = False
    latency_s: float = 0.0
    oversize_items: frozenset[str] = frozenset()
    unprintable_items: frozenset[str] = frozenset()

    @classmethod
    def from_json(cls, value: object) -> "FakeFaults":
        """Build from a JSON object with any of the fields; item id lists become frozensets.

        Raises:
            TypeError: if `value` is not an object or names an unknown field.
        """
        if not isinstance(value, dict):
            raise TypeError(f"faults must be an object, got {type(value).__name__}")
        fields: dict[str, Any] = {k: frozenset(v) if k in _ID_SETS else v for k, v in value.items()}
        return cls(**fields)


_ID_SETS = ("fail_items", "oom_items", "oversize_items", "unprintable_items")


class OutOfMemoryError(ResourceFault):
    """Out of memory, named as model frameworks name it."""


class UnprintableFault(ContentFault):
    """A content fault whose `__str__` and `__repr__` raise."""

    def __str__(self) -> str:
        """Raise: this fault cannot be printed."""
        raise RuntimeError("this fault cannot be printed")

    def __repr__(self) -> str:
        """Raise: this fault cannot be printed."""
        raise RuntimeError("this fault cannot be represented")


@dataclass
class _State:
    faults: FakeFaults
    handled: int = field(default=0)


def load(init: dict[str, Any]) -> object:
    """Return the fake's state; `init` may hold only `array_dir` and `faults`.

    Raises:
        ValueError: on any other key.
        TypeError: on malformed faults.
    """
    unknown = sorted(init.keys() - {"array_dir", "faults"})
    if unknown:
        raise ValueError(f"fake init takes only array_dir and faults, got {unknown}")
    return _State(FakeFaults.from_json(init.get("faults", {})))


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return `{"outputs": {"value": sha256(jcs(inputs and params))}}` per item, perturbed by faults.

    `params["faults"]` is excluded from the hash. Arrays enter it by the sha256 of their canonical
    `.npy` bytes.
    """
    if not isinstance(state, _State):
        raise TypeError("state must come from load")
    faults = FakeFaults.from_json(params["faults"]) if "faults" in params else state.faults
    rest = _identity_view({k: v for k, v in params.items() if k != "faults"})
    batch = [_identity_view(item["inputs"]) for item in items] if faults.batch_coupled else None
    results = []
    for item in items:
        if faults.crash_after is not None and state.handled >= faults.crash_after:
            print(f"fake crash after {state.handled} items", file=sys.stderr, flush=True)
            os._exit(EXIT_CRASH)
        state.handled += 1
        time.sleep(faults.latency_s)
        try:
            results.append(_one(item, rest, faults, batch))
        except Exception as exc:
            results.append({"error": exc})
    return results


def _one(item: dict[str, Any], params: object, faults: FakeFaults, batch: list[object] | None) -> dict[str, Any]:
    item_id = item["id"]
    if item_id in faults.fail_items:
        return {"error": ContentFault(f"fake failure for item {item_id}")}
    if item_id in faults.oom_items:
        return {"error": OutOfMemoryError(f"fake out of memory for item {item_id}")}
    if item_id in faults.unprintable_items:
        return {"error": UnprintableFault()}
    if item_id in faults.oversize_items:
        return {"outputs": {"value": "x" * (MAX_LINE_BYTES + 1)}}
    payload = {"inputs": _identity_view(item["inputs"]), "params": params}
    if batch is not None:
        payload["batch"] = batch
    data = jcs_dumps(payload)
    if faults.nondeterministic:
        data += os.urandom(_NONCE_BYTES)
    return {"outputs": {"value": sha256_bytes(data)}}


def _identity_view(value: object) -> object:
    if isinstance(value, numpy.ndarray):
        return sha256_bytes(canonical_npy_bytes(value))
    if isinstance(value, dict):
        return {k: _identity_view(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_identity_view(v) for v in value]
    return value
