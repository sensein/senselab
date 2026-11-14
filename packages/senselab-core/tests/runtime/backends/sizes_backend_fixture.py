"""A host backend that records the items of each request and fails the items its input asks it to."""

from pathlib import Path
from typing import Any

from senselab_worker.faults import ContentFault

SIZES: list[int] = []
"""The number of items of each request, in order."""


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Record the request's size; an input `fail` is a content fault, `gap` an untyped one, else its identity."""
    SIZES.append(len(items))
    entries: list[dict[str, Any]] = []
    for item in items:
        value = next(iter(item["inputs"].values()))
        if value == "fail":
            entries.append({"error": ContentFault("asked to fail")})
        elif value == "leak":
            entries.append({"error": ContentFault(f"cannot read {Path.home() / 'data' / 'a.wav'} from {Path.cwd()}")})
        elif value == "gap":
            entries.append({"error": RuntimeError("an untyped failure")})
        else:
            entries.append({"outputs": {"value": str(getattr(value, "identity", value))}})
    return entries
