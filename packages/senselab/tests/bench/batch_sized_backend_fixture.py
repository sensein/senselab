"""A host backend that claims batch invariance while each output depends on the size of the request it came in."""

import hashlib
from typing import Any


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return, per item, the sha256 of its input and the request's size."""
    return [
        {"outputs": {"value": hashlib.sha256(f"{item['inputs']}:{len(items)}".encode()).hexdigest()}} for item in items
    ]
