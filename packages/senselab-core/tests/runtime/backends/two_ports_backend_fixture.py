"""A host backend with two output ports, `x` and `y`, each naming its port and the item's input."""

from typing import Any


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    """Return `x:<input>` on `x` and `y:<input>` on `y` for each item."""
    entries = []
    for item in items:
        value = next(iter(item["inputs"].values()))
        entries.append({"outputs": {"x": f"x:{value}", "y": f"y:{value}"}})
    return entries
