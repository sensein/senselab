"""A host backend that fails each item whose payload says `fail` and echoes the rest."""

from typing import Any


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ARG001 - a worker handle
    """Return, per item, its payload as the value, or a `ValueError` when the payload says `fail`."""
    return [
        {"error": ValueError("asked to fail")}
        if item["inputs"]["payload"].get("fail")
        else {"outputs": {"value": str(item["inputs"]["payload"])}}
        for item in items
    ]
