"""A host backend whose transcript confidence is `step` times its `batch_size` backend param, the text fixed."""

from typing import Any


def load(init: dict[str, Any]) -> object:
    """Return `init` as the state."""
    return init


def handle(state: object, items: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:  # noqa: ARG001 - a worker handle
    """Return, per item, a one-segment transcript wire form whose confidence scales with the forward's batch size."""
    confidence = params["step"] * params["backend_params"]["batch_size"]
    segment = {"text": "a", "confidence": confidence, "span": None, "children": []}
    transcript = {"format": "transcript@2.0", "segments": [segment], "extent": None}
    return [{"outputs": {"transcript": transcript}} for _ in items]
