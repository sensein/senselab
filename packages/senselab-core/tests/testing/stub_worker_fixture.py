"""A worker that loads only the stub model `{"id": "stub"}`, as the fake stack's model."""

from typing import Any

from senselab_worker import fake

handle = fake.handle


def load(init: dict[str, Any]) -> object:
    """Return the fake model's state for a model whose id is `stub`; refuse any other model."""
    rest = {k: v for k, v in init.items() if k != "model"}
    model = init.get("model")
    if not isinstance(model, dict) or model.get("id") != "stub":
        raise ValueError(f"no stub model {model!r}: this worker has no weights")
    return fake.load(rest)
