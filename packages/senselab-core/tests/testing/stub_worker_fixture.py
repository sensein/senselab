"""A worker that loads only the stub `{"model": "stub"}`, as the fake stack's model."""

from typing import Any

from senselab_worker import fake

handle = fake.handle


def load(init: dict[str, Any]) -> object:
    """Return the fake model's state for `init["model"] == "stub"`; refuse any other model."""
    rest = {k: v for k, v in init.items() if k != "model"}
    if init.get("model") != "stub":
        raise ValueError(f"no stub model {init.get('model')!r}: this worker has no weights")
    return fake.load(rest)
