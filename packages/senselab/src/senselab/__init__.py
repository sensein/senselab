"""senselab 2.x: the public package over ``senselab_core`` and ``senselab_worker``.

The task classes and functions are generated from the declarations into ``senselab._api.generated`` and
load on first use, so importing this package does not import the declared tasks.
"""

import importlib
from typing import TYPE_CHECKING, Any

from senselab._api.describe import describe, tasks

if TYPE_CHECKING:
    from senselab._api.generated import *  # noqa: F403

__all__ = ["describe", "tasks"]


def __getattr__(name: str) -> Any:  # noqa: ANN401 - a generated task class or function
    """Return the generated task class or function `name`."""
    generated = importlib.import_module("senselab._api.generated")
    if name in generated.__all__:
        return getattr(generated, name)
    raise AttributeError(f"module 'senselab' has no attribute {name!r}")


def __dir__() -> list[str]:
    """List the facade and the generated names."""
    generated = importlib.import_module("senselab._api.generated")
    return sorted({*__all__, *generated.__all__})
