"""senselab 2.x: the public package over ``senselab_core`` and ``senselab_worker``."""

from senselab._api import generated
from senselab._api.describe import describe, tasks
from senselab._api.generated import *  # noqa: F403 - the generated task classes and functions

__all__ = ["describe", "tasks", *generated.__all__]  # noqa: PLE0604 - the generated module's own names
