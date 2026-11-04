"""senselab 2.x: the public package over ``senselab_core`` and ``senselab_worker``."""

from senselab_core.runtime.call import packaged

from senselab._api import generated
from senselab._api.describe import describe, models, tasks, validate
from senselab._api.generated import *  # noqa: F403 - the generated task classes and functions

__all__ = ["describe", "models", "tasks", "validate", *generated.__all__]  # noqa: PLE0604 - the generated module's own names

packaged()
