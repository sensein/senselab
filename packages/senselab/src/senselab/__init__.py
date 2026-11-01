"""senselab 2.x: the public package over ``senselab_core`` and ``senselab_worker``."""

from senselab._api import generated, packaged  # noqa: F401 - importing packaged hands run_call the catalogue
from senselab._api.describe import describe, models, tasks
from senselab._api.generated import *  # noqa: F403 - the generated task classes and functions

__all__ = ["describe", "models", "tasks", *generated.__all__]  # noqa: PLE0604 - the generated module's own names
