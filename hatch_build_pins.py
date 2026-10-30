"""Hatchling metadata hook: the distribution's dependencies plus one exact pin on a sibling."""

from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface


class PinHook(MetadataHookInterface):
    """Set ``dependencies`` to the hook's ``dependencies`` plus ``<pin>==<version>``."""

    PLUGIN_NAME = "custom"

    def update(self, metadata: dict[str, Any]) -> None:
        """Write the dependency list, pin included, into the project metadata."""
        metadata["dependencies"] = [*self.config["dependencies"], f"{self.config['pin']}=={metadata['version']}"]
