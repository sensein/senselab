"""Hatchling metadata hook: the distribution's dependencies plus one exact pin on a sibling."""

from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface
from packaging.version import Version


class PinHook(MetadataHookInterface):
    """Set ``dependencies`` to the hook's ``dependencies`` plus ``<pin>==<version>``."""

    PLUGIN_NAME = "custom"

    def update(self, metadata: dict[str, Any]) -> None:
        """Write the dependency list, pin included, into the project metadata.

        Raises:
            ValueError: If the version has a local segment, which ``==`` would also match.
        """
        version = str(metadata["version"])
        if Version(version).local is not None:
            raise ValueError(f"refusing to pin {self.config['pin']}=={version}: a local version segment")
        metadata["dependencies"] = [*self.config["dependencies"], f"{self.config['pin']}=={version}"]
