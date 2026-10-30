"""Hatchling metadata hook: pin one sibling distribution to this distribution's version."""

from typing import Any

from hatchling.metadata.plugin.interface import MetadataHookInterface


class PinHook(MetadataHookInterface):
    """Append ``<pin>==<version>`` to ``dependencies``; ``pin`` is read from the hook's config."""

    PLUGIN_NAME = "custom"

    def update(self, metadata: dict[str, Any]) -> None:
        """Add the exact pin to the project metadata."""
        pin = self.config["pin"]
        metadata["dependencies"] = [*metadata.get("dependencies", []), f"{pin}=={metadata['version']}"]
