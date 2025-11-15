"""Bridge2AI voice assessment taxonomy tree implementation."""

from senselab.audio.tasks.quality_control.taxonomies.bioacoustic import (
    build_bioacoustic_activity_taxonomy,
)
from senselab.audio.tasks.quality_control.taxonomy import TaxonomyNode


def build_bridge2ai_voice_taxonomy() -> TaxonomyNode:
    """Build the Bridge2AI voice assessment taxonomy tree.

    This reuses the bioacoustic taxonomy as the base and renames the root node.

    Returns:
        The root node of the Bridge2AI voice taxonomy tree
    """
    # Reuse the bioacoustic taxonomy
    root = build_bioacoustic_activity_taxonomy()

    # Change the root name to bridge2ai_voice
    root.name = "bridge2ai_voice"

    return root


# Create the global Bridge2AI voice taxonomy tree instance
BRIDGE2AI_VOICE_TAXONOMY = build_bridge2ai_voice_taxonomy()
