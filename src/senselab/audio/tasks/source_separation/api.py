"""Public API for the senselab source separation task.

Currently exposes only class-space resolution for unasdiff's sound prior
(:func:`resolve_source_classes`). The separation entry points themselves land in
a later task in this plan, once the subprocess worker exists.
"""

from __future__ import annotations

from senselab.audio.tasks.source_separation.unasdiff import load_fsd_class_map_document


def resolve_source_classes(names: list[str]) -> list[int]:
    """Resolve FSD sound-prior class names to their conditioning indices.

    Args:
        names: Class names as they appear in the FSD class map (e.g.
            ``["Applause", "Cello"]``).

    Returns:
        The conditioning index for each name, in the same order.

    Raises:
        ValueError: If any name is not in the class map. The sound prior's
            embedding has 50 slots but only 41 were trained (see
            ``data/fsd41_classes.json``'s derivation); silently mapping an
            unknown name to a fallback index would condition the prior on
            whatever class that fallback happens to name, while the caller's
            requested label -- and the output -- would go on reporting
            something else. The message enumerates the valid names so the
            caller can fix the typo without a second round trip.
    """
    classes = load_fsd_class_map_document()["classes"]
    unknown = [name for name in names if name not in classes]
    if unknown:
        valid = ", ".join(sorted(classes))
        raise ValueError(f"Unknown source class name(s): {', '.join(unknown)!r}. Valid classes are: {valid}")
    return [classes[name] for name in names]
