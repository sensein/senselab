""".. include:: ./doc.md"""  # noqa: D415

from senselab.audio.tasks.turn_taking_projection.api import (
    project_voice_activity,
    separate_dyad,
    summarize_projection,
)
from senselab.audio.tasks.turn_taking_projection.result import VAPResult

__all__ = ["VAPResult", "project_voice_activity", "separate_dyad", "summarize_projection"]
