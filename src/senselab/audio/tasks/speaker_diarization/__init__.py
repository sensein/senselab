""".. include:: ./doc.md"""  # noqa: D415

from .api import capabilities_for, diarize_audios  # noqa: F401
from .capabilities import DiarizationCapabilities  # noqa: F401

__all__ = ["diarize_audios", "capabilities_for", "DiarizationCapabilities"]
