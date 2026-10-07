""".. include:: ./doc.md"""  # noqa: D415

from .api import (  # noqa: F401
    HEAR_EVENT_LABELS,
    HearEmbeddings,
    centred_cosine_similarity,
    detect_health_acoustic_events,
    extract_hear_embeddings_at_times,
    extract_hear_embeddings_from_audios,
)

__all__ = [
    "HEAR_EVENT_LABELS",
    "HearEmbeddings",
    "centred_cosine_similarity",
    "detect_health_acoustic_events",
    "extract_hear_embeddings_at_times",
    "extract_hear_embeddings_from_audios",
]
