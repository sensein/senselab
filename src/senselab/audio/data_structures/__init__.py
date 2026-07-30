"""This module contains data structures for audio processing."""

from .audio import Audio, batch_audios, unbatch_audios  # noqa: F401
from .audio_classification_result import AudioClassificationResult  # noqa: F401
from .audio_plus import (  # noqa: F401
    AudioPlus,
    AudioPlusMetadata,
    MetadataProvider,
    NullMetadataProvider,
    SpeakerInfo,
    TaskInfo,
    build_audio_plus,
)
