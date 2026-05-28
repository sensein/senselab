"""Speaker profile embedding workflow.

Builds per-subject speaker profiles (dominant-cluster centroids over pooled
per-window embeddings) and feeds them into :mod:`analyze_audio` to flag
other-voice regions and estimate target-speaker recording quality.

Profile-building runs as a standalone stage *before* :mod:`analyze_audio`,
sharing the content-addressable cache so expensive tasks (diarization,
speaker embeddings, scene classification) are computed once.

See ``specs/20260527-151905-speaker-profile-embedding/`` for the full design.
"""

from __future__ import annotations

from senselab.audio.workflows.speaker_profile.types import (
    ClusterStats,
    ComparisonFlag,
    ProfileComparisonResult,
    ProfileConfidence,
    ProfileParams,
    ProfileSourceFile,
    RecordingOtherVoiceSummary,
    RecordingQualityIndicator,
    SpeakerProfile,
)

__all__ = [
    "ClusterStats",
    "ComparisonFlag",
    "ProfileComparisonResult",
    "ProfileConfidence",
    "ProfileParams",
    "ProfileSourceFile",
    "RecordingOtherVoiceSummary",
    "RecordingQualityIndicator",
    "SpeakerProfile",
]
