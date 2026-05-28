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

from senselab.audio.workflows.speaker_profile.build import (
    AggregationResult,
    ProfileInput,
    TaggedWindowEmbedding,
    aggregate_dominant_cluster,
    build_source_records,
    build_speaker_profile,
    decide_confidence,
    extract_speech_windows_for_file,
)
from senselab.audio.workflows.speaker_profile.compare import (
    compare_recording_to_profile,
    leave_one_file_out_profile,
    score_window,
    within_file_holdout_profile,
)
from senselab.audio.workflows.speaker_profile.io import (
    ProfileSchemaError,
    load_profile,
    save_profile,
)
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
    "AggregationResult",
    "ClusterStats",
    "ComparisonFlag",
    "ProfileComparisonResult",
    "ProfileConfidence",
    "ProfileInput",
    "ProfileParams",
    "ProfileSchemaError",
    "ProfileSourceFile",
    "RecordingOtherVoiceSummary",
    "RecordingQualityIndicator",
    "SpeakerProfile",
    "TaggedWindowEmbedding",
    "aggregate_dominant_cluster",
    "build_source_records",
    "build_speaker_profile",
    "compare_recording_to_profile",
    "decide_confidence",
    "extract_speech_windows_for_file",
    "leave_one_file_out_profile",
    "load_profile",
    "save_profile",
    "score_window",
    "within_file_holdout_profile",
]
