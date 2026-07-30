"""Speaker profile embedding workflow.

Builds per-subject speaker profiles (dominant-cluster centroids over pooled
per-window embeddings) and scores audio against them to flag other-voice regions,
name pooled voice groups, and estimate target-speaker recording quality.

Enrollment and comparison only: this package deliberately does not wire itself into
any uncertainty axis. :func:`score_voice_groups` is the interface an identity-scoring
step consumes. Profile-building runs as a standalone stage and shares the
content-addressable embedding cache, so per-window embeddings are computed once
across stages and re-runs.

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
    compute_target_quality,
    leave_one_file_out_profile,
    profile_votes_by_bucket,
    score_voice_groups,
    score_window,
    summarize_other_voice,
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
    VoiceGroupAssignment,
    VoiceGroupBasis,
    VoiceGroupMatch,
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
    "VoiceGroupAssignment",
    "VoiceGroupBasis",
    "VoiceGroupMatch",
    "aggregate_dominant_cluster",
    "build_source_records",
    "build_speaker_profile",
    "compare_recording_to_profile",
    "compute_target_quality",
    "decide_confidence",
    "extract_speech_windows_for_file",
    "leave_one_file_out_profile",
    "load_profile",
    "profile_votes_by_bucket",
    "save_profile",
    "score_voice_groups",
    "score_window",
    "summarize_other_voice",
    "within_file_holdout_profile",
]
