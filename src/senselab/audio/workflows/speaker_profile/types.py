"""Typed dataclasses for the speaker_profile workflow.

These match the data model in
``specs/20260527-151905-speaker-profile-embedding/data-model.md``.

Following the audio_analysis convention, these are plain ``@dataclass(slots=True)``
classes (not Pydantic) because they are workflow-internal — the artifact JSON
writer in ``io.py`` serializes them explicitly.

The two recording-level rollups (``RecordingOtherVoiceSummary``,
``RecordingQualityIndicator``) are *internal compute holders* whose fields are meant
to feed a consumer's own recording-level claims (FR-020 / FR-010). They are not
serialized as standalone top-level objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ProfileConfidence = Literal["ok", "low", "ambiguous", "insufficient"]
"""Confidence label on a built profile (FR-005, FR-014).

- ``ok``: aggregate speech ≥ ``TARGET_CONFIDENT_SPEECH_S`` and a coherent dominant cluster.
- ``low``: aggregate speech below floor (``MIN_CONFIDENT_SPEECH_S``) but a coherent cluster exists.
- ``ambiguous``: a runner-up cluster rivals the dominant one (see ``AMBIGUITY_SHARE_RATIO``).
- ``insufficient``: no usable profile (centroids may be empty); terminal.
"""

ComparisonFlag = Literal["target", "other_voice", "unavailable"]
"""Per-window flag (FR-008). ``unavailable`` when the speech-presence gate fails."""


@dataclass(slots=True)
class ClusterStats:
    """Stats describing one embedding cluster from the profile build."""

    n_windows: int
    speech_seconds: float
    silhouette: float | None
    share: float  # fraction of all clustered windows in this cluster


@dataclass(slots=True)
class ProfileSourceFile:
    """Per-file record within an enrollment set (FR-004, FR-016).

    Used for the per-profile usage record AND for leave-one-file-out scoring
    (FR-012) — ``file_id`` is the stable identifier matched at compare time.
    """

    file_id: str
    audio_signature: str  # sha256 of post-resample PCM; matches the shared cache key
    session_id: str | None
    speech_seconds_used: float
    windows_used: int
    kept: bool
    drop_reason: str | None
    # drop_reason values include "insufficient_speech", "outside_dominant_cluster",
    # "non_speech_task" (e.g., cough / breathing) — speech-presence gated (FR-008).


@dataclass(slots=True)
class ProfileParams:
    """Configuration used to build a profile (stamped into the artifact).

    See ``constants.py`` for the defaults and their provenance.
    """

    embedding_models: list[str]
    profile_window_s: float
    profile_hop_s: float
    detect_window_s: float
    detect_hop_s: float
    min_confident_speech_s: float
    target_confident_speech_s: float
    ambiguity_share_ratio: float
    prefer_session: str | None = None


@dataclass(slots=True)
class SpeakerProfile:
    """Persisted, reusable per-subject profile.

    See ``contracts/speaker-profile.schema.md`` for the JSON serialization.
    The profile is the L2-normalized centroid of the dominant cluster of
    per-window embeddings pooled across the subject's files, per embedding
    model (FR-001 / FR-003).
    """

    subject_id: str
    centroids: dict[str, list[float]]  # {embedding_model_id -> L2-normalized vector}
    confidence: ProfileConfidence
    aggregate_speech_seconds: float
    dominant_cluster: ClusterStats
    runner_up_cluster: ClusterStats | None  # populated iff confidence == "ambiguous"
    calibration_band: dict[str, tuple[float, float]]  # {model_id -> (same_floor, diff_floor)}
    sources: list[ProfileSourceFile] = field(default_factory=list)
    params: ProfileParams | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ProfileComparisonResult:
    """One window's comparison against a profile (FR-007 / FR-008).

    Emitted into the identity-axis ``model_votes`` for the bucket and into the
    per-pass ``<pass>/speaker_profile.json`` sidecar.
    """

    start: float
    end: float
    similarity: float | None
    other_voice_uncertainty: float | None
    flag: ComparisonFlag
    p_voice: float | None
    per_model: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class RecordingOtherVoiceSummary:
    """Internal compute holder for the per-recording other-voice rollup (FR-020).

    Its fields populate the existing per-pass ``single_speaker`` claim
    (``global_uncertainty.by_pass[<pass>]``) in ``summary.json``. Not serialized
    as a standalone top-level object.
    """

    profile_other_voice_fraction: float
    profile_other_voice_seconds: float
    profile_peak_other_voice_uncertainty: float
    profile_p95_other_voice_uncertainty: float
    profile_speech_present_seconds: float
    profile_confidence: ProfileConfidence
    # Continuous "is the dominant voice the enrolled subject" measure: the
    # voiced-time-weighted mean per-window subject similarity over scored
    # windows ([0, 1]; higher = more the subject). Its complement
    # (1 - dominance) is the "recording may not be the subject" / wrong-subject
    # uncertainty — the profile's unique reference-based signal that the
    # reference-free identity axis cannot produce. ``None`` when no voiced window
    # was scorable (no signal, distinct from "confidently wrong subject").
    # Pair with ``profile_confidence`` downstream (a weak profile down-weights it).
    profile_subject_dominance: float | None


@dataclass(slots=True)
class RecordingQualityIndicator:
    """Internal compute holder for the target-speaker quality rollup (FR-010).

    Its fields populate the existing per-pass ``quality`` claim
    (``global_uncertainty.by_pass[<pass>]``) in ``summary.json``. Not serialized
    as a standalone top-level object.
    """

    profile_target_quality: float | None  # normalized [0, 1]; higher = cleaner. None when nothing was scorable.
    profile_target_match_fraction: float  # 1 - other-voice rate over speech-present duration
    profile_mean_target_consistency: float  # mean within-profile cosine consistency on matched windows
    profile_squim: dict[str, float] | None  # STOI/PESQ/SI-SDR on target-matched windows
    profile_confidence: ProfileConfidence


VoiceGroupBasis = Literal["relative", "absolute", "unavailable"]
"""How a voice-group assignment was decided.

- ``relative``: 2+ scorable groups, so the subject is the *closest* one and the
  decision needed no absolute threshold. Both groups come from the same recording,
  so channel / SNR offsets largely cancel — the robust case.
- ``absolute``: exactly one scorable group. There is nothing to compare against, so
  the call rests on the calibrated similarity alone and inherits that threshold's
  sensitivity to recording conditions.
- ``unavailable``: no group shares an embedding model with the profile, so no call
  was made. Distinct from "the subject is absent".
"""


@dataclass(slots=True)
class VoiceGroupMatch:
    """How well one pooled voice group matches a profile.

    Attributes:
        group_id: The group's id, as assigned by whatever produced the grouping.
        similarity: Calibrated ``1 - other_voice_uncertainty``; higher = more like
            the enrolled subject. ``None`` when no model overlapped the profile.
        other_voice_uncertainty: Consensus calibrated uncertainty that this group is
            somebody other than the enrolled subject. ``None`` when unscorable.
        per_model: Each model's pre-fusion uncertainty, for auditing which backend
            drove the consensus.
    """

    group_id: str
    similarity: float | None
    other_voice_uncertainty: float | None
    per_model: dict[str, float]


@dataclass(slots=True)
class VoiceGroupAssignment:
    """Which pooled voice group is the enrolled subject, and how firmly.

    Attributes:
        matches: One :class:`VoiceGroupMatch` per input group, ordered most- to
            least-like the subject (unscorable groups last).
        subject_group_id: The best-matching group, or ``None`` when nothing scored.
        margin: ``best.similarity - runner_up.similarity``. ``None`` unless the basis
            is ``relative``. A small margin means the groups look alike to the
            profile — the caller should treat the assignment as weak rather than
            trusting the winner.
        basis: See :data:`VoiceGroupBasis`.
    """

    matches: list[VoiceGroupMatch]
    subject_group_id: str | None
    margin: float | None
    basis: VoiceGroupBasis
