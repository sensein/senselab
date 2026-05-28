"""Documented thresholds and defaults for the speaker_profile workflow.

Every threshold is a named module-level constant so it is easy to find, tune,
and audit. Each constant's comment states its value, **origin tag**, and
**validation status**:

- ``[reuse]`` — existing senselab default carried over from the cited source.
  Keep aligned with the source unless an empirical sweep says otherwise.
- ``[new]`` — introduced by this feature. Provisional values are flagged with
  ``VALIDATE (T028)`` and tuned against the synthetic fixtures (T010a/b).

References:
- research.md "Constants & Thresholds"
- ``senselab/audio/workflows/audio_analysis/{embeddings,clustering}.py``
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Embedding model defaults — three-model consensus (FR-018, R3).
#
# ECAPA + ResNet are SpeechBrain (existing in senselab and run by analyze_audio
# by default). WavLM is new (FR-019, transformers backend) and adds genuine
# error decorrelation via SSL on a large noise/overlap-aware corpus.
# ---------------------------------------------------------------------------

ECAPA_MODEL_ID: str = "speechbrain/spkrec-ecapa-voxceleb"
"""[reuse] SpeechBrain ECAPA-TDNN — analyze_audio default embedding model."""

RESNET_MODEL_ID: str = "speechbrain/spkrec-resnet-voxceleb"
"""[reuse] SpeechBrain ResNet-TDNN — second analyze_audio default."""

WAVLM_DEFAULT_CHECKPOINT: str = "microsoft/wavlm-base-plus-sv"
"""[new] WavLM base-plus + SV head — the only official WavLM checkpoint with
an X-vector head (``microsoft/wavlm-large`` is a headless backbone). Configurable
so a WavLM-Large SV checkpoint can be substituted if one becomes available
(FR-019)."""

DEFAULT_EMBEDDING_MODELS: tuple[str, ...] = (
    ECAPA_MODEL_ID,
    RESNET_MODEL_ID,
    WAVLM_DEFAULT_CHECKPOINT,
)
"""[new] Default three-model consensus set (FR-018). Configurable; a
single-model fallback (e.g., just ECAPA) MUST remain viable."""


# ---------------------------------------------------------------------------
# Windowing — long windows for profile centroid, short windows for detection.
# Source: ``audio_analysis/embeddings.py`` (per-window embedding extraction;
# trade-off documented at the top of that module).
# ---------------------------------------------------------------------------

PROFILE_WINDOW_S: float = 2.0
"""[reuse] Long window for clean centroid embeddings (R3/R4)."""

PROFILE_HOP_S: float = 1.0
"""[reuse] Hop for the profile-build window grid."""

DETECT_WINDOW_S: float = 1.0
"""[reuse] Short window for detection-pass temporal resolution (R4);
matches ``embeddings.extract_per_window_embeddings`` default."""

DETECT_HOP_S: float = 0.5
"""[reuse] Detection-pass hop; one embedding per 0.5 s bucket."""

MIN_CONTIGUOUS_SPEECH_S: float = 1.0
"""[new] VALIDATE (T028). Minimum contiguous speech per contributing window;
sub-1s fragments are dropped/merged. ECAPA stat-pooling is noisier below 1 s
(see embeddings.py)."""

SUBSECOND_INTRUSION_BOUNDARY_S: float = 1.0
"""[new] VALIDATE (T028). Below this duration, other-voice localization is
reported as lower-confidence (FR-017 — coarser sub-1s resolution acknowledged)."""


# ---------------------------------------------------------------------------
# Clustering — reused defaults from ``audio_analysis/clustering.py``
# (``cluster_pass_speakers``). Keep these aligned with that module's behavior;
# they are part of the contamination-tolerant dominant-cluster selection (R2).
# ---------------------------------------------------------------------------

N_CLUSTERS_MAX: int = 6
"""[reuse] ``cluster_pass_speakers`` — max speaker clusters considered."""

MIN_WINDOWS_FOR_CLUSTERING: int = 4
"""[reuse] ``cluster_pass_speakers`` — below this, fall back to single-cluster regime."""

COHERENT_SILHOUETTE_THRESHOLD: float = 0.10
"""[reuse] ``cluster_pass_speakers`` — multi- vs single-cluster gate."""

MIN_CLUSTER_FRACTION: float = 0.10
"""[reuse] ``cluster_pass_speakers`` — clusters smaller than this fraction
(floor 2 windows) of clustered windows are treated as outliers, not speakers."""

MERGE_THRESHOLD: float = 0.55
"""[reuse] ``_merge_close_clusters`` — collapse prosodic sub-clusters of the
same speaker when centroid cos_sim ≥ this. Below this, clusters are taken to
be genuinely different people."""

CLUSTERING_ALGORITHM: str = "spectral"
"""[reuse] ``cluster_pass_speakers`` — spectral on cosine-affinity matrix is
the standard for speaker diarization; k-means is the documented fallback."""


# ---------------------------------------------------------------------------
# Calibration band — ``calibrate_cosine_uncertainty`` /
# ``_empirical_calibration_band`` in ``audio_analysis/clustering.py``.
# These are the *fallback* values when too few cluster pairs exist; the
# empirical band is computed per profile from within-/between-cluster cos_dist
# percentiles.
# ---------------------------------------------------------------------------

SAME_SPEAKER_FLOOR_FALLBACK: float = 0.30
"""[reuse] Literature fallback — typical ECAPA same-speaker noise level."""

DIFF_SPEAKER_FLOOR_FALLBACK: float = 0.70
"""[reuse] Literature fallback — well above the VoxCeleb EER region."""


# ---------------------------------------------------------------------------
# Profile confidence policy (FR-005). VALIDATE these against real per-subject
# durations in T028; the synthetic fixtures (T010a) include thin / insufficient
# subjects specifically to exercise the boundaries.
# ---------------------------------------------------------------------------

MIN_CONFIDENT_SPEECH_S: float = 20.0
"""[new] VALIDATE (T028). Aggregate speech-present seconds (post-gating) below
which the profile is marked ``confidence="low"``. The ~20 s floor reflects
ECAPA enrollment stability; the synthetic 'thin subject' fixture sits here."""

TARGET_CONFIDENT_SPEECH_S: float = 30.0
"""[new] VALIDATE (T028). Target aggregate speech for ``confidence="ok"``."""

AMBIGUITY_SHARE_RATIO: float = 0.80
"""[new] VALIDATE (T028). Flag ``confidence="ambiguous"`` when
``runner_up_speech_s / dominant_speech_s >= AMBIGUITY_SHARE_RATIO`` (evaluated
only when ≥2 speech clusters exist; centroid separation is already guaranteed
by ``MERGE_THRESHOLD``). Acceptance target: balanced 50/50 fixture → ambiguous;
dominant ~85/15 fixture → confident (FR-014, T011 / T028)."""


# ---------------------------------------------------------------------------
# Session preference. Optional same-session up-weighting applied to
# dominant-cluster *selection* and *centroid direction* only — never to the
# reported ``speech_seconds`` (the artifact invariant
# ``aggregate_speech_seconds == dominant_cluster.speech_seconds`` reports real
# seconds). Default is unweighted: when no ``prefer_session`` is given, every
# window weighs 1.0 and the profile is identical with or without session
# metadata.
# ---------------------------------------------------------------------------

SESSION_PREFERENCE_WEIGHT: float = 2.0
"""[new] Validate empirically. Multiplier applied to windows whose ``session_id``
matches ``--prefer-session`` when selecting / centering the dominant cluster.
``1.0`` would disable the preference; ``2.0`` lets the preferred session lead a
near-tie without erasing the contribution of other sessions."""


# ---------------------------------------------------------------------------
# Comparison-time policy (FR-008 / R6). The other-voice threshold is adaptive
# per profile (derived from its empirical calibration band); a fixed override
# is available via ``--profile-other-voice-threshold``. The fusion weights are
# a placeholder pending T028's sweep.
# ---------------------------------------------------------------------------

OTHER_VOICE_THRESHOLD_DEFAULT: float | None = None
"""[new] ``None`` → adaptive (derived from the profile's calibration band).
A fixed value (CLI override) overrides per-subject calibration."""

CONSENSUS_FUSION_WEIGHTS_DEFAULT: dict[str, float] | None = None
"""[new] VALIDATE (T028). ``None`` → unweighted mean of per-model calibrated
uncertainties (R3). Revisit weighting if one model dominates errors in the
empirical sweep."""
