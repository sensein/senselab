# `senselab.audio.workflows.speaker_profile`

Per-subject **speaker profiles** and profile-based **other-voice flagging** /
**target-speaker quality** for `analyze_audio`.

The workflow has two stages:

1. **Build** (`build_speaker_profile`, run as a standalone stage *before*
   `analyze_audio`): pool per-window speaker embeddings across all of a
   subject's files, cluster them, and persist the **dominant cluster's
   centroid** per embedding model as a contamination-tolerant profile artifact.
2. **Compare** (inside `analyze_audio --speaker-profile`): score each analyzed
   window against the profile to flag likely **other-voice** regions and derive
   a **target-speaker recording-quality** indicator.

The profile is the L2-normalized centroid of the dominant (largest,
silhouette-coherent) cluster of pooled per-window embeddings — minority clusters
(other voices, noise) are discarded, which is what makes a profile tolerant to a
moderate fraction of contaminating audio. Diarization / presence are used only
to *locate* speech, never to assign identity.

## Pipeline

**Build** (`build.py`):

```
files → per-file speech-window extraction (presence-gated, ≥~1s windows)
      → pool windows (tagged by file_id)
      → cluster on a reference model (reuses cluster_pass_speakers)
      → dominant cluster → per-model L2-normalized centroid + empirical
        calibration band (reuses _empirical_calibration_band)
      → confidence policy (ok / low / ambiguous / insufficient)
      → per-file keep/drop usage records
      → SpeakerProfile artifact (JSON)
```

**Compare** (`compare.py`, wired into `analyze_audio`):

```
analyzed recording → reuse the run's per-window embeddings + presence p_voice
   → (if the file contributed to the profile) leave-one-file-out: re-extract the
      other source files and recompute the centroid; single-file → within-file
      holdout
   → per-window consensus score: per-model calibrated cosine-uncertainty vs the
      centroid (each model uses its own calibration band), fused to a consensus
   → speech-presence gate (low p_voice → "unavailable")
   → flag: target / other_voice / unavailable
   → recording rollups (other-voice summary, target-quality indicator)
```

## Outputs

- **Profile artifact** (one JSON per subject) — see
  `contracts/speaker-profile.schema.md`. Human-inspectable: subject id, per-model
  centroids, calibration band, dominant/runner-up cluster stats, per-file usage
  records, confidence, params, provenance.
- **`analyze_audio` additions** (only with `--speaker-profile`; otherwise output
  is byte-identical — SC-006, regression-tested in `regression_test.py`):
  - identity-axis `model_votes` gain `speaker_profile/<model>` +
    `speaker_profile/consensus` entries (additive — the identity aggregator
    ignores them, so existing uncertainties are unchanged);
  - a per-pass `speaker_profile.json` sidecar (per-window flags + the two
    rollups);
  - the existing per-pass `single_speaker` and `quality` claims gain
    `profile_*` sub-signals, and each folds a profile-derived uncertainty into
    its headline (the quality fold only when `confidence == "ok"`).

  See `contracts/analyze-audio-profile.md`.

## Confidence semantics

`SpeakerProfile.confidence` is one of:

- `ok` — coherent dominant cluster with aggregate speech ≥ the floor.
- `low` — coherent dominant cluster but `0 < aggregate < min_confident_speech_s`.
- `ambiguous` — a runner-up cluster rivals the dominant one
  (`runner_up / dominant ≥ AMBIGUITY_SHARE_RATIO`); the target voice's identity
  is in doubt. The runner-up stats are recorded.
- `insufficient` — no usable profile (terminal; centroids may be empty).
  Consumers treat this as "no profile".

Consumers must honor `confidence`: target-quality is discounted on
`low`/`ambiguous` and ignored on `insufficient`.

## Public API

```python
from senselab.audio.workflows.speaker_profile import (
    build_speaker_profile,            # build entrypoint
    SpeakerProfile, save_profile, load_profile,
    compare_recording_to_profile,     # per-window scoring
    summarize_other_voice,            # recording other-voice rollup
    compute_target_quality,           # recording target-quality rollup
    leave_one_file_out_profile, within_file_holdout_profile,
    profile_votes_by_bucket,
)
```

CLI: `scripts/build_speaker_profile.py` (see `contracts/build-profile-cli.md`).

```bash
build_speaker_profile --subject-id SUB --output SUB.profile.json FILE [FILE ...]
analyze_audio RECORDING.wav --speaker-profile SUB.profile.json
```

## Constants

Every threshold is a named, documented value in `constants.py` (origin tag
`[reuse]`/`[new]`; `[new]` values flagged for empirical validation). Key ones:
default embedding consensus (ECAPA + ResNet + WavLM), window/hop sizes,
`MIN/TARGET_CONFIDENT_SPEECH_S`, `AMBIGUITY_SHARE_RATIO`,
`OTHER_VOICE_CALIBRATED_CUTOFF`, `MIN_P_VOICE_FOR_COMPARISON`,
`SESSION_PREFERENCE_WEIGHT`.

## Caching note

The build stage and `analyze_audio` are designed to share the
content-addressable task cache (diarization, speaker embeddings, scene
classification) so the expensive per-file tasks are computed once across both
stages. The caller-agnostic key helper lives in `cache.py`
(`task_wrapper_hash`, keyed on each task's implementing library modules).
**Status**: `build_speaker_profile` uses it, but `analyze_audio` still keys its
task cache on its own script-source hash, so the two stages do not yet share
entries — finishing that swap (and the end-to-end cache-hit test) is the
remaining FR-015 work. See `research.md` R1.

See `specs/20260527-151905-speaker-profile-embedding/` for the full design.
