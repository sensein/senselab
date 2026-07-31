# `senselab.audio.workflows.speaker_profile`

Per-subject **speaker profiles** and profile-based **other-voice flagging** /
**target-speaker quality**.

The workflow has two stages:

1. **Build** (`build_speaker_profile`, a standalone stage): pool per-window speaker
   embeddings across all of a subject's files, cluster them, and persist the
   **dominant cluster's centroid** per embedding model as a contamination-tolerant
   profile artifact.
2. **Compare** (library calls): score audio against the profile to flag likely
   **other-voice** regions, name pooled voice groups, and derive a
   **target-speaker recording-quality** indicator.

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

**Compare** (`compare.py`):

```
recording → per-window embeddings + (optional) presence p_voice
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
  `contracts/speaker-profile.schema.md` in the design spec (PR #523). Human-inspectable: subject id, per-model
  centroids, calibration band, dominant/runner-up cluster stats, per-file usage
  records, confidence, params, provenance.
- **Per-window comparison results** (`ProfileComparisonResult`): calibrated
  `similarity` / `other_voice_uncertainty` and a `target` / `other_voice` /
  `unavailable` flag per window, plus two recording-level rollups
  (`summarize_other_voice`, `compute_target_quality`).
- **Voice-group assignment** (`score_voice_groups` → `VoiceGroupAssignment`): given a
  recording already grouped into distinct voices, which group is the enrolled
  subject, with a `relative` / `absolute` / `unavailable` basis and the margin to the
  runner-up. This is the preferred entry point for identity scoring — see **Scope**.

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
    score_voice_groups,               # name pooled voice groups (preferred)
    profile_from_related_audios,      # enroll from an Audio+ bundle's siblings
    check_grid_compatibility,         # guard: detection grid must match enrollment grid
    profile_votes_by_bucket,          # map per-window results onto a bucket grid
)
```

## Enrolling from an Audio+ bundle

`profile_from_related_audios` is the bridge from the metadata layer
(`senselab.audio.data_structures.audio_plus`) to enrollment: hand it the bundle derived
for the recording you are about to analyze and it enrolls from that speaker's *other*
recordings.

```python
ap = build_audio_plus(ref, audio_loader=load, metadata_provider=B2AIMetadataProvider(root))
profile = profile_from_related_audios(ap, audio_loader=load, cache_dir=cache)
```

This is **leave-one-out by construction** — a provider's `related_audio_refs` excludes the
queried recording, so the file being scored cannot contribute to the reference it is scored
against. Siblings load one at a time; a subject may have dozens. A sibling that fails to
load is skipped and reported via `load_failures`, not fatal.

## Window grid

A profile records the grid it was enrolled at (`params.profile_window_s` / `profile_hop_s`),
and `check_grid_compatibility` **warns** if detection windows were extracted at a different
*length* (`strict=True` raises instead). Measured on constructed intrusions, a mismatch costs
2–10 AUC points — degraded, not meaningless, so the default surfaces it and continues. Comparing across lengths adds a duration domain gap on top of any speaker
difference, and nothing downstream would catch it: measured, `calibration_band` does not
adapt to the grid — it came out as the fixed fallback values at both 2.0 s and 0.5 s.

The hop is deliberately not checked; `_window_step_seconds` derives coverage from the
results' own timestamps, so duration rollups are already hop-agnostic.

Measured on the synthetic fixtures, coarse windows separate speakers better
(cross-subject centroid similarity 0.27 at 2.0 s vs 0.41 at 0.5 s; zero false alarms vs
11%), while fine windows localize better (0.85 vs 0.50 detection inside a 3 s intrusion).
Enrollment wants the coarse grid; frame-level localization wants the fine one. Since
`DIFF_SPEAKER_FLOOR` is fixed at 0.70 regardless, choosing a grid also means choosing a
band.

CLI: `scripts/build_speaker_profile.py` (contract: `contracts/build-profile-cli.md` in the design spec (PR #523)).

```bash
build_speaker_profile --subject-id SUB --output SUB.profile.json FILE [FILE ...]
```

## Constants

Every threshold is a named, documented value in `constants.py` (origin tag
`[reuse]`/`[new]`; `[new]` values flagged for empirical validation). Key ones:
default embedding consensus (ECAPA + ResNet), window/hop sizes,
`MIN/TARGET_CONFIDENT_SPEECH_S`, `AMBIGUITY_SHARE_RATIO`,
`OTHER_VOICE_CALIBRATED_CUTOFF`, `MIN_P_VOICE_FOR_COMPARISON`,
`SESSION_PREFERENCE_WEIGHT`.

## Caching note

Per-window speaker embeddings are cached through the shared
`senselab.utils.tasks.cached_inference` store: pass `--cache-dir` to
`build_speaker_profile.py` and each model's window list is keyed on
`(audio signature, model, window/hop)`. The key carries no caller identity, so a
profile build and any other stage extracting per-window embeddings over the same
audio reuse each other's work instead of re-running the GPU models — which matters
most for leave-one-file-out, since that re-extracts the subject's sibling files.

Bump `_EMBEDDING_CACHE_CODE_VERSION` in `audio_analysis/embeddings.py` if a change
there would alter the vectors for an unchanged `(audio, model, grid)`.

## Scope

This module covers **enrollment and comparison only** — building a profile and
scoring audio against it. It deliberately does not wire itself into any uncertainty
axis: how a profile-derived signal should fold into speaker-identity scoring belongs
with the per-speaker uncertainty work, and `score_voice_groups` is the interface that
work consumes.

See the design spec (PR #523) for the full design.
