# Contract: `analyze_audio` profile inputs & outputs

Additive extension to `scripts/analyze_audio.py` and the identity axis (R10). **When no profile is supplied, every existing input and output is unchanged** (FR-011, SC-006).

## New input

| Arg | Type | Default | Meaning |
|-----|------|---------|---------|
| `--speaker-profile` | path | none | A [speaker-profile artifact](./speaker-profile.schema.md). When given, enables profile-based other-voice flagging and target-speaker quality. |
| `--profile-other-voice-threshold` | float | adaptive | Override the adaptive per-subject threshold (R6) with a fixed calibrated-uncertainty cutoff. |

- The analyzed file's `file_id` is matched against the profile's `sources[]` to apply **leave-one-file-out** (R5): if the file contributed to the profile, its windows are excluded from the centroid used to score it. Single-file subjects fall back to within-file holdout.
- If `confidence == "insufficient"`, the profile is treated as absent (a warning is printed); all other outputs proceed unchanged.

## New outputs (only when `--speaker-profile` is supplied)

1. **Per-bucket identity-axis votes**: each `identity.parquet` row gains `model_votes["speaker_profile/<model>"]` and `model_votes["speaker_profile/consensus"]` entries carrying `{similarity, other_voice_uncertainty, flag}` (see `ProfileComparisonResult`). `unavailable` flags appear where the speech-presence gate fails. Existing voters/columns are untouched.

2. **`speaker_profile.json`** (new file next to other per-run outputs):

```jsonc
{
  "subject_id": "sub-00123",
  "profile_confidence": "ok",
  "leave_one_file_out_applied": true,
  "windows": [
    { "start": 0.0, "end": 1.0, "similarity": 0.71, "other_voice_uncertainty": 0.12,
      "flag": "target", "p_voice": 0.95, "per_model": { "...ecapa...": 0.10, "...resnet...": 0.14 } }
    // ...
  ],
  "quality": {
    "quality": 0.82,
    "target_match_fraction": 0.93,
    "mean_target_consistency": 0.74,
    "mean_squim": { "stoi": 0.91, "pesq": 2.8, "si_sdr": 14.2 },
    "profile_confidence": "ok"
  }
}
```

(Written **per pass** — `raw_16k/speaker_profile.json`, `enhanced_16k/speaker_profile.json` — matching the per-pass `embeddings/*.json` sidecar convention.)

3. **Recording-level rollups — extend existing claims** in `summary.json` → `global_uncertainty.by_pass[<pass>]` (FR-020/FR-010). No new top-level objects; profile sub-signals are added to the existing `single_speaker` and `quality` claims:

```jsonc
"single_speaker": {
  "uncertainty": 0.07,                 // existing headline; now also folds in profile uncertainty via max()
  "n_speakers": 1,                     // existing
  "identity_axis_mean": 0.05,          // existing
  "expects_speech": true,              // existing
  "profile_other_voice_fraction": 0.04,        // added (FR-020)
  "profile_other_voice_seconds": 1.8,          // added
  "profile_peak_other_voice_uncertainty": 0.81,// added
  "profile_p95_other_voice_uncertainty": 0.22, // added
  "profile_speech_present_seconds": 45.0,      // added
  "profile_confidence": "ok"                   // added (downstream fail-safe)
},
"quality": {
  "uncertainty": 0.18,                 // existing headline; now also folds in target-quality via existing aggregation
  "pesq_mean": 2.8, "stoi_mean": 0.91, "sisdr_mean": 14.2,   // existing (all-window)
  "profile_target_quality": 0.82,              // added (FR-010)
  "profile_target_match_fraction": 0.93,       // added
  "profile_mean_target_consistency": 0.74,     // added
  "profile_squim": { "stoi": 0.93, "pesq": 2.9, "si_sdr": 15.0 }, // added (target-matched windows)
  "profile_confidence": "ok"                   // added
}
```

4. **`disagreements.json`**: profile-flagged `other_voice` buckets are eligible for the existing top-N ranking under the `identity` axis (no schema change — they ride the existing row format).

> **Scope note**: these are decision-ready signals only. The PASS-vs-manual-review decision (recall-biased operating point, fail-safe on low-confidence, optional agentic rationale over *all* `analyze_audio` measures) is a **separate, out-of-scope feature** that consumes them.

## Invariants

- With `--speaker-profile` absent: output set and bytes are identical to the pre-feature version (regression-tested per SC-006).
- `flag == "unavailable"` ⟺ speech-presence gate not met for that bucket (cough/breathing/non-speech) — never a false `other_voice` (FR-008).
- `quality` is reported with `profile_confidence`; consumers must discount quality when confidence is `low`/`ambiguous` and ignore it when `insufficient`.
- All profile-derived buckets are time-aligned to the existing bucket grid so they overlay diarization/ASR/identity rows (FR-009).
