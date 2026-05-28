# Contract: Speaker Profile Artifact (JSON)

The persisted profile produced by `build_speaker_profile` and consumed by `analyze_audio --speaker-profile`. One file per subject. Human-inspectable; vectors are small.

## Top-level object

```jsonc
{
  "schema_version": 1,                      // bumped on breaking changes
  "subject_id": "sub-00123",                // required, non-empty
  "confidence": "ok",                       // "ok" | "low" | "ambiguous" | "insufficient"
  "aggregate_speech_seconds": 41.5,
  "centroids": {                            // {embedding_model_id -> L2-normalized vector}
    "speechbrain/spkrec-ecapa-voxceleb": [/* floats, 192-D */],
    "speechbrain/spkrec-resnet-voxceleb": [/* floats */],
    "microsoft/wavlm-base-plus-sv": [/* floats, 512-D */]
  },
  "calibration_band": {                     // {model_id -> [same_speaker_floor, diff_speaker_floor]}
    "speechbrain/spkrec-ecapa-voxceleb": [0.31, 0.68]
  },
  "dominant_cluster": { "n_windows": 78, "speech_seconds": 41.5, "silhouette": 0.27, "share": 0.86 },
  "runner_up_cluster": null,                // ClusterStats when confidence == "ambiguous"
  "sources": [
    {
      "file_id": "sub-00123/ses-1/free-speech.wav",
      "audio_signature": "<sha256>",
      "session_id": "ses-1",
      "speech_seconds_used": 22.0,
      "windows_used": 41,
      "kept": true,
      "drop_reason": null
    },
    {
      "file_id": "sub-00123/ses-1/cough.wav",
      "audio_signature": "<sha256>",
      "session_id": "ses-1",
      "speech_seconds_used": 0.0,
      "windows_used": 0,
      "kept": false,
      "drop_reason": "non_speech_task"
    }
  ],
  "params": {
    "embedding_models": ["speechbrain/spkrec-ecapa-voxceleb", "speechbrain/spkrec-resnet-voxceleb", "microsoft/wavlm-base-plus-sv"],
    "profile_window_s": 2.0, "profile_hop_s": 1.0,
    "detect_window_s": 1.0, "detect_hop_s": 0.5,
    "min_confident_speech_s": 20.0, "target_confident_speech_s": 30.0,
    "ambiguity_share_ratio": 0.80,
    "prefer_session": null
  },
  "provenance": {
    "senselab_version": "x.y.z",
    "cache_key_basis": "module:senselab.audio.workflows.speaker_profile.build@<hash>",
    "built_at": "2026-05-27T15:30:00Z"
  }
}
```

## Invariants

- `centroids` has ≥1 entry unless `confidence == "insufficient"` (then it MAY be `{}`).
- Each `centroids[model]` vector is L2-normalized (‖v‖ ≈ 1) and length-consistent across profiles for the same model.
- `aggregate_speech_seconds == dominant_cluster.speech_seconds`.
- `runner_up_cluster` is non-null **iff** `confidence == "ambiguous"`.
- At least one `sources[*].kept == true` unless `confidence == "insufficient"`.
- `confidence == "low"` ⟺ `0 < aggregate_speech_seconds < min_confident_speech_s` and a coherent cluster exists.
- Consumers MUST honor `confidence`: treat `insufficient` as "no profile", and surface `low`/`ambiguous` in downstream output.

## Compatibility

- Unknown extra keys MUST be ignored by readers (forward-compatible).
- A reader encountering a higher `schema_version` than it supports MUST refuse rather than misinterpret.
