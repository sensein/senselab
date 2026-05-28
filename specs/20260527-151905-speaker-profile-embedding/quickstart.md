# Quickstart: Speaker Profile Embedding

Build a per-subject speaker profile, then use it in `analyze_audio` to flag other-voice regions and estimate target-speaker recording quality.

> Prereqs: senselab dev environment installed (`poetry install` / project env), `analyze_audio` already runnable.

## 1. Build a profile for a subject

Point it at the subject's files. The few solid free-speech / reading files will dominate; short clips, cough, and breathing files drop out automatically (no task labels needed).

```bash
python scripts/build_speaker_profile.py \
  --subject-id sub-00123 \
  --output profiles/sub-00123.json \
  --cache-dir artifacts/analyze_audio_cache \
  data/sub-00123/ses-1/free-speech.wav \
  data/sub-00123/ses-1/reading.wav \
  data/sub-00123/ses-1/cough.wav \
  data/sub-00123/ses-1/breathing.wav
```

Inspect the result — `confidence`, how many seconds were used, and which files were kept:

```bash
jq '{confidence, aggregate_speech_seconds,
     kept: [.sources[] | select(.kept) | .file_id],
     dropped: [.sources[] | select(.kept|not) | {file_id, drop_reason}]}' \
  profiles/sub-00123.json
```

Expected: `confidence: "ok"` when ≥~30s of speech was found; the two non-speech files appear under `dropped` with `drop_reason: "non_speech_task"`.

## 2. Analyze a recording with the profile

```bash
python scripts/analyze_audio.py \
  data/sub-00123/ses-1/reading.wav \
  --speaker-profile profiles/sub-00123.json \
  --cache-dir artifacts/analyze_audio_cache \
  --output-dir out/sub-00123-reading
```

Because the profile and `analyze_audio` share the cache, the diarization / embedding tasks should report `cache: "hit"` (no recomputation). Since `reading.wav` helped build the profile, **leave-one-file-out** is applied automatically.

## 3. Review profile-based signals

```bash
# Other-voice flags + target-quality summary
jq '{confidence: .profile_confidence, loo: .leave_one_file_out_applied,
     quality: .quality,
     other_voice: [.windows[] | select(.flag=="other_voice") | {start, end, other_voice_uncertainty}]}' \
  out/sub-00123-reading/speaker_profile.json
```

## Acceptance smoke checks (map to spec Success Criteria)

- **SC-001 / FR-001**: step 1 yields exactly one profile + a usage record.
- **FR-016**: cough/breathing files appear as `dropped`.
- **SC-006 / FR-011**: running step 2 **without** `--speaker-profile` produces the same non-profile outputs as before (diff against a baseline run).
- **FR-008**: non-speech buckets are `flag: "unavailable"`, never `other_voice`.
- **FR-012**: `leave_one_file_out_applied: true` for a file that contributed to the profile.

## Validating contamination tolerance (SC-002) — optional

Mix a known second speaker into ~20% of a subject's enrollment material, rebuild, and confirm the profile centroid is closer (lower calibrated uncertainty) to held-out clean target audio than to the intruder:

```bash
python scripts/build_speaker_profile.py --subject-id sub-test --output profiles/sub-test.json <files...>
# then score held-out clean target vs intruder clips against profiles/sub-test.json (see compare_test.py)
```
