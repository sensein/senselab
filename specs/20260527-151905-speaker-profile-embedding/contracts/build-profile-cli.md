# Contract: `build_speaker_profile` CLI

Thin wrapper (`scripts/build_speaker_profile.py`) over `senselab.audio.workflows.speaker_profile.build`. Mirrors `analyze_audio.py` conventions (same `--cache-dir`, `--device`, model flags, JSON output). Produces one [speaker-profile artifact](./speaker-profile.schema.md).

## Synopsis

```
build_speaker_profile --subject-id SUB --output PROFILE.json [options] FILE [FILE ...]
build_speaker_profile --subject-id SUB --output PROFILE.json --files-from manifest.txt [options]
```

## Inputs

| Arg | Type | Default | Meaning |
|-----|------|---------|---------|
| `FILE...` (positional) | paths | — | The subject's audio files. ≥1 required (or `--files-from`). |
| `--files-from` | path | — | Newline-delimited file list (optionally `path\tsession_id`). Alternative to positionals. |
| `--subject-id` | str | — | **Required.** Stamped into the artifact. |
| `--output` | path | — | **Required.** Where the profile JSON is written. |
| `--embedding-models` | str... | ECAPA, ResNet, WavLM (`microsoft/wavlm-base-plus-sv`) | Consensus models (R3). One model → single-model profile. WavLM uses the transformers backend (FR-019); degrades if unavailable. |
| `--profile-window-s` / `--profile-hop-s` | float | 2.0 / 1.0 | Long windows for the centroid. |
| `--min-confident-speech-s` | float | 20.0 | Below → `confidence="low"`. |
| `--target-confident-speech-s` | float | 30.0 | Target for `confidence="ok"`. |
| `--prefer-session` | str | none | Up-weight same-session windows (FR-013). |
| `--cache-dir` | path | `artifacts/analyze_audio_cache` | **Shared** with `analyze_audio` (R1). |
| `--no-cache` | flag | off | Disable cache lookup/store. |
| `--device` | {cpu,cuda,mps,auto} | auto | Compute device. |

## Behavior

1. Resample/downmix each file to 16 kHz mono (same `prepare_audio` as `analyze_audio`).
2. For each file: locate speech via diarization + presence `p_voice` (cache-shared), extract ≥~1s window embeddings per model, drop sub-1s fragments and non-speech windows.
3. Pool windows across files (tagged by `file_id`), cluster (`cluster_pass_speakers`), select dominant cluster → per-model centroids + calibration band.
4. Decide `confidence` from aggregate speech seconds and cluster coherence/ambiguity.
5. Write the artifact (atomic). Print a one-line summary to stdout.

## Exit codes / failure semantics

| Code | Condition |
|------|-----------|
| 0 | Profile written (any `confidence`, including `insufficient`). |
| 2 | Usage error (no files, missing `--subject-id`/`--output`). |
| 1 | Unrecoverable error (all files unreadable). |

- Per-file/per-model failures are **non-fatal**: recorded in `sources[].drop_reason` / provenance, not aborts (matches existing `failures` pattern).
- `insufficient` is a **success exit (0)** with an artifact whose `centroids` may be `{}` — it is a valid "declined" result, not an error (FR-005).

## Cache contract (R1)

Running this command then `analyze_audio` on the same file with identical task params MUST yield `cache: "hit"` for the shared tasks (diarization, speaker embeddings, scene classification) in the `analyze_audio` run.
