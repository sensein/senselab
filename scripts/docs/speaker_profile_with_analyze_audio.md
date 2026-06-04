# Combining a speaker profile with `analyze_audio`

This guide shows how to run a **full subject's audio files** through `analyze_audio`
*and* build a reusable **speaker profile** for that subject, so the per-subject
voice embeddings are ready for downstream work. It assumes the per-subject batch
workflow in `scripts/submit_subject_audio.sh` (one Slurm array task per file).

For the library internals see
[`src/senselab/audio/workflows/speaker_profile/doc.md`](../../src/senselab/audio/workflows/speaker_profile/doc.md);
for a single-file walkthrough see the feature
[`quickstart.md`](../../specs/20260527-151905-speaker-profile-embedding/quickstart.md).

## The mental model: two stages, in order

The profile is a **separate stage that runs first**, across *all* of a subject's
files, and produces **one artifact per subject**. `analyze_audio` then optionally
consumes that artifact per file.

```
                       build_speaker_profile.py          analyze_audio.py
   subject's files ─────────────►  profile.json  ─────────────►  per-file outputs
   (all of them, one job)        (1 per subject)   (--speaker-profile, one job/file)
```

- **`build_speaker_profile.py`** pools per-window speaker embeddings across the
  subject's files, clusters them, and stores the **dominant cluster's centroid
  per model** (ECAPA + ResNet + WavLM by default) plus a calibration band and a
  record of which files contributed. That centroid set *is* the subject's
  embedding representation — the thing future specs consume.
- **`analyze_audio.py --speaker-profile profile.json`** scores each analyzed
  window against the profile, flags likely other-voice regions, and extends the
  `single_speaker` / `quality` summary claims. With no `--speaker-profile`, every
  other output is byte-identical to a normal run.

You can do **just the build** now (embeddings ready, no analysis) and add
`--speaker-profile` to your `analyze_audio` runs whenever you're ready.

## Stage 1 — build the profile (embeddings ready)

`build_speaker_profile.py` takes the subject's files (positional, or
`--files-from` a newline-delimited list — the **same list format**
`submit_subject_audio.sh` already writes to
`logs/analyze_audio/<sub>/filelist.txt`):

```bash
uv run python scripts/build_speaker_profile.py \
    --files-from   logs/analyze_audio/sub-001/filelist.txt \
    --subject-id   sub-001 \
    --output       <dataset_root>/derivatives/speaker_profile/sub-001/profile.json \
    --cache-dir    <dataset_root>/derivatives/analyze_audio_cache \
    --device       cuda
```

Defaults build the full **ECAPA + ResNet + WavLM** consensus. Override with
`--embedding-models …` (a single model → single-model profile). Window/threshold
flags (`--profile-window-s`, `--min-confident-speech-s`, `--ambiguity-share-ratio`,
…) mirror the documented constants; the defaults are reasonable starting points.

The one-line summary reports confidence and how many files contributed:

```
profile sub-001: confidence=ok models=3 speech=42.3s kept=4/6 files → .../profile.json
```

`confidence` is one of `ok` / `low` / `insufficient`. An `insufficient` profile is
still written (exit 0) but `analyze_audio` treats it as *absent* (warns and scores
nothing) — so it's safe to build for every subject and let confidence gate use.

> **This is heavy model inference — run it on a GPU compute node (`sbatch`), never
> the login node.** A ready-made launcher is described under
> [One-command launcher](#one-command-launcher) below.

## Stage 2 — analyze with the profile (when you're ready)

Because `submit_subject_audio.sh` forwards any extra flags straight to
`analyze_audio.py`, wiring the profile into your existing per-subject run is just
one added flag — **once the profile exists**:

```bash
bash scripts/submit_subject_audio.sh <dataset_root> sub-001 \
    --speaker-profile <dataset_root>/derivatives/speaker_profile/sub-001/profile.json
```

That adds, per file: per-window other-voice flags on the identity axis, a
`<pass>/speaker_profile.json` sidecar, and profile sub-signals folded into the
`single_speaker` and `quality` summary claims. No PASS/REVIEW verdict is emitted —
these are raw signals for a future triage layer to consume.

`--profile-other-voice-threshold <0..1>` optionally pins a fixed other-voice cutoff
instead of the profile's adaptive band (leave it off to use the calibrated default).

## Three correctness points that matter

1. **Building from the same files you analyze is correct, not circular.**
   `analyze_audio` matches each analyzed recording to its profile contribution by
   **audio content hash** and applies **leave-one-file-out** automatically —
   it rebuilds the centroid *excluding that recording* before scoring it, so a
   file never inflates its own "target" similarity. You do **not** need to hold
   files out manually.

2. **Keep the profile's source files on disk at their build-time paths.**
   Leave-one-file-out re-extracts the *sibling* source files from the paths stored
   in the profile (the `file_id`s — i.e. whatever paths you passed at build time).
   If those siblings are moved/unreadable at analyze time, it falls back to scoring
   against the self-inclusive centroid and **warns** that the score is not
   leak-free. So: build with **absolute paths** (the `submit_subject_audio.sh`
   filelist already uses them) and don't relocate the audio between stages.

3. **Use the same embedding model set in both stages.** Score-level fusion only
   works on models the profile and the run share; `analyze_audio` warns if the
   profile's models don't overlap the run's `--embedding-models`. The defaults
   match (both default to the ECAPA + ResNet + WavLM trio), so the simplest path
   is to override neither.

## Performance caveat: the two stages don't share cache yet

The design *intends* for `build_speaker_profile` to pre-warm `analyze_audio`'s
per-file cache (diarization, speaker embeddings, scene classification) so each task
is computed once across both stages. **That cross-stage cache sharing is currently
deferred** (the helper ships but `analyze_audio` is not yet wired to it — see the
feature spec's FR-015 / Phase 6). Practically: building the profile does **not**
speed up a later `analyze_audio` run; the embedding compute happens in each stage
independently. Pointing both at the same `--cache-dir` is still correct and
harmless — it just won't produce cross-stage hits until that wiring lands.

## Recommended path for "embeddings ready, analysis later"

1. Run **Stage 1** for each subject now (GPU job) → one `profile.json` per subject
   under `derivatives/speaker_profile/<sub>/`. That is your ready-to-use embedding
   artifact.
2. Keep the source audio in place.
3. When a downstream spec needs them, add `--speaker-profile …/profile.json` to your
   existing `submit_subject_audio.sh` invocation (**Stage 2**) — no other change.

## One-command launcher

A local launcher chains both stages with a Slurm dependency, mirroring
`submit_subject_audio.sh`'s file discovery and output layout. It is **not tracked
in git** (it carries cluster-specific partitions/paths) — it lives under
`scripts/local/`:

- `scripts/local/slurm_build_profile_job.sh` — the GPU build job (one per subject).
- `scripts/local/submit_subject_with_profile.sh` — orchestrator: discovers the
  subject's files, submits the build job, then submits the analyze array gated on
  it (`--dependency=afterok`) with `--speaker-profile` already attached.

```bash
# Build the profile AND analyze every file with it (two chained jobs):
bash scripts/local/submit_subject_with_profile.sh <dataset_root> sub-001

# Build the profile only — embeddings ready, no analysis yet:
bash scripts/local/submit_subject_with_profile.sh <dataset_root> sub-001 --build-only

# Any trailing flags are forwarded to analyze_audio.py, e.g.:
bash scripts/local/submit_subject_with_profile.sh <dataset_root> sub-001 --skip yamnet
```

For your stated goal — embeddings ready now, analysis later — use `--build-only`
per subject, then run your normal `submit_subject_audio.sh … --speaker-profile …`
when you're ready.
