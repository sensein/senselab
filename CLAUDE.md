# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Senselab is a Python package for processing and analyzing behavioral data (primarily voice/speech, but also text and video) using reproducible pipelines. It uses uv for dependency management and requires Python 3.11-3.12.

## Build and Development Commands

```bash
# Install dependencies (full development setup)
uv sync --extra text --extra video --extra senselab-ai --extra nlp --extra pii --group dev --group docs

# Install the spaCy NLP model used by Presidio for PII detection
uv run python -m spacy download en_core_web_lg

# Install pre-commit hooks (required before committing)
uv run pre-commit install

# Run all tests with coverage
uv run pytest

# Run a single test file
uv run pytest src/tests/audio/tasks/preprocessing_test.py

# Run a specific test function
uv run pytest src/tests/audio/tasks/preprocessing_test.py::test_function_name -v

# Run tests in parallel
uv run pytest -n auto

# Type checking
uv run mypy .

# Linting
uv run ruff check
uv run ruff check --fix  # auto-fix issues

# Spell checking
uv run codespell

# Generate documentation locally
uv run pdoc src/senselab -t docs_style/pdoc-theme --docformat google
```

## Architecture

### Module Structure

```
src/senselab/
├── audio/           # Audio processing (largest module)
│   ├── data_structures/  # Audio, AudioClassificationResult
│   ├── tasks/            # Processing operations by capability
│   └── workflows/        # Composite pipelines (e.g., health_measurements)
├── video/           # Video processing
│   ├── data_structures/  # Video, Pose
│   └── tasks/            # pose_estimation, input_output
├── text/            # Text processing
│   └── tasks/            # embeddings_extraction
├── utils/           # Shared utilities
│   ├── data_structures/  # SenselabDataset, DeviceType, SenselabModel, Language
│   └── tasks/            # batching, dimensionality_reduction, pooling, etc.
└── agentic_interface/    # AI agent for Jupyter notebooks
```

### Key Design Patterns

**Data Containers**: `Audio` and `Video` classes use lazy loading (private `_waveform`/`_frames` with property accessors). All data structures are Pydantic v2 models.

**Model Abstraction**: `SenselabModel` is the base class with provider-specific subclasses (`HFModel`, `SpeechBrainModel`, `PyannoteAudioModel`, etc.) that validate model IDs on instantiation.

**Task Organization**: Each task module has an `api.py` with public functions and backend-specific implementations (e.g., `huggingface.py`, `speechbrain.py`). Functions accept data objects, model specs, device, and return typed results.

**Device Handling**: `DeviceType` enum (CPU, CUDA, MPS) with dtype mapping for cross-platform compatibility.

### Audio Tasks

Key audio processing capabilities in `audio/tasks/`:
- `speech_to_text/` - ASR transcription
- `speaker_verification/`, `speaker_embeddings/`, `speaker_diarization/`
- `speech_enhancement/`, `voice_activity_detection/`
- `voice_cloning/`, `text_to_speech/`
- `forced_alignment/`, `features_extraction/`
- `classification/` - includes speech emotion recognition
- `preprocessing/` - resample, filter, normalize
- `data_augmentation/`, `quality_control/`

## Code Style

- Google-style docstrings (enforced by ruff with `convention = "google"`)
- Line length: 120 characters
- Type hints required (mypy with pydantic plugin)
- Tests located in `src/tests/` mirroring the package structure
- Test files must be named `*_test.py`

## System Requirements

- macOS requires ARM64 (Apple Silicon); Intel Macs are not supported
- FFmpeg must be installed system-wide
- Docker required for some video models (MediaPipe-based estimators)
- CUDA 12.8 libraries for GPU support
- HuggingFace token (`HF_TOKEN` env var) for many models

## CUDA host configuration

Subprocess-venv backends (`nemo-canary-qwen`, `nemo`, `qwen-asr`) install
their own `torch` + `torchaudio` into isolated venvs at `~/.cache/senselab/venvs/`.
On hosts with system CUDA newer than the PyTorch default-wheels CUDA
(e.g. CUDA 12.9 against PyTorch's `cu128` default), the install would
otherwise resolve `torch` and `torchaudio` to mismatched CUDA toolchains
and break their ABI contract at import time.

The shared installer (`ensure_venv` in `src/senselab/utils/subprocess_venv.py`)
auto-detects the host CUDA via `nvidia-smi` / `nvcc` and routes the install
through the matching PyTorch wheel index (`cu128` / `cu126` / `cu124` /
`cu121` / `cpu`). No per-backend configuration needed.

Operator override for internal mirrors or unsupported CUDA versions:

```bash
# Force CPU wheels (e.g. unsupported CUDA, or testing CPU path on a GPU host)
SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu uv run python scripts/analyze_audio.py ...

# Internal PyPI mirror
SENSELAB_TORCH_INDEX_URL=https://pypi.internal.example.com/pytorch/cu128 uv run python ...
```

When no compatible wheel pair exists for the host (rare; happens for days
after a fresh CUDA major release), the failure surfaces as a
`SenselabCudaCompatibilityError` naming the host CUDA, the attempted index,
and the recommended action. See
`specs/20260512-204619-fix-canary-cuda-conflict/quickstart.md` for
validation recipes.

## Profiling with Scalene

Optional development tool for line-level CPU and memory profiling of any Python script or Jupyter notebook in the repo.

```bash
# Install (opt-in; default uv sync is unaffected)
uv sync --group profiling

# Profile a script
uv run python scripts/profile_with_scalene.py path/to/script.py

# Profile a tutorial notebook
uv run python scripts/profile_with_scalene.py tutorials/audio/speech_to_text.ipynb

# Restrict to senselab code only
uv run python scripts/profile_with_scalene.py --no-thirdparty examples/run.py
```

Reports are written to `artifacts/scalene/<target>_<timestamp>.html` (standalone, open in any browser). Constraints: notebooks must be non-interactive (no `input()` or widget event waits); child-process profiling is not supported by this wrapper in Scalene 2.2.

See `specs/20260503-235625-scalene-profiling/quickstart.md` for the full option reference, and `scripts/profile_imports.py` for the separate cold-start import-time profiler.

## Audio analysis script + ASR backend extensions

`scripts/analyze_audio.py` runs senselab's full task suite (diarization, AST, YAMNet, quality features, ASR, speaker embeddings) on an audio file with and without speech enhancement, with content-addressable caching plus full provenance and a hierarchical Label Studio export bundle.

**The CLI is two arguments.** Everything else is one versioned config with its derivation written beside each value: `src/senselab/audio/workflows/audio_analysis/data/run_config/default.yaml`.

```bash
# Full default pass (cache reused on subsequent runs)
uv run python scripts/analyze_audio.py path/to/audio.wav

# Somewhere else
uv run python scripts/analyze_audio.py audio.wav --out artifacts/experiment_3

# Change a value: a YAML with only the keys you are changing, deep-merged over the packaged one.
# The merged mapping's hash is stamped into every artifact, so the run can be named.
cat > one_asr.yaml <<'EOF'
models:
  asr: [openai/whisper-large-v3-turbo]
stages:
  align_asr: false
EOF
uv run python scripts/analyze_audio.py audio.wav --config one_asr.yaml
```

There are deliberately **no per-knob flags**. Seventy existed; the recipes in this file differed only
in flags whose right value a reader had no basis to choose, and the shipped defaults of the four grid
flags put the four uncertainty axes on four spacings sharing zero bucket keys — which disabled every
cross-axis coupling in the pipeline while reporting that it had run. Adding a flag back is adding an
unmeasured decision with a public interface; add a config key with its derivation instead. The
config's `{name, version, config_hash, sources}` travels into `final/summary.json`, the comparator
params on every fused row, and `disagreements.json`.

New senselab APIs landed alongside the script:

- `senselab.audio.tasks.forced_alignment` — multilingual MMS aligner (`facebook/mms-1b-all`) covering ~1100 languages. Pass `aligner_model=MMS_MODEL_ID` to `align_transcriptions`. Japanese / Chinese transcripts are auto-romanized via `uroman` (install via `uv sync --extra nlp`).
- `senselab.audio.tasks.speech_to_text.canary_qwen` — NVIDIA Canary-Qwen 2.5B (text-only) via NeMo SALM in an isolated `nemo-canary-qwen` venv. Auto-routed when the model id matches `nvidia/canary-`.
- `senselab.audio.tasks.speech_to_text.qwen` — Alibaba Qwen3-ASR 1.7B / 3B via the `qwen-asr` Python wrapper in an isolated `qwen-asr` venv, with the bundled `Qwen3-ForcedAligner-0.6B` companion enabled by default for native word-level timestamps. Auto-routed when the model id matches `Qwen/Qwen3-ASR`.
- IBM Granite Speech 3.3 8B — text-only via the existing HF pipeline path; the script's auto-align stage adds per-segment MMS timestamps downstream.

ASR cache and alignment cache are separable: re-running with a different aligner does not invalidate the (slow) ASR result. See `specs/20260506-154425-audio-analysis-asr-extensions/` for the full design.

### Adaptive uncertainty loop (acts on the three axes)

`scripts/adaptive_loop.py` runs a deterministic, budgeted intervention loop over a completed
analyze_audio run dir: ingests the 9 uncertainty parquets into a provenance-tagged vote store
(re-aggregation is pure — parity-checked against the stored parquets), proposes high-uncertainty
regions per axis, executes a policy-ranked catalog (stream election, hallucination adjudication,
cache-replay/live ASR escalation, embedding change-point + re-cluster identity repair, gated
segmentation-3.0 overlap detection), and fuses a consensus transcript / refined diarization /
presence track with a byte-reproducible decision log (`final/iterations.json`,
`final/convergence.json`, `final/timeline.png`). Policy (thresholds/budgets/model pools) is the
`adaptive:` **section of the run config** — `data/run_config/default.yaml`, override the whole file
with `--config`; a file with `thresholds:` / `fusion:` / `rules:` at the top level is *refused*
rather than merged where nothing reads it. It keeps its own `policy_hash` beside the config's
`config_hash`, because a policy change and a model change are not the same event. `enhancement.mode`
(`auto` | `always` | `never`) plus the `triage:` block carry the round-0 frame-posterior speech gate
and SNR enhancement gate (`always`, the default, runs both passes unconditionally — which is what
makes raw and enhanced a perturbation *sample*); `profiles.calibration` carries the US5 scene-quality
calibration (versioned dB→[0,1] anchors, fit via `scripts/calibrate_scene_quality.py`, bridge in
`workflows/audio_analysis/calibration.py`; its `temperature` block currently reaches no fold — see
the note there). The
comparator is split into `harvest_pass` (model-touching, `compute.py`) + `aggregate_pass` (pure,
`votes.py`); `compute_uncertainty_axes` is a compatible wrapper (`mutate_passes=False` for the
side-effect-free API). Design/spec/results: `specs/20260723-225523-dynamic-uncertainty-workflow/`
(tasks.md Phase 8 lists the open follow-ups; README "Adaptive audio analysis" has the user runbook).

### Three-axis uncertainty workflow

The reusable comparator lives at `senselab.audio.workflows.audio_analysis`. The CLI script `scripts/analyze_audio.py` is a thin wrapper: per-task pipeline → one call to `compute_uncertainty_axes(...)` → parquet writers + LS bundle + disagreements index + timeline plot.

The workflow emits four per-bucket uncertainty time series — `speech_presence` (was there a speaker?), `speaker` (**who** is speaking here?), `asr` (what was said?) and `background_mask` (is this region free of target activity?) — each in `[0, 1]`. Every model whose output naturally encodes an axis votes; `fuse.fuse_axis` is the **one** fold, weighting each signal by its measured perturbation stability and physical support and collapsing via `uncertainty.aggregator`.

**Every axis is on one grid** (`axes.DEFAULT_TIME_GRID`, 0.1 s window == 0.1 s hop), so row *i* of one axis is row *i* of another and a cross-axis join needs no reconciliation. Measured before it was: 242 / 242 / 19 / 8 rows on 0.1/0.02, 0.1/0.02, 0.25/0.25 and 1.0/0.5, sharing **zero** bucket keys — so coupling did nothing and every round came out byte-identical. Window equals hop deliberately: a 0.1 s window at a 0.02 s hop reports five near-duplicate rows per window, and nothing in the output said so.

The `speaker` axis measures **attribution**, composed by `attribution.py` from three voters: per-speaker presence doubt (`max` over the speakers present of the entropy of the model share — the same quantity `final/per_speaker_presence.parquet` publishes), ASR word-location doubt (`1 - temporal_confidence` over the words reaching the bucket, since word boundaries are what assign a word to a speaker's span), and target-activity doubt (the mask region's uncertainty, only where its `state` is not `target_active`). A bucket the mask confidently calls `target_free` carries no vote: there is nobody to attribute. It asked "was it the same speaker as before?" until 2026-08-05 — a change question asked at the grid rate against embeddings windowed ten times coarser, which read 0.666 on a clean two-speaker conversation whose per-speaker presence doubt was 0.168. The cosines, calibrated readings, change points and overlap distribution survive as L1 measurements.

The `asr` axis has **one** voter, `consensus_words`: the recognizers' words are fused once per pass (`asr._consensus_word_doubt`, graded phonemically by `asr.phoneme_similarity`) and each bucket takes the coverage-weighted mean of `1 - existence_confidence` over the words reaching it. There is no per-bucket text — that was a reconstruction of what `final/transcript.json` holds at word resolution, and it is what forced this axis onto a 1.0 s grid, since a fully-contained text read returns nothing from a bucket narrower than a word. Localisation lives on the word (`onset_confidence` / `offset_confidence`), not in the axis number.

**L1 measures, L2 decides.** L1 reports what a tool produced, in that tool's units, at its own
resolution: no thresholds, no rescaling to `[0, 1]` against an anchor, no reduction across a
dimension the tool reported separately, no selection among estimators. Every interpretation lives
at L2, where it is named and can be changed without re-running a model. Concretely for the
speech-presence axis: `speech_presence.harvest_speech_presence_evidence` emits measurements
(segment `covered_fraction`, transcript `word_overlap_s`, per-chunk `avg_logprobs`, `excess_db`,
`frame_mean` + `channel_means`), and `speech_presence_link.link_speech_presence` turns them into
votes under a `SpeechPresencePolicy` recorded in each row's provenance. Consumers that need beliefs
(`support.py`, `fuse.py`, `adaptive/belief.py`) call `speech_presence_link.votes_for_harvest`;
`PassHarvest.speech_presence_evidence` holds the measurements. Scene quality follows the same
split: `quality.harvest_quality_measurements` (dB / hertz / proportion) →
`degradation.scene_degradation` (anchored scores). Remaining violations and their status are
tracked one-by-one in
`specs/20260728-221507-per-speaker-identity-scene/l1-post-processing-register.md`; the governing
design and its decisions D-1 – D-16 are in the sibling `layered-architecture.md`.

**An uncertainty axis IS an aggregator** (D-16). It aggregates across signals *and* across
passes, so there is no such thing as a per-pass axis — a pass is an input dimension to the fold,
never an index on its output. Passes are a *perturbation sample*: a signal whose answer flips
between them has not earned its weight, which is what `reliability.signal_stability` measures,
per signal, and what sets each signal's fusion weight. L1 emits
`L1/<pass>/signals/<signal>.parquet` in native units and nothing under `L1/` is named for an
axis; the single fold is `fuse.fuse_axis`, which receives every pass at once and reports the pass
dimension only as each row's `contributing_passes` column.

Output:

- `<run_dir>/L1/<pass>/signals/<signal>.parquet` — one row per (signal, bucket), the measurement in the tool's own units; units / window / hop / model / revision in `schema.metadata`.
- `<run_dir>/L1/stability/<signal>.parquet` + `signals.json` — cross-pass `|Δ|` per bucket and the run-level mean that sets each signal's fusion weight.
- `<run_dir>/L1/passes.json` — the small index later stages read (duration, audio signature, input path).
- `<run_dir>/L2/round<N>/uncertainty/<axis>.parquet` — the four fused axes, all on one grid: `uncertainty`, `epistemic_uncertainty`, `confidence`, `variability`, `triage_score`, `contributing_signals`, `contributing_passes`, `signal_weights`, `weight_basis`, `round`.
- `<run_dir>/L2/round0/votes/<axis>.parquet` — the linked evidence at the vote level, keyed `(axis, bucket, source, pass, scope)`; what the adaptive store ingests.
- `<run_dir>/final/disagreements.json` — top-N ranked over the fused axes by `triage_score`, axis-priority tiebreak (asr > speaker > speech_presence). No `pass` field: an axis has no pass.
- `<run_dir>/final/uncertainty_detail.png` — one line per axis with `epistemic_uncertainty` shaded beneath, plus a per-signal stability strip and per-source detail rows. An axis view is a conclusion, so it lives under `final/`, never under `L1/`; the evidence figure with no conclusions on it is `L1/signals.png`.
- `<run_dir>/final/timeline.png` — the adaptive loop's view: fused words, interventions and run state.
- LS Labels tracks `uncertainty__<axis>` (attached once) plus per-pass evidence tracks `<pass>__signal__<signal>`. **No transcript TextArea**: the words are published at word resolution in `final/transcript.json` and rendered as `final__consensus_transcript__text` by `adaptive.ls_final`, so the bundle carries one rendering of the transcript rather than two at two resolutions.

```bash
# Default — runs the full pipeline including the workflow
uv run python scripts/analyze_audio.py audio.wav

# Standalone use of the workflow API (no script)
uv run python -c "
from senselab.audio.workflows.audio_analysis import BucketGrid, compute_uncertainty_axes
signals, fused_axes, incomparable, embeddings = compute_uncertainty_axes(
    passes=passes_summary,    # the dict produced by analyze_audio's per-task run_pass
    grid=BucketGrid(),         # = axes.DEFAULT_TIME_GRID; every axis is on it, no per-axis override
    params={...},
    audio={'raw_16k': audio},
    speaker_embedding_models=['speechbrain/spkrec-ecapa-voxceleb'],
    aggregator='min',
    speech_presence_labels=['Speech', 'Conversation', 'Narration, monologue'],
)
"

# Skip the workflow entirely, or switch the aggregator from "max-doubtful" (default min) to
# "average-doubt": both are run-config keys, not flags.
cat > variant.yaml <<'EOF'
stages:
  comparisons: false
uncertainty:
  aggregator: mean
EOF
uv run python scripts/analyze_audio.py audio.wav --config variant.yaml
```

See `specs/20260508-173136-compare-uncertainty/` for the full design (spec.md, plan.md, contracts/cli.md, contracts/uncertainty-row.parquet.md, contracts/disagreements.json.md, contracts/ls-bundle.md, quickstart.md).

### Background scene characterization and per-speaker identity

Background sound sources are detected by **per-band noise-floor subtraction**, not by
amplification. Measurement drove that: neither scene classifier normalizes input level
(both are amplitude-sensitive), and amplification changes no signal-to-noise ratio — it
moves a source and the residual foreground together. What gain fixes is a classifier's
absolute floor; what it cannot fix is a source buried under leaked foreground.

```bash
# Probe whether the classifiers self-normalize. Cached checkpoints only, never downloads.
uv run python scripts/probe_classifier_levels.py --input clip.wav --out artifacts/level_probe/

# Full run with the mask and background characterization
# task.type: speech in the run config selects what counts as the participant's own activity
uv run python scripts/analyze_audio.py clip.wav
```

Key pieces, and the reasoning that shaped each:

- **`noise_floor.py`** — bias-corrected per-band floor. A tenth-percentile estimate sits
  ~9.8 dB below the true mean noise power; uncorrected, every relative-dB gate is that much
  more permissive. Uses a 100 ms frame: the floor is a long-term percentile and needs
  *frequency* resolution, not time resolution — a 25 ms frame cannot resolve below ~140 Hz,
  where mains hum and ventilation live. A source running through the whole clip is absorbed
  into its own band floor, so `detect_stationary_sources` compares bands against their
  neighbours instead (ECMA-74 prominence, ≥9 dB).
- **`sources.py`** — the corroborated **3 / 6 / 10 dB** ladder above the band floor, plus
  four fabrication guards. The failure mode is not a missed source but a *fabricated* one:
  amplified noise floor produces confident water-like labels indistinguishable from genuine
  broadband noise.
- **`background_mask.py`** — regions free of **target** activity (not free of speech).
  What counts as target comes from `task.type`: in a breathing task, speech detection is
  silent during the target event, and since AudioSet maps `Breathing` to `people`, a mask
  built from voice activity alone reports the collected signal as a background source.
- **`foreground.py`** — suppression depth is the binding constraint, measured by
  *projection* rather than level. Two residuals at identical power license opposite
  conclusions (leaked speech vs genuine background).
- **`speaker_identity.py`** — speaker-count posterior keeping multi-modal disagreement, with
  source reliability **derived from perturbation evidence** rather than assigned. The raw
  and enhanced passes are the same recording under a transform, so they already constitute a
  stability sample; a source that flips between them has not earned its weight.
- **`adaptive/influence.py`, `adaptive/provenance.py`** — uncertainty-gated mutual influence,
  with the self-confirmation guard: uncertainty falling *because a value was overwritten* is
  not a confidence gain.

Thresholds live in `data/detection_margin/<version>.json` with a written derivation, never
as code literals. Regenerate one from measured verdicts rather than editing it by hand:

```bash
uv run python scripts/calibrate_detection_margin.py \
    --level-verdicts artifacts/level_probe/level-verdicts.json \
    --out src/senselab/audio/workflows/audio_analysis/data/detection_margin/<name>.json
```

It refuses to emit a profile with no measured floor, one whose confident tier sits above
every measured classifier floor (a threshold already known unreachable on that host), or one
carrying an unmarked provisional figure. `profile_version` is the *schema* version and is
never restamped; the profile's identity is `calibrated_as` plus its filename.

Outputs: `<pass>/background_mask.{parquet,json}`, `<pass>/noise_floor.parquet`,
`<pass>/background_sources.parquet`, `<pass>/suppression.json`, `final/speakers.json`,
`final/per_speaker_presence.parquet`, plus `<pass>__background__mask` and
`<pass>__speaker__presence` tracks in the Label Studio bundle. Design and evidence:
`specs/20260728-221507-per-speaker-identity-scene/`.

Three id namespaces stay distinct because all three once rendered as `S0`: a model's own
speaker labels (`SPEAKER_00`, `spk0`), the pass-wide cluster that harmonises labels across
diar models (`C0`), and the fused speaker id in `final/speakers.json` (`S0`).

## Active Technologies
- N/A (CI/CD configuration only — YAML, JSON) + Intuit Auto (v11.2.1), hatch-vcs, GitHub Actions (20260418-104204-alpha-prerelease-process)
- Bash (setup script), YAML (GitHub Actions workflows) + machulav/ec2-github-runner@v2.5.2, aws-actions/configure-aws-credentials@v6, aws CLI, gh CLI (20260418-120722-aws-gpu-test-setup)
- N/A (ephemeral instances) (20260418-120722-aws-gpu-test-setup)
- Python 3.11-3.12, Bash + orch, transformers, speechbrain, pyannote-audio, coqui-tts, ppgs/espnet, sentence-transformers (20260419-133236-test-classification-deps)
- Python 3.11-3.14 (Colab uses 3.12) + papermill (notebook execution), senselab (the library being tutorialized) (20260420-212321-fix-colab-tutorials)
- N/A (notebooks are files in the repo) (20260420-212321-fix-colab-tutorials)
- Python 3.11-3.12 (Colab uses 3.12) + senselab (the library being tutorialized), papermill (CI execution), ipywebrtc or JS widgets (recording) (20260423-213942-pedagogical-tutorials)
- Python 3.11-3.12 (Colab uses 3.12) + senselab, transformers (for text sentiment pipeline) (20260424-152323-improve-ser-tutorial)
- YAML (GitHub Actions), Markdown + pdoc, JamesIves/github-pages-deploy-action@v4, peter-evans/create-or-update-commen (20260424-232054-docs-pr-preview)
- GitHub Pages (`docs` branch) (20260424-232054-docs-pr-preview)
- Python 3.11-3.12 + s3prl (subprocess venv), speechbrain, pyannote-audio, nemo_toolkit (subprocess venv), transformers (20260428-101838-expand-speech-models)
- Python 3.11-3.12 + ransformers (HuggingFace audio-classification pipeline), existing senselab classification module (20260429-201758-auditory-scene-analysis)
- Python 3.11-3.12 (managed via uv) + stdlib only (subprocess, json, time, re, pathlib) — the profiling script itself has no heavy deps; it invokes senselab imports in child processes (20260501-154228-optimize-import-times)
- File-based (Markdown report output to `artifacts/`) (20260501-154228-optimize-import-times)
- Python 3.11-3.14 (managed via uv) — matches senselab's `requires-python` + scalene (new optional dep, opt-in via `--group profiling`); jupyter nbconvert (already present transitively via `senselab-ai` extra) (20260503-235625-scalene-profiling)
- File-based — HTML/JSON reports written to `artifacts/scalene/` (20260503-235625-scalene-profiling)
- Python 3.11–3.14 (managed via uv) — matches senselab's `requires-python`. (20260506-154425-audio-analysis-asr-extensions)
- File-based — JSON outputs under `artifacts/analyze_audio/`; persistent cache under `artifacts/analyze_audio_cache/`; subprocess venvs under `~/.cache/senselab/venvs/{nemo-canary-qwen,qwen-asr}/`. (20260506-154425-audio-analysis-asr-extensions)
- Python 3.11–3.14 (managed via uv) — matches senselab's `requires-python`. + senselab (the merged audio analysis module from PR #510), pandas + pyarrow (already in the active venv via the prior PR's features pipeline), `jiwer` for WER (already in the `[nlp]` extra), `g2p-en` or similar small G2P library for grapheme→phoneme on the ASR side (new, ~1 MB). (20260508-173136-compare-uncertainty)
- File-based — parquet under `<run_dir>/<pass>/comparisons/<task>.parquet` and `<run_dir>/<pass>/comparisons/cross_stream/<a>_vs_<b>.parquet`; JSON for `<run_dir>/disagreements.json`; XML/JSON appendage to the existing LS bundle. (20260508-173136-compare-uncertainty)
- Python 3.11–3.14 (managed via uv), matches senselab's `requires-python`. (20260508-173136-compare-uncertainty)
- file-based — (20260508-173136-compare-uncertainty)
- Python 3.11–3.14 (matches senselab's `requires-python = ">=3.11,<3.15"`). + `uv` (managed installer), `torch>=2.8,<2.9`, `torchaudio>=2.8,<2.9` (currently pinned at the subprocess-venv definitions), `nemo_toolkit[asr,tts]`, `qwen-asr`. Fix introduces no new runtime dependency. (20260512-204619-fix-canary-cuda-conflict)
- File-based — venvs live under `~/.cache/senselab/venvs/<name>/`, marker file `.senselab-installed` records the current resolved requirement set. (20260512-204619-fix-canary-cuda-conflict)
- Python 3.11–3.12 (repo `requires-python = ">=3.11,<3.15"`), managed via uv + pyannote-audio (existing — adds `segmentation-3.0` raw-scores + `brouhaha` via `Model`/`Inference`), transformers (AST + Whisper token logits), torchaudio + torchaudio-squim (existing), librosa (promote from transitive → explicit), numpy/scipy (calibration), pandas/pyarrow (existing parquet), jiwer (existing) (20260722-175022-scene-quality-utterance)
- File-based — parquet under `<run_dir>/<pass>/uncertainty/{presence,identity,utterance}.parquet`; checked-in category map JSON and calibration profile JSON under the package; validation artifacts under `artifacts/` (20260722-175022-scene-quality-utterance)
- Python 3.11–3.14 (repo `requires-python = ">=3.11,<3.15"`), managed via `uv` + numpy, scipy, pandas + pyarrow (parquet), torch/torchaudio, transformers (AST), TensorFlow Hub (YAMNet), pyannote-audio (diarization, brouhaha SNR/C50), speechbrain (embeddings, enhancement), librosa (**promote transitive → explicit**: `pcen`, `A_weighting`), pyloudnorm (**new**, BS.1770 LUFS; numpy/scipy only) (20260728-221507-per-speaker-identity-scene)
- File-based — parquet under `<run_dir>/<pass>/uncertainty/` and `<run_dir>/final/`, JSON for convergence/decision logs, content-addressable cache under `artifacts/analyze_audio_cache/` (20260728-221507-per-speaker-identity-scene)

## Recent Changes
- 20260418-104204-alpha-prerelease-process: Added N/A (CI/CD configuration only — YAML, JSON) + Intuit Auto (v11.2.1), hatch-vcs, GitHub Actions
