# CLAUDE.md

Guidance for Claude Code working in this repository.

**Setup, installation and the test/lint commands live in [README.md](README.md#development).**
That file is the source of truth and is verified against a clean clone; this file does not repeat it.
What is here is the rest: how the code is organised, the conventions that are not obvious from
reading it, and the traps that have cost real time.

## Project Overview

Senselab processes and analyses behavioural data — primarily voice and speech, also text and
video — through reproducible pipelines. uv for dependency management; the interpreter is pinned in
`.python-version` (3.12, matching CI).

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

- Google-style docstrings (enforced by ruff, `convention = "google"`)
- Line length 120; type hints required (mypy with the pydantic plugin)
- Tests in `src/tests/` mirroring the package, named `*_test.py`
- **Explain *why* in comments and docstrings, not *what*.** The codebase's convention is that a
  non-obvious choice records the measurement or failure that drove it, so a later reader can
  disagree with the reasoning rather than guess at it. Several sections below are that convention
  applied to the module docs.

## Traps that have cost time

- **Do not run `pytest -n auto`.** Each xdist worker is a separate interpreter that imports torch +
  transformers + speechbrain independently — **535 MB resident per worker before any test runs**,
  plus a private copy of any model weights that worker loads. It has exhausted a 32 GB machine.
  Measured on a 16-core node, `-n 16` also buys nothing on the fast suites: 68 s against 70 s
  serial, because the import cost equals the test time. Run the directory you changed instead.
  There is a second, independent reason, now **partly fixed**: `ensure_venv` does
  `shutil.rmtree(venv_dir)` before installing, and two workers wanting the same subprocess venv
  would delete each other's tree mid-install. It has held a `FileLock` around the whole
  marker-check / rmtree / install sequence since PR #444, gated on a `.senselab-installed`
  completion marker, so that specific race is closed. The macOS CI job carried `-n auto` and hung for 5.5 hours
  until GitHub's 6-hour ceiling killed it (run 31218624423) — all three workers stalled at the same
  instant, nothing completed afterwards. Every CI job now runs serially and the macOS job carries
  `timeout-minutes: 90`, so the next hang costs minutes. What the venv lock still lacks is a
  heartbeat: a holder that dies mid-install blocks every waiter for the full 600 s timeout and then
  raises, rather than being detected as dead and taken over.
- **`uv sync` is subtractive.** It removes extras not named in the command, so always pass the full
  set (`--all-extras`).
- **Cache invalidation is free.** Bump `CACHE_SCHEMA_VERSION` in
  `src/senselab/utils/tasks/cached_inference.py` rather than reasoning about which
  `artifacts/analyze_audio_cache/` entries survive. A stale entry that *looks* readable costs far
  more than recomputing one, and the wipe is automatic on every host.
- **Cache keys are commit-aware as of schema 23, so the first run after that change recomputes
  everything.** Keys used to carry a bare `model_id`, so an upstream push to a tracked ref loaded
  new weights under an unchanged key and served a result computed by the *old* commit as current.
  Keys now include the resolved 40-hex commit, which is what makes an upstream push invalidate on
  its own. Every pre-23 entry predates that and cannot be attributed to a commit, so none is
  reused — a one-time full recompute, not a regression, and worth saying out loud before someone
  reports it as one.
- **A model load must pass a commit SHA, never a ref.** Resolving `main` to a SHA binds nothing by
  itself: `snapshot_download(revision="main")` writes `refs/main` and the caller stays
  ref-addressed, so a later load passing `"main"` goes back through that pointer, which may have
  moved. Every load is therefore two calls — resolve, then load again with `revision=<sha>` — and
  the second is free, because a full SHA triggers `huggingface_hub`'s commit-hash shortcut and
  returns cached files with no network at all. `src/tests/utils/revision_pinning_guard_test.py`
  enforces this by AST sweep over the subprocess-worker files: a new worker payload carrying a
  `revision`-ish key fails the test until it is reviewed and allowlisted. Recording a SHA while
  loading through a ref is the one outcome worse than recording nothing, because the provenance is
  then confidently wrong.
- **Thresholds belong in `data/` with a written derivation, never as code literals.** Two defects
  this session came from literals that were never fitted: a silhouette coefficient read directly as
  a probability, and a 2→10 dB HNR ramp under which ordinary voiced speech (median 8.12 dB) read as
  only partly voiced. Regenerate a profile from measured verdicts; do not hand-edit one.
- **Pre-alpha: rename and replace outright.** No parallel fields, no aliases, no deprecation
  shims.

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

## Audio analysis: the uncertainty workflow

Two entry points over one module, `senselab.audio.workflows.audio_analysis`:

```bash
# Analyse: every model on the recording, content-addressably cached. Two arguments.
uv run python scripts/analyze_audio.py audio.wav [--out DIR]

# Adapt: the uncertainty-driven intervention loop over a completed run directory.
uv run python scripts/adaptive_loop.py artifacts/analyze_audio/<run>/
```

**The CLI is two arguments.** Everything else is one versioned config with each value's derivation
written beside it: `src/senselab/audio/workflows/audio_analysis/data/run_config/default.yaml`.
Override with a partial YAML deep-merged over the packaged one; the merged mapping's hash is stamped
into every artifact.

```bash
cat > variant.yaml <<'EOF'
models:
  asr: [openai/whisper-large-v3-turbo]
uncertainty:
  aggregator: mean
EOF
uv run python scripts/analyze_audio.py audio.wav --config variant.yaml
```

There are deliberately **no per-knob flags**. Seventy existed, and the shipped defaults of the four
grid flags put the four axes on four spacings sharing zero bucket keys — disabling every cross-axis
coupling while reporting that it had run. Adding a flag back is adding an unmeasured decision with a
public interface; add a config key with its derivation instead.

**The design and its reasoning live in the module's own docs**, which pdoc renders and which stay
next to the code they describe:

- [`workflows/audio_analysis/doc.md`](src/senselab/audio/workflows/audio_analysis/doc.md) — one grid,
  one fold, L1-measures/L2-decides, the four axes and their voters, outputs, public API, the
  background-scene and per-speaker-identity design.
- `specs/20260728-221507-per-speaker-identity-scene/layered-architecture.md` — decisions D-1…D-27.
- `specs/20260728-221507-per-speaker-identity-scene/l1-post-processing-register.md` — every L1/L2
  boundary violation, one row each, with its status and the measurement behind it. **Open items are
  tracked here, not in this file.**
- `specs/20260508-173136-compare-uncertainty/` — the comparator's contracts.
- `specs/20260506-154425-audio-analysis-asr-extensions/` — the ASR backend extensions
  (Canary-Qwen, Qwen3-ASR, MMS alignment) and the separable ASR/alignment caches.

Three id namespaces stay distinct because all three once rendered as `S0`: a model's own speaker
labels (`SPEAKER_00`, `spk0`), the pass-wide cluster harmonising labels across diarizers (`C0`), and
the fused speaker id in `final/speakers.json` (`S0`). Identity repair adds a fourth, `R*`.
