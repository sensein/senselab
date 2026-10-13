[![Build](https://github.com/sensein/senselab/actions/workflows/main-branch-status.yaml/badge.svg)](https://github.com/sensein/senselab/actions/workflows/main-branch-status.yaml)
[![codecov](https://codecov.io/gh/sensein/senselab/graph/badge.svg?token=9S8WY128PO)](https://codecov.io/gh/sensein/senselab)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[![PyPI](https://img.shields.io/pypi/v/senselab.svg)](https://pypi.org/project/senselab/)
[![Python Version](https://img.shields.io/pypi/pyversions/senselab)](https://pypi.org/project/senselab)
[![License](https://img.shields.io/pypi/l/senselab)](https://opensource.org/licenses/Apache-2.0)

[![pages](https://img.shields.io/badge/api-docs-blue)](https://sensein.group/senselab)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/sensein/senselab)

# ```senselab```
This Python package **streamlines**, **optimizes**, and **enforces best open-science practices** for processing and analyzing _behavioral data_ (primarily voice and speech, but also text and video) using robust reproducible pipelines and utilities.

## Quick start
```Python
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import resample_audios
from senselab.audio.tasks.features_extraction import extract_features_from_audios
from senselab.audio.tasks.speech_to_text import transcribe_audios

audio = Audio(filepath='path_to_audio_file.wav')
print(audio.sampling_rate)
# ➡️ 44100

[resampled_audio] = resample_audios([audio], resample_rate=16000)
print(resampled_audio.sampling_rate)
# ➡️ 16000

audio_features = extract_features_from_audios([audio])
print(audio_features[0].keys())
# ➡️ dict_keys(['opensmile', 'praat_parselmouth', 'torchaudio', 'torchaudio_squim', ...])

transcript = transcribe_audios([audio])
print(transcript)
# ➡️ "The quick brown fox jumps over the lazy dog."
```

For more detailed information, check out our [**Documentation**](https://sensein.group/senselab) and our [**Tutorials**](https://github.com/sensein/senselab/blob/main/tutorials/audio/00_getting_started.ipynb).

💡 **Tip**: Many tutorials include Google Colab badges and you can try them instantly without installing anything on your local machine.



### Why should you use ```senselab```?
- **Modular design**: Easily integrate or use standalone transformations for flexible data manipulation.
- **Pre-built pipelines**: Access pre-configured pipelines to reduce setup time and effort.
- **Reproducibility**: Ensure consistent and verifiable results with fixed seeds and version-controlled steps.
- **Easy integration**: Seamlessly fit into existing workflows with minimal configuration.
- **Extensible**: Modify and contribute custom transformations and pipelines to meet specific research needs.
- **Comprehensive documentation**: Detailed guides, examples, and documentation for all features and modules.
- **Performance optimized**: Efficiently process large datasets with optimized code and algorithms.
- **Interactive examples**: Jupyter notebooks provide practical examples for deriving insights from real-world datasets.
- **senselab AI**: Interact with your data through an AI-based chatbot. The AI agent generates and runs senselab-based code for you, making exploration easier and giving you both the results and the code used to produce them (perfect for quick experiments or for users who prefer not to code).


---

## Adaptive audio analysis (uncertainty-driven)

senselab can analyze a recording with its full task suite (diarization, scene/event tagging, quality
metrics, multi-model ASR + forced alignment, speaker embeddings), quantify **where** the models are
uncertain along three temporal axes — *presence* (is someone speaking?), *identity* (who?), and
*utterance* (what was said?) — and then **act on that uncertainty**: a deterministic, budgeted loop
re-processes only the uncertain regions (extra ASR models, embedding re-clustering, overlap
detection), fuses a consensus transcript/diarization, and explains any residual uncertainty instead
of hiding it. Design + results: `specs/20260723-225523-dynamic-uncertainty-workflow/`.

It runs in two steps:

```bash
# Step 1 — analyze: run every model on the recording (results are content-addressably
# cached, so re-runs are cheap). Two arguments: the audio, and where results go.
uv run python scripts/analyze_audio.py path/to/recording.wav
# → artifacts/analyze_audio/<name>_<timestamp>/  (L1 per-signal parquets, L2 fused
#   axes, Label Studio bundle, disagreements.json, final/ deliverables)

# Step 2 — adapt: run the uncertainty-driven loop over that run directory.
uv run python scripts/adaptive_loop.py artifacts/analyze_audio/<run_dir> \
    --cache-dir artifacts/analyze_audio_cache \
    --ground-truth path/to/labelstudio_export.json   # optional: scores vs human labels
```

**`analyze_audio.py` takes an audio file and `--out`, and nothing else.** Every other value — the
model ids, the bucket grid, the aggregator, the task type, the triage and enhancement gates, which
stages run — lives in one versioned file with its derivation recorded beside it:
`src/senselab/audio/workflows/audio_analysis/data/run_config/default.yaml`. To change something,
write a YAML holding only the keys you are changing and pass `--config my.yaml`; it deep-merges over
the packaged one, and the merged mapping's hash is stamped into every artifact's provenance, so a run
can always be named. There are deliberately no per-knob flags: the seventy that preceded this
differed in ways a reader had no basis to choose between, and the *shipped defaults* of the four grid
flags put the four uncertainty axes on four spacings that shared no bucket keys — silently disabling
every cross-axis coupling in the pipeline.

The loop writes, under the run directory (or `--out`):

- `final/transcript.json` — consensus word-level transcript (family-weighted voting across all ASR
  models) with speaker attribution, per-word confidence, and alternates where models disagree;
- `final/diarization.json` — refined speaker segments (embedding change-point + re-clustering
  repair);
- `final/estimates/<axis>.parquet` — the last round's estimate of every active axis, extracted
  verbatim from `L2/round/<last>/estimates/`. A number in `final/` that is not in the last round
  was computed at the wrong stage, so this directory only copies;
- `final/speakers.json` + `final/per_speaker_presence.parquet` — the speaker-count posterior with
  its per-speaker hypotheses, and one presence track per hypothesised speaker;
- `final/decisions.json` — what the loop did and why: every intervention (fired / deferred /
  blocked) with trigger values and measured uncertainty deltas, budget accounting, and regions
  marked `converged` / `irreducible` (with a machine-readable reason). Each round's own slice of
  this is in its `L2/round/<n>/summary.json`;
- `final/timeline.png` — ground truth (if given) vs presence / identity / utterance uncertainty per
  round, interventions, and the confidence-colored fused words;
- `final/labelstudio_{tasks,config}.{json,xml}` + `disagreements_resolved.json` — the original Label
  Studio bundle with `final__*` consensus tracks added, and the round-1 disagreements annotated with
  their resolutions.

All thresholds, budgets and model pools live in the `adaptive:` section of that same run config —
round count, aggregator, per-run intervention budgets, ASR reserve/escalation pools, identity-repair
parameters — and it keeps its own `policy_hash` beside the config's `config_hash`, because a policy
change and a model change are not the same event. `scripts/adaptive_loop.py` takes `--config` too.
Runs are **deterministic**: identical inputs + config produce byte-identical decision logs. `HF_TOKEN` enables the gated
`pyannote/brouhaha` frame posteriors; without it the loop degrades gracefully and records
the skipped intervention in `final/decisions.json → convergence.next_actions`.

---

## Background scene characterization

Detects background sound sources — people, machines, environment — beneath a
near-microphone foreground speaker, and reports **how far above the noise floor** each one
sits so a marginal finding is never mistaken for a confident one.

```bash
# `task.type` in the run config selects what counts as the participant's own activity.
uv run python scripts/analyze_audio.py recording.wav
```

Three things worth knowing before reading the output:

**Detection is floor subtraction, not amplification.** Amplification moves a source and the
leaked foreground together, so it changes no signal-to-noise ratio. It is capped at 10 dB
and used only to keep a classifier's absolute floor from destroying quiet content.

**Every finding carries its margin above the band noise floor**, on a 3 / 6 / 10 dB ladder
corroborated independently by human masked-threshold criteria, a dozen bioacoustics and
noise-standard traditions, and the classifiers' own measured detection floors.

**A null result is attributable.** Suppression depth is reported alongside, so "no
background found" is distinguishable from "suppression was too shallow to look".

The background mask marks where claims are trustworthy without relying on suppression at
all. `task.type` matters: in a breathing or cough task the target event *is* a non-speech
vocal sound, and a mask built from voice activity alone would report the collected signal as
a background source.

## ⚠️ System Requirements
1. **If on macOS, this package requires an ARM64 architecture** due to PyTorch 2.2.2+ dropping support for x86-64 on macOS.

    ❌ Unsupported systems include:
    - macOS (Intel x86-64)
    - Other platforms where dependencies are unavailable

    To check your system compatibility, please run this command:
    ```bash
    python -c "import platform; print(platform.machine())"
    ```

    If the output is:
    - `arm64` → ✅ Your system is compatible.
    - `x86_64` → ❌ Your system is not supported.

    If you attempt to install this package on an unsupported system, the installation or execution will fail.

2. **`FFmpeg` shared libraries** are required. The consumer is `torchcodec`, which `dlopen`s them at
   import time *by soname* (`libavutil.so.56` / `.57` / `.58` / `.59`, one attempt per supported major).
   Two consequences worth knowing before you debug this:

   - **The `av` (PyAV) wheel does not satisfy it**, even though it ships ffmpeg libraries inside your
     environment. PyAV mangles their filenames on purpose (`av.libs/libavutil-3591eddc.so.60.8.100`) so
     they cannot collide with a system ffmpeg, which also makes them invisible to a soname lookup.
   - **Without them, no test collects at all.** `src/tests/conftest.py` reports
     `Dependencies failed to import — test environment is broken`, including for tests that never open
     an audio file.

   If you have no system ffmpeg, or no root, this repo installs it for you via conda-forge into a
   prefix you choose:

   ```bash
   bash scripts/install-ffmpeg.sh                       # defaults to /opt/miniforge
   CONDA_PREFIX=~/ffmpeg bash scripts/install-ffmpeg.sh # anywhere writable
   export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"    # macOS: DYLD_LIBRARY_PATH
   ```

   This is what CI uses on every platform. Otherwise install ffmpeg (`<8`) system-wide — see
   [ffmpeg.org](https://www.ffmpeg.org/download.html).

3. CUDA libraries matching the CUDA version expected by the PyTorch wheels (e.g., the latest pytorch 2.8 expects cuda-12.8). To install those with conda, please do:
  - ```conda config --add channels nvidia```
  - ```conda install -y nvidia/label/cuda-12.8.1::cuda-libraries-dev```

    **Hosts with newer system CUDA** (e.g., CUDA 12.9): the subprocess-venv backends (`nemo-canary-qwen`, `nemo`, `qwen-asr`) auto-detect the host's CUDA version via `nvidia-smi` and route their `torch`/`torchaudio` installs through the matching PyTorch wheel index (`cu128` / `cu126` / `cu124` / `cu121` / `cpu`). No manual configuration needed.

    **Internal mirrors / unsupported CUDA / CPU fallback**: set the `SENSELAB_TORCH_INDEX_URL` environment variable to override the chosen index. Common values:

    ```bash
    # Force CPU wheels (e.g. testing CPU path on a GPU host, or unsupported CUDA)
    export SENSELAB_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

    # Internal PyPI mirror that proxies PyTorch wheels
    export SENSELAB_TORCH_INDEX_URL=https://pypi.internal.example.com/pytorch/cu128
    ```

    When no compatible `torch`+`torchaudio` binary pair exists for your host (rare; happens in the days after a CUDA major release), installation fails with a named `SenselabCudaCompatibilityError` that lists the detected host CUDA, the attempted index URL, and the recommended action — no opaque stack traces from inside `torchaudio`.
4. Docker is required and must be running for some video models (e.g., MediaPipe-based estimators).
Please follow the official installation instructions for your platform: [Install Docker](https://docs.docker.com/get-started/get-docker/).
5. Some functionalities rely on HuggingFace models, and increasingly, models require authentication and signed license agreements. Instructions on how to generate a Hugging Face access token can be found here: https://huggingface.co/docs/hub/security-tokens
  - You can provide your HuggingFace token either by exporting it in your shell:
    ```bash
    export HF_TOKEN=your_token_here
    ```
  - or by adding it to your `.env` file (see `.env.example` for reference).

---

## Installation

**Python 3.11–3.14**, declared as `>=3.11,<3.15` in `pyproject.toml`. For development the repo pins
the interpreter in `.python-version` (**3.12**, matching CI's default), which `uv` reads, so a bare
`uv sync` is deterministic.

Install this package via:

```sh
pip install 'senselab[all]'
```

Or get the newest development version via:

```sh
pip install 'git+https://github.com/sensein/senselab.git#egg=senselab[all]'
```

If you want to install only audio dependencies, you do:
```sh
pip install 'senselab'
```

The declared extras are `nlp`, `text`, `video`, `senselab-ai`, and `all` (every one of them).
To pick a subset:
```sh
pip install 'senselab[video,text,senselab-ai]'
```

There is no `articulatory` extra — it was documented here and in `CONTRIBUTING.md` but never declared,
so `uv sync --extra articulatory` fails outright and `pip install 'senselab[articulatory]'` warns and
installs base only.

### Released vs pre-release

Merging a PR into `alpha` publishes an **alpha pre-release** automatically (`release.yaml` →
`auto shipit`); merging into `main` publishes a release. So there are two lines on PyPI, and `--pre`
is how you choose:

```sh
pip install 'senselab[all]'          # the released line (currently 1.3.0)
pip install --pre 'senselab[all]'    # the newest alpha from the alpha branch (1.3.1aN)
```

This is why every tutorial carries `--pre` — notebooks track the alpha branch:

```python
!pip install -q uv
!uv pip install --pre --system "senselab[nlp,text,video]"
```

Colab images happen to ship `ffmpeg`, so notebooks work there without installing it. The guarded
fallback for images that do not — and the `HF_TOKEN`-from-Colab-secrets snippet — is the setup-cell
template in [`tutorials/README.md`](tutorials/README.md).

**None of this applies to development.** `--pre` installs a *published artifact*; a developer wants
the working tree. See [Development](#development) below, which builds from source and never fetches
`senselab` from PyPI.

---

## Development

Three steps, and the second is the one people miss:

Development installs **from source** — the checkout you are standing in. `uv sync` puts the working
tree in the environment, so an edit is live with no reinstall; nothing here fetches `senselab` from
PyPI, and `--pre` has no role. (The version you will see, `1.3.1aN.devM`, comes from `hatch-vcs`
reading git describe, which is also why a shallow clone with no tags reports a wrong version.)

```bash
# 1. Environment. --all-extras is what every CI workflow uses: it cannot go stale when an
#    extra is added, which `--extra all` can. The interpreter comes from .python-version
#    (3.12, matching CI's default), so no --python flag.
uv sync --all-extras --group dev --group docs

# 2. FFmpeg shared libraries for torchcodec. Skip this and NOTHING collects —
#    conftest.py aborts with "Dependencies failed to import", even for tests that
#    never open an audio file. See System Requirements above for why the PyAV wheel
#    does not cover it.
bash scripts/install-ffmpeg.sh
export LD_LIBRARY_PATH="/opt/miniforge/lib:$LD_LIBRARY_PATH"   # macOS: DYLD_LIBRARY_PATH

# 3. Hooks, required before committing.
uv run pre-commit install
```

Then:

```bash
uv run pytest                                    # everything, with coverage
uv run pytest src/tests/audio/tasks/preprocessing_test.py          # one file
uv run pytest src/tests/audio/tasks/preprocessing_test.py::test_x  # one test
uv run mypy .
uv run ruff check          # --fix to autofix
uv run ruff format
uv run codespell
```

**On `pytest -n auto`.** It is tempting and it is a memory hazard: `pytest-xdist` gives each worker its
own interpreter, and each one imports torch + transformers + speechbrain independently — measured at
**535 MB resident per worker before a single test runs**, plus a private copy of any model weights that
worker's tests load. On a 10-core / 32 GB laptop `-n auto` has exhausted memory. Prefer running the
directory you changed, or cap the workers (`-n 4`). The pure-Python workflow tests are fast serially:
`uv run pytest src/tests/audio/workflows/audio_analysis` is ~1400 tests in about 17 s.

Docs build locally with:

```bash
uv run pdoc src/senselab -t docs_style/pdoc-theme --docformat google
```

---

## senselab AI (our AI-based chatbot)

#### Development (with uv)

```bash
uv sync --extra senselab-ai
uv run senselab-ai
```

#### Production (with pip)

```bash
pip install 'senselab[senselab-ai]'
senselab-ai
```

Once started, you can open the provided JupyterLab interface, setup the agent and chat with it, and let it create and execute code for you.

![Example of how senselab-ai works](<tutorials/senselab-ai/resources/Screenshot 2025-09-02 at 8.52.31 PM.png>)

For a walkthrough, see: [`tutorials/senselab-ai/senselab_ai_intro.ipynb`](tutorials/senselab-ai/senselab_ai_intro.ipynb).


---

## Contributing
<ins>We welcome contributions from the community!</ins> Before proceeding with that, please review our [**CONTRIBUTING.md**](https://github.com/sensein/senselab/blob/main/CONTRIBUTING.md).

---

## Funding
`senselab` is mostly supported by the following organizations and initiatives:
- McGovern Institute ICON Fellowship
- NIH Bridge2AI Precision Public Health (OT2OD032720)
- Child Mind Institute
- ReadNet Project
- Chris and Lann Woehrle Psychiatric Fund

---

## Acknowledgments

`senselab` builds on the work of many open-source projects. We gratefully acknowledge the developers and maintainers of the following key dependencies:

* [PyTorch](https://github.com/pytorch/pytorch), [Torchvision](https://github.com/pytorch/vision), [Torchaudio](https://github.com/pytorch/audio)
_deep learning framework and audio/vision extensions_
* [Transformers](https://github.com/huggingface/transformers), [Datasets](https://github.com/huggingface/datasets), [Accelerate](https://github.com/huggingface/accelerate), [Huggingface Hub](https://github.com/huggingface/huggingface_hub)
_training and inference utilities plus (pre-)trained models and datasets_
* [Scikit-learn](https://github.com/scikit-learn/scikit-learn), [UMAP-learn](https://github.com/lmcinnes/umap)
_machine learning utilities_
* [Matplotlib](https://github.com/matplotlib/matplotlib)
_visualization toolkit_
* [Praat-Parselmouth](https://github.com/YannickJadoul/Parselmouth), [OpenSMILE](https://github.com/audeering/opensmile), [SpeechBrain](https://github.com/speechbrain/speechbrain), [SPARC](speech-articulatory-coding), [Pyannote-audio](https://github.com/pyannote/pyannote-audio), [Coqui-TTS](https://github.com/idiap/coqui-ai-TTS), [NVIDIA NeMo](https://github.com/NVIDIA/NeMo), [Vocos](https://github.com/gemelo-ai/vocos), [Audiomentations](https://github.com/iver56/audiomentations), [Torch-audiomentations](https://github.com/asteroid-team/torch-audiomentations)
_speech and audio processing tools_
* [NLTK](https://github.com/nltk/nltk), [Sentence-Transformers](https://github.com/UKPLab/sentence-transformers), [Pylangacq](https://github.com/jacksonllee/pylangacq), [Jiwer](https://github.com/jitsi/jiwer)
_text and language processing tools_
* [OpenCV](https://github.com/opencv/opencv-python), [Ultralytics](https://github.com/ultralytics/ultralytics), [mediapipe](https://github.com/google-ai-edge/mediapipe), [Python-ffmpeg](https://github.com/jonghwanhyeon/python-ffmpeg), [AV](https://github.com/PyAV-Org/PyAV)
_computer vision and pose estimation_
* [Pydantic](https://github.com/pydantic/pydantic), [Iso639](https://github.com/janpipek/iso639-python), [PyCountry](https://github.com/pycountry/pycountry), [Nest-asyncio](https://github.com/erdewit/nest_asyncio)
_validation, and utilities_
* [Ipywidgets](https://github.com/jupyter-widgets/ipywidgets), [IpKernel](https://github.com/ipython/ipykernel), [Nbformat](https://github.com/jupyter/nbformat), [Nbss-upload](https://github.com/notebook-sharing-space/nbss-upload), [Notebook-intelligence](https://github.com/notebook-intelligence/notebook-intelligence)
_Jupyter and notebook-related tools_

We are thankful to the open-source community for enabling this project! 🙏
