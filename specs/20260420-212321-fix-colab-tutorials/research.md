# Research: Fix All Tutorials to Run on Google Colab

## 1. Notebook Execution in CI

**Decision**: Use papermill for CI notebook execution
**Rationale**: papermill executes notebooks cell-by-cell, captures outputs, reports errors with cell numbers, supports parameterization (reduce epochs for CI), and has timeout support. Already used by major ML projects (StellarGraph, Hugging Face).
**Alternatives considered**:
- nbclient: lower-level, less CI-friendly output
- nbconvert --execute: works but less control over timeouts/parameters
- treon: simpler but less maintained

## 2. Colab Setup Cell Pattern

**Decision**: Bootstrap uv via pip, then use uv for all package installation (Constitution I compliant)
**Rationale**: Colab doesn't have uv pre-installed, but `pip install uv` is a one-time bootstrap. After that, all Python package management goes through uv per Constitution I. FFmpeg needs conda-forge (via miniforge) for torchcodec.
**Pattern**:
```python
# Bootstrap uv (Colab doesn't have it), then install senselab via uv
!pip install -q uv
!uv pip install --pre --system "senselab[nlp,text,video]"
# Install FFmpeg shared libraries for torchcodec
!bash scripts/install-ffmpeg.sh
```

## 3. GPU Detection in Notebooks

**Decision**: Use torch.cuda.is_available() with informative skip messages
**Rationale**: Standard pattern. Colab GPU runtimes have CUDA; CPU runtimes don't. No need for complex detection.
**Pattern**:
```python
import torch
GPU_AVAILABLE = torch.cuda.is_available()
if not GPU_AVAILABLE:
    print("⚠️ GPU not available. Some cells will be skipped. Enable GPU: Runtime > Change runtime type > T4 GPU")
```

## 4. HF_TOKEN Handling in Notebooks

**Decision**: Use Colab's userdata secrets for HF_TOKEN
**Rationale**: Colab has a built-in secrets manager (google.colab.userdata). More secure than env vars or inline tokens.
**Pattern**:
```python
import os
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except (ImportError, Exception):
    pass  # Not on Colab or no secret set
if not os.environ.get("HF_TOKEN"):
    print("⚠️ HF_TOKEN not set. Some models may not load. Set it in Colab: Secrets (🔑) > Add HF_TOKEN")
```

## 5. Tutorial Classification (CPU vs GPU)

**Decision**: Classify each notebook by hardware requirement based on the models it uses

| Tutorial | Requires GPU | Requires HF_TOKEN | Notes |
|----------|-------------|-------------------|-------|
| 00_getting_started | No | No | Basic intro |
| audio_data_augmentation | No | No | CPU augmentation |
| conversational_data_exploration | Yes | Yes | Pyannote (gated model) |
| extract_speaker_embeddings | Yes | No | SpeechBrain |
| features_extraction | No* | No | *PPGs need GPU, rest CPU |
| forced_alignment | No | No | Wav2Vec2 works on CPU |
| speaker_diarization | Yes | Yes | Pyannote (gated model) |
| speaker_verification | Yes | No | SpeechBrain |
| speech_emotion_recognition | Yes | No | Large wav2vec2 models |
| speech_enhancement | Yes | No | SpeechBrain |
| speech_to_text | No | No | Whisper-tiny works on CPU |
| text_to_speech | Yes | No | Bark, XTTS need GPU |
| voice_activity_detection | Yes | Yes | Pyannote (gated model) |
| voice_cloning | Yes | No | Coqui subprocess venv |
| pose_estimation | No | No | YOLO CPU works |
| dimensionality_reduction | No | No | sklearn/umap |
| senselab_ai_intro | No | No | Jupyter widgets |

## 6. Colab Badge Format

**Decision**: Standard GitHub badge linking to Colab
**Pattern**:
```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sensein/senselab/blob/alpha/tutorials/audio/00_getting_started.ipynb)
```
Note: Links target the `alpha` branch since tutorials install pre-release.
