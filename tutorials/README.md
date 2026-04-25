# Senselab Tutorials

Interactive Jupyter notebooks demonstrating senselab's capabilities.
Each notebook can be opened directly in Google Colab.

## Setup Cell Template

Every tutorial notebook should have this as its first code cell:

```python
# Install senselab (latest pre-release) and dependencies
!pip install -q uv
!uv pip install --pre --system "senselab[nlp,text,video]"

# Install FFmpeg for audio/video decoding (torchcodec)
import subprocess, shutil
if not shutil.which("ffmpeg"):
    subprocess.run(["bash", "-c",
        "curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname -s)-$(uname -m).sh -o /tmp/mf.sh && "
        "bash /tmp/mf.sh -b -p /opt/miniforge && "
        "/opt/miniforge/bin/conda install -y -p /opt/miniforge 'ffmpeg<8' && "
        "ln -sf /opt/miniforge/bin/ffmpeg /usr/local/bin/ffmpeg"
    ], check=True)

# For notebooks using pyannote models (diarization, VAD):
# Set your HuggingFace token in Colab Secrets (🔑 icon in sidebar)
import os
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
except Exception:
    pass
if not os.environ.get("HF_TOKEN"):
    print("⚠️ HF_TOKEN not set. Models requiring authentication may fail.")
    print("   Set it in Colab: click 🔑 in the sidebar > Add HF_TOKEN")
```

## Colab Badge

Add this markdown cell at the very top of each notebook:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sensein/senselab/blob/alpha/tutorials/<path>)
```

## GPU Detection

For notebooks that benefit from GPU acceleration:

```python
import torch
GPU_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
if not GPU_AVAILABLE:
    print("⚠️ Running on CPU. For faster execution, enable GPU:")
    print("   Runtime > Change runtime type > T4 GPU")
```

## Tutorials

### Audio
| Notebook | GPU | HF_TOKEN | Description |
|----------|-----|----------|-------------|
| 00_getting_started | Optional | Yes | Overview of all audio features |
| audio_data_augmentation | No | No | Data augmentation techniques |
| conversational_data_exploration | Yes | Yes* | Full conversation analysis pipeline |
| extract_speaker_embeddings | Optional | No | Speaker embedding extraction |
| features_extraction | No | No | OpenSMILE, parselmouth, torchaudio features |
| forced_alignment | No | No | Word/phoneme alignment |
| speaker_diarization | Optional | Yes* | Who spoke when |
| speaker_verification | Optional | No | Speaker identity verification |
| speech_emotion_recognition | Yes | No | Emotion classification |
| speech_enhancement | Optional | No | Noise reduction |
| speech_to_text | No | No | Transcription with Whisper |
| text_to_speech | Yes | No | Speech synthesis |
| voice_activity_detection | Optional | Yes* | Detect speech segments |
| voice_cloning | Yes | No | Voice conversion |
| audio_recording_and_acoustic_analysis | No | No | Recording and acoustic feature analysis |
| transcription_and_phonemic_analysis | Yes | No | Transcription and phoneme-level analysis |
| speech_representations_lab | Yes | No | Acoustic vs articulatory speech representations |

*Requires accepting pyannote model terms on HuggingFace

### Video
| Notebook | GPU | Description |
|----------|-----|-------------|
| pose_estimation | No | Body pose estimation with YOLO |

### Utils
| Notebook | GPU | Description |
|----------|-----|-------------|
| dimensionality_reduction | No | UMAP/t-SNE visualization |

### Senselab AI
| Notebook | GPU | Description |
|----------|-----|-------------|
| senselab_ai_intro | No | AI assistant for Jupyter |
