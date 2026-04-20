# Test Model Size & Device Tier Audit

## Tier Classification

- **Tier 1 (CPU-fast)**: <300MB download, inference <30s on CPU. Always run on all available devices.
- **Tier 2 (CPU-feasible)**: 300MB-1GB, inference <2min on CPU. Run on all available devices with extended timeout.
- **Tier 3 (GPU-preferred)**: >1GB or very slow on CPU. Skip on CPU, only run on MPS/CUDA.
- **Isolated**: Runs in subprocess venv. Device handled internally.

## Model Inventory

| Model | Download | Params | Test File(s) | Tier |
|-------|----------|--------|-------------|------|
| `speechbrain/metricgan-plus-voicebank` | ~5MB | small | speech_enhancement_test | 1 |
| `speechbrain/spkrec-xvect-voxceleb` | ~30MB | small | speaker_embeddings_test | 1 |
| `pyannote/speaker-diarization-community-1` | ~20MB | pipeline | speaker_diarization_test, vad_test | 2 |
| `sentence-transformers/all-MiniLM-L6-v2` | ~80MB | 22M | embeddings_extraction_test | 1 |
| `speechbrain/spkrec-resnet-voxceleb` | ~80MB | ~14M | speaker_embeddings_test | 1 |
| `speechbrain/spkrec-ecapa-voxceleb` | ~89MB | ~14M | speaker_embeddings_test | 1 |
| `speechbrain/sepformer-wham16k-enhancement` | ~113MB | 26M | speech_enhancement_test | 1 |
| `facebook/mms-tts-eng` | ~145MB | VITS | text_to_speech_test | 1 |
| `openai/whisper-tiny` | ~150MB | 39M | speech_to_text_test, forced_alignment_test | 1 |
| `facebook/wav2vec2-base-960h` | ~360MB | 95M | forced_alignment_test | 1 |
| `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | ~661MB | ~150M | classification_test | 2 |
| `facebook/seamless-m4t-unity-small` | ~783MB | 281M | speech_to_text_test | 2 |
| `nvidia/diar_sortformer_4spk-v1` | ~988MB | large | speaker_diarization_test | skip (too slow) |
| `knnvc` (Coqui) | ~1GB est | | voice_cloning_test | isolated |
| `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | ~1.27GB | 315M | classification_test | 3 |
| `suno/bark-small` | ~1.69GB | 3x80M | text_to_speech_test | 3 |
| `coqui/XTTS-v2` | ~1.87GB | large | text_to_speech_test | isolated |
| `ppgs` | varies | | features_extraction_test | isolated/GPU |

## Device Parameterization Design

### Available devices detection (conftest.py)

```python
import pytest
import torch

def get_available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices

AVAILABLE_DEVICES = get_available_devices()
GPU_DEVICES = [d for d in AVAILABLE_DEVICES if d != "cpu"]

# Fixtures
@pytest.fixture(params=AVAILABLE_DEVICES, ids=lambda d: f"device={d}")
def any_device(request):
    """Parameterize over all available devices (cpu, mps, cuda)."""
    return request.param

@pytest.fixture(params=GPU_DEVICES or pytest.param("skip", marks=pytest.mark.skip("No GPU")), 
                ids=lambda d: f"device={d}")
def gpu_device(request):
    """Parameterize over GPU-only devices (mps, cuda). Skip if none available."""
    return request.param
```

### Test patterns by tier

**Tier 1** — uses `any_device` fixture:
```python
def test_extract_speaker_embeddings(resampled_audio, ecapa_model, any_device):
    result = extract_speaker_embeddings_from_audios([resampled_audio], model=ecapa_model, device=any_device)
    assert len(result) == 1
```

**Tier 2** — uses `any_device` with timeout marker:
```python
@pytest.mark.timeout(120)
def test_diarize_audios(audio, pyannote_model, any_device):
    result = diarize_audios([audio], model=pyannote_model, device=any_device)
    ...
```

**Tier 3** — uses `gpu_device`:
```python
def test_bark_synthesis(gpu_device):
    result = synthesize_texts(["Hello"], model=HFModel(path_or_uri="suno/bark-small"), device=gpu_device)
    ...
```

### CI matrix coverage

| Environment | Tier 1 | Tier 2 | Tier 3 | Isolated |
|------------|--------|--------|--------|----------|
| pre-commit (ubuntu, no GPU) | skip (no models) | skip | skip | skip |
| macOS GHA (macos-test label) | cpu | cpu | skip | skip |
| EC2 GPU (ec2-gpu-test label) | cpu+cuda | cpu+cuda | cuda | cuda |

### Notes

- MPS (Apple Metal) is available on macOS with Apple Silicon. macOS GHA runners are M-series so MPS should work.
- The `device` parameter maps to `DeviceType` enum in senselab. Tests should convert string to DeviceType.
- Isolated backends (coqui, ppgs) manage their own device selection inside the subprocess venv.
- The nvidia sortformer test is already marked skip ("too slow") — leave as-is.
