Self-supervised learning (SSL) embedding extraction from audio.

This module extracts embeddings from pre-trained self-supervised models across three backends:

- **HuggingFace Transformers**: wav2vec2, HuBERT, WavLM, and other models hosted on the HuggingFace Hub.
- **SpeechBrain**: ECAPA-TDNN, x-vector, ResNet speaker encoder models via SpeechBrain's EncoderClassifier.
- **S3PRL** (isolated subprocess venv): APC, TERA, CPC, and other upstream models from the S3PRL toolkit.

These embeddings capture rich acoustic representations learned from large-scale unlabeled audio data and are useful as features for downstream tasks like emotion recognition, speaker identification, and health assessment.

## Supported Models

| Model | Backend | Embedding Dim | Parameters | Training Data | Use Case |
|-------|---------|---------------|------------|---------------|----------|
| wav2vec2-base | HuggingFace | 768 | 95M | LibriSpeech | General SSL features |
| HuBERT-large | HuggingFace | 1024 | 315M | LibriLight | General SSL features |
| WavLM-large | HuggingFace | 1024 | 315M | LibriLight | General SSL features |
| Whisper (encoder only) | HuggingFace | 1280 (large-v3) | 635M encoder | Weakly supervised ASR | Speech encoder features |
| ECAPA-TDNN | SpeechBrain | 192 | 7.3M | VoxCeleb | Speaker identification |
| APC | S3PRL | 512 | 4.1M | LibriSpeech | Generative SSL |
| TERA | S3PRL | 768 | 21M | LibriSpeech | Generative SSL |
| CPC | S3PRL | 256 | 1.8M | LibriSpeech | Contrastive SSL |

## Usage

```python
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.ssl_embeddings import extract_ssl_embeddings_from_audios
from senselab.utils.data_structures import HFModel, SpeechBrainModel

audio = Audio(filepath="sample.wav")

# HuggingFace: returns [num_layers, time_frames, embedding_dim]
hf_emb = extract_ssl_embeddings_from_audios([audio], HFModel(path_or_uri="facebook/wav2vec2-base"))

# SpeechBrain: returns [embedding_dim] (fixed-dimensional)
sb_emb = extract_ssl_embeddings_from_audios(
    [audio], SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb")
)

# S3PRL: returns [time_frames, embedding_dim] (last hidden state)
s3_emb = extract_ssl_embeddings_from_audios([audio], "apc")
```

## Backend Notes

**Whisper-style encoder-decoder models** (feature extractor returns `input_features`): only the encoder runs. Audio is cut into consecutive 30 s windows, each encoded on its own, and the states are concatenated in time, so the result keeps the `[num_layers, time_frames, embedding_dim]` shape (20 ms frames; window k fills frames `[1500k, 1500(k+1))`). The `revision` must be a 40-hex commit:

```python
whisper = HFModel(path_or_uri="openai/whisper-large-v3", revision="06f233fe06e710322aca913c1bc4249a0d71fce1")
w_emb = extract_ssl_embeddings_from_audios([audio], whisper)
```

**S3PRL** runs in an isolated subprocess venv because it requires torchaudio<2.5 (it uses the removed `torchaudio.set_audio_backend()` API). The venv is automatically provisioned on first use via `uv`.
