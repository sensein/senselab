# Quickstart: Expand Speech Representation Model Coverage

## Test S3PRL embeddings
```bash
uv run python -c "
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.ssl_embeddings import extract_ssl_embeddings
audio = Audio(filepath='tutorial_audio_files/audio_48khz_mono_16bits.wav')
embeddings = extract_ssl_embeddings([audio], model='apc')  # S3PRL model
print(f'Shape: {embeddings[0].shape}')
"
```

## Test NeMo ASR
```bash
uv run python -c "
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.speech_to_text.api import transcribe_audios
from senselab.utils.data_structures import NeMoModel
audio = Audio(filepath='tutorial_audio_files/audio_48khz_mono_16bits.wav')
result = transcribe_audios([audio], model=NeMoModel(path_or_uri='nvidia/stt_en_conformer_ctc_large'))
print(result[0].text)
"
```

## Build model registry
```bash
uv run python scripts/generate_model_registry.py > docs/model_registry.md
```
