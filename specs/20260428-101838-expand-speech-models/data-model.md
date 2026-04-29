# Data Model: Expand Speech Representation Model Coverage

**Date**: 2026-04-28

## New Entities

### S3PRLModel (new model type)
- Extends SenselabModel pattern
- `path_or_uri`: S3PRL model name (e.g., "apc", "tera", "cpc")
- Runs in isolated subprocess venv
- Returns embeddings as List[torch.Tensor]

### ModelRegistryEntry (documentation entity)
- `name`: Human-readable model name
- `task`: Which senselab task it applies to
- `source`: Backend (huggingface, speechbrain, s3prl, nemo, pyannote)
- `model_id`: Identifier used to load the model
- `embedding_dim`: Output dimension
- `parameters`: Parameter count
- `training_data`: What it was trained on
- `recommended_for`: Use case guidance

## Modified Entities

### ssl_embeddings module
- Currently: HuggingFace-only
- After: HuggingFace + S3PRL + SpeechBrain backends
- API signature stays the same; backend selected by model type

### speech_to_text module
- Currently: HuggingFace-only
- After: HuggingFace + NeMo (subprocess venv)

### voice_activity_detection module
- Currently: Reuses diarization output
- After: Also supports dedicated Pyannote VAD pipeline
