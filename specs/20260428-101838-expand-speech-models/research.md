# Research: Expand Speech Representation Model Coverage

**Date**: 2026-04-28

## R1: S3PRL Integration Approach

**Decision**: Add S3PRL as a subprocess venv backend for SSL embedding extraction. The S3PRL toolkit pins specific torch/torchaudio versions and has a large dependency tree that conflicts with the main environment.

**Rationale**: S3PRL provides 30+ pre-trained SSL models through a uniform `hub` API (`s3prl.hub`). The paper's encoder.py shows the pattern: `model = getattr(hub, model_name)()` → `model(wavs)` returns hidden states. This is a clean API that maps well to a subprocess worker script.

**S3PRL models to support** (from the paper + s3prl hub):
- APC, TERA, MockingJay, DeCoAR2, CPC (from paper)
- NPC, VQ-APC, Audio ALBERT (additional popular models)
- Filterbanks, MFCCs, mel (feature extractors, not SSL but useful baselines)

**Venv requirements**: `s3prl`, `torch>=2.0`, `torchaudio`, `numpy`, `soundfile`

**Alternatives considered**:
- Direct dependency: Too many conflicts (pins torch versions)
- Docker container: Overkill for model inference
- Skip S3PRL, use only HF equivalents: Many S3PRL models (APC, TERA, CPC) have no HF equivalent

## R2: SpeechBrain Speaker Encoder Access

**Decision**: Expose SpeechBrain's `EncoderClassifier.encode_batch()` for embedding extraction (not just verification). This is already partially done — the speaker_embeddings module uses it. The fix is to also expose it in ssl_embeddings with a clear "SpeechBrain speaker encoders" option.

**Rationale**: The paper uses 3 SpeechBrain models (ECAPA-TDNN, ResNet, x-vector) via `EncoderClassifier.from_hparams()` → `encode_batch()`. Senselab already has this in `speaker_embeddings/` but not unified with the SSL embeddings module.

**Action**: Unify the embedding extraction API so users can request embeddings from any backend (HF, S3PRL, SpeechBrain) through a single entry point, or keep them as separate but documented paths.

## R3: NeMo ASR via Subprocess Venv

**Decision**: Add NeMo ASR as a subprocess venv option in the speech_to_text module. NeMo has Conformer-CTC and Conformer-RNNT models that are state-of-the-art for English ASR.

**Rationale**: NeMo already has a subprocess venv pattern established for diarization. ASR can reuse the same venv with minimal additions.

**NeMo ASR models**: `nvidia/stt_en_conformer_ctc_large`, `nvidia/stt_en_conformer_transducer_large`

## R4: Pyannote Dedicated VAD

**Decision**: Expose `pyannote/voice-activity-detection` as a first-class option separate from the diarization-based VAD. The current VAD is just diarization with all segments merged.

**Rationale**: Pyannote's dedicated VAD model is faster and more accurate for voice detection than using full diarization. It's also a prerequisite for other tasks (trimming silence, detecting speech regions for processing).

## R5: Model Registry Documentation

**Decision**: Create a `docs/model_registry.md` (or `MODEL_REGISTRY.md` in repo root) that is a structured table of all supported models, generated from a YAML data source. This becomes part of the pdoc-generated docs.

**Format per model**:
```yaml
- name: ECAPA-TDNN
  task: speaker_embeddings
  source: speechbrain
  model_id: speechbrain/spkrec-ecapa-voxceleb
  embedding_dim: 192
  parameters: 7.3M
  training_data: VoxCeleb
  recommended_for: Speaker verification, speaker identification
```

## R6: Speaker Identity Coding Paper Reproducibility

**Decision**: Create a tutorial notebook that demonstrates extracting embeddings from the paper's 17 models using senselab, then performs basic speaker verification benchmarking. The paper's custom train/benchmark scripts are NOT reproduced — only the embedding extraction and evaluation steps.

**Models accessible through senselab** (15 of 17):
- HuggingFace (5/5): wav2vec2, W2V-BERT, HuBERT, WavLM, data2vec — all work via existing ssl_embeddings
- SpeechBrain (3/3): ECAPA-TDNN, ResNet, x-vector — work via speaker_embeddings
- S3PRL (5/7): APC, TERA, MockingJay, DeCoAR2, CPC — need new subprocess venv; filterbanks/MFCCs/mel are baseline features
- Not accessible (2): Hybrid BYOL-S (custom SERAB package), cochlear model (custom implementation)
