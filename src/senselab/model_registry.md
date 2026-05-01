# Senselab Model Registry

All models supported by senselab, organized by task.

## Speaker Embeddings

| Model | Source | Model ID | Embedding Dim | Parameters | Recommended For |
|-------|--------|----------|---------------|------------|-----------------|
| ECAPA-TDNN | speechbrain | `speechbrain/spkrec-ecapa-voxceleb` | 192 | 7.3M | Speaker verification, identification, embedding extraction |
| ResNet TDNN | speechbrain | `speechbrain/spkrec-resnet-voxceleb` | 192 | 7.3M | Speaker verification (alternative to ECAPA-TDNN) |
| X-Vector | speechbrain | `speechbrain/spkrec-xvect-voxceleb` | 192 | 7.3M | Speaker verification (classic approach) |

## Ssl Embeddings

| Model | Source | Model ID | Embedding Dim | Parameters | Recommended For |
|-------|--------|----------|---------------|------------|-----------------|
| wav2vec2-base | huggingface | `facebook/wav2vec2-base` | 768 | 95M | General-purpose speech representations |
| wav2vec2-large | huggingface | `facebook/wav2vec2-large-lv60` | 1024 | 315M | High-quality speech representations |
| HuBERT-large | huggingface | `facebook/hubert-large-ll60k` | 1024 | 315M | Speech representations with clustering-based pre-training |
| WavLM-large | huggingface | `microsoft/wavlm-large` | 1024 | 315M | Speaker verification, separation, and general speech |
| data2vec-audio-large | huggingface | `facebook/data2vec-audio-large` | 1024 | 313M | Multi-modal pre-training approach for speech |
| W2V-BERT 2.0 | huggingface | `facebook/w2v-bert-2.0` | 1024 | 600M | Multilingual speech representations |
| APC | s3prl | `apc` | 512 | 4.1M | Autoregressive predictive coding, lightweight SSL |
| TERA | s3prl | `tera` | 768 | 21.3M | Time-frequency representation learning |
| MockingJay | s3prl | `mockingjay` | 768 | 85.1M | Masked reconstruction pre-training |
| DeCoAR 2.0 | s3prl | `decoar2` | 768 | 89.8M | Deep contextualized acoustic representations |
| CPC | s3prl | `modified_cpc` | 256 | 1.8M | Contrastive predictive coding, smallest SSL model |

## Speech To Text

| Model | Source | Model ID | Embedding Dim | Parameters | Recommended For |
|-------|--------|----------|---------------|------------|-----------------|
| Whisper Tiny | huggingface | `openai/whisper-tiny` | — | 39M | Fast, resource-constrained ASR |
| Whisper Small | huggingface | `openai/whisper-small` | — | 244M | Balanced speed/accuracy ASR |
| Whisper Large v3 Turbo | huggingface | `openai/whisper-large-v3-turbo` | — | 809M | Best accuracy, multilingual ASR |
| NeMo Conformer CTC | nemo | `nvidia/stt_en_conformer_ctc_large` | — | 120M | High-accuracy English ASR (via subprocess venv) |

## Speaker Diarization

| Model | Source | Model ID | Embedding Dim | Parameters | Recommended For |
|-------|--------|----------|---------------|------------|-----------------|
| Pyannote Diarization | pyannote | `pyannote/speaker-diarization-community-1` | — | N/A | Multi-speaker diarization (requires HF token) |
| NeMo Sortformer | nemo | `nvidia/diar_sortformer_4spk-v1` | — | N/A | 4-speaker diarization (via subprocess venv) |

## Speech Emotion Recognition

| Model | Source | Model ID | Embedding Dim | Parameters | Recommended For |
|-------|--------|----------|---------------|------------|-----------------|
| SUPERB SER (IEMOCAP) | huggingface | `superb/wav2vec2-base-superb-er` | — | 95M | Conversational speech emotion (4 classes) |
| SpeechBrain SER (IEMOCAP) | speechbrain | `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` | — | 95M | Conversational speech emotion (4 classes, very confident) |
| XLSR SER (RAVDESS) | huggingface | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | — | 315M | Acted speech emotion (8 classes) |
| Continuous SER (MSP-Podcast) | huggingface | `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` | — | 315M | Dimensional emotion (valence/arousal/dominance) |

## Audio Scene Classification

| Model | Source | Model ID | Embedding Dim | Parameters | Recommended For |
|-------|--------|----------|---------------|------------|-----------------|
| Audio Spectrogram Transformer (AST) | huggingface | `MIT/ast-finetuned-audioset-10-10-0.4593` | — | 87M | General-purpose auditory scene analysis, sound event detection |
| YAMNet | tensorflow | `google/yamnet` | — | 3.2M | Lightweight audio scene classification (TensorFlow-based; not directly supported via classify_audios) |

## Speech Enhancement

| Model | Source | Model ID | Embedding Dim | Parameters | Recommended For |
|-------|--------|----------|---------------|------------|-----------------|
| SepFormer (16kHz) | speechbrain | `speechbrain/sepformer-wham16k-enhancement` | — | N/A | Speech enhancement at 16kHz |
| SepFormer (8kHz) | speechbrain | `speechbrain/sepformer-whamr-enhancement` | — | N/A | Speech enhancement at 8kHz (with reverb) |

## Voice Activity Detection

| Model | Source | Model ID | Embedding Dim | Parameters | Recommended For |
|-------|--------|----------|---------------|------------|-----------------|
| Pyannote VAD | pyannote | `pyannote/voice-activity-detection` | — | N/A | Dedicated voice activity detection (requires HF token) |

## Features Extraction

| Model | Source | Model ID | Embedding Dim | Parameters | Recommended For |
|-------|--------|----------|---------------|------------|-----------------|
| SPARC (Articulatory) | sparc | `speech-articulatory-coding` | 14 (EMA) + pitch/loudness/periodicity | N/A | Articulatory features, voice conversion, resynthesis |
| PPG (Phonetic Posteriorgrams) | ppgs | `ppgs` | 40 (phonemes) | N/A | Phoneme-level analysis, duration extraction |
| OpenSMILE | opensmile | `eGeMAPSv02` | 88 (functionals) | N/A | Standard acoustic features for emotion, health assessment |
