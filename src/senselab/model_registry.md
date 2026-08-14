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

| Model | Source | Model ID | Embedding Dim | Parameters | License | Speakers | Text | Recommended For |
|-------|--------|----------|---------------|------------|---------|---|---|-----------------|
| Pyannote Diarization | pyannote | `pyannote/speaker-diarization-community-1` | — | N/A | — | — | no | Multi-speaker diarization (requires HF token). Seed-17 probe: the only backend that counts speakers reliably at all — 100% exact-count accuracy at k=1, 85% at k=2 — but falls to <=45% for k>=3; treat counts above 2 as unverified on this evidence. |
| NeMo Sortformer | nemo | `nvidia/diar_sortformer_4spk-v1` | — | N/A | — | 4 | no | 4-speaker diarization (via subprocess venv). Structurally capped at 4 (confirmed, seed-17 probe: 20/20 k=8 sessions predicted exactly 4). Exact-count accuracy peaks at k=2..4 (65-80%) but is 0% at k=1 and at k>=5, where predictions clamp to 4 regardless of truth. |
| VibeVoice-ASR-HF | microsoft | `microsoft/VibeVoice-ASR-HF` | — | 7B | — | — | yes | Unified ASR + diarization (in-process, transformers>=5.3, CUDA recommended). Seed-17 probe: no structural ceiling observed (predicted up to 16, plus refusals, at k=8); exact-count accuracy is uneven (95% at k=2, down to 20% by k=8) — best where a rough count suffices. |
| USC-SAIL Child-Adult Classifier | usc-sail | `AlexXu811/whisper-child-adult` | — | Whisper-base + LoRA | — | 2 | no | Child/adult/overlap speaker-role labeling, not identity (via subprocess venv, CUDA only). Structurally capped at 2 (confirmed, seed-17 probe: 20/20 k=8 sessions counted exactly 2). Exact-count accuracy is 70% at k=2, 50% at k=1, and necessarily 0% for k>=3 since it cannot emit more than 2 speakers. |
| MOSS-Transcribe-Diarize | OpenMOSS-Team | `OpenMOSS-Team/MOSS-Transcribe-Diarize` | — | 0.9B | — | — | yes | Unified ASR + diarization, lightweight and CPU-plausible (via subprocess venv, transformers>=5.6). Seed-17 probe: no structural ceiling observed (predicted up to 12 at k=8), but exact-count accuracy is inconsistent (0% at k=1, 25-65% elsewhere) — do not rely on its speaker count without independent verification. |
| DiariZen | BUT-FIT | `BUT-FIT/diarizen-wavlm-large-s80-md` | — | WavLM-large + Conformer | CC BY-NC 4.0 — non-commercial only | — | no | Diarization only, no transcription (via subprocess venv installing DiariZen's forked pyannote-audio). Seed-17 probe: no structural ceiling observed (predicted up to 8 at k=8); best exact-count accuracy at k=2-3 (75-90%), degrading beyond. |

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
| YAMNet | tensorflow | `google/yamnet` | — | 3.2M | Lightweight audio scene classification (TensorFlow-based; runs via subprocess venv) |

## Speech Enhancement

| Model | Source | Model ID | Embedding Dim | Parameters | License | Recommended For |
|-------|--------|----------|---------------|------------|---------|-----------------|
| SepFormer (16kHz) | speechbrain | `speechbrain/sepformer-wham16k-enhancement` | — | N/A | — | Speech enhancement at 16kHz |
| SepFormer (8kHz) | speechbrain | `speechbrain/sepformer-whamr-enhancement` | — | N/A | — | Speech enhancement at 8kHz (with reverb) |
| DriftSE | LiangXu123 (unpackaged upstream repo, via subprocess venv) | `sensein/driftse-distilhubert-three-layers` | — | N/A | Unknown — unresolved upstream. The source repository https://github.com/LiangXu123/DriftSE carries no LICENSE file and no README licence statement, so no terms have been granted for these weights. A licence request was opened 2026-08-08 and is unanswered: https://github.com/LiangXu123/DriftSE/issues/2. The mirror is public so the backend is usable during the alpha, which is a deliberate decision taken with that unknown status visible rather than resolved: treat the weights as all-rights-reserved by default, and consult upstream before any use that depends on licence terms. | One-step (1 NFE) generative enhancement; reachable only via an explicit HFModel id, not in any default model list and not wired into audio_analysis (via subprocess venv, pinned upstream commit 695a64db187500fa0d7bae23912680bd5d4df613) |

## Separation

| Model | Source | Model ID | Embedding Dim | Parameters | License | Recommended For |
|-------|--------|----------|---------------|------------|---------|-----------------|
| unasdiff | RunwuShi (unpackaged upstream repo, via subprocess venv, Python 3.10, torch==2.6.0) | `sensein/unasdiff-diffusion-priors` | — | N/A | Unknown — unresolved upstream. The source repository https://github.com/RunwuShi/unasdiff carries no LICENSE file and no README licence statement, so no terms have been granted for these weights. A licence request was opened 2026-08-08 and is unanswered: https://github.com/RunwuShi/unasdiff/issues/1. The mirror is public so the backend is usable during the alpha, which is a deliberate decision taken with that unknown status visible rather than resolved: reachability grants no rights. Treat the weights as all-rights-reserved by default, and consult upstream before any use that depends on licence terms. SENSELAB_UNASDIFF_CHECKPOINTS remains for a caller supplying their own checkpoints. | Unsupervised speech/sound source separation via two independently trained diffusion priors (speech_sound, sound_sound, speech_speech modes); reachable only via an explicit HFModel id, not in any default model list and not wired into audio_analysis (via subprocess venv, pinned upstream commit 5a5d70cdc94fe9d034892a1c5bc68ad1a67d2daa) |

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

## Text To Speech

| Model | Source | Model ID | Embedding Dim | Parameters | Recommended For |
|-------|--------|----------|---------------|------------|-----------------|
| Qwen3-TTS (CustomVoice) | Alibaba Qwen (qwen-tts PyPI package, via subprocess venv) | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | — | 1.7B | Named-speaker (9 built-in identities, no reference-audio cloning required) speech synthesis -- the speech source for the speaker-ceiling probe, which needs N distinct identities with exact ground truth; reachable only via an explicit HFModel id starting with Qwen/Qwen3-TTS, not in any default model list (via subprocess venv, pinned commit 0c0e3051f131929182e2c023b9537f8b1c68adfe, licence apache-2.0) |
