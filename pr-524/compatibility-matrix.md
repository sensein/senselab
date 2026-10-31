# Senselab Compatibility Matrix

Minimum and maximum tested versions of senselab dependencies across Python versions.
Lower bounds verified by pinning each package to its minimum in an isolated venv.
Upper bounds verified by running the full test suite (490+ tests passing).

## Python Support

| Python | Status | Notes |
|--------|--------|-------|
| 3.11 | Supported | Lowest supported; all lower bounds verified |
| 3.12 | Supported (default) | Google Colab default |
| 3.13 | Supported | praat-parselmouth needs >=0.4.5 |
| 3.14 | Supported | torch needs >=2.9; praat-parselmouth needs >=0.4.7 |

## Core Dependencies

| Package | pyproject.toml | Py 3.11 min | Py 3.12 min | Py 3.13 min | Py 3.14 min | Tested upper |
|---------|---------------|-------------|-------------|-------------|-------------|-------------|
| torch | >=2.8 | 2.8.0 | 2.8.0 | 2.8.0 | 2.9.0 | 2.11.0 |
| torchaudio | >=2.8 | 2.8.0 | 2.8.0 | 2.8.0 | 2.9.0 | 2.11.0 |
| torchvision | >=0.23 | 0.23.0 | 0.23.0 | 0.23.0 | 0.24.0 | 0.26.0 |
| torchcodec | >=0.7 | 0.7.0 | 0.7.0 | 0.7.0 | 0.9.0 | 0.11.1 |
| transformers | >=5.0 | 5.0.0 | 5.0.0 | 5.0.0 | 5.0.0 | 5.5.4 |
| huggingface-hub | >=1.3 | 1.3.0 | 1.3.0 | 1.3.0 | 1.3.0 | 1.11.0 |
| speechbrain | >=1.0 | 1.0.0 | 1.0.0 | 1.0.0 | 1.0.0 | 1.1.0 |
| pyannote-audio | >=4.0 | 4.0.0 | 4.0.0 | 4.0.0 | 4.0.0 | 4.0.4 |
| pydantic | >=2.11 | 2.11.0 | 2.11.0 | 2.11.0 | 2.11.0 | 2.13.3 |
| scikit-learn | >=1.7 | 1.7.0 | 1.7.0 | 1.7.0 | 1.7.0 | 1.8.0 |
| praat-parselmouth | >=0.4.3 | 0.4.3 | 0.4.3 | 0.4.5 | 0.4.7 | 0.4.7 |
| opensmile | >=2.6 | 2.6.0 | 2.6.0 | 2.6.0 | 2.6.0 | 2.6.0 |
| umap-learn | >=0.5.4 | 0.5.4 | 0.5.4 | 0.5.4 | 0.5.4 | 0.5.12 |
| audiomentations | >=0.42 | 0.42.0 | 0.42.0 | 0.42.0 | 0.42.0 | 0.43.1 |
| vocos | >=0.1 | 0.1.0 | 0.1.0 | 0.1.0 | 0.1.0 | 0.1.0 |

## Optional Dependencies (extras)

| Package | Extra | pyproject.toml | Tested |
|---------|-------|---------------|--------|
| sentence-transformers | text | >=5.1 | 5.4.1 |
| pylangacq | text | >=0.20 | 0.23.0 |
| jiwer | nlp | >=3.0 | 4.0.0 |
| nltk | nlp | >=3.9 | 3.9.4 |
| av | video | >=15 | 17.0.1 |
| opencv-python-headless | video | >=4.11 | 4.13.0 |
| ultralytics | video | >=8.3 | 8.4.40 |

## Isolated Backends (subprocess venvs)

These packages run in their own isolated virtual environments managed by uv.
They are not installed in the main senselab environment.

| Backend | Venv Python | torch | Key deps | Purpose |
|---------|-------------|-------|----------|---------|
| coqui-tts ~0.27 | 3.11 | >=2.8,<2.9 | transformers >=4.52,<5 | Voice cloning (knnvc) |
| ppgs >=0.0.9 | 3.11 | >=2.8,<2.9 | espnet, snorkel, lightning | Phonetic posteriorgrams |
| sparc >=0.1 | 3.11 | >=2.8,<2.9 | librosa, torchcrepe, penn | Articulatory features + voice cloning |
| continuous-ser | 3.12 | >=2.8 | transformers <5, hf-hub <1.0 | Continuous emotion (audeering model) |

## System Dependencies

| Dependency | Required by | Install method |
|-----------|-------------|----------------|
| FFmpeg shared libs (<=7) | torchcodec | `bash scripts/install-ffmpeg.sh` (miniforge/conda-forge) |
| libsndfile | soundfile | Usually bundled with soundfile pip package |

## Notes

- Python 3.14 requires torch >=2.9 (2.8.x doesn't have 3.14 wheels)
- praat-parselmouth minimum varies by Python (build toolchain compatibility)
- torchcodec requires FFmpeg shared libraries at runtime (not just the CLI binary)
- On macOS, homebrew FFmpeg 8 is incompatible with torchcodec; use `scripts/install-ffmpeg.sh`
- torchaudio.info() removed in 2.11; senselab uses soundfile.info() as fallback
