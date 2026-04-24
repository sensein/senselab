# Data Model: Improve SER Tutorial

**Date**: 2026-04-24

No new data models needed. The tutorial uses existing senselab entities.

## Existing Entities Used

### AudioClassificationResult
- **labels**: List[str] — emotion class names, sorted by score descending
- **scores**: List[float] — confidence scores, sorted descending
- **top_label()**: Returns highest-confidence label
- **top_score()**: Returns highest confidence score

### Audio
- Standard senselab Audio object loaded from recording or file

### HFModel
- Model specification for HuggingFace models

## Models Used in Tutorial

| Model | Type | Training Data | Best For |
|-------|------|--------------|----------|
| superb/wav2vec2-base-superb-er | Discrete | IEMOCAP | Conversational speech |
| ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition | Discrete | RAVDESS | Acted speech |
| audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim | Continuous | MSP-Podcast | Dimensional emotion |
| cardiffnlp/twitter-roberta-base-sentiment-latest | Text | Twitter | Text sentiment |
