# Tutorial Notebook Audit

Generated: 2026-04-20

## Summary

| Notebook | Has Setup | Has Badge | Needs GPU | Needs HF_TOKEN | Status |
|----------|-----------|-----------|-----------|----------------|--------|
| 00_getting_started | Yes | Yes | No* | Yes | Fix setup cell |
| audio_data_augmentation | Yes | Yes | No | No | Fix setup cell |
| conversational_data_exploration | Yes | Yes | Yes | No** | Fix setup cell, add HF_TOKEN |
| extract_speaker_embeddings | Yes | Yes | No | No | Fix setup cell |
| features_extraction | Yes | Yes | No | No | Fix setup cell |
| forced_alignment | Yes | Yes | No | No | Fix setup cell |
| speaker_diarization | Yes | Yes | No | No** | Fix setup cell, add HF_TOKEN |
| speaker_verification | Yes | Yes | No | No | Fix setup cell |
| speech_emotion_recognition | Yes | No | Yes | No | Fix setup cell, add badge |
| speech_enhancement | Yes | Yes | No | No | Fix setup cell |
| speech_to_text | Yes | Yes | No | No | Fix setup cell |
| text_to_speech | Yes | Yes | No | No | Fix setup cell |
| voice_activity_detection | Yes | Yes | No | No** | Fix setup cell, add HF_TOKEN |
| voice_cloning | Yes | Yes | No | No | Fix setup cell |
| pose_estimation | Yes | Yes | No | No | Fix setup cell |
| dimensionality_reduction | Yes | Yes | No | No | Fix setup cell |
| senselab_ai_intro | No | No | No | No | Add setup + badge |

*Uses pyannote models that need HF_TOKEN
**Uses pyannote models but doesn't document HF_TOKEN requirement

## Key Findings

- All notebooks have existing setup cells using `%pip install senselab` — need updating to `uv pip install --pre`
- 16/17 have Colab badges (missing: senselab_ai_intro, speech_emotion_recognition)
- Badges link to `main` branch — need updating to `alpha`
- No broken imports, but torchaudio deprecation warnings throughout
- Only 00_getting_started documents HF_TOKEN for pyannote models
