# Senselab Compatibility Matrix

| Function | Required Deps | GPU | Isolated | Python | Torch |
|----------|--------------|-----|----------|--------|-------|
| `audio.tasks.classification.classify_audios` | transformers | Yes | No | >=3.11 | >=2.8 |
| `audio.tasks.features_extraction.extract_ppg_from_audios` | ppgs, espnet | Yes | Yes (ppgs) | >=3.11 | >=2.8 |
| `audio.tasks.forced_alignment.align_transcriptions` | transformers, torchaudio | Yes | No | >=3.11 | >=2.8 |
| `audio.tasks.speaker_diarization.diarize_audios` | pyannote-audio, torchaudio | Yes | No | >=3.11 | >=2.8 |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | speechbrain, torchaudio | Yes | No | >=3.11 | >=2.8 |
| `audio.tasks.speech_enhancement.enhance_audios` | speechbrain, torchaudio | Yes | No | >=3.11 | >=2.8 |
| `audio.tasks.speech_to_text.transcribe_audios` | transformers, torchaudio | No | No | >=3.11 | >=2.8 |
| `audio.tasks.text_to_speech.synthesize_texts` | transformers | Yes | No | >=3.11 | >=2.8 |
| `audio.tasks.voice_cloning.clone_voices` | coqui-tts | Yes | Yes (coqui) | >=3.11 | >=2.8 |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | transformers | Yes | No | >=3.11 | >=2.8 |
| `video.tasks.pose_estimation.estimate_pose` | ultralytics, opencv-python-headless | No | No | >=3.11 | >=2.8 |
