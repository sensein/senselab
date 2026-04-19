# Senselab Compatibility Matrix

| Function | Required Deps | Dep Versions | GPU | Isolated | Python | Torch |
|----------|--------------|-------------|-----|----------|--------|-------|
| `audio.tasks.classification.classify_audios` | transformers | transformers>=4.40,<5.0 | Yes | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.classification.classify_emotions_from_speech` | transformers | transformers>=4.40,<5.0 | Yes | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.data_augmentation.augment_audios` | audiomentations, torch-audiomentations | audiomentations>=0.42, torch-audiomentations>=0.12 | No | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.features_extraction.extract_features_from_audios` | torchaudio | torchaudio>=2.8,<3.0 | No | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.features_extraction.extract_ppg_from_audios` | ppgs, espnet | ppgs>=0.0.9,<0.0.10, espnet>=202205, torch>=2.0,<2.9 | Yes | Yes (ppgs) | >=3.10,<3.12 | >=2.8,<3.0 |
| `audio.tasks.forced_alignment.align_transcriptions` | transformers, torchaudio | transformers>=4.40,<5.0, torchaudio>=2.8,<3.0 | Yes | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.speaker_diarization.diarize_audios` | pyannote-audio, torchaudio, torchcodec | pyannote-audio>=3.0,<5.0, torchaudio>=2.8,<3.0, torchcodec>=0.7 | Yes | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | speechbrain, torchaudio | speechbrain>=1.0,<2.0, torchaudio>=2.8,<3.0 | Yes | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.speech_enhancement.enhance_audios` | speechbrain, torchaudio | speechbrain>=1.0,<2.0, torchaudio>=2.8,<3.0 | Yes | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.speech_to_text.transcribe_audios` | transformers, torchaudio | transformers>=4.40,<5.0, torchaudio>=2.8,<3.0 | No | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | transformers | transformers>=4.40,<5.0 | Yes | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.text_to_speech.synthesize_texts` | transformers | transformers>=4.40,<5.0 | Yes | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | pyannote-audio, torchaudio | pyannote-audio>=3.0,<5.0, torchaudio>=2.8,<3.0 | Yes | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.voice_cloning.clone_voices` | coqui-tts | coqui-tts>=0.27,<1.0, torch>=2.4,<2.9 | Yes | Yes (coqui) | >=3.10,<3.12 | >=2.8,<3.0 |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | transformers, sentence-transformers | transformers>=4.40,<5.0, sentence-transformers>=5.1 | Yes | No | >=3.11 | >=2.8,<3.0 |
| `video.tasks.pose_estimation.estimate_pose` | ultralytics, opencv-python-headless | ultralytics>=8.0,<9.0, opencv-python-headless>=4.8,<5.0 | No | No | >=3.11 | >=2.8,<3.0 |

## Test Matrix

| Function | Python | Torch | Deps | Isolated |
|----------|--------|-------|------|----------|
| `audio.tasks.speech_to_text.transcribe_audios` | 3.11 | 2.8 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speech_to_text.transcribe_audios` | 3.11 | 2.10 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speech_to_text.transcribe_audios` | 3.12 | 2.8 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speech_to_text.transcribe_audios` | 3.12 | 2.10 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speech_to_text.transcribe_audios` | 3.13 | 2.8 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speech_to_text.transcribe_audios` | 3.13 | 2.10 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speech_to_text.transcribe_audios` | 3.14 | 2.8 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speech_to_text.transcribe_audios` | 3.14 | 2.10 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.11 | 2.8 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0,torchcodec>=0.7 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.11 | 2.10 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0,torchcodec>=0.7 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.12 | 2.8 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0,torchcodec>=0.7 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.12 | 2.10 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0,torchcodec>=0.7 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.13 | 2.8 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0,torchcodec>=0.7 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.13 | 2.10 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0,torchcodec>=0.7 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.14 | 2.8 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0,torchcodec>=0.7 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.14 | 2.10 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0,torchcodec>=0.7 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.11 | 2.8 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.11 | 2.10 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.12 | 2.8 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.12 | 2.10 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.13 | 2.8 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.13 | 2.10 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.14 | 2.8 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.14 | 2.10 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.11 | 2.8 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.11 | 2.10 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.12 | 2.8 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.12 | 2.10 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.13 | 2.8 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.13 | 2.10 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.14 | 2.8 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.14 | 2.10 | speechbrain>=1.0,<2.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.voice_cloning.clone_voices` | 3.11 | venv-managed | coqui-tts~=0.27,torch~=2.8 | true |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.11 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.11 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.12 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.12 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.13 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.13 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.14 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.14 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.11 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.11 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.12 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.12 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.13 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.13 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.14 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.14 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.11 | 2.8 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.11 | 2.10 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.12 | 2.8 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.12 | 2.10 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.13 | 2.8 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.13 | 2.10 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.14 | 2.8 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.14 | 2.10 | transformers>=4.40,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.features_extraction.extract_ppg_from_audios` | 3.11 | venv-managed | ppgs>=0.0.9,<0.0.10,espnet,snorkel>=0.10.0,<0.11.0,lightning~=2.4 | true |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.11 | 2.8 | transformers>=4.40,<5.0,sentence-transformers>=5.1 | false |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.11 | 2.10 | transformers>=4.40,<5.0,sentence-transformers>=5.1 | false |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.12 | 2.8 | transformers>=4.40,<5.0,sentence-transformers>=5.1 | false |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.12 | 2.10 | transformers>=4.40,<5.0,sentence-transformers>=5.1 | false |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.13 | 2.8 | transformers>=4.40,<5.0,sentence-transformers>=5.1 | false |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.13 | 2.10 | transformers>=4.40,<5.0,sentence-transformers>=5.1 | false |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.14 | 2.8 | transformers>=4.40,<5.0,sentence-transformers>=5.1 | false |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.14 | 2.10 | transformers>=4.40,<5.0,sentence-transformers>=5.1 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.11 | 2.8 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.11 | 2.10 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.12 | 2.8 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.12 | 2.10 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.13 | 2.8 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.13 | 2.10 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.14 | 2.8 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.14 | 2.10 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.11 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.11 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.12 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.12 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.13 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.13 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.14 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.14 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.11 | 2.8 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.11 | 2.10 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.12 | 2.8 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.12 | 2.10 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.13 | 2.8 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.13 | 2.10 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.14 | 2.8 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.14 | 2.10 | pyannote-audio>=3.0,<5.0,torchaudio>=2.8,<3.0 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.11 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.11 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.12 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.12 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.13 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.13 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.14 | 2.8 | transformers>=4.40,<5.0 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.14 | 2.10 | transformers>=4.40,<5.0 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.11 | 2.8 | torchaudio>=2.8,<3.0 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.11 | 2.10 | torchaudio>=2.8,<3.0 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.12 | 2.8 | torchaudio>=2.8,<3.0 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.12 | 2.10 | torchaudio>=2.8,<3.0 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.13 | 2.8 | torchaudio>=2.8,<3.0 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.13 | 2.10 | torchaudio>=2.8,<3.0 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.14 | 2.8 | torchaudio>=2.8,<3.0 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.14 | 2.10 | torchaudio>=2.8,<3.0 | false |
| `video.tasks.pose_estimation.estimate_pose` | 3.11 | 2.8 | ultralytics>=8.0,<9.0,opencv-python-headless>=4.8,<5.0 | false |
| `video.tasks.pose_estimation.estimate_pose` | 3.11 | 2.10 | ultralytics>=8.0,<9.0,opencv-python-headless>=4.8,<5.0 | false |
| `video.tasks.pose_estimation.estimate_pose` | 3.12 | 2.8 | ultralytics>=8.0,<9.0,opencv-python-headless>=4.8,<5.0 | false |
| `video.tasks.pose_estimation.estimate_pose` | 3.12 | 2.10 | ultralytics>=8.0,<9.0,opencv-python-headless>=4.8,<5.0 | false |
| `video.tasks.pose_estimation.estimate_pose` | 3.13 | 2.8 | ultralytics>=8.0,<9.0,opencv-python-headless>=4.8,<5.0 | false |
| `video.tasks.pose_estimation.estimate_pose` | 3.13 | 2.10 | ultralytics>=8.0,<9.0,opencv-python-headless>=4.8,<5.0 | false |
| `video.tasks.pose_estimation.estimate_pose` | 3.14 | 2.8 | ultralytics>=8.0,<9.0,opencv-python-headless>=4.8,<5.0 | false |
| `video.tasks.pose_estimation.estimate_pose` | 3.14 | 2.10 | ultralytics>=8.0,<9.0,opencv-python-headless>=4.8,<5.0 | false |
