# Senselab Function Dependencies

| Function | Required Deps | Dep Versions | GPU | Isolated | Python | Torch |
|----------|--------------|-------------|-----|----------|--------|-------|
| `audio.tasks.classification.classify_audios` | transformers | transformers>=5.0 | No | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.classification.classify_emotions_from_speech` | transformers | transformers>=5.0 | No | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.data_augmentation.augment_audios` | audiomentations, torch-audiomentations | audiomentations>=0.42, torch-audiomentations>=0.12 | No | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.features_extraction.extract_features_from_audios` | torchaudio | torchaudio>=2.8 | No | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.features_extraction.extract_ppg_from_audios` | core | — | Yes | Yes (ppgs) | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.features_extraction.extract_sparc_features` | core | — | No | Yes (sparc) | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.features_extraction.extract_speechscore_metrics_from_audios` | core | — | No | Yes (clearvoice-speechscore) | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.forced_alignment.align_transcriptions` | transformers, torchaudio | transformers>=5.0, torchaudio>=2.8 | No | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.health_acoustics.detect_health_acoustic_events` | core | — | No | Yes (hear) | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.health_acoustics.extract_hear_embeddings_from_audios` | core | — | No | Yes (hear) | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.speaker_diarization.diarize_audios` | pyannote-audio, torchaudio | pyannote-audio>=4.0, torchaudio>=2.8 | No | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | speechbrain, torchaudio | speechbrain>=1.0, torchaudio>=2.8 | No | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.speech_enhancement.enhance_audios` | speechbrain, torchaudio | speechbrain>=1.0, torchaudio>=2.8 | No | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.speech_super_resolution.super_resolve_audios` | core | — | No | Yes (clearvoice) | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.speech_to_text.transcribe_audios` | transformers, torchaudio | transformers>=5.0, torchaudio>=2.8 | No | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.ssl_embeddings.extract_s3prl_embeddings` | core | — | No | Yes (s3prl) | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | transformers | transformers>=5.0 | No | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.target_speaker_extraction.extract_target_speakers_from_videos` | core | — | No | Yes (clearvoice) | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.text_to_speech.synthesize_texts` | transformers | transformers>=5.0 | No | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | pyannote-audio, torchaudio | pyannote-audio>=4.0, torchaudio>=2.8 | No | No | >=3.11 | >=2.8,<3.0 |
| `audio.tasks.voice_cloning.clone_voices` | core | — | No | Yes (coqui) | >=3.11 | >=2.8,<3.0 |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | transformers, sentence-transformers | transformers>=5.0, sentence-transformers>=5.1 | No | No | >=3.11 | >=2.8,<3.0 |
| `video.tasks.pose_estimation.estimate_pose` | ultralytics, opencv-python-headless | ultralytics>=8.0, opencv-python-headless>=4.8 | No | No | >=3.11 | >=2.8,<3.0 |

## Test Matrix

| Function | Python | Torch | Deps | Isolated |
|----------|--------|-------|------|----------|
| `audio.tasks.speech_to_text.transcribe_audios` | 3.11 | 2.8 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.speech_to_text.transcribe_audios` | 3.11 | 2.10 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.speech_to_text.transcribe_audios` | 3.12 | 2.8 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.speech_to_text.transcribe_audios` | 3.12 | 2.10 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.speech_to_text.transcribe_audios` | 3.13 | 2.8 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.speech_to_text.transcribe_audios` | 3.13 | 2.10 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.speech_to_text.transcribe_audios` | 3.14 | 2.8 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.speech_to_text.transcribe_audios` | 3.14 | 2.10 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.11 | 2.8 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.11 | 2.10 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.12 | 2.8 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.12 | 2.10 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.13 | 2.8 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.13 | 2.10 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.14 | 2.8 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_diarization.diarize_audios` | 3.14 | 2.10 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.11 | 2.8 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.11 | 2.10 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.12 | 2.8 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.12 | 2.10 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.13 | 2.8 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.13 | 2.10 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.14 | 2.8 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.speaker_embeddings.extract_speaker_embeddings_from_audios` | 3.14 | 2.10 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.11 | 2.8 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.11 | 2.10 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.12 | 2.8 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.12 | 2.10 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.13 | 2.8 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.13 | 2.10 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.14 | 2.8 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.speech_enhancement.enhance_audios` | 3.14 | 2.10 | speechbrain>=1.0,torchaudio>=2.8 | false |
| `audio.tasks.voice_cloning.clone_voices` | 3.11 | venv-managed |  | true |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.11 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.11 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.12 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.12 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.13 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.13 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.14 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.text_to_speech.synthesize_texts` | 3.14 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.11 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.11 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.12 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.12 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.13 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.13 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.14 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.classification.classify_audios` | 3.14 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.health_acoustics.extract_hear_embeddings_from_audios` | 3.11 | venv-managed | tensorflow>=2.16,<3,numpy,soundfile | true |
| `audio.tasks.health_acoustics.detect_health_acoustic_events` | 3.11 | venv-managed | tensorflow>=2.16,<3,numpy,soundfile | true |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.11 | 2.8 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.11 | 2.10 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.12 | 2.8 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.12 | 2.10 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.13 | 2.8 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.13 | 2.10 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.14 | 2.8 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.forced_alignment.align_transcriptions` | 3.14 | 2.10 | transformers>=5.0,torchaudio>=2.8 | false |
| `audio.tasks.features_extraction.extract_ppg_from_audios` | 3.11 | venv-managed |  | true |
| `audio.tasks.features_extraction.extract_sparc_features` | 3.11 | venv-managed |  | true |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.11 | 2.8 | transformers>=5.0,sentence-transformers>=5.1 | false |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.11 | 2.10 | transformers>=5.0,sentence-transformers>=5.1 | false |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.12 | 2.8 | transformers>=5.0,sentence-transformers>=5.1 | false |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.12 | 2.10 | transformers>=5.0,sentence-transformers>=5.1 | false |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.13 | 2.8 | transformers>=5.0,sentence-transformers>=5.1 | false |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.13 | 2.10 | transformers>=5.0,sentence-transformers>=5.1 | false |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.14 | 2.8 | transformers>=5.0,sentence-transformers>=5.1 | false |
| `text.tasks.embeddings_extraction.extract_embeddings_from_text` | 3.14 | 2.10 | transformers>=5.0,sentence-transformers>=5.1 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.11 | 2.8 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.11 | 2.10 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.12 | 2.8 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.12 | 2.10 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.13 | 2.8 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.13 | 2.10 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.14 | 2.8 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.data_augmentation.augment_audios` | 3.14 | 2.10 | audiomentations>=0.42,torch-audiomentations>=0.12 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.11 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.11 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.12 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.12 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.13 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.13 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.14 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.ssl_embeddings.extract_ssl_embeddings_from_audios` | 3.14 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.ssl_embeddings.extract_s3prl_embeddings` | 3.11 | venv-managed |  | true |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.11 | 2.8 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.11 | 2.10 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.12 | 2.8 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.12 | 2.10 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.13 | 2.8 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.13 | 2.10 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.14 | 2.8 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.voice_activity_detection.detect_human_voice_activity_in_audios` | 3.14 | 2.10 | pyannote-audio>=4.0,torchaudio>=2.8 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.11 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.11 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.12 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.12 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.13 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.13 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.14 | 2.8 | transformers>=5.0 | false |
| `audio.tasks.classification.classify_emotions_from_speech` | 3.14 | 2.10 | transformers>=5.0 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.11 | 2.8 | torchaudio>=2.8 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.11 | 2.10 | torchaudio>=2.8 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.12 | 2.8 | torchaudio>=2.8 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.12 | 2.10 | torchaudio>=2.8 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.13 | 2.8 | torchaudio>=2.8 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.13 | 2.10 | torchaudio>=2.8 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.14 | 2.8 | torchaudio>=2.8 | false |
| `audio.tasks.features_extraction.extract_features_from_audios` | 3.14 | 2.10 | torchaudio>=2.8 | false |
| `audio.tasks.features_extraction.extract_speechscore_metrics_from_audios` | 3.11 | venv-managed | torch>=2.0.1,torchaudio>=2.0.2,numpy<2.0,>=1.24.3,scipy>=1.10.1,librosa==0.10.2.post1,soundfile==0.12.1,resampy,museval,mir_eval==0.7,pesq==0.0.4,pystoi==0.3.3,onnxruntime,gammatone,pysptk,pyworld,fastdtw,xls_r_sqa,pandas,matplotlib,tqdm | true |
| `audio.tasks.speech_super_resolution.super_resolve_audios` | 3.11 | venv-managed | clearvoice==0.1.2,torch>=2.0.1,torchaudio>=2.0.2,numpy<2.0,>=1.24.3 | true |
| `audio.tasks.target_speaker_extraction.extract_target_speakers_from_videos` | 3.11 | venv-managed | clearvoice==0.1.2,torch>=2.0.1,torchaudio>=2.0.2,numpy<2.0,>=1.24.3 | true |
| `video.tasks.pose_estimation.estimate_pose` | 3.11 | 2.8 | ultralytics>=8.0,opencv-python-headless>=4.8 | false |
| `video.tasks.pose_estimation.estimate_pose` | 3.11 | 2.10 | ultralytics>=8.0,opencv-python-headless>=4.8 | false |
| `video.tasks.pose_estimation.estimate_pose` | 3.12 | 2.8 | ultralytics>=8.0,opencv-python-headless>=4.8 | false |
| `video.tasks.pose_estimation.estimate_pose` | 3.12 | 2.10 | ultralytics>=8.0,opencv-python-headless>=4.8 | false |
| `video.tasks.pose_estimation.estimate_pose` | 3.13 | 2.8 | ultralytics>=8.0,opencv-python-headless>=4.8 | false |
| `video.tasks.pose_estimation.estimate_pose` | 3.13 | 2.10 | ultralytics>=8.0,opencv-python-headless>=4.8 | false |
| `video.tasks.pose_estimation.estimate_pose` | 3.14 | 2.8 | ultralytics>=8.0,opencv-python-headless>=4.8 | false |
| `video.tasks.pose_estimation.estimate_pose` | 3.14 | 2.10 | ultralytics>=8.0,opencv-python-headless>=4.8 | false |
