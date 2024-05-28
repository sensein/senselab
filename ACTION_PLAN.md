# Action Plan for Senselab

Senselab is a Python package designed to streamline the processing and analysis of behavioral data, including voice and speech, text, and video. Our goal is to implement robust and reproducible methodologies. 

This action plan outlines our initial functionalities and integrations for Senselab. As we progress, we will continue to enhance and expand these capabilities to support more advanced and comprehensive behavioral data analysis.
Anyone should feel free to suggest more features and methods. 

For an updated project progress, please see the [Project Board](https://github.com/orgs/sensein/projects/45).


## AUDIO

1. **Speech to Text API**
   - Integrate models supported by the Huggingface “automatic-speech-recognition” pipeline.

2. **Forced Alignment API**
   - Integrate wav2vec2 similar to whisperX (but without using WhisperX).

3. **Text to Speech API**
   - Integrate models supported by the Huggingface “text-to-speech” pipeline.
   - Suggested models (by Stan):
     - [Parler-tts](https://github.com/huggingface/parler-tts) - Apache 2.0 License - Great generic voices.
     - [Metavoice-src](https://github.com/metavoiceio/metavoice-src) - Apache 2.0 License - Supports voice cloning with good quality.
     - [OpenVoice](https://github.com/myshell-ai/OpenVoice) - MIT License.
     - [Bark](https://huggingface.co/suno/bark) - MIT License.
     - [GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) - MIT License - Supports voice cloning.

4. **Speaker Diarization API**
   - Integrate all models for speaker diarization by pyannote.audio accessible through the Huggingface hub.

5. **Voice Activity Detection API**
   - Integrate all models for speaker diarization by pyannote.audio accessible through the Huggingface hub.
   - Include torchaudio methods for VAD.

6. **Deep-learning Embeddings Extraction API**
   - Integrate all audio models available in the Huggingface hub.

7. **Handcrafted Features Extraction API**
   - Integrate features from opensmile, praat, and torchaudio.
   - Consider integrating features from [Nick’s table](https://docs.google.com/spreadsheets/d/18V_FrE3jYm1Msl4rg8xl8WVBDprbZrRFIVm3nEf2uTQ/edit?usp=sharing) in the future.

8. **Data Augmentation API**
   - Integrate all methods from torch-audiomentations.

9. **Preprocessing API**
   - Integrate methods for converting stereo to mono and for resampling.

10. **Voice Cloning API**
    - Integrate KNNVC.

11. **Speech Emotion Recognition API**
    - Integrate some existing and original models from Huggingface.

## TEXT

1. **Semantic Embeddings Extraction API**
   - Integrate some existing models from Huggingface sentence-transformer.

2. **Text Emotion Recognition API**
   - Integrate some existing models from Huggingface.

3. **Text Sentiment Analysis API**
   - Integrate some existing models from Huggingface.

## VIDEO

1. **Preprocessing API**
   - Integrate methods for extracting audio and changing the frame rate.

2. **Landmark Extraction API**
   - Integrate Mediapipe.

3. **Facial Action Unit Extraction API**
   - Integrate PyFeat.

4. **Facial Emotion Recognition API**
   - Integrate PyFeat.

5. **Body Pose Estimation API**
   - Integrate Mediapipe.
