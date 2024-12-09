# Voice Activity Detection (VAD)

[![Tutorial](https://img.shields.io/badge/Tutorial-Click%20Here-blue?style=for-the-badge)](https://github.com/sensein/senselab/blob/main/tutorials/audio/voice_activity_detection.ipynb)

## Task Overview

Voice Activity Detection (VAD) is a binary classification task that identifies the presence of human voice in audio. The primary challenge in VAD lies in differentiating between noise and human voice, particularly in environments with significant background noise (e.g., fans, car engines). While VAD performs well in quiet environments where distinguishing between silence and speech is straightforward, the task becomes more difficult when background noise or non-standard speech patterns are present.

## Models

In `senselab`, we integrate [pyannote.audio](https://github.com/pyannote/pyannote-audio) models for VAD. These models can be explored on the [Hugging Face Hub](https://huggingface.co/pyannote). Additional approaches for VAD may be integrated into the package in the future.

## Evaluation

### Metrics

The primary metrics used to evaluate VAD modules are Detection Error Rate (DER) and Detection Cost Function (DCF).

- **Detection Error Rate (DER):**

  ```text
    DER = (false alarm + missed detection) / total duration of speech in reference
  ```

  - **False alarm:** duration of non-speech incorrectly classified as speech.
  - **Missed detection:** duration of speech incorrectly classified as non-speech.
  - **Total:** Total duration of speech in the reference.

- **Detection Cost Function (DCF):**

  ```text
  DCF = 0.25 * false alarm rate + 0.75 * miss rate
  ```

  - **False alarm rate:** Proportion of non-speech incorrectly classified as speech.
  - **Miss rate:** Proportion of speech incorrectly classified as non-speech.

### Additional Metrics

VAD systems may also be evaluated using the following metrics:

- **Accuracy:** Proportion of the input signal correctly classified.
- **Precision:** Proportion of detected speech that is actually speech.
- **Recall:** Proportion of speech that is correctly detected.

For more detailed information on these metrics, refer to the [pyannote.metrics documentation](https://pyannote.github.io/pyannote-metrics/reference.html).
