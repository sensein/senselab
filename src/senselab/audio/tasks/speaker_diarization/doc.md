# Speaker diarization

[![Tutorial](https://img.shields.io/badge/Tutorial-Click%20Here-blue?style=for-the-badge)](https://github.com/sensein/senselab/blob/main/tutorials/audio/speaker_diarization.ipynb)

## Task Overview
Speaker diarization is the process of segmenting audio recordings by speaker labels, aiming to answer the question: **"Who spoke when?"**

## Models

All supported models ([pyannote.audio](https://github.com/pyannote/pyannote-audio) models from [Hugging Face Hub](https://huggingface.co/pyannote) and the [NVIDIA nemo softformer](https://huggingface.co/nvidia/diar_sortformer_4spk-v1)) provide state-of-the-art diarization and can be used as follows:

```python
from senselab.audio.data_structures import Audio
from senselab.utils.data_structures.model import HFModel
from senselab.audio.tasks.speaker_diarization.api import diarize_audios

audio = Audio(filepath="path/to/audio_48khz_mono_16bits.wav")
model = HFModel(path_or_uri="nvidia/diar_sortformer_4spk-v1")
results = diarize_audios([audio], model=model)
# results is a list of lists of ScriptLine objects, e.g.:
# [[ScriptLine(speaker="speaker_0", start=0.08, end=4.95), ...]]
```

We may integrate additional approaches for speaker diarization into the package in the future.


## Evaluation

### Metrics

The **Diarization Error Rate (DER)** is the standard metric for evaluating and comparing speaker diarization systems. It is defined as:
```text
DER= (false alarm + missed detection + confusion) / total
```
where
- `false alarm` is the duration of non-speech incorrectly classified as speech, missed detection
- `missed detection` is the duration of speech incorrectly classified as non-speech, confusion
- `confusion` is the duration of speaker confusion, and total
- `total` is the sum over all speakers of their reference speech duration.

**Note:** DER takes overlapping speech into account. This can lead to increased missed detection rates if the speaker diarization system does not include an overlapping speech detection module.

### Benchmark

You can find a benchmark of the latest pyannote.audio model's performance on various time-stamped speech datasets [here](https://github.com/pyannote/pyannote-audio?tab=readme-ov-file#benchmark).
