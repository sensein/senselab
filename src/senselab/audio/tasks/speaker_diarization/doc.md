# Speaker diarization

[![Tutorial](https://img.shields.io/badge/Tutorial-Click%20Here-blue?style=for-the-badge)](https://github.com/sensein/senselab/blob/main/tutorials/audio/speaker_diarization.ipynb)

## Task Overview
Speaker diarization is the process of segmenting audio recordings by speaker labels, aiming to answer the question: **"Who spoke when?"**

## Models

In `senselab`, we integrate [pyannote.audio](https://github.com/pyannote/pyannote-audio) models for speaker diarization. These models can be explored on the [Hugging Face Hub](https://huggingface.co/pyannote). We may integrate additional approaches for speaker diarization into the package in the future.

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
