# Data Model: Pedagogical Audio Tutorials

**Date**: 2026-04-23

This feature is primarily tutorial/notebook content — no new persistent data models are introduced. The tutorials use existing senselab data structures.

## Existing Entities Used

### Audio
- **Source**: `senselab.audio.data_structures.Audio`
- **Created from**: file path, waveform tensor + sampling rate, or stream
- **Key attributes**: `waveform` (torch.Tensor), `sampling_rate` (int), `filepath` (optional Path)
- **Used in**: All tutorials (loaded from file or recording)

### AudioClassificationResult
- **Source**: `senselab.audio.data_structures.AudioClassificationResult`
- **Key attributes**: `labels` (List[str]), `scores` (List[float])
- **Used in**: Tutorial 1 (emotion detection)

### ScriptLine
- **Source**: `senselab.audio.data_structures.ScriptLine`
- **Key attributes**: `text` (str), `start` (float), `end` (float), `chunks` (List[ScriptLine])
- **Used in**: Tutorial 2 (ASR output, forced alignment)

### Model Types
- `HFModel`: Hugging Face model reference (ASR, emotion)
- `SpeechBrainModel`: SpeechBrain model (speaker verification, embeddings)
- `DeviceType`: CPU/CUDA device selection

## New API Surface (fresh implementation, superseding PR #431)

### extract_mean_phoneme_durations
- **Input**: `audio: Audio`, `posteriorgram: torch.Tensor`
- **Output**: `Dict[str, Any]` with `phoneme_durations` list
- **Used in**: Tutorial 2, SHBT205-Lab

### plot_ppg_phoneme_timeline
- **Input**: `audio: Audio`, `posteriorgram: torch.Tensor`, `title: str`
- **Output**: `matplotlib.figure.Figure`
- **Used in**: Tutorial 2, SHBT205-Lab

## Tutorial Manifest Entries (new)

```json
{
    "path": "tutorials/audio/audio_recording_and_acoustic_analysis.ipynb",
    "benefits_from_gpu": true,
    "requires_hf_token": false,
    "timeout_cpu": 600,
    "timeout_gpu": 300,
    "category": "audio"
},
{
    "path": "tutorials/audio/transcription_and_phonemic_analysis.ipynb",
    "benefits_from_gpu": true,
    "requires_hf_token": false,
    "timeout_cpu": 1800,
    "timeout_gpu": 600,
    "category": "audio"
},
{
    "path": "tutorials/audio/shbt205_lab.ipynb",
    "benefits_from_gpu": true,
    "requires_hf_token": true,
    "timeout_cpu": 1800,
    "timeout_gpu": 600,
    "category": "audio"
}
```
