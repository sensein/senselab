# PPG Mean Phoneme Duration Task

## Goal

Add a feature-extraction task that summarizes mean phoneme durations from PPG model output, while also allowing optional `start_times` and `end_times` windows for PPG extraction.

## Plan

- [x] Branch from `enh/uv-refactor`
- [x] Add stable unit tests for PPG extraction without relying on remote model downloads
- [x] Add a task that converts posteriorgrams into mean phoneme duration summaries
- [x] Support optional per-audio `start_times` and `end_times`
- [x] Add a synthesized-phrase test path that exercises TTS -> PPG duration extraction
- [x] Run focused validation and capture any follow-up cleanup

## Notes

- The duration summary normalizes raw PPG tensor layouts to frame-major form before decoding the dominant phoneme per frame.
- Time-window support is implemented before model inference so duration summaries naturally reflect the requested slice.
- The synthesized-phrase coverage uses mocked model backends to keep tests deterministic and offline-friendly.
