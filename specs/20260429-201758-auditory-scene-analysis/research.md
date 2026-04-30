# Research: Auditory Scene Analysis

**Date**: 2026-04-29

## R1: Primary Model Choice

**Decision**: Use HuggingFace Audio Spectrogram Transformer (AST) as the primary model instead of Google's TensorFlow-based YAMNet. Also support YAMNet via TFLite subprocess venv as secondary option.

**Rationale**: AST is natively supported in HuggingFace Transformers (PyTorch), achieves state-of-the-art 0.485 mAP on AudioSet (521 classes), and works with senselab's existing `audio-classification` pipeline. No TensorFlow dependency needed. YAMNet requires TF/TFLite which would conflict with the main PyTorch environment.

**Models to support**:
1. `MIT/ast-finetuned-audioset-10-10-0.4593` — AST fine-tuned on AudioSet (PyTorch, HuggingFace pipeline)
2. `google/yamnet` or `STMicroelectronics/yamnet` — YAMNet via HuggingFace if available, or TFLite subprocess venv
3. Any HuggingFace `audio-classification` model — generic support via existing pipeline

**Alternatives considered**:
- PANNs: Not natively on HuggingFace Transformers; would need custom loading
- BEATs: Available on HuggingFace but fewer pre-trained checkpoints for AudioSet
- CLAP: Good for zero-shot but different use case (text-audio retrieval)

## R2: Windowed Iteration Approach

**Decision**: Implement windowed classification by slicing the audio waveform tensor into overlapping windows, batching them, and running the classifier on each batch. Return results as a list of per-window classification outputs with timestamps.

**Rationale**: Simple tensor slicing is efficient and doesn't require creating separate Audio objects per window. The existing `classify_audios_with_transformers` can process a list of Audio objects, so we create lightweight Audio objects from window slices.

**Parameters**:
- `window_size`: float (seconds), default 1.0
- `hop_size`: float (seconds), default 0.5
- `top_k`: int (number of top predictions per window), default 5

## R3: Output Format

**Decision**: Return `List[Dict]` where each dict has `start`, `end`, `labels`, `scores`. This is compatible with the `segments` panel type in `plot_aligned_panels`.

**Rationale**: The dict format is simple, JSON-serializable, and compatible with existing visualization patterns. It can be easily converted to DataFrames or segment annotations.

## R4: Visualization

**Decision**: Reuse `plot_aligned_panels` with a new panel approach — the "segments" panel type already supports colored bars with labels. For scene classification, we show the top-1 predicted class per window as a colored bar.

**Rationale**: No new plotting code needed — existing infrastructure handles this.
