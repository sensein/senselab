Audio visualization and playback utilities.

This module provides functions for visualizing audio data: `plot_waveform` (amplitude over time), `plot_specgram` (frequency content over time, linear or mel scale), `plot_waveform_and_specgram` (stacked layout), and `plot_aligned_panels` (reusable multi-panel time-aligned visualization supporting waveform, spectrogram, feature overlays, and segment annotations on a shared time axis). Use `play_audio` for inline audio playback in Jupyter notebooks. All plotting functions support context-aware scaling for different display sizes.
