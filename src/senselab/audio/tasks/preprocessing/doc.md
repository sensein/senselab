Audio preprocessing utilities for preparing audio signals before analysis.

This module provides functions for resampling audio to different sample rates (`resample_audios`), downmixing multi-channel audio to mono (`downmix_audios_to_mono`), and chunking audio into segments (`chunk_audios`). These operations are typically the first step in any audio analysis pipeline — most models expect mono audio at a specific sample rate (commonly 16 kHz).
