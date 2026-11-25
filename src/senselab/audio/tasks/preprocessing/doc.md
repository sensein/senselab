Audio preprocessing utilities for preparing audio signals before analysis.

This module provides functions for resampling audio to different sample rates (`resample_audios`), downmixing multi-channel audio to mono (`downmix_audios_to_mono`), and chunking audio into segments (`chunk_audios`). These operations are typically the first step in any audio analysis pipeline — most models expect mono audio at a specific sample rate (commonly 16 kHz).

`resample_audios` computes its filter in float64 by default and returns float32. In tests on Intel and AMD CPUs this gave identical output on both, at about 2–2.5× the resample time of float32 (30 minutes of 48 kHz audio took 3–5 s on one thread, against 1.3–1.9 s). Pass `precision="float32"` for the faster computation; its output can differ in the last bit between CPU vendors. Each resampled `Audio` records the precision it ran with, and the CPU and library versions, in `metadata["processing_steps"]` and `metadata["environments"]`.
