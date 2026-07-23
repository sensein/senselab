"""Scene-quality models for the audio_analysis workflow.

Currently wraps ``pyannote/brouhaha`` — a multitask model that predicts
per-frame voice activity, signal-to-noise ratio (SNR), and room-acoustics
clarity (C50) in a single forward pass. Used by the presence axis to source
the ``quality_snr`` / ``quality_reverb`` degradation scores and a second
frame-level speech-presence voter.
"""

from senselab.audio.tasks.scene_quality.brouhaha import BrouhahaFrames, extract_brouhaha_frames

__all__ = ["BrouhahaFrames", "extract_brouhaha_frames"]
