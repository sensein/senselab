"""Unsupervised audio source separation.

Wraps unasdiff (https://github.com/RunwuShi/unasdiff), which separates a mixture
into speech and an FSD50K-conditioned sound source using two independently-trained
diffusion priors, without training on mixtures. See
``senselab.audio.tasks.source_separation.unasdiff`` for the backend's design,
licensing position, and why it runs in an isolated subprocess venv.
"""

from senselab.audio.tasks.source_separation.api import resolve_source_classes, separate_audios

__all__ = ["resolve_source_classes", "separate_audios"]
