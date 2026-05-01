""".. include:: ./doc.md"""  # noqa: D415

from .api import extract_features_from_audios  # noqa: F401
from .ppg import (  # noqa: F401
    extract_mean_phoneme_durations,
    extract_ppg_segments,
    extract_ppgs_from_audios,
    plot_ppg_phoneme_timeline,
    to_frame_major_posteriorgram,
)
from .sparc import SparcFeatureExtractor  # noqa: F401

__all__ = [
    "extract_features_from_audios",
    "extract_mean_phoneme_durations",
    "extract_ppg_segments",
    "extract_ppgs_from_audios",
    "plot_ppg_phoneme_timeline",
    "to_frame_major_posteriorgram",
    "SparcFeatureExtractor",
]
