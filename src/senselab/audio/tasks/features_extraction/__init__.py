""".. include:: ./doc.md"""  # noqa: D415

from .api import extract_features_from_audios  # noqa: F401
from .ppg import (  # noqa: F401
    extract_mean_phoneme_durations_from_audios,
    extract_ppgs_from_audios,
    plot_ppg_phoneme_timeline,
)
