""".. include:: ./doc.md"""  # noqa: D415

from .plotting import (  # noqa: F401
    play_audio,
    plot_aligned_panels,
    plot_range_with_ppg,
    plot_specgram,
    plot_waveform,
)

__all__ = ["play_audio", "plot_aligned_panels", "plot_range_with_ppg", "plot_specgram", "plot_waveform"]
