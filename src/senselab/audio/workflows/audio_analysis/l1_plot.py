"""``L1/signals.png`` — the evidence plot: every signal, plus level, and no conclusions.

L1 is evidence, so its figure shows what each signal reported and how loud the audio was while
it reported it. That pairing explains most disagreements: "the diarizer stopped here" next to
"the level fell to -60 dBFS here" is usually the whole story, and neither row says it alone.

Level is plotted in **dBFS** rather than raw RMS because a level track is read against full
scale — 0 dBFS is the anchor a reader already has — and amplitude-referenced, so halving the
amplitude reads as -6 dB rather than -3.

Two deliberate omissions:

**No uncertainty rows.** Those are level-2 conclusions drawn *from* this evidence. A figure
that mixes the two invites reading a conclusion as another observation, which is how a derived
signal came to be treated as a peer in the first place.

**No signal is dropped.** A model that ran and reported nothing still gets a row, because
otherwise its silence is indistinguishable from its absence — and "this model reported nothing
here" is frequently the informative part.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "DBFS_FLOOR",
    "build_l1_signal_plot",
    "rms_dbfs_track",
]

DBFS_FLOOR = -100.0
"""Floor for digital silence. ``-inf`` cannot be plotted; a floor keeps the axis usable while
staying visibly bottomed out."""


def rms_dbfs_track(
    waveform: np.ndarray,
    sampling_rate: int,
    *,
    hop_s: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Short-time RMS level in dBFS, at the requested hop.

    The level row is the one that should stay at native resolution rather than being pushed
    onto a bucket grid: a brief dropout is exactly what a coarse grid hides, and a brief
    dropout is what explains a diarizer's gap.

    Returns:
        ``(times, levels_dbfs)``, with ``levels_dbfs`` floored at :data:`DBFS_FLOOR`.
    """
    arr = np.asarray(waveform, dtype=np.float64).squeeze()
    hop = max(1, int(round(float(hop_s) * float(sampling_rate))))
    if arr.size < hop:
        return np.zeros(0), np.zeros(0)
    frames = arr[: (arr.size // hop) * hop].reshape(-1, hop)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    # 20*log10 on amplitude, so a half-amplitude signal reads -6 dB.
    with np.errstate(divide="ignore"):
        levels = 20.0 * np.log10(np.maximum(rms, 1e-12))
    times = np.arange(frames.shape[0]) * (hop / float(sampling_rate))
    return times, np.maximum(levels, DBFS_FLOOR)


def build_l1_signal_plot(
    out_dir: Path | str,
    *,
    signals: Mapping[str, Sequence[tuple[float, float]]],
    duration_s: float,
    waveform: np.ndarray | None = None,
    sampling_rate: int = 16000,
    title: str | None = None,
) -> Path:
    """Draw one row per L1 signal plus a dBFS level row, and save under ``L1/signals.png``.

    Args:
        out_dir: Run directory.
        signals: ``{signal → [(start, end), ...]}`` spans the signal reported.
        duration_s: Recording length, so empty rows still span the figure.
        waveform: Mono samples for the level row, or ``None`` to omit it.
        sampling_rate: Sample rate of ``waveform``.
        title: Figure title.

    Returns:
        The written path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from senselab.audio.workflows.audio_analysis.layout import evidence_dir

    names = sorted(signals)
    n_rows = len(names) + (1 if waveform is not None else 0)
    height = max(2.0, 0.45 * max(1, n_rows) + 1.2)
    fig, axes = plt.subplots(max(1, n_rows), 1, figsize=(14, height), sharex=True, squeeze=False)
    flat = [ax for (ax,) in axes]

    for ax, name in zip(flat, names):
        ax.set_ylabel(name[:34], rotation=0, ha="right", va="center", fontsize=7)
        ax.set_xlim(0, max(duration_s, 1e-6))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for start, end in signals[name] or ():
            ax.add_patch(Rectangle((float(start), 0.1), max(1e-3, float(end) - float(start)), 0.8, color="#3b6ea5"))

    if waveform is not None:
        ax = flat[-1]
        times, levels = rms_dbfs_track(waveform, sampling_rate)
        ax.set_ylabel("RMS\ndBFS", rotation=0, ha="right", va="center", fontsize=7)
        ax.set_xlim(0, max(duration_s, 1e-6))
        if times.size:
            ax.plot(times, levels, linewidth=0.7, color="#333333")
            ax.set_ylim(DBFS_FLOOR, 3.0)
        ax.grid(True, axis="y", alpha=0.25)

    flat[-1].set_xlabel("time (s)")
    fig.suptitle(title or "L1 signals (evidence)", fontsize=10)
    fig.tight_layout()

    dest = evidence_dir(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "signals.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
