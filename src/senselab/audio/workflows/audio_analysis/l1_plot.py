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


def spectrogram_db(
    waveform: np.ndarray,
    sampling_rate: int,
    *,
    n_fft: int = 1024,
    hop_s: float = 0.01,
    floor_db: float = -80.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Magnitude spectrogram in dB, normalised so the loudest bin is 0 dB.

    dB rather than linear magnitude: on a linear scale everything below the loudest bin
    collapses to black, which hides exactly the quiet background content the rest of this
    pipeline is about. Normalised to the peak so the colour scale means the same thing across
    recordings at different levels.

    Returns:
        ``(spec_db, times, freqs)``, floored at ``floor_db`` — digital silence would otherwise
        be ``-inf``, which cannot be rendered.
    """
    arr = np.asarray(waveform, dtype=np.float64).squeeze()
    hop = max(1, int(round(float(hop_s) * float(sampling_rate))))
    if arr.size < n_fft:
        arr = np.pad(arr, (0, n_fft - arr.size))
    window = np.hanning(n_fft)
    starts = range(0, max(1, arr.size - n_fft + 1), hop)
    frames = np.stack([arr[i : i + n_fft] * window for i in starts], axis=1)
    magnitude = np.abs(np.fft.rfft(frames, axis=0))
    peak = float(magnitude.max()) or 1.0
    with np.errstate(divide="ignore"):
        spec = 20.0 * np.log10(np.maximum(magnitude / peak, 1e-12))
    times = np.array([i / float(sampling_rate) for i in starts])
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sampling_rate))
    return np.maximum(spec, floor_db), times, freqs


def scene_composition(
    windows: Sequence[Mapping[str, object]],
    *,
    duration_s: float,
    hop_s: float = 0.1,
    categories: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-category share of a classifier's output over time, for a composition plot.

    Normalised per column so the rows read as *shares* — a composition plot whose columns do
    not sum to one is a stack of unrelated magnitudes and cannot be compared across time.

    Time the classifier never covered stays empty rather than becoming an even split: an even
    split would look like "all categories equally present" where the truth is "nothing was
    measured here", and those license opposite conclusions.

    Returns:
        ``(times, shares)`` with ``shares`` shaped ``(len(categories), len(times))``.
    """
    from senselab.audio.workflows.audio_analysis.sound_sources import (
        SOURCE_CATEGORIES,
        _category_for,
        load_source_category_map,
    )

    names = list(categories or SOURCE_CATEGORIES)
    try:
        doc = load_source_category_map()
        mapping, default = dict(doc.get("map") or {}), str(doc.get("default") or names[-1])
    except (OSError, ValueError):
        mapping, default = {}, names[-1]

    n_cols = max(1, int(round(max(duration_s, hop_s) / hop_s)))
    times = np.arange(n_cols) * hop_s
    mass = np.zeros((len(names), n_cols), dtype=np.float64)
    index = {name: i for i, name in enumerate(names)}

    for window in windows or ():
        if not isinstance(window, Mapping):
            continue
        w_start = float(window.get("start", 0.0) or 0.0)
        w_end = float(window.get("end", 0.0) or 0.0)
        cols = np.where((times + hop_s > w_start) & (times < w_end))[0]
        if cols.size == 0:
            continue
        for label, score in zip(window.get("labels") or [], window.get("scores") or []):
            category = _category_for(str(label), mapping, default)
            row = index.get(category)
            if row is None:
                continue
            mass[row, cols] += max(0.0, float(score))

    totals = mass.sum(axis=0)
    shares = np.zeros_like(mass)
    covered = totals > 0
    shares[:, covered] = mass[:, covered] / totals[covered]
    return times, shares


def build_l1_signal_plot(
    out_dir: Path | str,
    *,
    signals: Mapping[str, Sequence[tuple[float, float]]],
    duration_s: float,
    waveform: np.ndarray | None = None,
    sampling_rate: int = 16000,
    words: Sequence[Mapping[str, object]] | None = None,
    scene_by_classifier: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    title: str | None = None,
) -> Path:
    """Draw one row per L1 signal plus a dBFS level row, and save under ``L1/signals.png``.

    Args:
        out_dir: Run directory.
        signals: ``{signal → [(start, end), ...]}`` spans the signal reported.
        duration_s: Recording length, so empty rows still span the figure.
        waveform: Mono samples for the spectrogram and level rows, or ``None`` to omit them.
        sampling_rate: Sample rate of ``waveform``.
        words: Aligned words (``start``, ``end``, ``text``) drawn as a transcript row, so a
            disagreement can be read against what was actually said there.
        scene_by_classifier: ``{classifier → windows}`` drawn as stacked category shares, one
            row per classifier — AST and YAMNet disagree often enough that a merged row would
            hide which of them saw what.
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
    scene_names = sorted(scene_by_classifier or {})
    n_rows = (
        len(names) + len(scene_names) + (1 if words else 0) + (2 if waveform is not None else 0)  # spectrogram + level
    )
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

    row = len(names)
    for classifier in scene_names:
        ax = flat[row]
        row += 1
        ax.set_ylabel(f"scene\n{classifier}", rotation=0, ha="right", va="center", fontsize=7)
        ax.set_xlim(0, max(duration_s, 1e-6))
        times, shares = scene_composition(scene_by_classifier[classifier], duration_s=duration_s)
        if shares.size:
            ax.stackplot(times, shares, linewidth=0)
            ax.set_ylim(0, 1)
        ax.set_yticks([])

    if words:
        ax = flat[row]
        row += 1
        ax.set_ylabel("words", rotation=0, ha="right", va="center", fontsize=7)
        ax.set_xlim(0, max(duration_s, 1e-6))
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        for word in words:
            start = float(word.get("start", 0.0) or 0.0)
            end = float(word.get("end", start) or start)
            ax.text(
                (start + end) / 2.0,
                0.5,
                str(word.get("text") or ""),
                ha="center",
                va="center",
                fontsize=6,
                rotation=0,
                clip_on=True,
            )
            ax.axvline(start, color="#bbbbbb", linewidth=0.3)

    if waveform is not None:
        ax = flat[row]
        row += 1
        spec, s_times, s_freqs = spectrogram_db(waveform, sampling_rate)
        ax.set_ylabel("spectrogram", rotation=0, ha="right", va="center", fontsize=7)
        ax.imshow(
            spec,
            origin="lower",
            aspect="auto",
            extent=(0.0, float(s_times[-1] if s_times.size else duration_s), 0.0, float(s_freqs[-1])),
            cmap="magma",
        )
        ax.set_xlim(0, max(duration_s, 1e-6))
        ax.set_yticks([])

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
