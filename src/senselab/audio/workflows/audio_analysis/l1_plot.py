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
    "classify_signal",
    "SIGNAL_GROUPS",
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


SIGNAL_GROUPS: tuple[tuple[str, str], ...] = (
    ("frame", "frame posteriors"),
    ("acoustic", "acoustic proxies"),
    ("scene", "scene classifiers"),
    ("diarization", "diarization"),
    ("asr", "ASR"),
    ("other", "other"),
)
"""Display order, grouped by what kind of evidence a signal is.

Alphabetical order interleaved a frame VAD, an acoustic proxy and a diarizer, which made the
figure unreadable: every row looked identical, so a reader could not tell what kind of claim
any of them was making. Grouping is what lets the eye compare like with like."""

_ROW_HEIGHT = {
    "spectrogram": 3.0,
    "scene": 1.4,
    "asr": 1.6,
    "frame": 1.2,
    "acoustic": 1.0,
    "diarization": 0.9,
    "level": 1.2,
    "other": 0.9,
}
"""Relative row heights. A uniform height gave a binary on/off row the same space as a
spectrogram, which wastes the figure on the rows carrying least information."""


def classify_signal(name: str) -> str:
    """Group a signal name by the kind of evidence it is.

    Read from the naming the harvester already uses rather than a hand-maintained list, so a
    new voter lands in the right group without a second place to update.
    """
    text = str(name)
    if text.startswith("frame_"):
        return "frame"
    if text.startswith("acoustic_"):
        return "acoustic"
    if text in ("ast", "yamnet") or text.startswith("scene"):
        return "scene"
    if "diar" in text or text.startswith("pyannote") or text.startswith("embedding_silhouette"):
        return "diarization"
    if any(tag in text.lower() for tag in ("whisper", "canary", "qwen", "asr", "granite")):
        return "asr"
    return "other"


def build_l1_signal_plot(
    out_dir: Path | str,
    *,
    signals: Mapping[str, Sequence[tuple[float, float]]],
    duration_s: float,
    waveform: np.ndarray | None = None,
    sampling_rate: int = 16000,
    series: Mapping[str, tuple[Sequence[float], Sequence[float]]] | None = None,
    words_by_model: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    speakers_by_model: Mapping[str, Sequence[tuple[float, float, str]]] | None = None,
    scene_by_classifier: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
    failed: Sequence[str] = (),
    title: str | None = None,
) -> Path:
    """Draw the L1 evidence figure: grouped rows, each in the display type its data warrants.

    Args:
        out_dir: Run directory.
        signals: ``{signal → spans}`` for signals whose claim is on/off.
        duration_s: Recording length.
        waveform: Mono samples for the spectrogram and level rows.
        sampling_rate: Sample rate of ``waveform``.
        series: ``{signal → (times, values)}`` for signals that report a continuous
            confidence. Plotted as a trace: rendering a frame posterior as an on/off bar
            discards everything it measured, which is why both VAD rows previously drew as
            solid full-width blocks.
        words_by_model: ``{asr_model → aligned words}``. Drawn *inside* that model's own row —
            a shared words row collided every token into an unreadable smear and, worse,
            attributed a transcript to no model in particular.
        speakers_by_model: ``{diar_model → [(start, end, cluster_id)]}``. Each cluster gets
            its own colour: a flat row makes a two-speaker conversation look identical to a
            one-speaker one, which is exactly what the speaker axis is arguing about.
        scene_by_classifier: ``{classifier → windows}`` as stacked category shares.
        failed: Signals that ran and errored. They keep a row, marked as failed: omitting them
            makes a failure indistinguishable from a signal that was never configured.
        title: Figure title.

    Returns:
        The written path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from senselab.audio.workflows.audio_analysis.layout import evidence_dir

    series = dict(series or {})
    words_by_model = dict(words_by_model or {})
    speakers_by_model = dict(speakers_by_model or {})
    # One stable colour per cluster across every row, so the same speaker reads the same way
    # in each diarizer's row — the comparison the figure exists to support.
    cluster_ids = sorted({c for spans in speakers_by_model.values() for _s, _e, c in spans})
    cluster_colour = {cluster: plt.get_cmap("tab10")(i % 10) for i, cluster in enumerate(cluster_ids)}
    scene = dict(scene_by_classifier or {})

    # Build the row plan: (kind, name) in group order, so like sits with like.
    plan: list[tuple[str, str]] = []
    named = set(signals) | set(series) | set(scene) | set(failed) | set(speakers_by_model)
    for kind, _label in SIGNAL_GROUPS:
        for name in sorted(n for n in named if classify_signal(n) == kind):
            plan.append((kind, name))
    if waveform is not None:
        plan.append(("spectrogram", "spectrogram"))
        plan.append(("level", "RMS dBFS"))

    heights = [_ROW_HEIGHT.get(kind, 1.0) for kind, _n in plan]
    fig, axs = plt.subplots(
        len(plan),
        1,
        figsize=(15, 0.42 * sum(heights) + 1.2),
        sharex=True,
        squeeze=False,
        gridspec_kw={"height_ratios": heights},
    )
    flat = [ax for (ax,) in axs]

    for ax, (kind, name) in zip(flat, plan):
        ax.set_xlim(0, max(duration_s, 1e-6))
        ax.set_yticks([])
        ax.set_ylabel(f"{name[:40]}\n[{kind}]", rotation=0, ha="right", va="center", fontsize=6)

        if name in failed:
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, "failed", transform=ax.transAxes, ha="center", va="center", fontsize=7, color="#aa3333")
            continue

        if kind == "spectrogram":
            spec, s_times, s_freqs = spectrogram_db(waveform, sampling_rate)
            ax.imshow(
                spec,
                origin="lower",
                aspect="auto",
                cmap="magma",
                extent=(0.0, float(s_times[-1] if s_times.size else duration_s), 0.0, float(s_freqs[-1])),
            )
            continue

        if kind == "level":
            times, levels = rms_dbfs_track(waveform, sampling_rate)
            if times.size:
                ax.plot(times, levels, linewidth=0.7, color="#333333")
                ax.set_ylim(DBFS_FLOOR, 3.0)
                ax.set_yticks([DBFS_FLOOR, 0])
                ax.tick_params(labelsize=5)
            ax.grid(True, axis="y", alpha=0.25)
            continue

        if kind == "scene" and name in scene:
            times, shares = scene_composition(scene[name], duration_s=duration_s)
            if shares.size:
                ax.stackplot(times, shares, linewidth=0)
                ax.set_ylim(0, 1)
            continue

        if name in series:
            times, values = series[name]
            ax.plot(np.asarray(times, dtype=float), np.asarray(values, dtype=float), linewidth=0.7, color="#1f6f4a")
            ax.set_ylim(0, 1.02)
            ax.set_yticks([0, 1])
            ax.tick_params(labelsize=5)
            ax.grid(True, axis="y", alpha=0.2)
            continue

        # Binary spans.
        ax.set_ylim(0, 1)
        for start, end in signals.get(name) or ():
            ax.add_patch(Rectangle((float(start), 0.15), max(1e-3, float(end) - float(start)), 0.7, color="#3b6ea5"))
        # ASR words go in the model's own row, so a transcript is attributed to who produced it.
        for word in words_by_model.get(name) or ():
            w_start = float(word.get("start", 0.0) or 0.0)
            w_end = float(word.get("end", w_start) or w_start)
            ax.text(
                (w_start + w_end) / 2.0,
                0.5,
                str(word.get("text") or ""),
                ha="center",
                va="center",
                fontsize=4.5,
                color="white",
                clip_on=True,
            )

    if cluster_ids:
        from matplotlib.patches import Patch

        fig.legend(
            handles=[Patch(color=cluster_colour[c], label=c) for c in cluster_ids],
            loc="lower right",
            ncol=min(8, len(cluster_ids)),
            fontsize=6,
            frameon=False,
        )
    flat[-1].set_xlabel("time (s)")
    fig.suptitle(title or "L1 signals (evidence)", fontsize=10)
    fig.tight_layout()

    dest = evidence_dir(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "signals.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
