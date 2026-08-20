"""Sweep a 40 ms rectangular window across the recording and plot HeAR's response.

HeAR's input is hard-fixed at 2 s, so a 40 ms excerpt cannot be scored on its own. Each window is
embedded, unmodified, at the centre of an otherwise-silent 2 s buffer, which makes the response
attributable to those 40 ms alone. See design.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.health_acoustics.api import detect_health_acoustic_events
from senselab.audio.tasks.preprocessing import resample_audios

SR = 16000
HEAR_SECONDS = 2.0
WIN_MS = 40.0
HOP_MS = 20.0

# From ground-truth-2026-08-18.md, human-verified.
EVENTS: List[Tuple[float, float, str]] = [
    (0.893, 1.095, "mouth"),
    (2.275, 3.496, "breath"),
    (5.308, 6.291, "breath"),
    (7.926, 8.494, "cough"),
    (9.610, 10.250, "cough"),
    (11.62, 13.20, "speech"),
]
PLOT_LABELS = ["Breathe", "Cough", "Speech", "Throat Clear"]
# All eight detector labels, for the raster. They are independent presence probabilities.
ALL_LABELS = ["Baby Cough", "Breathe", "Cough", "Laugh", "Sneeze", "Snore", "Speech", "Throat Clear"]


def windows(n_samples: int, win: int, hop: int) -> List[int]:
    """Return the start sample of every window that fits.

    Args:
        n_samples: Length of the recording in samples.
        win: Window length in samples.
        hop: Hop in samples.

    Returns:
        Window start offsets.
    """
    return list(range(0, max(0, n_samples - win) + 1, hop))


def embed(excerpt: torch.Tensor, buffer_len: int) -> torch.Tensor:
    """Centre ``excerpt`` in a zero buffer of ``buffer_len`` samples.

    Args:
        excerpt: One channel of samples, shape (1, win).
        buffer_len: Output length in samples.

    Returns:
        Shape (1, buffer_len), zero everywhere but the centred excerpt.
    """
    out = torch.zeros((1, buffer_len), dtype=excerpt.dtype)
    start = (buffer_len - excerpt.shape[-1]) // 2
    out[:, start : start + excerpt.shape[-1]] = excerpt
    return out


def main() -> int:
    """Run the sweep and write the figure.

    Returns:
        Process exit status.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--out", type=Path, default=Path("hear_sweep.png"))
    ap.add_argument("--win-ms", type=float, default=WIN_MS)
    ap.add_argument("--hop-ms", type=float, default=HOP_MS)
    ap.add_argument("--raster", action="store_true", help="heatmap over all eight labels")
    args = ap.parse_args()

    base = Audio(filepath=args.audio)
    if base.sampling_rate != SR:
        base = resample_audios([base], SR)[0]
    wav = base.waveform[:1]
    n = wav.shape[-1]
    win = int(round(args.win_ms / 1000 * SR))
    hop = int(round(args.hop_ms / 1000 * SR))
    buf = int(HEAR_SECONDS * SR)
    starts = windows(n, win, hop)
    print(f"{len(starts)} windows of {args.win_ms:.0f} ms at {args.hop_ms:.0f} ms hop", flush=True)

    clips = [Audio(waveform=embed(wav[:, s : s + win], buf), sampling_rate=SR) for s in starts]
    detections = detect_health_acoustic_events(clips)

    labels = ALL_LABELS if args.raster else PLOT_LABELS
    curves: Dict[str, List[float]] = {lab: [] for lab in labels}
    for det in detections:
        flat: Dict[str, float] = {}
        for w in det:
            for d in w.get("label_scores", []):
                for k, v in d.items():
                    flat[k] = max(flat.get(k, 0.0), float(v))
        for lab in labels:
            curves[lab].append(flat.get(lab, 0.0))

    centres = np.array([(s + win / 2) / SR for s in starts])
    duration = n / SR

    # A dedicated colorbar column, so attaching one does not shrink the raster relative to the
    # spectrogram -- the two panels must keep identical width for their time axes to line up.
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[60, 1], height_ratios=[1.1, 1])
    ax_s = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[1, 0], sharex=ax_s)
    cax = fig.add_subplot(gs[1, 1])
    fig.add_subplot(gs[0, 1]).axis("off")

    ax_s.specgram(wav[0].numpy(), NFFT=512, Fs=SR, noverlap=384, cmap="magma")
    ax_s.set_xlim(0.0, duration)
    ax_s.tick_params(labelbottom=False)  # shared with the panel below
    ax_s.set_ylabel("Hz")
    ax_s.set_title(
        f"{args.audio.name} — {args.win_ms:.0f} ms rectangular window swept at {args.hop_ms:.0f} ms hop, "
        f"each embedded alone in a silent 2 s HeAR buffer"
    )

    colours = {"mouth": "#7d5ba6", "breath": "#2e86ab", "cough": "#c1440e", "speech": "#3d7a3d"}
    for a, b, kind in EVENTS:
        for ax in (ax_s, ax_h):
            ax.axvspan(a, b, color=colours[kind], alpha=0.16, lw=0)
        ax_s.text((a + b) / 2, ax_s.get_ylim()[1] * 0.92, kind, ha="center", fontsize=8, color=colours[kind])

    if args.raster:
        order = sorted(labels, key=lambda lab: -max(curves[lab]))
        mat = np.array([curves[lab] for lab in order])
        half = (hop / SR) / 2 if len(centres) > 1 else 0.5
        edges_x = np.append(centres - half, centres[-1] + half)
        im = ax_h.pcolormesh(
            edges_x, np.arange(len(order) + 1), mat, cmap="viridis", vmin=0.0, vmax=1.0, shading="flat"
        )
        ax_h.set_yticks(np.arange(len(order)) + 0.5)
        ax_h.set_yticklabels([f"{lab}  ({max(curves[lab]):.2f})" for lab in order], fontsize=8)
        ax_h.invert_yaxis()
        ax_h.set_xlabel("time (s)")
        for a, b, _ in EVENTS:
            ax_h.axvline(a, color="w", lw=0.7, alpha=0.7)
            ax_h.axvline(b, color="w", lw=0.7, alpha=0.7, ls=":")
        fig.colorbar(im, cax=cax, label="probability")
    else:
        for lab in labels:
            ax_h.plot(centres, curves[lab], lw=1.3, label=lab)
        ax_h.axhline(0.5, color="k", ls=":", lw=0.8)
        ax_h.set_ylim(-0.02, 1.02)
        ax_h.set_ylabel("HeAR probability")
        ax_h.set_xlabel("time (s)")
        ax_h.legend(loc="upper left", fontsize=8, ncols=4)
        ax_h.grid(alpha=0.25)
        cax.axis("off")

    ax_h.set_xlim(0.0, duration)
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}", flush=True)

    for lab in labels:
        arr = np.array(curves[lab])
        hits = [f"{centres[i]:.2f}" for i in np.where(arr > 0.5)[0]]
        print(
            f"{lab:14s} max={arr.max():.3f}  >0.5 at {len(hits)} windows"
            + (f": {', '.join(hits[:12])}{' ...' if len(hits) > 12 else ''}" if hits else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
