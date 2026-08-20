"""Raster YAMNet's top-k AudioSet classes over its own native windows.

No sweeping and no silent-buffer construction: YAMNet has a fixed 0.96 s window on a 0.48 s hop and
no minimum-duration constraint, so it is run as shipped. Classes are ranked by their peak over the
whole recording, which answers "which classes does this file elicit at all". See design.md.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.classification.api import classify_audios

YAMNET_ALL_LABELS = 521

EVENTS: List[Tuple[float, float, str]] = [
    (0.893, 1.095, "mouth"),
    (2.275, 3.496, "breath"),
    (5.308, 6.291, "breath"),
    (7.926, 8.494, "cough"),
    (9.610, 10.250, "cough"),
    (11.62, 13.20, "speech"),
]


def main() -> int:
    """Run YAMNet, rank classes by peak, and raster the top k.

    Returns:
        Process exit status.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--out", type=Path, default=Path("yamnet_raster.png"))
    args = ap.parse_args()

    base = Audio(filepath=args.audio)
    duration = base.waveform.shape[-1] / base.sampling_rate
    result = classify_audios([base], model="yamnet", top_k=YAMNET_ALL_LABELS)[0]
    print(f"{len(result)} native windows over {duration:.3f}s", flush=True)

    per_window: List[Dict[str, float]] = []
    starts, ends = [], []
    for w in result:
        flat = {k: float(v) for d in w.get("label_scores", []) for k, v in d.items()}
        per_window.append(flat)
        starts.append(float(w["start"]))
        ends.append(float(w["end"]))

    peaks: Dict[str, float] = {}
    for flat in per_window:
        for k, v in flat.items():
            peaks[k] = max(peaks.get(k, 0.0), v)
    top = sorted(peaks, key=lambda k: -peaks[k])[: args.top]
    print(f"top {args.top} by peak: " + ", ".join(f"{k} ({peaks[k]:.3f})" for k in top), flush=True)

    mat = np.array([[flat.get(lab, 0.0) for flat in per_window] for lab in top])
    edges_x = np.append(np.array(starts), ends[-1])

    fig = plt.figure(figsize=(14, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[60, 1], height_ratios=[1.1, 1])
    ax_s = fig.add_subplot(gs[0, 0])
    ax_y = fig.add_subplot(gs[1, 0], sharex=ax_s)
    cax = fig.add_subplot(gs[1, 1])
    fig.add_subplot(gs[0, 1]).axis("off")

    wav = base.waveform[0].numpy()
    ax_s.specgram(wav, NFFT=512, Fs=base.sampling_rate, noverlap=384, cmap="magma")
    ax_s.set_xlim(0.0, duration)
    ax_s.tick_params(labelbottom=False)
    ax_s.set_ylabel("Hz")
    ax_s.set_title(
        f"{args.audio.name} — YAMNet, native 0.96 s window / 0.48 s hop, "
        f"top {args.top} of 521 classes by peak over the recording"
    )

    colours = {"mouth": "#7d5ba6", "breath": "#2e86ab", "cough": "#c1440e", "speech": "#3d7a3d"}
    for a, b, kind in EVENTS:
        ax_s.axvspan(a, b, color=colours[kind], alpha=0.16, lw=0)
        ax_s.text((a + b) / 2, ax_s.get_ylim()[1] * 0.92, kind, ha="center", fontsize=8, color=colours[kind])

    im = ax_y.pcolormesh(edges_x, np.arange(len(top) + 1), mat, cmap="viridis", vmin=0.0, vmax=1.0, shading="flat")
    ax_y.set_yticks(np.arange(len(top)) + 0.5)
    ax_y.set_yticklabels([f"{lab}  ({peaks[lab]:.2f})" for lab in top], fontsize=8)
    ax_y.invert_yaxis()
    ax_y.set_xlim(0.0, duration)
    ax_y.set_xlabel("time (s)")
    for a, b, _ in EVENTS:
        ax_y.axvline(a, color="w", lw=0.7, alpha=0.7)
        ax_y.axvline(b, color="w", lw=0.7, alpha=0.7, ls=":")
    fig.colorbar(im, cax=cax, label="score")
    fig.savefig(args.out, dpi=140)
    print(f"wrote {args.out}", flush=True)

    for lab in top:
        arr = np.array([flat.get(lab, 0.0) for flat in per_window])
        hi = np.where(arr > 0.5)[0]
        spans = ", ".join(f"{starts[i]:.2f}-{ends[i]:.2f}" for i in hi[:8])
        print(
            f"{lab:34s} peak={arr.max():.3f}  >0.5 in {len(hi)}/{len(arr)} windows"
            + (f": {spans}{' ...' if len(hi) > 8 else ''}" if len(hi) else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
