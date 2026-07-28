"""Visual timeline for an adaptive-loop run (best-effort sidecar, like analyze_audio's).

Reads the persisted round artifacts (no live state needed) and renders
``final/timeline.png``:

1. spectrogram — the acoustic evidence every row below is derived from, so a
   reviewer can see *why* a span is uncertain (noise, silence, overlap) rather
   than only that it is;
2. ground-truth segments (when an LS export is provided) — untranscribed spans hatched;
3. presence — final p_voice + uncertainty band;
4. identity — round-1 vs final uncertainty, GT speaker boundaries dashed;
5. utterance — round-1 vs final uncertainty, proposed regions, fired
   interventions, irreducible buckets hatched;
6. fused words — text colored by confidence (green→red), speaker ticks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_adaptive_timeline(out_dir: Path, *, gt_path: Path | None = None, title: str = "") -> Path | None:
    """Render ``<out_dir>/final/timeline.png``; returns the path (None on failure)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import Rectangle

    out_dir = Path(out_dir)
    final = out_dir / "final"
    transcript = json.loads((final / "transcript.json").read_text())
    stream = transcript.get("stream", "raw_16k")
    iterations = json.loads((final / "iterations.json").read_text())["entries"]
    convergence = json.loads((final / "convergence.json").read_text())

    rounds_dir = out_dir / "rounds"
    round_ids = sorted(int(p.name) for p in rounds_dir.iterdir() if p.name.isdigit())
    first_r, last_r = round_ids[0], round_ids[-1]

    def _belief(round_idx: int, axis: str) -> "pd.DataFrame":
        df = pd.read_parquet(rounds_dir / str(round_idx) / "belief" / f"{axis}.parquet")
        return df[df["stream"] == stream].sort_values("start")

    pres, ident_0, ident_k = _belief(last_r, "presence"), _belief(first_r, "identity"), _belief(last_r, "identity")
    utt_0, utt_k = _belief(first_r, "utterance"), _belief(last_r, "utterance")
    duration = float(pres["end"].max()) if len(pres) else 5.0

    gt = None
    if gt_path is not None:
        from senselab.audio.workflows.audio_analysis.adaptive.evaluate import load_ls_ground_truth

        gt = load_ls_ground_truth(gt_path)

    fig, axes = plt.subplots(6, 1, figsize=(14, 13), sharex=True, height_ratios=[1.3, 0.8, 1.2, 1.2, 1.4, 1.0])
    ax_spec, ax_gt, ax_p, ax_i, ax_u, ax_w = axes

    # ── row 0: spectrogram ──────────────────────────────────────────────
    # The acoustic evidence every row below is derived from. Put first so a
    # reviewer can see *why* a span is uncertain (noise, silence, overlap)
    # rather than only that it is.
    _draw_spectrogram(ax_spec, out_dir, duration)

    # ── row 1: ground truth ─────────────────────────────────────────────
    ax_gt.set_ylabel("ground\ntruth", rotation=0, ha="right", va="center")
    ax_gt.set_ylim(0, 1)
    ax_gt.set_yticks([])
    if gt:
        speakers = sorted({s["speaker"] for s in gt["segments"] if s["speaker"]})
        cmap = plt.get_cmap("tab10")
        colors = {spk: cmap(i % 10) for i, spk in enumerate(speakers)}
        for seg in gt["segments"]:
            hatch = None if seg.get("text") else "///"
            ax_gt.add_patch(
                Rectangle(
                    (seg["start"], 0.15),
                    seg["end"] - seg["start"],
                    0.7,
                    facecolor=colors.get(seg["speaker"], "grey"),
                    alpha=0.55,
                    hatch=hatch,
                    edgecolor="black",
                    linewidth=0.6,
                )
            )
            label = seg["speaker"] or "?"
            if not seg.get("text"):
                label += " (untranscribed)"
            ax_gt.text((seg["start"] + seg["end"]) / 2, 0.5, label, ha="center", va="center", fontsize=7, rotation=0)
    else:
        ax_gt.text(duration / 2, 0.5, "no ground truth provided", ha="center", va="center", fontsize=8, alpha=0.6)

    # ── row 2: presence ─────────────────────────────────────────────────
    mids_p = (pres["start"] + pres["end"]) / 2
    ax_p.plot(mids_p, pres["p_voice"], color="tab:blue", lw=1.2, label="p_voice (final)")
    ax_p.fill_between(
        mids_p, 0, pres["aggregated_uncertainty"].fillna(0), color="tab:red", alpha=0.18, label="uncertainty"
    )
    ax_p.axhline(0.5, color="grey", lw=0.6, ls=":")
    ax_p.set_ylabel("presence", rotation=0, ha="right", va="center")
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.legend(loc="upper right", fontsize=7, ncol=2)

    # ── row 3: identity ─────────────────────────────────────────────────
    _step(ax_i, ident_0, color="silver", label=f"round {first_r}")
    _step(ax_i, ident_k, color="tab:purple", label=f"round {last_r} (final)")
    if gt:
        for b in [s["start"] for s in gt["segments"][1:]]:
            ax_i.axvline(b, color="black", lw=0.8, ls="--", alpha=0.6)
    ax_i.set_ylabel("identity\nuncertainty", rotation=0, ha="right", va="center")
    ax_i.set_ylim(-0.02, 1.05)
    ax_i.legend(loc="lower right", fontsize=7, ncol=2, title="GT boundaries dashed", title_fontsize=6)

    # ── row 4: utterance + regions + interventions ─────────────────────
    _step(ax_u, utt_0, color="silver", label=f"round {first_r}")
    _step(ax_u, utt_k, color="tab:red", label=f"round {last_r} (final)")
    for _, row in utt_k.iterrows():
        if row.get("status") == "irreducible":
            ax_u.add_patch(
                Rectangle(
                    (row["start"], 0),
                    row["end"] - row["start"],
                    1.0,
                    facecolor="none",
                    hatch="xx",
                    edgecolor="tab:red",
                    linewidth=0.0,
                    alpha=0.35,
                )
            )
    seen_regions = set()
    for r_idx in round_ids:
        regions_file = rounds_dir / str(r_idx) / "regions.json"
        if not regions_file.exists():
            continue
        for reg in json.loads(regions_file.read_text()):
            if reg["axis"] != "utterance" or reg["stream"] != stream:
                continue
            span = (round(reg["core_start"], 3), round(reg["core_end"], 3))
            if span in seen_regions:
                continue
            seen_regions.add(span)
            ax_u.axvspan(reg["core_start"], reg["core_end"], color="orange", alpha=0.12)
            ax_u.annotate(
                reg["region_id"], (reg["core_start"], 1.02), fontsize=6, color="darkorange", annotation_clip=False
            )
    y_note = 0.92
    for e in iterations:
        if e["status"] != "fired" or not e.get("region_id"):
            continue
        reg_delta = next(iter((e.get("delta") or {}).values()), None)
        d_txt = f" Δ{reg_delta['delta']:+.2f}" if reg_delta else ""
        x = _region_mid(e, rounds_dir, axis="utterance")
        if x is not None:
            ax_u.annotate(
                f"{e['rule'].split('_')[0]} r{e['round']}{d_txt}",
                (x, y_note),
                fontsize=6.5,
                ha="center",
                color="darkred",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "darkred", "lw": 0.5, "alpha": 0.85},
            )
            y_note -= 0.14
    ax_u.set_ylabel("utterance\nuncertainty", rotation=0, ha="right", va="center")
    ax_u.set_ylim(-0.02, 1.08)
    ax_u.axhline(0.66, color="tab:red", lw=0.6, ls=":", alpha=0.7)
    ax_u.axhline(0.33, color="tab:green", lw=0.6, ls=":", alpha=0.7)
    ax_u.legend(
        loc="upper left", fontsize=7, ncol=2, title="θ_high/θ_low dotted; irreducible hatched", title_fontsize=6
    )

    # ── row 5: fused words ──────────────────────────────────────────────
    cmap_conf = plt.get_cmap("RdYlGn")
    for w in transcript["words"]:
        mid = (w["start"] + w["end"]) / 2
        conf = float(w.get("confidence") or 0.0)
        ax_w.add_patch(
            Rectangle(
                (w["start"], 0.25),
                max(0.02, w["end"] - w["start"]),
                0.5,
                facecolor=cmap_conf(conf),
                alpha=0.8,
                edgecolor="black",
                linewidth=0.4,
            )
        )
        ax_w.text(mid, 0.87, w["text"], ha="center", va="center", fontsize=7, rotation=35)
        ax_w.text(mid, 0.5, f"{conf:.2f}", ha="center", va="center", fontsize=5.5)
        if w.get("alternates"):
            alt_txt = "|".join(a["text"] for a in w["alternates"][:2])
            ax_w.text(mid, 0.12, alt_txt, ha="center", fontsize=5, color="grey")
    ax_w.set_ylabel("fused words\n(conf color)", rotation=0, ha="right", va="center")
    ax_w.set_ylim(0, 1.1)
    ax_w.set_yticks([])
    ax_w.set_xlabel("time (s)")
    ax_w.set_xlim(0, duration)

    n_fired = sum(1 for e in iterations if e["status"] == "fired")
    fig.suptitle(
        (title or out_dir.name) + f"  ·  stream={stream}  ·  run_state={convergence['run_state']}  ·  "
        f"{n_fired} interventions  ·  policy={str(convergence.get('policy_hash'))[:8]}",
        fontsize=10,
    )
    fig.align_ylabels(axes)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    dest = final / "timeline.png"
    fig.savefig(dest, dpi=150)
    plt.close(fig)
    return dest


def _draw_spectrogram(ax: Any, out_dir: Path, duration: float) -> None:  # noqa: ANN401
    """Render the run's input audio as a dB-scaled STFT on ``ax``.

    Best-effort and self-contained: the audio path comes from the run's
    ``summary.json`` (``input_audio``), re-rooted if the run came from another
    machine. Any failure leaves an annotated empty axis rather than losing the
    whole figure — the spectrogram is context, not the point of the plot.
    """
    ax.set_ylabel("spectrogram\n(kHz)", rotation=0, ha="right", va="center")
    try:
        import numpy as np

        from senselab.audio.workflows.audio_analysis.adaptive.loop import _resolve_input_audio

        summary = json.loads((out_dir / "summary.json").read_text())
        path = _resolve_input_audio(summary.get("input_audio"), out_dir)
        if not path:
            raise FileNotFoundError("input_audio not recorded in summary.json")

        from senselab.audio.data_structures import Audio
        from senselab.audio.tasks.preprocessing import downmix_audios_to_mono, resample_audios

        audio = Audio(filepath=str(path))
        if audio.waveform.shape[0] > 1:
            audio = downmix_audios_to_mono([audio])[0]
        audio = resample_audios([audio], 16000)[0]
        y = audio.waveform.squeeze().detach().cpu().numpy()

        n_fft, hop = 512, 128
        # magnitude STFT via numpy so the plot adds no new dependency
        win = np.hanning(n_fft)
        n_frames = max(1, 1 + (len(y) - n_fft) // hop)
        frames = np.stack([y[i * hop : i * hop + n_fft] * win for i in range(n_frames)], axis=1)
        mag = np.abs(np.fft.rfft(frames, n=n_fft, axis=0))
        db = 20.0 * np.log10(mag + 1e-8)
        db = np.maximum(db, db.max() - 70.0)  # 70 dB dynamic range

        ax.imshow(
            db,
            origin="lower",
            aspect="auto",
            extent=(0.0, len(y) / 16000.0, 0.0, 8.0),
            cmap="magma",
        )
        ax.set_ylim(0, 8)
        ax.set_yticks([0, 4, 8])
    except Exception as exc:  # noqa: BLE001 — context row, never fatal
        ax.set_yticks([])
        ax.set_ylim(0, 1)
        ax.text(
            duration / 2,
            0.5,
            f"spectrogram unavailable ({type(exc).__name__})",
            ha="center",
            va="center",
            fontsize=8,
            alpha=0.6,
        )


def _step(ax: Any, df: Any, *, color: str, label: str) -> None:  # noqa: ANN401
    """Step-plot per-bucket uncertainty (buckets may overlap; drawn at midpoints)."""
    if not len(df):
        return
    mids = (df["start"] + df["end"]) / 2
    ax.plot(
        mids, df["aggregated_uncertainty"], drawstyle="steps-mid", color=color, lw=1.4, label=label, marker="o", ms=2.5
    )


def _region_mid(entry: dict[str, Any], rounds_dir: Path, *, axis: str | None = None) -> float | None:
    """Midpoint of the region an iteration entry acted on; optionally filter by region axis."""
    regions_file = rounds_dir / str(entry["round"]) / "regions.json"
    if not regions_file.exists():
        return None
    for reg in json.loads(regions_file.read_text()):
        if reg.get("region_id") == entry.get("region_id"):
            if axis is not None and reg.get("axis") != axis:
                return None
            return (reg["core_start"] + reg["core_end"]) / 2
    return None
