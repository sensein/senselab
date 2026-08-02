"""Visual timeline for an adaptive-loop run (best-effort sidecar, like analyze_audio's).

Reads the persisted round artifacts (no live state needed) and renders
``final/timeline.png``:

1. spectrogram — the acoustic evidence every row below is derived from, so a
   reviewer can see *why* a span is uncertain (noise, silence, overlap) rather
   than only that it is;
2. ground-truth segments (when an LS export is provided) — untranscribed spans hatched;
3. speech_presence — final p_voice + uncertainty band;
4. speaker — round-1 vs final uncertainty, GT speaker boundaries dashed;
5. asr — round-1 vs final uncertainty, proposed regions, fired
   interventions, irreducible buckets hatched;
6. fused words — text colored by confidence (green→red), speaker ticks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from senselab.audio.workflows.audio_analysis.layout import belief_dir, evidence_dir, final_dir


def build_adaptive_timeline(
    out_dir: Path,
    *,
    transcript: dict[str, Any],
    gt_path: Path | None = None,
    title: str = "",
) -> Path | None:
    """Render ``<out_dir>/final/timeline.png``; returns the path (None on failure).

    ``transcript`` is required. The figure renders the converged answer, and its caller has just
    produced that answer, so it hands it over; the fallback that read ``final/transcript.json``
    when the argument was omitted made a deliverable an input to the stage that writes the
    deliverable next to it — and, being a default, it was the path the standalone driver took.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.patches import Rectangle

    out_dir = Path(out_dir)
    final = final_dir(out_dir)
    # Belief artifacts (posterior, speech_presence, convergence) are level 2; the deliverables
    # (transcript, diarization, timeline, summary) stay in final/. Different questions:
    # "what do we believe" is per bucket and per round, "what do we hand over" is one answer.
    belief = belief_dir(out_dir)
    final.mkdir(parents=True, exist_ok=True)
    belief.mkdir(parents=True, exist_ok=True)
    stream = transcript.get("stream", "raw_16k")
    iterations = json.loads((belief / "iterations.json").read_text())["entries"]
    convergence = json.loads((belief / "convergence.json").read_text())

    rounds_dir = belief_dir(out_dir) / "rounds"
    round_ids = sorted(int(p.name) for p in rounds_dir.iterdir() if p.name.isdigit())
    first_r, last_r = round_ids[0], round_ids[-1]

    def _belief(round_idx: int, axis: str) -> "pd.DataFrame":
        # No stream filter: the belief file holds one row per bucket, already folded across
        # passes by the writer under a recorded policy. Filtering here picked one pass's reading
        # and called it the run's, which is a fold nobody wrote down.
        return pd.read_parquet(rounds_dir / str(round_idx) / "belief" / f"{axis}.parquet").sort_values("start")

    pres, ident_0, ident_k = _belief(last_r, "speech_presence"), _belief(first_r, "speaker"), _belief(last_r, "speaker")
    utt_0, utt_k = _belief(first_r, "asr"), _belief(last_r, "asr")
    duration = float(pres["end"].max()) if len(pres) else 5.0

    gt = None
    if gt_path is not None:
        from senselab.audio.workflows.audio_analysis.adaptive.evaluate import load_ls_ground_truth

        gt = load_ls_ground_truth(gt_path)

    fig, axes = plt.subplots(
        8, 1, figsize=(14, 15.5), sharex=True, height_ratios=[1.3, 0.8, 0.6, 1.2, 1.2, 0.9, 1.4, 1.0]
    )
    ax_spec, ax_gt, ax_mask, ax_p, ax_i, ax_spk, ax_u, ax_w = axes

    # ── row 0: spectrogram ──────────────────────────────────────────────
    # The acoustic evidence every row below is derived from. Put first so a
    # reviewer can see *why* a span is uncertain (noise, silence, overlap)
    # rather than only that it is.
    _draw_spectrogram(ax_spec, out_dir, duration)

    # ── row 2: background mask ──────────────────────────────────────────
    # Sits directly under the ground truth and above the axes, because it says
    # *where the background findings below can be trusted at all*: a target-free
    # span has no foreground to leak, so a claim there does not depend on
    # suppression depth. Reading an uncertainty row without knowing which spans
    # were target-free invites treating a leakage artifact as a finding.
    _draw_background_mask(ax_mask, out_dir, duration)

    # ── per-speaker speech_presence ────────────────────────────────────────────
    # Directly under the speaker axis, because it is what that axis's number could not
    # say: whether a high value means "we disagree about who spoke" or "we disagree about
    # how many people are here". One lane per hypothesised speaker makes the count visible
    # at a glance, and the header carries the posterior when it is multi-modal.
    _draw_per_speaker(ax_spk, out_dir, duration)

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

    # ── row 2: speech_presence ─────────────────────────────────────────────────
    mids_p = (pres["start"] + pres["end"]) / 2
    ax_p.plot(mids_p, pres["p_voice"], color="tab:blue", lw=1.2, label="p_voice (final)")
    ax_p.fill_between(mids_p, 0, pres["uncertainty"].fillna(0), color="tab:red", alpha=0.18, label="uncertainty")
    ax_p.axhline(0.5, color="grey", lw=0.6, ls=":")
    ax_p.set_ylabel("speech_presence", rotation=0, ha="right", va="center")
    ax_p.set_ylim(-0.02, 1.02)
    ax_p.legend(loc="upper right", fontsize=7, ncol=2)

    # ── row 3: speaker ─────────────────────────────────────────────────
    _step(ax_i, ident_0, color="silver", label=f"round {first_r}")
    _step(ax_i, ident_k, color="tab:purple", label=f"round {last_r} (final)")
    if gt:
        for b in [s["start"] for s in gt["segments"][1:]]:
            ax_i.axvline(b, color="black", lw=0.8, ls="--", alpha=0.6)
    # The L2 fused speaker axis, beside the belief store's. Two different numbers sharing a name
    # (item 27); the fused one is the only one cross-axis coupling can reach, so omitting it makes
    # coupling look broken when it is simply not what this row was drawing.
    fused_spk = _fused_axis(out_dir, "speaker")
    if fused_spk is not None and len(fused_spk):
        mids = (fused_spk["start"] + fused_spk["end"]) / 2
        ax_i.plot(
            mids,
            fused_spk["uncertainty"],
            color="tab:green",
            lw=1.1,
            ls="-.",
            label="L2 fused (coupled)",
        )
    ax_i.set_ylabel("speaker\nuncertainty", rotation=0, ha="right", va="center")
    ax_i.set_ylim(-0.02, 1.05)
    ax_i.legend(
        loc="lower right",
        fontsize=7,
        ncol=3,
        title="belief store vs L2 fused — different quantities (item 27)",
        title_fontsize=6,
    )

    # ── row 4: asr + regions + interventions ─────────────────────
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
            # No stream filter: a region is a span of the recording, proposed from an axis that
            # already folds across passes. Filtering by one dropped every region on the run's
            # other pass from the figure, which is the same collapse the store used to force on
            # every reader — invisibly, because a missing overlay looks like a quiet stretch.
            if reg["axis"] != "asr":
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
        x = _region_mid(e, rounds_dir, axis="asr")
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
    ax_u.set_ylabel("asr\nuncertainty", rotation=0, ha="right", va="center")
    ax_u.set_ylim(-0.02, 1.08)
    ax_u.axhline(0.66, color="tab:red", lw=0.6, ls=":", alpha=0.7)
    ax_u.axhline(0.33, color="tab:green", lw=0.6, ls=":", alpha=0.7)
    ax_u.legend(
        loc="upper left", fontsize=7, ncol=2, title="θ_high/θ_low dotted; irreducible hatched", title_fontsize=6
    )

    # ── row 5: fused words ──────────────────────────────────────────────
    cmap_conf = plt.get_cmap("RdYlGn")
    # Word labels cycle through three text lanes (word i -> lane i % 3). Speech runs at roughly
    # three words a second, so at any readable font size consecutive labels overlap when they share
    # one lane; rotating them 35 degrees traded one kind of illegibility for another. Staggering
    # gives each label three words' worth of horizontal room and lets the text sit upright.
    text_lanes = (0.78, 0.50, 0.22)
    for idx, w in enumerate(transcript["words"]):
        mid = (w["start"] + w["end"]) / 2
        conf = float(w.get("confidence") or 0.0)
        lane = idx % len(text_lanes)
        # Confidence lives on the label's own background. The separate box-plus-number below each
        # word encoded the same quantity three ways — box colour, printed value, and position —
        # while the staggering had already pulled the label away from the box it belonged to, so
        # the colour stopped reading as that word's.
        ax_w.text(
            mid,
            text_lanes[lane],
            w["text"],
            ha="center",
            va="center",
            fontsize=7,
            zorder=2,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": cmap_conf(conf),
                "edgecolor": "black",
                "linewidth": 0.3,
                "alpha": 0.9,
            },
        )
        if w.get("alternates"):
            alt_txt = "|".join(a["text"] for a in w["alternates"][:2])
            ax_w.text(mid, text_lanes[lane] - 0.09, alt_txt, ha="center", va="center", fontsize=5, color="grey")
    # A colorbar, because a colour scale with no key requires the reader to guess which end is
    # confident. Horizontal and inset so it costs no row height.
    if transcript["words"]:
        cax = ax_w.inset_axes((0.35, 1.02, 0.3, 0.05))
        fig.colorbar(
            plt.cm.ScalarMappable(norm=plt.Normalize(0.0, 1.0), cmap=cmap_conf),
            cax=cax,
            orientation="horizontal",
        )
        cax.tick_params(labelsize=5, length=2, pad=1)
        cax.set_title("word confidence", fontsize=6, pad=2)
    ax_w.set_ylabel("fused words", rotation=0, ha="right", va="center")
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


_MASK_STATE_STYLE = {
    # Green reads as "usable" and grey as "unknown" without needing the legend;
    # target-active is deliberately muted so the eye lands on the usable spans.
    "target_free": ("#2e7d32", "target-free"),
    # Target absent, something else audible — the state a background claim is actually *made*
    # from, since a silent stretch characterises nothing. It was missing from this table, so it
    # fell through to the ``indeterminate`` default and every such region rendered as "cannot
    # tell": on a clip whose only finding was a 5.5 s non-target span, the mask row showed
    # nothing but grey. Blue rather than another green, because it answers a different question
    # from ``target_free`` — the two are kept apart everywhere else for the same reason.
    "nontarget_active": ("#1565c0", "non-target active"),
    "indeterminate": ("#9e9e9e", "cannot tell"),
    "target_active": ("#c8c8c8", "target active"),
}


def _draw_per_speaker(ax: Any, out_dir: Path, duration: float) -> None:  # noqa: ANN401 — matplotlib Axes
    """Draw one speech_presence lane per hypothesised speaker, with the count posterior in view.

    This is the row the single speaker scalar could not provide. A high speaker
    uncertainty is ambiguous between disagreement about *who* spoke and disagreement about
    *how many* people are present; lanes plus the posterior separate the two.
    """
    import json as _json

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    from senselab.utils.data_structures.logging import logger

    ax.set_ylabel("per-speaker\npresence", rotation=0, ha="right", va="center", fontsize=9)
    ax.set_xlim(0, duration)
    ax.set_yticks([])

    final = final_dir(out_dir)
    belief = belief_dir(out_dir)
    final.mkdir(parents=True, exist_ok=True)
    belief.mkdir(parents=True, exist_ok=True)
    speakers_path, speech_presence_path = belief / "speakers.json", belief / "per_speaker_presence.parquet"
    if not speakers_path.exists() or not speech_presence_path.exists():
        ax.text(
            0.5,
            0.5,
            "no per-speaker speaker output",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#888888",
        )
        ax.set_ylim(0, 1)
        return

    try:
        import pandas as pd

        doc = _json.loads(speakers_path.read_text())
        rows = pd.read_parquet(speech_presence_path).to_dict("records")
    except Exception as exc:  # noqa: BLE001 — a plot must never fail a run
        logger.debug("per-speaker outputs unreadable: %s", exc)
        ax.set_ylim(0, 1)
        return

    speaker_ids = sorted({str(r["speaker_id"]) for r in rows}) or [
        str(s["speaker_id"]) for s in doc.get("speakers", [])
    ]
    if not speaker_ids:
        ax.text(
            0.5,
            0.5,
            "no speaker hypotheses",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#888888",
        )
        ax.set_ylim(0, 1)
        return

    lanes = {sid: i for i, sid in enumerate(speaker_ids)}
    ax.set_ylim(-0.5, len(speaker_ids) - 0.5)
    ax.set_yticks(list(lanes.values()))
    ax.set_yticklabels(speaker_ids, fontsize=7)
    cmap = plt.get_cmap("tab10")

    for row in rows:
        conf = row.get("speech_presence_confidence")
        if conf is None:
            continue
        lane = lanes.get(str(row["speaker_id"]))
        if lane is None:
            continue
        unc = float(row.get("speech_presence_uncertainty") or 0.0)
        ax.add_patch(
            Rectangle(
                (float(row["start"]), lane - 0.35),
                max(float(row["end"]) - float(row["start"]), 1e-6),
                0.7,
                facecolor=cmap(lane % 10),
                edgecolor="none",
                # Height encodes confidence, transparency encodes uncertainty about it --
                # a faint short bar is a speaker the analysis is unsure about twice over.
                alpha=max(0.12, float(conf) * (1.0 - 0.6 * unc)),
            )
        )

    cp = doc.get("count_posterior") or {}
    probs = cp.get("probabilities") or {}
    if probs:
        summary = "  ".join(f"{k}:{float(v):.2f}" for k, v in sorted(probs.items()))
        flag = "  MULTI-MODAL" if cp.get("is_multimodal") else ""
        ax.text(
            0.005,
            0.97,
            f"speaker-count posterior  {summary}{flag}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.5,
            color="#b00020" if cp.get("is_multimodal") else "#444444",
        )


def _draw_background_mask(ax: Any, out_dir: Path, duration: float) -> None:  # noqa: ANN401 — matplotlib Axes
    """Draw the background mask as a four-state strip, with uncertainty as alpha.

    Absent mask parquet leaves an explicitly labelled empty row rather than a silently
    blank one — "no mask was produced" and "the mask was empty" are different facts, and a
    blank strip would conflate them.
    """
    from matplotlib.patches import Rectangle

    from senselab.utils.data_structures.logging import logger

    ax.set_ylabel("background\nmask", rotation=0, ha="right", va="center", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlim(0, duration)

    # ``L2/background_mask.parquet`` — one named path. Every previous form of this read was a
    # glob, and every one of them drifted: first against the flat layout, then one level short of
    # ``L1/<pass>/``. A glob that matches nothing is indistinguishable from a stage that produced
    # nothing, and this row said "no background mask" on runs whose mask had found regions.
    rows: list[dict[str, Any]] = []
    candidate = belief_dir(out_dir) / "background_mask.parquet"
    if candidate.exists():
        try:
            import pandas as pd

            rows = pd.read_parquet(candidate).to_dict("records")
        except Exception as exc:  # noqa: BLE001 — a plot must not fail a run
            logger.debug("background mask unreadable at %s: %s", candidate, exc)

    if not rows:
        ax.text(
            0.5,
            0.5,
            "no background mask",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#888888",
        )
        return

    seen: set[str] = set()
    for row in rows:
        state = str(row.get("state", "indeterminate"))
        color, label = _MASK_STATE_STYLE.get(state, _MASK_STATE_STYLE["indeterminate"])
        start, end = float(row["start"]), float(row["end"])
        # Confident spans read solid; uncertain ones fade. A washed-out green is a
        # region whose "usable" claim is itself shaky.
        unc = float(row.get("uncertainty") or 0.0)
        alpha = 0.85 if state == "target_active" else max(0.15, 0.85 * (1.0 - unc))
        ax.add_patch(
            Rectangle(
                (start, 0.2),
                max(end - start, 1e-6),
                0.6,
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                label=label if label not in seen else None,
            )
        )
        seen.add(label)
        if float(row.get("guard_trimmed_s") or 0.0) > 0.0:
            # Hatch what the guard interval removed, so a shrunken mask is visibly
            # the guard's doing rather than an absence of quiet audio.
            ax.add_patch(
                Rectangle(
                    (start, 0.2),
                    max(end - start, 1e-6),
                    0.6,
                    facecolor="none",
                    edgecolor="#555555",
                    hatch="///",
                    linewidth=0.0,
                    alpha=0.5,
                )
            )
    if seen:
        ax.legend(loc="upper right", fontsize=6, ncol=3, framealpha=0.6)


def _draw_spectrogram(ax: Any, out_dir: Path, duration: float) -> None:  # noqa: ANN401
    """Render the run's input audio as a dB-scaled STFT on ``ax``.

    Best-effort and self-contained: the audio path comes from the run's
    ``L1/passes.json`` (``input_audio``), re-rooted if the run came from another
    machine. Any failure leaves an annotated empty axis rather than losing the
    whole figure — the spectrogram is context, not the point of the plot.
    """
    ax.set_ylabel("spectrogram\n(kHz)", rotation=0, ha="right", va="center")
    try:
        import numpy as np

        from senselab.audio.workflows.audio_analysis.adaptive.loop import _resolve_input_audio

        # The input path is evidence about the run, not a deliverable.
        summary = json.loads((evidence_dir(out_dir) / "passes.json").read_text())
        path = _resolve_input_audio(summary.get("input_audio"), out_dir)
        if not path:
            raise FileNotFoundError("input_audio not recorded in L1/passes.json")

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
    ax.plot(mids, df["uncertainty"], drawstyle="steps-mid", color=color, lw=1.4, label=label, marker="o", ms=2.5)


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


def _fused_axis(out_dir: Path, axis: str) -> Any:  # noqa: ANN401 — pd.DataFrame or None
    """The L2 fused axis for ``axis``, from the last round that wrote one.

    A *different quantity* from the belief store this figure otherwise draws, and drawn beside it
    deliberately (register item 27). They share a name and nothing else: different grids, different
    provenance — the belief store ingests L1's per-pass axis folds — and only the fused one is
    reachable by D-11's cross-axis coupling. Showing one and labelling it "speaker uncertainty"
    invites the reading that per-speaker presence failed to move it, when in fact the coupled
    quantity was never on the figure.

    **This function is scaffolding for a defect, and should be deleted rather than maintained.**
    The layered design has exactly one axis lineage: L1 emits per-signal measurements, L2 fuses
    them, `final/` holds the result. A second speaker axis exists only because L1 emits a per-pass
    axis fold it is not supposed to compute (item 25) and the belief store was built to ingest it.
    Remove that fold and the belief store has nothing to read but L2's axes — one number, and no
    reason for this comparison to exist.

    Returns ``None`` when no fused parquet exists, so older runs still render.
    """
    import pandas as pd

    from senselab.audio.workflows.audio_analysis.layout import belief_dir

    found = sorted(belief_dir(out_dir).glob(f"round*/uncertainty/{axis}.parquet"))
    if not found:
        return None
    try:
        return pd.read_parquet(found[-1]).sort_values("start")
    except Exception:  # noqa: BLE001 — a plot must not fail a run
        return None
