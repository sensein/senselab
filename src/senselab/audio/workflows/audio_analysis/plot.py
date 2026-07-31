"""6-row aggregate-uncertainty + per-source-detail timeline plot.

Per FR-006 (revised 2026-05-09): the plot must let a reviewer answer "WHY is this
bucket uncertain?" in addition to "HOW uncertain is it?". Three uncertainty rows show
the headline scalars; three detail rows show the underlying source signals so the
reviewer can drill in directly without opening the parquets.

Rows top-to-bottom:

1. **speech_presence_uncertainty** — raw solid + enhanced dashed in [0, 1]
2. **speaker_uncertainty** — raw solid + enhanced dashed
3. **asr_uncertainty** — raw solid + enhanced dashed
4. **Diarization detail** — per (pass, diar_model), speaker bars at native segment
   times, colored by speaker label. Lets the reviewer see where each diar model
   thinks each speaker is.
5. **Embedding similarity (adjacent windows)** — per (pass, embedding_model), a line
   of ``1 − cos_sim`` between consecutive uniform-window embeddings (default
   2 s window / 1 s hop). Spikes mark speaker-change events independent of any diar
   model's segmentation. Lets the reviewer compare the diar models' label
   transitions against what the audio itself says.
6. **ASR output** — per (pass, asr_model), token-level spans at the actual
   timestamps from the resolved (post-MMS-aligned) ASR result. Lets the reviewer see
   which models returned text where and confirm whether high asr uncertainty
   is real disagreement or punctuation/hesitation noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from senselab.audio.workflows.audio_analysis.layout import final_dir
from senselab.audio.workflows.audio_analysis.types import AxisResult
from senselab.utils.data_structures.logging import logger


def _series_for(rows: list, duration_s: float, hop_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(centers, values)`` for plotting one (pass, axis) line in [0, 1].

    Plots one point per row at the row's own midpoint — this matches whatever
    bucket grid ``BucketGrid.iter_buckets`` actually produced (handling
    overlapping grids correctly), rather than re-deriving a separate bucket
    count from ``duration_s / hop_s`` which would disagree with the parquet for
    any non-trivial grid.
    """
    if duration_s <= 0 or not rows:
        return np.array([]), np.array([])
    centers = np.array([0.5 * (r.start + r.end) for r in rows], dtype=np.float64)
    values = np.array(
        [float(r.within_pass_uncertainty) if r.within_pass_uncertainty is not None else np.nan for r in rows],
        dtype=np.float64,
    )
    # Sort by center so plotting doesn't draw a zigzag if rows arrive out of order.
    order = np.argsort(centers)
    return centers[order], values[order]


def _attr_series(rows: list, attr: str) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted ``(centers, values)`` for an arbitrary numeric row attribute.

    Used for the scene-quality / sound-source rows, whose values live on the
    speech_presence rows' additive columns (``quality_*`` / ``src_*``). ``None`` values
    become ``nan`` so gaps don't draw as zeros.
    """
    if not rows:
        return np.array([]), np.array([])
    centers = np.array([0.5 * (r.start + r.end) for r in rows], dtype=np.float64)
    values = np.array(
        [float(v) if (v := getattr(r, attr, None)) is not None else np.nan for r in rows],
        dtype=np.float64,
    )
    order = np.argsort(centers)
    return centers[order], values[order]


def _speech_presence_has_attr(axis_results: dict, attrs: tuple[str, ...]) -> bool:
    """True if any speech_presence row across passes carries a non-null value for any of ``attrs``."""
    for (_pl, axis), result in axis_results.items():
        if axis != "speech_presence":
            continue
        for r in result.rows:
            if any(getattr(r, a, None) is not None for a in attrs):
                return True
    return False


def _seg_attr(seg: Any, name: str) -> Any:  # noqa: ANN401
    """Tolerant getter for ScriptLine vs dict shapes."""
    if isinstance(seg, dict):
        return seg.get(name)
    return getattr(seg, name, None)


def _cluster_speakers_by_embedding(
    detail_by_pass: dict[str, dict[str, Any]],
    cosine_threshold: float = 0.5,
) -> dict[tuple[str, str, str], str]:
    """Cluster ``(pass, diar_model, raw_label)`` triples across **all** passes.

    Speakers on raw and enhanced passes are the same physical people; ECAPA /
    ResNet embeddings on the two passes shift slightly from enhancement but
    typically stay well within the same-speaker cosine band (≤0.3). Sharing
    the centroid accumulator across passes lets a speaker keep one cluster id
    (and one plot color) end-to-end, regardless of which pass / diar model is
    asking. Falls back to label-string speaker when no embeddings are
    available for a given (pass, diar_model).
    """
    from senselab.audio.workflows.audio_analysis.clustering import (
        _diar_segments,
        _mean_window_embedding_over_segments,
        assign_unified_clusters_with_seed_phase,
    )

    out: dict[tuple[str, str, str], str] = {}
    # One seed group per (pass, synthetic embedding source) so that within
    # a pass, the per-pass clusterer's already-distinct speaker labels never
    # merge with each other. Cross-pass matching (raw_Peter ↔ enh_Peter)
    # is handled inside ``assign_unified_clusters_with_seed_phase`` at the
    # higher ``cross_group_threshold``.
    seed_groups: dict[tuple[str, str], list[tuple[tuple[str, str, str], np.ndarray]]] = {}
    other_items: list[tuple[tuple[str, str, str], np.ndarray]] = []

    for pass_label, detail in detail_by_pass.items():
        per_window_emb_by_model = detail.get("per_window_embeddings") or {}
        diar_by_model = detail.get("diar_by_model") or {}
        if not per_window_emb_by_model or not any(per_window_emb_by_model.values()):
            for m, segs in diar_by_model.items():
                for seg in segs:
                    spk = str(_seg_attr(seg, "speaker") or "?")
                    out.setdefault((pass_label, m, spk), spk)
            continue
        emb_model = sorted(per_window_emb_by_model)[0]
        windows = per_window_emb_by_model[emb_model]
        for m, segs in diar_by_model.items():
            block = {"status": "ok", "result": [list(segs)]}
            label_segs: dict[str, list[Any]] = {}
            for seg in _diar_segments(block):
                spk = str(_seg_attr(seg, "speaker") or "?")
                label_segs.setdefault(spk, []).append(seg)
            is_seed = m.startswith("embedding_silhouette/")
            for label, label_seg_list in label_segs.items():
                mean_emb = _mean_window_embedding_over_segments(label_seg_list, windows)
                if mean_emb is None or mean_emb.size == 0:
                    out[(pass_label, m, label)] = label
                    continue
                if is_seed:
                    seed_groups.setdefault((pass_label, m), []).append(((pass_label, m, label), mean_emb))
                else:
                    other_items.append(((pass_label, m, label), mean_emb))

    out.update(
        assign_unified_clusters_with_seed_phase(
            list(seed_groups.values()), other_items, cosine_threshold=cosine_threshold
        )
    )
    return out


def _iter_leaf_tokens(asr_result: Any) -> Any:  # noqa: ANN401
    """Yield ``(start, end, text)`` for every leaf-level chunk in an ASR result.

    Handles both shapes the comparator sees:

    - **Whisper-style** (native per-token chunks on a single ScriptLine): the line's
      ``chunks`` field is already at word level.
    - **Post-MMS-aligned** (text-only ASR resolved through the alignment block): the
      structure is nested — outer ScriptLine → asr ScriptLine →
      word ScriptLines. Recurse to the leaves to get the per-word text + timestamps.

    A "leaf" is a chunk with no further sub-chunks (or whose sub-chunks have no
    timestamps).
    """
    if not asr_result:
        return
    items = asr_result if isinstance(asr_result, list) else [asr_result]
    stack: list[Any] = list(items)
    while stack:
        node = stack.pop(0)
        if node is None:
            continue
        children = _seg_attr(node, "chunks") or []
        cs = _seg_attr(node, "start")
        ce = _seg_attr(node, "end")
        text = _seg_attr(node, "text") or ""
        if children:
            # Walk into children depth-first; preserve ordering.
            stack = list(children) + stack
        elif cs is not None and ce is not None and text.strip():
            yield float(cs), float(ce), str(text).strip()


def _adjacent_window_cosine_series(
    window_embeddings: list[Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(timestamps, 1 − cos_sim)`` between consecutive window embeddings.

    Each entry is a ``WindowEmbedding`` (dataclass with ``start_s`` / ``end_s`` /
    ``vector``) or a 3-tuple ``(start_s, end_s, vector)``. Adjacent-window cosine
    distance is a model-free indicator of speaker change — independent of any
    diarization model's segmentation.

    Each distance value is anchored at the midpoint between the two windows'
    centers (i.e. where the change is happening), not at the later window's
    center — that way a speaker turn at time ``T`` shows up as a spike at ``T``,
    not at ``T + window_s/2``.
    """
    if not window_embeddings:
        return np.array([]), np.array([])
    window_centers: list[float] = []
    vectors: list[np.ndarray] = []
    for w in window_embeddings:
        if hasattr(w, "start_s"):
            s, e, v = float(w.start_s), float(w.end_s), w.vector
        else:
            s, e, v = float(w[0]), float(w[1]), w[2]
        window_centers.append(0.5 * (s + e))
        vectors.append(np.asarray(v, dtype=np.float64))
    n = len(vectors)
    if n < 2:
        return np.array([]), np.array([])
    timestamps = np.array([0.5 * (window_centers[i - 1] + window_centers[i]) for i in range(1, n)])
    distances = np.full(n - 1, np.nan)
    for i in range(1, n):
        a, b = vectors[i - 1], vectors[i]
        if a.size == 0 or b.size == 0 or a.size != b.size:
            continue
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
        sim = float(np.dot(a, b) / denom)
        distances[i - 1] = max(0.0, min(1.0, 1.0 - sim))
    return timestamps, distances


def _pass_color_alpha(pass_label: str) -> tuple[float, str]:
    """Lighter alpha + dashed for enhanced; full alpha + solid for raw."""
    if pass_label == "raw_16k":
        return 0.85, "-"
    return 0.55, "--"


_MASK_STATE_STYLE = {
    "target_free": ("#2e7d32", "target-free"),
    "indeterminate": ("#9e9e9e", "cannot tell"),
    "target_active": ("#c8c8c8", "target active"),
}


def _load_background_mask_rows(run_dir: Path) -> list[dict[str, Any]]:
    """Read the unmodified pass's background mask, if one was written.

    Only the unmodified pass is consulted: enhancement removes the non-speech evidence
    target activity is read from, so an enhanced-pass mask reports more of the recording as
    safe for background claims exactly where the background was destroyed.
    """
    for candidate in sorted(Path(run_dir).glob("*/background_mask.parquet")):
        if "enhanced" in candidate.parent.name:
            continue
        try:
            import pandas as pd

            return list(pd.read_parquet(candidate).to_dict("records"))
        except Exception as exc:  # noqa: BLE001 — a plot must never fail a run
            logger.debug("background mask unreadable at %s: %s", candidate, exc)
    return []


def _draw_background_mask_row(ax: Any, rows: list[dict[str, Any]], duration_s: float) -> None:  # noqa: ANN401
    """Draw the mask as a three-state strip, with uncertainty as alpha."""
    from matplotlib.patches import Rectangle

    ax.set_ylabel("background\nmask", rotation=0, ha="right", va="center", fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlim(0, duration_s)
    seen: set[str] = set()
    for row in rows:
        state = str(row.get("state", "indeterminate"))
        color, label = _MASK_STATE_STYLE.get(state, _MASK_STATE_STYLE["indeterminate"])
        start, end = float(row["start"]), float(row["end"])
        unc = float(row.get("uncertainty") or 0.0)
        # A washed-out green marks a region whose own "usable" claim is shaky.
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
            # Hatch what the guard removed, so a shrunken mask is visibly the guard's
            # doing rather than an absence of quiet audio.
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


def build_aligned_timeline_plot(
    *,
    run_dir: Path,
    axis_results: dict[tuple[Any, Any], AxisResult],
    duration_s: float,
    grid_hop: float,
    asr_grid_hop: float | None = None,
    detail_by_pass: dict[str, dict[str, Any]] | None = None,
    save_path: Path | None = None,
    title: str | None = None,
    audio_waveform: np.ndarray | None = None,
    audio_sr: int = 16000,
    chunk_duration_s: float | None = None,
) -> Path | None:
    """Render the aggregate-uncertainty + per-source-detail figure.

    Adds a pre-emphasized broad-band spectrogram (5 ms analysis window, 256-pt FFT)
    at the top when ``audio_waveform`` is provided. The short window favors time
    resolution over frequency resolution, which is appropriate when the audio mixes
    child and adult voices (high f0 → narrowband would smear formants into harmonic
    bands; broadband shows formant trajectories cleanly for both ranges).

    Chunked output is opt-in via ``chunk_duration_s``. It wrote ``timeline_001.png``,
    ``timeline_002.png`` … whose panels were mostly empty, because a fixed window rarely lines
    up with where anything happened; one figure per L2 round, drawn after fusion, replaced it.
    When enabled, the figure is rendered once and saved
    repeatedly with ``xlim`` adjusted to each ``chunk_duration_s``-second window. The
    files are written as ``timeline_001.png``, ``timeline_002.png``, …. For shorter
    audio a single ``timeline.png`` is written.

    Args:
        run_dir: Run directory; the figure lands in ``final/`` with the other deliverables.
        axis_results: ``{(pass_label, axis) → AxisResult}`` from ``compute_uncertainty_axes``.
        duration_s: Audio duration in seconds — drives the x-axis extent.
        grid_hop: Bucket hop length (seconds) — matches the comparator grid.
        asr_grid_hop: Hop length for the asr grid (typically wider than
            ``grid_hop``, e.g. 0.5 s with a 1.0 s window). When ``None``, falls back
            to ``grid_hop`` for the asr row.
        detail_by_pass: ``{pass_label → {"diar_by_model": {..}, "asr_by_model": {..},
            "per_window_embeddings": {emb_model → [WindowEmbedding, ...]},
            "ppg": {"per_frame_phonemes": [..], "frame_hop": float}}}``.
            Populates the four detail rows (diar / embedding / PPG / ASR). ``None``
            collapses the figure to the three uncertainty rows alone.
        save_path: Override path for the single PNG (only honored when
            ``duration_s <= chunk_duration_s``; chunked output always writes into ``run_dir``).
        title: Optional figure title.
        audio_waveform: Optional mono PCM samples to drive the top spectrogram row.
            Pass a 1-D float array (or 2-D array shaped ``(1, N)``); 2-D input is
            squeezed.
        audio_sr: Sampling rate for ``audio_waveform``. Defaults to 16 kHz.
        chunk_duration_s: Maximum visible time per figure, or ``None`` (default) for no
            chunked output at all. Audio longer than this
            triggers chunked output. Defaults to 20 s.

    Returns:
        Output path of the first PNG written, or ``None`` when ``duration_s <= 0``.
    """
    if duration_s <= 0:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    has_detail = bool(detail_by_pass)
    has_spec = audio_waveform is not None and np.asarray(audio_waveform).size > 1
    # Pass labels actually present in the detail bundle — any pass beyond the
    # default raw/enhanced pair shows up in every detail row. Sorted so that
    # "raw_16k" sorts before "enhanced_16k" by convention but extension passes
    # land in alphabetical order after.
    pass_order: list[str] = sorted(detail_by_pass.keys()) if detail_by_pass else []
    # Size the diar / PPG / ASR detail rows by stripe count (each stripe is one
    # (pass, model)) so tall rows aren't squashed and short rows don't waste space.
    n_diar_stripes = 0
    n_asr_stripes = 0
    n_ppg_stripes = 0
    if has_detail:
        assert detail_by_pass is not None
        for pl in pass_order:
            n_diar_stripes += len((detail_by_pass.get(pl) or {}).get("diar_by_model", {}))
            n_asr_stripes += len((detail_by_pass.get(pl) or {}).get("asr_by_model", {}))
            ppg = (detail_by_pass.get(pl) or {}).get("ppg")
            if ppg and ppg.get("per_frame_phonemes"):
                n_ppg_stripes += 1
    # Optional scene rows (feature 20260722-175022) — only when the speech_presence
    # axis actually carries the additive columns.
    has_quality = _speech_presence_has_attr(
        axis_results, ("quality_snr", "quality_clip", "quality_reverb", "quality_bandwidth")
    )
    has_sources = _speech_presence_has_attr(
        axis_results, ("src_speech", "src_people", "src_machine", "src_environment")
    )

    # Build the row-index map with a running counter so inserting rows never
    # requires re-deriving fragile offsets.
    height_ratios: list[float] = []
    row_idx: dict[str, int] = {}

    def _add_row(name: str, height: float) -> None:
        row_idx[name] = len(height_ratios)
        height_ratios.append(height)

    if has_spec:
        _add_row("spec", 1.6)  # broadband spectrogram
    mask_rows = _load_background_mask_rows(run_dir)
    if mask_rows:
        # Above the axis rows on purpose: it says *where the rows below can be trusted*.
        # A target-free span has no foreground to leak, so a background claim there does
        # not depend on suppression depth. Reading an axis without knowing which spans
        # were target-free invites treating a leakage artifact as a finding.
        _add_row("mask", 0.55)
    _add_row("speech_presence", 1.4)
    _add_row("speaker", 1.4)
    _add_row("asr", 1.4)
    if has_quality:
        _add_row("quality", 1.2)
    if has_sources:
        _add_row("sources", 1.2)
    if has_detail:
        diar_h = max(0.9, 0.3 * max(1, n_diar_stripes))
        asr_h = max(0.9, 0.3 * max(1, n_asr_stripes))
        _add_row("diar", diar_h)
        _add_row("emb", 1.2)
        if n_ppg_stripes > 0:
            _add_row("ppg", max(0.9, 0.3 * n_ppg_stripes))
        _add_row("asr_words", asr_h)
    n_rows = len(height_ratios)
    fig_height = sum(height_ratios) + 1.0

    spec_row = row_idx.get("spec")
    mask_row = row_idx.get("mask")
    speech_presence_row = row_idx["speech_presence"]
    speaker_row = row_idx["speaker"]
    asr_row = row_idx["asr"]
    quality_row = row_idx.get("quality")
    sources_row = row_idx.get("sources")
    diar_row = row_idx.get("diar")
    emb_row = row_idx.get("emb")
    ppg_row = row_idx.get("ppg")

    # ``None`` means chunking is off, so nothing will chunk regardless of duration. Narrowed to a
    # local rather than relying on ``bool(...) and ...`` short-circuiting, which mypy cannot follow.
    chunk_s = None if chunk_duration_s is None else float(chunk_duration_s)
    will_chunk = chunk_s is not None and chunk_s > 0 and duration_s > chunk_s + 0.5
    # When chunking, fix the figure width per-chunk; otherwise scale with duration.
    fig_width = 12.0 if will_chunk else max(10.0, duration_s / 4.0)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    if n_rows == 1:
        axes = np.array([axes])
    if title:
        fig.suptitle(title, fontsize=11)

    axis_color = {"speech_presence": "#1f77b4", "speaker": "#ff7f0e", "asr": "#2ca02c"}
    utt_hop = asr_grid_hop if asr_grid_hop is not None else grid_hop

    if mask_row is not None and mask_rows:
        _draw_background_mask_row(axes[mask_row], mask_rows, duration_s)

    if has_spec and spec_row is not None:
        from scipy.signal import spectrogram

        ax_spec = axes[spec_row]
        wf = np.asarray(audio_waveform, dtype=np.float32).squeeze()
        if wf.ndim > 1:
            wf = wf[0]
        # Pre-emphasis: y[n] = x[n] - 0.97 * x[n-1] — flattens spectral tilt and
        # boosts upper formants (helps see the high-frequency child-voice formants
        # without losing the adult-range lower formants).
        pre = np.empty_like(wf)
        pre[0] = wf[0]
        pre[1:] = wf[1:] - 0.97 * wf[:-1]
        # Broadband: 5 ms Hann window (80 samples @ 16 kHz) → good time resolution
        # so adjacent voiced frames don't smear together. nfft=256 zero-pads for
        # smoother frequency display. 75% overlap keeps the time grid dense.
        nperseg = max(16, int(round(0.005 * audio_sr)))
        noverlap = int(nperseg * 0.75)
        nfft = max(256, 1 << (nperseg - 1).bit_length())
        f, t, Sxx = spectrogram(
            pre,
            fs=audio_sr,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            scaling="density",
        )
        Sxx_db = 10.0 * np.log10(Sxx + 1e-10)
        # Robust dynamic range: clip to [median - 5 dB, max] so a few outliers
        # don't crush the contrast.
        vmin = np.percentile(Sxx_db, 5)
        vmax = np.percentile(Sxx_db, 99.5)
        ax_spec.pcolormesh(t, f, Sxx_db, cmap="magma", shading="auto", vmin=vmin, vmax=vmax, rasterized=True)
        ax_spec.set_ylim(0, audio_sr / 2)
        ax_spec.set_ylabel("freq (Hz)\nbroadband", fontsize=8)
        ax_spec.set_xlim(0, duration_s)

    # Rows 1–3: per-axis raw + enhanced overlay.
    for axis, row_i in (("speech_presence", speech_presence_row), ("speaker", speaker_row), ("asr", asr_row)):
        ax = axes[row_i]
        # Utterance has its own (possibly wider+overlapping) grid.
        axis_hop = utt_hop if axis == "asr" else grid_hop
        for pass_label in pass_order:
            result = axis_results.get((pass_label, axis))
            if result is None:
                continue
            centers, values = _series_for(result.rows, duration_s, axis_hop)
            if centers.size == 0:
                continue
            alpha, style = _pass_color_alpha(pass_label)
            label = f"{'raw' if pass_label == 'raw_16k' else 'enhanced'} {axis}"
            ax.plot(centers, values, linestyle=style, color=axis_color[axis], linewidth=1.0, alpha=alpha, label=label)
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"{axis}\nuncertainty", fontsize=8)
        ax.grid(axis="x", alpha=0.2)
        if any(line.get_label() and not line.get_label().startswith("_") for line in ax.lines):
            ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.85)

    # Scene quality: the four [0,1] degradation scores over time (0 = clean).
    if quality_row is not None:
        ax = axes[quality_row]
        q_colors = {
            "quality_snr": "#d62728",
            "quality_clip": "#9467bd",
            "quality_reverb": "#8c564b",
            "quality_bandwidth": "#e377c2",
        }
        for pass_label in pass_order:
            result = axis_results.get((pass_label, "speech_presence"))
            if result is None:
                continue
            alpha, style = _pass_color_alpha(pass_label)
            for attr, color in q_colors.items():
                centers, values = _attr_series(result.rows, attr)
                if centers.size == 0 or np.all(np.isnan(values)):
                    continue
                suffix = "" if pass_label == "raw_16k" else " (enh)"
                label = attr.removeprefix("quality_") + suffix
                ax.plot(centers, values, linestyle=style, color=color, linewidth=1.0, alpha=alpha, label=label)
        ax.set_ylim(0, 1)
        ax.set_ylabel("scene quality\n(0=clean)", fontsize=8)
        ax.grid(axis="x", alpha=0.2)
        if any(line.get_label() and not line.get_label().startswith("_") for line in ax.lines):
            ax.legend(loc="upper right", fontsize=6, ncol=4, framealpha=0.85)

    # Sound sources: stacked category masses (raw pass) — speech/people/machine/environment.
    if sources_row is not None:
        ax = axes[sources_row]
        src_cats = ("src_speech", "src_people", "src_machine", "src_environment")
        src_colors = {
            "src_speech": "#2ca02c",
            "src_people": "#1f77b4",
            "src_machine": "#7f7f7f",
            "src_environment": "#bcbd22",
        }
        # Prefer the raw pass; fall back to the first speech_presence pass available.
        result = axis_results.get(("raw_16k", "speech_presence"))
        if result is None:
            for pass_label in pass_order:
                if (pass_label, "speech_presence") in axis_results:
                    result = axis_results[(pass_label, "speech_presence")]
                    break
        if result is not None:
            rows = sorted(
                (r for r in result.rows if getattr(r, "src_speech", None) is not None),
                key=lambda r: r.start,
            )
            if rows:
                centers = np.array([0.5 * (r.start + r.end) for r in rows], dtype=np.float64)
                stacks = [np.array([float(getattr(r, c) or 0.0) for r in rows], dtype=np.float64) for c in src_cats]
                ax.stackplot(
                    centers,
                    *stacks,
                    labels=[c.removeprefix("src_") for c in src_cats],
                    colors=[src_colors[c] for c in src_cats],
                    alpha=0.85,
                )
                ax.legend(loc="upper right", fontsize=6, ncol=4, framealpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_ylabel("sound\nsources", fontsize=8)
        ax.grid(axis="x", alpha=0.2)

    if has_detail:
        assert detail_by_pass is not None  # narrow for type checker
        # Row 4: Diarization detail — one stripe per (pass, diar_model). Color is
        # assigned by clustering each (pass, model, raw_label) via cosine similarity
        # of the mean speaker embedding (sampled from the primary diar's per-segment
        # embedding history). Speakers identified as the same person across
        # diar models (or across passes) end up in the same cluster and therefore
        # the same color, regardless of label naming convention.
        ax_diar = axes[diar_row]
        diar_cmap = matplotlib.colormaps.get_cmap("tab10")
        cluster_map = _cluster_speakers_by_embedding(detail_by_pass)
        cluster_color: dict[str, Any] = {}
        for cluster_id in cluster_map.values():
            if cluster_id not in cluster_color:
                cluster_color[cluster_id] = diar_cmap(len(cluster_color) % 10)

        diar_stripes: list[tuple[str, str, list[Any]]] = []
        for pass_label in pass_order:
            for m, segs in (detail_by_pass.get(pass_label) or {}).get("diar_by_model", {}).items():
                diar_stripes.append((pass_label, m, segs))
        if diar_stripes:
            for k, (pass_label, m, segs) in enumerate(diar_stripes):
                y = k
                # Distinguish pass via edge color (raw=solid black; enhanced=grey),
                # not via fill — fill is reserved for speaker speaker.
                edge = "black" if pass_label == "raw_16k" else "0.4"
                for seg in segs:
                    s = _seg_attr(seg, "start")
                    e = _seg_attr(seg, "end")
                    spk_raw = str(_seg_attr(seg, "speaker") or "?")
                    if s is None or e is None:
                        continue
                    cluster_id = cluster_map.get((pass_label, m, spk_raw), spk_raw)
                    color_ = cluster_color.get(cluster_id)
                    if color_ is None:
                        # Fallback for labels not seen during clustering.
                        cluster_color[cluster_id] = diar_cmap(len(cluster_color) % 10)
                        color_ = cluster_color[cluster_id]
                    ax_diar.barh(
                        y + 0.15,
                        float(e) - float(s),
                        left=float(s),
                        height=0.7,
                        color=color_,
                        edgecolor=edge,
                        linewidth=0.3,
                    )
            ax_diar.set_yticks([k + 0.5 for k in range(len(diar_stripes))])
            ax_diar.set_yticklabels([f"{pl[:3]} {m.split('/')[-1][:18]}" for pl, m, _ in diar_stripes], fontsize=7)
            ax_diar.set_ylim(0, max(1, len(diar_stripes)))
            # Legend: one entry per cluster (= "speaker as identified by embedding").
            if cluster_color:
                import matplotlib.patches as mpatches

                handles = [
                    mpatches.Patch(color=c, label=k_) for k_, c in sorted(cluster_color.items(), key=lambda kv: kv[0])
                ]
                ax_diar.legend(
                    handles=handles,
                    loc="upper right",
                    fontsize=7,
                    ncol=min(len(handles), 4),
                    framealpha=0.85,
                    title="speakers (embedding-clustered)",
                    title_fontsize=7,
                )
        ax_diar.set_ylabel("diar\nspeakers", fontsize=8)
        ax_diar.grid(axis="x", alpha=0.2)

        # Row 5: Embedding adjacent-window cosine — one line per (pass, embedding_model).
        # The series walks the uniform window grid (default 2 s window / 1 s hop) and
        # plots ``1 − cos`` between consecutive windows. Spikes mark speaker-change
        # events independent of any diar model's segmentation.
        ax_emb = axes[emb_row]
        emb_palette = {
            "speechbrain/spkrec-ecapa-voxceleb": "#9467bd",
            "speechbrain/spkrec-resnet-voxceleb": "#8c564b",
        }
        any_emb_plotted = False
        for pass_label in pass_order:
            detail = detail_by_pass.get(pass_label) or {}
            for emb_model, windows in (detail.get("per_window_embeddings") or {}).items():
                if not windows:
                    continue
                centers, dists = _adjacent_window_cosine_series(windows)
                if centers.size == 0 or np.all(np.isnan(dists)):
                    continue
                alpha, style = _pass_color_alpha(pass_label)
                emb_color = emb_palette.get(emb_model, "#7f7f7f")
                short = emb_model.split("/")[-1].replace("spkrec-", "")
                label = f"{'raw' if pass_label == 'raw_16k' else 'enh'} {short}"
                ax_emb.plot(centers, dists, color=emb_color, linestyle=style, linewidth=1.0, alpha=alpha, label=label)
                any_emb_plotted = True
        ax_emb.set_ylim(0, 1)
        ax_emb.set_ylabel("embedding\n1−cos(adj win)", fontsize=8)
        ax_emb.grid(axis="x", alpha=0.2)
        if any_emb_plotted:
            ax_emb.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.85)

        # Row 6 (when present): PPG argmax phonetic sequence, time-aligned. One stripe
        # per pass with PPG data; each contiguous run of the same argmax phoneme
        # renders as a bar from the run's start to its end with the phoneme letter(s)
        # as small text. ``<silent>`` runs render as a faint background bar.
        if ppg_row is not None:
            ax_ppg = axes[ppg_row]
            ppg_stripes: list[tuple[str, list[tuple[float, float, str]]]] = []
            for pass_label in pass_order:
                ppg_detail = (detail_by_pass.get(pass_label) or {}).get("ppg") or {}
                per_frame = ppg_detail.get("per_frame_phonemes") or []
                frame_hop = float(ppg_detail.get("frame_hop") or 0.0)
                if per_frame and frame_hop > 0:
                    runs = []
                    cur = per_frame[0]
                    cur_start = 0
                    for f in range(1, len(per_frame)):
                        if per_frame[f] != cur:
                            runs.append((cur_start * frame_hop, f * frame_hop, cur))
                            cur = per_frame[f]
                            cur_start = f
                    runs.append((cur_start * frame_hop, len(per_frame) * frame_hop, cur))
                    ppg_stripes.append((pass_label, runs))

            if ppg_stripes:
                for k, (pass_label, runs) in enumerate(ppg_stripes):
                    y = k
                    alpha, _ = _pass_color_alpha(pass_label)
                    for rs, re_, phon in runs:
                        is_silent = phon == "<silent>"
                        ax_ppg.barh(
                            y + 0.15,
                            re_ - rs,
                            left=rs,
                            height=0.7,
                            color="#cccccc" if is_silent else "#9467bd",
                            alpha=alpha if not is_silent else alpha * 0.4,
                            edgecolor="none",
                        )
                        if not is_silent and (re_ - rs) >= 0.04:
                            ax_ppg.text(
                                (rs + re_) / 2,
                                y + 0.22,
                                phon,
                                ha="center",
                                va="bottom",
                                fontsize=3.5,
                                color="black",
                                clip_on=True,
                            )
                ax_ppg.set_yticks([k + 0.5 for k in range(len(ppg_stripes))])
                ax_ppg.set_yticklabels([f"{pl[:3]} ppg" for pl, _ in ppg_stripes], fontsize=7)
                ax_ppg.set_ylim(0, max(1, len(ppg_stripes)))
            ax_ppg.set_ylabel("PPG\nphonemes", fontsize=8)
            ax_ppg.grid(axis="x", alpha=0.2)

        # Row 6/7: ASR output — one stripe per (pass, asr_model), token spans at native
        # timestamps with the actual text rendered on each bar (small font) so the
        # reviewer can see WHY asr uncertainty is high (punctuation differences,
        # partial words, hesitation tokens).
        # Read the index directly rather than via the Optional local: this block runs under the
        # same ``has_detail`` guard that added the row, so it is present by construction. The
        # Optional lookup is what let the key collision with the axis row go unnoticed.
        ax_asr = axes[row_idx["asr_words"]]
        asr_stripes: list[tuple[str, str, Any]] = []
        for pass_label in pass_order:
            for m, asr_result in (detail_by_pass.get(pass_label) or {}).get("asr_by_model", {}).items():
                asr_stripes.append((pass_label, m, asr_result))
        asr_palette = list(mcolors.TABLEAU_COLORS.values())
        if asr_stripes:
            for k, (pass_label, m, asr_result) in enumerate(asr_stripes):
                y = k
                stripe_color = asr_palette[k % len(asr_palette)]
                alpha, _ = _pass_color_alpha(pass_label)
                tokens = list(_iter_leaf_tokens(asr_result))
                if tokens:
                    for cs, ce, ct in tokens:
                        width = ce - cs
                        ax_asr.barh(
                            y + 0.15,
                            width,
                            left=cs,
                            height=0.7,
                            color=stripe_color,
                            alpha=alpha,
                            edgecolor="none",
                        )
                        # Render the token's text inside the bar, anchored to the
                        # bottom of the bar (so the speaker stripe hue stays clean
                        # at the top). Skip very short tokens (< 60 ms) where the
                        # text would overflow.
                        if width >= 0.06 and ct:
                            ax_asr.text(
                                cs + width / 2,
                                y + 0.22,  # just above bar bottom (bar is y+0.15..y+0.85)
                                ct.strip(),
                                ha="center",
                                va="bottom",
                                fontsize=3.5,
                                color="black",
                                clip_on=True,
                            )
                else:
                    # Text-only ASR with no alignment block — paint a faint
                    # whole-row stripe so the reviewer sees the model contributed.
                    line0 = (asr_result if isinstance(asr_result, list) else [asr_result])[0] if asr_result else None
                    if line0 is not None:
                        ls = _seg_attr(line0, "start")
                        le = _seg_attr(line0, "end")
                        if ls is not None and le is not None:
                            ax_asr.barh(
                                y + 0.15,
                                float(le) - float(ls),
                                left=float(ls),
                                height=0.7,
                                color=stripe_color,
                                alpha=alpha * 0.4,
                                edgecolor="none",
                            )
            ax_asr.set_yticks([k + 0.5 for k in range(len(asr_stripes))])
            ax_asr.set_yticklabels([f"{pl[:3]} {m.split('/')[-1][:18]}" for pl, m, _ in asr_stripes], fontsize=7)
            ax_asr.set_ylim(0, max(1, len(asr_stripes)))
        ax_asr.set_ylabel("ASR\noutput", fontsize=8)
        ax_asr.set_xlabel("Time (s)")
        ax_asr.grid(axis="x", alpha=0.2)
        ax_asr.set_xlim(0, duration_s)

    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    if not will_chunk:
        out = save_path or (final_dir(run_dir) / "timeline.png")
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140)
        plt.close(fig)
        return out
    # Chunked output is opt-in. It produced timeline_001.png, timeline_002.png … whose panels
    # were mostly empty, because a fixed window rarely lines up with where anything happened.
    # One figure per L2 round, drawn after fusion, is the useful unit instead.
    if not chunk_duration_s:
        plt.close(fig)
        return None

    import math

    n_chunks = math.ceil(duration_s / chunk_duration_s)
    out_paths: list[Path] = []
    run_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_chunks):
        t0 = i * chunk_duration_s
        t1 = min((i + 1) * chunk_duration_s, duration_s)
        for ax in axes:
            ax.set_xlim(t0, t1)
        chunk_dir = final_dir(run_dir)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        out_path = chunk_dir / f"timeline_{i + 1:03d}.png"
        fig.savefig(out_path, dpi=140)
        out_paths.append(out_path)
    plt.close(fig)
    return out_paths[0]
