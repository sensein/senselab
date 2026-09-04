"""FIGURE — PREPROCESS's and TAXONOMY's own output, drawn from the store.

Reads the two nodes' elements and their sidecars and writes one image per fixed-width page. It runs
no model, reads no hint, decides nothing, and writes nothing back to the store, so it can be
re-invoked over a completed run directory exactly as ``report()`` can.

Its configuration is split in two, and the split is enforced by construction: every pipeline value
comes from the packaged :class:`TriageConfig` and is only ever read, while every value that governs
the drawing itself lives in :class:`FigureStyle`. A page whose panel has nothing to draw says which
element is absent and why, taking the reason from PREPROCESS's own verdict rather than supplying a
value of its own.

See ``specs/20260904-preprocess-taxonomy-figure/design.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import (
    find_measurement,
    find_measurements,
    live_entities,
    resolve_stream,
)
from senselab.utils.prov_store import Entity, ProvStore

NODE = "FIGURE"

_STREAM = "preemphasised"
_FALLBACK_STREAM = "plain"

#: The derivative each TAXONOMY evidence line reads, so an unavailable line can name what is missing.
_LINE_SOURCE: dict[tuple[str, str], tuple[str, ...]] = {
    ("speech", "acoustic"): ("yamnet_windows", "ast_windows"),
    ("speech", "lexical"): ("consensus_transcript",),
    ("airway", "health_acoustic"): ("span_hear",),
    ("airway", "acoustic"): ("span_yamnet",),
    ("voice", "phonation"): ("phonation_tracks",),
}

_SUMMARISED_CLASSIFIERS = ("yamnet", "ast", "hear")

# E=envelope (primary amplitude), C=continuity, A=asr, S=normalization (supplementary amplitude).
_MEASURE_CODE = {"amplitude": "E", "continuity": "C", "asr": "A"}
_SPAN_ROWS = ("E", "C", "A", "S")


@dataclass(frozen=True)
class FigureStyle:
    """How the figure is drawn. Nothing here is read by the pipeline.

    Attributes:
        page_seconds: The width of every page, in recording seconds.
        pad_short_pages: Whether a final page shorter than ``page_seconds`` is padded out to it, so
            every image spans the same duration and panels are comparable page to page.
        figure_inches: ``(width, height)`` of one page.
        dpi: Raster resolution.
        height_ratios: One entry per panel, top first.
        spectrogram_dynamic_range_db: Colour floor, in dB below the page's own peak bin.
        top_labels: How many labels a per-span raster shows.
        summary_labels: How many labels the whole-file taxonomy panel lists per classifier.
        colour_primary: Envelope and primary-amplitude spans.
        colour_supplement: Normalization-derived spans.
        colour_continuity: The continuity trace and its spans.
        colour_asr: ASR-derived spans and the word lane.
        colour_clip: Clip-event accents.
        colour_padding: The shading that marks a padded tail.
        cmap_spectrogram: Spectrogram colormap.
        cmap_yamnet: YAMNet raster colormap.
        cmap_hear: HeAR raster colormap.
        cmap_squim: SQUIM raster colormap.
        word_colours: The consensus-word fill cycle.
        title_fontsize: Panel title size.
        tick_fontsize: Axis tick-label size.
        cell_fontsize: Score text drawn inside a raster cell.
        marker_size: Raster cell area.
        text_fontsize: The taxonomy panel's monospaced lines.
        absent_fontsize: The note a panel prints when its element is absent.
    """

    page_seconds: float = 20.0
    pad_short_pages: bool = True
    figure_inches: tuple[float, float] = (14.0, 17.0)
    dpi: int = 130
    height_ratios: tuple[float, ...] = (2.2, 2.0, 0.5, 0.5, 0.32, 0.5, 1.0, 1.6)
    spectrogram_dynamic_range_db: float = 80.0
    top_labels: int = 4
    summary_labels: int = 6
    colour_primary: str = "steelblue"
    colour_supplement: str = "darkorange"
    colour_continuity: str = "mediumseagreen"
    colour_asr: str = "mediumpurple"
    colour_clip: str = "crimson"
    colour_padding: str = "0.55"
    cmap_spectrogram: str = "magma"
    cmap_yamnet: str = "BuGn"
    cmap_hear: str = "OrRd"
    cmap_squim: str = "Purples"
    word_colours: tuple[str, ...] = ("#6a51a3", "#3182bd", "#31a354", "#e6550d")
    title_fontsize: float = 9.0
    tick_fontsize: float = 6.0
    cell_fontsize: float = 5.0
    marker_size: float = 260.0
    text_fontsize: float = 7.5
    absent_fontsize: float = 7.0
    span_row_colours: dict[str, str] = field(default_factory=dict)

    def row_colour(self, code: str) -> str:
        """The colour for one span-source row.

        Args:
            code: ``"E"``, ``"C"``, ``"A"`` or ``"S"``.

        Returns:
            The configured colour.
        """
        default = {
            "E": self.colour_primary,
            "C": self.colour_continuity,
            "A": self.colour_asr,
            "S": self.colour_supplement,
        }
        return self.span_row_colours.get(code, default[code])


def pages(duration_s: float, style: FigureStyle) -> list[tuple[float, float]]:
    """The page windows covering a recording, every one the same width.

    A recording is cut into ``style.page_seconds`` pages and the final page keeps that full width
    even when the recording stops inside it, so a span's drawn width means the same thing on every
    page. The uncovered tail is padding, not silence, and :func:`_mark_padding` says so on the page.

    Args:
        duration_s: The recording's real duration.
        style: The drawing configuration.

    Returns:
        ``[(start, end), ...]``, always at least one page.

    Raises:
        ValueError: If ``page_seconds`` is not positive, since no number of pages would cover the
            recording.
    """
    if style.page_seconds <= 0:
        raise ValueError(f"page_seconds must be positive, got {style.page_seconds}")
    n_pages = max(1, ceil(duration_s / style.page_seconds))
    if not style.pad_short_pages:
        return [
            (index * style.page_seconds, min((index + 1) * style.page_seconds, duration_s)) for index in range(n_pages)
        ]
    return [(index * style.page_seconds, (index + 1) * style.page_seconds) for index in range(n_pages)]


def _mark_padding(axes: Sequence[Axes], duration_s: float, t1: float, style: FigureStyle) -> bool:
    """Shade the part of a page that is past the end of the recording.

    Args:
        axes: Every panel on the page.
        duration_s: The recording's real duration.
        t1: The page's right edge.
        style: The drawing configuration.

    Returns:
        Whether any padding was drawn.
    """
    if t1 <= duration_s:
        return False
    for axis in axes:
        axis.axvspan(
            duration_s,
            t1,
            facecolor=style.colour_padding,
            alpha=0.18,
            hatch="xx",
            edgecolor=style.colour_padding,
            linewidth=0.0,
            zorder=5,
        )
        axis.axvline(duration_s, color=style.colour_padding, linewidth=1.0, linestyle="--", zorder=6)
    axes[0].text(
        (duration_s + t1) / 2,
        0.94,
        "padding — recording ended",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=style.tick_fontsize,
        color="0.25",
        zorder=7,
    )
    return True


def _stream_path(store: ProvStore, run_dir: Path) -> Path | None:
    """The conditioned stream this figure draws, preferring the pre-emphasised one.

    Args:
        store: The provenance store.
        run_dir: The run directory the stream sits under.

    Returns:
        The WAV path, or None when neither stream is in the store.
    """
    for name in (_STREAM, _FALLBACK_STREAM):
        try:
            entity_id, _ = resolve_stream(store, run_dir, name)
        except LookupError:
            continue
        entity = store.get_entity(entity_id)
        path = Path(str(entity.attributes["path"]))
        candidate = path if path.is_absolute() else run_dir / path
        if candidate.is_file():
            return candidate
    return None


def _absent_reasons(store: ProvStore) -> dict[str, str]:
    """Which PREPROCESS derivatives are absent, and the exception that made each one absent.

    Args:
        store: The provenance store.

    Returns:
        ``{derivative: reason}``, empty when PREPROCESS recorded no verdict.
    """
    for entity in store.entities("verdict"):
        if store.is_invalidated(entity.id) or entity.attributes.get("node") != "PREPROCESS":
            continue
        detail = entity.attributes.get("detail") or {}
        absent = detail.get("absent") or {}
        return {str(name): str(reason) for name, reason in absent.items()}
    return {}


def _npz(run_dir: Path, store: ProvStore, name: str, key: str) -> np.ndarray | None:
    """One array out of a persisted derivative, or None when the derivative never reached the store.

    Args:
        run_dir: The run directory.
        store: The provenance store.
        name: The measurement's name.
        key: The array's key inside the ``.npz``.

    Returns:
        The array, or None.
    """
    measurement = find_measurement(store, name)
    if measurement is None:
        return None
    path = measurement.attributes.get("path")
    if not path:
        return None
    sidecar = run_dir / str(path)
    if not sidecar.is_file():
        return None
    with np.load(sidecar) as loaded:
        if key not in loaded:
            return None
        return np.asarray(loaded[key])


def _span_code(signal_name: str, measure: str) -> str:
    """The row code for one span's proposing source.

    Args:
        signal_name: The span's signal.
        measure: The span's measure.

    Returns:
        ``"E"``, ``"C"``, ``"A"``, ``"S"`` or ``"?"``.
    """
    if measure == "amplitude" and signal_name == "normalized":
        return "S"
    return _MEASURE_CODE.get(measure, "?")


def _spans(store: ProvStore) -> list[dict[str, Any]]:
    """Every live general span, with what the lane panel needs to draw it.

    Args:
        store: The provenance store.

    Returns:
        One dict per span.
    """
    return [
        {
            "id": entity.id,
            "extent": entity.extent,
            "signal": entity.attributes.get("signal"),
            "measure": entity.attributes.get("measure"),
            "contains_clip": bool(entity.attributes.get("contains_clip")),
            "corroborated_by": entity.attributes.get("corroborated_by") or [],
        }
        for entity in live_entities(store, "span")
        if entity.attributes.get("family") is None and entity.extent is not None
    ]


def _clip_extents(store: ProvStore) -> list[tuple[float, float]]:
    """Every live clip-event span's extent.

    Args:
        store: The provenance store.

    Returns:
        The extents.
    """
    return [
        entity.extent
        for entity in live_entities(store, "span")
        if entity.attributes.get("family") == "clip" and entity.extent is not None
    ]


def _span_scores(store: ProvStore, measurement_name: str) -> dict[str, dict[str, float]]:
    """Per-span classifier scores, keyed by span id and reduced to the best score per label.

    Args:
        store: The provenance store.
        measurement_name: ``"span_yamnet"`` or ``"span_hear"``.

    Returns:
        ``{span_id: {label: score}}``.
    """
    by_span: dict[str, dict[str, float]] = {}
    for measurement in find_measurements(store, measurement_name):
        span_id = measurement.attributes.get("span_id")
        if span_id is None:
            continue
        slot = by_span.setdefault(str(span_id), {})
        for label, score in (measurement.attributes.get("scores") or {}).items():
            slot[str(label)] = max(slot.get(str(label), 0.0), float(score))
    return by_span


def _top_labels(per_span: dict[str, dict[str, float]], limit: int) -> list[str]:
    """The labels appearing on the most spans.

    Args:
        per_span: :func:`_span_scores`'s result.
        limit: How many to keep.

    Returns:
        The labels, most frequent first.
    """
    counts: dict[str, int] = {}
    for scores in per_span.values():
        for label in scores:
            counts[label] = counts.get(label, 0) + 1
    return [label for label, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]]


def _words(store: ProvStore) -> list[dict[str, Any]]:
    """Every live consensus word with an extent.

    Args:
        store: The provenance store.

    Returns:
        One dict per word.
    """
    return [
        {"extent": entity.extent, "text": str(entity.attributes.get("text") or "")}
        for entity in live_entities(store, "word")
        if entity.extent is not None
    ]


def _squim_by_span(store: ProvStore) -> dict[str, dict[str, float | None]]:
    """SQUIM's three metrics per span.

    Args:
        store: The provenance store.

    Returns:
        ``{span_id: {stoi, pesq, si_sdr}}``.
    """
    by_span: dict[str, dict[str, float | None]] = {}
    for entity in live_entities(store, "assertion"):
        if entity.attributes.get("name") != "squim" or "stoi" not in entity.attributes:
            continue
        for span_id in store.derived_from(entity.id):
            by_span[span_id] = {
                "stoi": entity.attributes.get("stoi"),
                "pesq": entity.attributes.get("pesq"),
                "si_sdr": entity.attributes.get("si_sdr"),
            }
    return by_span


def _kind_entities(store: ProvStore) -> list[Entity]:
    """TAXONOMY's whole-file kind elements.

    Args:
        store: The provenance store.

    Returns:
        The live kind entities.
    """
    return [entity for entity in live_entities(store, "kind")]


def _label_summaries(store: ProvStore) -> dict[str, Entity]:
    """Each classifier's whole-file label-score summary.

    Args:
        store: The provenance store.

    Returns:
        ``{classifier: entity}`` over those TAXONOMY summarised.
    """
    found: dict[str, Entity] = {}
    for classifier in _SUMMARISED_CLASSIFIERS:
        summary = find_measurement(store, f"{classifier}_label_summary")
        if summary is not None:
            found[classifier] = summary
    return found


def taxonomy_summary_lines(store: ProvStore, style: FigureStyle) -> list[str]:
    """The whole-file taxonomy readout, as the lines the panel prints.

    Two aggregations, both file-scoped: each classifier's label-score distribution over every window
    it produced, and each kind's folded state with its evidence lines. A line whose derivative is
    absent prints the reason PREPROCESS recorded, so a null configuration key is named on the page
    instead of being filled in with a value this figure invented.

    Args:
        store: The provenance store.
        style: The drawing configuration, for how many labels to list.

    Returns:
        The lines, in print order.
    """
    lines: list[str] = ["WHOLE-FILE CLASSIFICATION SUMMARY"]
    absent = _absent_reasons(store)
    summaries = _label_summaries(store)
    if not summaries:
        lines.append("  no classifier produced a label-score summary")
    for classifier in _SUMMARISED_CLASSIFIERS:
        summary = summaries.get(classifier)
        if summary is None:
            reason = absent.get(f"{classifier}_scores")
            detail = f" — {reason}" if reason else ""
            lines.append(f"  {classifier}: absent{detail}")
            continue
        attributes = summary.attributes
        labels: dict[str, dict[str, float]] = attributes.get("labels") or {}
        head = (
            f"  {classifier}: {attributes.get('n_windows')} windows "
            f"@ {attributes.get('win_length_s')}s/{attributes.get('hop_s')}s hop, {len(labels)} labels"
        )
        lines.append(head)
        for label, stats in list(labels.items())[: style.summary_labels]:
            lines.append(
                f"      {label:<28} peak {float(stats['peak']):.3f}  "
                f"median {float(stats['median']):.3f}  in {int(stats['n_windows'])} windows"
            )
    lines.append("")
    lines.append("KIND STATES AND EVIDENCE LINES")
    kinds = _kind_entities(store)
    if not kinds:
        lines.append("  TAXONOMY wrote no kind element")
        return lines
    for entity in sorted(kinds, key=lambda item: str(item.attributes.get("kind"))):
        kind = str(entity.attributes.get("kind"))
        lines.append(f"  {kind}: {entity.attributes.get('state')}")
        for name, line in (entity.attributes.get("lines") or {}).items():
            floor = line.get("floor")
            floor_text = "floor —" if floor is None else f"floor {floor}"
            body = (
                f"      {name:<18} {str(line.get('state')):<12} "
                f"{line.get('evidence')} {line.get('unit')}  ({floor_text})"
            )
            lines.append(body)
            if line.get("state") != "unavailable":
                continue
            for source in _LINE_SOURCE.get((kind, name), ()):
                reason = absent.get(source)
                if reason:
                    lines.append(f"          {source} absent: {reason}")
    return lines


def _spectrogram_panel(
    axis: Axes,
    power: np.ndarray | None,
    hop_s: float,
    sampling_rate: int,
    window: tuple[float, float],
    title: str,
    style: FigureStyle,
    absent_note: str,
) -> None:
    """Display a spectrogram the pipeline already wrote, never a fresh STFT of similar shape.

    Args:
        axis: The panel.
        power: The persisted power array, or None when it is absent.
        hop_s: The hop the STFT used.
        sampling_rate: The stream's rate, fixing the bin axis.
        window: The page's ``(start, end)``.
        title: The panel title.
        style: The drawing configuration.
        absent_note: What to print when ``power`` is None.
    """
    t0, t1 = window
    axis.set_xlim(t0, t1)
    axis.set_title(title, fontsize=style.title_fontsize)
    axis.set_ylabel("Hz")
    if power is None:
        _absent_panel(axis, window, absent_note, style)
        return
    axis.set_ylim(0, sampling_rate / 2)
    frame_t = np.arange(power.shape[1]) * hop_s
    lo = int(np.searchsorted(frame_t, t0, side="left"))
    hi = int(np.searchsorted(frame_t, t1, side="right"))
    if hi - lo < 1:
        return
    seg_db = 10.0 * np.log10(np.maximum(power[:, lo:hi], 1e-12))
    vmax = float(seg_db.max())
    axis.imshow(
        seg_db,
        origin="lower",
        aspect="auto",
        cmap=style.cmap_spectrogram,
        interpolation="nearest",
        extent=(float(frame_t[lo]), float(frame_t[hi - 1] + hop_s), 0.0, sampling_rate / 2.0),
        vmin=vmax - style.spectrogram_dynamic_range_db,
        vmax=vmax,
        zorder=1,
    )
    axis.set_xlim(t0, t1)


def _absent_panel(axis: Axes, window: tuple[float, float], note: str, style: FigureStyle) -> None:
    """Say which element is missing and why, in the panel that would have drawn it.

    Args:
        axis: The panel.
        window: The page's ``(start, end)``.
        note: The reason, as recorded by the node that could not produce the element.
        style: The drawing configuration.
    """
    t0, t1 = window
    axis.set_xlim(t0, t1)
    axis.set_yticks([])
    # Axes fraction, not data coordinates, and above the padding hatch: an absent note placed at the
    # page's time centre lands inside a padded tail on the final page and is drawn over.
    axis.text(
        0.01,
        0.5,
        note,
        transform=axis.transAxes,
        ha="left",
        va="center",
        fontsize=style.absent_fontsize,
        color="0.3",
        style="italic",
        zorder=8,
    )
    for spine in axis.spines.values():
        spine.set_edgecolor("0.75")


def _waveform_panel(
    axis: Axes,
    samples: np.ndarray,
    sampling_rate: int,
    envelope_db: np.ndarray | None,
    floor_db: float | None,
    continuity: np.ndarray | None,
    window: tuple[float, float],
    style: FigureStyle,
    k_db: float | None,
    cut_level: float | None,
    cut_percentile: float | None,
) -> None:
    """The conditioned waveform, its envelope and floor, and the continuity trace on their own scales.

    Args:
        axis: The panel.
        samples: The conditioned stream.
        sampling_rate: Its rate.
        envelope_db: PREPROCESS's persisted envelope, or None when absent.
        floor_db: Its floor, or None.
        continuity: The continuity trace, or None when the narrowband array is absent.
        window: The page's ``(start, end)``.
        style: The drawing configuration.
        k_db: Amplitude's detection threshold above the floor.
        cut_level: The trace value the rank cut lands on.
        cut_percentile: The percentile that produced it.
    """
    t0, t1 = window
    times = np.arange(len(samples)) / sampling_rate
    mask = (times >= t0) & (times < t1)
    axis.plot(times[mask], samples[mask], linewidth=0.3, color="0.4", zorder=1)
    axis.set_ylim(-1.05, 1.05)
    axis.set_xlim(t0, t1)
    axis.set_ylabel("Amplitude")
    axis.set_title("conditioned waveform + envelope + floor + continuity", fontsize=style.title_fontsize)

    if envelope_db is not None and floor_db is not None:
        twin = axis.twinx()
        window_env = envelope_db[: len(times)][mask[: len(envelope_db)]] if len(envelope_db) else envelope_db
        twin.plot(
            times[mask][: len(window_env)],
            window_env,
            color=style.colour_primary,
            linewidth=0.9,
            label="envelope dBFS",
            zorder=2,
        )
        twin.axhline(floor_db, color="firebrick", linewidth=1.0, linestyle="--", label=f"floor ({floor_db:.1f} dBFS)")
        if k_db is not None:
            twin.axhline(
                floor_db + k_db,
                color="firebrick",
                linewidth=1.2,
                alpha=0.9,
                label=f"k_db ({floor_db + k_db:.1f} dBFS)",
            )
        twin.set_ylabel("dBFS")
        finite = window_env[np.isfinite(window_env)] if len(window_env) else window_env
        if len(finite):
            twin.set_ylim(min(floor_db, float(finite.min())) - 5, float(finite.max()) + 5)
        handles, labels = twin.get_legend_handles_labels()
    else:
        handles, labels = [], []

    if continuity is not None and len(continuity):
        cont_axis = axis.twinx()
        cont_axis.spines["right"].set_position(("outward", 55))
        trace = continuity[: len(times)]
        cont_axis.plot(
            times[: len(trace)][mask[: len(trace)]],
            trace[mask[: len(trace)]],
            color=style.colour_continuity,
            linewidth=0.8,
            label="continuity",
            zorder=2,
        )
        if cut_level is not None:
            cont_axis.axhline(
                cut_level,
                color="darkgreen",
                linewidth=1.2,
                alpha=0.9,
                label=f"rank cut p{cut_percentile:g} ({cut_level:.3f})",
            )
        cont_axis.set_ylim(0.0, 1.05)
        cont_axis.set_ylabel("continuity", color=style.colour_continuity)
        cont_axis.tick_params(axis="y", colors=style.colour_continuity)
        extra_handles, extra_labels = cont_axis.get_legend_handles_labels()
        handles, labels = handles + extra_handles, labels + extra_labels
    if handles:
        axis.legend(handles, labels, loc="upper right", fontsize=style.tick_fontsize)


def _span_row_cells(spans: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Per-source rows of every proposal, flagged by whether dedup kept it.

    Args:
        spans: :func:`_spans`' result.

    Returns:
        ``{code: [cell, ...]}``.
    """
    cells: dict[str, list[dict[str, Any]]] = {code: [] for code in _SPAN_ROWS}
    for span in spans:
        start, end = span["extent"]
        code = _span_code(str(span["signal"] or ""), str(span["measure"] or ""))
        if code in cells:
            cells[code].append({"start": start, "end": end, "owned": True, "span": span})
        for record in span["corroborated_by"]:
            record_code = _span_code(str(record.get("signal") or ""), str(record.get("measure") or ""))
            if record_code in cells:
                cells[record_code].append(
                    {
                        "start": float(record["start"]),
                        "end": float(record["end"]),
                        "owned": False,
                        "span": span,
                    }
                )
    return cells


def _span_lane_panel(
    axis: Axes,
    spans: list[dict[str, Any]],
    clips: list[tuple[float, float]],
    window: tuple[float, float],
    style: FigureStyle,
) -> None:
    """One compact row per proposing source, hatched where dedup kept the proposal.

    Args:
        axis: The panel.
        spans: :func:`_spans`' result.
        clips: Clip-event extents.
        window: The page's ``(start, end)``.
        style: The drawing configuration.
    """
    from matplotlib.patches import Rectangle

    t0, t1 = window
    axis.set_title(
        "spans by source (E=envelope C=continuity A=asr S=normalization) — hatched=kept after dedup",
        fontsize=style.title_fontsize,
    )
    axis.set_xlim(t0, t1)
    n_rows = len(_SPAN_ROWS)
    axis.set_yticks(range(n_rows))
    axis.set_yticklabels(list(reversed(_SPAN_ROWS)), fontsize=style.tick_fontsize)
    axis.set_ylim(-0.5, n_rows - 0.5)
    axis.tick_params(axis="y", length=0)
    for row in range(n_rows - 1):
        axis.axhline(row + 0.5, color="0.85", linewidth=0.5, zorder=0)
    for start, end in clips:
        if end < t0 or start > t1:
            continue
        axis.axvspan(max(start, t0), min(end, t1), color=style.colour_clip, alpha=0.12, zorder=1)
    cells = _span_row_cells(spans)
    for index, code in enumerate(_SPAN_ROWS):
        y = n_rows - 1 - index
        for cell in cells[code]:
            left, right = max(cell["start"], t0), min(cell["end"], t1)
            if right <= left:
                continue
            kept = bool(cell["owned"])
            edge = style.colour_clip if cell["span"]["contains_clip"] else "0.15"
            axis.add_patch(
                Rectangle(
                    (left, y - 0.36),
                    right - left,
                    0.72,
                    facecolor=style.row_colour(code),
                    edgecolor=edge,
                    linewidth=0.8 if kept else 0.4,
                    hatch="///" if kept else None,
                    alpha=0.9,
                    zorder=3 if kept else 2,
                )
            )


def _raster_panel(
    axis: Axes,
    spans: list[dict[str, Any]],
    per_span: dict[str, dict[str, float]],
    labels: list[str],
    cmap_name: str,
    title: str,
    window: tuple[float, float],
    style: FigureStyle,
    absent_note: str,
    marker_size: float | None = None,
) -> None:
    """One fixed row per label, each span's cell coloured by its score.

    Args:
        axis: The panel.
        spans: :func:`_spans`' result.
        per_span: :func:`_span_scores`' result.
        labels: The rows to draw.
        cmap_name: Colormap.
        title: Panel title.
        window: The page's ``(start, end)``.
        style: The drawing configuration.
        absent_note: What to print when the measurement never ran.
        marker_size: Cell area, defaulting to the style's.
    """
    import matplotlib.pyplot as plt

    t0, t1 = window
    axis.set_title(title, fontsize=style.title_fontsize)
    axis.set_xlim(t0, t1)
    if not labels:
        _absent_panel(axis, window, absent_note, style)
        return
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=style.tick_fontsize)
    axis.set_ylim(-0.5, len(labels) - 0.5)
    cmap = plt.get_cmap(cmap_name)
    for span in spans:
        start, end = span["extent"]
        if end < t0 or start > t1:
            continue
        mid = max(start, t0) + (min(end, t1) - max(start, t0)) / 2
        scores = per_span.get(span["id"], {})
        for row, label in enumerate(labels):
            score = scores.get(label)
            if score is None:
                continue
            axis.scatter(
                [mid],
                [row],
                s=marker_size or style.marker_size,
                marker="s",
                c=[cmap(0.25 + 0.75 * max(0.0, min(1.0, score)))],
                edgecolors="0.3",
                linewidths=0.4,
                zorder=3,
            )
            axis.text(mid, row, f"{score:.2f}", ha="center", va="center", fontsize=style.cell_fontsize, zorder=4)


def _squim_panel(
    axis: Axes,
    spans: list[dict[str, Any]],
    squim: dict[str, dict[str, float | None]],
    window: tuple[float, float],
    style: FigureStyle,
) -> None:
    """SQUIM's three metrics per span.

    Args:
        axis: The panel.
        spans: :func:`_spans`' result.
        squim: :func:`_squim_by_span`'s result.
        window: The page's ``(start, end)``.
        style: The drawing configuration.
    """
    import matplotlib.pyplot as plt

    t0, t1 = window
    metrics = ("stoi", "pesq", "si_sdr")
    ranges = {"stoi": (0.0, 1.0), "pesq": (1.0, 4.5), "si_sdr": (-10.0, 30.0)}
    axis.set_title("SQUIM per span (STOI / PESQ / SI-SDR)", fontsize=style.title_fontsize)
    axis.set_xlim(t0, t1)
    if not squim:
        _absent_panel(axis, window, "no SQUIM assertion in the store", style)
        return
    axis.set_yticks(range(len(metrics)))
    axis.set_yticklabels(list(metrics), fontsize=style.tick_fontsize)
    axis.set_ylim(-0.5, len(metrics) - 0.5)
    cmap = plt.get_cmap(style.cmap_squim)
    for span in spans:
        start, end = span["extent"]
        if end < t0 or start > t1:
            continue
        mid = max(start, t0) + (min(end, t1) - max(start, t0)) / 2
        scores = squim.get(span["id"])
        if scores is None:
            continue
        for row, metric in enumerate(metrics):
            value = scores.get(metric)
            if value is None:
                continue
            low, high = ranges[metric]
            frac = max(0.0, min(1.0, (float(value) - low) / (high - low)))
            axis.scatter(
                [mid],
                [row],
                s=style.marker_size,
                marker="s",
                c=[cmap(0.25 + 0.75 * frac)],
                edgecolors="0.3",
                linewidths=0.4,
                zorder=3,
            )
            axis.text(
                mid,
                row,
                f"{float(value):.2f}",
                ha="center",
                va="center",
                fontsize=style.cell_fontsize,
                zorder=4,
            )


def _asr_lane_panel(
    axis: Axes, words: list[dict[str, Any]], window: tuple[float, float], style: FigureStyle, absent_note: str
) -> None:
    """One bar per consensus word with its text drawn on the bar.

    Args:
        axis: The panel.
        words: :func:`_words`' result.
        window: The page's ``(start, end)``.
        style: The drawing configuration.
        absent_note: What to print when no consensus transcript reached the store.
    """
    from matplotlib.patches import Rectangle

    t0, t1 = window
    axis.set_title("consensus ASR", fontsize=style.title_fontsize)
    axis.set_xlim(t0, t1)
    if not words:
        _absent_panel(axis, window, absent_note, style)
        return
    axis.set_ylim(-0.5, 2.5)
    axis.set_yticks([])
    here = [word for word in words if word["extent"][1] >= t0 and word["extent"][0] <= t1]
    for index, word in enumerate(here):
        start, end = word["extent"]
        row = index % 3
        axis.add_patch(
            Rectangle(
                (max(start, t0), row - 0.4),
                max(min(end, t1) - max(start, t0), 0.01),
                0.8,
                facecolor=style.word_colours[index % len(style.word_colours)],
                alpha=0.8,
            )
        )
        axis.text(
            max(start, t0), row, word["text"], fontsize=style.cell_fontsize, ha="left", va="center", color="white"
        )


def _taxonomy_panel(axis: Axes, lines: list[str], style: FigureStyle) -> None:
    """The whole-file taxonomy readout, monospaced and off the shared time axis.

    Args:
        axis: The panel.
        lines: :func:`taxonomy_summary_lines`' result.
        style: The drawing configuration.
    """
    axis.set_axis_off()
    axis.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=style.text_fontsize,
        family="monospace",
    )


def _continuity_trace(
    store: ProvStore, run_dir: Path, config: TriageConfig, sampling_rate: int, n_samples: int
) -> np.ndarray | None:
    """The continuity trace, recomputed from the persisted narrowband spectrogram.

    PREPROCESS computes this trace inside ``_spans`` and does not persist it, so unlike every other
    curve here it cannot simply be read back. It is recomputed from the stored narrowband array under
    the run's own configuration, which is deterministic but not the same guarantee as reading a
    sidecar.

    Args:
        store: The provenance store.
        run_dir: The run directory.
        config: The run's configuration.
        sampling_rate: The stream's rate.
        n_samples: How many samples the trace is resampled to.

    Returns:
        The trace, or None when the narrowband spectrogram is absent.
    """
    from senselab.audio.tasks.envelope.api import ButterworthSmoothing
    from senselab.audio.tasks.spectral_continuity.api import spectral_continuity

    power = _npz(run_dir, store, "spectrogram_narrowband", "spectrogram")
    if power is None:
        return None
    smoothing = ButterworthSmoothing(
        cutoff_hz=float(config.require("envelope.lowpass_hz")),
        order=int(config.require("envelope.filter_order")),
    )
    return spectral_continuity(
        np.sqrt(np.maximum(power, 0.0)),
        hop_s=float(config.require("spectrogram.hop_ms")) / 1000.0,
        sampling_rate=sampling_rate,
        n_samples=n_samples,
        smoothing=smoothing,
    )


def _cut_level(trace: np.ndarray | None, cut_percentile: float) -> float | None:
    """The trace value the rank cut lands on.

    Args:
        trace: The continuity trace, or None.
        cut_percentile: The configured percentile.

    Returns:
        The level, or None when there is no trace or the cut takes no samples.
    """
    if trace is None or not len(trace):
        return None
    n_change_points = int(round(len(trace) * cut_percentile / 100.0))
    if n_change_points <= 0:
        return None
    return float(np.sort(trace, kind="stable")[n_change_points - 1])


def preprocess_figure(
    store: ProvStore,
    figure_dir: Path,
    config: TriageConfig,
    *,
    run_dir: Path,
    style: FigureStyle | None = None,
    stem: str | None = None,
) -> dict[str, Path]:
    """Draw PREPROCESS's and TAXONOMY's output from the store, one image per page.

    Reads only what the two nodes left behind and writes nothing back to the store, so it can be
    re-invoked over a completed run directory. The pipeline configuration is read and never
    overridden: a panel whose element is absent under the packaged configuration says which
    derivative is missing and prints the reason the producing node recorded.

    Args:
        store: The provenance store, after PREPROCESS and TAXONOMY have run.
        figure_dir: Where the images are written; created if absent.
        config: The run's configuration, read for the values the panels annotate.
        run_dir: Where PREPROCESS wrote its streams and derivative sidecars.
        style: How to draw. Defaults to :class:`FigureStyle`, whose every field governs the drawing
            alone.
        stem: The filename stem, defaulting to the run id.

    Returns:
        ``{"page01": path, ...}`` in page order.

    Raises:
        LookupError: If no conditioned stream is in the store, since there is nothing to draw
            against and a blank page would misreport that as a measurement.
    """
    import matplotlib.pyplot as plt

    style = style or FigureStyle()
    stream_path = _stream_path(store, run_dir)
    if stream_path is None:
        raise LookupError("no conditioned stream in the store; PREPROCESS must run before FIGURE")
    samples, sampling_rate = sf.read(str(stream_path), dtype="float32", always_2d=False)
    samples = np.asarray(samples, dtype="float64")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    duration_s = len(samples) / float(sampling_rate)

    absent = _absent_reasons(store)
    envelope = _npz(run_dir, store, "energy_envelope", "envelope_dbfs")
    floor_array = _npz(run_dir, store, "energy_envelope", "floor_dbfs")
    floor_db = float(floor_array[0]) if floor_array is not None and len(floor_array) else None
    wideband = _npz(run_dir, store, "spectrogram_wideband", "spectrogram")
    hop_s = float(config.require("spectrogram.hop_ms")) / 1000.0
    trace = _continuity_trace(store, run_dir, config, sampling_rate, len(samples))
    cut_percentile = float(config.require("spans.continuity_cut_percentile"))

    spans = _spans(store)
    clips = _clip_extents(store)
    yamnet = _span_scores(store, "span_yamnet")
    hear = _span_scores(store, "span_hear")
    words = _words(store)
    squim = _squim_by_span(store)
    summary_lines = taxonomy_summary_lines(store, style)

    wideband_title = (
        f"wideband spectrogram ({float(config.require('spectrogram.wideband_window_ms')):.0f} ms window, "
        f"{float(config.require('spectrogram.hop_ms')):.0f} ms hop) — the speech-analysis view; "
        "continuity runs on the narrowband array, not this one"
    )

    figure_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for index, window in enumerate(pages(duration_s, style), start=1):
        figure: Figure
        figure, axes = plt.subplots(
            len(style.height_ratios),
            1,
            figsize=style.figure_inches,
            constrained_layout=True,
            gridspec_kw={"height_ratios": list(style.height_ratios)},
        )
        (
            axis_wide,
            axis_wave,
            axis_spans,
            axis_yamnet,
            axis_hear,
            axis_squim,
            axis_asr,
            axis_taxonomy,
        ) = axes
        timed = [axis_wide, axis_wave, axis_spans, axis_yamnet, axis_hear, axis_squim, axis_asr]

        _spectrogram_panel(
            axis_wide,
            wideband,
            hop_s,
            sampling_rate,
            window,
            wideband_title,
            style,
            absent.get("spectrogram_wideband", "spectrogram_wideband is absent from the store"),
        )
        _waveform_panel(
            axis_wave,
            samples,
            sampling_rate,
            envelope,
            floor_db,
            trace,
            window,
            style,
            k_db=float(config.require("spans.k_db")),
            cut_level=_cut_level(trace, cut_percentile),
            cut_percentile=cut_percentile,
        )
        _span_lane_panel(axis_spans, spans, clips, window, style)
        _raster_panel(
            axis_yamnet,
            spans,
            yamnet,
            _top_labels(yamnet, style.top_labels),
            style.cmap_yamnet,
            f"YAMNet per-span labels (top-{style.top_labels})",
            window,
            style,
            absent.get("span_yamnet", "span_yamnet is absent from the store"),
        )
        _raster_panel(
            axis_hear,
            spans,
            hear,
            _top_labels(hear, style.top_labels),
            style.cmap_hear,
            f"HeAR per-span labels (top-{style.top_labels})",
            window,
            style,
            absent.get("span_hear", "span_hear is absent from the store"),
            marker_size=150.0,
        )
        _squim_panel(axis_squim, spans, squim, window, style)
        _asr_lane_panel(
            axis_asr,
            words,
            window,
            style,
            absent.get("consensus_transcript", "no consensus word in the store"),
        )
        _taxonomy_panel(axis_taxonomy, summary_lines, style)

        for axis in timed:
            axis.set_xlim(*window)
            axis.tick_params(axis="x", labelsize=style.tick_fontsize)
        # Only the last timed panel carries the tick labels: repeated on every panel they collide
        # with the title of the panel below, which is what the scratch tool's pages did.
        for axis in timed[:-1]:
            axis.tick_params(axis="x", labelbottom=False)
        padded = _mark_padding(timed, duration_s, window[1], style)
        axis_asr.set_xlabel("Time (s)")
        pad_note = "  ·  padded to a uniform page" if padded else ""
        figure.suptitle(
            f"{stem or store.run_id} — page {index}, {window[0]:.0f}-{window[1]:.0f}s of {duration_s:.2f}s{pad_note}",
            fontsize=10,
        )
        out_path = figure_dir / f"{stem or store.run_id}__page{index:02d}.png"
        figure.savefig(out_path, dpi=style.dpi)
        plt.close(figure)
        written[f"page{index:02d}"] = out_path

    (figure_dir / "taxonomy_summary.json").write_text(json.dumps({"lines": summary_lines}, indent=1) + "\n")
    written["taxonomy_summary"] = figure_dir / "taxonomy_summary.json"
    return written
