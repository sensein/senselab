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
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
from matplotlib.text import Text

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


_SUMMARY_LABEL_WIDTH = 20
"""Characters a label field takes in the whole-file summary."""

_SUMMARY_COLUMN_WIDTH = 52
"""Characters one classifier column takes when the summary is laid out across the page."""


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
        top_labels: How many of its own highest-scoring labels each span contributes to a
            per-span raster's row set.
        raster_row_ratio: Height one raster row takes, as a share of the page's height ratios. A
            raster is at least its declared height and grows past it rather than compressing its
            rows.
        raster_rows_scope: Where a raster's row set is unioned. Only ``"file"`` is implemented:
            every page draws the same rows in the same order, so a label can be scanned down
            across pages and a page carrying none of a row's label shows that row empty.
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
        word_fill: The consensus-word bar fill.
        word_text_colour: The consensus-word text colour.
        title_fontsize: Panel title size.
        tick_fontsize: Axis tick-label size.
        cell_fontsize: Score text drawn inside a raster cell.
        marker_size: Raster cell area.
        text_fontsize: The taxonomy panel's monospaced lines.
        absent_fontsize: The note a panel prints when its element is absent.
        cell_ramp: The span of a colormap a value is mapped onto. Ink means attention: the end
            of the ramp that matters runs to near-full colour, and the other end to near-white so
            it recedes into the page.
        raster_cell_height: A cell's height in row units, leaving a gap between rows so a dense
            band of cells does not read as one solid row.
        raster_min_cell_s: The narrowest a cell is drawn, in seconds. A cell takes its span's width,
            and a span shorter than this would otherwise render as an invisible hairline.
        also_write_pngs: Whether each page is additionally written as its own PNG. Off: a recording's
            pages are one PDF, because 388 recordings emitted 538 loose pages whose only ordering was
            their filename. A test that must inspect one page's pixels turns it on.
        colorbar_width_ratio: The colorbar column's width, as a fraction of the panel's. The column
            exists on every row so that every panel is drawn to the same width and the shared time
            axis stays aligned; rows with nothing to scale leave their slot blank.
        colorbar_tick_fontsize: The colorbar's tick labels.
        squim_ranges: The value range each SQUIM row is normalised over for colour. One shared scale
            across rows would be meaningless, the three metrics having unrelated units.
        cell_floor_fontsize: The smallest a raster cell's score text is shrunk to before it is
            dropped. The cell itself is never dropped.
        waveform_headroom: The page's own peak amplitude is scaled by this to set the waveform's
            y-limits, so a signal well below full scale still fills its panel.
        waveform_min_amplitude: A floor on those limits, so a near-silent page does not zoom into
            its own noise.
        absent_height_ratio: The height an absent panel collapses to, freeing its remaining share
            for the panels that have something to draw.
        raster_paint_floor: A raster cell scoring below this is left unpainted, so the eye finds the
            spans where something registered. A display choice only: it changes no measurement, and the
            row stays present because the label is still part of the file's union.
        asr_rows: How many staggered rows the consensus-word lane uses.
        asr_row_height: The bar height within one word-lane row, in row units.
    """

    page_seconds: float = 20.0
    pad_short_pages: bool = True
    figure_inches: tuple[float, float] = (14.0, 17.0)
    dpi: int = 130
    height_ratios: tuple[float, ...] = (1.32, 2.0, 0.5, 0.5, 0.32, 0.5, 0.40, 1.6)
    spectrogram_dynamic_range_db: float = 80.0
    top_labels: int = 4
    raster_rows_scope: str = "file"
    raster_row_ratio: float = 0.075
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
    cmap_squim: str = "Purples_r"
    word_fill: str = "#fdd0a2"
    word_text_colour: str = "black"
    title_fontsize: float = 9.0
    tick_fontsize: float = 6.0
    cell_fontsize: float = 5.0
    marker_size: float = 260.0
    text_fontsize: float = 7.5
    absent_fontsize: float = 7.0
    cell_floor_fontsize: float = 4.0
    cell_ramp: tuple[float, float] = (0.05, 0.95)
    raster_cell_height: float = 0.72
    raster_min_cell_s: float = 0.02
    also_write_pngs: bool = False
    colorbar_width_ratio: float = 0.014
    colorbar_tick_fontsize: float = 5.0
    squim_ranges: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {"stoi": (0.0, 1.0), "pesq": (1.0, 4.5), "si_sdr": (-10.0, 30.0)}
    )
    waveform_headroom: float = 1.15
    waveform_min_amplitude: float = 0.02
    absent_height_ratio: float = 0.2
    raster_paint_floor: float = 0.1
    asr_rows: int = 3
    asr_row_height: float = 0.44
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
    # Rotated inside the band: a padded tail is often a fraction of a second wide, and a horizontal
    # label centred on it overflows onto the recording it is meant to be distinguished from.
    axes[0].text(
        (duration_s + t1) / 2,
        0.5,
        "padding — recording ended",
        transform=axes[0].get_xaxis_transform(),
        ha="center",
        va="center",
        rotation=90,
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
        ``{span_id: {label: score}}``, read from ``raw_scores`` — the model's own output for the
        window, written whatever the configuration says. No labelling threshold takes part.
    """
    by_span: dict[str, dict[str, float]] = {}
    for measurement in find_measurements(store, measurement_name):
        span_id = measurement.attributes.get("span_id")
        if span_id is None:
            continue
        slot = by_span.setdefault(str(span_id), {})
        for label, score in (measurement.attributes.get("raw_scores") or {}).items():
            slot[str(label)] = max(slot.get(str(label), 0.0), float(score))
    return by_span


def _raster_rows(
    per_span: dict[str, dict[str, float]], per_span_top_k: int, scope: str, floor: float | None = None
) -> list[str]:
    """The raster's row set: the union of each span's highest-scoring labels, over the whole file.

    Args:
        per_span: :func:`_span_scores`'s result, every span in the recording.
        per_span_top_k: How many of its own labels each span contributes.
        scope: Where the union is taken. Only ``"file"`` is implemented.
        floor: A label whose file-wide peak falls under this contributes no row, or ``None`` to keep
            every label a span ranked. Without it a span where nothing fires still contributes its
            four highest, which are all near zero.

    Returns:
        The rows, highest file-wide peak first, so a row holds the same position on every page.

    Raises:
        ValueError: If ``scope`` is not ``"file"``.
    """
    if scope != "file":
        raise ValueError(f"raster_rows_scope must be 'file', got {scope!r}")
    rows: set[str] = set()
    peaks: dict[str, float] = {}
    for scores in per_span.values():
        carried = sorted(
            ((label, score) for label, score in scores.items() if float(score) > 0.0),
            key=lambda pair: (-float(pair[1]), pair[0]),
        )
        rows.update(label for label, _ in carried[:per_span_top_k])
        for label, score in scores.items():
            peaks[label] = max(peaks.get(label, 0.0), float(score))
    if floor is not None:
        rows = {label for label in rows if peaks.get(label, 0.0) >= floor}
    return [label for label in sorted(rows, key=lambda label: (-peaks.get(label, 0.0), label))]


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


def _summary_sections(store: ProvStore, style: FigureStyle) -> tuple[list[list[str]], list[str]]:
    """The whole-file readout as its parts: one block per classifier, then the kind block.

    Args:
        store: The provenance store.
        style: The drawing configuration, for how many labels to list.

    Returns:
        ``(classifier_blocks, kind_lines)``. Each classifier block leads with its own name, so a
        block stands alone in a column.
    """
    absent = _absent_reasons(store)
    summaries = _label_summaries(store)
    blocks: list[list[str]] = []
    for classifier in _SUMMARISED_CLASSIFIERS:
        summary = summaries.get(classifier)
        if summary is None:
            reason = absent.get(f"{classifier}_scores")
            blocks.append([f"{classifier}: absent", f"  {reason}" if reason else "  no reason recorded"])
            continue
        attributes = summary.attributes
        labels: dict[str, dict[str, float]] = attributes.get("labels") or {}
        block = [
            f"{classifier}: {attributes.get('n_windows')} win "
            f"@ {attributes.get('win_length_s')}/{attributes.get('hop_s')}s, {len(labels)} labels"
        ]
        ranked = sorted(
            ((name, stats) for name, stats in labels.items() if float(stats["peak"]) > 0.0),
            key=lambda item: (float(item[1]["peak"]), float(item[1]["median"])),
            reverse=True,
        )
        if not ranked:
            block.append("  every label scored 0.000")
        for label, stats in ranked[: style.summary_labels]:
            block.append(
                f"  {label:<{_SUMMARY_LABEL_WIDTH}.{_SUMMARY_LABEL_WIDTH}} "
                f"peak {float(stats['peak']):.2f} median {float(stats['median']):.2f} "
                f"({int(stats['n_windows'])})"
            )
        blocks.append(block)

    kind_lines: list[str] = []
    kinds = _kind_entities(store)
    if not kinds:
        return blocks, ["  TAXONOMY wrote no kind element"]
    for entity in sorted(kinds, key=lambda item: str(item.attributes.get("kind"))):
        kind = str(entity.attributes.get("kind"))
        kind_lines.append(f"  {kind}: {entity.attributes.get('state')}")
        for name, line in (entity.attributes.get("lines") or {}).items():
            floor = line.get("floor")
            floor_text = "floor —" if floor is None else f"floor {floor}"
            unit = line.get("unit")
            evidence = f"{line.get('evidence')}" + (f" {unit}" if unit else "")
            kind_lines.append(f"      {name:<18} {str(line.get('state')):<12} {evidence}  ({floor_text})")
            if line.get("state") != "unavailable":
                continue
            why = line.get("why")
            if why:
                kind_lines.append(f"          {why}")
            for source in _LINE_SOURCE.get((kind, name), ()):
                reason = absent.get(source)
                if reason:
                    kind_lines.append(f"          {source} absent: {reason}")
    return blocks, kind_lines


def taxonomy_summary_lines(store: ProvStore, style: FigureStyle) -> list[str]:
    """The whole-file taxonomy readout, one line under another, as the sidecar JSON records it.

    Two aggregations, both file-scoped: each classifier's label-score distribution over every window
    it produced, and each kind's folded state with its evidence lines. A line whose derivative is
    absent prints the reason PREPROCESS recorded, so a null configuration key is named instead of
    being filled in with a value this figure invented.

    Args:
        store: The provenance store.
        style: The drawing configuration, for how many labels to list.

    Returns:
        The lines, in print order.
    """
    blocks, kind_lines = _summary_sections(store, style)
    lines: list[str] = ["WHOLE-FILE CLASSIFICATION SUMMARY"]
    if not _label_summaries(store):
        lines.append("  no classifier produced a label-score summary")
    for block in blocks:
        lines.append(f"  {block[0]}")
        lines.extend(f"    {line.strip()}" for line in block[1:])
    lines.append("")
    lines.append("KIND STATES AND EVIDENCE LINES")
    lines.extend(kind_lines)
    return lines


def summary_panel_lines(store: ProvStore, style: FigureStyle) -> list[str]:
    """The same readout laid out across the page: the classifier blocks side by side in columns.

    Every column is padded to :data:`_SUMMARY_COLUMN_WIDTH`, so the longest possible line is a
    known number of monospaced characters and cannot run past the axis.

    Args:
        store: The provenance store.
        style: The drawing configuration.

    Returns:
        The lines, in print order.
    """
    blocks, kind_lines = _summary_sections(store, style)
    lines: list[str] = ["WHOLE-FILE CLASSIFICATION SUMMARY"]
    depth = max((len(block) for block in blocks), default=0)
    for row in range(depth):
        cells = [block[row] if row < len(block) else "" for block in blocks]
        lines.append(
            "  " + "".join(f"{cell:<{_SUMMARY_COLUMN_WIDTH}.{_SUMMARY_COLUMN_WIDTH}}" for cell in cells).rstrip()
        )
    lines.append("")
    lines.append("KIND STATES AND EVIDENCE LINES")
    lines.extend(kind_lines)
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
    continuity_absent: str,
) -> None:
    """The conditioned waveform, its envelope and floor, and the continuity trace on their own scales.

    The scalar readings go in the title rather than a legend, which at this panel's density covered
    the traces it was labelling.

    Args:
        axis: The panel.
        samples: The conditioned stream.
        sampling_rate: Its rate.
        envelope_db: PREPROCESS's persisted envelope, or None when absent.
        floor_db: Its floor, or None.
        continuity: PREPROCESS's persisted continuity trace, or None when absent.
        window: The page's ``(start, end)``.
        style: The drawing configuration.
        k_db: Amplitude's detection threshold above the floor.
        cut_level: The trace value the rank cut landed on, as PREPROCESS recorded it.
        cut_percentile: The percentile that produced it.
        continuity_absent: What to name when the trace is absent.
    """
    t0, t1 = window
    times = np.arange(len(samples)) / sampling_rate
    mask = (times >= t0) & (times < t1)
    page = samples[mask[: len(samples)]] if len(samples) else samples
    axis.plot(times[mask], samples[mask], linewidth=0.3, color="0.4", zorder=1)
    peak = float(np.abs(page).max()) if len(page) else 0.0
    limit = max(peak * style.waveform_headroom, style.waveform_min_amplitude)
    axis.set_ylim(-limit, limit)
    axis.set_xlim(t0, t1)
    axis.set_ylabel("Amplitude")

    readings = [f"waveform peak {peak:.3f}" if len(page) else "waveform absent"]
    if floor_db is not None:
        readings.append(f"floor {floor_db:.1f} dBFS (dashed)")
        if k_db is not None:
            readings.append(f"k_db {floor_db + k_db:.1f} dBFS (solid)")
    if cut_level is not None and cut_percentile is not None:
        readings.append(f"rank cut p{cut_percentile:g} {cut_level:.3f}")
    axis.set_title(
        "conditioned waveform + envelope + floor + continuity — " + "  ·  ".join(readings),
        fontsize=style.title_fontsize,
    )

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
        twin.axhline(floor_db, color="firebrick", linewidth=1.0, linestyle="--")
        if k_db is not None:
            twin.axhline(floor_db + k_db, color="firebrick", linewidth=1.2, alpha=0.9)
        twin.set_ylabel("dBFS")
        finite = window_env[np.isfinite(window_env)] if len(window_env) else window_env
        if len(finite):
            twin.set_ylim(min(floor_db, float(finite.min())) - 5, float(finite.max()) + 5)

    if continuity is not None and len(continuity):
        cont_axis = axis.twinx()
        cont_axis.spines["right"].set_position(("outward", 55))
        trace = continuity[: len(times)]
        cont_axis.plot(
            times[: len(trace)][mask[: len(trace)]],
            trace[mask[: len(trace)]],
            color=style.colour_continuity,
            linewidth=0.8,
            zorder=2,
        )
        if cut_level is not None:
            cont_axis.axhline(
                cut_level,
                color="darkgreen",
                linewidth=1.2,
                alpha=0.9,
            )
        cont_axis.set_ylim(0.0, 1.05)
        cont_axis.set_ylabel("continuity", color=style.colour_continuity)
        cont_axis.tick_params(axis="y", colors=style.colour_continuity)
    else:
        axis.text(
            0.006,
            0.06,
            continuity_absent,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=style.absent_fontsize,
            style="italic",
            color="0.35",
            zorder=7,
        )


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
    row_absent: dict[str, str],
) -> None:
    """One compact row per proposing source, hatched where dedup kept the proposal.

    A row that proposed nothing anywhere in the recording says why on the row itself, so an empty
    row is never mistaken for a source that ran and found nothing.

    Args:
        axis: The panel.
        spans: :func:`_spans`' result.
        clips: Clip-event extents.
        window: The page's ``(start, end)``.
        style: The drawing configuration.
        row_absent: ``{code: reason}`` for a source that contributed no span to the whole recording.
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
        if not cells[code] and code in row_absent:
            axis.text(
                t0 + (t1 - t0) * 0.004,
                y,
                row_absent[code],
                ha="left",
                va="center",
                fontsize=style.absent_fontsize,
                style="italic",
                color="0.4",
                zorder=4,
            )


def _ramped(cmap_name: str, style: FigureStyle) -> Colormap:
    """The portion of a colormap scores are drawn on.

    Args:
        cmap_name: The full colormap's name.
        style: The drawing configuration, for the ramp's bounds.

    Returns:
        A colormap spanning only ``style.cell_ramp`` of the original, so a cell and the colorbar
        beside it are the same scale.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    low, high = style.cell_ramp
    full = plt.get_cmap(cmap_name)
    return LinearSegmentedColormap.from_list(
        f"{cmap_name}-ramped", [full(low + (high - low) * step / 255.0) for step in range(256)]
    )


def _score_colorbar(axis: Axes, cmap_name: str, style: FigureStyle, *, label: str) -> None:
    """Draw a panel's colour scale in the slot reserved to its right.

    The slot is its own gridspec column rather than space taken from the panel, so adding a scale
    never narrows the panel and the shared time axis stays aligned down the page.

    Args:
        axis: The reserved slot.
        cmap_name: The colormap the panel drew with.
        style: The drawing configuration.
        label: What the scale measures.
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    axis.set_axis_on()
    mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=_ramped(cmap_name, style))
    bar = axis.figure.colorbar(mappable, cax=axis)
    bar.set_label(label, fontsize=style.colorbar_tick_fontsize)
    bar.ax.tick_params(labelsize=style.colorbar_tick_fontsize, length=2, pad=1)


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
    colorbar_axis: Axes,
) -> None:
    """One fixed row per label, each span's cell drawn at its span's width and coloured by its score.

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
        colorbar_axis: The slot to the panel's right, where the score scale is drawn.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    t0, t1 = window
    axis.set_title(title, fontsize=style.title_fontsize)
    axis.set_xlim(t0, t1)
    if not labels:
        _absent_panel(axis, window, absent_note, style)
        colorbar_axis.set_axis_off()
        return
    axis.set_yticks(range(len(labels)))
    axis.set_yticklabels(labels, fontsize=style.tick_fontsize)
    axis.set_ylim(-0.5, len(labels) - 0.5)
    cmap = _ramped(cmap_name, style)
    for span in spans:
        start, end = span["extent"]
        if end < t0 or start > t1:
            continue
        left = max(start, t0)
        width = min(end, t1) - left
        scores = per_span.get(span["id"], {})
        for row, label in enumerate(labels):
            score = scores.get(label)
            if score is None or score < style.raster_paint_floor:
                continue
            axis.add_patch(
                Rectangle(
                    (left, row - style.raster_cell_height / 2),
                    max(width, style.raster_min_cell_s),
                    style.raster_cell_height,
                    facecolor=cmap(max(0.0, min(1.0, score))),
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=3,
                )
            )
    _score_colorbar(colorbar_axis, cmap_name, style, label="probability — dark = present")


def _readable_on(rgba: tuple[float, float, float, float]) -> str:
    """Black or white, whichever reads on a cell of this colour.

    Args:
        rgba: The cell's fill.

    Returns:
        The text colour. Uses Rec. 601 luminance, so the choice follows perceived brightness rather
        than the colormap's position — which is what a reversed map needs, since its dark end and
        its light end swap.
    """
    red, green, blue = rgba[0], rgba[1], rgba[2]
    return "black" if (0.299 * red + 0.587 * green + 0.114 * blue) > 0.55 else "white"


def _squim_panel(
    axis: Axes,
    spans: list[dict[str, Any]],
    squim: dict[str, dict[str, float | None]],
    window: tuple[float, float],
    style: FigureStyle,
    colorbar_axis: Axes,
) -> None:
    """SQUIM's three metrics per span, each cell drawn at its span's width.

    Ink means attention, so the two kinds of panel ink opposite ends of their scales and still
    read the same way: a raster darkens where a label fires, and this panel darkens where quality is
    poor. STOI, PESQ and SI-SDR are all higher-is-better, so low is dark here. Their units are
    unrelated, so each row is normalised over its own range from ``style.squim_ranges`` and the
    colorbar reads as a normalised fraction rather than a value.

    Args:
        axis: The panel.
        spans: :func:`_spans`' result.
        squim: :func:`_squim_by_span`'s result.
        window: The page's ``(start, end)``.
        style: The drawing configuration.
        colorbar_axis: The slot to the panel's right, where the scale is drawn.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    t0, t1 = window
    metrics = ("stoi", "pesq", "si_sdr")
    axis.set_title("SQUIM per span (STOI / PESQ / SI-SDR)", fontsize=style.title_fontsize)
    axis.set_xlim(t0, t1)
    if not squim:
        _absent_panel(axis, window, "no SQUIM assertion in the store", style)
        colorbar_axis.set_axis_off()
        return
    axis.set_yticks(range(len(metrics)))
    axis.set_yticklabels(list(metrics), fontsize=style.tick_fontsize)
    axis.set_ylim(-0.5, len(metrics) - 0.5)
    cmap = _ramped(style.cmap_squim, style)
    for span in spans:
        start, end = span["extent"]
        scores = squim.get(span["id"])
        if end < t0 or start > t1 or scores is None:
            continue
        left = max(start, t0)
        width = max(min(end, t1) - left, style.raster_min_cell_s)
        for row, metric in enumerate(metrics):
            value = scores.get(metric)
            if value is None:
                continue
            low, high = style.squim_ranges[metric]
            frac = max(0.0, min(1.0, (float(value) - low) / (high - low)))
            axis.add_patch(
                Rectangle(
                    (left, row - style.raster_cell_height / 2),
                    width,
                    style.raster_cell_height,
                    facecolor=cmap(frac),
                    edgecolor="none",
                    linewidth=0.0,
                    zorder=3,
                )
            )
    _score_colorbar(colorbar_axis, style.cmap_squim, style, label="per row — dark = poor")


def _renderer(axis: Axes) -> RendererBase | None:
    """The canvas renderer, for measuring a label before it is committed to the page.

    Args:
        axis: Any panel on the figure.

    Returns:
        The renderer, or None where the backend exposes none — text extents then fall back to
        matplotlib's own cached renderer, which is accurate enough for a fit decision.
    """
    getter = getattr(axis.figure.canvas, "get_renderer", None)
    renderer = getter() if callable(getter) else None
    return renderer if isinstance(renderer, RendererBase) else None


def _axis_points_per_second(axis: Axes, window: tuple[float, float], renderer: RendererBase | None) -> float:
    """How many typographic points one second of the shared time axis occupies.

    Args:
        axis: The panel.
        window: The page's ``(start, end)``.
        renderer: :func:`_renderer`'s result.

    Returns:
        Points per second, using the axes' own drawn width rather than the figure's, since the
        margins are not available to a label.
    """
    t0, t1 = window
    width_px = axis.get_window_extent(renderer=renderer).width
    return float(width_px) / float(axis.figure.dpi) * 72.0 / max(t1 - t0, 1e-9)


def _fit_cell_text(
    axis: Axes,
    x: float,
    y: float,
    text: str,
    cell_points: float,
    style: FigureStyle,
    renderer: RendererBase | None,
) -> bool:
    """Draw a raster cell's score text only at a size that fits the space the cell has.

    Args:
        axis: The panel.
        x: The cell's centre on the time axis.
        y: The cell's row.
        text: The score, already formatted.
        cell_points: The width available to this cell, in points.
        style: The drawing configuration.
        renderer: :func:`_renderer`'s result.

    Returns:
        Whether the text was drawn. The caller has already drawn the marker, which is never dropped:
        a missing number means the cell was too small for one, not that nothing was measured.
    """
    sizes = (style.cell_fontsize, (style.cell_fontsize + style.cell_floor_fontsize) / 2, style.cell_floor_fontsize)
    for size in sizes:
        artist = axis.text(x, y, text, ha="center", va="center", fontsize=size, zorder=4)
        width = artist.get_window_extent(renderer=renderer).width / float(axis.figure.dpi) * 72.0
        if width <= cell_points:
            return True
        artist.remove()
    return False


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
    half = style.asr_row_height / 2.0
    axis.set_ylim(-0.5, style.asr_rows - 0.5)
    axis.set_yticks([])
    here = [word for word in words if word["extent"][1] >= t0 and word["extent"][0] <= t1]
    for index, word in enumerate(here):
        start, end = word["extent"]
        row = index % style.asr_rows
        axis.add_patch(
            Rectangle(
                (max(start, t0), row - half),
                max(min(end, t1) - max(start, t0), 0.01),
                style.asr_row_height,
                facecolor=style.word_fill,
                edgecolor="none",
                linewidth=0.0,
                alpha=0.8,
            )
        )
        axis.text(
            max(start, t0),
            row,
            word["text"],
            fontsize=style.cell_fontsize,
            ha="left",
            va="center",
            color=style.word_text_colour,
        )


def _taxonomy_panel(axis: Axes, lines: list[str], style: FigureStyle) -> Text:
    """The whole-file taxonomy readout, monospaced and off the shared time axis.

    Args:
        axis: The panel.
        lines: :func:`summary_panel_lines`' result.
        style: The drawing configuration.

    Returns:
        The artist, so a test can measure its extent against the axis.
    """
    axis.set_axis_off()
    return axis.text(
        0.0,
        1.0,
        "\n".join(lines),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=style.text_fontsize,
        family="monospace",
    )


def _continuity(store: ProvStore, run_dir: Path) -> tuple[np.ndarray | None, float | None, float | None]:
    """PREPROCESS's persisted continuity trace, with the rank cut it recorded.

    Read, never recomputed: a trace derived here could differ from the one the spans in this same
    store were proposed against, and the page would then annotate spans with a threshold that never
    produced them.

    Args:
        store: The provenance store.
        run_dir: The run directory.

    Returns:
        ``(trace, cut_level, cut_percentile)``. Every element is None when the derivative is absent;
        ``cut_level`` alone is None when the cut marked no sample.
    """
    measurement = find_measurement(store, "continuity_trace")
    if measurement is None:
        return None, None, None
    trace = _npz(run_dir, store, "continuity_trace", "continuity")
    level = measurement.attributes.get("cut_level")
    percentile = measurement.attributes.get("cut_percentile")
    return (
        trace,
        None if level is None else float(level),
        None if percentile is None else float(percentile),
    )


def _span_row_absence(absent: dict[str, str], spans: list[dict[str, Any]]) -> dict[str, str]:
    """Why a span-source row is empty over the whole recording.

    A source contributes nothing either because its own upstream derivative is absent or because
    every candidate it proposed corroborated a span an earlier source already covered. Neither
    means the source ran and found nothing, so the row says which it was.

    Args:
        absent: :func:`_absent_reasons`' result.
        spans: :func:`_spans`' result.

    Returns:
        ``{code: reason}`` for each row that contributed nothing.
    """
    present = {_span_code(str(span["signal"] or ""), str(span["measure"] or "")) for span in spans}
    reasons: dict[str, str] = {}
    if "E" not in present:
        reasons["E"] = absent.get("energy_envelope", "no amplitude span reached the store")
    if "S" not in present:
        reasons["S"] = absent.get("normalized_envelope", "no normalization-derived span was novel")
    if "C" not in present:
        reasons["C"] = absent.get("continuity_trace", "no continuity span was novel")
    if "A" not in present:
        reasons["A"] = absent.get("consensus_transcript", "no asr span was novel")
    return reasons


def _page_height_ratios(
    style: FigureStyle, collapsed: Sequence[int], raster_rows: Mapping[int, int] | None = None
) -> list[float]:
    """The page's panel heights, with absent panels collapsed and their share redistributed.

    The figure's total height is unchanged, so pages stay comparable: what an absent panel gives up
    goes to the panels that have something to draw, in proportion to what they already had. A
    raster's own height grows with how many rows it draws, so its tick labels keep their point size
    however large the row union turns out to be.

    Args:
        style: The drawing configuration.
        collapsed: Indices of the panels to collapse.
        raster_rows: Row counts by panel index, for the panels whose height follows their rows.

    Returns:
        One height per panel, in panel order.
    """
    ratios = list(style.height_ratios)
    for index, rows in (raster_rows or {}).items():
        if index not in set(collapsed):
            ratios[index] = max(ratios[index], rows * style.raster_row_ratio)
    freed = 0.0
    for index in collapsed:
        if ratios[index] > style.absent_height_ratio:
            freed += ratios[index] - style.absent_height_ratio
            ratios[index] = style.absent_height_ratio
    keep = [index for index in range(len(ratios)) if index not in set(collapsed)]
    total = sum(ratios[index] for index in keep)
    if freed > 0 and total > 0:
        for index in keep:
            ratios[index] += freed * ratios[index] / total
    return ratios


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
    trace, cut_level, cut_percentile = _continuity(store, run_dir)

    spans = _spans(store)
    clips = _clip_extents(store)
    yamnet = _span_scores(store, "span_yamnet")
    hear = _span_scores(store, "span_hear")
    words = _words(store)
    squim = _squim_by_span(store)
    summary_lines = taxonomy_summary_lines(store, style)
    panel_lines = summary_panel_lines(store, style)

    wideband_title = (
        f"wideband spectrogram ({float(config.require('spectrogram.wideband_window_ms')):.0f} ms window, "
        f"{float(config.require('spectrogram.hop_ms')):.0f} ms hop) — the speech-analysis view; "
        "continuity runs on the narrowband array, not this one"
    )

    raw_floor = config.get("taxonomy.consolidation_floor")
    floor = None if raw_floor is None else float(raw_floor)
    yamnet_rows = _raster_rows(yamnet, style.top_labels, style.raster_rows_scope, floor)
    hear_rows = _raster_rows(hear, style.top_labels, style.raster_rows_scope, floor)

    row_absent = _span_row_absence(absent, spans)
    # Panel indices, in the order they are unpacked below.
    collapsed = [
        index
        for index, empty in ((0, wideband is None), (3, not yamnet), (4, not hear), (5, not squim), (6, not words))
        if empty
    ]
    height_ratios = _page_height_ratios(style, collapsed, {3: len(yamnet_rows), 4: len(hear_rows)})

    figure_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    pdf_path = figure_dir / f"{stem or store.run_id}.pdf"
    pdf = PdfPages(pdf_path)
    for index, window in enumerate(pages(duration_s, style), start=1):
        figure: Figure
        figure, axes = plt.subplots(
            len(height_ratios),
            2,
            figsize=style.figure_inches,
            constrained_layout=True,
            gridspec_kw={"height_ratios": height_ratios, "width_ratios": [1.0, style.colorbar_width_ratio]},
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
        ) = axes[:, 0]
        # Every row owns a slot in the second column so that all panels are laid out to one width
        # and the shared time axis stays aligned; a row with no scale to show blanks its slot.
        slots = list(axes[:, 1])
        for slot in slots:
            slot.set_axis_off()
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
            cut_level=cut_level,
            cut_percentile=cut_percentile,
            continuity_absent=absent.get("continuity_trace", "continuity_trace is absent from the store"),
        )
        _span_lane_panel(axis_spans, spans, clips, window, style, row_absent)
        _raster_panel(
            axis_yamnet,
            spans,
            yamnet,
            yamnet_rows,
            style.cmap_yamnet,
            f"YAMNet per-span scores — rows: union of each span's top-{style.top_labels} over the file",
            window,
            style,
            absent.get("span_yamnet", "span_yamnet is absent from the store"),
            slots[3],
        )
        _raster_panel(
            axis_hear,
            spans,
            hear,
            hear_rows,
            style.cmap_hear,
            f"HeAR per-span scores — rows: union of each span's top-{style.top_labels} over the file",
            window,
            style,
            absent.get("span_hear", "span_hear is absent from the store"),
            slots[4],
        )
        _squim_panel(axis_squim, spans, squim, window, style, slots[5])
        _asr_lane_panel(
            axis_asr,
            words,
            window,
            style,
            absent.get("consensus_transcript", "no consensus word in the store"),
        )
        _taxonomy_panel(axis_taxonomy, panel_lines, style)

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
        pdf.savefig(figure, dpi=style.dpi)
        if style.also_write_pngs:
            page_path = figure_dir / f"{stem or store.run_id}__page{index:02d}.png"
            figure.savefig(page_path, dpi=style.dpi)
            written[f"page{index:02d}"] = page_path
        plt.close(figure)

    pdf.close()
    written["figure"] = pdf_path
    (figure_dir / "taxonomy_summary.json").write_text(json.dumps({"lines": summary_lines}, indent=1) + "\n")
    written["taxonomy_summary"] = figure_dir / "taxonomy_summary.json"
    return written
