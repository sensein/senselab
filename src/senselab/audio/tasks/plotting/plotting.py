"""This module contains functions for plotting audio-related data."""

import math
import os
import textwrap
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union, cast

# Use non-interactive backend when not in a notebook (e.g., papermill, CI)
if not os.environ.get("DISPLAY") and "inline" not in os.environ.get("MPLBACKEND", ""):
    import matplotlib

    matplotlib.use("agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rc_context
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import logger

# ---------------------------
# Plot context & scaling
# ---------------------------

_Context = Union[str, float]  # "auto" | "small" | "medium" | "large" | float scale
_INCHES_PER_RATIO = 1.8  # what one unit of plot_aligned_panels' height_ratios is worth
TEXT_PANEL_INCHES_PER_LINE = 0.18  # 8 pt at 1.2 line spacing is 0.133 in; the rest is headroom
MIN_FIGURE_HEIGHT_IN = 4.0  # the floor plot_aligned_panels puts under a short panel stack
TOKEN_LABEL_FONTSIZE = 5.0  # the default point size of the text a tokens panel draws on a bar
TOKEN_LABEL_FLOOR_FONTSIZE = 4.0  # the smallest point size a token label is shrunk to before it is dropped
TOKEN_LABEL_PADDING_PT = 1.0  # points of the bar kept clear of its label, shared by the two ends
TOKEN_ROW_PITCH_EM = 2.0  # the least pitch a staggered row is given, in multiples of the label's point size
TOKEN_BAR_HEIGHT_FRACTION = 0.7  # the share of its row's pitch a token's bar fills
REPORT_LANE_GUTTER_MIN_IN = 0.72  # a separate left column for a panel's descriptive lane title
REPORT_LANE_GUTTER_MAX_IN = 1.45  # long names wrap rather than taking time pixels from the report
REPORT_COLORBAR_GUTTER_IN = 0.72  # a shared right column for score-raster probability scales


def _fitted_token_fontsize(
    measure: Callable[[float], float],
    available_pt: float,
    full_fontsize: float,
    floor_fontsize: float,
) -> Optional[float]:
    """The largest point size in ``[floor_fontsize, full_fontsize]`` at which a label fits its bar.

    Args:
        measure: The label's rendered width in points at a given point size.
        available_pt: The bar's width in points, less the padding kept clear of the label.
        full_fontsize: The size the label is drawn at when it fits.
        floor_fontsize: The size below which the label is dropped rather than shrunk further.

    Returns:
        The point size to draw at, or ``None`` when the label does not fit even at the floor.
    """
    if available_pt <= 0.0:
        return None
    width = measure(full_fontsize)
    if width <= available_pt:
        return full_fontsize
    if width <= 0.0:
        return None
    candidate = min(full_fontsize, max(full_fontsize * available_pt / width, floor_fontsize))
    if measure(candidate) <= available_pt:
        return candidate
    if candidate > floor_fontsize and measure(floor_fontsize) <= available_pt:
        return floor_fontsize
    return None


def _staggered_row_ceiling(block_height_pt: float, fontsize: float) -> int:
    """The most rows a lane of this height can hold at this point size.

    Args:
        block_height_pt: The height available to the staggered block, in points.
        fontsize: The point size the block's labels are drawn at when they fit.

    Returns:
        The row count above which a row is shorter than ``TOKEN_ROW_PITCH_EM`` of its own font.
    """
    pitch = TOKEN_ROW_PITCH_EM * fontsize
    return max(1, int(block_height_pt // pitch)) if pitch > 0.0 else 1


def _staggered_row_count(
    label_widths_pt: Sequence[float],
    available_pt: float,
    padding_pt: float,
    max_rows: int,
) -> int:
    """The number of rows a lane's labels take to lie side by side, capped by legibility.

    Args:
        label_widths_pt: Each label's rendered width in points at the size below which it is dropped.
        available_pt: The width of one row, in points.
        padding_pt: The points kept clear beside each label.
        max_rows: The ceiling from ``_staggered_row_ceiling``.

    Returns:
        A row count of at least 1 and at most ``max_rows``.
    """
    if not label_widths_pt or available_pt <= 0.0 or max_rows <= 1:
        return 1
    demand = sum(float(width) + padding_pt for width in label_widths_pt)
    return int(min(max_rows, max(1, math.ceil(demand / available_pt))))


def _token_label_slots(
    centres: Sequence[float],
    half_spans: Sequence[float],
    rows: Sequence[int],
    row_count: int,
    limits: Tuple[float, float],
    *,
    expand_to_row_neighbours: bool = False,
) -> List[Tuple[float, float]]:
    """The extent on the time axis each label is measured against, one per token.

    Each slot is centred on the token's bar and reaches no further than the midpoint to the nearest
    token sharing its row or the edge of the window. By default it additionally stays within the
    bar's cycling-row reach; a report can opt into the unused row space for short transcript words.

    Args:
        centres: Each token's bar centre, on the time axis.
        half_spans: Half of each token's bar width, on the time axis.
        rows: Each token's row.
        row_count: The number of rows the tokens are spread over.
        limits: The window's ``(lower, upper)`` extent on the time axis.
        expand_to_row_neighbours: Whether labels may use all unused space in their cycling row.

    Returns:
        One ``(lower, upper)`` extent per token, in the order the tokens were given.
    """
    lower, upper = limits
    order = sorted(range(len(centres)), key=lambda index: centres[index])
    left = [lower] * len(centres)
    right = [upper] * len(centres)
    seen: Dict[int, int] = {}
    for index in order:
        previous = seen.get(rows[index])
        if previous is not None:
            left[index] = (centres[index] + centres[previous]) / 2.0
        seen[rows[index]] = index
    seen.clear()
    for index in reversed(order):
        following = seen.get(rows[index])
        if following is not None:
            right[index] = (centres[index] + centres[following]) / 2.0
        seen[rows[index]] = index
    slots: List[Tuple[float, float]] = []
    for index, centre in enumerate(centres):
        reach = min(centre - left[index], right[index] - centre)
        if not expand_to_row_neighbours:
            reach = min(reach, row_count * half_spans[index])
        reach = max(0.0, reach)
        slots.append((centre - reach, centre + reach))
    return slots


def _lane_title_lines(title: str, *, width: int = 18, max_lines: int = 2) -> List[str]:
    """Wrap one report lane title to its dedicated, bounded gutter.

    A panel title is descriptive metadata, not an axis value. Keeping it to two lines means a
    short lane cannot paint across the panel above or below it when a report page is dense.
    """
    lines = textwrap.wrap(title, width=width, break_long_words=True, break_on_hyphens=False) or [title]
    if len(lines) <= max_lines:
        return lines
    clipped = lines[:max_lines]
    clipped[-1] = f"{clipped[-1].rstrip()}..."
    return clipped


class _FittedTokenLabel(Text):
    """A token's text, drawn only at a point size at which it fits the span it is given.

    The decision is taken against the renderer that is about to draw it, so it holds for a figure
    saved at any width or dpi, and is retaken from ``full_fontsize`` on every draw.
    """

    def __init__(
        self,
        x: float,
        y: float,
        text: str,
        *,
        span: Tuple[float, float],
        full_fontsize: float,
        floor_fontsize: float = TOKEN_LABEL_FLOOR_FONTSIZE,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Place ``text`` at ``(x, y)``, measured against ``span`` on the time axis.

        Args:
            x: The label's x position, in data coordinates.
            y: The label's y position, in data coordinates.
            text: The label's text.
            span: The ``(lower, upper)`` extent on the time axis the label must fit inside.
            full_fontsize: The point size the label is drawn at when it fits.
            floor_fontsize: The point size below which the label is dropped rather than shrunk.
            **kwargs: Forwarded to ``matplotlib.text.Text``.
        """
        super().__init__(x, y, text, **kwargs)
        self._span = span
        self._full_fontsize = full_fontsize
        self._floor_fontsize = floor_fontsize

    def set_span(self, span: Tuple[float, float]) -> None:
        """Set the extent on the time axis the label is measured against.

        Args:
            span: The ``(lower, upper)`` extent, in data coordinates.
        """
        self._span = span

    def _points_per_pixel(self) -> float:
        """Display pixels are dpi-dependent; the fit is stated in points, which are not."""
        figure = self.get_figure()
        return 72.0 / float(figure.dpi) if figure is not None else 1.0

    def _span_width_pt(self) -> float:
        """The span's width in points, under the axes transform that is current for this draw."""
        axes = self.axes
        if axes is None:
            return 0.0
        (x0, _), (x1, _) = axes.transData.transform([(self._span[0], 0.0), (self._span[1], 0.0)])
        return abs(float(x1) - float(x0)) * self._points_per_pixel()

    def _text_width_pt(self, renderer: RendererBase, fontsize: float) -> float:
        """The label's rendered width in points at ``fontsize``, on the renderer about to draw it.

        Args:
            renderer: The renderer to measure against.
            fontsize: The point size to measure at.

        Returns:
            The width in points. A hidden ``Text`` measures as a unit box, so it is shown first.
        """
        self.set_visible(True)
        self.set_fontsize(fontsize)
        return float(self.get_window_extent(renderer).width) * self._points_per_pixel()

    def draw(self, renderer: RendererBase) -> None:
        """Draw the label at a size at which it fits its span, or not at all.

        Args:
            renderer: The renderer this draw is going through.
        """
        callback, self.stale_callback = self.stale_callback, None
        try:
            self.set_visible(True)
            self.set_fontsize(self._full_fontsize)
            available = self._span_width_pt() - TOKEN_LABEL_PADDING_PT
            fitted = _fitted_token_fontsize(
                lambda fontsize: self._text_width_pt(renderer, fontsize),
                available,
                self._full_fontsize,
                self._floor_fontsize,
            )
            self.set_fontsize(self._full_fontsize if fitted is None else fitted)
            self.set_visible(fitted is not None)
        finally:
            self.stale_callback = callback
            self.stale = False
        super().draw(renderer)


class _TokenPlacement(NamedTuple):
    """One token's bar, its label if it has one, and where on the time axis it sits."""

    block: int
    bar: Rectangle
    label: Optional[_FittedTokenLabel]
    centre: float
    half_span: float


class _StaggeredTokenLane(Artist):
    """The vertical layout of a tokens panel, decided against the renderer that is about to draw it.

    Draws nothing itself. It sits below every bar and label in the lane so that its ``draw`` runs
    first, sets each bar's row and each label's slot, and leaves the drawing to them.
    """

    zorder = -1.0

    def __init__(
        self,
        placements: List[_TokenPlacement],
        blocks: int,
        staggered_block: Optional[int],
        fontsize: float,
        floor_fontsize: float,
        expand_to_row_neighbours: bool = False,
    ) -> None:
        """Lay ``placements`` out over ``blocks`` stacked blocks of the lane.

        Args:
            placements: One entry per token, in the order the tokens were given.
            blocks: The number of blocks stacked in the lane.
            staggered_block: The block holding the tokens that declared no row, if there is one.
            fontsize: The point size the labels are drawn at when they fit.
            floor_fontsize: The point size below which a label is dropped rather than shrunk.
            expand_to_row_neighbours: Whether labels may use unused horizontal room in their cycling row.
        """
        super().__init__()
        self._placements = placements
        self._blocks = max(blocks, 1)
        self._staggered_block = staggered_block
        self._fontsize = fontsize
        self._floor_fontsize = floor_fontsize
        self._expand_to_row_neighbours = expand_to_row_neighbours

    def _row_count(self, renderer: RendererBase, axes: Axes, points_per_pixel: float) -> int:
        """The number of rows the block of undeclared tokens is spread over for this draw.

        Args:
            renderer: The renderer to measure the labels against.
            axes: The axes the lane is drawn on.
            points_per_pixel: The display-to-point conversion for this figure.

        Returns:
            A row count of at least 1.
        """
        if self._staggered_block is None:
            return 1
        lower, upper = axes.get_xlim()
        (x0, _), (x1, _) = axes.transData.transform([(lower, 0.0), (upper, 0.0)])
        width_pt = abs(float(x1) - float(x0)) * points_per_pixel
        block_pt = float(axes.get_window_extent(renderer).height) * points_per_pixel / float(self._blocks)
        widths = [
            placement.label._text_width_pt(renderer, self._floor_fontsize)
            for placement in self._placements
            if placement.block == self._staggered_block and placement.label is not None
        ]
        ceiling = _staggered_row_ceiling(block_pt, self._fontsize)
        return _staggered_row_count(widths, width_pt, TOKEN_LABEL_PADDING_PT, ceiling)

    def draw(self, renderer: RendererBase) -> None:
        """Place every bar in its row and hand every label the slot it is measured against.

        Args:
            renderer: The renderer this draw is going through.
        """
        axes = self.axes
        if axes is None or not self._placements:
            self.stale = False
            return
        figure = self.get_figure()
        points_per_pixel = 72.0 / float(figure.dpi) if figure is not None else 1.0
        limits = axes.get_xlim()
        rows = self._row_count(renderer, cast(Axes, axes), points_per_pixel)
        touched: List[Artist] = [placement.bar for placement in self._placements]
        touched += [placement.label for placement in self._placements if placement.label is not None]
        callbacks = [artist.stale_callback for artist in touched]
        for artist in touched:
            artist.stale_callback = None
        try:
            for block in range(self._blocks):
                members = [placement for placement in self._placements if placement.block == block]
                count = rows if block == self._staggered_block else 1
                assignment = [index % count for index in range(len(members))]
                slots = _token_label_slots(
                    [placement.centre for placement in members],
                    [placement.half_span for placement in members],
                    assignment,
                    count,
                    limits,
                    expand_to_row_neighbours=self._expand_to_row_neighbours,
                )
                pitch = 1.0 / float(self._blocks * count)
                top = float(block + 1) / float(self._blocks)
                for placement, row, slot in zip(members, assignment, slots):
                    centre = top - (row + 0.5) * pitch
                    height = TOKEN_BAR_HEIGHT_FRACTION * pitch
                    placement.bar.set_y(centre - height / 2.0)
                    placement.bar.set_height(height)
                    if placement.label is not None:
                        placement.label.set_y(centre)
                        placement.label.set_span(slot)
        finally:
            for artist, callback in zip(touched, callbacks):
                artist.stale_callback = callback
            self.stale = False


def _detect_screen_resolution() -> Tuple[int, int]:
    """Best-effort screen resolution detection. Falls back to 1920x1080."""
    # Try TkAgg; only through an already-open figure -- get_current_fig_manager() would create one.
    try:
        mgr = plt.get_current_fig_manager() if plt.get_fignums() else None
        win = getattr(mgr, "window", None)
        if win is not None and hasattr(win, "winfo_screenwidth"):
            return int(win.winfo_screenwidth()), int(win.winfo_screenheight())
    except Exception:
        pass
    # Try Qt
    try:
        from PyQt5 import QtWidgets  # type: ignore

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        screen = app.primaryScreen()
        size = screen.size()
        return int(size.width()), int(size.height())
    except Exception:
        pass
    # Fallback
    return 1920, 1080


def _context_scale_from_resolution() -> float:
    """Map screen width → a sensible scale factor."""
    width, _ = _detect_screen_resolution()
    # Simple, readable buckets
    if width <= 1366:
        return 0.9
    if width <= 1920:
        return 1.0
    if width <= 2560:
        return 1.25
    if width <= 3840:
        return 1.5
    return 2.0


def _resolve_scale(context: _Context) -> float:
    if isinstance(context, (int, float)):
        return float(context)
    ctx = str(context).lower()
    if ctx == "auto":
        return _context_scale_from_resolution()
    if ctx in ("paper", "small"):
        return 0.9
    if ctx in ("notebook", "medium"):
        return 1.0
    if ctx in ("talk", "large"):
        return 1.3
    # Default
    return 1.0


def _rc_for_scale(scale: float) -> Dict[str, Any]:
    """Return rcParams tuned for the given scale (seaborn-like)."""
    base = 10.0 * scale
    return {
        "font.size": base,
        "axes.titlesize": base * 1.2,
        "axes.labelsize": base,
        "xtick.labelsize": base * 0.9,
        "ytick.labelsize": base * 0.9,
        "legend.fontsize": base * 0.95,
        "lines.linewidth": 1.25 * scale,
        "grid.linewidth": 0.8 * scale,
        "axes.linewidth": 0.8 * scale,
        "figure.titlesize": base * 1.3,
    }


# ---------------------------
# Helpers
# ---------------------------


def _power_to_db(spectrogram: np.ndarray, ref: float = 1.0, amin: float = 1e-10, top_db: float = 80.0) -> np.ndarray:
    """Converts a power spectrogram (amplitude squared) to decibel (dB) units."""
    S = np.asarray(spectrogram)

    if amin <= 0:
        raise ValueError("amin must be strictly positive")

    if np.issubdtype(S.dtype, np.complexfloating):
        logger.warning(
            "_power_to_db was called on complex input so phase information will be discarded. "
            "To suppress this warning, call power_to_db(np.abs(D)**2) instead.",
            stacklevel=2,
        )
        magnitude = np.abs(S)
    else:
        magnitude = S

    ref_value = ref(magnitude) if callable(ref) else np.abs(ref)
    log_spec: np.ndarray = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ValueError("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def _as_numpy(values: Any) -> np.ndarray:  # noqa: ANN401 — a tensor, a sequence or an array
    """One curve's x or y values as an array, whatever the caller handed in."""
    return values.cpu().numpy() if torch.is_tensor(values) else np.asarray(values)


def _draw_waveform_overlays(ax: Axes, panel: Dict[str, Any], *, time_limits: Tuple[float, float] | None = None) -> None:
    """Draw a waveform panel's span overlay and its twin-axis curves.

    The spans go on the waveform's own axis, behind everything, so the twin's curves stay on top of
    them; the twin's y-label names both, since the overlay has no scale of its own to be labelled by.

    Args:
        ax: The waveform panel's own axis, carrying amplitude on the left.
        panel: The panel specification, read for its optional ``spans`` and ``twin`` blocks.
        time_limits: Optional recording-time interval. Span labels outside it are not drawn: an
            evidence page must contain only evidence visible on its time axis.
    """
    spans = panel.get("spans") or {}
    twin_spec = panel.get("twin") or {}
    if not spans and not twin_spec:
        return
    for segment in spans.get("segments", []):
        start, end = float(segment["start"]), float(segment["end"])
        visible_start = max(start, time_limits[0]) if time_limits is not None else start
        visible_end = min(end, time_limits[1]) if time_limits is not None else end
        if visible_end <= visible_start:
            continue
        ax.axvspan(visible_start, visible_end, color="darkorange", alpha=0.18, linewidth=0, zorder=0)
        ax.annotate(
            str(segment["label"]),
            xy=((visible_start + visible_end) / 2.0, 0.99),
            xycoords=("data", "axes fraction"),
            ha="center",
            va="top",
            rotation=90,
            fontsize=6,
            color="saddlebrown",
        )
    twin = ax.twinx()
    for times, values, label, color in twin_spec.get("data", []):
        time_values, value_values = _as_numpy(times), _as_numpy(values)
        if time_limits is not None:
            visible = (time_values >= time_limits[0]) & (time_values <= time_limits[1])
            time_values, value_values = time_values[visible], value_values[visible]
        twin.plot(time_values, value_values, color=color, label=label, linewidth=0.9, alpha=0.9)
    names = [str(name) for name in (twin_spec.get("name"), spans.get("name")) if name]
    twin.set_ylabel(str(twin_spec.get("axis_label")) if twin_spec.get("axis_label") else " · ".join(names) or "Value")
    if twin_spec.get("data"):
        twin.legend(loc="upper right", fontsize=7)


# ---------------------------
# Public API
# ---------------------------


def plot_waveform(
    audio: Audio,
    title: str = "Waveform",
    fast: bool = False,
    *,
    context: _Context = "auto",
    figsize: Tuple[float, float] | None = None,
) -> Figure:
    """Plot the time-domain waveform of an `Audio` object and return the Figure.

    The plot is automatically scaled for readability using a *context* scale
    (similar to seaborn). Use `fast=True` to lightly decimate the signal for
    quicker rendering on very long waveforms.

    Args:
        audio (Audio):
            Input audio containing `.waveform` (shape `[C, T]`) and `.sampling_rate`.
        title (str, optional):
            Figure title. Defaults to `"Waveform"`.
        fast (bool, optional):
            If `True`, plots a 10× downsampled view for speed. Defaults to `False`.
        context (_Context, optional):
            Size preset or numeric scale. Accepted values:
              * `"auto"` (detect from screen), `"small"`, `"medium"`, `"large"`,
              * or a float scale factor (e.g., `1.25`). Defaults to `"auto"`.
        figsize (tuple[float, float] | None, optional):
            Base `(width, height)` in inches **before** context scaling.
            Defaults to `(12, 2×channels)`.

    Returns:
        matplotlib.figure.Figure: The created figure (also displayed).

    Example:
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> fig = plot_waveform(a1, title="Sample 1", fast=True, context="medium")
        >>> # fig.savefig("waveform.png")  # optional
    """
    waveform = audio.waveform
    sample_rate = audio.sampling_rate

    if fast:
        waveform = waveform[..., ::10]

    num_channels, num_frames = waveform.shape
    time_axis = torch.linspace(0, num_frames / sample_rate, num_frames)

    scale = _resolve_scale(context)
    rc = _rc_for_scale(scale)
    if figsize is None:
        base = (12.0, max(2.0 * num_channels, 2.5))
    else:
        base = figsize
    scaled_size = (base[0] * scale, base[1] * scale)

    with rc_context(rc):
        fig, axes = plt.subplots(num_channels, 1, figsize=scaled_size, sharex=True)
        if num_channels == 1:
            axes = [axes]  # ensure iterable
        for c, ax in enumerate(axes):
            ax.plot(time_axis.numpy(), waveform[c].cpu().numpy())
            ax.set_ylabel(f"Ch {c + 1}")
            ax.grid(True, alpha=0.3)
        fig.suptitle(title)
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show(block=False)
        return fig


def plot_specgram(
    audio: Audio,
    mel_scale: bool = False,
    title: str = "Spectrogram",
    *,
    context: _Context = "auto",
    figsize: Tuple[float, float] | None = None,
    **spect_kwargs: Any,  # noqa: ANN401
) -> Figure:
    """Plot a (mel-)spectrogram for a **mono** `Audio` object and return the Figure.

    Internally calls senselab's torchaudio-based extractors:
    `extract_spectrogram_from_audios` or `extract_mel_spectrogram_from_audios`.
    The function expects a 2D spectrogram `[freq_bins, time_frames]`; multi-channel
    inputs should be downmixed beforehand.

    Args:
        audio (Audio):
            Input **mono** audio. If multi-channel, downmix first.
        mel_scale (bool, optional):
            If `True`, plots a mel spectrogram; otherwise linear frequency. Defaults to `False`.
        title (str, optional):
            Figure title. Defaults to `"Spectrogram"`.
        context (_Context, optional):
            Size preset or numeric scale (`"auto"`, `"small"`, `"medium"`, `"large"`, or float).
            Defaults to `"auto"`.
        figsize (tuple[float, float] | None, optional):
            Base `(width, height)` in inches **before** context scaling. Defaults to `(10, 4)`.
        **spect_kwargs:
            Passed to the underlying extractor (e.g., `n_fft=1024`, `hop_length=256`,
            `n_mels=80`, `win_length=1024`, `f_min=0`, `f_max=None`).

    Returns:
        matplotlib.figure.Figure: The created figure (also displayed).

    Raises:
        ValueError: If spectrogram extraction fails, contains NaNs, or the result is not 2D.

    Example (linear spectrogram):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> fig = plot_specgram(a1, mel_scale=False, n_fft=1024, hop_length=256)
        >>> # fig.savefig("spec.png")

    Example (mel spectrogram):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> fig = plot_specgram(a1, mel_scale=True, n_mels=80, n_fft=1024, hop_length=256)
    """
    # Extract the spectrogram
    if mel_scale:
        from senselab.audio.tasks.features_extraction.torchaudio import (
            extract_mel_spectrogram_from_audios,
        )

        spectrogram = extract_mel_spectrogram_from_audios([audio], **spect_kwargs)[0]["mel_spectrogram"]
        y_axis_label = "Mel frequency (bins)"
    else:
        from senselab.audio.tasks.features_extraction.torchaudio import (
            extract_spectrogram_from_audios,
        )

        spectrogram = extract_spectrogram_from_audios([audio], **spect_kwargs)[0]["spectrogram"]
        y_axis_label = "Frequency [Hz]"

    # ---- Guard against invalid/short-audio outputs (must be exactly this phrase)
    if not torch.is_tensor(spectrogram):
        raise ValueError("Spectrogram extraction failed")
    if spectrogram.ndim == 0 or spectrogram.numel() == 0:
        raise ValueError("Spectrogram extraction failed")
    if spectrogram.dtype.is_floating_point and torch.isnan(spectrogram).any():
        raise ValueError("Spectrogram extraction failed")

    if spectrogram.dim() != 2:
        raise ValueError(
            "Spectrogram must be a 2D tensor. Got shape: {}".format(spectrogram.shape),
            "Please make sure the input audio is mono.",
        )

    # Determine time and frequency scale
    # num_frames = spectrogram.size(1)
    num_freq_bins = spectrogram.size(0)

    # Time axis in seconds
    duration_sec = audio.waveform.size(-1) / audio.sampling_rate
    time_axis_start = 0.0
    time_axis_end = float(duration_sec)

    # Frequency axis
    if mel_scale:
        freq_start, freq_end = 0.0, float(num_freq_bins - 1)
    else:
        freq_start, freq_end = 0.0, float(audio.sampling_rate / 2)

    scale = _resolve_scale(context)
    rc = _rc_for_scale(scale)
    if figsize is None:
        base = (10.0, 4.0)
    else:
        base = figsize
    scaled_size = (base[0] * scale, base[1] * scale)

    with rc_context(rc):
        fig = plt.figure(figsize=scaled_size)
        plt.imshow(
            _power_to_db(spectrogram.cpu().numpy()),
            aspect="auto",
            origin="lower",
            extent=(time_axis_start, time_axis_end, freq_start, freq_end),
            cmap="viridis",
        )
        plt.colorbar(label="Magnitude (dB)")
        plt.title(title)
        plt.ylabel(y_axis_label)
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.show(block=False)
        return fig


def plot_waveform_and_specgram(
    audio: Audio,
    *,
    title: str = "Waveform + Spectrogram",
    mel_scale: bool = False,
    fast_wave: bool = False,
    context: "_Context" = "auto",
    figsize: Tuple[float, float] | None = None,
    **spect_kwargs: Any,  # noqa: ANN401  # forwarded to spectrogram extraction
) -> Figure:
    """Stacked layout: waveform (top) and **mono** spectrogram (bottom). Returns the Figure.

    The waveform can be drawn in a faster, lightly decimated mode for long signals.
    Spectrogram extraction is delegated to senselab's torchaudio-based utilities
    and requires mono input.

    Args:
        audio (Audio):
            Input audio. **Spectrogram requires mono**; downmix multi-channel first.
        title (str, optional):
            Overall figure title. Defaults to `"Waveform + Spectrogram"`.
        mel_scale (bool, optional):
            If `True`, bottom panel is a mel spectrogram; otherwise linear frequency. Defaults to `False`.
        fast_wave (bool, optional):
            If `True`, waveform panel is downsampled for speed. Defaults to `False`.
        context (_Context, optional):
            Size preset or numeric scale (`"auto"`, `"small"`, `"medium"`, `"large"`, or float).
            Defaults to `"auto"`.
        figsize (tuple[float, float] | None, optional):
            Base `(width, height)` in inches **before** context scaling. Defaults to a balanced height.
        **spect_kwargs:
            Forwarded to the underlying spectrogram extractor (e.g., `n_fft`, `hop_length`, `n_mels`).

    Returns:
        matplotlib.figure.Figure: The created figure (also displayed).

    Raises:
        ValueError: If audio is not mono, or spectrogram extraction fails.

    Example:
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> fig = plot_waveform_and_specgram(
        ...     a1,
        ...     mel_scale=True,
        ...     fast_wave=True,
        ...     context="large",
        ...     n_fft=1024,
        ...     hop_length=256,
        ...     n_mels=80,
        ... )
        >>> # fig.savefig("wave_plus_mel.png")
    """
    # ---- Core timing info from ORIGINAL (non-decimated) data
    sr = audio.sampling_rate
    orig_num_frames = int(audio.waveform.size(-1))
    duration_sec = orig_num_frames / sr
    t0, t1 = 0.0, float(duration_sec)

    # ---- Prepare waveform (optionally decimated for speed)
    waveform = audio.waveform
    if fast_wave:
        waveform = waveform[..., ::10]  # decimate samples
    num_channels, num_frames = waveform.shape
    time_axis = np.linspace(0.0, duration_sec, num_frames, endpoint=False)

    # ---- Guardrail: spectrogram plotting requires mono input
    if audio.waveform.shape[0] != 1:
        raise ValueError("Only mono audio is supported for spectrogram plotting")

    # ---- Spectrogram (2D tensor: [freq_bins, time_frames])
    if mel_scale:
        from senselab.audio.tasks.features_extraction.torchaudio import (
            extract_mel_spectrogram_from_audios,
        )

        spec = extract_mel_spectrogram_from_audios([audio], **spect_kwargs)[0]["mel_spectrogram"]
        ylab = "Mel bins"
        f0, f1 = 0.0, float(spec.size(0) - 1) if torch.is_tensor(spec) and spec.ndim >= 1 else (0.0, 0.0)
        spec_title = "Mel Spectrogram"
    else:
        from senselab.audio.tasks.features_extraction.torchaudio import (
            extract_spectrogram_from_audios,
        )

        spec = extract_spectrogram_from_audios([audio], **spect_kwargs)[0]["spectrogram"]
        ylab = "Frequency [Hz]"
        f0, f1 = 0.0, float(sr / 2)
        spec_title = "Spectrogram"

    # ---- Guardrails for short/invalid outputs (exact phrase expected by tests)
    if not torch.is_tensor(spec):
        raise ValueError("Spectrogram extraction failed")
    if spec.ndim == 0 or spec.numel() == 0:
        raise ValueError("Spectrogram extraction failed")
    if spec.dtype.is_floating_point and torch.isnan(spec).any():
        raise ValueError("Spectrogram extraction failed")

    # We require a 2D (F x T) spectrogram. Anything else → fail (don’t auto-pick channels).
    if spec.ndim != 2:
        raise ValueError("Spectrogram extraction failed")

    # ---- Layout & context
    scale = _resolve_scale(context)
    rc = _rc_for_scale(scale)
    if figsize is None:
        base_h = max(2.0, 0.9 * num_channels) + 4.0  # waveform height + spectrogram
        base = (12.0, base_h)
    else:
        base = figsize
    size = (base[0] * scale, base[1] * scale)

    with rc_context(rc):
        fig, (ax_wav, ax_spec) = plt.subplots(2, 1, figsize=size, sharex=True, gridspec_kw={"height_ratios": [1, 2]})

        # ---- Waveform (top)
        if num_channels == 1:
            ax_wav.plot(time_axis, waveform[0].cpu().numpy())
            ax_wav.set_ylabel("Amp")
        else:
            for c in range(num_channels):
                ax_wav.plot(time_axis, waveform[c].cpu().numpy(), alpha=0.9 if c == 0 else 0.7)
            ax_wav.set_ylabel("Amp (multi-ch)")
        ax_wav.grid(True, alpha=0.3)
        ax_wav.set_title("Waveform")

        # ---- Spectrogram (bottom)
        im = ax_spec.imshow(
            _power_to_db(spec.cpu().numpy()),
            aspect="auto",
            origin="lower",
            extent=(t0, t1, f0, f1),
            cmap="viridis",
        )
        ax_spec.set_ylabel(ylab)
        ax_spec.set_xlabel("Time [s]")
        ax_spec.set_title(spec_title)

        # Keep both axes aligned in time
        ax_wav.set_xlim(t0, t1)
        ax_spec.set_xlim(t0, t1)

        # ---- Horizontal colorbar below the spectrogram
        divider = make_axes_locatable(ax_spec)
        cax = divider.append_axes("bottom", size="5%", pad=0.6)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("Magnitude (dB)")

        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show(block=False)
        return fig


def plot_aligned_panels(
    audio: Audio,
    panels: List[Dict[str, Any]],
    title: str = "",
    header_lines: Sequence[str] | None = None,
    header: Mapping[str, str] | None = None,
    figsize: Tuple[float, float] | None = None,
    spectrogram_params: Dict[str, Any] | None = None,
    context: _Context = "auto",
    time_limits: Tuple[float, float] | None = None,
) -> Figure:
    """Create a multi-panel time-aligned visualization from an Audio object.

    Each panel shares the same time axis. Supported panel types:

    - ``{"type": "waveform", "twin": {"name": str, "data": [(times, values, label, color), ...]},
      "spans": {"name": str, "segments": [{"label": str, "start": float, "end": float}, ...]}}`` --
      waveform amplitude on the left y-axis; the optional ``twin`` block's curves against a
      right-hand scale of their own; the optional ``spans`` block as translucent bars behind both,
      each annotated with its label. The right-hand y-label names whichever of the two are present.
      A row carrying either is drawn twice as tall, since more than one reading shares it.
    - ``{"type": "spectrogram", "mel": True/False}`` -- linear or mel spectrogram.
    - ``{"type": "features", "data": [(times, values, label, color), ...], "name": str}`` --
      scatter/line overlay of feature curves (e.g., pitch, formants). ``name`` becomes the
      panel's y-label.
    - ``{"type": "segments", "segments": [{"label": str, "start": float, "end": float}, ...],
      "name": str}`` -- colored horizontal bars for phoneme/word segments. ``name`` becomes the
      panel's y-label, so a figure stacking several lanes says which is which. Each distinct label
      is a y-tick, so this type suits a lane of a few repeating labels rather than of many texts.
    - ``{"type": "tokens", "tokens": [{"text": str, "start": float, "end": float,
      "row": str (optional)}, ...], "name": str, "fontsize": float (optional),
      "floor_fontsize": float (optional)}`` -- one bar per timed token with the token's **text drawn
      on the bar**, for a lane of many distinct texts: words, phones, a recognizer's tokens. The
      y-axis carries a tick per declared ``row`` and none at all when no token declares one, so 40
      words are 40 labelled bars rather than 40 y-ticks. A token that declares a ``row`` is drawn in
      it. Those that declare none share a block of the lane which is spread over as many unnamed
      rows as their own rendered widths take to lie side by side, decided at draw time; token ``i``
      of that block goes in row ``i mod R``, counted from the top of the block. Each label is
      measured at draw time, in points, against the slot its row leaves it — its own bar at one row,
      up to ``R`` times its bar when the row's neighbours leave the space — and one that does not fit
      at ``fontsize`` is shrunk towards ``floor_fontsize`` and dropped if it does not fit there
      either. The bar is never dropped.
    - ``{"type": "score_raster", "rows": [str, ...], "windows": [{"start": float,
      "end": float, "scores": {str: float}}], "name": str}`` -- one fixed row per selected
      label, with each native classifier window colored by its score. Missing cells mean the label
      did not clear the caller's reporting threshold, not that the classifier was never run.
    - ``{"type": "overlay_on_spectrogram", "mel": True/False, "overlays": [...]}`` --
      spectrogram with scatter overlays (each overlay is a dict with keys
      ``times``, ``values``, ``label``, ``color``, and optional ``size``).
    - ``{"type": "text", "lines": [str, ...], "fontsize": int}`` -- monospaced prose on an
      axis with no time scale, for blocks that accompany the shared axis rather than share it.
      Its height is ``max(1.0, TEXT_PANEL_INCHES_PER_LINE * len(lines))`` inches, so a long block grows the figure
      rather than overflowing its own axis.

    Args:
        audio: Input mono audio.
        panels: List of panel specification dicts (see above).
        title: Overall figure title.
        header_lines: Optional short, non-time-aligned lines placed below ``title`` and above the
            panel stack. This is for a concise decision context; it does not become another lane.
        header: Optional structured, typographic header. Its labels and values are placed above the
            panel stack as context, primary decision, leading evidence, and report-only summary;
            it takes precedence over ``header_lines``.
        figsize: Base ``(width, height)`` in inches **before** context scaling.
            Defaults to ``(14, sum_of_panel_heights)``.
        spectrogram_params: Parameters forwarded to torchaudio spectrogram transforms.
            Defaults to ``{"n_fft": 256, "hop_length": 80, "win_length": 160}``.
        context: Size preset or numeric scale.
        time_limits: Optional ``(start_s, end_s)`` interval to display. The audio and every panel
            retain their recording-time coordinates, but the shared axis is restricted to this
            interval. Both values must be within the recording duration.

    Returns:
        matplotlib.figure.Figure: The created figure.

    Raises:
        ValueError: If ``audio`` is not mono, or if a panel names a type this function does not
            draw. An unrecognised type used to yield a blank axis, so a caller's typo produced a
            figure that looked finished and said nothing.
    """
    import torchaudio.transforms as T

    # Enforce mono
    if audio.waveform.shape[0] != 1:
        raise ValueError("plot_aligned_panels requires mono audio. Downmix multi-channel first.")

    if spectrogram_params is None:
        spectrogram_params = {"n_fft": 256, "hop_length": 80, "win_length": 160}

    sr = audio.sampling_rate
    duration = audio.waveform.shape[1] / sr
    if time_limits is None:
        x_limits = (0.0, duration)
    else:
        start_s, end_s = (float(value) for value in time_limits)
        if not 0.0 <= start_s < end_s <= duration:
            raise ValueError(
                f"time_limits must satisfy 0 <= start < end <= {duration:g}; received ({start_s:g}, {end_s:g})"
            )
        x_limits = (start_s, end_s)

    # Height ratios
    ratio_map = {
        "waveform": 1,
        "spectrogram": 2,
        "features": 1,
        "segments": 1,
        "tokens": 1,
        "score_raster": 1,
        "overlay_on_spectrogram": 2,
    }

    def _ratio(panel: Dict[str, Any]) -> float:
        """One panel's share of the figure's height."""
        if "height_ratio" in panel:
            return float(panel["height_ratio"])
        ptype = panel.get("type", "waveform")
        if ptype == "text":
            return max(1.0, TEXT_PANEL_INCHES_PER_LINE * len(panel.get("lines", []))) / _INCHES_PER_RATIO
        if ptype == "waveform" and (panel.get("twin") or panel.get("spans")):
            return 2.0
        if ptype == "score_raster":
            return max(1.0, 0.45 * len(panel.get("rows") or []))
        return float(ratio_map.get(ptype, 1))

    height_ratios = [_ratio(p) for p in panels]

    scale = _resolve_scale(context)
    rc = _rc_for_scale(scale)
    if figsize is None:
        base_h = float(sum(height_ratios)) * _INCHES_PER_RATIO
        base = (14.0, max(base_h, MIN_FIGURE_HEIGHT_IN))
    else:
        base = figsize
    scaled_size = (base[0] * scale, base[1] * scale)

    start_sample = int(round(x_limits[0] * sr))
    end_sample = int(round(x_limits[1] * sr))
    waveform = audio.waveform.squeeze().cpu()[start_sample:end_sample]
    waveform_np = waveform.numpy()
    time_wav = np.arange(start_sample, end_sample, dtype=float) / sr

    def _visible_interval(start: float, end: float) -> tuple[float, float] | None:
        """The portion of one timed artist that belongs on this page."""
        start, end = max(start, x_limits[0]), min(end, x_limits[1])
        return (start, end) if end > start else None

    # Compute a spectrogram only for the page window. Rendering a 10-second page must not allocate
    # a full-recording spectrogram, particularly for long recordings on the batch runner.
    _spec_cache: Dict[str, np.ndarray] = {}

    def _get_spec(mel: bool) -> Tuple[np.ndarray, float, float]:
        key = "mel" if mel else "linear"
        if key not in _spec_cache:
            # Filter out sample_rate from params for non-mel transforms
            filtered_params = {k: v for k, v in spectrogram_params.items() if k != "sample_rate"}
            if mel:
                transform = T.MelSpectrogram(sample_rate=sr, **filtered_params)
            else:
                transform = T.Spectrogram(**filtered_params)
            spec_tensor = transform(waveform)
            _spec_cache[key] = _power_to_db(spec_tensor.cpu().numpy())
        spec_db = _spec_cache[key]
        if mel:
            return spec_db, 0.0, float(spec_db.shape[0] - 1)
        else:
            return spec_db, 0.0, float(sr / 2)

    with rc_context(rc):
        fig, axes = plt.subplots(
            len(panels),
            1,
            figsize=scaled_size,
            sharex=True,
            gridspec_kw={"height_ratios": height_ratios},
            squeeze=False,
        )
        axes_list = [axes[i, 0] for i in range(len(panels))]

        for ax, panel in zip(axes_list, panels):
            ptype = panel.get("type", "waveform")

            if ptype == "waveform":
                ax.plot(time_wav, waveform_np, linewidth=0.3, color="0.45")
                ax.set_ylabel("Amplitude")
                ax.grid(True, alpha=0.2)
                _draw_waveform_overlays(ax, panel, time_limits=x_limits)

            elif ptype == "spectrogram":
                mel = panel.get("mel", False)
                spec_db, f0, f1 = _get_spec(mel)
                ax.imshow(
                    spec_db,
                    aspect="auto",
                    origin="lower",
                    extent=[x_limits[0], x_limits[1], f0, f1],
                    cmap="magma",
                )
                ax.set_ylabel("Mel bins" if mel else "Frequency (Hz)")

            elif ptype == "features":
                data = panel.get("data", [])
                style = panel.get("style", "scatter")
                for times, values, label, color in data:
                    t_np = times.cpu().numpy() if torch.is_tensor(times) else np.asarray(times)
                    v_np = values.cpu().numpy() if torch.is_tensor(values) else np.asarray(values)
                    visible = (t_np >= x_limits[0]) & (t_np <= x_limits[1])
                    t_np, v_np = t_np[visible], v_np[visible]
                    if style == "line":
                        ax.plot(t_np, v_np, color=color, label=label, linewidth=0.8, alpha=0.8)
                    else:
                        ax.scatter(t_np, v_np, s=3, c=color, label=label, alpha=0.7)
                ax.set_ylabel(panel.get("name") or "Value")
                ax.legend(loc="upper right", fontsize=7)
                ax.grid(True, alpha=0.3)

            elif ptype == "segments":
                seg_list = [
                    seg
                    for seg in panel.get("segments", [])
                    if _visible_interval(float(seg["start"]), float(seg["end"])) is not None
                ]
                unique_labels = sorted({s["label"] for s in seg_list})
                y_map = {lbl: i for i, lbl in enumerate(unique_labels)}
                cmap = plt.get_cmap("tab20", max(len(unique_labels), 1))
                for seg in seg_list:
                    y = y_map[seg["label"]]
                    visible = _visible_interval(float(seg["start"]), float(seg["end"]))
                    if visible is None:
                        continue
                    start, end = visible
                    ax.barh(y, end - start, left=start, height=0.7, color=cmap(y), alpha=0.85, edgecolor="none")
                ax.set_yticks(range(len(unique_labels)))
                ax.set_yticklabels(unique_labels, fontsize=7)
                ax.set_ylabel(panel.get("name") or "Segment")
                ax.grid(axis="x", linestyle="--", alpha=0.3)

            elif ptype == "tokens":
                tokens = panel.get("tokens", [])
                named_rows = list(dict.fromkeys(str(token["row"]) for token in tokens if token.get("row")))
                free_rows = [""] if any(not token.get("row") for token in tokens) else []
                blocks = named_rows + free_rows
                block_of = {row: index for index, row in enumerate(blocks)}
                fontsize = float(panel.get("fontsize", TOKEN_LABEL_FONTSIZE))
                floor = float(panel.get("floor_fontsize", TOKEN_LABEL_FLOOR_FONTSIZE))
                count = max(len(blocks), 1)
                cmap = plt.get_cmap("tab20", count)
                placements: List[_TokenPlacement] = []
                for token in tokens:
                    block = block_of[str(token.get("row") or "")]
                    start, end = float(token["start"]), float(token["end"])
                    visible = _visible_interval(start, end)
                    if visible is None:
                        continue
                    start, end = visible
                    width = end - start
                    height = TOKEN_BAR_HEIGHT_FRACTION / float(count)
                    centre = (block + 0.5) / float(count)
                    color = token.get("color") or cmap(block)
                    bars = ax.barh(centre, width, left=start, height=height, color=color, alpha=0.92, edgecolor="none")
                    text = str(token.get("text") or "")
                    label = None
                    if text:
                        label = _FittedTokenLabel(
                            start + width / 2.0,
                            centre,
                            text,
                            span=(start, end),
                            full_fontsize=fontsize,
                            floor_fontsize=floor,
                            ha="center",
                            va="center",
                            fontsize=fontsize,
                            color="black",
                            transform=ax.transData,
                            clip_on=True,
                        )
                        label.set_clip_path(ax.patch)
                        ax.add_artist(label)
                    placements.append(_TokenPlacement(block, bars[0], label, start + width / 2.0, width / 2.0))
                if placements:
                    ax.add_artist(
                        _StaggeredTokenLane(
                            placements,
                            count,
                            block_of[""] if free_rows else None,
                            fontsize,
                            floor,
                            bool(panel.get("expand_label_slots", False)),
                        )
                    )
                ax.set_ylim(0.0, 1.0)
                show_row_labels = bool(panel.get("show_row_labels", True))
                ax.set_yticks(
                    [(index + 0.5) / float(count) for index in range(len(blocks))]
                    if named_rows and show_row_labels
                    else []
                )
                if named_rows and show_row_labels:
                    ax.set_yticklabels(blocks, fontsize=7)
                ax.set_ylabel(panel.get("name") or "Tokens")
                ax.grid(axis="x", linestyle="--", alpha=0.3)

            elif ptype == "score_raster":
                rows = [str(label) for label in panel.get("rows") or []]
                windows = panel.get("windows") or []
                cmap = plt.get_cmap("viridis")
                for row_index, label in enumerate(rows):
                    for window in windows:
                        scores = window.get("scores") or {}
                        if label not in scores:
                            continue
                        score = float(scores[label])
                        visible = _visible_interval(float(window["start"]), float(window["end"]))
                        if visible is None:
                            continue
                        start, end = visible
                        ax.add_patch(
                            Rectangle(
                                (start, row_index - 0.42),
                                end - start,
                                0.84,
                                facecolor=cmap(np.clip(score, 0.0, 1.0)),
                                edgecolor="none",
                            )
                        )
                ax.set_ylim(-0.5, max(len(rows) - 0.5, 0.5))
                ax.set_yticks(range(len(rows)))
                ax.set_yticklabels(rows, fontsize=7)
                ax.set_ylabel(panel.get("name") or "Label probability")
                ax.grid(axis="x", linestyle="--", alpha=0.3)
                image = ax.imshow(np.array([[0.0, 1.0]]), cmap=cmap, vmin=0.0, vmax=1.0, visible=False, aspect="auto")
                # Every timed row ends at the same x coordinate. The figure reserves a right
                # gutter for this scale, so it never steals width from just this raster row.
                color_axis = ax.inset_axes([1.025, 0.0, 0.022, 1.0])
                colorbar = fig.colorbar(image, cax=color_axis)
                colorbar.set_label("Probability", fontsize=7)
                colorbar.ax.tick_params(labelsize=6)

            elif ptype == "overlay_on_spectrogram":
                mel = panel.get("mel", False)
                spec_db, f0, f1 = _get_spec(mel)
                ax.imshow(
                    spec_db,
                    aspect="auto",
                    origin="lower",
                    extent=[x_limits[0], x_limits[1], f0, f1],
                    cmap="magma",
                    alpha=0.9,
                )
                overlays = panel.get("overlays", [])
                for ov in overlays:
                    t_np = ov["times"].cpu().numpy() if torch.is_tensor(ov["times"]) else np.asarray(ov["times"])
                    v_np = ov["values"].cpu().numpy() if torch.is_tensor(ov["values"]) else np.asarray(ov["values"])
                    visible = (t_np >= x_limits[0]) & (t_np <= x_limits[1])
                    t_np, v_np = t_np[visible], v_np[visible]
                    ax.scatter(
                        t_np,
                        v_np,
                        s=ov.get("size", 3),
                        c=ov["color"],
                        label=ov.get("label", ""),
                        zorder=3,
                        alpha=0.7,
                    )
                ax.set_ylabel("Mel bins" if mel else "Frequency (Hz)")
                if overlays:
                    ax.legend(loc="upper right", fontsize=7)

            elif ptype == "text":
                ax.axis("off")
                lines = [str(line) for line in panel.get("lines", [])]
                ax.text(
                    0.01,
                    0.98,
                    "\n".join(lines),
                    transform=ax.transAxes,
                    va="top",
                    ha="left",
                    family=panel.get("family", "monospace"),
                    fontsize=panel.get("fontsize", 8),
                )

            else:
                raise ValueError(
                    f"unknown panel type {ptype!r}; plot_aligned_panels supports "
                    "waveform, spectrogram, features, segments, tokens, score_raster, overlay_on_spectrogram and text"
                )

        # Shared x-axis config. The scale belongs to the last panel that USES it: a text panel has
        # its axis off, and with sharex that hides the tick labels for every panel above it.
        timed = [ax for ax, panel in zip(axes_list, panels) if panel.get("type", "waveform") != "text"]
        bottom = timed[-1] if timed else axes_list[-1]
        bottom.set_xlabel("Time (seconds)")
        bottom.tick_params(labelbottom=True)
        axes_list[0].set_xlim(*x_limits)

        plain_header = [str(line) for line in (header_lines or ()) if str(line)]
        # A structured report header owns the page's heading hierarchy. Drawing the legacy centered
        # title as well makes two competing headings and can clip it at the fixed PDF page boundary.
        if title and not header:
            fig.suptitle(title, y=0.997, fontsize=11)
        if header:

            def _header_lines(value: object, width: int) -> list[str]:
                return [
                    line
                    for paragraph in str(value).splitlines() or [""]
                    for line in (textwrap.wrap(paragraph, width=width, break_long_words=True) or [""])
                ]

            def _line_height_fraction(point_size: float, *, linespacing: float = 1.25) -> float:
                """One text line's height, as a fraction of this figure's actual height.

                No renderer exists yet to measure against at this point in the draw, so the step is
                derived from the figure's real physical height in inches rather than a constant tuned
                only for one figsize; it stays correct when figsize or the context scale changes.
                """
                return (point_size * linespacing) / 72.0 / scaled_size[1]

            # Header text is figure-relative rather than axis-relative. Reserve its measured line
            # count before tight_layout so it never clips into the right margin or over the first lane.
            y = 0.972
            label_gap = _line_height_fraction(8.0) + 0.003
            for label_key, value_key, fontsize, width, weight in (
                ("context_label", "context", 9.0, 168, "normal"),
                ("decision_label", "decision", 15.0, 104, "bold"),
                ("evidence_label", "evidence", 10.0, 150, "normal"),
                ("support_label", "support", 8.5, 168, "normal"),
            ):
                fig.text(0.015, y, header.get(label_key, ""), va="top", ha="left", fontsize=8, weight="bold")
                y -= label_gap
                lines = _header_lines(header.get(value_key, ""), width)
                fig.text(
                    0.015,
                    y,
                    "\n".join(lines),
                    va="top",
                    ha="left",
                    fontsize=fontsize,
                    weight=weight,
                    linespacing=1.25,
                )
                y -= 0.003 + len(lines) * _line_height_fraction(fontsize)
        elif plain_header:
            fig.text(0.015, 0.972, "\n".join(plain_header), va="top", ha="left", fontsize=8, family="sans-serif")
        top = max(0.48, y - 0.006) if header else (0.90 if plain_header else (0.96 if title else 1.0))

        # The y tick labels live in the automatic inner gutter that tight_layout measures. Lane
        # names get a distinct outer gutter, rather than sharing a rotated ylabel with tick text.
        # Score rasters additionally get a fixed right gutter for their colour scale. This keeps
        # time pixels aligned across all panels and makes the page plan independent of row count.
        lane_titles = [
            ax.get_ylabel() if panel.get("type", "waveform") != "text" else "" for ax, panel in zip(axes_list, panels)
        ]
        longest_lane_line = max(
            (len(line) for title in lane_titles for line in _lane_title_lines(title)),
            default=0,
        )
        lane_gutter_in = min(
            REPORT_LANE_GUTTER_MAX_IN,
            max(REPORT_LANE_GUTTER_MIN_IN, 0.18 + 0.055 * float(longest_lane_line)),
        )
        for ax, title in zip(axes_list, lane_titles):
            if title:
                # Preserve ylabel metadata for callers that use it to identify an axis, but draw
                # its visible representation in the dedicated report lane-title gutter below.
                ax.yaxis.label.set_visible(False)
        fig.tight_layout(
            rect=(lane_gutter_in / scaled_size[0], 0.0, 1.0 - REPORT_COLORBAR_GUTTER_IN / scaled_size[0], top),
            pad=0.6,
            h_pad=0.8,
        )
        for ax, title in zip(axes_list, lane_titles):
            if not title:
                continue
            position = ax.get_position()
            available_height_pt = position.height * scaled_size[1] * 72.0
            max_lines = max(1, int(available_height_pt // (8.0 * 1.35)))
            lane_text = fig.text(
                lane_gutter_in / (2.0 * scaled_size[0]),
                (position.y0 + position.y1) / 2.0,
                "\n".join(_lane_title_lines(title, max_lines=max_lines)),
                va="center",
                ha="center",
                fontsize=8,
                linespacing=1.2,
            )
            lane_text.set_gid("senselab-lane-title")
        plt.show(block=False)
        return fig


def play_audio(audio: Audio) -> None:
    """Play an `Audio` object inline (Jupyter/IPython), supporting 1–2 channels.

    Uses `IPython.display.Audio` to render audio widgets in notebooks. For more
    than two channels, downmix first.

    Args:
        audio (Audio):
            Input audio to play (mono or stereo). Sampling rate is preserved.

    Raises:
        ValueError: If the waveform has more than 2 channels.

    Example:
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> play_audio(a1)
    """
    from IPython.display import Audio as DisplayAudio
    from IPython.display import display

    waveform = audio.waveform.cpu().numpy()
    sample_rate = audio.sampling_rate

    num_channels = waveform.shape[0]
    if num_channels == 1:
        display(DisplayAudio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(DisplayAudio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels is not supported.")
