"""Drawing the family x gate matrix as a heatmap.

Everything in :class:`HeatmapStyle` governs the drawing and nothing else. The figure is a view of a
:class:`~senselab.audio.workflows.triage.routing_analysis.gate_matrix.GateMatrix` and computes
nothing of its own.

A cell no recording could evaluate is hatched and left unannotated rather than painted at the
bottom of the scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from senselab.audio.workflows.triage.routing_analysis.gate_matrix import GateMatrix, routing_branch_of
from senselab.audio.workflows.triage.routing_analysis.ruleset import Ruleset

FIRED_PANEL = "fired_rate_evaluable"
"""The panel showing how often a gate fired among the recordings it could read."""

UNAVAILABLE_PANEL = "unavailable_rate"
"""The panel showing how often a gate's evidence was never written."""

PANELS: tuple[str, ...] = (FIRED_PANEL, UNAVAILABLE_PANEL)
"""Both panels the heatmap can draw, in page order."""

_TITLES = {
    FIRED_PANEL: "fired, of the recordings the gate could read",
    UNAVAILABLE_PANEL: "evidence never written",
}


@dataclass(frozen=True)
class HeatmapStyle:
    """How the matrix is drawn. Nothing here is read by the pipeline or by the aggregation.

    Attributes:
        cell_width_in: Width one gate column takes, in inches.
        cell_height_in: Height one family row takes, in inches.
        label_width_in: Width reserved for the family labels, in inches.
        header_height_in: Height reserved above the cells for the title, in inches.
        gate_label_height_in: Height reserved below the cells for the rotated gate labels, in
            inches.
        colorbar_width_in: Width reserved to the right of the cells for the colour bar, in inches.
        fired_cmap: Colormap for the fired-rate panel.
        unavailable_cmap: Colormap for the unavailable-rate panel.
        missing_color: What a cell with no rate to draw is painted.
        missing_hatch: The hatch drawn over such a cell.
        annotate: Whether each cell carries its rate as text.
        annotate_counts: Whether an annotated cell also carries its ``n``.
        light_text_over: Rate above which a cell's annotation is drawn light rather than dark.
        value_fontsize: Point size of a cell annotation.
        label_fontsize: Point size of the family and gate labels.
        title_fontsize: Point size of the panel titles.
        dpi: Raster resolution of the saved figure.
        grid_color: Colour of the lines between cells.
        grid_width: Width of those lines, in points.
        branch_rule_width: Width of the heavier line between one branch's gates and the next.
        sort_families_by_n: Whether rows are ordered by descending recording count rather than by
            name.
    """

    cell_width_in: float = 0.62
    cell_height_in: float = 0.26
    label_width_in: float = 3.1
    header_height_in: float = 1.0
    gate_label_height_in: float = 2.0
    colorbar_width_in: float = 0.95
    fired_cmap: str = "viridis"
    unavailable_cmap: str = "magma"
    missing_color: str = "#d9d9d9"
    missing_hatch: str = "xx"
    annotate: bool = True
    annotate_counts: bool = False
    light_text_over: float = 0.55
    value_fontsize: float = 5.0
    label_fontsize: float = 6.0
    title_fontsize: float = 9.0
    dpi: int = 200
    grid_color: str = "white"
    grid_width: float = 0.5
    branch_rule_width: float = 1.6
    sort_families_by_n: bool = True


def ordered_families(matrix: GateMatrix, style: HeatmapStyle) -> tuple[str, ...]:
    """The row order one panel is drawn in.

    Args:
        matrix: The matrix.
        style: The drawing style.

    Returns:
        The families, by descending recording count then by name, or by name alone.
    """
    if not style.sort_families_by_n:
        return matrix.families
    return tuple(sorted(matrix.families, key=lambda family: (-matrix.recordings[family], family)))


def panel_values(matrix: GateMatrix, families: tuple[str, ...], panel: str) -> np.ndarray:
    """One panel's rates, with NaN wherever there is no rate to draw.

    Args:
        matrix: The matrix.
        families: The row order.
        panel: One of :data:`PANELS`.

    Returns:
        A ``len(families) x len(matrix.gates)`` array. A NaN entry is a cell no recording could
        evaluate.
    """
    grid = np.full((len(families), len(matrix.gates)), np.nan, dtype=float)
    for row, family in enumerate(families):
        for column, gate in enumerate(matrix.gates):
            rate = getattr(matrix.cell(family, gate), panel)
            if rate is not None:
                grid[row, column] = rate
    return grid


def branch_boundaries(matrix: GateMatrix, ruleset: Ruleset) -> list[int]:
    """Where one branch's gates give way to the next, as column indices.

    Args:
        matrix: The matrix.
        ruleset: The loaded ruleset.

    Returns:
        Each column index a heavier vertical rule is drawn before.
    """
    branches = [routing_branch_of(ruleset, gate) for gate in matrix.gates]
    return [index for index in range(1, len(branches)) if branches[index] != branches[index - 1]]


def _mark_missing(axes: Axes, grid: np.ndarray, style: HeatmapStyle) -> None:
    """Hatch every cell that has no rate to draw.

    Args:
        axes: The axes to draw on.
        grid: The panel's rates.
        style: The drawing style.
    """
    for row, column in zip(*np.nonzero(np.isnan(grid))):
        axes.add_patch(
            Rectangle(
                (column - 0.5, row - 0.5),
                1.0,
                1.0,
                facecolor=style.missing_color,
                edgecolor=style.grid_color,
                hatch=style.missing_hatch,
                linewidth=style.grid_width,
                zorder=2.0,
            )
        )


def _annotate(axes: Axes, grid: np.ndarray, matrix: GateMatrix, families: tuple[str, ...], style: HeatmapStyle) -> None:
    """Write each cell's rate over it, and nothing over a cell that has none.

    Args:
        axes: The axes to draw on.
        grid: The panel's rates.
        matrix: The matrix, read for each cell's ``n``.
        families: The row order.
        style: The drawing style.
    """
    for row, family in enumerate(families):
        for column, gate in enumerate(matrix.gates):
            rate = float(grid[row, column])
            if np.isnan(rate):
                continue
            text = f"{rate:.2f}"
            if style.annotate_counts:
                text = f"{text}\nn{matrix.cell(family, gate).n}"
            axes.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                fontsize=style.value_fontsize,
                color="white" if rate > style.light_text_over else "black",
                zorder=3.0,
            )


def draw_gate_matrix(
    matrix: GateMatrix,
    ruleset: Ruleset,
    *,
    panel: str = FIRED_PANEL,
    style: HeatmapStyle | None = None,
    subtitle: str = "",
) -> Figure:
    """Draw one panel of the matrix: task family down, gate across.

    Args:
        matrix: The matrix to draw.
        ruleset: The loaded ruleset, read only for where the branch rules go.
        panel: One of :data:`PANELS`.
        style: The drawing style, or None for the default.
        subtitle: A line under the title, naming the corpus the matrix was built on.

    Returns:
        The figure.

    Raises:
        ValueError: When ``panel`` is not one of :data:`PANELS`, or when the matrix has no axes
            to draw.
    """
    if panel not in PANELS:
        raise ValueError(f"panel is not one of {PANELS}: {panel!r}")
    if not matrix.families or not matrix.gates:
        raise ValueError(f"nothing to draw: {len(matrix.families)} families and {len(matrix.gates)} gates")
    style = style or HeatmapStyle()
    families = ordered_families(matrix, style)
    grid = panel_values(matrix, families, panel)

    cells_width = style.cell_width_in * len(matrix.gates)
    cells_height = style.cell_height_in * len(families)
    width = style.label_width_in + cells_width + style.colorbar_width_in
    height = style.header_height_in + cells_height + style.gate_label_height_in
    figure = Figure(figsize=(width, height), dpi=style.dpi)
    axes = figure.add_axes(
        (
            style.label_width_in / width,
            style.gate_label_height_in / height,
            cells_width / width,
            cells_height / height,
        )
    )
    axes.set_facecolor(style.missing_color)
    image = axes.imshow(
        np.ma.masked_invalid(grid),
        aspect="auto",
        cmap=style.fired_cmap if panel == FIRED_PANEL else style.unavailable_cmap,
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    _mark_missing(axes, grid, style)
    axes.set_xticks(range(len(matrix.gates)))
    axes.set_xticklabels(matrix.gates, rotation=90, fontsize=style.label_fontsize)
    axes.set_yticks(range(len(families)))
    axes.set_yticklabels(
        [f"{family}  (n={matrix.recordings[family]})" for family in families],
        fontsize=style.label_fontsize,
    )
    axes.set_xticks(np.arange(-0.5, len(matrix.gates), 1.0), minor=True)
    axes.set_yticks(np.arange(-0.5, len(families), 1.0), minor=True)
    axes.grid(which="minor", color=style.grid_color, linewidth=style.grid_width)
    axes.tick_params(which="minor", length=0)
    for boundary in branch_boundaries(matrix, ruleset):
        axes.axvline(boundary - 0.5, color="black", linewidth=style.branch_rule_width, zorder=4.0)
    if style.annotate:
        _annotate(axes, grid, matrix, families, style)

    title = f"ruleset gates by task family — {_TITLES[panel]}"
    hatched = int(np.count_nonzero(np.isnan(grid)))
    legend = f"hatched: no recording of the family could evaluate the gate ({hatched} cells)"
    lines = [title, subtitle, legend] if subtitle else [title, legend]
    axes.set_title("\n".join(lines), fontsize=style.title_fontsize, pad=10)
    bar = figure.colorbar(image, ax=axes, fraction=0.025, pad=0.012)
    bar.ax.tick_params(labelsize=style.label_fontsize)
    bar.set_label("rate", fontsize=style.label_fontsize)
    return figure


def write_gate_matrix(
    matrix: GateMatrix,
    ruleset: Ruleset,
    path: Path,
    *,
    panel: str = FIRED_PANEL,
    style: HeatmapStyle | None = None,
    subtitle: str = "",
) -> Path:
    """Draw one panel and save it.

    Args:
        matrix: The matrix to draw.
        ruleset: The loaded ruleset.
        path: Where to write. The suffix decides the format.
        panel: One of :data:`PANELS`.
        style: The drawing style, or None for the default.
        subtitle: A line under the title.

    Returns:
        The path written.
    """
    resolved = style or HeatmapStyle()
    figure = draw_gate_matrix(matrix, ruleset, panel=panel, style=resolved, subtitle=subtitle)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=resolved.dpi)
    return path
