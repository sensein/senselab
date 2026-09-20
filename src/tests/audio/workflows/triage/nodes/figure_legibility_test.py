"""FIGURE's drawing decisions: the waveform's scale, absent panels' height, and empty span rows."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402 — the Agg backend must be selected before pyplot loads

from senselab.audio.workflows.triage.config import TriageConfig  # noqa: E402
from senselab.audio.workflows.triage.nodes.figure import (  # noqa: E402
    FigureStyle,
    _page_height_ratios,
    _span_row_absence,
    _waveform_panel,
)


def _draw_waveform(samples: np.ndarray, style: FigureStyle) -> tuple[float, float]:
    """Draw one waveform panel and return the y-limits it chose.

    Args:
        samples: The conditioned stream.
        style: The drawing configuration.

    Returns:
        ``(low, high)``.
    """
    figure, axis = plt.subplots()
    try:
        _waveform_panel(
            axis,
            samples,
            16000,
            None,
            None,
            None,
            (0.0, 1.0),
            style,
            k_db=None,
            cut_level=None,
            cut_percentile=None,
            continuity_absent="continuity_trace is absent from the store",
        )
        return axis.get_ylim()
    finally:
        plt.close(figure)


class TestTheWaveformIsVisible:
    """A conditioned stream sits well below full scale, where a full-scale axis hides it."""

    def test_the_axis_tracks_the_signal_rather_than_full_scale(self) -> None:
        """A 0.05-peak signal fills its panel instead of reading as a flat line on zero."""
        low, high = _draw_waveform(0.05 * np.sin(np.linspace(0.0, 40.0, 16000)), FigureStyle())

        assert 0.05 < high < 0.1, f"expected the axis near the 0.05 peak, got {high}"
        assert low == pytest.approx(-high), "the axis stays symmetric about zero"

    def test_the_peak_stays_inside_the_axis(self) -> None:
        """Headroom is above one, so the loudest sample is never clipped by the frame."""
        samples = 0.3 * np.sin(np.linspace(0.0, 40.0, 16000))
        _, high = _draw_waveform(samples, FigureStyle())

        assert high > float(np.abs(samples).max())

    def test_a_near_silent_page_does_not_zoom_into_its_own_noise(self) -> None:
        """A floor keeps a silent page from magnifying dither into apparent signal."""
        style = FigureStyle()
        _, high = _draw_waveform(np.full(16000, 1e-6), style)

        assert high == pytest.approx(style.waveform_min_amplitude)

    def test_the_scale_is_a_drawing_choice(self) -> None:
        """Both limits come from the style, so neither is a pipeline value in disguise."""
        wide = _draw_waveform(0.05 * np.ones(16000), FigureStyle(waveform_headroom=4.0))
        narrow = _draw_waveform(0.05 * np.ones(16000), FigureStyle(waveform_headroom=1.05))

        assert wide[1] > narrow[1]


class TestAnAbsentPanelGivesUpItsHeight:
    """An absent panel says one line; it should not spend a fifth of the page saying it."""

    def test_collapsing_preserves_the_page_total(self) -> None:
        """Redistribution keeps the figure's height, so pages stay comparable."""
        style = FigureStyle()

        assert sum(_page_height_ratios(style, [])) == pytest.approx(sum(style.height_ratios))
        assert sum(_page_height_ratios(style, [3, 4])) == pytest.approx(sum(style.height_ratios))

    def test_a_collapsed_panel_shrinks_and_the_rest_grow(self) -> None:
        """The rasters collapse to a strip and the panels with data take what they gave up."""
        style = FigureStyle()

        ratios = _page_height_ratios(style, [3, 4])

        assert ratios[3] == pytest.approx(style.absent_height_ratio)
        assert ratios[4] == pytest.approx(style.absent_height_ratio)
        assert ratios[0] > style.height_ratios[0]
        assert ratios[1] > style.height_ratios[1]

    def test_a_panel_already_shorter_than_the_strip_is_left_alone(self) -> None:
        """Collapsing never makes a panel taller than it was."""
        style = FigureStyle()

        ratios = _page_height_ratios(style, [4])

        assert ratios[4] == pytest.approx(min(style.height_ratios[4], style.absent_height_ratio))


class TestAnEmptySpanRowSaysWhy:
    """An empty row is a skipped source, never a source that ran and found nothing."""

    def test_the_asr_row_names_the_absent_transcript(self, config: TriageConfig) -> None:
        """No transcript, no ASR span — and the row says which, rather than reading as a finding."""
        reasons = _span_row_absence({"consensus_transcript": "ValueError: no recognizer agreed"}, [])

        assert "recognizer" in reasons["A"]

    def test_an_empty_asr_row_with_a_transcript_means_nothing_was_novel(self, config: TriageConfig) -> None:
        """Every candidate corroborated a span an earlier source already covered."""
        reasons = _span_row_absence({}, [])

        assert reasons["A"] == "no asr span was novel"

    def test_a_row_that_proposed_a_span_says_nothing(self, config: TriageConfig) -> None:
        """A source with a span of its own needs no explanation."""
        spans = [{"signal": "preemphasised", "measure": "amplitude"}]

        assert "E" not in _span_row_absence({}, spans)

    def test_it_prefers_the_producing_node_s_own_reason(self, config: TriageConfig) -> None:
        """Where PREPROCESS recorded why a derivative is absent, that text is used verbatim."""
        reasons = _span_row_absence({"continuity_trace": "ValueError: nobody measured it"}, [])

        assert reasons["C"] == "ValueError: nobody measured it"


class TestTheWordLaneReadsAsSpeech:
    """The staggered rows stay; their height is what makes the words scan continuously."""

    def test_a_bar_is_shorter_than_its_row_pitch(self) -> None:
        """Rows one unit apart with a sub-unit bar leaves a visible gap rather than a solid band."""
        style = FigureStyle()

        assert 0.0 < style.asr_row_height < 1.0

    def test_the_staggering_is_preserved(self) -> None:
        """More than one row, so a word's label can use the width its neighbours are not using."""
        assert FigureStyle().asr_rows > 1


class TestCoverTextFitsThePage:
    """A cover line wider than the page is truncated mid-word with no error of any kind."""

    def test_a_full_width_line_stays_inside_the_printed_margins(self) -> None:
        """Measured against the rendered extent, so page size, point size and column count agree."""
        from senselab.audio.workflows.triage.nodes.figure import monospace_columns
        from senselab.audio.workflows.triage.nodes.report import _BLOCK_COLUMNS, _BLOCK_FONTSIZE

        style = FigureStyle()
        figure = plt.figure(figsize=style.figure_inches)
        renderer = figure.canvas.get_renderer()
        text = figure.text(0.0, 0.5, "M" * _BLOCK_COLUMNS, family="monospace", fontsize=_BLOCK_FONTSIZE)
        drawn_in = text.get_window_extent(renderer=renderer).width / figure.dpi
        plt.close(figure)
        drawable_in = style.figure_inches[0] - 2.0 * style.cover_margin_in
        assert drawn_in <= drawable_in, f"{_BLOCK_COLUMNS} columns draw {drawn_in:.3f}in into {drawable_in:.3f}in"
        assert _BLOCK_COLUMNS == monospace_columns(style, _BLOCK_FONTSIZE)

    def test_one_more_column_would_not_fit(self) -> None:
        """The width is the page's, not a round number that happens to be under it."""
        from senselab.audio.workflows.triage.nodes.report import _BLOCK_COLUMNS, _BLOCK_FONTSIZE

        style = FigureStyle()
        figure = plt.figure(figsize=style.figure_inches)
        renderer = figure.canvas.get_renderer()
        text = figure.text(0.0, 0.5, "M" * (_BLOCK_COLUMNS + 1), family="monospace", fontsize=_BLOCK_FONTSIZE)
        drawn_in = text.get_window_extent(renderer=renderer).width / figure.dpi
        plt.close(figure)
        assert drawn_in > style.figure_inches[0] - 2.0 * style.cover_margin_in

    def test_every_wrapped_cover_line_is_within_the_width(self) -> None:
        """The wrapper is what enforces it, so a long measurement line must come back folded."""
        from senselab.audio.workflows.triage.nodes.report import _BLOCK_COLUMNS, _wrapped

        long_line = "  SPEECH: " + "; ".join(f"measure_{index}=0.{index:03d}" for index in range(40))
        assert len(long_line) > _BLOCK_COLUMNS
        folded = _wrapped([long_line])
        assert len(folded) > 1
        assert all(len(line) <= _BLOCK_COLUMNS for line in folded)


class TestPageTitleFitsThePage:
    """A span page's heading names the recording; clipped, it names nothing."""

    @staticmethod
    def _drawn_in(figure: "plt.Figure", line: str, fontsize: float) -> float:
        """How wide one line draws, in inches."""
        artist = figure.text(0.0, 0.0, line, fontsize=fontsize)
        width = artist.get_window_extent().width / figure.dpi
        artist.remove()
        return width

    def test_a_real_stem_is_folded_rather_than_run_off_both_edges(self) -> None:
        """The stems this corpus carries are longer than a landscape page at 10 pt."""
        from senselab.audio.workflows.triage.nodes.figure import PAGE_TITLE_FONTSIZE, wrap_measured

        style = FigureStyle()
        figure = plt.figure(figsize=style.figure_inches)
        drawable = style.figure_inches[0] - 2.0 * style.cover_margin_in
        heading = (
            "sub-00053adb-a1f4-4724-a694-c10e01b8cbe6_ses-33F6D051-4580-43FF-BC3B-14CD8B86CA3D"
            "_task-diadochokinesis-buttercup_20260919-200235 — page 1, 0-20s of 6.27s"
        )
        assert self._drawn_in(figure, heading, PAGE_TITLE_FONTSIZE) > drawable
        folded = wrap_measured(figure, heading, fontsize=PAGE_TITLE_FONTSIZE, drawable_in=drawable)
        assert "\n" in folded
        for line in folded.split("\n"):
            assert self._drawn_in(figure, line, PAGE_TITLE_FONTSIZE) <= drawable
        plt.close(figure)

    def test_one_unbroken_word_is_split_rather_than_overflowed(self) -> None:
        """A stem carries no spaces, so word wrapping alone cannot fold it."""
        from senselab.audio.workflows.triage.nodes.figure import PAGE_TITLE_FONTSIZE, wrap_measured

        style = FigureStyle()
        figure = plt.figure(figsize=style.figure_inches)
        drawable = style.figure_inches[0] - 2.0 * style.cover_margin_in
        folded = wrap_measured(figure, "A" * 400, fontsize=PAGE_TITLE_FONTSIZE, drawable_in=drawable)
        assert len(folded.split("\n")) > 1
        for line in folded.split("\n"):
            assert self._drawn_in(figure, line, PAGE_TITLE_FONTSIZE) <= drawable
        plt.close(figure)

    def test_a_heading_that_already_fits_is_left_on_one_line(self) -> None:
        """Folding a short heading would cost a line of the page for nothing."""
        from senselab.audio.workflows.triage.nodes.figure import PAGE_TITLE_FONTSIZE, wrap_measured

        style = FigureStyle()
        figure = plt.figure(figsize=style.figure_inches)
        drawable = style.figure_inches[0] - 2.0 * style.cover_margin_in
        heading = "sub-a_task-loudness — page 1, 0-20s of 6.27s"
        assert wrap_measured(figure, heading, fontsize=PAGE_TITLE_FONTSIZE, drawable_in=drawable) == heading
        plt.close(figure)


class TestCoverBodyFitsThePage:
    """A cover body longer than the page is drawn from the top and clipped, losing its tail."""

    def test_a_body_at_capacity_stays_inside_its_axis(self) -> None:
        """Measured on the drawn artist against the drawn axis, not on a line count."""
        from matplotlib.layout_engine import ConstrainedLayoutEngine

        from senselab.audio.workflows.triage.nodes.figure import (
            _taxonomy_panel,
            cover_body_capacity,
            cover_body_rect,
        )

        style = FigureStyle(figure_inches=(11.0, 8.5))
        capacity = cover_body_capacity(style, 1)
        figure = plt.figure(figsize=style.figure_inches, layout=ConstrainedLayoutEngine(rect=cover_body_rect(style, 1)))
        axis = figure.add_subplot()
        artist = _taxonomy_panel(axis, [f"line {index:03d}" for index in range(capacity)], style)
        figure.canvas.draw()
        assert artist.get_window_extent().y0 >= axis.get_window_extent().y0
        plt.close(figure)

    def test_a_body_well_past_capacity_would_overflow_undrawn(self) -> None:
        """The capacity is the page's, not a number that happens to be under it."""
        from matplotlib.layout_engine import ConstrainedLayoutEngine

        from senselab.audio.workflows.triage.nodes.figure import (
            _taxonomy_panel,
            cover_body_capacity,
            cover_body_rect,
        )

        style = FigureStyle(figure_inches=(11.0, 8.5))
        over = cover_body_capacity(style, 1) + 12
        figure = plt.figure(figsize=style.figure_inches, layout=ConstrainedLayoutEngine(rect=cover_body_rect(style, 1)))
        axis = figure.add_subplot()
        artist = _taxonomy_panel(axis, [f"line {index:03d}" for index in range(over)], style)
        figure.canvas.draw()
        assert artist.get_window_extent().y0 < axis.get_window_extent().y0
        plt.close(figure)

    def test_every_paginated_page_is_within_capacity_and_nothing_is_lost(self) -> None:
        """Pagination must preserve every non-blank line, in order."""
        from senselab.audio.workflows.triage.nodes.figure import paginate_body

        body = [f"line {index:03d}" if index % 17 else "" for index in range(240)]
        pages_out = paginate_body(body, 40)
        assert len(pages_out) > 1
        assert all(len(page) <= 40 for page in pages_out)
        kept = [line for page in pages_out for line in page if line.strip()]
        assert kept == [line for line in body if line.strip()]

    def test_a_body_that_already_fits_stays_one_page(self) -> None:
        """Paginating a short body would cost a page for nothing."""
        from senselab.audio.workflows.triage.nodes.figure import paginate_body

        body = ["one", "", "two", "three"]
        assert paginate_body(body, 40) == [body]

    def test_a_break_prefers_a_blank_line(self) -> None:
        """A section split mid-block reads as two unrelated fragments."""
        from senselab.audio.workflows.triage.nodes.figure import paginate_body

        body = ["a1", "a2", "", "b1", "b2", "b3", "b4"]
        assert paginate_body(body, 5) == [["a1", "a2"], ["b1", "b2", "b3", "b4"]]
