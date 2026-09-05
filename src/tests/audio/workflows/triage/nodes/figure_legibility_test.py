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

    def test_the_asr_row_names_the_null_key_that_skipped_it(self, config: TriageConfig) -> None:
        """The packaged config leaves speech.word_gap_ms null, which drops the source silently."""
        assert config.get("speech.word_gap_ms") is None

        reasons = _span_row_absence(config, {}, [])

        assert "speech.word_gap_ms" in reasons["A"]

    def test_a_row_that_proposed_a_span_says_nothing(self, config: TriageConfig) -> None:
        """A source with a span of its own needs no explanation."""
        spans = [{"signal": "preemphasised", "measure": "amplitude"}]

        assert "E" not in _span_row_absence(config, {}, spans)

    def test_it_prefers_the_producing_node_s_own_reason(self, config: TriageConfig) -> None:
        """Where PREPROCESS recorded why a derivative is absent, that text is used verbatim."""
        reasons = _span_row_absence(config, {"continuity_trace": "ValueError: nobody measured it"}, [])

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
