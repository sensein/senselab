"""This script contains unit tests for the plotting tasks."""

from typing import Any, Dict, List, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from matplotlib.artist import Artist
from matplotlib.backend_bases import RendererBase
from matplotlib.patches import Rectangle
from matplotlib.pyplot import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.plotting.plotting import (
    TOKEN_LABEL_FLOOR_FONTSIZE,
    TOKEN_LABEL_FONTSIZE,
    TOKEN_ROW_PITCH_EM,
    _fitted_token_fontsize,
    _staggered_row_ceiling,
    _staggered_row_count,
    _token_label_slots,
    play_audio,
    plot_aligned_panels,
    plot_specgram,
    plot_waveform,
)


def _tone(seconds: float = 1.0) -> Audio:
    """A mono waveform of the requested length, enough for a shared time axis."""
    samples = int(16000 * seconds)
    return Audio(waveform=torch.linspace(-0.5, 0.5, samples).unsqueeze(0), sampling_rate=16000)


_STORY_RECALL_PROSE = (
    "The story was about a grandfather who lived alone beside the river "
    "and mended nets for the village children every summer"
)


def _connected_speech(duration: float) -> List[Dict[str, Any]]:
    """Word extents at the density of the Story-recall page: 0.20-0.40 s bars, 3-9 character words."""
    tokens: List[Dict[str, Any]] = []
    cursor = 0.4
    words = _STORY_RECALL_PROSE.split()
    for index in range(len(words) * 8):
        word = words[index % len(words)]
        width = 0.20 + 0.02 * (len(word) % 11)
        if cursor + width > duration:
            break
        tokens.append({"text": word, "start": cursor, "end": cursor + width})
        cursor += width + 0.05
    return tokens


class TestTheTextPanel:
    """A panel that carries prose beside the shared time axis, for a report's per-step blocks."""

    def test_a_text_panel_renders_its_lines(self) -> None:
        """The lines reach the axis; a blank panel is the failure this type exists to prevent."""
        figure = plot_aligned_panels(
            _tone(), [{"type": "waveform"}, {"type": "text", "lines": ["triage: pass", "release: withheld"]}]
        )
        texts = [t.get_text() for ax in figure.axes for t in ax.texts]
        assert any("triage: pass" in text for text in texts)
        assert any("release: withheld" in text for text in texts)

    def test_a_text_panel_has_no_time_axis(self) -> None:
        """It carries no data over time, so it must not claim a shared x-scale it does not use."""
        figure = plot_aligned_panels(_tone(), [{"type": "waveform"}, {"type": "text", "lines": ["a"]}])
        assert not figure.axes[-1].axison

    def test_a_long_block_grows_the_figure(self) -> None:
        """A verdict block with many reasons must grow the page, not overflow its own axis."""
        short = plot_aligned_panels(_tone(), [{"type": "waveform"}, {"type": "text", "lines": ["a"]}])
        tall = plot_aligned_panels(
            _tone(), [{"type": "waveform"}, {"type": "text", "lines": [f"line {i}" for i in range(60)]}]
        )
        assert tall.get_size_inches()[1] > short.get_size_inches()[1]

    def test_the_time_axis_stays_labelled_under_a_trailing_text_panel(self) -> None:
        """A text panel turns its axis off; with sharex that used to hide every panel's time scale."""
        figure = plot_aligned_panels(
            _tone(), [{"type": "waveform"}, {"type": "spectrogram"}, {"type": "text", "lines": ["a"]}]
        )
        assert figure.axes[1].get_xlabel() == "Time (seconds)"
        assert any(tick.get_text() for tick in figure.axes[1].get_xticklabels())

    def test_time_limits_restrict_the_shared_axis_without_retiming_data(self) -> None:
        """A report page may show one recording-time slice without retiming the recording."""
        figure = plot_aligned_panels(
            _tone(30.0),
            [{"type": "waveform"}, {"type": "segments", "segments": [], "name": "airway"}],
            time_limits=(10.0, 20.0),
        )
        assert figure.axes[0].get_xlim() == pytest.approx((10.0, 20.0))
        assert figure.axes[1].get_xlim() == pytest.approx((10.0, 20.0))

    def test_time_limits_draw_only_waveform_and_span_labels_on_the_visible_page(self) -> None:
        """An off-page label must not be rendered or expand a fixed-size PDF page."""
        figure = plot_aligned_panels(
            _tone(30.0),
            [
                {
                    "type": "waveform",
                    "spans": {
                        "name": "proposals",
                        "segments": [
                            {"label": "early", "start": 1.0, "end": 2.0},
                            {"label": "visible", "start": 11.0, "end": 12.0},
                        ],
                    },
                },
                {"type": "spectrogram"},
            ],
            time_limits=(10.0, 20.0),
        )
        waveform = figure.axes[0]
        assert waveform.lines[0].get_xdata().min() >= 10.0
        assert waveform.lines[0].get_xdata().max() < 20.0
        assert [text.get_text() for text in waveform.texts] == ["visible"]
        spectrogram = next(axis for axis in figure.axes if axis.get_ylabel() == "Frequency (Hz)")
        assert spectrogram.images[0].get_extent()[:2] == [10.0, 20.0]

    def test_a_zero_width_span_inside_the_page_is_not_drawn(self) -> None:
        """A degenerate span must not draw a label with nothing visible behind it."""
        figure = plot_aligned_panels(
            _tone(30.0),
            [
                {
                    "type": "waveform",
                    "spans": {
                        "name": "proposals",
                        "segments": [{"label": "instant", "start": 15.0, "end": 15.0}],
                    },
                },
            ],
            time_limits=(10.0, 20.0),
        )
        waveform = figure.axes[0]
        assert [text.get_text() for text in waveform.texts] == []

    def test_time_limits_reject_an_interval_outside_the_recording(self) -> None:
        """A misleading evidence page is worse than a rendering error."""
        with pytest.raises(ValueError, match="time_limits"):
            plot_aligned_panels(_tone(), [{"type": "waveform"}], time_limits=(0.0, 2.0))

    def test_score_raster_uses_fixed_label_rows_and_a_probability_legend(self) -> None:
        """Classifier scores should read as a heat map, not an unstable comma-separated label list."""
        figure = plot_aligned_panels(
            _tone(10.0),
            [
                {
                    "type": "score_raster",
                    "name": "yamnet labels",
                    "rows": ["Speech", "Music"],
                    "windows": [
                        {"start": 0.0, "end": 1.0, "scores": {"Speech": 0.8, "Music": 0.6}},
                        {"start": 1.0, "end": 2.0, "scores": {"Speech": 0.9}},
                    ],
                }
            ],
        )
        assert [tick.get_text() for tick in figure.axes[0].get_yticklabels()] == ["Speech", "Music"]
        assert figure.axes[0].child_axes[0].get_ylabel() == "Probability"

    def test_a_probability_colorbar_does_not_narrow_its_shared_time_axis(self) -> None:
        """The raster must align horizontally with the waveform it explains."""
        figure = plot_aligned_panels(
            _tone(10.0),
            [
                {"type": "waveform"},
                {
                    "type": "score_raster",
                    "name": "yamnet labels",
                    "rows": ["Speech"],
                    "windows": [{"start": 0.0, "end": 1.0, "scores": {"Speech": 0.8}}],
                },
            ],
        )
        waveform_position = figure.axes[0].get_position()
        raster_position = figure.axes[1].get_position()
        assert raster_position.x0 == pytest.approx(waveform_position.x0)
        assert raster_position.x1 == pytest.approx(waveform_position.x1)

    def test_every_timed_panel_uses_the_same_time_to_pixel_transform(self) -> None:
        """A timestamp must land on the same pixel in every report row, not merely share x-limits."""
        figure = plot_aligned_panels(
            _tone(10.0),
            [
                {
                    "type": "waveform",
                    "twin": {"name": "dBFS", "data": [([2.0, 8.0], [-40.0, -30.0], "floor", "firebrick")]},
                },
                {"type": "segments", "name": "airway", "segments": [{"label": "Breathe", "start": 2.0, "end": 8.0}]},
                {"type": "tokens", "name": "consensus ASR", "tokens": [{"text": "word", "start": 2.0, "end": 8.0}]},
                {
                    "type": "score_raster",
                    "name": "yamnet labels",
                    "rows": ["Breathing"],
                    "windows": [{"start": 2.0, "end": 8.0, "scores": {"Breathing": 0.8}}],
                },
                {"type": "spectrogram"},
            ],
            time_limits=(1.0, 9.0),
        )
        figure.canvas.draw()
        timed_axes = [axis for axis in figure.axes if axis.axison and axis.get_xlim() == pytest.approx((1.0, 9.0))]
        reference = timed_axes[0].transData.transform([(2.0, 0.0), (8.0, 0.0)])[:, 0]
        assert len(timed_axes) == 6  # Five rows plus the waveform's twin dBFS axis.
        for axis in timed_axes[1:]:
            pixels = axis.transData.transform([(2.0, 0.0), (8.0, 0.0)])[:, 0]
            assert pixels == pytest.approx(reference)

    def test_a_long_structured_header_wraps_inside_the_figure_above_the_timeline(self) -> None:
        """A decision sentence must not be cut off at the page edge or overlap evidence lanes."""
        figure = plot_aligned_panels(
            _tone(10.0),
            [{"type": "waveform"}, {"type": "spectrogram"}],
            header={
                "context_label": "TASK / CONTEXT",
                "context": "task: narrative recall | declared hints: speech=claimed_and_found; voice=no_claim",
                "decision_label": "PRIMARY FILE DECISION",
                "decision": "TRIAGE: FLAG · RELEASE: WITHHELD",
                "evidence_label": "LEADING DECISION EVIDENCE",
                "evidence": (
                    "TAXONOMY: voice uncertain because the longest phonation span falls below the present "
                    "threshold but above the uncertainty threshold; SPEECH: a long supporting explanation follows."
                ),
                "support_label": "SCREENING / ROUTING",
                "support": (
                    "screened: airway=absent; speech=present; voice=uncertain\n"
                    "routing: AIRWAY skipped; SPEECH run; VOICE run"
                ),
            },
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        header_boxes = [
            text.get_window_extent(renderer)
            for text in figure.texts
            if text.get_text() and text.get_gid() != "senselab-lane-title"
        ]
        assert any("\n" in text.get_text() for text in figure.texts)
        assert all(box.x0 >= 0.0 and box.x1 <= figure.bbox.x1 for box in header_boxes)
        assert max(axis.get_window_extent(renderer).y1 for axis in figure.axes) < min(box.y0 for box in header_boxes)

    def test_report_lane_titles_and_eight_row_raster_labels_stay_in_separate_gutters(self) -> None:
        """A dense classifier lane must not paint either descriptive or row text into another panel."""
        labels = [
            "Breathe",
            "Cough",
            "Speech",
            "Sneeze",
            "Throat clear",
            "Wheeze",
            "Snore",
            "Silence",
        ]
        figure = plot_aligned_panels(
            _tone(10.0),
            [
                {"type": "waveform", "name": "recording amplitude and dBFS envelope"},
                {
                    "type": "score_raster",
                    "name": "HEAR candidate probabilities within evaluated airway spans",
                    "rows": labels,
                    "windows": [
                        {
                            "start": 0.0,
                            "end": 2.0,
                            "scores": {label: index / float(len(labels) - 1) for index, label in enumerate(labels)},
                        }
                    ],
                },
                {"type": "segments", "name": "long airway candidate review decisions", "segments": []},
            ],
        )
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        data_axes = [axis for axis in figure.axes if axis.axison]
        lane_titles = [text for text in figure.texts if text.get_gid() == "senselab-lane-title"]
        assert len(lane_titles) == len(data_axes)

        axis_boxes = [axis.get_window_extent(renderer) for axis in data_axes]
        lane_boxes = [text.get_window_extent(renderer) for text in lane_titles]
        assert all(lane_box.x1 < min(axis_box.x0 for axis_box in axis_boxes) for lane_box in lane_boxes)
        for lane_box, axis_box in zip(lane_boxes, axis_boxes):
            assert lane_box.y0 >= axis_box.y0
            assert lane_box.y1 <= axis_box.y1

        raster = next(axis for axis in data_axes if axis.get_ylabel().startswith("HEAR candidate"))
        raster_box = raster.get_window_extent(renderer)
        row_boxes = [tick.get_window_extent(renderer) for tick in raster.get_yticklabels()]
        assert len(row_boxes) == len(labels)
        assert all(box.x1 < raster_box.x0 and box.y0 >= raster_box.y0 and box.y1 <= raster_box.y1 for box in row_boxes)

        colorbar = raster.child_axes[0]
        colorbar_box = colorbar.get_window_extent(renderer)
        assert colorbar_box.x0 > raster_box.x1
        assert colorbar_box.x1 <= figure.bbox.x1

    def test_a_structured_header_is_the_only_figure_heading(self) -> None:
        """A PDF report must not clip a second, legacy title above its structured header."""
        figure = plot_aligned_panels(
            _tone(1.0),
            [{"type": "waveform"}],
            title="A legacy title that belongs in the structured report header",
            header={
                "context_label": "TASK / CONTEXT",
                "context": "task: sustained phonation",
                "decision_label": "PRIMARY FILE DECISION",
                "decision": "TRIAGE: REVIEW",
                "evidence_label": "LEADING DECISION EVIDENCE",
                "evidence": "voice review selected",
                "support_label": "SCREENING / ROUTING",
                "support": "voice=present; VOICE run",
            },
        )
        assert figure._suptitle is None

    def test_a_partial_header_degrades_instead_of_raising(self) -> None:
        """A caller supplying only some header fields must not crash the render with a KeyError."""
        figure = plot_aligned_panels(
            _tone(1.0),
            [{"type": "waveform"}],
            header={"decision_label": "PRIMARY FILE DECISION", "decision": "TRIAGE: FLAG"},
        )
        assert figure is not None

    def test_a_segments_panel_can_name_its_lane(self) -> None:
        """A figure stacking several segment lanes is unreadable if every one says "Segment"."""
        figure = plot_aligned_panels(
            _tone(),
            [
                {"type": "segments", "segments": [{"label": "Cough", "start": 0.1, "end": 0.2}], "name": "airway"},
                {"type": "segments", "segments": [{"label": "a", "start": 0.1, "end": 0.2}]},
            ],
        )
        assert figure.axes[0].get_ylabel() == "airway"
        assert figure.axes[1].get_ylabel() == "Segment"

    def test_a_features_panel_can_name_its_lane(self) -> None:
        """A stack of feature curves labelled "Value" says nothing about which curve is which."""
        figure = plot_aligned_panels(
            _tone(),
            [{"type": "features", "data": [([0.1], [1.0], "f0", "steelblue")], "name": "envelope"}],
        )
        assert figure.axes[0].get_ylabel() == "envelope"

    def test_a_waveform_panel_can_carry_a_twin_axis(self) -> None:
        """Amplitude and a dB curve are two scales over one signal; one row can hold both."""
        figure = plot_aligned_panels(
            _tone(),
            [
                {
                    "type": "waveform",
                    "twin": {
                        "name": "envelope dBFS",
                        "data": [([0.1, 0.2], [-40.0, -30.0], "envelope dBFS", "steelblue")],
                    },
                }
            ],
        )
        assert figure.axes[0].get_ylabel() == "Amplitude"
        assert figure.axes[1].get_ylabel() == "envelope dBFS"
        assert figure.axes[1].yaxis.get_label_position() == "right"

    def test_a_waveform_panel_can_carry_a_span_overlay(self) -> None:
        """A lane of bars over the signal it was measured from beats a lane of bars beneath it."""
        figure = plot_aligned_panels(
            _tone(),
            [
                {
                    "type": "waveform",
                    "spans": {
                        "name": "spans (dB over floor)",
                        "segments": [{"label": "18 dB", "start": 0.1, "end": 0.3}],
                    },
                }
            ],
        )
        assert len(figure.axes[0].patches) == 1
        assert any("18 dB" in text.get_text() for text in figure.axes[0].texts)

    def test_the_overlay_names_itself_on_the_right_hand_scale(self) -> None:
        """A translucent bar with no label is decoration; the scale must say what it is."""
        figure = plot_aligned_panels(
            _tone(),
            [
                {
                    "type": "waveform",
                    "twin": {"name": "envelope dBFS", "data": [([0.1], [-40.0], "envelope dBFS", "steelblue")]},
                    "spans": {
                        "name": "spans (dB over floor)",
                        "segments": [{"label": "18 dB", "start": 0.1, "end": 0.3}],
                    },
                }
            ],
        )
        label = figure.axes[1].get_ylabel()
        assert "envelope dBFS" in label and "spans (dB over floor)" in label

    def test_a_waveform_panel_with_no_twin_draws_one_axis(self) -> None:
        """The twin is opt-in; a bare waveform panel must not sprout an empty right-hand scale."""
        figure = plot_aligned_panels(_tone(), [{"type": "waveform"}])
        assert len(figure.axes) == 1

    def test_an_unknown_panel_type_now_raises(self) -> None:
        """A typo used to yield a blank axis and a report that looked finished."""
        with pytest.raises(ValueError, match="unknown panel type"):
            plot_aligned_panels(_tone(), [{"type": "sepctrogram"}])


class TestTheTokenLane:
    """A lane of timed tokens — words, phones — each carrying its own text on its own bar."""

    @staticmethod
    def _tokens(figure: Figure, index: int = 0) -> list[str]:
        """The texts a real draw put on one panel's axis, in draw order."""
        figure.canvas.draw()
        return [text.get_text() for text in figure.axes[index].texts if text.get_visible()]

    @staticmethod
    def _placed(figure: Figure, index: int = 0) -> list[Text]:
        """The token labels a real draw placed on one panel's axis."""
        figure.canvas.draw()
        return [text for text in figure.axes[index].texts if text.get_visible()]

    @staticmethod
    def _renderer(figure: Figure) -> RendererBase:
        """The renderer the figure's own canvas draws through."""
        return cast(RendererBase, figure.canvas.get_renderer())  # type: ignore[attr-defined]

    @classmethod
    def _extent(cls, figure: Figure, artist: Artist) -> Bbox:
        """One drawn artist's extent, in the display space bars and labels share."""
        return artist.get_window_extent(cls._renderer(figure))

    def test_a_token_carries_its_text_on_the_axis(self) -> None:
        """The text belongs on the bar; as a y-tick, 40 words collapse into one unreadable stack."""
        figure = plot_aligned_panels(
            _tone(),
            [
                {
                    "type": "tokens",
                    "name": "words",
                    "tokens": [
                        {"text": "hello", "start": 0.1, "end": 0.35},
                        {"text": "world", "start": 0.4, "end": 0.65},
                    ],
                }
            ],
        )
        assert self._tokens(figure) == ["hello", "world"]

    def test_every_token_is_drawn_as_a_bar_at_its_own_extent(self) -> None:
        """One bar per token, positioned in time, whether or not its text was placed."""
        figure = plot_aligned_panels(
            _tone(),
            [
                {
                    "type": "tokens",
                    "name": "words",
                    "tokens": [
                        {"text": f"w{index}", "start": 0.1 * index, "end": 0.1 * index + 0.08} for index in range(6)
                    ],
                }
            ],
        )
        bars = [cast(Rectangle, patch) for patch in figure.axes[0].patches]
        assert len(bars) == 6
        assert bars[0].get_x() == pytest.approx(0.0)
        assert bars[5].get_width() == pytest.approx(0.08)

    def test_the_lane_carries_no_tick_per_token(self) -> None:
        """The y-axis is not a legend: one tick per word is the defect this panel type replaces."""
        figure = plot_aligned_panels(
            _tone(),
            [
                {
                    "type": "tokens",
                    "name": "words",
                    "tokens": [
                        {"text": f"w{index}", "start": 0.02 * index, "end": 0.02 * index + 0.015} for index in range(40)
                    ],
                }
            ],
        )
        assert list(figure.axes[0].get_yticks()) == []

    def test_a_bar_too_narrow_for_its_text_keeps_the_bar(self) -> None:
        """A long word on a narrow bar loses its text, not its bar; the bar is the measurement."""
        figure = plot_aligned_panels(
            _tone(),
            [
                {
                    "type": "tokens",
                    "name": "words",
                    "tokens": [
                        {"text": "wide", "start": 0.1, "end": 0.4},
                        {"text": "grandfather", "start": 0.5, "end": 0.505},
                    ],
                }
            ],
            context=1.0,
        )
        assert len(figure.axes[0].patches) == 2
        assert self._tokens(figure) == ["wide"]

    def test_a_placed_label_never_outgrows_its_own_bar(self) -> None:
        """The rendered text is measured against the bar in display space, not assumed in seconds."""
        tokens = [
            {"text": "grandfather", "start": 0.05, "end": 0.09},
            {"text": "was", "start": 0.2, "end": 0.42},
            {"text": "village", "start": 0.5, "end": 0.56},
            {"text": "a", "start": 0.7, "end": 0.95},
        ]
        figure = plot_aligned_panels(_tone(), [{"type": "tokens", "name": "words", "tokens": tokens}], context=1.0)
        bars = {token["text"]: patch for token, patch in zip(tokens, figure.axes[0].patches)}
        assert len(self._placed(figure)) >= 1
        for label in self._placed(figure):
            bar = self._extent(figure, bars[label.get_text()])
            assert self._extent(figure, label).width <= bar.width

    def test_a_short_word_on_a_wide_bar_keeps_its_full_size(self) -> None:
        """Shrinking is what a narrow bar buys; a wide one must not pay for it."""
        figure = plot_aligned_panels(
            _tone(),
            [{"type": "tokens", "name": "words", "tokens": [{"text": "a", "start": 0.1, "end": 0.9}]}],
            context=1.0,
        )
        (label,) = self._placed(figure)
        assert label.get_fontsize() == pytest.approx(TOKEN_LABEL_FONTSIZE)

    def test_no_two_placed_labels_overlap(self) -> None:
        """Connected speech is where the pileup was: adjacent words ran into overlapping glyphs."""
        tokens = _connected_speech(duration=30.0)
        figure = plot_aligned_panels(_tone(30.0), [{"type": "tokens", "name": "words", "tokens": tokens}], context=1.0)
        assert len(figure.axes[0].patches) == len(tokens)
        extents = [self._extent(figure, label) for label in self._placed(figure)]
        overlapping = [
            (one, two) for index, one in enumerate(extents) for two in extents[index + 1 :] if one.overlaps(two)
        ]
        assert overlapping == []

    def test_the_fit_decision_follows_the_figure_width(self) -> None:
        """The gate is scale-free: the same lane at twice the width places strictly more labels."""
        tokens = _connected_speech(duration=30.0)
        panel = {"type": "tokens", "name": "words", "tokens": tokens}
        narrow = plot_aligned_panels(_tone(30.0), [panel], figsize=(8.0, 2.0), context=1.0)
        wide = plot_aligned_panels(_tone(30.0), [panel], figsize=(64.0, 2.0), context=1.0)
        assert len(self._placed(narrow)) < len(self._placed(wide))

    def test_a_label_is_shrunk_toward_the_floor_before_it_is_dropped(self) -> None:
        """The policy is shrink-then-drop: a bar that fits the text only smaller still carries it."""
        widths = [0.05 + 0.05 * step for step in range(29)]
        tokens = [
            {"text": "village", "start": 0.2 + index * 2.0, "end": 0.2 + index * 2.0 + width}
            for index, width in enumerate(widths)
        ]
        figure = plot_aligned_panels(
            _tone(60.0), [{"type": "tokens", "name": "words", "tokens": tokens}], figsize=(20.0, 2.0), context=1.0
        )
        sizes = {float(label.get_fontsize()) for label in self._placed(figure)}
        assert sizes, "a shrinkable bar must still carry its label"
        assert min(sizes) >= TOKEN_LABEL_FLOOR_FONTSIZE
        assert any(size < TOKEN_LABEL_FONTSIZE for size in sizes)

    def test_drawing_twice_does_not_shrink_twice(self) -> None:
        """A figure saved as a PNG and then into the PDF must not compound its own decision."""
        tokens = _connected_speech(duration=30.0)
        figure = plot_aligned_panels(
            _tone(30.0), [{"type": "tokens", "name": "words", "tokens": tokens}], figsize=(20.0, 2.0), context=1.0
        )
        first = [(label.get_text(), label.get_fontsize()) for label in self._placed(figure)]
        second = [(label.get_text(), label.get_fontsize()) for label in self._placed(figure)]
        assert first == second

    def test_a_png_and_a_pdf_of_one_figure_agree(self, tmp_path: Any) -> None:  # noqa: ANN401
        """Both output paths render through their own renderer; neither may re-decide the other's."""
        from matplotlib.backends.backend_pdf import PdfPages

        tokens = _connected_speech(duration=30.0)
        figure = plot_aligned_panels(
            _tone(30.0), [{"type": "tokens", "name": "words", "tokens": tokens}], figsize=(20.0, 2.0), context=1.0
        )
        figure.savefig(tmp_path / "page.png", bbox_inches="tight")
        after_png = [(label.get_text(), label.get_fontsize()) for label in self._placed(figure)]
        with PdfPages(tmp_path / "page.pdf") as pages:
            pages.savefig(figure, bbox_inches="tight")
        after_pdf = [(label.get_text(), label.get_fontsize()) for label in self._placed(figure)]
        assert after_png == after_pdf
        assert (tmp_path / "page.pdf").stat().st_size > 0


class TestTheTokenLabelFitPolicy:
    """The arithmetic behind shrink-then-drop, apart from any renderer."""

    @staticmethod
    def _measure(width_per_point: float) -> Any:  # noqa: ANN401
        """A text whose rendered width is proportional to its point size."""
        return lambda fontsize: width_per_point * fontsize

    def test_a_fitting_label_keeps_its_full_size(self) -> None:
        """Nothing is bought by shrinking a label that already fits."""
        assert _fitted_token_fontsize(self._measure(2.0), 20.0, 5.0, 4.0) == pytest.approx(5.0)

    def test_a_label_that_fits_only_smaller_is_shrunk(self) -> None:
        """The size returned is the largest at which the measured width still fits."""
        fitted = _fitted_token_fontsize(self._measure(2.0), 9.0, 5.0, 4.0)
        assert fitted is not None
        assert fitted == pytest.approx(4.5)

    def test_a_label_that_does_not_fit_at_the_floor_is_dropped(self) -> None:
        """Below the floor the text is no longer legible, so it is not drawn at all."""
        assert _fitted_token_fontsize(self._measure(2.0), 4.0, 5.0, 4.0) is None

    def test_a_bar_with_no_room_at_all_is_dropped(self) -> None:
        """A bar narrower than the padding carries no text, whatever the text is."""
        assert _fitted_token_fontsize(self._measure(2.0), 0.0, 5.0, 4.0) is None
        assert _fitted_token_fontsize(self._measure(2.0), -3.0, 5.0, 4.0) is None

    def test_the_text_is_clipped_to_the_axis(self) -> None:
        """A token at the edge of the window must not draw outside the panel it belongs to."""
        figure = plot_aligned_panels(
            _tone(),
            [{"type": "tokens", "name": "words", "tokens": [{"text": "edge", "start": 0.8, "end": 0.99}]}],
        )
        assert figure.axes[0].texts[0].get_clip_on()

    def test_the_lane_names_itself(self) -> None:
        """A stack of lanes is unreadable if none of them says which reading it is."""
        figure = plot_aligned_panels(
            _tone(), [{"type": "tokens", "name": "words", "tokens": [{"text": "a", "start": 0.1, "end": 0.4}]}]
        )
        assert figure.axes[0].get_ylabel() == "words"

    def test_declared_rows_become_the_only_y_ticks(self) -> None:
        """Several stripes over one axis are named by stripe, still never by token."""
        figure = plot_aligned_panels(
            _tone(),
            [
                {
                    "type": "tokens",
                    "name": "words",
                    "tokens": [
                        {"text": "one", "start": 0.1, "end": 0.4, "row": "consensus"},
                        {"text": "two", "start": 0.1, "end": 0.4, "row": "whisper"},
                    ],
                }
            ],
        )
        assert [tick.get_text() for tick in figure.axes[0].get_yticklabels()] == ["consensus", "whisper"]

    def test_a_lane_with_no_token_still_draws(self) -> None:
        """An empty lane is an empty stripe, not a raise and not a blank unnamed axis."""
        figure = plot_aligned_panels(_tone(), [{"type": "tokens", "name": "words", "tokens": []}])
        assert figure.axes[0].get_ylabel() == "words"
        assert len(figure.axes[0].patches) == 0


class TestTheStaggeredTokenRows:
    """A lane whose labels cannot lie side by side in one row spreads them over rows it derives."""

    @staticmethod
    def _draw(duration: float, size: tuple[float, float]) -> Figure:
        """A connected-speech lane of the given length, drawn at the given figure size."""
        tokens = _connected_speech(duration=duration)
        figure = plot_aligned_panels(
            _tone(duration),
            [{"type": "tokens", "name": "words", "tokens": tokens}],
            figsize=size,
            context=1.0,
        )
        figure.canvas.draw()
        return figure

    @staticmethod
    def _bar_tops(figure: Figure, index: int = 0) -> list[float]:
        """Each bar's lower edge, in the order the tokens were given."""
        return [round(float(cast(Rectangle, patch).get_y()), 6) for patch in figure.axes[index].patches]

    @classmethod
    def _row_tops(cls, figure: Figure, index: int = 0) -> list[float]:
        """The distinct bar rows a real draw laid out, from the top of the lane downwards."""
        return sorted(set(cls._bar_tops(figure, index)), reverse=True)

    @classmethod
    def _rows(cls, figure: Figure, index: int = 0) -> list[int]:
        """Each bar's row, counted from the top of the lane, in the order the tokens were given."""
        tops = cls._row_tops(figure, index)
        return [tops.index(top) for top in cls._bar_tops(figure, index)]

    @staticmethod
    def _drawn(figure: Figure, index: int = 0) -> list[Text]:
        """The token labels a real draw placed on one panel's axis."""
        return [text for text in figure.axes[index].texts if text.get_visible()]

    @classmethod
    def _overlaps(cls, figure: Figure, index: int = 0) -> list[tuple[str, str]]:
        """Every pair of drawn labels whose display-space boxes intersect."""
        renderer = cast(RendererBase, figure.canvas.get_renderer())  # type: ignore[attr-defined]
        drawn = cls._drawn(figure, index)
        boxes = [(text.get_text(), text.get_window_extent(renderer)) for text in drawn]
        return [
            (one[0], two[0])
            for position, one in enumerate(boxes)
            for two in boxes[position + 1 :]
            if one[1].overlaps(two[1])
        ]

    def test_a_campaign_length_lane_is_no_longer_nearly_wordless(self) -> None:
        """The defect: 30 s of speech on a report-width page drew 21 of its 88 words and no more."""
        tokens = _connected_speech(duration=30.0)
        figure = self._draw(30.0, (14.0, 4.0))
        assert len(figure.axes[0].patches) == len(tokens)
        assert len(self._drawn(figure)) >= 66

    def test_a_sixty_second_lane_carries_words_at_all(self) -> None:
        """At one row a 60 s page drew nothing; the reviewer met a row of bare bars."""
        tokens = _connected_speech(duration=60.0)
        figure = self._draw(60.0, (14.0, 4.0))
        assert len(figure.axes[0].patches) == len(tokens)
        assert len(self._drawn(figure)) >= 126

    def test_no_two_drawn_labels_overlap_at_campaign_length(self) -> None:
        """The load-bearing invariant: more rows may never buy a single touching pair of glyphs."""
        assert self._overlaps(self._draw(30.0, (14.0, 4.0))) == []

    def test_no_two_drawn_labels_overlap_on_the_longest_page(self) -> None:
        """Sixty seconds is the top of the campaign range and the densest lane the panel meets."""
        assert self._overlaps(self._draw(60.0, (14.0, 4.0))) == []

    def test_no_two_drawn_labels_overlap_on_a_narrow_page(self) -> None:
        """A page narrow enough to exhaust the row ceiling must still not overlap; it drops."""
        assert self._overlaps(self._draw(60.0, (6.0, 1.6))) == []

    def test_a_wider_page_uses_fewer_rows(self) -> None:
        """The row count is derived from the width the labels are given, so it must follow it."""
        narrow = self._draw(30.0, (8.0, 2.0))
        wide = self._draw(30.0, (64.0, 2.0))
        assert len(self._row_tops(wide)) == 1
        assert len(self._row_tops(narrow)) > len(self._row_tops(wide))

    def test_a_lane_whose_labels_already_fit_side_by_side_stays_on_one_row(self) -> None:
        """A row is spent only when the labels cannot lie side by side; twelve seconds is not that."""
        assert len(self._row_tops(self._draw(12.0, (14.0, 4.0)))) == 1

    def test_a_sparse_lane_stays_on_one_row(self) -> None:
        """Rows relieve crowding. Twenty-nine words over sixty seconds are not crowded."""
        tokens = [{"text": "village", "start": 0.2 + index * 2.0, "end": 0.7 + index * 2.0} for index in range(29)]
        figure = plot_aligned_panels(
            _tone(60.0), [{"type": "tokens", "name": "words", "tokens": tokens}], figsize=(20.0, 2.0), context=1.0
        )
        figure.canvas.draw()
        assert len(self._row_tops(figure)) == 1

    def test_the_reading_order_cycles_by_index(self) -> None:
        """Token i is in row i mod R, so the sequence is a staircase that never reorders time."""
        figure = self._draw(30.0, (14.0, 4.0))
        rows = self._rows(figure)
        row_count = len(self._row_tops(figure))
        assert row_count > 1
        assert rows == [index % row_count for index in range(len(rows))]

    def test_a_row_never_shrinks_below_what_its_font_needs(self) -> None:
        """The ceiling is typographic: a row too short to hold its own label is not a row."""
        figure = self._draw(60.0, (6.0, 1.0))
        renderer = cast(RendererBase, figure.canvas.get_renderer())  # type: ignore[attr-defined]
        lane = figure.axes[0].get_window_extent(renderer).height * 72.0 / figure.dpi
        assert lane / len(self._row_tops(figure)) >= TOKEN_ROW_PITCH_EM * TOKEN_LABEL_FONTSIZE

    def test_a_label_may_use_the_width_its_row_neighbours_leave(self) -> None:
        """This is the whole mechanism: a staggered label is measured against its row's slot."""
        figure = self._draw(30.0, (14.0, 4.0))
        renderer = cast(RendererBase, figure.canvas.get_renderer())  # type: ignore[attr-defined]
        axis = figure.axes[0]
        pairs = [(text, bar) for text, bar in zip(axis.texts, axis.patches) if text.get_visible()]
        assert pairs
        widest = max(
            text.get_window_extent(renderer).width / bar.get_window_extent(renderer).width for text, bar in pairs
        )
        assert widest > 1.0

    def test_a_declared_row_wins_and_is_not_restaggered(self) -> None:
        """The per-model stripe contract: a token that names its row is placed in it, and stays."""
        tokens = [
            {
                "text": f"w{index}",
                "start": 0.4 * index,
                "end": 0.4 * index + 0.3,
                "row": "whisper" if index % 2 else "canary",
            }
            for index in range(24)
        ]
        figure = plot_aligned_panels(
            _tone(12.0), [{"type": "tokens", "name": "words", "tokens": tokens}], figsize=(6.0, 2.0), context=1.0
        )
        figure.canvas.draw()
        rows = self._rows(figure)
        assert len(set(rows)) == 2
        assert len(set(rows[0::2])) == 1
        assert len(set(rows[1::2])) == 1
        assert [tick.get_text() for tick in figure.axes[0].get_yticklabels()] == ["canary", "whisper"]

    def test_only_the_tokens_that_declare_no_row_are_staggered(self) -> None:
        """Precedence is explicit: the named stripe is one row however dense the unnamed block is."""
        declared = [
            {"text": f"d{index}", "start": 0.35 * index, "end": 0.35 * index + 0.3, "row": "canary"}
            for index in range(80)
        ]
        free = [{"text": word["text"], "start": word["start"], "end": word["end"]} for word in _connected_speech(30.0)]
        figure = plot_aligned_panels(
            _tone(30.0),
            [{"type": "tokens", "name": "words", "tokens": declared + free}],
            figsize=(14.0, 4.0),
            context=1.0,
        )
        figure.canvas.draw()
        rows = self._rows(figure)
        assert len(set(rows[: len(declared)])) == 1
        assert len(set(rows[len(declared) :])) > 1

    def test_the_staggered_rows_carry_no_tick_of_their_own(self) -> None:
        """Rows must not read as named lanes; only a declared row is ever named on the axis."""
        assert list(self._draw(30.0, (14.0, 4.0)).axes[0].get_yticks()) == []

    def test_the_staggered_rows_share_one_colour(self) -> None:
        """A row is a wrap of one reading, so colouring rows apart would imply a grouping there is not."""
        figure = self._draw(30.0, (14.0, 4.0))
        assert len({tuple(patch.get_facecolor()) for patch in figure.axes[0].patches}) == 1


class TestTheStaggeredRowArithmetic:
    """How many rows a lane takes, and how wide a label may grow in one, apart from any renderer."""

    def test_labels_that_fit_side_by_side_take_one_row(self) -> None:
        """The demand is the labels' own widths; a lane under its supply needs no second row."""
        assert _staggered_row_count([10.0] * 8, 200.0, 1.0, 9) == 1

    def test_a_lane_over_its_width_takes_the_rows_its_labels_demand(self) -> None:
        """Eleven points of ink each over a hundred points of page is two rows, not a guess."""
        assert _staggered_row_count([10.0] * 20, 100.0, 1.0, 9) == 3

    def test_the_row_count_never_passes_its_ceiling(self) -> None:
        """A row shorter than its own font is illegible, so the ceiling wins over the demand."""
        assert _staggered_row_count([10.0] * 200, 100.0, 1.0, 4) == 4

    def test_an_empty_lane_takes_one_row(self) -> None:
        """No label demands anything; the lane is still one row of bars."""
        assert _staggered_row_count([], 100.0, 1.0, 9) == 1

    def test_a_lane_with_no_width_takes_one_row(self) -> None:
        """An axis of no width cannot be divided into rows that help."""
        assert _staggered_row_count([10.0] * 20, 0.0, 1.0, 9) == 1

    def test_the_ceiling_is_the_lane_over_the_row_pitch(self) -> None:
        """The pitch is stated in multiples of the label's point size, which is dpi-free."""
        assert _staggered_row_ceiling(92.0, 5.0) == 9
        assert _staggered_row_ceiling(9.0, 5.0) == 1

    def test_one_row_measures_a_label_against_its_own_bar(self) -> None:
        """At one row the slot is the bar, which is what the panel did before it staggered."""
        slots = _token_label_slots([1.0, 2.0, 3.0], [0.4, 0.4, 0.4], [0, 0, 0], 1, (0.0, 4.0))
        assert slots == pytest.approx([(0.6, 1.4), (1.6, 2.4), (2.6, 3.4)])

    def test_a_staggered_label_may_reach_across_the_rows_it_is_not_in(self) -> None:
        """Two rows give a label twice its bar, bounded by the neighbour that shares its row."""
        slots = _token_label_slots([1.0, 2.0, 3.0, 4.0], [0.4] * 4, [0, 1, 0, 1], 2, (0.0, 5.0))
        assert slots[0] == pytest.approx((0.2, 1.8))
        assert slots[2] == pytest.approx((2.2, 3.8))

    def test_expanded_slots_use_the_full_gap_in_their_own_row(self) -> None:
        """A report can show short transcript words without moving their timing bars."""
        slots = _token_label_slots(
            [1.0, 2.0, 3.0, 4.0],
            [0.4] * 4,
            [0, 1, 0, 1],
            2,
            (0.0, 5.0),
            expand_to_row_neighbours=True,
        )
        assert slots[0] == pytest.approx((0.0, 2.0))
        assert slots[2] == pytest.approx((2.0, 4.0))

    def test_a_slot_stops_at_the_midpoint_to_its_row_neighbour(self) -> None:
        """The bound that makes the pairwise overlap impossible rather than merely unobserved."""
        slots = _token_label_slots([1.0, 1.4, 2.0], [1.0, 1.0, 1.0], [0, 1, 0], 2, (0.0, 3.0))
        assert slots[0][1] == pytest.approx(1.5)
        assert slots[2][0] == pytest.approx(1.5)

    def test_a_slot_stops_at_the_edge_of_the_axis(self) -> None:
        """A row's first and last label reach to the window, never past it."""
        slots = _token_label_slots([0.2, 2.8], [2.0, 2.0], [0, 0], 2, (0.0, 3.0))
        assert slots[0] == pytest.approx((0.0, 0.4))
        assert slots[1] == pytest.approx((2.6, 3.0))


class TestPlotWaveform:
    """Tests for the plot_waveform function."""

    def test_plot_waveform_mono_audio(self, mono_audio_sample: Audio) -> None:
        """Test plotting waveform with mono audio."""
        figure = plot_waveform(mono_audio_sample, title="Test Mono Waveform")

        assert isinstance(figure, Figure)
        # Access the stored suptitle (private attribute but safe for testing)
        suptitle: Text = getattr(figure, "_suptitle")
        assert suptitle.get_text() == "Test Mono Waveform"

        # Check that we have the expected number of subplots (1 for mono)
        assert len(figure.axes) == 1

    def test_plot_waveform_stereo_audio(self, stereo_audio_sample: Audio) -> None:
        """Test plotting waveform with stereo audio."""
        figure = plot_waveform(stereo_audio_sample, title="Test Stereo Waveform")

        assert isinstance(figure, Figure)
        suptitle: Text = getattr(figure, "_suptitle")
        assert suptitle.get_text() == "Test Stereo Waveform"

        # Check that we have the expected number of subplots (2 for stereo)
        assert len(figure.axes) == 2

    def test_plot_waveform_with_fast_option(self, mono_audio_sample: Audio) -> None:
        """Test plotting waveform with fast option enabled."""
        figure = plot_waveform(mono_audio_sample, title="Fast Plot", fast=True)

        assert isinstance(figure, Figure)
        suptitle: Text = getattr(figure, "_suptitle")
        assert suptitle.get_text() == "Fast Plot"

    def test_plot_waveform_default_title(self, mono_audio_sample: Audio) -> None:
        """Test plotting waveform with default title."""
        figure = plot_waveform(mono_audio_sample)

        assert isinstance(figure, Figure)
        suptitle: Text = getattr(figure, "_suptitle")
        assert suptitle.get_text() == "Waveform"

    def test_plot_waveform_multi_channel_audio(self) -> None:
        """Test waveform with multi-channel audio (more than 2 channels)."""
        # Create a 4-channel audio
        waveform = torch.randn(4, 16000)
        audio = Audio(waveform=waveform, sampling_rate=16000)

        figure = plot_waveform(audio, title="Multi-channel Test")

        assert isinstance(figure, Figure)
        assert len(figure.axes) == 4  # Should have 4 subplots for 4 channels

    def test_plot_waveform_empty_audio(self) -> None:
        """Test plotting waveform with empty audio."""
        # Create empty audio (0 frames)
        waveform = torch.empty(1, 0)
        audio = Audio(waveform=waveform, sampling_rate=16000)

        figure = plot_waveform(audio)
        assert isinstance(figure, Figure)

    @patch("matplotlib.pyplot.show")
    def test_plot_waveform_show_called(self, mock_show: MagicMock, mono_audio_sample: Audio) -> None:
        """Test that plt.show is called with block=False."""
        plot_waveform(mono_audio_sample)
        mock_show.assert_called_once_with(block=False)


class TestPlotSpecgram:
    """Tests for the plot_specgram function."""

    def test_plot_specgram_regular(self, mono_audio_sample: Audio) -> None:
        """Test plotting regular spectrogram."""
        figure = plot_specgram(mono_audio_sample, mel_scale=False, title="Test Spectrogram")

        assert isinstance(figure, Figure)

    def test_plot_specgram_mel_scale(self, mono_audio_sample: Audio) -> None:
        """Test plotting mel spectrogram."""
        figure = plot_specgram(mono_audio_sample, mel_scale=True, title="Test Mel Spectrogram")

        assert isinstance(figure, Figure)

    def test_plot_specgram_default_title(self, mono_audio_sample: Audio) -> None:
        """Test plotting spectrogram with default title."""
        figure = plot_specgram(mono_audio_sample)

        assert isinstance(figure, Figure)

    def test_plot_specgram_with_kwargs(self, mono_audio_sample: Audio) -> None:
        """Test plotting spectrogram with additional keyword arguments."""
        # Test with some common spectrogram parameters
        figure = plot_specgram(
            mono_audio_sample,
            mel_scale=False,
            title="Test with kwargs",
            n_fft=512,
            hop_length=256,
        )

        assert isinstance(figure, Figure)

    def test_plot_specgram_stereo_audio(self, stereo_audio_sample: Audio) -> None:
        """Stereo should error (we require mono)."""
        with pytest.raises(ValueError, match="Spectrogram must be a 2D tensor."):
            plot_specgram(stereo_audio_sample, title="Stereo Spectrogram")

    @patch("matplotlib.pyplot.show")
    def test_plot_specgram_show_called(self, mock_show: MagicMock, mono_audio_sample: Audio) -> None:
        """Test that plt.show is called with block=False."""
        plot_specgram(mono_audio_sample)
        mock_show.assert_called_once_with(block=False)

    def test_plot_specgram_short_audio(self) -> None:
        """Test plotting spectrogram with very short audio."""
        # Create very short audio (100 samples)
        waveform = torch.randn(1, 100)
        audio = Audio(waveform=waveform, sampling_rate=16000)

        # Very short audio should raise an appropriate error
        with pytest.raises(ValueError, match="Spectrogram extraction failed"):
            plot_specgram(audio)


class TestPlayAudio:
    """Tests for the play_audio function."""

    @patch("IPython.display.display")
    @patch("IPython.display.Audio")
    def test_play_audio_mono(
        self,
        mock_audio: MagicMock,
        mock_display: MagicMock,
        mono_audio_sample: Audio,
    ) -> None:
        """Test playing mono audio."""
        play_audio(mono_audio_sample)

        # Check that Audio was called with the correct parameters
        mock_audio.assert_called_once()
        call_args = mock_audio.call_args
        assert call_args[1]["rate"] == mono_audio_sample.sampling_rate

        # Check that display was called
        mock_display.assert_called_once()

    @patch("IPython.display.display")
    @patch("IPython.display.Audio")
    def test_play_audio_stereo(
        self,
        mock_audio: MagicMock,
        mock_display: MagicMock,
        stereo_audio_sample: Audio,
    ) -> None:
        """Test playing stereo audio."""
        play_audio(stereo_audio_sample)

        # Check that Audio was called with the correct parameters
        mock_audio.assert_called_once()
        call_args = mock_audio.call_args
        assert call_args[1]["rate"] == stereo_audio_sample.sampling_rate

        # For stereo, the first argument should be a tuple of two channels
        audio_data = call_args[0][0]
        assert isinstance(audio_data, tuple)
        assert len(audio_data) == 2

        # Check that display was called
        mock_display.assert_called_once()

    def test_play_audio_more_than_two_channels(self) -> None:
        """Test that playing n-channel audio for n > 2 raises ValueError.

        This test ensures that audio with more than 2 channels cannot be played
        using the play_audio function, as it only supports mono and stereo audio.
        """
        # Create 3-channel audio
        waveform = torch.randn(3, 16000)
        audio = Audio(waveform=waveform, sampling_rate=16000)

        expected_msg = "Waveform with more than 2 channels is not supported"
        with pytest.raises(ValueError, match=expected_msg):
            play_audio(audio)

    def test_play_audio_four_channels(self) -> None:
        """Test that playing audio with 4 channels raises ValueError."""
        # Create 4-channel audio
        waveform = torch.randn(4, 16000)
        audio = Audio(waveform=waveform, sampling_rate=16000)

        expected_msg = "Waveform with more than 2 channels is not supported"
        with pytest.raises(ValueError, match=expected_msg):
            play_audio(audio)

    @patch("IPython.display.display")
    @patch("IPython.display.Audio")
    def test_play_audio_empty_mono(self, mock_audio: MagicMock, mock_display: MagicMock) -> None:
        """Test playing empty mono audio."""
        waveform = torch.empty(1, 0)
        audio = Audio(waveform=waveform, sampling_rate=16000)

        play_audio(audio)

        mock_audio.assert_called_once()
        mock_display.assert_called_once()

    @patch("IPython.display.display")
    @patch("IPython.display.Audio")
    def test_play_audio_empty_stereo(self, mock_audio: MagicMock, mock_display: MagicMock) -> None:
        """Test playing empty stereo audio."""
        waveform = torch.empty(2, 0)
        audio = Audio(waveform=waveform, sampling_rate=16000)

        play_audio(audio)

        mock_audio.assert_called_once()
        mock_display.assert_called_once()


class TestPlottingIntegration:
    """Integration tests for plotting functions."""

    def test_all_functions_with_same_audio(self, mono_audio_sample: Audio) -> None:
        """Test that all plotting functions work with the same audio input."""
        # Test waveform plotting
        waveform_fig = plot_waveform(mono_audio_sample, title="Integration Test Waveform")
        assert isinstance(waveform_fig, Figure)

        # Test spectrogram plotting
        specgram_fig = plot_specgram(mono_audio_sample, title="Integration Test Spectrogram")
        assert isinstance(specgram_fig, Figure)

        # Test audio playing (mocked to avoid actual playback)
        with patch("IPython.display.display"), patch("IPython.display.Audio"):
            play_audio(mono_audio_sample)

    @patch("matplotlib.pyplot.show")
    def test_plotting_functions_dont_block(self, mock_show: MagicMock, mono_audio_sample: Audio) -> None:
        """Test that plotting functions don't block execution."""
        plot_waveform(mono_audio_sample)
        plot_specgram(mono_audio_sample)

        # Verify show was called twice with block=False
        assert mock_show.call_count == 2
        for call in mock_show.call_args_list:
            assert call[1]["block"] is False

    def test_plotting_with_different_sampling_rates(self) -> None:
        """Test plotting functions with different sampling rates."""
        sampling_rates = [8000, 16000, 22050, 44100, 48000]

        for sr in sampling_rates:
            waveform = torch.randn(1, sr)  # 1 second of audio
            audio = Audio(waveform=waveform, sampling_rate=sr)

            # Test both plotting functions
            waveform_fig = plot_waveform(audio, title=f"SR: {sr}")
            specgram_fig = plot_specgram(audio, title=f"Specgram SR: {sr}")

            assert isinstance(waveform_fig, Figure)
            assert isinstance(specgram_fig, Figure)

    def test_plotting_with_various_durations(self) -> None:
        """Test plotting functions with audio of various durations."""
        durations = [0.1, 0.5, 1.0, 2.0, 5.0]  # seconds
        sampling_rate = 16000

        for duration in durations:
            num_samples = int(duration * sampling_rate)
            waveform = torch.randn(1, num_samples)
            audio = Audio(waveform=waveform, sampling_rate=sampling_rate)

            waveform_fig = plot_waveform(audio, title=f"Duration: {duration}s")
            specgram_fig = plot_specgram(audio, title=f"Specgram Duration: {duration}s")

            assert isinstance(waveform_fig, Figure)
            assert isinstance(specgram_fig, Figure)
