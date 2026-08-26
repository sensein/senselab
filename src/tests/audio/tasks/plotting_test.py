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
    _fitted_token_fontsize,
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
