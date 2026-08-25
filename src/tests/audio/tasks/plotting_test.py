"""This script contains unit tests for the plotting tasks."""

from unittest.mock import MagicMock, patch

import pytest
import torch
from matplotlib.pyplot import Figure
from matplotlib.text import Text

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.plotting.plotting import (
    play_audio,
    plot_aligned_panels,
    plot_specgram,
    plot_waveform,
)


def _tone() -> Audio:
    """A short mono waveform, enough for a shared time axis."""
    return Audio(waveform=torch.linspace(-0.5, 0.5, 16000).unsqueeze(0), sampling_rate=16000)


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
