"""This script contains unit tests for the plotting tasks."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torchaudio
from matplotlib.pyplot import Figure

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.plotting.plotting import (
    play_audio,
    plot_specgram,
    plot_waveform,
)

try:
    import torchaudio  # noqa: F401

    TORCHAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    TORCHAUDIO_AVAILABLE = False


@pytest.mark.skipif(
    not TORCHAUDIO_AVAILABLE,
    reason="Torchaudio is not available.",
)
class TestPlotWaveform:
    """Tests for the plot_waveform function."""

    def test_plot_waveform_mono_audio(self, mono_audio_sample: Audio) -> None:
        """Test plotting waveform with mono audio."""
        figure = plot_waveform(mono_audio_sample, title="Test Mono Waveform")

        assert isinstance(figure, Figure)
        assert figure._suptitle.get_text() == "Test Mono Waveform"

        # Check that we have the expected number of subplots (1 for mono)
        assert len(figure.axes) == 1

    def test_plot_waveform_stereo_audio(self, stereo_audio_sample: Audio) -> None:
        """Test plotting waveform with stereo audio."""
        figure = plot_waveform(stereo_audio_sample, title="Test Stereo Waveform")

        assert isinstance(figure, Figure)
        assert figure._suptitle.get_text() == "Test Stereo Waveform"

        # Check that we have the expected number of subplots (2 for stereo)
        assert len(figure.axes) == 2

    def test_plot_waveform_with_fast_option(self, mono_audio_sample: Audio) -> None:
        """Test plotting waveform with fast option enabled."""
        figure = plot_waveform(mono_audio_sample, title="Fast Plot", fast=True)

        assert isinstance(figure, Figure)
        assert figure._suptitle.get_text() == "Fast Plot"

    def test_plot_waveform_default_title(self, mono_audio_sample: Audio) -> None:
        """Test plotting waveform with default title."""
        figure = plot_waveform(mono_audio_sample)

        assert isinstance(figure, Figure)
        assert figure._suptitle.get_text() == "Waveform"

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


@pytest.mark.skipif(
    not TORCHAUDIO_AVAILABLE,
    reason="Torchaudio is not available.",
)
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
        figure = plot_specgram(mono_audio_sample, mel_scale=False, title="Test with kwargs", n_fft=512, hop_length=256)

        assert isinstance(figure, Figure)

    def test_plot_specgram_stereo_audio(self, stereo_audio_sample: Audio) -> None:
        """Test spectrogram with stereo audio (should use first channel)."""
        figure = plot_specgram(stereo_audio_sample, title="Stereo Spectrogram")

        assert isinstance(figure, Figure)

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


@pytest.mark.skipif(
    not TORCHAUDIO_AVAILABLE,
    reason="Torchaudio is not available.",
)
class TestPlayAudio:
    """Tests for the play_audio function."""

    @patch("IPython.display.display")
    @patch("IPython.display.Audio")
    def test_play_audio_mono(self, mock_audio: MagicMock, mock_display: MagicMock, mono_audio_sample: Audio) -> None:
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
        self, mock_audio: MagicMock, mock_display: MagicMock, stereo_audio_sample: Audio
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
        """Test that playing audio with more than 2 channels raises ValueError."""
        # Create 3-channel audio
        waveform = torch.randn(3, 16000)
        audio = Audio(waveform=waveform, sampling_rate=16000)

        expected_msg = "Waveform with more than 2 channels are not supported"
        with pytest.raises(ValueError, match=expected_msg):
            play_audio(audio)

    def test_play_audio_four_channels(self) -> None:
        """Test that playing audio with 4 channels raises ValueError."""
        # Create 4-channel audio
        waveform = torch.randn(4, 16000)
        audio = Audio(waveform=waveform, sampling_rate=16000)

        expected_msg = "Waveform with more than 2 channels are not supported"
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


@pytest.mark.skipif(
    not TORCHAUDIO_AVAILABLE,
    reason="Torchaudio is not available.",
)
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
