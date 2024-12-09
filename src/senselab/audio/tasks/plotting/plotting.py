"""This module contains functions for plotting audio-related data."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Audio, display

from senselab.audio.data_structures import Audio as AudioData
from senselab.audio.tasks.features_extraction.torchaudio import (
    extract_mel_spectrogram_from_audios,
    extract_spectrogram_from_audios,
)
from senselab.utils.data_structures import logger


def plot_waveform(audio: AudioData, title: str = "Waveform", fast: bool = False) -> None:
    """Plots the waveform of an Audio object.

    Args:
        audio (AudioData): An instance of Audio containing waveform data and sampling rate.
        title (str): Title of the plot.
        fast (bool): If True, plots a downsampled version for a faster but less detailed view.

    Todo:
        - Add option to save the plot
        - Add option to choose the size of the Figure
    """
    waveform = audio.waveform
    sample_rate = audio.sampling_rate

    if fast:
        # Downsampling the waveform and sample rate for a quicker plot
        waveform = waveform[::10]

    num_channels, num_frames = waveform.shape
    time_axis = torch.linspace(0, num_frames / sample_rate, num_frames)

    figure, axes = plt.subplots(num_channels, 1, figsize=(12, num_channels * 2))
    if num_channels == 1:
        axes = [axes]  # Ensure axes is iterable
    for c, ax in enumerate(axes):
        ax.plot(time_axis, waveform[c].numpy(), linewidth=1)
        ax.set_ylabel(f"Channel {c + 1}")
        ax.grid(True)

    figure.suptitle(title)
    plt.xlabel("Time [s]")
    plt.show(block=False)


def plot_specgram(audio: AudioData, mel_scale: bool = False, title: str = "Spectrogram", **spect_kwargs: Any) -> None:  # noqa : ANN401
    """Plots the spectrogram of an Audio object.

    Args:
        audio: (AudioData): An instance of Audio containing waveform data and sampling rate.
        mel_scale (bool): Whether to plot a mel spectrogram or a regular spectrogram.
        title (str): Title of the spectrogram plot.
        **spect_kwargs: Additional keyword arguments to pass to the spectrogram function.

    Todo:
        - Add option to save the plot
        - Add option to choose the size of the Figure
    """

    def _power_to_db(
        spectrogram: np.ndarray, ref: float = 1.0, amin: float = 1e-10, top_db: float = 80.0
    ) -> np.ndarray:
        """Converts a power spectrogram (amplitude squared) to decibel (dB) units.

        Args:
            spectrogram (np.ndarray): Power spectrogram to convert.
            ref (float): Reference power. Default is 1.0.
            amin (float): Minimum power. Default is 1e-10.
            top_db (float): Minimum decibel. Default is 80.0.

        Returns:
            np.ndarray: Decibel spectrogram.
        """
        S = np.asarray(spectrogram)

        if amin <= 0:
            raise ValueError("amin must be strictly positive")

        if np.issubdtype(S.dtype, np.complexfloating):
            logger.warning(
                "_power_to_db was called on complex input so phase "
                "information will be discarded. To suppress this warning, "
                "call power_to_db(np.abs(D)**2) instead.",
                stacklevel=2,
            )
            magnitude = np.abs(S)
        else:
            magnitude = S

        if callable(ref):
            # User supplied a function to calculate reference power
            ref_value = ref(magnitude)
        else:
            ref_value = np.abs(ref)

        log_spec: np.ndarray = 10.0 * np.log10(np.maximum(amin, magnitude))
        log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

        if top_db is not None:
            if top_db < 0:
                raise ValueError("top_db must be non-negative")
            log_spec = np.maximum(log_spec, log_spec.max() - top_db)

        return log_spec

    # Extract the spectrogram
    if mel_scale:
        spectrogram = extract_mel_spectrogram_from_audios([audio], **spect_kwargs)[0]["mel_spectrogram"]
        y_axis_label = "Mel Frequency"
    else:
        spectrogram = extract_spectrogram_from_audios([audio], **spect_kwargs)[0]["spectrogram"]
        y_axis_label = "Frequency [Hz]"

    if spectrogram.dim() != 2:
        raise ValueError("Spectrogram must be a 2D tensor.")

    # Determine time and frequency scale
    num_frames = spectrogram.size(1)
    num_freq_bins = spectrogram.size(0)

    # Time axis in seconds
    time_axis = (audio.waveform.size(-1) / audio.sampling_rate) * (torch.arange(0, num_frames).float() / num_frames)

    # Frequency axis in Hz (for non-mel spectrograms)
    if mel_scale:
        freq_axis = torch.arange(num_freq_bins)  # For mel spectrogram, keep the bins as discrete values
    else:
        freq_axis = torch.linspace(0, audio.sampling_rate / 2, num_freq_bins)

    plt.figure(figsize=(10, 4))
    plt.imshow(
        _power_to_db(spectrogram.numpy()),
        aspect="auto",
        origin="lower",
        extent=(float(time_axis[0]), float(time_axis[-1]), float(freq_axis[0]), float(freq_axis[-1])),
        cmap="viridis",
    )
    plt.colorbar(label="Magnitude (dB)")
    plt.title(title)
    plt.ylabel(y_axis_label)
    plt.xlabel("Time [Sec]")
    plt.show(block=False)


def play_audio(audio: AudioData) -> None:
    """Plays an audio file.

    Args:
        audio (AudioData): An instance of Audio containing waveform data and sampling rate.

    Raises:
        ValueError: If the number of channels is more than 2.
    """
    waveform = audio.waveform.numpy()
    sample_rate = audio.sampling_rate

    num_channels = waveform.shape[0]
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")
