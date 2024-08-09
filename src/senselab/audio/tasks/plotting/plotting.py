"""This module contains functions for plotting audio-related data."""

import matplotlib.pyplot as plt
import torch
from IPython.display import Audio, display

from senselab.audio.data_structures.audio import Audio as AudioData


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


def plot_specgram(spectrogram: torch.Tensor, sample_rate: int, title: str = "Spectrogram") -> None:
    """Plots the spectrogram of an Audio object.

    Args:
        spectrogram (torch.Tensor): A tensor representing the spectrogram.
        sample_rate (int): The sampling rate of the audio data.
        title (str): Title of the spectrogram plot.

    Todo:
        - Add option to save the plot
        - Add option to choose the size of the Figure
    """
    if spectrogram.dim() != 2:
        raise ValueError("Spectrogram must be a 2D tensor.")
    plt.figure(figsize=(10, 4))
    plt.imshow(
        spectrogram.numpy(),
        aspect="auto",
        origin="lower",
        extent=(0, spectrogram.size(1) / sample_rate, 0, sample_rate / 2),
        cmap="viridis",
    )
    plt.colorbar(label="Magnitude (dB)")
    plt.title(title)
    plt.ylabel("Frequency [Hz]")
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
