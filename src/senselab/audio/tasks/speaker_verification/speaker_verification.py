"""Audio Processing and Speaker Verification Module.

This module provides functions for resampling audio using an IIR filter and
verifying if two audio samples or files are from the same speaker using a
specified model.
"""

import typing as ty
from pathlib import Path

import torch
from scipy import signal
from speechbrain.augment.time_domain import Resample

from senselab.audio.data_structures.audio import Audio


def _resample_iir(audio: Audio, lowcut: float, new_sample_rate: int, order: int = 4) -> Audio:
    """Resamples audio using an IIR filter.

    Args:
        audio (Audio): The audio signal to resample.
        lowcut (float): The low cut frequency for the IIR filter.
        new_sample_rate (int): The new sample rate after resampling.
        order (int, optional): The order of the IIR filter. Defaults to 4.

    Returns:
        Audio: The resampled audio signal.
    """
    sos = signal.butter(order, lowcut, btype="low", output="sos", fs=new_sample_rate)
    filtered = torch.from_numpy(signal.sosfiltfilt(sos, audio.waveform.squeeze().numpy()).copy()).float()
    resampler = Resample(orig_freq=audio.sampling_rate, new_freq=new_sample_rate)
    return Audio(waveform=resampler(filtered.unsqueeze(0)).squeeze(0), sampling_rate=new_sample_rate)


def verify_speaker(
    audio1: Audio, audio2: Audio, model: str, model_rate: int, device: ty.Optional[str] = None
) -> ty.Tuple[float, float]:
    """Verifies if two audio samples are from the same speaker.

    Args:
        audio1 (Audio): The first audio sample.
        audio2 (Audio): The second audio sample.
        model (str): The path to the speaker verification model.
        model_rate (int): The sample rate expected by the model.
        device (str, optional): The device to run the model on. Defaults to None.

    Returns:
        Tuple[float, float]: The verification score and prediction.
    """
    from speechbrain.inference.speaker import SpeakerRecognition

    if model_rate != audio1.sampling_rate:
        audio1 = _resample_iir(audio1, model_rate / 2 - 100, model_rate)
    if model_rate != audio2.sampling_rate:
        audio2 = _resample_iir(audio2, model_rate / 2 - 100, model_rate)

    verification = SpeakerRecognition.from_hparams(source=model, run_opts={"device": device})
    score, prediction = verification.verify_batch(audio1.waveform.T, audio2.waveform.T)
    return score, prediction


def verify_speaker_from_files(
    file1: Path, file2: Path, model: str, device: ty.Optional[str] = None
) -> ty.Tuple[float, float]:
    """Verifies if two audio files are from the same speaker.

    Args:
        file1 (Path): The path to the first audio file.
        file2 (Path): The path to the second audio file.
        model (str): The path to the speaker verification model.
        device (str, optional): The device to run the model on. Defaults to None.

    Returns:
        Tuple[float, float]: The verification score and prediction.
    """
    from speechbrain.inference.speaker import SpeakerRecognition

    verification = SpeakerRecognition.from_hparams(source=model, run_opts={"device": device})
    return verification.verify_files(file1, file2)
