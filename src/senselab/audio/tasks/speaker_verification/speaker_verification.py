"""Audio Processing and Speaker Verification Module.

This module provides functions for resampling audio using an IIR filter and
verifying if two audio samples or files are from the same speaker using a
specified model.
"""

import typing as ty

from speechbrain.inference.speaker import SpeakerRecognition

from senselab.audio.data_structures.audio import Audio
from senselab.audio.tasks.preprocessing.preprocessing import resample_audios
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import SpeechBrainModel


def verify_speaker(
    audio1: Audio,
    audio2: Audio,
    model: SpeechBrainModel = SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main"),
    resample_rate: int = 16000,
    device: DeviceType = DeviceType.CPU,
) -> ty.Tuple[float, bool]:
    """Verifies if two audio samples are from the same speaker.

    Args:
        audio1 (Audio): The first audio sample.
        audio2 (Audio): The second audio sample.
        model (str): The path to the speaker verification model.
        resample_rate (int): The sample rate expected by the model.
        device (str, optional): The device to run the model on. Defaults to None.

    Returns:
        Tuple[float, bool]: The verification score and prediction.
    """
    if resample_rate != audio1.sampling_rate:
        audio1_resampled = resample_audios([audio1], resample_rate, method="iir", lowcut=resample_rate / 2 - 100)
    if resample_rate != audio2.sampling_rate:
        audio2_resampled = resample_audios([audio2], resample_rate, method="iir", lowcut=resample_rate / 2 - 100)
    verification = SpeakerRecognition.from_hparams(source=model.path_or_uri, run_opts={"device": device.value})
    score, prediction = verification.verify_batch(audio1_resampled[0].waveform, audio2_resampled[0].waveform)
    return float(score), bool(prediction)
