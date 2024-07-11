"""Audio Processing and Speaker Verification Module.

This module provides functions for resampling audio using an IIR filter and
verifying if two audio samples or files are from the same speaker using a
specified model.
"""

import typing as ty

from speechbrain.inference.speaker import SpeakerRecognition

from senselab.audio.data_structures.audio import Audio
from senselab.utils.data_structures.device import DeviceType
from senselab.utils.data_structures.model import SpeechBrainModel


def verify_speaker(
    audio1: Audio,
    audio2: Audio,
    model: SpeechBrainModel = SpeechBrainModel(path_or_uri="speechbrain/spkrec-ecapa-voxceleb", revision="main"),
    device: DeviceType = DeviceType.CPU,
) -> ty.Tuple[float, bool]:
    """Verifies if two audio samples are from the same speaker.

    Args:
        audio1 (Audio): The first audio sample.
        audio2 (Audio): The second audio sample.
        model (SpeechBrainModel): The model for speaker verification.
                                  Defaults to the ECAPA-TDNN model.
        device (DeviceType): The device to run the model on. Defaults to CPU.

    Returns:
        Tuple[float, bool]: The verification score and prediction.
                            The score is a float, and the prediction is a boolean.
    """
    verification = SpeakerRecognition.from_hparams(source=model.path_or_uri, run_opts={"device": device.value})
    score, prediction = verification.verify_batch(audio1.waveform, audio2.waveform)
    return float(score), bool(prediction)
