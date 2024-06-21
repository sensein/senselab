"""This module contains functions that extract features from audio files using the PRAAT library."""

from typing import Dict, List

import parselmouth
import pydra
from parselmouth.praat import call

from senselab.audio.data_structures.audio import Audio


def get_audios_durations(audios: List[Audio]) -> List[Dict[str, float]]:
    """Returns the duration of the Audio objects."""

    def get_audio_duration(audio: Audio) -> Dict[str, float]:
        def _get_duration(sound: parselmouth.Sound) -> float:
            return call(sound, "Get total duration")

        waveform = audio.waveform
        sampling_rate = audio.sampling_rate
        sound = parselmouth.Sound(waveform, sampling_frequency=sampling_rate)
        duration = _get_duration(sound)
        return {"duration": duration}

    durations = []
    for audio in audios:
        durations.append(get_audio_duration(audio))
    return durations


def get_audios_f0_descriptors(
    audios: List[Audio],
    f0min: float,
    f0max: float,
    unit: str = "Hertz",
) -> List[Dict[str, float]]:
    """Returns the fundamental frequency descriptors of the audios."""

    def get_audio_f0(audio: Audio, f0min: float, f0max: float, unit: str) -> Dict[str, float]:
        def _to_pitch(sound: parselmouth.Sound, f0min: float, f0max: float) -> parselmouth.Pitch:
            return call(sound, "To Pitch", 0.0, f0min, f0max)

        def _get_mean_f0(pitch: parselmouth.Pitch, unit: str) -> float:
            return call(pitch, "Get mean", 0, 0, unit)

        def _get_std_dev_f0(pitch: parselmouth.Pitch, unit: str) -> float:
            return call(pitch, "Get standard deviation", 0, 0, unit)

        waveform = audio.waveform
        sampling_rate = audio.sampling_rate
        sound = parselmouth.Sound(waveform, sampling_frequency=sampling_rate)

        pitch = _to_pitch(sound, f0min, f0max)
        return {
            f"f0_mean_{unit}": _get_mean_f0(pitch, unit),
            f"f0_std_dev_{unit}": _get_std_dev_f0(pitch, unit),
        }

    return [get_audio_f0(audio, f0min, f0max, unit) for audio in audios]


def get_audios_harmonicity_descriptors(audios: List[Audio], f0min: float) -> List[Dict[str, float]]:
    """Returns the harmonicity descriptors of the audios."""

    def get_audio_harmonicity(audio: Audio, f0min: float) -> Dict[str, float]:
        def _to_harmonicity(sound: parselmouth.Sound, f0min: float) -> parselmouth.Harmonicity:
            return call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)

        def _get_mean_hnr(harmonicity: parselmouth.Harmonicity) -> float:
            return call(harmonicity, "Get mean", 0, 0)

        def _get_std_dev_hnr(harmonicity: parselmouth.Harmonicity) -> float:
            return call(harmonicity, "Get standard deviation", 0, 0)

        waveform = audio.waveform
        sampling_rate = audio.sampling_rate
        sound = parselmouth.Sound(waveform, sampling_frequency=sampling_rate)

        harmonicity = _to_harmonicity(sound, f0min)
        return {
            "harmonicity_mean": _get_mean_hnr(harmonicity),
            "harmonicity_std_dev": _get_std_dev_hnr(harmonicity),
        }

    return [get_audio_harmonicity(audio, f0min) for audio in audios]


def get_audios_jitter_descriptors(
    audios: List[Audio],
    f0min: float,
    f0max: float,
) -> List[Dict[str, float]]:
    """Returns the jitter descriptors of the audios."""

    def get_audio_jitter(audio: Audio, f0min: float, f0max: float) -> Dict[str, float]:
        def _to_point_process(sound: parselmouth.Sound, f0min: float, f0max: float) -> parselmouth.Data:
            return call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

        def _get_jitter(type: str, point_process: parselmouth.Data) -> float:
            return call(point_process, f"Get jitter ({type})", 0, 0, 0.0001, 0.02, 1.3)

        waveform = audio.waveform
        sampling_rate = audio.sampling_rate
        sound = parselmouth.Sound(waveform, sampling_frequency=sampling_rate)
        point_process = _to_point_process(sound, f0min, f0max)
        return {
            "local_jitter": _get_jitter("local", point_process),
            "localabsolute_jitter": _get_jitter("local, absolute", point_process),
            "rap_jitter": _get_jitter("rap", point_process),
            "ppq5_jitter": _get_jitter("ppq5", point_process),
            "ddp_jitter": _get_jitter("ddp", point_process),
        }

    return [get_audio_jitter(audio, f0min, f0max) for audio in audios]


def get_audios_shimmer_descriptors(
    audios: List[Audio],
    f0min: float,
    f0max: float,
) -> List[Dict[str, float]]:
    """Returns the shimmer descriptors of the audios."""

    def get_audio_shimmer(audio: Audio, f0min: float, f0max: float) -> Dict[str, float]:
        def _to_point_process(sound: parselmouth.Sound, f0min: float, f0max: float) -> parselmouth.Data:
            return call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

        def _get_shimmer(type: str, sound: parselmouth.Sound, point_process: parselmouth.Data) -> float:
            return call([sound, point_process], f"Get shimmer ({type})", 0, 0, 0.0001, 0.02, 1.3, 1.6)

        waveform = audio.waveform
        sampling_rate = audio.sampling_rate
        sound = parselmouth.Sound(waveform, sampling_frequency=sampling_rate)
        point_process = _to_point_process(sound, f0min, f0max)
        return {
            "local_shimmer": _get_shimmer("local", sound, point_process),
            "localDB_shimmer": _get_shimmer("local_dB", sound, point_process),
            "apq3_shimmer": _get_shimmer("apq3", sound, point_process),
            "apq5_shimmer": _get_shimmer("apq5", sound, point_process),
            "apq11_shimmer": _get_shimmer("apq11", sound, point_process),
            "dda_shimmer": _get_shimmer("dda", sound, point_process),
        }

    return [get_audio_shimmer(audio, f0min, f0max) for audio in audios]


get_audios_durations_pt = pydra.mark.task(get_audios_durations)
get_audios_f0_descriptors_pt = pydra.mark.task(get_audios_f0_descriptors)
get_audios_harmonicity_descriptors_pt = pydra.mark.task(get_audios_harmonicity_descriptors)
get_audios_jitter_descriptors_pt = pydra.mark.task(get_audios_jitter_descriptors)
get_audios_shimmer_descriptors_pt = pydra.mark.task(get_audios_shimmer_descriptors)
