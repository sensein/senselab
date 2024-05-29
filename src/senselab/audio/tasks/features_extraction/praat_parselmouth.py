"""This module contains functions for extracting Praat features."""

from typing import Any, Dict

import parselmouth
from datasets import Dataset
from parselmouth.praat import call

from senselab.utils.tasks.input_output import (
    _from_dict_to_hf_dataset,
    _from_hf_dataset_to_dict,
)


def get_hf_dataset_durations(
    dataset: Dict[str, Any], audio_column: str = "audio"
) -> Dict[str, float]:
    """Returns the duration of the audios as a HuggingFace `Dataset` object."""
    hf_dataset = _from_dict_to_hf_dataset(dataset)
    unnecessary_columns = [
        col for col in hf_dataset.column_names if col != audio_column
    ]
    hf_dataset = hf_dataset.remove_columns(unnecessary_columns)

    def get_hf_dataset_row_duration(
        row: Dataset, audio_column: str
    ) -> Dict[str, float]:
        def _get_duration(sound: parselmouth.Sound) -> float:
            return call(sound, "Get total duration")

        waveform = row[audio_column]["array"]
        sampling_rate = row[audio_column]["sampling_rate"]
        sound = parselmouth.Sound(waveform, sampling_frequency=sampling_rate)
        duration = _get_duration(sound)
        return {"duration_seconds": duration}

    durations_hf_dataset = hf_dataset.map(
        lambda x: get_hf_dataset_row_duration(x, audio_column)
    )
    durations_hf_dataset = durations_hf_dataset.remove_columns([audio_column])
    return _from_hf_dataset_to_dict(durations_hf_dataset)


def get_hf_dataset_f0_descriptors(
    dataset: Dict[str, Any],
    f0min: float,
    f0max: float,
    audio_column: str = "audio",
    unit: str = "Hertz",
) -> Dict[str, float]:
    """Returns the fundamental frequency descriptors of the audios."""
    hf_dataset = _from_dict_to_hf_dataset(dataset)
    unnecessary_columns = [
        col for col in hf_dataset.column_names if col != audio_column
    ]
    hf_dataset = hf_dataset.remove_columns(unnecessary_columns)

    def get_hf_dataset_row_f0_descriptors(
        row: Dataset, audio_column: str, f0min: float, f0max: float, unit: str
    ) -> Dict[str, float]:
        def _to_pitch(
            sound: parselmouth.Sound, f0min: float, f0max: float
        ) -> parselmouth.Pitch:
            return call(sound, "To Pitch", 0.0, f0min, f0max)

        def _get_mean_f0(pitch: parselmouth.Pitch, unit: str) -> float:
            return call(pitch, "Get mean", 0, 0, unit)

        def _get_std_dev_f0(pitch: parselmouth.Pitch, unit: str) -> float:
            return call(pitch, "Get standard deviation", 0, 0, unit)

        waveform = row[audio_column]["array"]
        sampling_rate = row[audio_column]["sampling_rate"]
        sound = parselmouth.Sound(waveform, sampling_frequency=sampling_rate)

        pitch = _to_pitch(sound, f0min, f0max)
        return {
            f"f0_mean_{unit}": _get_mean_f0(pitch, unit),
            f"f0_std_dev_{unit}": _get_std_dev_f0(pitch, unit),
        }

    f0_descriptors_hf_dataset = hf_dataset.map(
        lambda x: get_hf_dataset_row_f0_descriptors(
            x, audio_column, f0min, f0max, unit
        )
    )
    f0_descriptors_hf_dataset = f0_descriptors_hf_dataset.remove_columns(
        [audio_column]
    )
    return _from_hf_dataset_to_dict(f0_descriptors_hf_dataset)


def get_hf_dataset_harmonicity_descriptors(
    dataset: Dict[str, Any], f0min: float, audio_column: str = "audio"
) -> Dict[str, float]:
    """Returns the harmonicity descriptors of the audios."""
    hf_dataset = _from_dict_to_hf_dataset(dataset)
    unnecessary_columns = [
        col for col in hf_dataset.column_names if col != audio_column
    ]
    hf_dataset = hf_dataset.remove_columns(unnecessary_columns)

    def _get_hf_dataset_row_harmonicity_descriptors(
        row: Dataset, audio_column: str, f0min: float
    ) -> Dict[str, float]:
        def _to_harmonicity(
            sound: parselmouth.Sound, f0min: float
        ) -> parselmouth.Harmonicity:
            return call(sound, "To Harmonicity (cc)", 0.01, f0min, 0.1, 1.0)

        def _get_mean_hnr(harmonicity: parselmouth.Harmonicity) -> float:
            return call(harmonicity, "Get mean", 0, 0)

        def _get_std_dev_hnr(harmonicity: parselmouth.Harmonicity) -> float:
            return call(harmonicity, "Get standard deviation", 0, 0)

        waveform = row[audio_column]["array"]
        sampling_rate = row[audio_column]["sampling_rate"]
        sound = parselmouth.Sound(waveform, sampling_frequency=sampling_rate)

        harmonicity = _to_harmonicity(sound, f0min)
        return {
            "harmonicity_mean": _get_mean_hnr(harmonicity),
            "harmonicity_std_dev": _get_std_dev_hnr(harmonicity),
        }

    harmonicity_descriptors_hf_dataset = hf_dataset.map(
        lambda x: _get_hf_dataset_row_harmonicity_descriptors(
            x, audio_column, f0min
        )
    )
    harmonicity_descriptors_hf_dataset = (
        harmonicity_descriptors_hf_dataset.remove_columns([audio_column])
    )
    return _from_hf_dataset_to_dict(harmonicity_descriptors_hf_dataset)


def get_hf_dataset_jitter_descriptors(
    dataset: Dict[str, Any],
    f0min: float,
    f0max: float,
    audio_column: str = "audio",
) -> Dict[str, float]:
    """Returns the jitter descriptors of the audios."""
    hf_dataset = _from_dict_to_hf_dataset(dataset)
    unnecessary_columns = [
        col for col in hf_dataset.column_names if col != audio_column
    ]
    hf_dataset = hf_dataset.remove_columns(unnecessary_columns)

    def _get_hf_dataset_row_jitter_descriptors(
        row: Dataset, audio_column: str, f0min: float, f0max: float
    ) -> Dict[str, float]:
        def _to_point_process(
            sound: parselmouth.Sound, f0min: float, f0max: float
        ) -> parselmouth.Data:
            return call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

        def _get_jitter(type: str, point_process: parselmouth.Data) -> float:
            return call(
                point_process, f"Get jitter ({type})", 0, 0, 0.0001, 0.02, 1.3
            )

        waveform = row[audio_column]["array"]
        sampling_rate = row[audio_column]["sampling_rate"]
        sound = parselmouth.Sound(waveform, sampling_frequency=sampling_rate)
        point_process = _to_point_process(sound, f0min, f0max)
        return {
            "local_jitter": _get_jitter("local", point_process),
            "localabsolute_jitter": _get_jitter(
                "local, absolute", point_process
            ),
            "rap_jitter": _get_jitter("rap", point_process),
            "ppq5_jitter": _get_jitter("ppq5", point_process),
            "ddp_jitter": _get_jitter("ddp", point_process),
        }

    jitter_descriptors_hf_dataset = hf_dataset.map(
        lambda x: _get_hf_dataset_row_jitter_descriptors(
            x, audio_column, f0min, f0max
        )
    )
    jitter_descriptors_hf_dataset = (
        jitter_descriptors_hf_dataset.remove_columns([audio_column])
    )
    return _from_hf_dataset_to_dict(jitter_descriptors_hf_dataset)


def get_hf_dataset_shimmer_descriptors(
    dataset: Dict[str, Any],
    f0min: float,
    f0max: float,
    audio_column: str = "audio",
) -> Dict[str, float]:
    """Returns the shimmer descriptors of the audios."""
    hf_dataset = _from_dict_to_hf_dataset(dataset)
    unnecessary_columns = [
        col for col in hf_dataset.column_names if col != audio_column
    ]
    hf_dataset = hf_dataset.remove_columns(unnecessary_columns)

    def _get_hf_dataset_row_shimmer_descriptors(
        row: Dataset, audio_column: str, f0min: float, f0max: float
    ) -> Dict[str, float]:
        def _to_point_process(
            sound: parselmouth.Sound, f0min: float, f0max: float
        ) -> parselmouth.Data:
            return call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

        def _get_shimmer(
            type: str, sound: parselmouth.Sound, point_process: parselmouth.Data
        ) -> float:
            # Use a single function call with flexible shimmer type
            return call(
                [sound, point_process],
                f"Get shimmer ({type})",
                0,
                0,
                0.0001,
                0.02,
                1.3,
                1.6,
            )

        waveform = row[audio_column]["array"]
        sampling_rate = row[audio_column]["sampling_rate"]
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

    shimmer_descriptors_hf_dataset = hf_dataset.map(
        lambda x: _get_hf_dataset_row_shimmer_descriptors(
            x, audio_column, f0min, f0max
        )
    )
    shimmer_descriptors_hf_dataset = (
        shimmer_descriptors_hf_dataset.remove_columns([audio_column])
    )
    return _from_hf_dataset_to_dict(shimmer_descriptors_hf_dataset)
