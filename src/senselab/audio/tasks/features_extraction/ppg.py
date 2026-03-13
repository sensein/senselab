"""This module provides the implementation of Phonetic Posteriorgrams (PPGs) for audio features extraction."""

import traceback
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional

import torch
from matplotlib.figure import Figure

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.preprocessing import chunk_audios
from senselab.utils.data_structures import DeviceType, _select_device_and_dtype, logger

try:
    import ppgs

    PPGS_AVAILABLE = True
except ModuleNotFoundError:
    PPGS_AVAILABLE = False


def extract_ppgs_from_audios(
    audios: List["Audio"],
    device: Optional[DeviceType] = None,
    start_times: Optional[List[Optional[float]]] = None,
    end_times: Optional[List[Optional[float]]] = None,
) -> List[torch.Tensor]:
    """Extracts phonetic posteriorgrams (PPGs) from every audio.

    Args:
        audios (List[Audio]): The audios to extract PPGs from
        device (Optional[DeviceType]): Device to use for extracting PPGs
        start_times (Optional[List[Optional[float]]]): Optional per-audio start times in seconds
        end_times (Optional[List[Optional[float]]]): Optional per-audio end times in seconds

    Returns:
        List[Tensor]: The PPG for each input audio
    """
    if not PPGS_AVAILABLE:
        raise ModuleNotFoundError(
            "`ppgs` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    device, _ = _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU])
    prepared_audios = _slice_audios_for_ppg_extraction(audios=audios, start_times=start_times, end_times=end_times)

    if any(audio.waveform.shape[0] != 1 for audio in prepared_audios):
        raise ValueError("Only mono audio is supported by ppgs model.")

    posteriorgrams = []
    for audio in prepared_audios:
        try:
            posteriorgrams.append(
                ppgs.from_audio(
                    torch.unsqueeze(audio.waveform, dim=0),
                    ppgs.SAMPLE_RATE,
                    gpu=0 if device == DeviceType.CUDA else None,
                ).cpu()
            )

        except RuntimeError as e:
            logger.error(f"Encountered RuntimeError when extracting ppgs: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            posteriorgrams.append(torch.tensor(torch.nan))

    return posteriorgrams


def extract_mean_phoneme_durations(audio: Audio, posteriorgram: torch.Tensor) -> Dict[str, Any]:
    """Summarize mean phoneme durations from a posteriorgram.

    Args:
        audio (Audio): Audio corresponding to the posteriorgram.
        posteriorgram (torch.Tensor): Raw model output, typically shaped
            ``(1, phonemes, frames)`` or ``(phonemes, frames)``.

    Returns:
        Dict[str, Any]: Aggregate duration statistics per phoneme.
    """
    frame_major = _to_frame_major_posteriorgram(posteriorgram)
    if frame_major is None:
        return {
            "frame_count": 0,
            "phoneme_count": 0,
            "analysis_duration_seconds": 0.0,
            "seconds_per_frame": 0.0,
            "mean_segment_duration_seconds": 0.0,
            "phoneme_durations": [],
        }

    segments = _extract_ppg_segments(audio=audio, frame_major_posteriorgram=frame_major)
    frame_count = int(frame_major.shape[0])
    analysis_duration_seconds = float(audio.waveform.shape[1] / audio.sampling_rate)
    seconds_per_frame = analysis_duration_seconds / max(frame_count, 1)

    durations_by_phoneme: dict[int, list[float]] = defaultdict(list)
    segment_durations: list[float] = []
    for segment in segments:
        segment_duration = float(segment["duration_seconds"])
        segment_durations.append(segment_duration)
        durations_by_phoneme[int(segment["phoneme_index"])].append(segment_duration)

    phoneme_durations = []
    for phoneme_index, phoneme in enumerate(ppgs.PHONEMES):
        durations = durations_by_phoneme.get(phoneme_index)
        if not durations:
            continue
        phoneme_durations.append(
            {
                "phoneme": phoneme,
                "count": len(durations),
                "mean_duration_seconds": round(mean(durations), 6),
                "total_duration_seconds": round(sum(durations), 6),
            }
        )

    return {
        "frame_count": frame_count,
        "phoneme_count": len(phoneme_durations),
        "analysis_duration_seconds": round(analysis_duration_seconds, 6),
        "seconds_per_frame": round(seconds_per_frame, 6),
        "mean_segment_duration_seconds": round(mean(segment_durations), 6) if segment_durations else 0.0,
        "phoneme_durations": phoneme_durations,
    }


def plot_ppg_phoneme_timeline(
    audio: Audio,
    posteriorgram: torch.Tensor,
    title: str = "PPG phoneme timeline",
    show: bool = True,
) -> Figure:
    """Plot contiguous PPG phoneme segments with onset and offset markers.

    Args:
        audio (Audio): Audio corresponding to the posteriorgram.
        posteriorgram (torch.Tensor): Raw model output, typically shaped
            ``(1, phonemes, frames)`` or ``(phonemes, frames)``.
        title (str): Figure title.
        show (bool): Whether to display the figure with ``plt.show(block=False)``.

    Returns:
        matplotlib.figure.Figure: The created figure.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("`matplotlib` is required for PPG timeline plotting.") from exc

    frame_major = _to_frame_major_posteriorgram(posteriorgram)
    if frame_major is None:
        raise ValueError("Cannot plot an empty or NaN-only posteriorgram.")

    segments = _extract_ppg_segments(audio=audio, frame_major_posteriorgram=frame_major)
    present_indices = sorted({int(segment["phoneme_index"]) for segment in segments})
    y_positions = {phoneme_index: position for position, phoneme_index in enumerate(present_indices)}
    colors = plt.get_cmap("tab20", len(present_indices))

    figure_height = max(6.0, len(present_indices) * 0.35)
    fig, ax = plt.subplots(figsize=(18, figure_height))

    for segment in segments:
        phoneme_index = int(segment["phoneme_index"])
        y_position = y_positions[phoneme_index]
        start_seconds = float(segment["start_seconds"])
        duration_seconds = float(segment["duration_seconds"])
        color = colors(y_position)
        ax.broken_barh(
            [(start_seconds, duration_seconds)],
            (y_position - 0.35, 0.7),
            facecolors=color,
            edgecolors=color,
            alpha=0.85,
        )
        ax.plot(
            [start_seconds, start_seconds + duration_seconds],
            [y_position, y_position],
            "|",
            color="black",
            markersize=8,
            markeredgewidth=1,
        )

    duration_seconds = float(audio.waveform.shape[1] / audio.sampling_rate)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Phoneme")
    ax.set_xlim(0, duration_seconds)
    ax.set_ylim(-1, len(present_indices))
    ax.set_yticks(range(len(present_indices)))
    ax.set_yticklabels([ppgs.PHONEMES[index] for index in present_indices])
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show(block=False)

    return fig


def extract_mean_phoneme_durations_from_audios(
    audios: List["Audio"],
    device: Optional[DeviceType] = None,
    start_times: Optional[List[Optional[float]]] = None,
    end_times: Optional[List[Optional[float]]] = None,
) -> List[Dict[str, Any]]:
    """Extract mean phoneme duration summaries for a batch of audios.

    Args:
        audios (List[Audio]): Input audios to summarize.
        device (Optional[DeviceType]): Device used for PPG extraction.
        start_times (Optional[List[Optional[float]]]): Optional per-audio start times in seconds.
        end_times (Optional[List[Optional[float]]]): Optional per-audio end times in seconds.

    Returns:
        List[Dict[str, Any]]: One duration summary per audio.
    """
    prepared_audios = _slice_audios_for_ppg_extraction(audios=audios, start_times=start_times, end_times=end_times)
    posteriorgrams = extract_ppgs_from_audios(prepared_audios, device=device)
    return [
        extract_mean_phoneme_durations(audio=audio, posteriorgram=posteriorgram)
        for audio, posteriorgram in zip(prepared_audios, posteriorgrams)
    ]


def _slice_audios_for_ppg_extraction(
    audios: List["Audio"],
    start_times: Optional[List[Optional[float]]],
    end_times: Optional[List[Optional[float]]],
) -> List[Audio]:
    """Slice audios to optional per-item windows before PPG extraction."""
    if start_times is None and end_times is None:
        return audios

    if start_times is not None and len(start_times) != len(audios):
        raise ValueError("start_times should have the same length as audios.")
    if end_times is not None and len(end_times) != len(audios):
        raise ValueError("end_times should have the same length as audios.")

    chunk_specs = []
    for index, audio in enumerate(audios):
        duration_seconds = float(audio.waveform.shape[1] / audio.sampling_rate)
        start_time = 0.0 if start_times is None or start_times[index] is None else float(start_times[index])
        end_time = duration_seconds if end_times is None or end_times[index] is None else float(end_times[index])
        chunk_specs.append((audio, (start_time, end_time)))

    return chunk_audios(chunk_specs)


def _to_frame_major_posteriorgram(posteriorgram: torch.Tensor) -> Optional[torch.Tensor]:
    """Normalize a posteriorgram to ``(frames, phonemes)`` layout."""
    if not isinstance(posteriorgram, torch.Tensor) or posteriorgram.numel() == 0:
        return None
    if torch.isnan(posteriorgram).all():
        return None

    normalized = posteriorgram.detach().cpu()
    if normalized.ndim == 3:
        if normalized.shape[0] != 1:
            raise ValueError("Expected batched posteriorgrams to contain exactly one audio item.")
        normalized = normalized.squeeze(0)

    if normalized.ndim != 2:
        raise ValueError("Expected posteriorgram to have 2 or 3 dimensions.")

    phoneme_count = len(ppgs.PHONEMES)
    if normalized.shape[0] == phoneme_count:
        return normalized.transpose(0, 1)
    if normalized.shape[1] == phoneme_count:
        return normalized

    raise ValueError(
        "Expected one posteriorgram dimension to match the PPG phoneme inventory size "
        f"({phoneme_count}), received shape {tuple(normalized.shape)}."
    )


def _extract_ppg_segments(audio: Audio, frame_major_posteriorgram: torch.Tensor) -> List[Dict[str, Any]]:
    """Extract contiguous argmax phoneme segments from a frame-major posteriorgram."""
    frame_count = int(frame_major_posteriorgram.shape[0])
    duration_seconds = float(audio.waveform.shape[1] / audio.sampling_rate)
    seconds_per_frame = duration_seconds / max(frame_count, 1)
    best_labels = frame_major_posteriorgram.argmax(dim=1).tolist()

    segments: List[Dict[str, Any]] = []
    start_index = 0
    current_label = int(best_labels[0])

    for frame_index in range(1, frame_count):
        label_index = int(best_labels[frame_index])
        if label_index == current_label:
            continue

        segments.append(
            {
                "phoneme": ppgs.PHONEMES[current_label],
                "phoneme_index": current_label,
                "start_frame": start_index,
                "end_frame": frame_index - 1,
                "start_seconds": round(start_index * seconds_per_frame, 6),
                "end_seconds": round(frame_index * seconds_per_frame, 6),
                "duration_seconds": round((frame_index - start_index) * seconds_per_frame, 6),
            }
        )
        start_index = frame_index
        current_label = label_index

    segments.append(
        {
            "phoneme": ppgs.PHONEMES[current_label],
            "phoneme_index": current_label,
            "start_frame": start_index,
            "end_frame": frame_count - 1,
            "start_seconds": round(start_index * seconds_per_frame, 6),
            "end_seconds": round(duration_seconds, 6),
            "duration_seconds": round((frame_count - start_index) * seconds_per_frame, 6),
        }
    )

    return segments
