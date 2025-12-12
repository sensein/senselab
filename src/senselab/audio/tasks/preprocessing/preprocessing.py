"""Audio preprocessing utilities.

This module provides preprocessing primitives operating on in-memory
`senselab.audio.data_structures.Audio` objects.

Notes:
    - These functions operate on `Audio` objects (already loaded in memory); they
      do not read or write files.
"""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from scipy import signal

from senselab.audio.data_structures import Audio

try:
    from speechbrain.augment.time_domain import Resample

    SPEECHBRAIN_AVAILABLE = True
except ModuleNotFoundError:
    SPEECHBRAIN_AVAILABLE = False


# ---------------------------------------------------------------------
# Resample Audios
# ---------------------------------------------------------------------
def resample_audios(
    audios: List[Audio], resample_rate: int, lowcut: Optional[float] = None, order: int = 4
) -> List[Audio]:
    """Resample a batch of `Audio` objects to a target sampling rate.

    For each channel, a zero-phase IIR low-pass filter (Butterworth, SOS) is applied,
    then resampling is performed with `speechbrain.augment.time_domain.Resample`.

    Args:
        audios (list[Audio]):
            Input audio objects to be resampled.
        resample_rate (int):
            Target sampling rate in Hz.
        lowcut (float, optional):
            Low-pass cutoff frequency in Hz (applied before resampling). If ``None``,
            defaults to ``resample_rate / 2 - 100`` to avoid Nyquist artifacts.
        order (int, optional):
            Butterworth filter order (default: 4).

    Returns:
        list[Audio]: Resampled `Audio` objects (sampling rate set to `resample_rate`).

    Raises:
        ModuleNotFoundError:
            If `speechbrain` is not installed.
        ValueError:
            If provided parameters are invalid.

    Example:
        >>> from senselab.audio.tasks.preprocessing import resample_audios
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> out = resample_audios([a1], resample_rate=16000)
        >>> out[0].sampling_rate
        16000
    """
    if not SPEECHBRAIN_AVAILABLE:
        raise ModuleNotFoundError(
            "`speechbrain` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    outs: List[Audio] = []

    for audio in audios:
        # Defensive copy of metadata
        md = audio.metadata.copy()

        # Design low-pass if not provided
        _lowcut = lowcut if lowcut is not None else (resample_rate / 2 - 100.0)
        sos = signal.butter(order, _lowcut, btype="low", output="sos", fs=resample_rate)

        # Per-channel IIR (zero-phase) then resample
        resampler = Resample(orig_freq=audio.sampling_rate, new_freq=resample_rate)
        channels = []
        for ch in audio.waveform:
            # sosfiltfilt works on numpy
            filtered = signal.sosfiltfilt(sos, ch.detach().cpu().numpy()).copy()
            filtered_t = torch.from_numpy(filtered).float().unsqueeze(0)  # [1, T]
            res_ch = resampler(filtered_t).squeeze(0)  # [T']
            channels.append(res_ch)

        resampled_waveform = torch.stack(channels, dim=0)  # [C, T']
        outs.append(Audio(waveform=resampled_waveform, sampling_rate=resample_rate, metadata=md))

    return outs


# ---------------------------------------------------------------------
# Downmix to Mono
# ---------------------------------------------------------------------
def downmix_audios_to_mono(audios: List[Audio]) -> List[Audio]:
    """Downmix each `Audio` to mono by averaging channels.

    Channel averaging preserves the original sampling rate and copies metadata.

    Args:
        audios (list[Audio]):
            Input `Audio` objects. Works for mono and multi-channel inputs.

    Returns:
        list[Audio]: Mono `Audio` objects (shape [1, T]).

    Example:
        >>> from senselab.audio.tasks.preprocessing import downmix_audios_to_mono
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> out = downmix_audios_to_mono([a1])
        >>> out[0].waveform.shape[0]
        1
    """
    outs: List[Audio] = []
    for audio in audios:
        outs.append(
            Audio(
                waveform=audio.waveform.mean(dim=0, keepdim=True),
                sampling_rate=audio.sampling_rate,
                metadata=audio.metadata.copy(),
            )
        )
    return outs


# ---------------------------------------------------------------------
# Select Channel
# ---------------------------------------------------------------------
def select_channel_from_audios(audios: List[Audio], channel_index: int) -> List[Audio]:
    """Select a specific channel from each `Audio`.

    Args:
        audios (list[Audio]):
            Input `Audio` objects (mono or multi-channel).
        channel_index (int):
            Zero-based channel index to extract.

    Returns:
        list[Audio]: `Audio` objects containing only the selected channel (shape [1, T]).

    Raises:
        ValueError:
            If `channel_index` is out of range for any input.

    Example:
        >>> from senselab.audio.tasks.preprocessing import select_channel_from_audios
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> ch0 = select_channel_from_audios([a1], channel_index=0)
        >>> ch0[0].waveform.shape[0]
        1
    """
    outs: List[Audio] = []
    for audio in audios:
        if channel_index < 0 or audio.waveform.size(0) <= channel_index:
            raise ValueError("channel_index should be a valid non-negative integer within channel range.")
        wf = audio.waveform[channel_index, :].unsqueeze(0)  # keep [1, T]
        outs.append(Audio(waveform=wf, sampling_rate=audio.sampling_rate, metadata=audio.metadata.copy()))
    return outs


# ---------------------------------------------------------------------
# Chunk Audios (by single [start, end])
# ---------------------------------------------------------------------
def chunk_audios(
    data: Sequence[Tuple[Audio, Tuple[float, float]]],
) -> List[Audio]:
    """Extract a single time chunk from each `Audio`.

    Each pair provides the `(start, end)` times (in seconds) of the desired chunk.

    Args:
        data (Sequence[tuple[Audio, tuple[float, float]]]):
            Sequence like ``[(audio, (start, end)), ...]``.

    Returns:
        list[Audio]: One chunked `Audio` per input pair.

    Raises:
        ValueError:
            If `start < 0`, `start >= end`, or `end` exceeds audio duration.

    Example:
        >>> from senselab.audio.tasks.preprocessing import chunk_audios
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> pairs = [(a1, (0.5, 2.0)), (a1, (1.0, 3.0))]
        >>> chunks = chunk_audios(pairs)
        >>> len(chunks)
        2
    """
    outs: List[Audio] = []
    for audio, (start, end) in data:
        if start < 0:
            raise ValueError("Start time must be >= 0.")
        if start >= end:
            raise ValueError(f"Start time ({start}) must be < end time ({end}).")

        sr = audio.sampling_rate
        wf = audio.waveform  # shape: (C, T)
        duration = wf.shape[1] / sr
        if end > duration:
            raise ValueError(f"End time ({end}) must be <= audio duration ({duration:.6f} s).")

        i0 = int(start * sr)
        i1 = int(end * sr)
        chunked_wf = wf[:, i0:i1]

        outs.append(
            Audio(
                waveform=chunked_wf,
                sampling_rate=sr,
                metadata={**audio.metadata, "segment_start": float(start), "segment_end": float(end)},
            )
        )

    return outs


# ---------------------------------------------------------------------
# Extract multiple segments per audio
# ---------------------------------------------------------------------
def extract_segments(data: List[Tuple[Audio, List[Tuple[float, float]]]]) -> List[List[Audio]]:
    """Extract multiple time segments per `Audio`.

    For each input `Audio`, a list of `(start, end)` segments (in seconds) is provided,
    and the function returns a list of chunked `Audio` objects.

    Args:
        data (list[tuple[Audio, list[tuple[float, float]]]]):
            Items like ``(audio, [(s1, e1), (s2, e2), ...])``.

    Returns:
        list[list[Audio]]: For each input audio, a list of extracted segments.

    Raises:
        ValueError:
            If any segment is invalid or exceeds audio duration.

    Example:
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.audio.tasks.preprocessing import extract_segments
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> specs = [(a1, [(0.0, 1.0), (1.5, 2.0)])]
        >>> segmented = extract_segments(specs)
        >>> len(segmented[0])
        2
    """
    result: List[List[Audio]] = []

    for audio, timestamps in data:
        sr = audio.sampling_rate
        dur = audio.waveform.shape[1] / sr
        out: List[Audio] = []
        for start, end in timestamps:
            if start < 0:
                raise ValueError("Start time must be >= 0.")
            if end > dur:
                raise ValueError(f"End must be <= duration of the audio ({dur} sec).")
            s = int(start * sr)
            e = int(end * sr)
            wf = audio.waveform[:, s:e]
            out.append(Audio(waveform=wf, sampling_rate=sr, metadata=audio.metadata.copy()))
        result.append(out)

    return result


# ---------------------------------------------------------------------
# Pad Audios to desired length (samples)
# ---------------------------------------------------------------------
def pad_audios(audios: List[Audio], desired_samples: int) -> List[Audio]:
    """Right-pad each audio to a desired number of samples.

    Pads with zeros to reach `desired_samples`. If an audio is already longer or equal,
    it is returned unchanged.

    Args:
        audios (list[Audio]):
            Input `Audio` objects.
        desired_samples (int):
            Target length in samples (per channel).

    Returns:
        list[Audio]: Padded `Audio` objects.

    Example:
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.audio.tasks.preprocessing import pad_audios
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> padded = pad_audios([a1], desired_samples=16000)
        >>> padded[0].waveform.shape[1] >= 16000
        True
    """
    outs: List[Audio] = []
    for audio in audios:
        current = audio.waveform.shape[1]
        if current >= desired_samples:
            outs.append(audio)
            continue
        padding_needed = desired_samples - current
        padded_waveform = torch.nn.functional.pad(audio.waveform, (0, padding_needed))
        outs.append(
            Audio(
                waveform=padded_waveform,
                sampling_rate=audio.sampling_rate,
                metadata=audio.metadata.copy(),
            )
        )

    return outs


# ---------------------------------------------------------------------
# Evenly segment audios (optionally pad last)
# ---------------------------------------------------------------------
def evenly_segment_audios(
    audios: List[Audio], segment_length: float, pad_last_segment: bool = True
) -> List[List[Audio]]:
    """Split each audio into fixed-length segments (seconds), optionally padding the last.

    If `pad_last_segment` is True and the final segment is shorter than `segment_length`,
    it is right-padded with zeros to match the target duration.

    Args:
        audios (list[Audio]):
            Input `Audio` objects.
        segment_length (float):
            Segment duration in seconds.
        pad_last_segment (bool, optional):
            Whether to pad the last segment to full length (default: True).

    Returns:
        list[list[Audio]]: For each input, a list of fixed-length segments.

    Example:
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.audio.tasks.preprocessing import evenly_segment_audios
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> segs = evenly_segment_audios([a1], segment_length=1.0, pad_last_segment=True)
        >>> len(segs[0]) >= 1
        True
    """
    all_segments: List[List[Audio]] = []

    for audio in audios:
        sr = audio.sampling_rate
        total_samples = audio.waveform.shape[1]
        seg_samples = max(1, int(round(segment_length * sr)))
        segments: List[Audio] = []

        if total_samples == 0:
            all_segments.append(
                [Audio(waveform=audio.waveform.clone(), sampling_rate=sr, metadata=audio.metadata.copy())]
            )
            continue

        # Full segments
        n_full = total_samples // seg_samples
        for i in range(n_full):
            s = i * seg_samples
            e = s + seg_samples
            wf = audio.waveform[:, s:e]
            segments.append(Audio(waveform=wf, sampling_rate=sr, metadata=audio.metadata.copy()))

        # Remainder
        rem = total_samples % seg_samples
        if rem > 0:
            s = n_full * seg_samples
            wf = audio.waveform[:, s:total_samples]
            if pad_last_segment:
                pad_needed = seg_samples - wf.shape[1]
                wf = torch.nn.functional.pad(wf, (0, pad_needed))
            segments.append(Audio(waveform=wf, sampling_rate=sr, metadata=audio.metadata.copy()))

        # Edge case: if audio shorter than one segment and no remainder branch executed (because rem==total)
        if len(segments) == 0:
            wf = audio.waveform
            if pad_last_segment and wf.shape[1] < seg_samples:
                pad_needed = seg_samples - wf.shape[1]
                wf = torch.nn.functional.pad(wf, (0, pad_needed))
            segments.append(Audio(waveform=wf, sampling_rate=sr, metadata=audio.metadata.copy()))

        all_segments.append(segments)

    return all_segments


# ---------------------------------------------------------------------
# Concatenate audio objects
# ---------------------------------------------------------------------
def concatenate_audios(audios: List[Audio]) -> Audio:
    """Concatenate multiple `Audio` objects along the time axis.

    All inputs must have the same sampling rate and number of channels. Metadata
    is not merged; the output `Audio` contains the concatenated waveform and
    the common sampling rate.

    Args:
        audios (list[Audio]):
            Non-empty list of `Audio` objects to concatenate.

    Returns:
        Audio: A single `Audio` with waveform concatenated along time.

    Raises:
        ValueError:
            If the list is empty, or inputs differ in sampling rate or channel count.

    Example:
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.audio.tasks.preprocessing import concatenate_audios
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> a2 = Audio(filepath=Path("sample2.wav").resolve())
        >>> a3 = Audio(filepath=Path("sample3.wav").resolve())
        >>> out = concatenate_audios([a1, a2, a3])
        >>> out.waveform.shape[1] == a1.waveform.shape[1] + a2.waveform.shape[1] + a3.waveform.shape[1]
        True
    """
    if not audios:
        raise ValueError("The input list is empty. Please provide a list with at least one Audio object.")

    sampling_rate = audios[0].sampling_rate
    num_channels = audios[0].waveform.shape[0]

    for audio in audios:
        if audio.sampling_rate != sampling_rate:
            raise ValueError("All audios must have the same sampling rate to concatenate.")
        if audio.waveform.shape[0] != num_channels:
            raise ValueError("All audios must have the same number of channels (mono or stereo) to concatenate.")

    concatenated_waveform = torch.cat([audio.waveform.cpu() for audio in audios], dim=1)
    return Audio(
        waveform=concatenated_waveform,
        sampling_rate=sampling_rate,
    )
