"""This module implements audio preprocessing utilities with Pydra workflows."""

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from pydra.compose import python, workflow
from scipy import signal

from senselab.audio.data_structures import Audio

try:
    from speechbrain.augment.time_domain import Resample

    SPEECHBRAIN_AVAILABLE = True
except ModuleNotFoundError:
    SPEECHBRAIN_AVAILABLE = False


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _select_worker(plugin: str) -> str:
    return "debug" if plugin in ("serial", "debug") else plugin


# ---------------------------------------------------------------------
# Resample Audios
# ---------------------------------------------------------------------
def resample_audios(
    audios: List[Audio],
    resample_rate: int,
    lowcut: Optional[float] = None,
    order: int = 4,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    plugin: str = "debug",
    plugin_args: Optional[Dict[str, Any]] = None,
) -> List[Audio]:
    """Resamples a list of audio signals to a given sampling rate (parallelized with Pydra)."""
    if not SPEECHBRAIN_AVAILABLE:
        raise ModuleNotFoundError(
            "`speechbrain` is not installed. "
            "Please install senselab audio dependencies using `pip install 'senselab[audio]'`."
        )
    plugin_args = plugin_args or {}

    @python.define
    def _resample(audio: Audio, resample_rate: int, lowcut: Optional[float], order: int) -> Audio:
        # Defensive copy of metadata
        md = audio.metadata.copy()

        # Design low-pass if not provided
        if lowcut is None:
            lowcut = resample_rate / 2 - 100.0
        sos = signal.butter(order, lowcut, btype="low", output="sos", fs=resample_rate)

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
        return Audio(waveform=resampled_waveform, sampling_rate=resample_rate, metadata=md)

    @workflow.define
    def _wf(xs: Sequence[Audio], resample_rate: int, lowcut: Optional[float], order: int) -> List[Audio]:
        node = workflow.add(
            _resample(resample_rate=resample_rate, lowcut=lowcut, order=order).split(audio=xs),
            name="map_resample_audio",
        )
        return node.out

    worker = _select_worker(plugin)
    res = _wf(xs=audios, resample_rate=resample_rate, lowcut=lowcut, order=order)(
        worker=worker, cache_root=cache_dir, **plugin_args
    )
    return list(res.out)


# ---------------------------------------------------------------------
# Downmix to Mono
# ---------------------------------------------------------------------
def downmix_audios_to_mono(
    audios: List[Audio],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    plugin: str = "debug",
    plugin_args: Optional[Dict[str, Any]] = None,
) -> List[Audio]:
    """Downmix Audio objects to mono by averaging channels (parallelized with Pydra)."""
    plugin_args = plugin_args or {}

    @python.define
    def _downmix(audio: Audio) -> Audio:
        return Audio(
            waveform=audio.waveform.mean(dim=0, keepdim=True),
            sampling_rate=audio.sampling_rate,
            metadata=audio.metadata.copy(),
        )

    @workflow.define
    def _wf(xs: Sequence[Audio]) -> List[Audio]:
        node = workflow.add(_downmix().split(audio=xs), name="map_downmix_audio")
        return node.out

    worker = _select_worker(plugin)
    res = _wf(xs=audios)(worker=worker, cache_root=cache_dir, **plugin_args)
    return list(res.out)


# ---------------------------------------------------------------------
# Select Channel
# ---------------------------------------------------------------------
def select_channel_from_audios(
    audios: List[Audio],
    channel_index: int,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    plugin: str = "debug",
    plugin_args: Optional[Dict[str, Any]] = None,
) -> List[Audio]:
    """Select a specific channel from each Audio (parallelized with Pydra)."""
    plugin_args = plugin_args or {}

    @python.define
    def _select(audio: Audio, channel_index: int) -> Audio:
        if channel_index < 0 or audio.waveform.size(0) <= channel_index:
            raise ValueError("channel_index should be a valid non-negative integer within channel range.")
        wf = audio.waveform[channel_index, :].unsqueeze(0)  # keep [1, T]
        return Audio(waveform=wf, sampling_rate=audio.sampling_rate, metadata=audio.metadata.copy())

    @workflow.define
    def _wf(xs: Sequence[Audio], channel_index: int) -> List[Audio]:
        node = workflow.add(_select(channel_index=channel_index).split(audio=xs), name="map_select_channel")
        return node.out

    worker = _select_worker(plugin)
    res = _wf(xs=audios, channel_index=channel_index)(worker=worker, cache_root=cache_dir, **plugin_args)
    return list(res.out)


# ---------------------------------------------------------------------
# Chunk Audios (by single [start, end])
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# Chunk Audios (by single [start, end]) via Pydra
# ---------------------------------------------------------------------
def chunk_audios(
    data: Sequence[Tuple[Audio, Tuple[float, float]]],
    cache_dir: Optional[Union[str, "os.PathLike"]] = None,
    plugin: str = "debug",
    plugin_args: Optional[Dict[str, Any]] = None,
) -> List[Audio]:
    """Chunk input audios based on provided start/end timestamps using a Pydra workflow.

    Args:
        data: A sequence of tuples in the form:
              [(audio: Audio, (start: float, end: float)), ...]
              where `start` and `end` are in seconds.
        cache_dir: Directory to use for caching the workflow (optional).
        plugin: Pydra plugin to use for submission. Defaults to "debug".
        plugin_args: Additional arguments to pass to the plugin (optional).

    Returns:
        List[Audio]: New Audio objects corresponding to the requested chunks.

    Raises:
        ValueError: If start < 0, end exceeds duration, or start >= end.
    """
    plugin_args = plugin_args or {}

    @python.define
    def _chunk_pair(pair: Tuple[Audio, Tuple[float, float]]) -> Audio:
        """Chunk a single (Audio, (start, end)) pair."""
        audio, (start, end) = pair

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

        return Audio(
            waveform=chunked_wf,
            sampling_rate=sr,
            metadata={**audio.metadata, "segment_start": float(start), "segment_end": float(end)},
        )

    @workflow.define
    def _wf(pairs: Sequence[Tuple[Audio, Tuple[float, float]]]) -> List[Audio]:
        node = workflow.add(
            _chunk_pair().split(pair=pairs),
            name="map_chunk_audio",
        )
        return node.out  # list[Audio]

    worker = "debug" if plugin in ("serial", "debug") else plugin
    res = _wf(pairs=data)(worker=worker, cache_root=cache_dir, **plugin_args)
    return list(res.out)


'''
def chunk_audios(data: List[Tuple[Audio, Tuple[float, float]]]) -> List[Audio]:
    """
    Chunk input audios based on provided start/end timestamps.

    Args:
        data: A list of tuples in the form:
              [(audio: Audio, (start: float, end: float)), ...]
              where `start` and `end` are in seconds.

    Returns:
        List[Audio]: New `Audio` objects corresponding to the requested chunks.

    Raises:
        ValueError: If start < 0, end exceeds duration, or start >= end.
    """
    chunked_audios: List[Audio] = []

    for audio, (start, end) in data:
        if start < 0:
            raise ValueError("Start time must be >= 0.")
        if start >= end:
            raise ValueError(f"Start time ({start}) must be < end time ({end}).")

        duration = audio.waveform.shape[1] / audio.sampling_rate
        if end > duration:
            raise ValueError(
                f"End time must be <= audio duration ({duration:.6f} s)."
            )

        start_sample = int(start * audio.sampling_rate)
        end_sample = int(end * audio.sampling_rate)

        chunked_waveform = audio.waveform[:, start_sample:end_sample]

        chunked_audios.append(
            Audio(
                waveform=chunked_waveform,
                sampling_rate=audio.sampling_rate,
                metadata=audio.metadata.copy(),
            )
        )

    return chunked_audios
'''


# ---------------------------------------------------------------------
# Extract multiple segments per audio
# ---------------------------------------------------------------------
def extract_segments(
    data: List[Tuple[Audio, List[Tuple[float, float]]]],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    plugin: str = "debug",
    plugin_args: Optional[Dict[str, Any]] = None,
) -> List[List[Audio]]:
    """Extract multiple segments for each input Audio (maps over audios with Pydra).

    Returns a list aligned with `data`, where each element is a list[Audio] (the extracted segments).
    """
    plugin_args = plugin_args or {}

    @python.define
    def _extract(audio: Audio, timestamps: List[Tuple[float, float]]) -> List[Audio]:
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
        return out

    @workflow.define
    def _wf(pairs: Sequence[Tuple[Audio, List[Tuple[float, float]]]]) -> List[List[Audio]]:
        # We split separately for the two items in the tuple by providing two sequences
        audios = [p[0] for p in pairs]
        tss = [p[1] for p in pairs]
        node = workflow.add(_extract().split(audio=audios, timestamps=tss), name="map_extract_segments")
        return node.out  # -> List[List[Audio]]

    worker = _select_worker(plugin)
    res = _wf(pairs=data)(worker=worker, cache_root=cache_dir, **plugin_args)
    return list(res.out)


# ---------------------------------------------------------------------
# Pad Audios to desired length (samples)
# ---------------------------------------------------------------------
def pad_audios(
    audios: List[Audio],
    desired_samples: int,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    plugin: str = "debug",
    plugin_args: Optional[Dict[str, Any]] = None,
) -> List[Audio]:
    """Right-pad each audio to `desired_samples` (parallelized with Pydra)."""
    plugin_args = plugin_args or {}

    @python.define
    def _pad(audio: Audio, desired_samples: int) -> Audio:
        current = audio.waveform.shape[1]
        if current >= desired_samples:
            return audio
        padding_needed = desired_samples - current
        padded_waveform = torch.nn.functional.pad(audio.waveform, (0, padding_needed))
        return Audio(
            waveform=padded_waveform,
            sampling_rate=audio.sampling_rate,
            metadata=audio.metadata.copy(),
        )

    @workflow.define
    def _wf(xs: Sequence[Audio], desired_samples: int) -> List[Audio]:
        node = workflow.add(_pad(desired_samples=desired_samples).split(audio=xs), name="map_pad_audio")
        return node.out

    worker = _select_worker(plugin)
    res = _wf(xs=audios, desired_samples=desired_samples)(worker=worker, cache_root=cache_dir, **plugin_args)
    return list(res.out)


# ---------------------------------------------------------------------
# Evenly segment audios (optionally pad last)
# ---------------------------------------------------------------------
def evenly_segment_audios(
    audios: List[Audio],
    segment_length: float,
    pad_last_segment: bool = True,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    plugin: str = "debug",
    plugin_args: Optional[Dict[str, Any]] = None,
) -> List[List[Audio]]:
    """Split each audio into fixed-length segments (seconds). Optionally pad the last segment.

    Returns:
        List[List[Audio]]: For each input Audio, a list of segments as Audio objects.
    """
    plugin_args = plugin_args or {}

    @python.define
    def _even_segments(audio: Audio, segment_length: float, pad_last_segment: bool) -> List[Audio]:
        sr = audio.sampling_rate
        total_samples = audio.waveform.shape[1]
        seg_samples = max(1, int(round(segment_length * sr)))
        segments: List[Audio] = []

        if total_samples == 0:
            return [Audio(waveform=audio.waveform.clone(), sampling_rate=sr, metadata=audio.metadata.copy())]

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

        return segments

    @workflow.define
    def _wf(xs: Sequence[Audio], segment_length: float, pad_last_segment: bool) -> List[List[Audio]]:
        node = workflow.add(
            _even_segments(segment_length=segment_length, pad_last_segment=pad_last_segment).split(audio=xs),
            name="map_evenly_segment",
        )
        return node.out  # -> List[List[Audio]]

    worker = _select_worker(plugin)
    res = _wf(xs=audios, segment_length=segment_length, pad_last_segment=pad_last_segment)(
        worker=worker, cache_root=cache_dir, **plugin_args
    )
    return list(res.out)


# ---------------------------------------------------------------------
# Concatenate audio objects
# ---------------------------------------------------------------------
def concatenate_audios(audios: List[Audio]) -> Audio:
    """Concatenates all audios in the list, ensuring they have the same sampling rate and shape.

    Args:
        audios: List of Audio objects to concatenate.

    Returns:
        A single Audio object that is the concatenation of all input audios.

    Raises:
        ValueError: If the audios do not all have the same sampling rate or shape.
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
