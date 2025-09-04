"""Audio preprocessing utilities implemented with Pydra workflows.

This module provides parallelizable preprocessing primitives operating on in-memory
`senselab.audio.data_structures.Audio` objects. Each function exposes a simple API
and runs as a small Pydra workflow so that batches can be processed concurrently
(using the selected plugin).

Notes:
    - These functions operate on `Audio` objects (already loaded in memory); they
      do not read or write files.
    - For execution backends, see Pydra v1 plugins:
      https://nipype.github.io/pydra/

Common plugins:
    * "serial" / "debug" — sequential execution (default).
    * "cf" — concurrent futures for parallel execution.
    * "slurm" — submit tasks to a SLURM cluster.

Use `plugin_args` to tune the backend, e.g. `{"n_procs": 8}` for "cf".
"""

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
    """Resample a batch of `Audio` objects to a target sampling rate via Pydra.

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
        cache_dir (str | os.PathLike, optional):
            Pydra cache directory. If ``None``, Pydra uses its default.
        plugin (str, optional):
            Pydra execution plugin. Common options:
              * ``"serial"`` / ``"debug"``: sequential (default).
              * ``"cf"``: concurrent futures (parallel).
              * ``"slurm"``: SLURM scheduler.
        plugin_args (dict, optional):
            Extra options for the chosen plugin (e.g., ``{"n_procs": 8}`` for "cf").

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
        >>> out = resample_audios([a1], resample_rate=16000, plugin="cf")
        >>> out[0].sampling_rate
        16000
    """
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
    """Downmix each `Audio` to mono by averaging channels (parallel via Pydra).

    Channel averaging preserves the original sampling rate and copies metadata.

    Args:
        audios (list[Audio]):
            Input `Audio` objects. Works for mono and multi-channel inputs.
        cache_dir (str | os.PathLike, optional):
            Pydra cache directory. If ``None``, Pydra uses its default.
        plugin (str, optional):
            Pydra execution plugin. Common options:
              * ``"serial"`` / ``"debug"``: sequential (default).
              * ``"cf"``: concurrent futures (parallel).
              * ``"slurm"``: SLURM scheduler.
        plugin_args (dict, optional):
            Extra options for the chosen plugin (e.g., ``{"n_procs": 8}``).

    Returns:
        list[Audio]: Mono `Audio` objects (shape [1, T]).

    Example:
        >>> from senselab.audio.tasks.preprocessing import downmix_audios_to_mono
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> out = downmix_audios_to_mono([a1], plugin="cf")
        >>> out[0].waveform.shape[0]
        1
    """
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
    """Select a specific channel from each `Audio` (parallel via Pydra).

    Args:
        audios (list[Audio]):
            Input `Audio` objects (mono or multi-channel).
        channel_index (int):
            Zero-based channel index to extract.
        cache_dir (str | os.PathLike, optional):
            Pydra cache directory.
        plugin (str, optional):
            Pydra execution plugin. Common options:
              * ``"serial"`` / ``"debug"``: sequential (default).
              * ``"cf"``: concurrent futures (parallel).
              * ``"slurm"``: SLURM scheduler.
        plugin_args (dict, optional):
            Extra options for the chosen plugin.

    Returns:
        list[Audio]: `Audio` objects containing only the selected channel (shape [1, T]).

    Raises:
        ValueError:
            If `channel_index` is out of range for any input.

    Example:
        >>> from senselab.audio.tasks.preprocessing import select_channel_from_audios
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> ch0 = select_channel_from_audios([a1], channel_index=0, plugin="cf")
        >>> ch0[0].waveform.shape[0]
        1
    """
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
# Chunk Audios (by single [start, end]) via Pydra
# ---------------------------------------------------------------------
def chunk_audios(
    data: Sequence[Tuple[Audio, Tuple[float, float]]],
    cache_dir: Optional[Union[str, "os.PathLike"]] = None,
    plugin: str = "debug",
    plugin_args: Optional[Dict[str, Any]] = None,
) -> List[Audio]:
    """Extract a single time chunk from each `Audio` via a Pydra workflow.

    Each pair provides the `(start, end)` times (in seconds) of the desired chunk.

    Args:
        data (Sequence[tuple[Audio, tuple[float, float]]]):
            Sequence like ``[(audio, (start, end)), ...]``.
        cache_dir (str | os.PathLike, optional):
            Pydra cache directory.
        plugin (str, optional):
            Pydra execution plugin. Common options:
              * ``"serial"`` / ``"debug"``: sequential (default).
              * ``"cf"``: concurrent futures (parallel).
              * ``"slurm"``: SLURM scheduler.
        plugin_args (dict, optional):
            Extra options for the chosen plugin.

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
        >>> chunks = chunk_audios(pairs, plugin="cf", plugin_args={"n_procs": 8})
        >>> len(chunks)
        2
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


# ---------------------------------------------------------------------
# Extract multiple segments per audio
# ---------------------------------------------------------------------
def extract_segments(
    data: List[Tuple[Audio, List[Tuple[float, float]]]],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    plugin: str = "debug",
    plugin_args: Optional[Dict[str, Any]] = None,
) -> List[List[Audio]]:
    """Extract multiple time segments per `Audio` (maps over inputs via Pydra).

    For each input `Audio`, a list of `(start, end)` segments (in seconds) is provided,
    and the function returns a list of chunked `Audio` objects.

    Args:
        data (list[tuple[Audio, list[tuple[float, float]]]]):
            Items like ``(audio, [(s1, e1), (s2, e2), ...])``.
        cache_dir (str | os.PathLike, optional):
            Pydra cache directory.
        plugin (str, optional):
            Pydra execution plugin. Common options:
              * ``"serial"`` / ``"debug"``: sequential (default).
              * ``"cf"``: concurrent futures (parallel).
              * ``"slurm"``: SLURM scheduler.
        plugin_args (dict, optional):
            Extra options for the chosen plugin.

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
        >>> segmented = extract_segments(specs, plugin="cf")
        >>> len(segmented[0])
        2
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
    """Right-pad each audio to a desired number of samples (parallel via Pydra).

    Pads with zeros to reach `desired_samples`. If an audio is already longer or equal,
    it is returned unchanged.

    Args:
        audios (list[Audio]):
            Input `Audio` objects.
        desired_samples (int):
            Target length in samples (per channel).
        cache_dir (str | os.PathLike, optional):
            Pydra cache directory.
        plugin (str, optional):
            Pydra execution plugin. Common options:
              * ``"serial"`` / ``"debug"``: sequential (default).
              * ``"cf"``: concurrent futures (parallel).
              * ``"slurm"``: SLURM scheduler.
        plugin_args (dict, optional):
            Extra options for the chosen plugin.

    Returns:
        list[Audio]: Padded `Audio` objects.

    Example:
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.audio.tasks.preprocessing import pad_audios
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> padded = pad_audios([a1], desired_samples=16000, plugin="cf")
        >>> padded[0].waveform.shape[1] >= 16000
        True
    """
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
        cache_dir (str | os.PathLike, optional):
            Pydra cache directory.
        plugin (str, optional):
            Pydra execution plugin. Common options:
              * ``"serial"`` / ``"debug"``: sequential (default).
              * ``"cf"``: concurrent futures (parallel).
              * ``"slurm"``: SLURM scheduler.
        plugin_args (dict, optional):
            Extra options for the chosen plugin.

    Returns:
        list[list[Audio]]: For each input, a list of fixed-length segments.

    Example:
        >>> from senselab.audio.data_structures import Audio
        >>> from senselab.audio.tasks.preprocessing import evenly_segment_audios
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> segs = evenly_segment_audios([a1], segment_length=1.0, pad_last_segment=True, plugin="cf")
        >>> len(segs[0]) >= 1
        True
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
