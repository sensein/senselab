"""The one definition of how senselab reads and writes audio files.

Every audio read in senselab is :func:`read_audio`, :func:`read_audio_blocks` or :func:`audio_info`;
``Audio``'s lazy load calls them, and ``src/tests/utils/portable_audio_read_guard_test.py`` fails on
any other decoder call. :meth:`senselab.audio.data_structures.Audio.save_to_file` delegates to
:func:`write_audio`. Subprocess-venv workers -- which run in isolated environments where senselab is
not installed -- get this file staged next to their payload by
:func:`senselab.utils.subprocess_venv.stage_portable_audio_io` and import it directly.

**This module must import nothing from senselab, and at module level nothing beyond numpy, soundfile
and the standard library.** torchcodec is imported inside the one function that needs it, so a worker
reading lossless files never needs it. ``src/tests/utils/portable_audio_io_test.py`` enforces this by
AST sweep; a convenient ``from senselab...`` here breaks every worker the next time a venv is rebuilt.

Two decisions live here and nowhere else:

* which subtype a destination gets (:func:`resolve_subtype`), and
* what happens when the samples cannot be represented in it (:func:`apply_range_policy`).

The measurements behind both, and why the default preserves rather than clips:
``specs/20260819-091500-wav-subtype-sweep/design.md``.
"""

from __future__ import annotations

import logging
import os
from typing import IO, Any, Iterator, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf

logger = logging.getLogger("senselab")

LOSSLESS_WAV_SUBTYPE = "FLOAT"
"""The subtype a WAV gets when the caller does not ask for another one."""

RAISE = "raise"
WARN = "warn"
NORMALIZE = "normalize"
OUT_OF_RANGE_POLICIES = (RAISE, WARN, NORMALIZE)

# Preference order per input dtype; the first subtype the destination supports wins. FLAC has no
# float subtype at any depth, so float data resolves to PCM_24 there and the range policy applies.
_PREFERENCE_BY_KIND = {
    "float": ("FLOAT", "DOUBLE", "PCM_32", "PCM_24", "PCM_16"),
    "int16": ("PCM_16", "PCM_24", "PCM_32", "FLOAT"),
    "int32": ("PCM_32", "FLOAT", "PCM_24", "PCM_16"),
}

_FLOAT_SUBTYPES = ("FLOAT", "DOUBLE")


class AudioWriteReport(NamedTuple):
    """What a write did, so a caller can record it rather than assume it."""

    path: str
    format: str
    subtype: str
    peak: float
    out_of_range_fraction: float
    gain: float


def is_float_subtype(subtype: str) -> bool:
    """Whether ``subtype`` stores samples as floats, i.e. carries values beyond ±1 unchanged."""
    return subtype in _FLOAT_SUBTYPES


def format_for(path: Union[str, os.PathLike], format: Optional[str] = None) -> str:
    """Return the libsndfile format name for a destination.

    Args:
        path: The destination path; its extension is used when ``format`` is None.
        format: An explicit format name, case-insensitive.

    Returns:
        An upper-case format name, e.g. ``"WAV"``. Not necessarily one libsndfile can write --
        use :func:`libsndfile_can_write` to find out.
    """
    if format:
        return format.upper()
    return os.fspath(path).rsplit(".", 1)[-1].upper() if "." in os.fspath(path) else ""


def libsndfile_can_write(fmt: str) -> bool:
    """Whether libsndfile can write ``fmt``, and therefore whether a subtype can be chosen."""
    return fmt.upper() in sf.available_formats()


def widest_subtype(fmt: str, dtype: Union[np.dtype, type, str] = np.float32) -> str:
    """Return the subtype of ``fmt`` that loses the least of a ``dtype`` array.

    Args:
        fmt: A libsndfile format name, case-insensitive.
        dtype: The dtype of the array about to be written. Only the dtypes ``soundfile.write``
            accepts are distinguished (float, ``int32``, ``int16``); anything else is float.

    Returns:
        A subtype name ``soundfile.check_format(fmt, subtype)`` accepts. For a lossy container
        with no PCM subtype, the container's own default.

    Raises:
        ValueError: If ``fmt`` is not a format libsndfile can write, or has no default subtype.
    """
    normalized = fmt.upper()
    try:
        available = set(sf.available_subtypes(normalized))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{fmt!r} is not a format soundfile can write") from exc
    if not available:
        raise ValueError(f"{fmt!r} is not a format soundfile can write")

    resolved = np.dtype(dtype)
    if np.issubdtype(resolved, np.integer):
        kind = "int16" if resolved.itemsize <= 2 else "int32"
    else:
        kind = "float"

    for candidate in _PREFERENCE_BY_KIND[kind]:
        if candidate in available:
            return candidate

    default = sf.default_subtype(normalized)
    if default is None:
        raise ValueError(f"{fmt!r} has no default subtype; name one explicitly")
    return default


def resolve_subtype(
    fmt: str,
    dtype: Union[np.dtype, type, str] = np.float32,
    subtype: Optional[str] = None,
) -> str:
    """Decide the subtype for a write, validating an explicit request against the format.

    Args:
        fmt: A libsndfile format name.
        dtype: The dtype of the array about to be written.
        subtype: The caller's request, or None to preserve as much as ``fmt`` allows.

    Returns:
        The subtype to write.

    Raises:
        ValueError: If ``subtype`` is not one ``fmt`` can carry. ``subtype="FLOAT"`` with a FLAC
            destination is the case worth naming: FLAC has no float subtype at any depth, so
            "preserve exactly" and "write FLAC" are mutually exclusive and must not resolve to a
            quiet quantization.
    """
    normalized = fmt.upper()
    if subtype is None:
        return widest_subtype(normalized, dtype)
    requested = subtype.upper()
    if not sf.check_format(normalized, requested):
        raise ValueError(
            f"{normalized} cannot carry subtype {requested}; it supports "
            f"{sorted(sf.available_subtypes(normalized))}. "
            f"{'FLAC is an integer codec and has no float subtype at any depth. ' if normalized == 'FLAC' else ''}"
            "Choose a format that supports it, or a subtype this one has."
        )
    return requested


def subtype_preference(
    fmt: str,
    preferred: Optional[str],
    dtype: Union[np.dtype, type, str] = np.float32,
) -> Optional[str]:
    """Degrade a subtype *preference* to what ``fmt`` can carry.

    The distinction from :func:`resolve_subtype` is the point. An explicit subtype is a demand and
    must raise when the format cannot carry it, because a caller asking for FLOAT and silently
    receiving PCM_16 is the defect this module exists to remove. A preference -- a codec hint
    derived from a source stream, say -- is not a demand: the caller wants as much fidelity as the
    container allows and has no opinion beyond that.

    Args:
        fmt: A libsndfile format name.
        preferred: The subtype the caller would like, or None.
        dtype: The dtype of the array about to be written.

    Returns:
        ``preferred`` when ``fmt`` can carry it, otherwise None, which leaves
        :func:`resolve_subtype` to pick the widest subtype the format has.
    """
    if preferred is None:
        return None
    normalized, requested = fmt.upper(), preferred.upper()
    return requested if sf.check_format(normalized, requested) else None


def out_of_range_fraction(samples: Union[np.ndarray, Sequence[float]]) -> float:
    """Fraction of samples a fixed-point subtype would clip, i.e. those outside ±1.

    Args:
        samples: Samples, in any shape.

    Returns:
        The fraction in ``[0, 1]``; ``0.0`` for an empty array.
    """
    arr = np.asarray(samples)
    if arr.size == 0:
        return 0.0
    return float(np.count_nonzero(np.abs(arr) > 1.0) / arr.size)


def apply_range_policy(
    samples: np.ndarray,
    fmt: str,
    subtype: str,
    out_of_range: str = RAISE,
    destination: str = "",
) -> Tuple[np.ndarray, float, float, float]:
    """Enforce the range policy for a write that is about to happen.

    The one place the "can this data be represented, and if not, what then" question is answered.
    :func:`write_audio` calls it, and so does ``Audio.save_to_file`` on the paths where libsndfile
    is not the writer -- so a float-incapable destination behaves identically either way.

    Args:
        samples: The samples about to be written.
        fmt: The resolved libsndfile format name (used only in messages).
        subtype: The resolved subtype. A float one makes this a no-op.
        out_of_range: ``"raise"`` (default), ``"warn"`` or ``"normalize"``.
        destination: The path, for the message.

    Returns:
        ``(samples, peak, fraction, gain)``. ``samples`` is rescaled only under ``"normalize"``,
        and ``gain`` is the factor applied, so the original is recoverable by dividing by it.

    Raises:
        ValueError: If ``out_of_range`` is not a known policy, or if it is ``"raise"`` and the
            samples do not fit.
    """
    if out_of_range not in OUT_OF_RANGE_POLICIES:
        raise ValueError(f"out_of_range must be one of {OUT_OF_RANGE_POLICIES}, got {out_of_range!r}")

    peak = float(np.abs(samples).max()) if samples.size else 0.0
    if is_float_subtype(subtype) or peak <= 1.0:
        return samples, peak, 0.0, 1.0

    fraction = out_of_range_fraction(samples)
    where = f" writing {destination}" if destination else ""
    detail = (
        f"{100 * fraction:.4g}% of samples exceed ±1 (peak {peak:.6g}) and {fmt}/{subtype} cannot represent them{where}"
    )
    if out_of_range == RAISE:
        raise ValueError(
            detail + ". Write a float-capable format (wav, aiff, w64, caf leave subtype=None and "
            "get FLOAT), or pass out_of_range='normalize' to rescale with the gain reported, or "
            "out_of_range='warn' to accept the clipping."
        )
    if out_of_range == WARN:
        logger.warning("%s; those samples are being clipped.", detail)
        return samples, peak, fraction, 1.0

    gain = 1.0 / peak
    logger.warning("%s; rescaling by %.6g. Divide by that gain to recover the original.", detail, gain)
    return (samples * gain).astype(samples.dtype, copy=False), peak, fraction, gain


def write_audio(
    path: Union[str, os.PathLike],
    samples: np.ndarray,
    sampling_rate: int,
    format: Optional[str] = None,
    subtype: Optional[str] = None,
    out_of_range: str = RAISE,
    channels_first: bool = True,
) -> AudioWriteReport:
    """Write audio, preserving it exactly unless the destination cannot or the caller says otherwise.

    Args:
        path: Destination file. Its extension picks the format when ``format`` is None.
        samples: 1-D samples, or 2-D shaped by ``channels_first``.
        sampling_rate: Sample rate in Hz.
        format: Explicit libsndfile format name, case-insensitive.
        subtype: Explicit subtype. None resolves to the widest the format supports, which is
            ``FLOAT`` for WAV -- so a plain ``.wav`` write preserves float samples exactly.
        out_of_range: What to do when the resolved subtype cannot represent the samples:
            ``"raise"`` (default), ``"warn"`` (clip, loudly) or ``"normalize"`` (rescale, with
            the gain in the returned report).
        channels_first: For 2-D ``samples``, whether the first axis is channels (senselab's
            convention). Ignored for 1-D input.

    Returns:
        An :class:`AudioWriteReport` naming what was written.

    Raises:
        ValueError: If the format cannot be written, the subtype does not fit it, or the samples
            do not fit the subtype under ``out_of_range="raise"``.
    """
    fmt = format_for(path, format)
    if not libsndfile_can_write(fmt):
        raise ValueError(f"{fmt!r} is not a format soundfile can write; got destination {os.fspath(path)!r}")

    array = np.asarray(samples)
    if array.ndim == 2 and channels_first:
        array = array.T

    resolved = resolve_subtype(fmt, array.dtype, subtype)
    array, peak, fraction, gain = apply_range_policy(array, fmt, resolved, out_of_range, os.fspath(path))

    sf.write(os.fspath(path), array, sampling_rate, subtype=resolved, format=fmt)
    return AudioWriteReport(
        path=os.fspath(path),
        format=fmt,
        subtype=resolved,
        peak=peak,
        out_of_range_fraction=fraction,
        gain=gain,
    )


LIBSNDFILE = "libsndfile"
TORCHCODEC = "torchcodec"

EXACT_FRAME_COUNT_FORMATS = frozenset({"FLAC", "WAV", "WAVEX", "W64", "RF64", "AIFF"})
"""libsndfile major formats whose header frame count is exact; :func:`read_audio` reads these with libsndfile."""


def _is_short_data_chunk(line: str) -> bool:
    """Whether a line of libsndfile's ``extra_info`` reports a data chunk longer than the file."""
    return line.lstrip().startswith("data") and "(should be" in line


class AudioInfo(NamedTuple):
    """What a file holds, and which decoder :func:`read_audio` reads it with."""

    sampling_rate: int
    channels: int
    frames: Optional[int]
    format: str
    decoder: str


def _libsndfile_info(source: str) -> Optional["sf._SoundFileInfo"]:
    """``sf.info(source)``, or None where libsndfile cannot open it."""
    try:
        return sf.info(source)
    except (RuntimeError, OSError, TypeError, ValueError):
        return None


def _torchcodec_decoder(source: str) -> Any:  # noqa: ANN401
    """An open torchcodec ``AudioDecoder``; torchcodec is imported here so workers never need it.

    Raises:
        RuntimeError: If torchcodec cannot be imported or cannot open ``source``.
    """
    try:
        from torchcodec.decoders import AudioDecoder
    except (ImportError, RuntimeError, OSError) as exc:
        raise RuntimeError(f"torchcodec is needed to decode {source!r} and cannot be imported: {exc}") from exc
    try:
        return AudioDecoder(source)
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"torchcodec cannot open {source!r}: {exc}") from exc


def audio_info(path: Union[str, os.PathLike]) -> AudioInfo:
    """Describe an audio file without decoding its samples.

    Args:
        path: Source file.

    Returns:
        An :class:`AudioInfo`. ``frames`` is the header's exact count for a libsndfile read, and the
        header's estimate (or None) for a torchcodec one. ``decoder`` names the decoder
        :func:`read_audio` uses for this file.

    Raises:
        RuntimeError: If no decoder can open the file.
    """
    source = os.fspath(path)
    info = _libsndfile_info(source)
    if info is not None and info.format in EXACT_FRAME_COUNT_FORMATS:
        return AudioInfo(int(info.samplerate), int(info.channels), int(info.frames), info.format, LIBSNDFILE)
    try:
        metadata = _torchcodec_decoder(source).metadata
    except RuntimeError:
        if info is None:
            raise
        return AudioInfo(int(info.samplerate), int(info.channels), None, info.format, LIBSNDFILE)
    duration = getattr(metadata, "duration_seconds_from_header", None)
    estimate = None if duration is None else round(duration * metadata.sample_rate)
    codec = str(getattr(metadata, "codec", "") or "").upper()
    return AudioInfo(int(metadata.sample_rate), int(metadata.num_channels), estimate, codec, TORCHCODEC)


def _frame_window(
    start: Optional[int], stop: Optional[int], total: Optional[int], source: str, rate: int
) -> Tuple[int, Optional[int]]:
    """Validate a ``[start, stop)`` frame window against a file of ``total`` frames (None: unknown)."""
    first = 0 if start is None else int(start)
    if first < 0 or (stop is not None and int(stop) < first):
        raise ValueError(f"Invalid frame window [{start}, {stop}) for {source!r}.")
    if total is not None and total > 0 and first >= total:
        raise ValueError(f"Offset ({first / rate} s) exceeds the audio file duration ({total / rate:.2f} s).")
    if stop is None:
        return first, None
    return first, int(stop) if total is None else min(int(stop), total)


def _read_libsndfile(source: str, info: "sf._SoundFileInfo", start: Optional[int], stop: Optional[int]) -> np.ndarray:
    """Decode ``[start, stop)`` with libsndfile as ``(channels, frames)``, refusing a short read."""
    total = int(info.frames)
    short_chunk = next((ln.strip() for ln in info.extra_info.splitlines() if _is_short_data_chunk(ln)), None)
    if short_chunk:
        logger.warning(
            "%s: the header's data chunk is longer than the file (%s); reading the %d frames present.",
            source,
            short_chunk,
            total,
        )
    first, last = _frame_window(start, stop, total, source, int(info.samplerate))
    data, _ = sf.read(source, start=first, stop=last, dtype="float32", always_2d=True)
    expected = (total if last is None else last) - first
    if info.format in EXACT_FRAME_COUNT_FORMATS and data.shape[0] != expected:
        raise RuntimeError(
            f"libsndfile decoded {data.shape[0]} frames of {source!r} where its header promises {expected}; "
            "the file is truncated or corrupt."
        )
    return data.T


def _read_torchcodec(source: str, start: Optional[int], stop: Optional[int]) -> Tuple[np.ndarray, int]:
    """Decode ``[start, stop)`` with torchcodec as ``(channels, frames)``, refusing an empty or non-finite one."""
    decoder = _torchcodec_decoder(source)
    rate = int(decoder.metadata.sample_rate)
    if start is None and stop is None:
        samples = decoder.get_all_samples().data
    elif getattr(decoder.metadata, "begin_stream_seconds", None) == 0:
        first, last = _frame_window(start, stop, None, source, rate)
        if last is None:
            samples = decoder.get_samples_played_in_range(start_seconds=first / rate).data
        else:
            samples = decoder.get_samples_played_in_range(start_seconds=first / rate, stop_seconds=last / rate).data
    else:
        whole = decoder.get_all_samples().data
        first, last = _frame_window(start, stop, int(whole.shape[-1]), source, rate)
        samples = whole[:, first:last]
    array = np.asarray(samples.numpy(), dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise RuntimeError(f"torchcodec decoded no channels from {source!r}.")
    if not np.isfinite(array).all():
        raise RuntimeError(f"torchcodec decoded non-finite samples from {source!r}.")
    return array, rate


def read_audio(
    path: Union[str, os.PathLike],
    always_2d: bool = False,
    channels_first: bool = True,
    *,
    start: Optional[int] = None,
    stop: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Read audio as float32, in senselab's axis order.

    FLAC, WAV, WAVEX, W64, RF64 and AIFF are decoded by libsndfile, and a decode shorter than the
    header's frame count raises. Every other format is decoded by torchcodec; where torchcodec cannot
    decode it, a format libsndfile opens is decoded by libsndfile. :func:`audio_info` names the decoder.

    Args:
        path: Source file.
        always_2d: Return a 2-D array even for mono.
        channels_first: For 2-D output, put channels on the first axis. Ignored when the result
            is 1-D.
        start: First frame to read; None reads from the beginning.
        stop: Frame to stop before; None reads to the end. Clamped to the file's length.

    Returns:
        ``(samples, sampling_rate)``.

    Raises:
        ValueError: If ``start`` lies at or beyond the end of the file, or the window is inverted.
        RuntimeError: If no decoder can read the file, or a lossless decode is short of its header.
    """
    source = os.fspath(path)
    info = _libsndfile_info(source)
    if info is not None and info.format in EXACT_FRAME_COUNT_FORMATS:
        data, rate = _read_libsndfile(source, info, start, stop), int(info.samplerate)
    else:
        try:
            data, rate = _read_torchcodec(source, start, stop)
        except RuntimeError:
            if info is None:
                raise
            data, rate = _read_libsndfile(source, info, start, stop), int(info.samplerate)
    if data.shape[0] == 1 and not always_2d:
        return np.ascontiguousarray(data[0]), rate
    return np.ascontiguousarray(data if channels_first else data.T), rate


def read_audio_blocks(
    source: Union[str, os.PathLike, IO[bytes]], block_seconds: float
) -> Iterator[Tuple[np.ndarray, int]]:
    """Read audio in consecutive blocks without decoding the rest of the source first.

    Args:
        source: A path, or a readable binary file-like object libsndfile can open.
        block_seconds: Duration of each block; the last block may be shorter.

    Yields:
        ``(samples, sampling_rate)``, samples float32 shaped ``(channels, frames)``.
    """
    opened = source if hasattr(source, "read") else os.fspath(source)  # type: ignore[arg-type]
    with sf.SoundFile(opened, "r") as handle:
        rate = int(handle.samplerate)
        block_frames = int(block_seconds * rate)
        while True:
            block = handle.read(frames=block_frames, dtype="float32", always_2d=True)
            if block.shape[0] == 0:
                return
            yield np.ascontiguousarray(block.T), rate
