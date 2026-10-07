"""The one definition of how senselab reads and writes audio files.

Two callers implement nothing of their own: :meth:`senselab.audio.data_structures.Audio.save_to_file`
delegates to :func:`write_audio`, and subprocess-venv workers -- which run in isolated
environments where senselab is not installed -- get this file staged next to their payload by
:func:`senselab.utils.subprocess_venv.stage_portable_audio_io` and import it directly.

**This module must import nothing from senselab, and nothing beyond numpy, soundfile and the
standard library.** ``src/tests/utils/portable_audio_io_test.py`` enforces both by AST sweep; a
convenient ``from senselab...`` here breaks every worker the next time a venv is rebuilt.

Two decisions live here and nowhere else:

* which subtype a destination gets (:func:`resolve_subtype`), and
* what happens when the samples cannot be represented in it (:func:`apply_range_policy`).

The measurements behind both, and why the default preserves rather than clips:
``specs/20260819-091500-wav-subtype-sweep/design.md``.
"""

from __future__ import annotations

import logging
import os
from typing import NamedTuple, Optional, Sequence, Tuple, Union

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


def read_audio(
    path: Union[str, os.PathLike],
    always_2d: bool = False,
    channels_first: bool = True,
) -> Tuple[np.ndarray, int]:
    """Read audio as float32, in senselab's axis order.

    Args:
        path: Source file.
        always_2d: Return a 2-D array even for mono.
        channels_first: For 2-D output, put channels on the first axis. Ignored when the result
            is 1-D.

    Returns:
        ``(samples, sampling_rate)``.
    """
    data, sampling_rate = sf.read(os.fspath(path), dtype="float32", always_2d=always_2d)
    if data.ndim == 2 and channels_first:
        data = data.T
    return data, int(sampling_rate)
