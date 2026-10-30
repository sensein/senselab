"""Torchaudio SQUIM objective and subjective speech-quality estimates, the objective one over fixed windows."""

import math
import traceback
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import torch.nn

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import DeviceType, _select_device_and_dtype, logger
from senselab.utils.dependencies import torchaudio_available

TORCHAUDIO_AVAILABLE = torchaudio_available()
if TORCHAUDIO_AVAILABLE:
    from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE

_objective_model = None
_subjective_model = None


def _get_objective_model() -> torch.nn.Module:
    """Lazily load the SQUIM objective quality model."""
    global _objective_model  # noqa: PLW0603
    if _objective_model is None:
        _objective_model = SQUIM_OBJECTIVE.get_model()
    return _objective_model


def _get_subjective_model() -> torch.nn.Module:
    """Lazily load the SQUIM subjective quality model."""
    global _subjective_model  # noqa: PLW0603
    if _subjective_model is None:
        _subjective_model = SQUIM_SUBJECTIVE.get_model()
    return _subjective_model


_EPS = 1e-9

DEFAULT_WINDOW_S = 10.0
"""The default window length, in seconds."""

DEFAULT_MIN_WINDOW_S = 3.0
"""The default shortest trailing window kept as a window of its own, in seconds."""

DEFAULT_FLOOR_S = 0.0
"""The default shortest span read at all, in seconds: every non-empty span is read."""

OBJECTIVE_HEADS = ("stoi", "pesq", "si_sdr")
"""The objective head's three readings."""


def plan_windows(
    start_s: float,
    end_s: float,
    *,
    window_s: Optional[float],
    min_window_s: float,
    floor_s: float,
    short_tail: Literal["fold", "drop"] = "fold",
) -> List[Tuple[float, float]]:
    """Non-overlapping fixed windows across ``[start_s, end_s]``.

    Full windows of ``window_s`` tile from ``start_s``. A trailing remainder of at least ``min_window_s``
    is a window of its own; a shorter one is folded into the previous window (``short_tail="fold"``)
    or dropped (``short_tail="drop"``). A span shorter than ``min_window_s`` is one window over the whole
    span, and one shorter than ``floor_s`` gets none. With ``window_s=None`` the span is one window,
    or none under the floor.

    Args:
        start_s: The span's start, in seconds.
        end_s: The span's end, in seconds.
        window_s: The window length, or None for one window over the span.
        min_window_s: The shortest trailing window kept, and the length under which a span is one window.
        floor_s: The shortest span given any window.
        short_tail: What happens to a trailing remainder shorter than ``min_window_s``.

    Returns:
        ``(start, end)`` per window, in order.

    Raises:
        ValueError: If ``window_s`` is not positive, ``min_window_s`` exceeds it, or a length is negative.
    """
    if floor_s < 0 or min_window_s < 0:
        raise ValueError(f"floor_s and min_window_s must be >= 0, got floor_s={floor_s}, min_window_s={min_window_s}")
    if window_s is not None and (window_s <= 0 or min_window_s > window_s):
        raise ValueError(f"windows need 0 < window_s and min_window_s <= window_s, got {window_s} and {min_window_s}")
    duration = end_s - start_s
    if duration < floor_s - _EPS:
        return []
    if window_s is None or duration < min_window_s - _EPS:
        return [(start_s, end_s)]
    full = int(math.floor(duration / window_s + _EPS))
    windows = [(start_s + i * window_s, start_s + (i + 1) * window_s) for i in range(full)]
    tail_start = start_s + full * window_s
    tail = end_s - tail_start
    if tail >= min_window_s - _EPS:
        windows.append((tail_start, end_s))
    elif tail > _EPS and short_tail == "fold":
        windows[-1] = (windows[-1][0], end_s)
    return windows


def window_sample_bounds(start_s: float, end_s: float, sampling_rate: int, n_samples: int) -> Tuple[int, int]:
    """The half-open sample range ``[lo, hi)`` a window in seconds selects.

    The start rounds down and the end rounds up, both clamped to the waveform.

    Args:
        start_s: The start, in seconds.
        end_s: The end, in seconds.
        sampling_rate: The waveform's rate.
        n_samples: The waveform's length in samples.

    Returns:
        ``(lo, hi)`` with ``0 <= lo <= hi <= n_samples``.
    """
    lo = min(n_samples, max(0, int(start_s * sampling_rate)))
    hi = max(lo, min(n_samples, math.ceil(end_s * sampling_rate)))
    return lo, hi


def _span_of(audio: Audio, start_s: Optional[float], end_s: Optional[float]) -> Tuple[float, float]:
    duration = audio.waveform.shape[-1] / audio.sampling_rate
    start = 0.0 if start_s is None else float(start_s)
    end = duration if end_s is None else float(end_s)
    if start < 0 or end > duration + _EPS or start >= end:
        raise ValueError(f"span must satisfy 0 <= start_s < end_s <= duration ({duration} s), got [{start}, {end}]")
    return start, min(end, duration)


def _median(values: List[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return float(np.median(finite)) if finite else float("nan")


def extract_objective_quality_features_from_audios(
    audios: List[Audio],
    window_s: Optional[float] = DEFAULT_WINDOW_S,
    min_window_s: float = DEFAULT_MIN_WINDOW_S,
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    floor_s: float = DEFAULT_FLOOR_S,
    device: Optional[DeviceType] = None,
) -> List[Dict[str, Any]]:
    """Objective SQUIM quality (STOI, PESQ, SI-SDR) over fixed windows of a span of each audio.

    The span ``[start_s, end_s]`` is cut into windows by :func:`plan_windows` (a trailing remainder
    shorter than ``min_window_s`` folds into the previous window), each window is cut by sample index
    and read by the SQUIM objective model, and the reported values are the medians over the windows.
    With ``window_s=None`` the span is read in one pass.

    Every non-empty span is read by default. Values from spans under about 0.5 s vary more than those
    from 10 s windows (sensein/senselab#614); set ``floor_s`` for a cut-off, under which a span's values
    are NaN and ``windows_n`` is 0. An empty span, including an empty audio, raises ``ValueError``.

    The SQUIM model only supports mono audio at 16000 Hz.

    Args:
        audios: Mono 16 kHz audios.
        window_s: The window length in seconds, or None for one pass over the span.
        min_window_s: A trailing remainder at least this long is its own window; a shorter one folds
            into the previous window. A span shorter than this is one window.
        start_s: The span's start in seconds; 0 when None.
        end_s: The span's end in seconds; the audio's duration when None.
        floor_s: The shortest span read, in seconds; shorter spans return NaN. 0 reads every span.
        device: The device to run the model on.

    Returns:
        One dict per audio with ``stoi``, ``pesq`` and ``si_sdr`` (medians over the windows, or the
        single-pass values), ``windows`` (one ``{start_s, end_s, stoi, pesq, si_sdr}`` per window, times
        in seconds of the audio), ``windows_n`` and ``span_s`` (the span's length in seconds).

    Raises:
        ModuleNotFoundError: If torchaudio is not installed.
        ValueError: If an audio is not mono 16 kHz, a span is outside ``0 <= start_s < end_s <= duration``
            (an empty span included), or the window lengths are inconsistent.
        RuntimeError: If the model fails on a window, as it does on one shorter than its 64-sample kernel.
    """
    if not TORCHAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`torchaudio` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    device, _ = _select_device_and_dtype(user_preference=device, compatible_devices=[DeviceType.CUDA, DeviceType.CPU])
    if any(audio.waveform.shape[0] != 1 for audio in audios):
        raise ValueError("Only mono audio is supported by Torchaudio-Squim model.")

    if any(audio.sampling_rate != 16000 for audio in audios):
        raise ValueError("Only 16000 Hz sampling rate is supported by Torchaudio-Squim model.")

    plans = []
    for audio in audios:
        span_start, span_end = _span_of(audio, start_s, end_s)
        windows = plan_windows(
            span_start, span_end, window_s=window_s, min_window_s=min_window_s, floor_s=floor_s, short_tail="fold"
        )
        plans.append((span_start, span_end, windows))

    features: List[Dict[str, Any]] = []
    model = _get_objective_model().to(device.value) if any(windows for _, _, windows in plans) else None

    for audio, (span_start, span_end, windows) in zip(audios, plans):
        waveform = audio.waveform
        n_samples = int(waveform.shape[-1])
        readings: List[Dict[str, Any]] = []
        for a, b in windows:
            lo, hi = window_sample_bounds(a, b, audio.sampling_rate, n_samples)
            assert model is not None
            stoi, pesq, si_sdr = model(waveform[:, lo:hi].to(device.value))
            readings.append(
                {
                    "start_s": a,
                    "end_s": b,
                    "stoi": stoi.cpu().item(),
                    "pesq": pesq.cpu().item(),
                    "si_sdr": si_sdr.cpu().item(),
                }
            )
        result: Dict[str, Any] = {head: _median([float(r[head]) for r in readings]) for head in OBJECTIVE_HEADS}
        result["windows"] = readings
        result["windows_n"] = len(readings)
        result["span_s"] = span_end - span_start
        features.append(result)

    return features


def extract_subjective_quality_features_from_audios(
    audios: List["Audio"], non_matching_references: List["Audio"]
) -> List[Dict[str, Any]]:
    """Extracts subjective audio features from a list of Audio objects.

    Currently, Torchaudio-Squim model only supports mono audio at 16000 Hz sampling rate.

    Args:
        audios (List[Audio]): List of Audio objects.
        non_matching_references (List[Audio]): Reference Audio objects for the subjective model.

    Returns:
        List[Dict[str, Any]]: List of dictionaries, each containing extracted features for an audio input.
    """
    if not TORCHAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`torchaudio` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    # Check if any audio is not mono
    if any(audio.waveform.shape[0] != 1 for audio in audios) or any(
        ref.waveform.shape[0] != 1 for ref in non_matching_references
    ):
        raise ValueError("Only mono audio is supported by Torchaudio-Squim model.")

    # Check if any audio has a sampling rate other than 16000 Hz
    if any(audio.sampling_rate != 16000 for audio in audios) or any(
        ref.sampling_rate != 16000 for ref in non_matching_references
    ):
        raise ValueError("Only 16000 Hz sampling rate is supported by Torchaudio-Squim model.")

    features: List[Dict[str, Any]] = []

    for i, audio in enumerate(audios):
        audio_features = {}
        try:
            mos = _get_subjective_model()(audio.waveform, non_matching_references[i].waveform)
            audio_features["mos"] = mos.item()
        except RuntimeError as e:
            audio_features["mos"] = np.nan
            logger.error(f"RuntimeException encountered: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")

        features.append(audio_features)

    return features
