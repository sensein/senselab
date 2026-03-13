"""This module contains functions for plotting audio-related data."""

from typing import Any, Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import rc_context
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from senselab.audio.data_structures import Audio
from senselab.utils.data_structures import logger

# ---------------------------
# Plot context & scaling
# ---------------------------

_Context = Union[str, float]  # "auto" | "small" | "medium" | "large" | float scale


def _detect_screen_resolution() -> Tuple[int, int]:
    """Best-effort screen resolution detection. Falls back to 1920x1080."""
    # Try TkAgg
    try:
        mgr = plt.get_current_fig_manager()
        win = getattr(mgr, "window", None)
        if win is not None and hasattr(win, "winfo_screenwidth"):
            return int(win.winfo_screenwidth()), int(win.winfo_screenheight())
    except Exception:
        pass
    # Try Qt
    try:
        from PyQt5 import QtWidgets  # type: ignore

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        screen = app.primaryScreen()
        size = screen.size()
        return int(size.width()), int(size.height())
    except Exception:
        pass
    # Fallback
    return 1920, 1080


def _context_scale_from_resolution() -> float:
    """Map screen width → a sensible scale factor."""
    width, _ = _detect_screen_resolution()
    # Simple, readable buckets
    if width <= 1366:
        return 0.9
    if width <= 1920:
        return 1.0
    if width <= 2560:
        return 1.25
    if width <= 3840:
        return 1.5
    return 2.0


def _resolve_scale(context: _Context) -> float:
    if isinstance(context, (int, float)):
        return float(context)
    ctx = str(context).lower()
    if ctx == "auto":
        return _context_scale_from_resolution()
    if ctx in ("paper", "small"):
        return 0.9
    if ctx in ("notebook", "medium"):
        return 1.0
    if ctx in ("talk", "large"):
        return 1.3
    # Default
    return 1.0


def _rc_for_scale(scale: float) -> Dict[str, Any]:
    """Return rcParams tuned for the given scale (seaborn-like)."""
    base = 10.0 * scale
    return {
        "font.size": base,
        "axes.titlesize": base * 1.2,
        "axes.labelsize": base,
        "xtick.labelsize": base * 0.9,
        "ytick.labelsize": base * 0.9,
        "legend.fontsize": base * 0.95,
        "lines.linewidth": 1.25 * scale,
        "grid.linewidth": 0.8 * scale,
        "axes.linewidth": 0.8 * scale,
        "figure.titlesize": base * 1.3,
    }


# ---------------------------
# Helpers
# ---------------------------


def _power_to_db(spectrogram: np.ndarray, ref: float = 1.0, amin: float = 1e-10, top_db: float = 80.0) -> np.ndarray:
    """Converts a power spectrogram (amplitude squared) to decibel (dB) units."""
    S = np.asarray(spectrogram)

    if amin <= 0:
        raise ValueError("amin must be strictly positive")

    if np.issubdtype(S.dtype, np.complexfloating):
        logger.warning(
            "_power_to_db was called on complex input so phase information will be discarded. "
            "To suppress this warning, call power_to_db(np.abs(D)**2) instead.",
            stacklevel=2,
        )
        magnitude = np.abs(S)
    else:
        magnitude = S

    ref_value = ref(magnitude) if callable(ref) else np.abs(ref)
    log_spec: np.ndarray = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise ValueError("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


# ---------------------------
# Public API
# ---------------------------


def plot_waveform(
    audio: Audio,
    title: str = "Waveform",
    fast: bool = False,
    *,
    context: _Context = "auto",
    figsize: Tuple[float, float] | None = None,
) -> Figure:
    """Plot the time-domain waveform of an `Audio` object and return the Figure.

    The plot is automatically scaled for readability using a *context* scale
    (similar to seaborn). Use `fast=True` to lightly decimate the signal for
    quicker rendering on very long waveforms.

    Args:
        audio (Audio):
            Input audio containing `.waveform` (shape `[C, T]`) and `.sampling_rate`.
        title (str, optional):
            Figure title. Defaults to `"Waveform"`.
        fast (bool, optional):
            If `True`, plots a 10× downsampled view for speed. Defaults to `False`.
        context (_Context, optional):
            Size preset or numeric scale. Accepted values:
              * `"auto"` (detect from screen), `"small"`, `"medium"`, `"large"`,
              * or a float scale factor (e.g., `1.25`). Defaults to `"auto"`.
        figsize (tuple[float, float] | None, optional):
            Base `(width, height)` in inches **before** context scaling.
            Defaults to `(12, 2×channels)`.

    Returns:
        matplotlib.figure.Figure: The created figure (also displayed).

    Example:
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> fig = plot_waveform(a1, title="Sample 1", fast=True, context="medium")
        >>> # fig.savefig("waveform.png")  # optional
    """
    waveform = audio.waveform
    sample_rate = audio.sampling_rate

    if fast:
        waveform = waveform[..., ::10]

    num_channels, num_frames = waveform.shape
    time_axis = torch.linspace(0, num_frames / sample_rate, num_frames)

    scale = _resolve_scale(context)
    rc = _rc_for_scale(scale)
    if figsize is None:
        base = (12.0, max(2.0 * num_channels, 2.5))
    else:
        base = figsize
    scaled_size = (base[0] * scale, base[1] * scale)

    with rc_context(rc):
        fig, axes = plt.subplots(num_channels, 1, figsize=scaled_size, sharex=True)
        if num_channels == 1:
            axes = [axes]  # ensure iterable
        for c, ax in enumerate(axes):
            ax.plot(time_axis.numpy(), waveform[c].cpu().numpy())
            ax.set_ylabel(f"Ch {c + 1}")
            ax.grid(True, alpha=0.3)
        fig.suptitle(title)
        axes[-1].set_xlabel("Time [s]")
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show(block=False)
        return fig


def plot_specgram(
    audio: Audio,
    mel_scale: bool = False,
    title: str = "Spectrogram",
    *,
    context: _Context = "auto",
    figsize: Tuple[float, float] | None = None,
    **spect_kwargs: Any,  # noqa: ANN401
) -> Figure:
    """Plot a (mel-)spectrogram for a **mono** `Audio` object and return the Figure.

    Internally calls senselab's torchaudio-based extractors:
    `extract_spectrogram_from_audios` or `extract_mel_spectrogram_from_audios`.
    The function expects a 2D spectrogram `[freq_bins, time_frames]`; multi-channel
    inputs should be downmixed beforehand.

    Args:
        audio (Audio):
            Input **mono** audio. If multi-channel, downmix first.
        mel_scale (bool, optional):
            If `True`, plots a mel spectrogram; otherwise linear frequency. Defaults to `False`.
        title (str, optional):
            Figure title. Defaults to `"Spectrogram"`.
        context (_Context, optional):
            Size preset or numeric scale (`"auto"`, `"small"`, `"medium"`, `"large"`, or float).
            Defaults to `"auto"`.
        figsize (tuple[float, float] | None, optional):
            Base `(width, height)` in inches **before** context scaling. Defaults to `(10, 4)`.
        **spect_kwargs:
            Passed to the underlying extractor (e.g., `n_fft=1024`, `hop_length=256`,
            `n_mels=80`, `win_length=1024`, `f_min=0`, `f_max=None`).

    Returns:
        matplotlib.figure.Figure: The created figure (also displayed).

    Raises:
        ValueError: If spectrogram extraction fails, contains NaNs, or the result is not 2D.

    Example (linear spectrogram):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> fig = plot_specgram(a1, mel_scale=False, n_fft=1024, hop_length=256)
        >>> # fig.savefig("spec.png")

    Example (mel spectrogram):
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> fig = plot_specgram(a1, mel_scale=True, n_mels=80, n_fft=1024, hop_length=256)
    """
    # Extract the spectrogram
    if mel_scale:
        from senselab.audio.tasks.features_extraction.torchaudio import (
            extract_mel_spectrogram_from_audios,
        )

        spectrogram = extract_mel_spectrogram_from_audios([audio], **spect_kwargs)[0]["mel_spectrogram"]
        y_axis_label = "Mel frequency (bins)"
    else:
        from senselab.audio.tasks.features_extraction.torchaudio import (
            extract_spectrogram_from_audios,
        )

        spectrogram = extract_spectrogram_from_audios([audio], **spect_kwargs)[0]["spectrogram"]
        y_axis_label = "Frequency [Hz]"

    # ---- Guard against invalid/short-audio outputs (must be exactly this phrase)
    if not torch.is_tensor(spectrogram):
        raise ValueError("Spectrogram extraction failed")
    if spectrogram.ndim == 0 or spectrogram.numel() == 0:
        raise ValueError("Spectrogram extraction failed")
    if spectrogram.dtype.is_floating_point and torch.isnan(spectrogram).any():
        raise ValueError("Spectrogram extraction failed")

    if spectrogram.dim() != 2:
        raise ValueError(
            "Spectrogram must be a 2D tensor. Got shape: {}".format(spectrogram.shape),
            "Please make sure the input audio is mono.",
        )

    # Determine time and frequency scale
    # num_frames = spectrogram.size(1)
    num_freq_bins = spectrogram.size(0)

    # Time axis in seconds
    duration_sec = audio.waveform.size(-1) / audio.sampling_rate
    time_axis_start = 0.0
    time_axis_end = float(duration_sec)

    # Frequency axis
    if mel_scale:
        freq_start, freq_end = 0.0, float(num_freq_bins - 1)
    else:
        freq_start, freq_end = 0.0, float(audio.sampling_rate / 2)

    scale = _resolve_scale(context)
    rc = _rc_for_scale(scale)
    if figsize is None:
        base = (10.0, 4.0)
    else:
        base = figsize
    scaled_size = (base[0] * scale, base[1] * scale)

    with rc_context(rc):
        fig = plt.figure(figsize=scaled_size)
        plt.imshow(
            _power_to_db(spectrogram.cpu().numpy()),
            aspect="auto",
            origin="lower",
            extent=(time_axis_start, time_axis_end, freq_start, freq_end),
            cmap="viridis",
        )
        plt.colorbar(label="Magnitude (dB)")
        plt.title(title)
        plt.ylabel(y_axis_label)
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.show(block=False)
        return fig


def plot_waveform_and_specgram(
    audio: Audio,
    *,
    title: str = "Waveform + Spectrogram",
    mel_scale: bool = False,
    fast_wave: bool = False,
    context: "_Context" = "auto",
    figsize: Tuple[float, float] | None = None,
    **spect_kwargs: Any,  # noqa: ANN401  # forwarded to spectrogram extraction
) -> Figure:
    """Stacked layout: waveform (top) and **mono** spectrogram (bottom). Returns the Figure.

    The waveform can be drawn in a faster, lightly decimated mode for long signals.
    Spectrogram extraction is delegated to senselab's torchaudio-based utilities
    and requires mono input.

    Args:
        audio (Audio):
            Input audio. **Spectrogram requires mono**; downmix multi-channel first.
        title (str, optional):
            Overall figure title. Defaults to `"Waveform + Spectrogram"`.
        mel_scale (bool, optional):
            If `True`, bottom panel is a mel spectrogram; otherwise linear frequency. Defaults to `False`.
        fast_wave (bool, optional):
            If `True`, waveform panel is downsampled for speed. Defaults to `False`.
        context (_Context, optional):
            Size preset or numeric scale (`"auto"`, `"small"`, `"medium"`, `"large"`, or float).
            Defaults to `"auto"`.
        figsize (tuple[float, float] | None, optional):
            Base `(width, height)` in inches **before** context scaling. Defaults to a balanced height.
        **spect_kwargs:
            Forwarded to the underlying spectrogram extractor (e.g., `n_fft`, `hop_length`, `n_mels`).

    Returns:
        matplotlib.figure.Figure: The created figure (also displayed).

    Raises:
        ValueError: If audio is not mono, or spectrogram extraction fails.

    Example:
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> fig = plot_waveform_and_specgram(
        ...     a1,
        ...     mel_scale=True,
        ...     fast_wave=True,
        ...     context="large",
        ...     n_fft=1024,
        ...     hop_length=256,
        ...     n_mels=80,
        ... )
        >>> # fig.savefig("wave_plus_mel.png")
    """
    # ---- Core timing info from ORIGINAL (non-decimated) data
    sr = audio.sampling_rate
    orig_num_frames = int(audio.waveform.size(-1))
    duration_sec = orig_num_frames / sr
    t0, t1 = 0.0, float(duration_sec)

    # ---- Prepare waveform (optionally decimated for speed)
    waveform = audio.waveform
    if fast_wave:
        waveform = waveform[..., ::10]  # decimate samples
    num_channels, num_frames = waveform.shape
    time_axis = np.linspace(0.0, duration_sec, num_frames, endpoint=False)

    # ---- Guardrail: spectrogram plotting requires mono input
    if audio.waveform.shape[0] != 1:
        raise ValueError("Only mono audio is supported for spectrogram plotting")

    # ---- Spectrogram (2D tensor: [freq_bins, time_frames])
    if mel_scale:
        from senselab.audio.tasks.features_extraction.torchaudio import (
            extract_mel_spectrogram_from_audios,
        )

        spec = extract_mel_spectrogram_from_audios([audio], **spect_kwargs)[0]["mel_spectrogram"]
        ylab = "Mel bins"
        f0, f1 = 0.0, float(spec.size(0) - 1) if torch.is_tensor(spec) and spec.ndim >= 1 else (0.0, 0.0)
        spec_title = "Mel Spectrogram"
    else:
        from senselab.audio.tasks.features_extraction.torchaudio import (
            extract_spectrogram_from_audios,
        )

        spec = extract_spectrogram_from_audios([audio], **spect_kwargs)[0]["spectrogram"]
        ylab = "Frequency [Hz]"
        f0, f1 = 0.0, float(sr / 2)
        spec_title = "Spectrogram"

    # ---- Guardrails for short/invalid outputs (exact phrase expected by tests)
    if not torch.is_tensor(spec):
        raise ValueError("Spectrogram extraction failed")
    if spec.ndim == 0 or spec.numel() == 0:
        raise ValueError("Spectrogram extraction failed")
    if spec.dtype.is_floating_point and torch.isnan(spec).any():
        raise ValueError("Spectrogram extraction failed")

    # We require a 2D (F x T) spectrogram. Anything else → fail (don’t auto-pick channels).
    if spec.ndim != 2:
        raise ValueError("Spectrogram extraction failed")

    # ---- Layout & context
    scale = _resolve_scale(context)
    rc = _rc_for_scale(scale)
    if figsize is None:
        base_h = max(2.0, 0.9 * num_channels) + 4.0  # waveform height + spectrogram
        base = (12.0, base_h)
    else:
        base = figsize
    size = (base[0] * scale, base[1] * scale)

    with rc_context(rc):
        fig, (ax_wav, ax_spec) = plt.subplots(2, 1, figsize=size, sharex=True, gridspec_kw={"height_ratios": [1, 2]})

        # ---- Waveform (top)
        if num_channels == 1:
            ax_wav.plot(time_axis, waveform[0].cpu().numpy())
            ax_wav.set_ylabel("Amp")
        else:
            for c in range(num_channels):
                ax_wav.plot(time_axis, waveform[c].cpu().numpy(), alpha=0.9 if c == 0 else 0.7)
            ax_wav.set_ylabel("Amp (multi-ch)")
        ax_wav.grid(True, alpha=0.3)
        ax_wav.set_title("Waveform")

        # ---- Spectrogram (bottom)
        im = ax_spec.imshow(
            _power_to_db(spec.cpu().numpy()),
            aspect="auto",
            origin="lower",
            extent=(t0, t1, f0, f1),
            cmap="viridis",
        )
        ax_spec.set_ylabel(ylab)
        ax_spec.set_xlabel("Time [s]")
        ax_spec.set_title(spec_title)

        # Keep both axes aligned in time
        ax_wav.set_xlim(t0, t1)
        ax_spec.set_xlim(t0, t1)

        # ---- Horizontal colorbar below the spectrogram
        divider = make_axes_locatable(ax_spec)
        cax = divider.append_axes("bottom", size="5%", pad=0.6)
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("Magnitude (dB)")

        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        plt.show(block=False)
        return fig


def play_audio(audio: Audio) -> None:
    """Play an `Audio` object inline (Jupyter/IPython), supporting 1–2 channels.

    Uses `IPython.display.Audio` to render audio widgets in notebooks. For more
    than two channels, downmix first.

    Args:
        audio (Audio):
            Input audio to play (mono or stereo). Sampling rate is preserved.

    Raises:
        ValueError: If the waveform has more than 2 channels.

    Example:
        >>> from pathlib import Path
        >>> from senselab.audio.data_structures import Audio
        >>> a1 = Audio(filepath=Path("sample1.wav").resolve())
        >>> play_audio(a1)
    """
    from IPython.display import Audio as DisplayAudio
    from IPython.display import display

    waveform = audio.waveform.cpu().numpy()
    sample_rate = audio.sampling_rate

    num_channels = waveform.shape[0]
    if num_channels == 1:
        display(DisplayAudio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(DisplayAudio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels is not supported.")
