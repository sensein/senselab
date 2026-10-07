"""Background speech inside an airway task, read off what speech enhancement removed.

PREPROCESS writes a ``residual`` stream (the plain stream less the gain-fitted, lag-aligned enhanced
stream) and classifies it with YAMNet. Two readings find another voice inside the task's extent:

- a residual window YAMNet hears as speech, where the enhancer kept the foreground (the enhanced
  stream well over the residual), the residual stands well over its own floor, and the residual's
  envelope does not follow the enhanced one (a residual that tracks the foreground is the enhancer's
  leakage of the participant's own sound, not another source);
- a harmonic run in the residual away from the participant's own events, for a voice too faint or
  band-limited for YAMNet to name: what the enhancer removed is periodic there, as a voice is and
  breath turbulence or room noise is not.

The plain stream's own speech score is recorded beside them, as context only.

Every parameter is in ``data/background_speech.yaml``. See ``specs/20261006-cough-pattern/design.md``.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from senselab.audio.tasks.classification.label_scores import label_scores
from senselab.audio.workflows.triage.nodes.common import find_measurement
from senselab.utils.prov_store import ProvStore

BACKGROUND_SPEECH_PATH = Path(__file__).parent / "data" / "background_speech.yaml"
PLAIN_STREAM = "plain"
RESIDUAL_STREAM = "residual"
ENHANCED_STREAM = "enhanced"
RESIDUAL_YAMNET = "residual_yamnet_scores"
PLAIN_YAMNET = "yamnet_scores"

Signal = tuple[np.ndarray, int]


@dataclass(frozen=True)
class BackgroundSpeechParameters:
    """The parameters of ``data/background_speech.yaml``.

    Attributes:
        speech_labels: The YAMNet labels that are speech.
        speech_min: The highest speech-label score a residual window reaches to hear speech.
        keep_db: How far the enhanced stream stands over the residual in a window the enhancer kept.
        residual_rise_db: How far over its own floor the residual stands where the enhancer removed something.
        floor_percentile: The percentile of the residual's window levels taken as its floor.
        envelope_frame_s: The frame the residual and enhanced envelopes are read in.
        leakage_corr_min: The envelope correlation at or above which a residual window is leakage.
        windows_min: The residual windows that must read as background speech.
        harmonic_band_hz: The band the residual's harmonicity is read in.
        f0_range_hz: The fundamental-frequency range its autocorrelation peak is searched over.
        frame_s: The harmonicity frame.
        frame_hop_s: Its hop.
        harmonicity_min: The normalised autocorrelation peak a harmonic frame reaches.
        harmonic_rise_db: How far over the residual's own frame floor a harmonic frame stands.
        event_pad_s: How far either side of the participant's own events harmonic frames are set aside.
        run_gap_s: The longest gap inside one harmonic run.
        run_min_s: The shortest harmonic run.
        run_voiced_min_s: The harmonic frames a run holds, in seconds.
        thinned_db: The range of plain over enhanced level over a run in which the enhancer removed part of
            the sound and kept the foreground.
        pad_s: How far either side of the task extent is read.
    """

    speech_labels: tuple[str, ...]
    speech_min: float
    keep_db: float
    residual_rise_db: float
    floor_percentile: float
    envelope_frame_s: float
    leakage_corr_min: float
    windows_min: int
    harmonic_band_hz: tuple[float, float]
    f0_range_hz: tuple[float, float]
    frame_s: float
    frame_hop_s: float
    harmonicity_min: float
    harmonic_rise_db: float
    event_pad_s: float
    run_gap_s: float
    run_min_s: float
    run_voiced_min_s: float
    thinned_db: tuple[float, float]
    pad_s: float


@functools.cache
def background_speech_parameters() -> BackgroundSpeechParameters:
    """The parameters of ``data/background_speech.yaml``.

    Returns:
        The parameters.
    """
    held = dict(yaml.safe_load(BACKGROUND_SPEECH_PATH.read_text()) or {})
    labels = tuple(str(label) for label in held.pop("speech_labels"))
    band, f0, thinned = held.pop("harmonic_band_hz"), held.pop("f0_range_hz"), held.pop("thinned_db")
    return BackgroundSpeechParameters(
        speech_labels=labels,
        windows_min=int(held.pop("windows_min")),
        harmonic_band_hz=(float(band[0]), float(band[1])),
        f0_range_hz=(float(f0[0]), float(f0[1])),
        thinned_db=(float(thinned[0]), float(thinned[1])),
        **{key: float(value) for key, value in held.items()},
    )


@dataclass(frozen=True)
class BackgroundSpeech:
    """What :func:`measure_background_speech` read over a task extent.

    Attributes:
        windows: Each residual window YAMNet hears as speech where the enhancer kept the foreground:
            ``(start, end, speech score, residual dB over its floor, enhanced dB over the residual,
            envelope correlation)``.
        runs: Each harmonic run in the residual off the participant's events: ``(start, end, harmonic
            seconds, plain dB over enhanced)``.
        residual_floor_db: The residual's floor, in dB.
        plain_speech_max: The highest speech-label score on the plain stream inside the extent, or None.
        extent: The extent read, padded.
        parameters: The parameters the reading was judged on.
    """

    windows: tuple[tuple[float, ...], ...] = ()
    runs: tuple[tuple[float, ...], ...] = ()
    residual_floor_db: float | None = None
    plain_speech_max: float | None = None
    extent: tuple[float, float] | None = None
    parameters: BackgroundSpeechParameters | None = None

    @property
    def speech_windows(self) -> tuple[tuple[float, ...], ...]:
        """The residual windows that are not the participant's own sound leaking."""
        p = self.parameters or background_speech_parameters()
        return tuple(w for w in self.windows if w[5] < p.leakage_corr_min)

    @property
    def voice_runs(self) -> tuple[tuple[float, ...], ...]:
        """The harmonic residual runs over which the enhancer kept the foreground."""
        p = self.parameters or background_speech_parameters()
        return tuple(r for r in self.runs if p.thinned_db[0] <= r[3] <= p.thinned_db[1])

    @property
    def heard(self) -> bool:
        """Whether another voice was read inside the extent."""
        p = self.parameters or background_speech_parameters()
        return len(self.speech_windows) >= p.windows_min or bool(self.voice_runs)

    def record(self) -> dict[str, Any]:
        """The reading, as JSON-ready values.

        Returns:
            The fields, keyed by name.
        """
        return {
            "heard": self.heard,
            "speech_windows": [list(window) for window in self.speech_windows],
            "voice_runs": [list(run) for run in self.voice_runs],
            "windows": [list(window) for window in self.windows],
            "runs": [list(run) for run in self.runs],
            "residual_floor_db": self.residual_floor_db,
            "plain_speech_max": self.plain_speech_max,
            "extent": list(self.extent) if self.extent is not None else None,
        }


def _classifier_windows(store: ProvStore, run_dir: Path, name: str) -> list[dict[str, Any]] | None:
    measurement = find_measurement(store, name)
    path = run_dir / str(measurement.attributes.get("path") or "") if measurement is not None else None
    if path is None or not path.is_file():
        return None
    return list(json.loads(path.read_text()))


def _speech(window: dict[str, Any], labels: Sequence[str]) -> float:
    scores = {key: float(value) for pair in label_scores(window) for key, value in pair.items()}
    return max((scores.get(label, 0.0) for label in labels), default=0.0)


def _named_stream(store: ProvStore, run_dir: Path, name: str) -> Signal | None:
    import soundfile  # noqa: PLC0415 -- decoding is only needed where a reading runs

    stream = next(
        (s for s in store.entities("stream") if s.attributes.get("name") == name and not store.is_invalidated(s.id)),
        None,
    )
    path = run_dir / str(stream.attributes.get("path") or "") if stream is not None else None
    if path is None or not path.is_file():
        return None
    samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
    return samples.mean(axis=1), int(rate)


def _piece(signal: Signal, start: float, end: float) -> np.ndarray:
    samples, rate = signal
    return samples[max(0, int(start * rate)) : max(0, int(end * rate))].astype(np.float64)


def _level_db(signal: Signal, start: float, end: float) -> float:
    piece = _piece(signal, start, end)
    return float(10.0 * np.log10(np.mean(piece**2) + 1e-12)) if len(piece) else -120.0


def _envelope_corr(a: Signal, b: Signal, start: float, end: float, frame_s: float) -> float:
    n = max(1, int(frame_s * a[1]))
    x, y = _piece(a, start, end), _piece(b, start, end)
    k = min(len(x), len(y)) // n
    if k < 3:
        return 1.0
    ex = 10 * np.log10((x[: k * n].reshape(k, n) ** 2).mean(axis=1) + 1e-12)
    ey = 10 * np.log10((y[: k * n].reshape(k, n) ** 2).mean(axis=1) + 1e-12)
    if ex.std() < 1e-9 or ey.std() < 1e-9:
        return 1.0
    return float(np.corrcoef(ex, ey)[0, 1])


def residual_harmonicity(residual: Signal, p: BackgroundSpeechParameters) -> tuple[np.ndarray, np.ndarray]:
    """Which residual frames are harmonic: periodic in the fundamental range and over the residual's floor.

    Args:
        residual: The residual samples and their rate.
        p: The parameters.

    Returns:
        Each frame's start time, and whether it is harmonic.
    """
    from scipy.signal import butter, sosfiltfilt  # noqa: PLC0415

    samples, rate = residual
    frame, hop = int(p.frame_s * rate), int(p.frame_hop_s * rate)
    if len(samples) < max(frame, 32):
        return np.zeros(0), np.zeros(0, dtype=bool)
    high = min(p.harmonic_band_hz[1], 0.45 * rate)
    sos = butter(4, (p.harmonic_band_hz[0], high), btype="band", fs=rate, output="sos")
    x = sosfiltfilt(sos, samples.astype(np.float64))
    frames = np.lib.stride_tricks.sliding_window_view(x, frame)[::hop] * np.hanning(frame)
    spectrum = np.fft.rfft(frames, n=2 * frame, axis=1)
    ac = np.fft.irfft(np.abs(spectrum) ** 2, axis=1)[:, :frame]
    lo, hi = int(rate / p.f0_range_hz[1]), int(rate / p.f0_range_hz[0])
    peak = ac[:, lo:hi].max(axis=1) / (ac[:, 0] + 1e-12)
    level = 10 * np.log10(np.mean(frames**2, axis=1) + 1e-12)
    floor = float(np.percentile(level, 10))
    return np.arange(len(frames)) * hop / rate, (peak >= p.harmonicity_min) & (level >= floor + p.harmonic_rise_db)


def _voiced_runs(
    times: np.ndarray,
    voiced: np.ndarray,
    *,
    low: float,
    high: float,
    events: Sequence[tuple[float, float]],
    p: BackgroundSpeechParameters,
) -> list[tuple[float, float, float]]:
    keep = voiced & (times >= low) & (times <= high)
    for start, end in events:
        keep &= ~((times >= start - p.event_pad_s) & (times <= end + p.event_pad_s))
    hop = float(np.median(np.diff(times))) if len(times) > 1 else 0.01
    runs: list[tuple[float, float, float]] = []
    for t in times[keep]:
        if runs and t - runs[-1][1] <= p.run_gap_s:
            runs[-1] = (runs[-1][0], float(t), runs[-1][2] + hop)
        else:
            runs.append((float(t), float(t), hop))
    return [run for run in runs if run[1] - run[0] >= p.run_min_s and run[2] >= p.run_voiced_min_s]


def measure_background_speech(
    residual_windows: Sequence[dict[str, Any]],
    *,
    plain: Signal,
    enhanced: Signal,
    residual: Signal,
    extent: tuple[float, float],
    events: Sequence[tuple[float, float]] = (),
    plain_windows: Sequence[dict[str, Any]] | None = None,
    parameters: BackgroundSpeechParameters | None = None,
) -> BackgroundSpeech:
    """Another voice inside a task extent, read off what the enhancer removed.

    Args:
        residual_windows: The residual stream's YAMNet windows.
        plain: The plain samples and their rate.
        enhanced: The enhanced samples and their rate.
        residual: The residual samples and their rate.
        extent: The task extent.
        events: The participant's own events (coughs, breaths), whose harmonic frames are set aside.
        plain_windows: The plain stream's YAMNet windows, read for context; None reads none.
        parameters: The parameters; ``data/background_speech.yaml`` when None.

    Returns:
        The reading.
    """
    p = parameters or background_speech_parameters()
    low, high = extent[0] - p.pad_s, extent[1] + p.pad_s
    spans = [(float(w.get("start", 0.0)), float(w.get("end", 0.0))) for w in residual_windows]
    levels = [_level_db(residual, a, b) for a, b in spans]
    floor = float(np.percentile(levels, p.floor_percentile)) if levels else None
    found = []
    for window, (a, b), level in zip(residual_windows, spans, levels):
        if b <= low or a >= high or floor is None:
            continue
        score, kept = _speech(window, p.speech_labels), _level_db(enhanced, a, b) - level
        if score >= p.speech_min and kept >= p.keep_db and level - floor >= p.residual_rise_db:
            corr = _envelope_corr(residual, enhanced, a, b, p.envelope_frame_s)
            found.append(
                (round(a, 3), round(b, 3), round(score, 3), round(level - floor, 2), round(kept, 2), round(corr, 3))
            )
    runs = []
    for a, b, voiced_s in _voiced_runs(*residual_harmonicity(residual, p), low=low, high=high, events=events, p=p):
        thinned = _level_db(plain, a, b) - _level_db(enhanced, a, b)
        runs.append((round(a, 3), round(b, 3), round(voiced_s, 3), round(thinned, 2)))
    inside = [
        _speech(w, p.speech_labels)
        for w in plain_windows or ()
        if float(w.get("end", 0.0)) > low and float(w.get("start", 0.0)) < high
    ]
    return BackgroundSpeech(
        windows=tuple(found),
        runs=tuple(runs),
        residual_floor_db=round(floor, 2) if floor is not None else None,
        plain_speech_max=round(max(inside), 3) if inside else None,
        extent=(round(low, 3), round(high, 3)),
        parameters=p,
    )


def background_speech_of(
    store: ProvStore, run_dir: Path, extent: tuple[float, float], events: Sequence[tuple[float, float]] = ()
) -> BackgroundSpeech | None:
    """Another voice inside a task extent, read off the stored streams and their classifications.

    Args:
        store: The provenance store, read for the plain, enhanced and residual streams, the residual
            and plain YAMNet windows.
        run_dir: The run directory their paths are relative to.
        extent: The task extent.
        events: The participant's own events inside it.

    Returns:
        The reading, or None where the residual's windows or any of the three streams is not stored.
    """
    windows = _classifier_windows(store, run_dir, RESIDUAL_YAMNET)
    streams = [_named_stream(store, run_dir, name) for name in (PLAIN_STREAM, ENHANCED_STREAM, RESIDUAL_STREAM)]
    plain, enhanced, residual = streams
    if windows is None or plain is None or enhanced is None or residual is None:
        return None
    p = background_speech_parameters()
    return measure_background_speech(
        windows,
        plain=plain,
        enhanced=enhanced,
        residual=residual,
        extent=extent,
        events=events,
        plain_windows=_classifier_windows(store, run_dir, PLAIN_YAMNET),
        parameters=p,
    )
