"""Tests for background speech read off what speech enhancement removed (``background_speech.py``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from senselab.audio.workflows.triage.background_speech import background_speech_of, measure_background_speech
from senselab.utils.prov_store import ProvStore

RATE = 1000
DURATION = 10.0


def _windows(scores: list[float]) -> list[dict]:
    return [
        {"start": k * 0.5, "end": k * 0.5 + 1.0, "label_scores": [{"Speech": score}, {"Cough": 0.1}]}
        for k, score in enumerate(scores)
    ]


def _signal(levels_db: list[tuple[float, float, float]], base_db: float, seed: int) -> np.ndarray:
    """A noise signal at ``base_db`` with ``(start, end, level_db)`` stretches raised."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(0.0, 1.0, int(DURATION * RATE)) * 10 ** (base_db / 20)
    for start, end, level in levels_db:
        a, b = int(start * RATE), int(end * RATE)
        samples[a:b] = rng.normal(0.0, 1.0, b - a) * 10 ** (level / 20)
    return samples


def _speech_at(window: int) -> list[float]:
    scores = [0.0] * 19
    scores[window] = 0.9
    return scores


def _modulated(start: float, end: float, level_db: float, seed: int) -> np.ndarray:
    """A noise burst whose envelope moves at 4 Hz, independent of anything else in the file."""
    rng = np.random.default_rng(seed)
    samples = rng.normal(0.0, 1.0, int(DURATION * RATE)) * 10 ** (-90 / 20)
    t = np.arange(int((end - start) * RATE)) / RATE
    a = int(start * RATE)
    samples[a : a + len(t)] = rng.normal(0.0, 1.0, len(t)) * 10 ** (level_db / 20) * (1.2 + np.sin(2 * np.pi * 4 * t))
    return samples


def test_an_independent_voice_the_enhancer_removed_is_heard() -> None:
    """6ca9935e: a voice the enhancer took out while keeping the participant is background speech."""
    residual = _modulated(5.0, 6.0, -40.0, 1)
    enhanced = _signal([(5.0, 6.0, -20.0)], -60.0, 2)
    reading = measure_background_speech(
        _windows(_speech_at(10)),
        plain=(enhanced + residual, RATE),
        enhanced=(enhanced, RATE),
        residual=(residual, RATE),
        extent=(2.0, 8.0),
    )
    assert reading.heard
    assert reading.speech_windows[0][0] == 5.0


def test_the_participants_own_cough_leaking_is_not_background_speech() -> None:
    """56baa32a: a residual whose envelope follows the enhanced cough is the cough leaking."""
    enhanced = _modulated(5.0, 6.0, -20.0, 3)
    residual = enhanced * 0.05
    reading = measure_background_speech(
        _windows(_speech_at(10)),
        plain=(enhanced + residual, RATE),
        enhanced=(enhanced, RATE),
        residual=(residual, RATE),
        extent=(2.0, 8.0),
    )
    assert reading.windows and not reading.speech_windows
    assert not reading.heard


def test_a_cough_the_enhancer_removed_whole_is_not_background_speech() -> None:
    """53dddba6: the enhancer took the coughs themselves; nothing of the foreground was kept."""
    residual = _signal([(5.0, 6.0, -20.0)], -80.0, 4)
    enhanced = _signal([], -90.0, 5)
    reading = measure_background_speech(
        _windows(_speech_at(10)),
        plain=(residual, RATE),
        enhanced=(enhanced, RATE),
        residual=(residual, RATE),
        extent=(2.0, 8.0),
    )
    assert not reading.heard


def _harmonic_residual(seed: int) -> np.ndarray:
    """Low noise with a 150 Hz harmonic tone from 5 to 7 s, sampled at 16 kHz."""
    rate = 16000
    rng = np.random.default_rng(seed)
    t = np.arange(int(DURATION * rate)) / rate
    x = rng.normal(0.0, 1e-4, len(t))
    on = (t >= 5.0) & (t < 7.0)
    x[on] += 0.01 * sum(np.sin(2 * np.pi * 150 * k * t[on]) / k for k in range(1, 6))
    return x


def test_a_harmonic_residual_run_away_from_the_coughs_is_heard() -> None:
    """6ca9935e: a faint voice YAMNet does not name is periodic in what the enhancer removed."""
    residual = _harmonic_residual(6)
    reading = measure_background_speech(
        [],
        plain=(residual * 2, 16000),
        enhanced=(residual, 16000),
        residual=(residual, 16000),
        extent=(1.0, 9.0),
        events=[(2.0, 2.5)],
    )
    assert reading.heard
    assert reading.runs[0][0] >= 4.9


def test_harmonicity_inside_a_cough_is_the_cough() -> None:
    """A voiced cough is the participant's own event, not another voice."""
    residual = _harmonic_residual(7)
    reading = measure_background_speech(
        [],
        plain=(residual * 2, 16000),
        enhanced=(residual, 16000),
        residual=(residual, 16000),
        extent=(1.0, 9.0),
        events=[(4.9, 7.1)],
    )
    assert not reading.heard


def test_an_absent_residual_reads_nothing(tmp_path: Path) -> None:
    """No stored residual windows: no reading, and no flag."""
    assert background_speech_of(ProvStore(run_id="t"), tmp_path, (0.0, 5.0)) is None
