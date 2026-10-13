"""The VOICE phonation measure over synthetic recordings."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.workflows.triage.voice_phonation import measure_phonation

pytest.importorskip("parselmouth")

RATE = 16000


def _tone(f0: np.ndarray, level: float = 0.2) -> np.ndarray:
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    return level * np.sum([np.sin(k * phase) / k for k in range(1, 8)], axis=0)


def _recording(
    pieces: list[tuple[float, float, float]], total_s: float, noise: float = 1e-3, seed: int = 0
) -> np.ndarray:
    """``(start, end, f0)`` held tones over room noise; a negative f0 is a soft breath-noise burst."""
    rng = np.random.default_rng(seed)
    x = noise * rng.standard_normal(int(total_s * RATE))
    for start, end, f0 in pieces:
        a, b = int(start * RATE), int(end * RATE)
        x[a:b] += _tone(np.full(b - a, f0)) if f0 > 0 else 2.7 * noise * rng.standard_normal(b - a)
    return x.astype(np.float32)


def _measure(x: np.ndarray, direction: str | None = None, words: tuple = ()) -> dict:
    return measure_phonation((x, RATE), residual=None, enhanced=None, words=words, glide_direction=direction).record()


def test_a_held_vowel_is_its_extent() -> None:
    """A held vowel over room noise is one hold, onset to offset."""
    reading = _measure(_recording([(1.0, 6.0, 140.0)], 8.0))
    assert reading["found"]
    assert reading["extent"]["onset_s"] == pytest.approx(1.0, abs=0.1)
    assert reading["extent"]["end_s"] == pytest.approx(6.0, abs=0.1)
    assert reading["holds_n"] == 1


def test_a_vowel_filling_the_file_is_found() -> None:
    """A vowel that fills the file is found: the floor is read outside it."""
    reading = _measure(_recording([(0.15, 9.95, 120.0)], 10.0))
    assert reading["found"]
    assert reading["extent"]["end_s"] - reading["extent"]["onset_s"] > 9.0


def test_restarted_holds_merge_into_one_extent() -> None:
    """Restarted holds merge into one extent, first onset to last offset."""
    reading = _measure(_recording([(1.0, 2.5, 150.0), (3.2, 4.8, 150.0), (5.4, 7.0, 150.0)], 8.0))
    assert reading["extent"]["onset_s"] == pytest.approx(1.0, abs=0.1)
    assert reading["extent"]["end_s"] == pytest.approx(7.0, abs=0.1)
    assert reading["holds_n"] == 3
    assert len(reading["breaks"]) == 2


def test_the_inhale_before_the_onset_attaches() -> None:
    """A soft breath burst just before the onset becomes the extent's start."""
    reading = _measure(_recording([(0.6, 1.2, -1.0), (1.35, 5.0, 140.0)], 6.0))
    assert reading["extent"]["inhale"]
    assert reading["extent"]["start_s"] == pytest.approx(0.6, abs=0.15)


def test_silence_is_nothing_captured() -> None:
    """Room noise alone has no extent."""
    assert not _measure(_recording([], 5.0))["found"]


def test_a_falling_glide_travels_in_its_declared_direction() -> None:
    """A falling glide travels down, and is no mismatch."""
    n = int(3.0 * RATE)
    f0 = np.geomspace(600.0, 200.0, n)
    x = _recording([], 5.0)
    x[RATE : RATE + n] += _tone(f0).astype(np.float32)
    reading = _measure(x, "down")
    assert reading["shape"] == "declared"
    assert reading["glide"]["travel_declared_semitones"] > 15.0
    assert reading["mismatch"] is None


def test_a_held_vowel_in_a_glide_task_is_a_mismatch() -> None:
    """A held vowel in a rising-glide task is a held mismatch."""
    reading = _measure(_recording([(1.0, 5.0, 190.0)], 6.0), "up")
    assert reading["shape"] == "held"
    assert reading["mismatch"].startswith("held near")


def test_leading_speech_is_reported_with_its_words() -> None:
    """Count-in words before the vowel stay outside the extent, reported with their words."""
    x = _recording([(0.5, 0.8, 180.0), (1.1, 1.4, 170.0), (3.0, 8.0, 140.0)], 9.0)
    reading = _measure(x, words=((0.5, 0.8, "one"), (1.1, 1.4, "two")))
    assert reading["extent"]["onset_s"] == pytest.approx(3.0, abs=0.1)
    assert [word for run in reading["outside_speech"] for word in run["words"]] == ["one", "two"]


def test_a_shutoff_during_phonation_cuts_the_task() -> None:
    """A drop to digital silence during the vowel cuts the task there."""
    x = _recording([(0.5, 6.0, 140.0)], 8.0)
    x[int(5.0 * RATE) : int(7.0 * RATE)] = 0.0
    reading = _measure(x)
    assert reading["shutoff"]["during_task"]
    assert reading["extent"]["end_s"] == pytest.approx(5.0, abs=0.1)


def test_mains_hum_is_not_phonation() -> None:
    """Mains hum fires the guard and the extent is the vowel, not the hum."""
    t = np.arange(int(6.0 * RATE)) / RATE
    hum = (0.02 * np.sum([np.sin(2 * np.pi * 60 * k * t) / k for k in range(1, 6)], axis=0)).astype(np.float32)
    x = _recording([(2.0, 3.0, 220.0)], 6.0) + hum
    reading = measure_phonation((x, RATE), residual=(hum, RATE), enhanced=None, words=(), glide_direction=None)
    assert reading.hum["fired"]
    assert reading.extent is not None
    assert reading.extent["onset_s"] == pytest.approx(2.0, abs=0.15)
    assert reading.extent["end_s"] == pytest.approx(3.0, abs=0.15)
