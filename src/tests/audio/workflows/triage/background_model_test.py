"""The recording-level background model, on synthetic recordings with a known answer."""

from __future__ import annotations

import numpy as np

from senselab.audio.workflows.triage.background_model import measure_background

RATE = 16000


def _noise(seconds: float, db: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(int(seconds * RATE)) * 10 ** (db / 20.0)


def _ramped(x: np.ndarray, ramp_s: float) -> np.ndarray:
    n = min(int(ramp_s * RATE), len(x) // 2)
    envelope = np.ones(len(x))
    envelope[:n] = np.linspace(0.0, 1.0, n)
    envelope[len(x) - n :] = np.linspace(1.0, 0.0, n)
    return x * envelope


def _vowel(seconds: float, db: float, f0: float = 150.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    tone = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, 20) if k * f0 < RATE / 2)
    return tone / np.max(np.abs(tone)) * 10 ** (db / 20.0)


def _overlap(region: tuple[float, float], span: tuple[float, float]) -> bool:
    return min(region[1], span[1]) > max(region[0], span[0])


def test_a_quick_breath_is_activity_and_a_click_is_an_impulse() -> None:
    """A 0.25 s breath with a noise body is activity; a 1 ms click is an impulse and not activity."""
    x = _noise(4.0, -60.0)
    breath = _ramped(_noise(0.25, -25.0, seed=1), 0.06)
    start = int(1.0 * RATE)
    x[start : start + len(breath)] += breath
    click = int(2.5 * RATE)
    x[click : click + 16] += 0.5 * np.hanning(16)
    reading = measure_background((x, RATE), residual=None, recording=None, clips=())
    regions = [(r.start_s, r.end_s) for r in reading.regions]
    assert any(_overlap(r, (1.0, 1.25)) for r in regions)
    assert not any(_overlap(r, (2.49, 2.52)) for r in regions)
    assert any(abs(imp.peak_s - 2.5) < 0.01 for imp in reading.impulses)
    assert not any(1.0 <= imp.peak_s <= 1.25 for imp in reading.impulses)


def test_residual_hum_is_read_as_mains_lines() -> None:
    """Mains multiples standing over their neighbourhood in the residual fire the hum reading."""
    t = np.arange(3 * RATE) / RATE
    hum = sum(np.sin(2 * np.pi * 60.0 * k * t) for k in range(1, 6)) * 10 ** (-40 / 20.0)
    residual = hum + _noise(3.0, -70.0)
    reading = measure_background((residual.copy(), RATE), residual=(residual, RATE), recording=None, clips=())
    assert reading.hum.mains_hz == (60.0,)
    assert reading.hum.lines["60"] >= 3


def test_a_shutoff_ends_in_a_flat_digital_floor() -> None:
    """An abrupt fall to digital silence after activity is a shutoff span."""
    x = np.concatenate([_noise(1.0, -60.0), _vowel(2.0, -20.0) + _noise(2.0, -60.0), np.zeros(RATE)])
    reading = measure_background((x, RATE), residual=None, recording=None, clips=())
    assert len(reading.shutoffs) == 1
    assert abs(reading.shutoffs[0][0] - 3.0) < 0.1


def test_a_task_that_fills_the_file_reads_its_floor_off_the_residual() -> None:
    """A vowel with no quiet frames takes its floor from the residual and stays active."""
    noise = _noise(4.0, -70.0)
    x = _vowel(4.0, -20.0) + noise
    reading = measure_background((x, RATE), residual=(noise, RATE), recording=None, clips=())
    assert reading.floor.source == "residual"
    assert reading.active_s >= 3.5
    assert reading.impulses == ()


def test_silence_has_no_activity_and_a_digital_floor() -> None:
    """Digital silence has a digital floor, no activity and no impulses."""
    reading = measure_background((np.zeros(2 * RATE), RATE), residual=None, recording=None, clips=())
    assert reading.floor.source == "digital"
    assert reading.regions == ()
    assert reading.impulses == ()


def test_a_dropout_is_a_span_on_the_original_recording() -> None:
    """A run of exact zeros is a dropout span; the kept clip spans are carried as given."""
    x = _noise(2.0, -30.0)
    x[RATE : RATE + 800] = 0.0
    reading = measure_background((x, RATE), residual=None, recording=(x, RATE), clips=((0.2, 0.21),))
    assert any(abs(a - 1.0) < 0.01 and abs(b - 1.05) < 0.01 for a, b in reading.dropouts)
    assert reading.record()["faults"]["clip"] == [[0.2, 0.21]]
