"""Regenerate the F0-range tables in ``specs/20260817-triage-workflow-dag/praat-instrument-audit.md``.

A probe, not a test: it lives beside the document whose tables it regenerates. Run it with::

    uv run python specs/20260817-triage-workflow-dag/f0-range-probe.py

It prints three tables: the adversarial case set with the range each rule variant derives, the
shipped rule's range and pinned-contour verdict per case, and the coefficient sweep that is the
evidence for ``pitch_excursion_multiplier`` and ``pitch_pinned_octave_ratio`` being insensitive
across their plausible spans rather than fitted.

Every denominator printed here is the same one: ``len(CASES)`` sources, of which
``len(CASES) - 1`` place pitch at a 50 Hz search floor. The 45 Hz fry is the exception and is
counted as an absence, never as a hit or a miss.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Tuple

import numpy as np
import parselmouth

from senselab.audio.data_structures import Audio
from senselab.audio.tasks.features_extraction.praat_parselmouth import extract_pitch_values, get_sound

SR = 16000


def buzz(f0: float, seconds: float = 1.0, harmonics: int = 6) -> np.ndarray:
    """Return a harmonic buzz at ``f0``, amplitude 0.3/h on the h-th harmonic."""
    t = np.arange(int(seconds * SR)) / SR
    return sum(
        (0.3 / (h + 1) * np.sin(2 * np.pi * f0 * (h + 1) * t) for h in range(harmonics)),
        np.zeros_like(t),
    ).astype(np.float32)


def tone(f0: float, level_db: float, seconds: float = 1.0) -> np.ndarray:
    """Return a sine at ``f0`` scaled to ``level_db`` relative to unit amplitude."""
    t = np.arange(int(seconds * SR)) / SR
    return (10.0 ** (level_db / 20.0) * np.sin(2 * np.pi * f0 * t)).astype(np.float32)


def glide(f_lo: float, f_hi: float, seconds: float = 2.0) -> np.ndarray:
    """Return an exponential F0 sweep from ``f_lo`` to ``f_hi``."""
    t = np.arange(int(seconds * SR)) / SR
    f0 = f_lo * ((f_hi / f_lo) ** (t / t[-1]))
    return np.sin(2 * np.pi * np.cumsum(f0) / SR).astype(np.float32)


def excursion(base: float, peak: float, peak_seconds: float, seconds: float = 2.0) -> np.ndarray:
    """Return a buzz at ``base`` with a ``peak_seconds`` window at ``peak``, placed mid-utterance."""
    n = int(seconds * SR)
    n_peak = int(peak_seconds * SR)
    start = (n - n_peak) // 2
    f0 = np.full(n, base, dtype=np.float64)
    f0[start : start + n_peak] = peak
    return np.sin(2 * np.pi * np.cumsum(f0) / SR).astype(np.float32)


def noisy(f0: float, snr_db: float, seconds: float = 2.0, seed: int = 0) -> np.ndarray:
    """Return a buzz at ``f0`` plus seeded broadband noise at ``snr_db``."""
    voice = buzz(f0, seconds=seconds)
    rng = np.random.default_rng(seed)
    scale = float(np.sqrt((voice**2).mean())) * 10.0 ** (-snr_db / 20.0)
    return (voice + rng.standard_normal(voice.size).astype(np.float32) * scale).astype(np.float32)


def as_audio(wave: np.ndarray) -> Audio:
    """Wrap a mono waveform as a 16 kHz ``Audio``."""
    return Audio(waveform=wave.astype(np.float32)[None, :], sampling_rate=SR)


def contour(audio: Audio, floor: float, ceiling: float) -> np.ndarray:
    """Return the nonzero frequencies of the wide autocorrelation pass."""
    snd = get_sound(audio)
    assert isinstance(snd, parselmouth.Sound)
    pitch = snd.to_pitch_ac(time_step=0.005, pitch_floor=floor, pitch_ceiling=ceiling)
    values = pitch.selected_array["frequency"]
    return np.asarray(values[values != 0], dtype=np.float64)


def variants(values: np.ndarray, floor: float, ceiling: float) -> Dict[str, Tuple[float, float]]:
    """Return the three rule variants the audit's table compares, as (floor, ceiling) pairs."""
    q1, p5, q3, p95 = (float(x) for x in np.percentile(values, [25.0, 5.0, 75.0, 95.0]))
    return {
        "hirst quartiles": (max(floor, 0.75 * q1), min(ceiling, 2.5 * q3)),
        "p5 floor + 2.5*q3": (max(floor, p5 / 1.5), min(ceiling, 2.5 * q3)),
        "p5 floor + 1.5*p95": (max(floor, p5 / 1.5), min(ceiling, 1.5 * p95)),
    }


CASES: List[Tuple[str, Callable[[], np.ndarray], float]] = [
    ("330 Hz + strong 55 Hz", lambda: buzz(330.0) + tone(55.0, -6.0), 330.0),
    ("440 Hz + strong 60 Hz", lambda: buzz(440.0) + tone(60.0, -6.0), 440.0),
    ("120 Hz + 60 Hz hum -22 dB", lambda: buzz(120.0, seconds=2.0) + tone(60.0, -22.0, seconds=2.0), 120.0),
    ("220 Hz + 120 Hz hum", lambda: buzz(220.0) + tone(120.0, -6.0), 220.0),
    ("90 Hz clean (low male)", lambda: buzz(90.0), 90.0),
    ("120 Hz clean", lambda: buzz(120.0), 120.0),
    ("220 Hz clean", lambda: buzz(220.0), 220.0),
    ("420 Hz clean (child)", lambda: buzz(420.0), 420.0),
    ("55 Hz fry", lambda: buzz(55.0), 55.0),
    ("glide 100->400", lambda: glide(100.0, 400.0), 400.0),
    ("register break 110->440 0.4 s", lambda: excursion(110.0, 440.0, 0.4), 440.0),
    ("emphatic peak 150->330 0.35 s", lambda: excursion(150.0, 330.0, 0.35), 330.0),
    ("120 Hz at 0 dB SNR", lambda: noisy(120.0, 0.0), 120.0),
    ("45 Hz fry", lambda: buzz(45.0), 45.0),
]


def main() -> None:
    """Print the three tables."""
    floor, ceiling = 50.0, 600.0

    print("## Rule variants, 14 adversarial cases (search range [50, 600])\n")
    header = f"{'source':>30} | {'true F0':>8} | " + " | ".join(
        f"{k:>22}" for k in variants(np.array([100.0]), floor, ceiling)
    )
    print(header)
    print("-" * len(header))
    tallies: Dict[str, int] = {}
    for name, build, true_f0 in CASES:
        values = contour(as_audio(build()), floor, ceiling)
        if values.size == 0:
            print(f"{name:>30} | {true_f0:8.1f} | {'absence (0 frames)':>22}")
            continue
        cells = []
        for key, (lo, hi) in variants(values, floor, ceiling).items():
            ok = lo <= true_f0 <= hi
            tallies[key] = tallies.get(key, 0) + int(ok)
            cells.append(f"[{lo:6.1f}, {hi:6.1f}] {'OK' if ok else 'MISS'}")
        print(f"{name:>30} | {true_f0:8.1f} | " + " | ".join(f"{c:>22}" for c in cells))
    tracked = len(CASES) - 1
    print(f"\ntally over the {tracked} tracked cases: " + ", ".join(f"{k} {v}/{tracked}" for k, v in tallies.items()))
    print("The hum levels are this probe's (-6 dB for 'strong', -22 dB for 'hum'). Earlier drafts of")
    print("the audit quoted x/14 counts that predate this file and are not reproducible; quote these.")

    print("\n## The shipped rule, with the fallback's verdict\n")
    header = f"{'source':>30} | {'frames':>6} | {'fell back':>9} | {'range':>20} | contains F0"
    print(header)
    print("-" * len(header))
    shipped = 0
    for name, build, true_f0 in CASES:
        derived = extract_pitch_values(as_audio(build()), search_floor_hz=floor, search_ceiling_hz=ceiling)
        if derived["pitch_frames"] == 0.0:
            print(f"{name:>30} | {0:6.0f} | {'-':>9} | {'absence':>20} | unchanged")
            continue
        lo, hi = derived["pitch_floor"], derived["pitch_ceiling"]
        ok = lo <= true_f0 <= hi
        shipped += int(ok)
        print(
            f"{name:>30} | {derived['pitch_frames']:6.0f} | {derived['pitch_range_fell_back']:9.0f} | "
            f"{f'[{lo:.1f}, {hi:.1f}]':>20} | {'yes' if ok else 'NO'}"
        )
    print(f"\nshipped rule: {shipped} of {len(CASES) - 1} tracked cases contain the true F0")

    print("\n## Coefficient sweeps: insensitivity, which is not derivation\n")
    for name, span in (
        ("pitch_excursion_multiplier", (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5)),
        ("pitch_pinned_octave_ratio", (1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0)),
    ):
        print(f"{name}:")
        for value in span:
            hits = 0
            for _label, build, true_f0 in CASES:
                derived = extract_pitch_values(
                    as_audio(build()), search_floor_hz=floor, search_ceiling_hz=ceiling, **{name: value}
                )
                if derived["pitch_frames"] == 0.0:
                    continue
                hits += int(derived["pitch_floor"] <= true_f0 <= derived["pitch_ceiling"])
            print(f"  {value:>5} -> {hits}/{tracked}")
        print()


if __name__ == "__main__":
    main()
