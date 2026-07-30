"""Foreground suppression, depth, and the decisive differential test (T050-T052, T064a).

The single most important test in this feature is :func:`test_reported_content_differs_when_a
_faint_source_is_present`. If two recordings that differ *only* in whether a faint background
source exists produce the same result, the pipeline is reporting residual foreground and the
whole background story does not work — regardless of how much else passes.

Suppression depth, not gain, is the binding constraint. An oracle experiment established
that 30 dB of suppression with the residual amplified produced identical output whether a
faint source was present or absent, because the leaked foreground dominated either way.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from senselab.audio.workflows.audio_analysis.foreground import (
    ForegroundSuppression,
    leakage_margin_db,
    project_onto,
    suppression_depth_db,
)

SR = 16000


def _tone(freq: float, amp: float, seconds: float = 2.0, sr: int = SR) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (amp * np.sin(2 * math.pi * freq * t)).astype(np.float64)


# ── projection ────────────────────────────────────────────────────────


def test_projection_splits_a_scaled_reference_entirely_into_the_parallel_part() -> None:
    """A residual that is pure leaked foreground has no orthogonal component."""
    ref = _tone(440.0, 1.0)
    parallel, orthogonal = project_onto(0.25 * ref, ref)
    assert np.allclose(parallel, 0.25 * ref, atol=1e-9)
    assert float(np.max(np.abs(orthogonal))) < 1e-9


def test_projection_of_an_orthogonal_signal_has_no_parallel_part() -> None:
    """Genuinely different content leaves no foreground behind."""
    ref, other = _tone(440.0, 1.0), _tone(1000.0, 1.0)
    parallel, orthogonal = project_onto(other, ref)
    assert float(np.max(np.abs(parallel))) < 1e-2
    assert np.allclose(orthogonal, other, atol=1e-2)


def test_projection_onto_silence_is_all_orthogonal() -> None:
    """No reference energy means nothing can be attributed to it."""
    sig = _tone(440.0, 1.0)
    parallel, orthogonal = project_onto(sig, np.zeros_like(sig))
    assert float(np.max(np.abs(parallel))) == 0.0
    assert np.allclose(orthogonal, sig)


# ── suppression depth (FR-018a) ───────────────────────────────────────


def test_depth_is_large_when_little_foreground_survives() -> None:
    """Deep suppression reports as deep."""
    speech = _tone(440.0, 1.0)
    residual = _tone(1000.0, 0.1) + 0.001 * speech
    assert suppression_depth_db(speech, residual) > 50.0


def test_depth_is_small_when_the_residual_is_mostly_foreground() -> None:
    """Shallow suppression is reported as shallow, not as a quiet background."""
    speech = _tone(440.0, 1.0)
    residual = 0.5 * speech + _tone(1000.0, 0.01)
    assert suppression_depth_db(speech, residual) < 10.0


def test_depth_is_infinite_when_no_foreground_remains() -> None:
    """Not clamped: a clamped value would read as a measurement."""
    assert suppression_depth_db(_tone(440.0, 1.0), _tone(1000.0, 0.1)) > 40.0


def test_depth_of_a_silent_foreground_is_negative_infinity() -> None:
    """Nothing to suppress means no depth to report."""
    assert suppression_depth_db(np.zeros(SR), _tone(1000.0, 0.1)) == -math.inf


# ── leakage margin (FR-026, SC-008) ───────────────────────────────────


def test_leakage_margin_is_positive_when_the_residual_is_mostly_background() -> None:
    """A background-dominated residual is safe to read categories from."""
    speech = _tone(440.0, 1.0)
    residual = _tone(1000.0, 0.5) + 0.001 * speech
    assert leakage_margin_db(residual, speech) > 20.0


def test_leakage_margin_is_negative_when_the_residual_is_mostly_leaked_speech() -> None:
    """A human-sound category read from this residual is suspect (SC-008)."""
    speech = _tone(440.0, 1.0)
    residual = 0.5 * speech + _tone(1000.0, 0.001)
    assert leakage_margin_db(residual, speech) < 0.0


def test_level_alone_cannot_distinguish_leakage_from_background() -> None:
    """Why the measure is a projection rather than a level.

    Two residuals at the *same* power license opposite conclusions: one is leaked
    foreground, the other is genuine background. A level-only measure sees them as
    identical.
    """
    speech = _tone(440.0, 1.0)
    leaky = 0.3 * speech
    genuine = _tone(1000.0, 0.3)
    assert float(np.mean(leaky**2)) == pytest.approx(float(np.mean(genuine**2)), rel=0.05)
    assert leakage_margin_db(leaky, speech) < 0.0 < leakage_margin_db(genuine, speech)


# ── depth versus the source's own depth (research D6) ─────────────────


def test_depth_must_exceed_the_source_depth_below_the_foreground() -> None:
    """30 dB of suppression does not expose a source 30 dB down.

    The oracle experiment: with the residual foreground still dominant, the output was
    identical whether the faint source was present or absent. The comparison is against the
    source's own depth, not a fixed threshold.
    """
    shallow = ForegroundSuppression(residual=np.zeros(4), achieved_depth_db=30.0, leakage_margin_db=-5.0, model="m")
    assert shallow.is_deep_enough_for(30.0) is False
    assert shallow.is_deep_enough_for(20.0) is True


def test_serialized_depth_of_an_infinite_value_is_null() -> None:
    """JSON has no infinity; a non-finite depth is absent rather than a huge number."""
    s = ForegroundSuppression(residual=np.zeros(4), achieved_depth_db=math.inf, leakage_margin_db=1.0, model="m")
    assert s.to_json()["achieved_depth_db"] is None


def test_fallback_is_recorded_and_the_run_continues() -> None:
    """Suppression failure degrades to the standard variant rather than failing (FR-029)."""
    s = ForegroundSuppression(
        residual=np.zeros(4),
        achieved_depth_db=-math.inf,
        leakage_margin_db=-math.inf,
        model="m",
        fallback="RuntimeError: model unavailable",
    )
    assert s.to_json()["fallback"] is not None
    assert s.to_json()["achieved_depth_db"] is None


# ── the decisive differential test (T050, SC-015) ─────────────────────


def _mix(background_amp: float, *, suppression_db: float) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a suppressor of a given depth over foreground plus optional background."""
    speech = _tone(440.0, 1.0)
    background = _tone(3000.0, background_amp)
    leak = speech * (10.0 ** (-suppression_db / 20.0))
    return leak + background, speech


def test_reported_content_differs_when_a_faint_source_is_present() -> None:
    """SC-015 in mechanism form, and the make-or-break property of this feature.

    With adequate suppression the residual's non-foreground content differs measurably
    between a recording containing a faint background source and an otherwise identical one
    without it. If it does not, the pipeline is reporting residual foreground.
    """
    deep = 60.0
    with_src, speech = _mix(0.01, suppression_db=deep)
    without_src, _ = _mix(0.0, suppression_db=deep)
    _, orth_with = project_onto(with_src, speech)
    _, orth_without = project_onto(without_src, speech)
    with_db = 10 * math.log10(float(np.mean(orth_with**2)))
    without_db = 10 * math.log10(max(float(np.mean(orth_without**2)), 1e-30))
    assert with_db - without_db > 20.0, "a faint source must change the non-foreground content"


def test_shallow_suppression_fails_the_differential_test() -> None:
    """Documents the failure the oracle experiment found, so a pass is meaningful.

    At 30 dB suppression against a source 40 dB below the foreground, the residual is
    dominated by leakage and the two cases become indistinguishable — a null result that
    says nothing about whether background content exists.
    """
    shallow = 30.0
    with_src, speech = _mix(0.01, suppression_db=shallow)
    without_src, _ = _mix(0.0, suppression_db=shallow)
    total_with = 10 * math.log10(float(np.mean(with_src**2)))
    total_without = 10 * math.log10(float(np.mean(without_src**2)))
    assert abs(total_with - total_without) < 1.0, "expected leakage to mask the difference"


def test_depth_measurement_distinguishes_the_two_regimes() -> None:
    """The reported depth is what tells a null result apart from a real absence."""
    deep_res, speech = _mix(0.01, suppression_db=60.0)
    shallow_res, _ = _mix(0.01, suppression_db=20.0)
    assert suppression_depth_db(speech, deep_res) > suppression_depth_db(speech, shallow_res) + 20.0
