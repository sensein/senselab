"""Invariance probes: perturbations that must not change a model's answer.

The stability factor in ``reliability.py`` uses the raw-vs-enhanced pair, but enhancement is a
genuine transform — a model is entitled to answer differently on enhanced audio. These
perturbations are different in kind: each is chosen so that a *correct* model returns the same
answer, which makes any change a defect in the model rather than a response to the signal.

Because measuring it requires re-running inference, it is opt-in.
"""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.workflows.audio_analysis.invariance import (
    INVARIANT_PERTURBATIONS,
    invariance_score,
    perturb,
)

SR = 16000


def _speech_like() -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(0, 1.0, 1 / SR)
    return (0.3 * np.sin(2 * np.pi * 150 * t) + 0.05 * rng.normal(size=t.size)).astype(np.float64)


# ── the perturbations preserve what a correct model should report ───────


def test_every_declared_perturbation_is_applicable() -> None:
    """A named perturbation that cannot run would silently score as perfect invariance."""
    x = _speech_like()
    for name in INVARIANT_PERTURBATIONS:
        out = perturb(x, SR, name)
        assert out.size > 0
        assert np.isfinite(out).all()


def test_gain_scaling_changes_no_signal_to_noise_ratio() -> None:
    """The property that makes gain output-preserving: it moves signal and noise together."""
    x = _speech_like()
    scaled = perturb(x, SR, "gain_down_6db")
    ratio = np.linalg.norm(scaled) / np.linalg.norm(x)
    assert ratio == pytest.approx(10 ** (-6 / 20), rel=1e-6)


def test_gain_does_not_clip() -> None:
    """A perturbation that clips is no longer output-preserving — it adds distortion."""
    x = _speech_like() * 3.0
    assert np.abs(perturb(x, SR, "gain_down_6db")).max() <= np.abs(x).max()


def test_a_whole_sample_shift_preserves_content() -> None:
    """Padding shifts the timeline without resampling, so the audio itself is unaltered."""
    x = _speech_like()
    shifted = perturb(x, SR, "shift_10ms")
    offset = int(0.010 * SR)
    assert np.allclose(shifted[offset : offset + x.size], x)


def test_dc_offset_leaves_the_ac_content_intact() -> None:
    """Speech models work on mean-removed spectra, so a small DC step should be invisible."""
    x = _speech_like()
    shifted = perturb(x, SR, "dc_offset")
    assert np.allclose(shifted - shifted.mean(), x - x.mean(), atol=1e-9)


def test_an_unknown_perturbation_is_refused() -> None:
    """Silently returning the input would score as perfect invariance on a typo."""
    with pytest.raises(ValueError, match="unknown perturbation"):
        perturb(_speech_like(), SR, "reverse_time")


# ── scoring ────────────────────────────────────────────────────────────


def test_a_model_that_answers_identically_is_fully_invariant() -> None:
    """The anchor: unchanged answers under output-preserving change means no defect seen."""
    assert invariance_score(2, {"gain_down_6db": 2, "shift_10ms": 2}) == pytest.approx(1.0)


def test_a_model_that_changes_its_answer_loses_invariance() -> None:
    """A different answer to the same question is a defect, not a response to the signal."""
    score = invariance_score(2, {"gain_down_6db": 5, "shift_10ms": 2})
    assert score is not None and score < 1.0


def test_disagreement_is_graded_by_how_far_the_answer_moved() -> None:
    """One extra speaker and three extra speakers are not equally wrong."""
    near = invariance_score(2, {"gain_down_6db": 3})
    far = invariance_score(2, {"gain_down_6db": 8})
    assert near is not None and far is not None
    assert far < near


def test_no_probe_results_means_no_invariance_claim() -> None:
    """Never measured must not read as measured-and-perfect."""
    assert invariance_score(2, {}) is None


def test_invariance_never_reaches_zero() -> None:
    """Mirrors the other floors: attenuate a signal, never erase its claim."""
    score = invariance_score(1, {"gain_down_6db": 99})
    assert score is not None and score > 0.0


# ── the probe runner ───────────────────────────────────────────────────


def test_a_model_stable_under_every_probe_scores_full_invariance() -> None:
    """A diarizer whose count is level- and framing-independent is behaving correctly."""
    from senselab.audio.workflows.audio_analysis.invariance import probe_diarization_invariance

    scores = probe_diarization_invariance(
        _speech_like(),
        SR,
        reference_counts={"steady": 2},
        run_diarization=lambda w, sr: {"steady": 2},
    )
    assert scores["steady"] == pytest.approx(1.0)


def test_a_model_whose_count_follows_the_gain_is_penalised() -> None:
    """Gain changes no signal-to-noise ratio, so a count that moves with it is an artifact."""
    from senselab.audio.workflows.audio_analysis.invariance import probe_diarization_invariance

    def run(w: np.ndarray, sr: int) -> dict[str, int]:
        return {"level_sensitive": 5 if np.abs(w).max() < 0.2 else 2}

    scores = probe_diarization_invariance(
        _speech_like() * 0.5, SR, reference_counts={"level_sensitive": 2}, run_diarization=run
    )
    assert scores["level_sensitive"] < 1.0


def test_a_probe_that_raises_yields_no_verdict() -> None:
    """A failed probe is absent evidence, not evidence of instability."""
    from senselab.audio.workflows.audio_analysis.invariance import probe_diarization_invariance

    def boom(w: np.ndarray, sr: int) -> dict[str, int]:
        raise RuntimeError("model unavailable")

    assert probe_diarization_invariance(_speech_like(), SR, reference_counts={"m": 2}, run_diarization=boom) == {}


def test_a_model_absent_from_the_reference_is_not_scored() -> None:
    """Without an unperturbed answer there is nothing to compare against."""
    from senselab.audio.workflows.audio_analysis.invariance import probe_diarization_invariance

    scores = probe_diarization_invariance(
        _speech_like(), SR, reference_counts={}, run_diarization=lambda w, sr: {"ghost": 3}
    )
    assert scores == {}
