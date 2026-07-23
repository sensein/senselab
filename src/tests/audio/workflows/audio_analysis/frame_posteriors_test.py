"""Tests for FramePosterior bucket aggregation (feature 20260722-175022, US3).

Model-free: the segmentation-3.0 loader is validated end-to-end later; here we
exercise the pure bucket-aggregation math (mean + within-bucket std) and the
powerset→P(speech) reduction.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import senselab.audio.tasks.voice_activity_detection.frame_posteriors as fp_mod
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.voice_activity_detection.frame_posteriors import (
    FramePosterior,
    _speech_prob_from_output,
)


def test_mean_std_over_overlapping_frames() -> None:
    """mean_std_in_window averages the frames overlapping the window."""
    probs = np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0])  # hop 0.1 s
    fp = FramePosterior(probs=probs, frame_hop_s=0.1)
    mean, std = fp.mean_std_in_window(0.2, 0.5)  # frames 2,3,4 → all 1.0
    assert abs(mean - 1.0) < 1e-9
    assert abs(std - 0.0) < 1e-9


def test_std_high_across_onset() -> None:
    """A window straddling an onset has high within-bucket std (instability)."""
    probs = np.concatenate([np.zeros(5), np.ones(5)])
    fp = FramePosterior(probs=probs, frame_hop_s=0.1)
    _mean, std = fp.mean_std_in_window(0.0, 1.0)  # spans the 0→1 transition
    assert std > 0.4  # std of half-0/half-1 ≈ 0.5


def test_empty_overlap_returns_nan() -> None:
    """A window past the end of the posterior returns (nan, nan)."""
    fp = FramePosterior(probs=np.ones(3), frame_hop_s=0.1)
    mean, std = fp.mean_std_in_window(5.0, 6.0)
    assert np.isnan(mean) and np.isnan(std)


def test_powerset_reduction_to_speech_prob() -> None:
    """Powerset softmax rows → P(speech) = 1 − P(no-speaker class 0)."""
    # 3 frames, 3 powerset classes summing to 1; class 0 = silence.
    data = np.array([[0.9, 0.05, 0.05], [0.2, 0.7, 0.1], [0.0, 0.5, 0.5]])
    speech = _speech_prob_from_output(data)
    assert np.allclose(speech, [0.1, 0.8, 1.0], atol=1e-9)


def test_multilabel_reduction_falls_back_to_max() -> None:
    """Non-normalized (multilabel) rows → max over the class axis."""
    data = np.array([[0.3, 0.9], [0.8, 0.7]])  # rows sum to 1.2 / 1.5, clearly not softmax
    speech = _speech_prob_from_output(data)
    assert np.allclose(speech, [0.9, 0.8], atol=1e-9)


def test_null_safe_when_model_construction_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-023: a gated/inaccessible model (ValidationError at construction) → [None], no raise.

    Regression for the real-model finding: PyannoteAudioModel validates against HF at
    construction, so its failure must be caught inside the extractor's guard.
    """
    if not fp_mod.PYANNOTEAUDIO_AVAILABLE:
        pytest.skip("pyannote-audio not installed")

    def _boom(*args: object, **kwargs: object) -> None:
        raise ValueError("gated repo — not in authorized list")

    monkeypatch.setattr(fp_mod, "PyannoteAudioModel", _boom)
    audio = Audio(waveform=torch.zeros(1, 16000, dtype=torch.float32), sampling_rate=16000)
    assert fp_mod.extract_speech_frame_posteriors([audio]) == [None]
