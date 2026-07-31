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


def test_multilabel_reduction_uses_noisy_or_not_max() -> None:
    """Non-normalized (multilabel) rows are per-speaker activations, so P(at least one).

    This test previously asserted the max over the class axis. The max saturates as soon as
    any one channel is confident, and measured on a real recording that produced a posterior
    of *exactly* 1.0000 in all 1070 buckets — a voice detector reporting speech everywhere
    across a conversation with four clear pauses. The noisy-or keeps quiet frames quiet while
    still reaching ~1 when a speaker is clearly present.
    """
    data = np.array([[0.3, 0.9], [0.8, 0.7]])  # rows sum to 1.2 / 1.5, clearly not softmax
    speech = _speech_prob_from_output(data)
    assert np.allclose(speech, [1 - 0.7 * 0.1, 1 - 0.2 * 0.3], atol=1e-9)
    assert (speech >= np.max(data, axis=1)).all(), "at least one speaker is at least as likely"


class _FakeInference:
    """Fake pyannote Inference: emits a constant per-frame value per chunk."""

    class _RF:
        def __init__(self, step: float) -> None:
            self.step = step
            self.duration = step * 4

    class _Model:
        def __init__(self, step: float) -> None:
            self.receptive_field = _FakeInference._RF(step)

    def __init__(self, step: float, value: float) -> None:
        self.model = _FakeInference._Model(step)
        self._step = step
        self._value = value
        self.calls = 0

    def __call__(self, d: dict) -> np.ndarray:
        self.calls += 1
        n_samples = int(d["waveform"].shape[-1])
        sr = int(d["sample_rate"])
        n_frames = max(1, int(round((n_samples / sr) / self._step)))
        return np.full((n_frames, 1), self._value, dtype=np.float64)


def test_chunked_inference_single_pass_short_clip() -> None:
    """A clip <= chunk length is a single pass (one inference call)."""
    audio = Audio(waveform=torch.full((1, 16000 * 5), 0.1, dtype=torch.float32), sampling_rate=16000)
    inf = _FakeInference(step=0.02, value=0.8)
    data, hop, _win = fp_mod.chunked_frame_inference(
        inf,  # type: ignore[arg-type]  # duck-typed stand-in for pyannote Inference
        audio,
        chunk_s=10.0,
        step_s=8.0,
    )
    assert inf.calls == 1
    assert abs(hop - 0.02) < 1e-9
    assert np.allclose(data, 0.8)


def test_chunked_inference_stitches_long_clip() -> None:
    """A clip longer than one chunk is stitched from overlapping windows."""
    dur_s = 25.0
    audio = Audio(waveform=torch.full((1, int(16000 * dur_s)), 0.1, dtype=torch.float32), sampling_rate=16000)
    inf = _FakeInference(step=0.02, value=0.8)
    data, hop, _win = fp_mod.chunked_frame_inference(
        inf,  # type: ignore[arg-type]  # duck-typed stand-in for pyannote Inference
        audio,
        chunk_s=10.0,
        step_s=8.0,
    )
    assert inf.calls > 1  # multiple chunks
    assert abs(hop - 0.02) < 1e-9
    # Continuous timeline spanning ~the whole clip at native resolution.
    assert data.shape[0] > int(dur_s / 0.02) * 0.9
    assert np.allclose(data, 0.8)  # overlap-averaging preserves the constant


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
