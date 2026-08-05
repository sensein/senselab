"""Tests for FramePosterior: the L1 activation matrix and the L2 collapses over it.

L1 keeps every channel the model reported (D-5). The pooled P(speech) is a *derived* quantity
computed here rather than a stored field, because storing it as though it were the measurement is
what let a wrong collapse go unnoticed.

Model-free: the segmentation-3.0 loader is validated end to end elsewhere; here we exercise the
pure math and the channel-format decision.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import senselab.audio.tasks.voice_activity_detection.frame_posteriors as fp_mod
from senselab.audio.data_structures import Audio
from senselab.audio.tasks.voice_activity_detection.frame_posteriors import (
    FramePosterior,
    channel_format_for,
    collapse_to_speech_prob,
)


def _single(probs: np.ndarray, hop: float = 0.1) -> FramePosterior:
    """A one-channel posterior (e.g. Brouhaha's VAD head)."""
    return FramePosterior(activations=np.asarray(probs, dtype=np.float64)[:, None], frame_hop_s=hop)


def test_mean_std_over_overlapping_frames() -> None:
    """mean_std_in_window averages the frames overlapping the window."""
    fp = _single(np.array([0.0, 0.0, 1.0, 1.0, 1.0, 0.0]))  # hop 0.1 s
    mean, std = fp.mean_std_in_window(0.2, 0.5)  # frames 2,3,4 → all 1.0
    assert abs(mean - 1.0) < 1e-9
    assert abs(std - 0.0) < 1e-9


def test_std_high_across_onset() -> None:
    """A window straddling an onset has high within-bucket std (instability)."""
    fp = _single(np.concatenate([np.zeros(5), np.ones(5)]))
    _mean, std = fp.mean_std_in_window(0.0, 1.0)  # spans the 0→1 transition
    assert std > 0.4  # std of half-0/half-1 ≈ 0.5


def test_empty_overlap_returns_nan() -> None:
    """A window past the end of the posterior returns (nan, nan)."""
    fp = _single(np.ones(3))
    mean, std = fp.mean_std_in_window(5.0, 6.0)
    assert np.isnan(mean) and np.isnan(std)


def test_per_speaker_rows_summing_to_one_are_not_powerset() -> None:
    """Regression for the exactly-1.0000 saturation.

    segmentation-3.0 declares ``powerset=True``, but pyannote 4.x converts to per-speaker
    activations before returning, so the output has one column per speaker. When a single speaker
    is fully active those rows sum to 1.0 — and the old row-sum sniff therefore took the powerset
    branch and computed ``1 - data[:, 0]``, treating *speaker#1* as the no-speaker class. Measured
    on real audio that read exactly 1.0000 in 100% of frames, including 4 s of digital silence.
    """
    # One speaker fully active, then silence. Rows sum to 1.0 in the active half.
    data = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.02, 0.0]])
    speech = collapse_to_speech_prob(data, channel_format="per_speaker")
    assert speech[0] == pytest.approx(1.0)
    assert speech[3] == pytest.approx(0.02), "silence must not read as speech"
    # The defect, stated explicitly so it cannot come back.
    assert not np.allclose(speech, 1.0 - data[:, 0])


def test_per_speaker_collapse_is_noisy_or() -> None:
    """P(at least one speaker) over independent per-speaker activations."""
    data = np.array([[0.3, 0.9], [0.8, 0.7]])
    speech = collapse_to_speech_prob(data, channel_format="per_speaker")
    assert np.allclose(speech, [1 - 0.7 * 0.1, 1 - 0.2 * 0.3], atol=1e-9)
    assert (speech >= np.max(data, axis=1)).all(), "at least one speaker is at least as likely"


def test_powerset_collapse_uses_the_empty_class() -> None:
    """A genuine powerset output reduces as 1 - P(no-speaker)."""
    data = np.array([[0.9, 0.05, 0.05], [0.2, 0.7, 0.1], [0.0, 0.5, 0.5]])
    speech = collapse_to_speech_prob(data, channel_format="powerset")
    assert np.allclose(speech, [0.1, 0.8, 1.0], atol=1e-9)


def test_single_channel_collapse_is_speaker() -> None:
    """A one-channel posterior (Brouhaha VAD) passes through, only bounded."""
    data = np.array([[0.42], [1.3], [-0.1]])
    assert np.allclose(collapse_to_speech_prob(data, channel_format="single"), [0.42, 1.0, 0.0])


def test_channel_format_read_from_declared_classes_not_row_sums() -> None:
    """Format comes from the model's declaration, compared against the output width.

    Powerset over 3 speakers has 7 classes; per-speaker has 3. Comparing the output width to
    ``len(specifications.classes)`` distinguishes them, where row sums cannot: a single active
    speaker makes per-speaker rows sum to exactly 1.0.
    """
    classes = ["speaker#1", "speaker#2", "speaker#3"]
    assert channel_format_for(n_columns=3, declared_classes=classes) == "per_speaker"
    assert channel_format_for(n_columns=7, declared_classes=classes) == "powerset"
    assert channel_format_for(n_columns=1, declared_classes=classes) == "single"
    # No declaration available: a single column is still unambiguous.
    assert channel_format_for(n_columns=1, declared_classes=None) == "single"


def test_l1_keeps_every_channel() -> None:
    """The activation matrix is the L1 measurement, and per-channel access survives."""
    data = np.array([[0.1, 0.9, 0.0], [0.2, 0.8, 0.0]])
    fp = FramePosterior(activations=data, frame_hop_s=0.1, channel_format="per_speaker")
    assert fp.activations.shape == (2, 3)
    means = fp.per_channel_mean_in_window(0.0, 0.2)
    assert means is not None
    assert np.allclose(means, [0.15, 0.85, 0.0])


def test_pooled_speech_prob_is_derived_not_stored() -> None:
    """``speech_prob()`` is computed from the channels on demand.

    Storing the collapse alongside the matrix would let the two disagree, and the stored value is
    what consumers would read — reintroducing exactly the failure this split removes.
    """
    fp = FramePosterior(
        activations=np.array([[0.0, 1.0, 0.0], [0.0, 0.05, 0.0]]),
        frame_hop_s=0.1,
        channel_format="per_speaker",
    )
    assert not hasattr(fp, "probs")
    assert np.allclose(fp.speech_prob(), [1.0, 0.05])


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


# ── chunk stitching must not average two different speakers together ─────────


def test_stitching_averages_mismatched_speakers_without_permutation_alignment() -> None:
    """The failure mode, pinned so the fix below has something to be a fix *of*.

    segmentation-3.0's speaker columns are arbitrary per inference, so the same person can be
    column 0 in one chunk and column 1 in the next. Averaging by column index then splits one
    speaker across two half-strength channels through the whole overlap region.
    """
    import numpy as np

    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import stitch_frames

    hop = 0.1
    a = np.zeros((10, 2))
    a[:, 0] = 1.0  # speaker is column 0 here
    b = np.zeros((10, 2))
    b[:, 1] = 1.0  # ...and column 1 here — same person, relabelled
    naive = stitch_frames([a, b], [0.0, 0.5], hop)
    overlap = naive[5:10]
    assert overlap == pytest.approx(np.full((5, 2), 0.5)), "one speaker smeared across two channels"


def test_permutation_alignment_keeps_one_speaker_in_one_channel() -> None:
    """With alignment, the relabelled chunk is matched to the timeline before it is averaged in."""
    import numpy as np

    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import stitch_frames

    a = np.zeros((10, 2))
    a[:, 0] = 1.0
    b = np.zeros((10, 2))
    b[:, 1] = 1.0
    aligned = stitch_frames([a, b], [0.0, 0.5], 0.1, align_permutations=True)
    assert aligned[:, 0] == pytest.approx(np.ones(15)), "the speaker stays at full strength"
    assert aligned[:, 1] == pytest.approx(np.zeros(15)), "and the empty channel stays empty"


def test_permutation_alignment_leaves_an_already_consistent_chunk_alone() -> None:
    """No flip to undo means no change — alignment must not invent one."""
    import numpy as np

    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import stitch_frames

    a = np.zeros((10, 2))
    a[:, 0] = 1.0
    b = np.zeros((10, 2))
    b[:, 0] = 1.0
    aligned = stitch_frames([a, b], [0.0, 0.5], 0.1, align_permutations=True)
    assert aligned[:, 0] == pytest.approx(np.ones(15))


def test_fixed_semantic_channels_are_never_permuted() -> None:
    """Brouhaha's columns are [vad, snr, c50] — permuting them would swap unrelated quantities.

    The alignment is opt-in for exactly this reason: it is only sound where the channel ordering
    carries no meaning.
    """
    import numpy as np

    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import stitch_frames

    # A chunk where SNR happens to look more like the previous chunk's VAD than its own SNR does.
    a = np.tile(np.array([[1.0, 0.0, 0.0]]), (10, 1))
    b = np.tile(np.array([[0.0, 1.0, 0.0]]), (10, 1))
    stitched = stitch_frames([a, b], [0.0, 0.5], 0.1)
    assert stitched[5:10, 0] == pytest.approx(np.full(5, 0.5))
    assert stitched[5:10, 1] == pytest.approx(np.full(5, 0.5))


def test_a_seam_with_no_confident_speaker_is_left_alone() -> None:
    """Alignment must decline when the overlap has nothing to key on.

    Measured on the validation run: 4 of 8 seams fell in silence. There the cost matrix is
    near-uniform and the "best" permutation is arbitrary — and because it is applied to the *whole*
    chunk, acting on it would scramble frames that had nothing wrong with them. Exact zeros happen
    to make the assignment degenerate to identity, so the hazard only shows up with faint noise,
    which is what this pins.
    """
    import numpy as np

    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import stitch_frames

    rng = np.random.default_rng(0)
    a = rng.uniform(0.0, 0.02, size=(10, 3))  # silence with a little jitter
    b = np.zeros((10, 3))
    b[:, 2] = 0.9  # a real speaker, but only *after* the overlap region
    b[:5] = rng.uniform(0.0, 0.02, size=(5, 3))
    aligned = stitch_frames([a, b], [0.0, 0.5], 0.1, align_permutations=True)
    # The speaker must still be in the channel the model put them in.
    assert aligned[10:, 2] == pytest.approx(np.full(5, 0.9))
