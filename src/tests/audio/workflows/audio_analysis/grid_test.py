"""Tests for per-axis bucket grids (feature 20260722-175022, US3).

Verifies BucketGrid.iter_buckets counts and that a distinct (finer) presence
grid coexists with the shared identity grid and the utterance grid, with each
axis recording its own grid in provenance.
"""

from __future__ import annotations

import pytest
import torch

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.audio_analysis import BucketGrid, compute_uncertainty_axes


@pytest.fixture(autouse=True)
def _offline_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub gated model loaders so the compute call stays offline."""
    import senselab.audio.tasks.scene_quality as sq
    import senselab.audio.tasks.voice_activity_detection.frame_posteriors as fp

    monkeypatch.setattr(sq, "extract_brouhaha_frames", lambda audios, *a, **k: [None] * len(audios))
    monkeypatch.setattr(fp, "extract_speech_frame_posteriors", lambda audios, *a, **k: [None] * len(audios))


def _silent_audio(duration_s: float, sr: int = 16000) -> Audio:
    """Trivial Audio object for the compute call."""
    return Audio(waveform=torch.zeros(1, int(duration_s * sr), dtype=torch.float32), sampling_rate=sr)


def _diar_block(segments: list[tuple[float, float, str]]) -> dict:
    """Minimal diar by-model block."""
    from types import SimpleNamespace

    segs = [SimpleNamespace(start=s, end=e, speaker=spk, text="") for s, e, spk in segments]
    return {"status": "ok", "result": [segs], "cache_key": "diar_k"}


def test_bucketgrid_iter_counts() -> None:
    """A 0.1 s / 0.02 s grid yields far more buckets than 0.5 s over the same span."""
    fine = list(BucketGrid(win_length=0.1, hop_length=0.02).iter_buckets(1.0))
    coarse = list(BucketGrid(win_length=0.5, hop_length=0.5).iter_buckets(1.0))
    assert len(fine) > len(coarse)
    assert len(coarse) == 2  # [0,0.5], [0.5,1.0]


def test_per_axis_grids_coexist_and_are_recorded() -> None:
    """Presence (fine), identity (shared), and utterance grids each apply independently."""
    raw_pass = {
        "duration_s": 2.0,
        "diarization": {"by_model": {"pyannote": _diar_block([(0.0, 2.0, "SPEAKER_00")])}},
    }
    axis_results, _, _ = compute_uncertainty_axes(
        passes={"raw_16k": raw_pass},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
        presence_grid=BucketGrid(win_length=0.1, hop_length=0.02),
        utterance_grid=BucketGrid(win_length=1.0, hop_length=0.5),
        params={},
        audio={"raw_16k": _silent_audio(2.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
    )
    presence = axis_results[("raw_16k", "presence")]
    identity = axis_results[("raw_16k", "identity")]
    # Fine presence grid → many more presence rows than the 0.5 s identity grid.
    assert len(presence.rows) > len(identity.rows)
    assert presence.provenance["grid"]["win_length"] == 0.1
    assert presence.provenance["grid"]["hop_length"] == 0.02
    assert identity.provenance["grid"]["win_length"] == 0.5


def test_presence_grid_defaults_to_shared_when_absent() -> None:
    """Without a presence_grid, presence uses the shared grid (legacy behavior)."""
    raw_pass = {
        "duration_s": 2.0,
        "diarization": {"by_model": {"pyannote": _diar_block([(0.0, 2.0, "SPEAKER_00")])}},
    }
    axis_results, _, _ = compute_uncertainty_axes(
        passes={"raw_16k": raw_pass},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
        params={},
        audio={"raw_16k": _silent_audio(2.0)},
        speaker_embedding_models=[],
        aggregator="min",
        speech_presence_labels=["Speech"],
    )
    presence = axis_results[("raw_16k", "presence")]
    assert presence.provenance["grid"]["win_length"] == 0.5
