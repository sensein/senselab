"""J4 — bind the harmonised speakers to the activation channels, and say how firmly.

D-7: this is a *joint* space, not an inheritance. The channels are permutation-arbitrary, so they
cannot name a speaker on their own; the harmonised speaker space has names but only segment-level
timing, so it cannot place them at frame resolution. Each supplies what the other lacks, and how
well-determined the binding is *is* part of the speaker uncertainty rather than an input to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.tasks.voice_activity_detection.frame_posteriors import FramePosterior
from senselab.audio.workflows.audio_analysis.joint import per_speaker_presence


def _fp(rows: list[list[float]], hop: float = 0.1, labels: tuple[str, ...] = ()) -> FramePosterior:
    arr = np.asarray(rows, dtype=np.float64)
    return FramePosterior(
        activations=arr,
        frame_hop_s=hop,
        channel_format="per_speaker",
        channel_labels=labels or tuple(f"speaker#{i + 1}" for i in range(arr.shape[1])),
    )


def test_speakers_bind_to_the_channels_that_were_active_when_they_spoke() -> None:
    """The binding is decided by temporal agreement, which is the only evidence linking the two."""
    # Channel 0 active in the first second, channel 1 in the second.
    rows = [[1.0, 0.0]] * 10 + [[0.0, 1.0]] * 10
    result = per_speaker_presence(
        {"S0": [(0.0, 1.0)], "S1": [(1.0, 2.0)]},
        _fp(rows),
    )
    assert result is not None
    assert result["assignment"] == {"S0": "speaker#1", "S1": "speaker#2"}


def test_the_binding_is_permutation_proof() -> None:
    """Relabelling the channels must move the binding with them, not break it.

    This is the property that makes J4 possible at all: channel identity is arbitrary, so the
    binding has to be re-derived from evidence rather than assumed from an index.
    """
    rows = [[0.0, 1.0]] * 10 + [[1.0, 0.0]] * 10
    result = per_speaker_presence({"S0": [(0.0, 1.0)], "S1": [(1.0, 2.0)]}, _fp(rows))
    assert result is not None
    assert result["assignment"] == {"S0": "speaker#2", "S1": "speaker#1"}


def test_presence_is_reported_at_frame_resolution_not_segment_resolution() -> None:
    """What the speaker space gains from the binding: timing it never had.

    The diarization span says "S0 spoke somewhere in 0-2 s"; the bound channel says where inside it.
    """
    rows = [[1.0, 0.0]] * 5 + [[0.0, 0.0]] * 5 + [[1.0, 0.0]] * 10
    result = per_speaker_presence({"S0": [(0.0, 2.0)]}, _fp(rows))
    assert result is not None
    times, probs = result["presence"]["S0"]
    assert len(times) == 20
    assert probs[7] == pytest.approx(0.0), "the pause inside the segment survives"
    assert probs[2] == pytest.approx(1.0)


def test_a_channel_no_speaker_claimed_is_reported_not_discarded() -> None:
    """An active channel nobody accounts for is the shape a missed speaker takes."""
    rows = [[1.0, 0.0, 0.0]] * 10 + [[0.0, 0.0, 1.0]] * 10
    result = per_speaker_presence({"S0": [(0.0, 1.0)]}, _fp(rows))
    assert result is not None
    assert "speaker#3" in result["unassigned_channels"]


def test_a_speaker_with_no_frame_support_is_reported_not_bound_anyway() -> None:
    """A claim the frames do not back must not be given a channel to make the table tidy."""
    rows = [[1.0, 0.0]] * 20
    result = per_speaker_presence({"S0": [(0.0, 2.0)], "S1": [(5.0, 6.0)]}, _fp(rows))
    assert result is not None
    assert result["assignment"]["S1"] is None
    assert "S1" in result["unassigned_speakers"]


def test_an_ambiguous_binding_reports_a_small_margin() -> None:
    """How firmly the binding is determined is the measurement C2 needs.

    Two channels equally active while one speaker is claimed cannot distinguish which is theirs.
    The margin says so; no threshold decides it here, because "enough" is the caller's question.
    """
    clear = per_speaker_presence({"S0": [(0.0, 1.0)]}, _fp([[1.0, 0.0]] * 10 + [[0.0, 0.0]] * 10))
    tied = per_speaker_presence({"S0": [(0.0, 1.0)]}, _fp([[1.0, 1.0]] * 10 + [[0.0, 0.0]] * 10))
    assert clear is not None and tied is not None
    assert tied["assignment_margin"]["S0"] < clear["assignment_margin"]["S0"]
    assert tied["uncertainty"] > clear["uncertainty"]


def test_a_single_pooled_channel_cannot_place_speakers() -> None:
    """One collapsed speech probability has already discarded who was speaking."""
    fp = FramePosterior(activations=np.ones((10, 1)), frame_hop_s=0.1, channel_format="single")
    assert per_speaker_presence({"S0": [(0.0, 1.0)]}, fp) is None


def test_no_speakers_means_no_binding_to_report() -> None:
    """Nothing to bind is not a binding of nothing."""
    assert per_speaker_presence({}, _fp([[1.0, 0.0]] * 10)) is None


def test_spans_are_derived_from_the_harmonised_cluster_not_a_model_label() -> None:
    """C2 is about the harmonised space, so the spans that feed it must be too.

    A raw ``SPEAKER_00`` means different people to different diarizers; the cluster id is what H2
    exists to produce. A bucket counts for a cluster when *any* diar model placed it there — the
    same union rule coverage already uses, so two models agreeing cannot inflate a span.
    """
    from senselab.audio.workflows.audio_analysis.joint import speaker_spans_from_votes

    votes = [
        {"start": 0.0, "end": 0.5, "votes": {"x": {"cluster_ids": {"pyannote": "C0", "sortformer": "C0"}}}},
        {"start": 0.5, "end": 1.0, "votes": {"x": {"cluster_ids": {"pyannote": "C0"}}}},
        {"start": 1.0, "end": 1.5, "votes": {"x": {"cluster_ids": {"pyannote": "C1"}}}},
    ]
    spans = speaker_spans_from_votes(votes)
    assert spans["C0"] == [(0.0, 1.0)], "contiguous buckets merge into one span"
    assert spans["C1"] == [(1.0, 1.5)]


def test_silence_is_not_a_speaker() -> None:
    """The silent cluster is a placeholder, not someone to bind a channel to."""
    from senselab.audio.workflows.audio_analysis.joint import speaker_spans_from_votes

    votes = [{"start": 0.0, "end": 0.5, "votes": {"x": {"cluster_ids": {"pyannote": "SIL"}}}}]
    assert speaker_spans_from_votes(votes) == {}
