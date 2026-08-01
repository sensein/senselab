"""J1 — how many speakers are simultaneously active, from the intact activation channels.

J1 is available now and J4 is not, for one reason worth pinning in a test: `segmentation-3.0`'s
channels are permutation-arbitrary within a window, so channel *k* is not a stable speaker across
the recording (D-7). A *count* of active channels does not care which channel is whom, so it is
well-defined without resolving the speaker↔channel assignment that J4 needs rounds for.

The count is reported as a distribution rather than a number. Entropy needs a distribution, and
"probably one speaker, possibly two" is a different state from "certainly one" even when both round
to the same expected count.
"""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.tasks.voice_activity_detection.frame_posteriors import FramePosterior
from senselab.audio.workflows.audio_analysis.joint import overlap_count_posterior


def _fp(rows: list[list[float]], hop: float = 0.01) -> FramePosterior:
    return FramePosterior(
        activations=np.asarray(rows, dtype=np.float64),
        frame_hop_s=hop,
        channel_format="per_speaker",
        channel_labels=tuple(f"speaker#{i + 1}" for i in range(len(rows[0]))),
    )


def test_two_confident_speakers_give_a_certain_count_of_two() -> None:
    """Both channels asserted → all the mass on 2, and no doubt about it."""
    result = overlap_count_posterior(_fp([[1.0, 1.0]] * 20), 0.0, 0.2)
    assert result is not None
    assert result["counts"][2] == pytest.approx(1.0)
    assert result["expected_count"] == pytest.approx(2.0)
    assert result["uncertainty"] == pytest.approx(0.0)


def test_one_speaker_active_gives_a_certain_count_of_one() -> None:
    """The other channel being silent is evidence, not absence of evidence."""
    result = overlap_count_posterior(_fp([[1.0, 0.0]] * 20), 0.0, 0.2)
    assert result is not None
    assert result["counts"][1] == pytest.approx(1.0)
    assert result["uncertainty"] == pytest.approx(0.0)


def test_two_coin_flip_channels_spread_the_count() -> None:
    """Independent Bernoullis at 0.5 give the binomial 1/4, 1/2, 1/4 — and real doubt.

    Expected count is 1.0, the same as a confident single speaker, which is exactly why the
    distribution has to be reported instead of its mean.
    """
    result = overlap_count_posterior(_fp([[0.5, 0.5]] * 20), 0.0, 0.2)
    assert result is not None
    assert result["counts"] == pytest.approx({0: 0.25, 1: 0.5, 2: 0.25})
    assert result["expected_count"] == pytest.approx(1.0)
    assert result["uncertainty"] > 0.9


def test_the_count_is_invariant_to_channel_permutation() -> None:
    """The property that makes J1 answerable while J4 still needs rounds.

    Channels are permutation-arbitrary within a window, so any quantity that depends on *which*
    channel is which is not yet well-defined. A count does not.
    """
    rows = [[0.9, 0.2, 0.6]] * 10
    permuted = [[0.6, 0.9, 0.2]] * 10
    a = overlap_count_posterior(_fp(rows), 0.0, 0.1)
    b = overlap_count_posterior(_fp(permuted), 0.0, 0.1)
    assert a is not None and b is not None
    assert a["counts"] == pytest.approx(b["counts"])
    assert a["expected_count"] == pytest.approx(b["expected_count"])


def test_silence_concentrates_on_zero_speakers() -> None:
    """All channels quiet → confidently nobody, which is a claim, not a missing answer."""
    result = overlap_count_posterior(_fp([[0.0, 0.0, 0.0]] * 10), 0.0, 0.1)
    assert result is not None
    assert result["counts"][0] == pytest.approx(1.0)
    assert result["expected_count"] == pytest.approx(0.0)


def test_within_bucket_timing_is_not_averaged_away() -> None:
    """Two speakers taking turns is not two speakers overlapping.

    Averaging the channels over the bucket first would give both 0.5 and report a 25% chance of
    overlap that never happened. The posterior is built per frame and then pooled, so a bucket where
    exactly one speaker is active at every instant reports one speaker.
    """
    turns = [[1.0, 0.0]] * 10 + [[0.0, 1.0]] * 10
    result = overlap_count_posterior(_fp(turns), 0.0, 0.2)
    assert result is not None
    assert result["counts"][1] == pytest.approx(1.0)
    assert result["counts"].get(2, 0.0) == pytest.approx(0.0)
    assert result["uncertainty"] == pytest.approx(0.0)


def test_a_single_collapsed_channel_cannot_answer_the_question() -> None:
    """One pooled speech probability has already discarded the count; it must not guess one."""
    fp = FramePosterior(activations=np.full((10, 1), 0.9), frame_hop_s=0.01, channel_format="single")
    assert overlap_count_posterior(fp, 0.0, 0.1) is None


def test_a_bucket_with_no_frames_yields_no_claim() -> None:
    """Outside the recording there is nothing to count."""
    assert overlap_count_posterior(_fp([[1.0, 1.0]] * 10), 5.0, 6.0) is None


def test_overlap_probability_is_the_mass_above_one_speaker() -> None:
    """The convenience field consumers actually want, derived from the same distribution."""
    result = overlap_count_posterior(_fp([[0.5, 0.5]] * 20), 0.0, 0.2)
    assert result is not None
    assert result["p_overlap"] == pytest.approx(0.25)


# ── J2: speaker change points from windowed embeddings ──────────────────────


def _windows(vectors: list[list[float]], width_s: float = 2.0, hop_s: float = 0.05) -> list:
    from senselab.audio.workflows.audio_analysis.embeddings import WindowEmbedding

    return [
        WindowEmbedding(start_s=i * hop_s, end_s=i * hop_s + width_s, vector=np.asarray(v, dtype=np.float64))
        for i, v in enumerate(vectors)
    ]


def test_change_detection_compares_across_a_whole_window_not_adjacent_ones() -> None:
    """D-2: at a 50 ms hop, adjacent 2 s windows share 97.5% of their audio.

    Their distance is therefore dominated by the 2.5% that is new, which is not a speaker-change
    signal at all. The comparison has to span a whole window so the two sides are disjoint spans
    meeting at the boundary — the fine hop buys *localisation*, not independent samples.
    """
    from senselab.audio.workflows.audio_analysis.joint import speaker_change_series

    result = speaker_change_series(_windows([[1.0, 0.0]] * 100, width_s=2.0, hop_s=0.05))
    assert result is not None
    assert result["lag_steps"] == 40, "2.0 s window / 0.05 s hop = 40 steps to reach a disjoint span"


def test_a_speaker_change_shows_up_at_the_boundary() -> None:
    """Two speakers back to back: change evidence peaks where they meet."""
    from senselab.audio.workflows.audio_analysis.joint import speaker_change_series

    a, b = [1.0, 0.0], [0.0, 1.0]
    entries = _windows([a] * 60 + [b] * 60)
    result = speaker_change_series(entries)
    assert result is not None
    peak_idx = int(np.argmax(result["p_change"]))
    peak_t = result["times"][peak_idx]
    # Speaker A occupies windows 0-59 (starts 0.00-2.95 s); the first all-B window starts at 3.0 s,
    # so the disjoint-span boundary lands there.
    assert result["p_change"][peak_idx] > 0.9
    assert 2.0 <= peak_t <= 5.0


def test_one_speaker_throughout_yields_no_change_evidence() -> None:
    """A steady speaker must not produce change points, or every recording has them."""
    from senselab.audio.workflows.audio_analysis.joint import speaker_change_series

    result = speaker_change_series(_windows([[1.0, 0.0]] * 120))
    assert result is not None
    assert float(np.max(result["p_change"])) < 0.1
    assert float(np.mean(result["uncertainty"])) < 0.2


def test_change_uncertainty_peaks_where_the_evidence_is_ambiguous() -> None:
    """Uncertainty is the entropy of {change, no change}, so it is highest mid-band.

    A confident change and a confident continuation are both certain; the doubt lives at the
    distances the calibration band cannot resolve.
    """
    from senselab.audio.workflows.audio_analysis.joint import speaker_change_series

    # 60 degrees off, giving cosine similarity 0.5 and therefore distance 0.5 -- the middle of the
    # 0.30-0.70 calibration band. A 45-degree offset would *not* work: distance 0.293 sits below the
    # same-speaker floor, i.e. inside the phonetic noise floor where a small distance is no evidence
    # at all, so it reads as a confident continuation rather than as ambiguity.
    mid = [float(np.cos(np.pi / 3)), float(np.sin(np.pi / 3))]
    ambiguous = speaker_change_series(_windows([[1.0, 0.0]] * 60 + [mid] * 60))
    confident = speaker_change_series(_windows([[1.0, 0.0]] * 120))
    assert ambiguous is not None and confident is not None
    assert float(np.max(ambiguous["uncertainty"])) > float(np.max(confident["uncertainty"]))


def test_too_few_windows_to_span_a_lag_yields_no_claim() -> None:
    """With less than one window-length of hops there is no disjoint pair to compare."""
    from senselab.audio.workflows.audio_analysis.joint import speaker_change_series

    assert speaker_change_series(_windows([[1.0, 0.0]] * 10)) is None
    assert speaker_change_series([]) is None
