"""Per-voter presence mapping — the part of ``aggregate`` that has production callers.

The per-axis folds this file used to exercise (``aggregate_speech_presence``,
``aggregate_speaker``, ``aggregate_asr``) are gone: they were complete and tested and nothing
called them, while the run's single fold is ``fuse.fuse_axis``. Their tests went with them —
keeping them would have kept a second, unexercised definition of every axis alive in the suite,
which is how the suite comes to disagree with the pipeline.

What remains is ``(speaks, native_confidence) → p_voice``, read by the belief store's ingest and by
S1's stream election. The cases are the same ones, restated on the quantity that is actually
produced: ``p_voice`` rather than the symmetric uncertainty derived from it.
"""

from __future__ import annotations

from typing import Any

import pytest

from senselab.audio.workflows.audio_analysis.aggregate import per_source_voice, speech_presence_p_voice

# ── the per-voter mapping ─────────────────────────────────────────────


def test_a_committed_vote_with_no_confidence_reads_as_certainty() -> None:
    """``speaks=True`` and no ``native_confidence`` is a full-strength claim, not a missing one."""
    votes = {f"m{i}": {"speaks": True, "native_confidence": None} for i in range(4)}
    assert speech_presence_p_voice(votes) == pytest.approx(1.0, abs=1e-6)


def test_a_partial_confidence_carries_through_unchanged() -> None:
    """``speaks=True`` at 0.8 is p_voice 0.8 — the confidence *is* the probability of voice."""
    votes = {f"m{i}": {"speaks": True, "native_confidence": 0.8} for i in range(4)}
    assert speech_presence_p_voice(votes) == pytest.approx(0.8, abs=1e-6)


def test_an_even_split_lands_at_one_half() -> None:
    """One committed voter each way: the mean is 0.5, which is where doubt is maximal."""
    votes = {
        "m0": {"speaks": True, "native_confidence": None},
        "m1": {"speaks": False, "native_confidence": None},
    }
    assert speech_presence_p_voice(votes) == pytest.approx(0.5)


def test_three_to_one_is_a_plain_mean_over_voters() -> None:
    """Equal weights, so 3 True + 1 False is 0.75 — no voter counts twice."""
    votes = {
        "m0": {"speaks": True},
        "m1": {"speaks": True},
        "m2": {"speaks": True},
        "m3": {"speaks": False},
    }
    assert speech_presence_p_voice(votes) == pytest.approx(0.75, abs=1e-6)


def test_a_confident_silence_claim_is_complemented_not_dropped() -> None:
    """``speaks=False`` at 0.9 is p_voice 0.1: the confidence is in the claim, so it inverts."""
    votes = {f"m{i}": {"speaks": False, "native_confidence": 0.9} for i in range(3)}
    assert speech_presence_p_voice(votes) == pytest.approx(0.1, abs=1e-6)


def test_committed_silence_reads_as_zero() -> None:
    """``speaks=False`` with no confidence is certainty of silence, not absence of an opinion."""
    votes = {f"m{i}": {"speaks": False, "native_confidence": None} for i in range(3)}
    assert speech_presence_p_voice(votes) == pytest.approx(0.0, abs=1e-6)


def test_a_scored_voter_and_a_binary_dissenter_both_count() -> None:
    """A 0.99 YAMNet claim against a committed 'no' averages to 0.495 — neither side is dropped."""
    votes: dict[str, dict[str, Any]] = {
        "yamnet": {"speaks": True, "native_confidence": 0.99},
        "binary_dissenter": {"speaks": False, "native_confidence": None},
    }
    assert speech_presence_p_voice(votes) == pytest.approx((0.99 + 0.0) / 2, abs=1e-6)


def test_one_voter_is_its_own_answer() -> None:
    """With a single witness there is nothing to average, so the reading passes through."""
    assert speech_presence_p_voice({"m0": {"speaks": True, "native_confidence": 0.7}}) == pytest.approx(0.7, abs=1e-6)


def test_no_voters_is_none_and_never_a_number() -> None:
    """Nothing said anything, which is not the same as a bucket at p_voice 0.5 (FR-007)."""
    assert speech_presence_p_voice({}) is None


def test_a_hallucination_flag_votes_against_voice() -> None:
    """An indicted ASR claim is attenuated to 0.1 rather than removed, so it stays visible."""
    assert per_source_voice({"asr": {"speaks": True, "hallucinated": True}})["asr"][0] == pytest.approx(0.1)


def test_an_unmeasured_source_keeps_full_weight() -> None:
    """Absent from ``weights`` means never measured, and a factor nobody gathered is not a discount."""
    votes = {"measured": {"speaks": True}, "unmeasured": {"speaks": False}}
    per_source = per_source_voice(votes, weights={"measured": 0.25})
    assert per_source["measured"][1] == pytest.approx(0.25)
    assert per_source["unmeasured"][1] == pytest.approx(1.0)
