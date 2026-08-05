"""H3 — put several ASR transcripts into one word space.

Each transcript is already aligned to the *audio*, independently. That is not enough to compare
them: two models that heard the same sentence can still disagree about where a word starts, and one
that inserted a word shifts every timestamp after it. Aligning the transcripts to *each other*
gives a slot structure in which "these three models produced different words for the same position"
is expressible at all — which is what the ASR axis needs and what a per-window WER cannot say.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.harmonize import harmonize_transcripts


def _w(pairs: list[tuple[float, float, str]]) -> list[tuple[float, float, str]]:
    return pairs


def test_identical_transcripts_produce_one_slot_per_word_and_no_disagreement() -> None:
    """The floor case: agreement must read as agreement, not as a slot per model."""
    words = _w([(0.0, 0.4, "the"), (0.4, 0.9, "cat"), (0.9, 1.4, "sat")])
    result = harmonize_transcripts({"a": words, "b": list(words)})
    assert len(result.slots) == 3
    assert [s.consensus for s in result.slots] == ["the", "cat", "sat"]
    assert all(s.disagreement == pytest.approx(0.0) for s in result.slots)
    assert result.gap_rate["a"] == pytest.approx(0.0)
    assert result.insertion_rate["b"] == pytest.approx(0.0)


def test_a_substitution_is_one_slot_with_two_readings() -> None:
    """Disagreement belongs in a slot, not in two slots that happen to overlap in time."""
    a = _w([(0.0, 0.4, "the"), (0.4, 0.9, "cat"), (0.9, 1.4, "sat")])
    b = _w([(0.0, 0.4, "the"), (0.4, 0.9, "bat"), (0.9, 1.4, "sat")])
    result = harmonize_transcripts({"a": a, "b": b})
    assert len(result.slots) == 3
    middle = result.slots[1]
    assert set(middle.words.values()) == {"cat", "bat"}
    assert middle.disagreement > 0.0
    # A two-way tie has no majority, so neither reading may be published as the consensus.
    assert middle.consensus is None


def test_a_deletion_is_a_gap_not_a_shifted_word() -> None:
    """A model that missed a word must not have its later words counted as substitutions.

    This is the failure a time-only comparison makes: every word after the deletion lines up with
    the wrong one, so a single miss reads as a whole tail of disagreements.
    """
    a = _w([(0.0, 0.4, "the"), (0.4, 0.9, "big"), (0.9, 1.4, "cat"), (1.4, 1.9, "sat")])
    b = _w([(0.0, 0.4, "the"), (0.4, 0.9, "cat"), (0.9, 1.4, "sat")])
    result = harmonize_transcripts({"a": a, "b": b})
    assert len(result.slots) == 4
    assert result.slots[1].words.get("b") is None, "the missing word is a gap in b"
    assert result.slots[2].words == {"a": "cat", "b": "cat"}
    assert result.slots[3].words == {"a": "sat", "b": "sat"}
    assert result.gap_rate["b"] == pytest.approx(0.25)


def test_a_hallucinated_run_is_an_insertion_by_one_model() -> None:
    """Words no other model produced are insertions, and are reported as that model's rate."""
    a = _w([(0.0, 0.4, "the"), (0.4, 0.9, "cat")])
    b = _w([(0.0, 0.4, "the"), (0.4, 0.9, "cat"), (0.9, 1.4, "thanks"), (1.4, 1.9, "for"), (1.9, 2.4, "watching")])
    result = harmonize_transcripts({"a": a, "b": b})
    assert result.insertion_rate["b"] > 0.5
    assert result.insertion_rate["a"] == pytest.approx(0.0)
    tail = [s for s in result.slots if s.words.get("a") is None]
    assert [s.words["b"] for s in tail] == ["thanks", "for", "watching"]


def test_majority_carries_the_consensus_and_dissent_is_kept() -> None:
    """Two of three agreeing is a consensus; the third's reading is not discarded."""
    base = _w([(0.0, 0.4, "the"), (0.4, 0.9, "cat")])
    odd = _w([(0.0, 0.4, "the"), (0.4, 0.9, "hat")])
    result = harmonize_transcripts({"a": base, "b": list(base), "c": odd})
    slot = result.slots[1]
    assert slot.consensus == "cat"
    assert slot.words["c"] == "hat"
    assert 0.0 < slot.disagreement < 1.0


def test_slot_times_come_from_the_models_that_filled_it() -> None:
    """A slot spans what the contributing models actually reported, not an invented midpoint."""
    a = _w([(0.0, 0.5, "hello")])
    b = _w([(0.2, 0.8, "hello")])
    slot = harmonize_transcripts({"a": a, "b": b}).slots[0]
    assert slot.start_s == pytest.approx(0.0)
    assert slot.end_s == pytest.approx(0.8)


def test_case_and_punctuation_do_not_count_as_disagreement() -> None:
    """Models differ in casing and punctuation conventions; that is not a transcription dispute."""
    a = _w([(0.0, 0.4, "The"), (0.4, 0.9, "cat.")])
    b = _w([(0.0, 0.4, "the"), (0.4, 0.9, "cat")])
    result = harmonize_transcripts({"a": a, "b": b})
    assert all(s.disagreement == pytest.approx(0.0) for s in result.slots)
    # The surface forms are still recoverable — normalisation decides agreement, it does not
    # overwrite what a model said.
    assert result.slots[0].words["a"] == "The"


def test_a_single_model_yields_slots_but_no_disagreement() -> None:
    """One transcript is a lattice of width one, not an error."""
    result = harmonize_transcripts({"a": _w([(0.0, 0.4, "hi")])})
    assert len(result.slots) == 1
    assert result.slots[0].disagreement == pytest.approx(0.0)
    assert result.gap_rate == {"a": 0.0}


def test_no_transcripts_yields_nothing() -> None:
    """Empty in, empty out — not a lattice of zero-width slots."""
    result = harmonize_transcripts({})
    assert result.slots == []
    assert result.gap_rate == {}


def test_each_model_reports_its_own_span_in_a_slot() -> None:
    """``times`` must be per model, or a consumer cannot measure boundary disagreement.

    The first version of this field used a bare loop variable inside a comprehension, so every
    model reported the *last* member's span. The lattice still looked like a lattice while placing
    one word in two columns and losing another — the failure mode of a wrong answer that type-checks.
    """
    a = [(0.0, 0.4, "hi"), (0.5, 0.9, "there")]
    b = [(0.02, 0.45, "hi"), (0.55, 0.95, "there")]
    slots = harmonize_transcripts({"a": a, "b": b}).slots

    assert slots[0].times["a"] == (0.0, 0.4) and slots[0].times["b"] == (0.02, 0.45)
    assert slots[1].times["a"] == (0.5, 0.9) and slots[1].times["b"] == (0.55, 0.95)


def test_a_model_absent_from_a_slot_reports_no_span() -> None:
    """Absent is ``None``, so a consumer counts witnesses rather than inventing a boundary."""
    slots = harmonize_transcripts(
        {"a": [(0.0, 0.4, "hi"), (0.5, 0.9, "um"), (1.0, 1.4, "there")], "b": [(0.0, 0.4, "hi"), (1.0, 1.4, "there")]}
    ).slots
    filler = next(s for s in slots if s.words.get("a") == "um")
    assert filler.times["a"] == (0.5, 0.9) and filler.times["b"] is None


def test_a_slot_identifies_each_model_word_by_index_not_by_onset() -> None:
    """Onsets do not identify words, so the lattice carries the index.

    Measured on the 5-speaker clip: a recognizer placed "Josh" at ``[2.72, 2.72]`` — zero duration —
    and another placed two words starting at 2.72. A consumer rebuilding richer word objects by
    ``(model, onset)`` therefore fetched the wrong word, put one word in two columns and dropped
    another, turning "wanted to take" into "wanted take take". The index makes the rebuild exact.
    """
    a = [(0.0, 0.4, "hi"), (0.4, 0.4, "there"), (0.4, 0.9, "friend")]
    slots = harmonize_transcripts({"a": a, "b": list(a)}).slots

    assert [s.indices["a"] for s in slots] == [0, 1, 2]
    assert [s.words["a"] for s in slots] == ["hi", "there", "friend"]
    # Two words share onset 0.4; only the index tells them apart.
    shared = [s for s in slots if s.times["a"] is not None and s.times["a"][0] == 0.4]
    assert {s.indices["a"] for s in shared} == {1, 2}
