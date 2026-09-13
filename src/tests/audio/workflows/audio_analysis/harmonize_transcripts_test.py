"""H3 — put several ASR transcripts into one word space.

Each transcript is already aligned to the *audio*, independently. That is not enough to compare
them: two models that heard the same sentence can still disagree about where a word starts, and one
that inserted a word shifts every timestamp after it. Aligning the transcripts to *each other*
gives a slot structure in which "these three models produced different words for the same position"
is expressible at all — which is what the ASR axis needs and what a per-window WER cannot say.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.harmonize import TranscriptSlot, _align_pair, harmonize_transcripts


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


def _shape(slots: list[TranscriptSlot]) -> list[tuple[str | None, str | None]]:
    """``(cw, qwen)`` surface forms per slot, ``None`` for a gap."""
    return [(s.words.get("cw"), s.words.get("qwen")) for s in slots]


def test_a_filler_and_a_tail_word_are_two_insertions_around_an_agreement() -> None:
    """CW "I uh think" against Qwen "I think so": "uh" and "so" are insertions, "think" is agreed.

    Under unit costs with a diagonal-first backtrace this came out as {I,I} {uh,think} {think,so} —
    two substitutions in place of two insertions, and "think" never recorded as agreed.
    """
    cw = _w([(0.0, 0.2, "I"), (0.25, 0.45, "uh"), (0.5, 0.9, "think")])
    qwen = _w([(0.0, 0.2, "I"), (0.3, 0.9, "think"), (0.9, 1.1, "so")])
    slots = harmonize_transcripts({"cw": cw, "qwen": qwen}).slots

    assert _shape(slots) == [("I", "I"), ("uh", None), ("think", "think"), (None, "so")]
    assert [s.consensus for s in slots] == ["I", "uh", "think", "so"]


def test_a_leading_insertion_does_not_turn_the_rest_into_substitutions() -> None:
    """CW "oh I uh think" against Qwen "I think so": three insertions around two agreements."""
    cw = _w([(0.0, 0.1, "oh"), (0.1, 0.2, "I"), (0.25, 0.45, "uh"), (0.5, 0.9, "think")])
    qwen = _w([(0.1, 0.2, "I"), (0.3, 0.9, "think"), (0.9, 1.1, "so")])
    slots = harmonize_transcripts({"cw": cw, "qwen": qwen}).slots

    assert _shape(slots) == [("oh", None), ("I", "I"), ("uh", None), ("think", "think"), (None, "so")]


def test_a_one_for_one_substitution_stays_one_slot_with_an_insertion_nearby() -> None:
    """A single differing word is a substitution, not a deletion plus an insertion.

    The insertion sits one word away from the substitution so the alignment is unambiguous; a cost
    model that under-priced indels would split "think"/"thing" into two single-source slots.
    """
    cw = _w([(0.0, 0.2, "I"), (0.25, 0.45, "uh"), (0.5, 0.8, "really"), (0.8, 1.1, "think")])
    qwen = _w([(0.0, 0.2, "I"), (0.3, 0.8, "really"), (0.8, 1.1, "thing")])
    slots = harmonize_transcripts({"cw": cw, "qwen": qwen}).slots

    assert _shape(slots) == [("I", "I"), ("uh", None), ("really", "really"), ("think", "thing")]
    assert slots[3].consensus is None and slots[3].disagreement > 0.0

    plain_cw = _w([(0.0, 0.4, "the"), (0.4, 0.9, "cat"), (0.9, 1.4, "sat")])
    plain_qwen = _w([(0.0, 0.4, "the"), (0.4, 0.9, "bat"), (0.9, 1.4, "sat")])
    assert _shape(harmonize_transcripts({"cw": plain_cw, "qwen": plain_qwen}).slots) == [
        ("the", "the"),
        ("cat", "bat"),
        ("sat", "sat"),
    ]


def test_a_substitution_beside_an_insertion_lands_on_the_earlier_token() -> None:
    """CW "I uh think" against Qwen "I thing": the substitution pairs "uh" with "thing".

    Which of two adjacent tokens is the substituted one is a tie under the cost model, and the
    backtrace resolves it toward the earlier token. Pinned as behaviour, not as correctness — see
    ``specs/20260817-triage-workflow-dag/transcript-alignment.md``.
    """
    cw = _w([(0.0, 0.2, "I"), (0.25, 0.45, "uh"), (0.5, 0.9, "think")])
    qwen = _w([(0.0, 0.2, "I"), (0.3, 0.9, "thing")])
    slots = harmonize_transcripts({"cw": cw, "qwen": qwen}).slots

    assert _shape(slots) == [("I", "I"), ("uh", "thing"), ("think", None)]


def test_a_repetition_against_one_token_aligns_its_last_copy() -> None:
    """Three copies of "the" against one: the last copy is agreed, the earlier copies are insertions.

    Pinned because the fused transcript's handling of repeated words depends on which copy carries
    two sources; it holds in both directions.
    """
    cw = _w([(0.0, 0.2, "the"), (0.3, 0.5, "the"), (0.6, 0.8, "the")])
    qwen = _w([(0.6, 0.8, "the")])

    forward = harmonize_transcripts({"cw": cw, "qwen": qwen}).slots
    assert _shape(forward) == [("the", None), ("the", None), ("the", "the")]
    assert [s.indices["cw"] for s in forward] == [0, 1, 2] and forward[2].indices["qwen"] == 0

    mirrored = harmonize_transcripts({"cw": qwen, "qwen": cw}).slots
    assert _shape(mirrored) == [(None, "the"), (None, "the"), ("the", "the")]
    assert mirrored[2].indices == {"cw": 0, "qwen": 2}


def test_without_timings_a_repetition_still_aligns_its_last_copy() -> None:
    """``_align_pair`` with no timings is the untouched rule: last copy, earlier token on a substitution tie."""
    assert _align_pair(["the", "the", "the"], ["the"]) == [(0, None), (1, None), (2, 0)]
    assert _align_pair(["the"], ["the", "the", "the"]) == [(None, 0), (None, 1), (0, 2)]
    assert _align_pair(["i", "uh", "think"], ["i", "thing"]) == [(0, 0), (1, 1), (2, None)]


def test_the_gets_case_pairs_the_temporally_coincident_first_copy() -> None:
    """CW "gets a- gets" against Qwen "gets", with the recording's timings: Qwen pairs with the first copy.

    Both pairings cost 6; Qwen's ``gets``@9.60-9.84 overlaps CW's first copy at 9.66-9.74 and sits
    0.15 s from the second, so time breaks the tie toward the first.
    """
    cw = _w([(9.66, 9.74, "gets"), (9.88, 9.99, "a-"), (9.99, 10.10, "gets")])
    qwen = _w([(9.60, 9.84, "gets")])
    slots = harmonize_transcripts({"cw": cw, "qwen": qwen}).slots

    assert _shape(slots) == [("gets", "gets"), ("a-", None), ("gets", None)]
    assert slots[0].indices == {"cw": 0, "qwen": 0}

    mirrored = harmonize_transcripts({"cw": qwen, "qwen": cw}).slots
    assert _shape(mirrored) == [("gets", "gets"), (None, "a-"), (None, "gets")]
    assert mirrored[0].indices == {"cw": 0, "qwen": 0}


def test_time_picks_whichever_copy_coincides_and_falls_back_to_the_last_when_it_cannot() -> None:
    """The copy chosen follows the single token's span, so it is time being read, not a fixed position."""
    cw = _w([(0.0, 0.2, "the"), (0.3, 0.5, "the"), (0.6, 0.8, "the")])

    last = harmonize_transcripts({"cw": cw, "qwen": _w([(0.62, 0.78, "the")])}).slots
    assert [s.indices["qwen"] for s in last] == [None, None, 0]

    middle = harmonize_transcripts({"cw": cw, "qwen": _w([(0.32, 0.48, "the")])}).slots
    assert [s.indices["qwen"] for s in middle] == [None, 0, None]

    # A span covering every copy separates none of them: the untimed rule stands and the last copy is taken.
    covering = harmonize_transcripts({"cw": cw, "qwen": _w([(0.0, 0.8, "the")])}).slots
    assert [s.indices["qwen"] for s in covering] == [None, None, 0]

    # Nearest wins by interval distance when no copy overlaps: 0.51-0.55 is 10 ms past the middle
    # copy's end and 50 ms short of the last copy's start, so the middle copy is taken.
    between = harmonize_transcripts({"cw": cw, "qwen": _w([(0.51, 0.55, "the")])}).slots
    assert [s.indices["qwen"] for s in between] == [None, 0, None]
    nearer_last = harmonize_transcripts({"cw": cw, "qwen": _w([(0.57, 0.59, "the")])}).slots
    assert [s.indices["qwen"] for s in nearer_last] == [None, None, 0]
    nearer_first = harmonize_transcripts({"cw": cw, "qwen": _w([(0.21, 0.24, "the")])}).slots
    assert [s.indices["qwen"] for s in nearer_first] == [0, None, None]


def test_time_never_buys_a_costlier_path() -> None:
    """``the the`` at 0 s and 5 s against ``the cat`` at 5 s: cost pairs the first ``the`` 4.6 s away.

    ``match + substitution`` costs 4; the time-preferred ``deletion + match + insertion`` costs 6, so
    the cheaper path wins although its pairing is the more distant one.
    """
    a, a_times = ["the", "the"], [(0.0, 0.4), (5.0, 5.4)]
    b, b_times = ["the", "cat"], [(5.0, 5.4), (5.5, 5.9)]
    timed = _align_pair(a, b, a_times, b_times)
    assert timed == [(0, 0), (1, 1)] == _align_pair(a, b)
    assert _align_pair(a, b, a_times, b_times) == timed, "deterministic on repeat"


def test_time_breaks_a_substitution_position_tie_when_the_spans_separate_the_tokens() -> None:
    """CW "I uh think" against Qwen "I thing": the substitution lands on the token ``thing`` overlaps.

    Placed over ``think`` it pairs with ``think``; placed over ``uh`` it pairs with ``uh``; spanning
    both, the untimed earlier-token rule stands.
    """
    cw = _w([(0.0, 0.2, "I"), (0.25, 0.45, "uh"), (0.5, 0.9, "think")])
    over_think = harmonize_transcripts({"cw": cw, "qwen": _w([(0.0, 0.2, "I"), (0.55, 0.9, "thing")])}).slots
    assert _shape(over_think) == [("I", "I"), ("uh", None), ("think", "thing")]
    over_uh = harmonize_transcripts({"cw": cw, "qwen": _w([(0.0, 0.2, "I"), (0.25, 0.45, "thing")])}).slots
    assert _shape(over_uh) == [("I", "I"), ("uh", "thing"), ("think", None)]
    spanning = harmonize_transcripts({"cw": cw, "qwen": _w([(0.0, 0.2, "I"), (0.3, 0.9, "thing")])}).slots
    assert _shape(spanning) == [("I", "I"), ("uh", "thing"), ("think", None)]


def test_timings_must_be_one_span_per_token() -> None:
    """A timing list of the wrong length is refused rather than silently misread."""
    with pytest.raises(ValueError, match="one span per token"):
        _align_pair(["a", "b"], ["a"], [(0.0, 0.1)], [(0.0, 0.1)])
