"""IPUs, backchannels, turns, floor transfers, overlaps and the turn-taking summary."""

from typing import Any, Dict, List

import pytest

from senselab.text.tasks.conversation_analysis import ConversationAnalysisConfig, analyze_conversation


def utterance(speaker: str, start: float, end: float, text: str) -> List[Dict[str, Any]]:
    """Contiguous words splitting ``[start, end]`` evenly, one per whitespace token."""
    tokens = text.split()
    step = (end - start) / len(tokens)
    return [
        {"speaker": speaker, "start": start + i * step, "end": start + (i + 1) * step, "text": tok}
        for i, tok in enumerate(tokens)
    ]


def test_ipus_split_on_intra_speaker_gap() -> None:
    """Words join across a 0.1 s gap and split across a 0.3 s gap."""
    words = [
        {"speaker": "A", "start": 0.0, "end": 0.5, "text": "first"},
        {"speaker": "A", "start": 0.6, "end": 1.0, "text": "second"},
        {"speaker": "A", "start": 1.3, "end": 1.6, "text": "third"},
    ]
    result = analyze_conversation(words)
    assert [(i.start, i.end, i.n_words) for i in result.ipus] == [(0.0, 1.0, 2), (1.3, 1.6, 1)]
    assert len(result.turns) == 1
    assert result.turns[0].ipu_ids == [0, 1]


def test_floor_transfer_offsets_and_types() -> None:
    """A gap, an overlap and a no-gap-no-overlap transition, with their exact FTOs."""
    words = (
        utterance("A", 0.0, 2.0, "we started the project last spring")
        + utterance("B", 2.5, 4.0, "and how did that go for everyone")
        + utterance("A", 3.8, 5.0, "it went quite well overall honestly")
        + utterance("B", 5.02, 6.0, "that sounds really good to hear")
    )
    result = analyze_conversation(words)
    assert [t.speaker for t in result.turns] == ["A", "B", "A", "B"]
    ftos = [f.fto for f in result.floor_transfers]
    assert ftos == pytest.approx([0.5, -0.2, 0.02])
    assert [f.transfer_type for f in result.floor_transfers] == ["gap", "overlap", "no_gap_no_overlap"]
    assert len(result.overlaps) == 1
    ov = result.overlaps[0]
    assert (ov.holder_speaker, ov.overlapper_speaker, ov.overlap_type) == ("B", "A", "non_competitive")
    assert (ov.start, ov.end) == pytest.approx((3.8, 4.0))
    assert ov.interruption_outcome is None

    summary = result.summary()
    assert summary["A"]["n_floor_takings"] == 1
    assert summary["A"]["fto_mean_s"] == pytest.approx(-0.2)
    assert summary["B"]["fto_mean_s"] == pytest.approx(0.26)
    assert summary["B"]["prop_gap"] == pytest.approx(0.5)
    assert summary["B"]["prop_no_gap_no_overlap"] == pytest.approx(0.5)
    assert summary["A"]["prop_overlap"] == pytest.approx(1.0)


def test_successful_interruption() -> None:
    """B starts mid-IPU; A stops 1.5 s later and B outlasts A."""
    words = utterance("A", 0.0, 3.5, "I was going to say that the plan was") + utterance(
        "B", 2.0, 6.0, "wait I really need to stop you there because that is wrong"
    )
    result = analyze_conversation(words)
    assert len(result.overlaps) == 1
    ov = result.overlaps[0]
    assert ov.overlap_type == "competitive"
    assert ov.interruption_outcome == "successful"
    assert (ov.holder_speaker, ov.overlapper_speaker) == ("A", "B")
    assert result.floor_transfers[0].fto == pytest.approx(-1.5)
    s = result.summary()
    assert s["B"]["interruptions_made_successful"] == 1
    assert s["A"]["interruptions_received_successful"] == 1
    assert s["B"]["floor_holding_index"] == pytest.approx(1.0)
    assert s["A"]["floor_holding_index"] == pytest.approx(0.0)


def test_unsuccessful_interruption_is_embedded_turn() -> None:
    """B's non-backchannel IPU inside A's turn does not take the floor."""
    words = utterance("A", 0.0, 6.0, "so the main thing we learned from the study was that people adapt") + utterance(
        "B", 2.0, 3.0, "no that is wrong"
    )
    result = analyze_conversation(words)
    assert [(t.speaker, t.embedded) for t in result.turns] == [("A", False), ("B", True)]
    assert result.floor_transfers == []
    ov = result.overlaps[0]
    assert (ov.overlap_type, ov.interruption_outcome) == ("competitive", "unsuccessful")
    s = result.summary()
    assert s["A"]["interruptions_received_unsuccessful"] == 1
    assert s["A"]["floor_holding_index"] == pytest.approx(1.0)
    assert s["B"]["interruptions_made_unsuccessful"] == 1


def test_backchannels_overlapping_and_in_pause() -> None:
    """An overlapping "mm-hm", an in-pause "yeah" and a wholly overlapped non-lexical IPU."""
    words = (
        utterance("A", 0.0, 3.0, "I grew up in a small town by the coast")
        + utterance("B", 1.0, 1.3, "mm-hm")
        + utterance("B", 1.8, 2.3, "the coast")
        + utterance("A", 3.5, 4.0, "near the harbour")
        + utterance("B", 4.1, 4.4, "yeah.")
        + utterance("A", 4.6, 6.0, "and we moved away later")
    )
    result = analyze_conversation(words)
    bcs = [(i.text, i.backchannel_type) for i in result.ipus if i.is_backchannel]
    assert bcs == [("mm-hm", "overlapping"), ("the coast", "overlapping"), ("yeah.", "in_pause")]
    assert len(result.turns) == 1
    assert result.turns[0].ipu_ids == [i.id for i in result.ipus if i.speaker == "A"]
    assert result.floor_transfers == []
    assert {o.overlap_type for o in result.overlaps} == {"backchannel"}
    s = result.summary()
    assert s["B"]["n_backchannels"] == 3
    assert s["B"]["n_backchannels_overlapping"] == 2
    assert s["B"]["n_backchannels_in_pause"] == 1
    assert s["B"]["backchannels_per_min"] == pytest.approx(3 / (6.0 / 60))
    assert s["B"]["n_turns"] == 0


def test_short_lexical_ipu_that_takes_floor_is_not_backchannel() -> None:
    """A "yeah" followed by the same speaker continuing, the other silent, is a turn onset."""
    words = (
        utterance("A", 0.0, 2.0, "that was the whole story then")
        + utterance("B", 2.1, 2.4, "yeah")
        + utterance("B", 2.7, 5.0, "I think that is right actually")
    )
    result = analyze_conversation(words)
    assert not any(i.is_backchannel for i in result.ipus)
    assert [t.speaker for t in result.turns] == ["A", "B"]
    assert result.floor_transfers[0].fto == pytest.approx(0.1)


def test_answer_to_question_is_not_backchannel() -> None:
    """A bare "Yes." after a question is an answer, even when the asker resumes quickly."""
    words = (
        utterance("A", 0.0, 1.5, "Do you like your job?")
        + utterance("B", 1.8, 2.1, "Yes.")
        + utterance("A", 2.5, 4.0, "What do you like about it?")
    )
    result = analyze_conversation(words)
    assert not any(i.is_backchannel for i in result.ipus)
    assert [t.dialogue_act for t in result.turns] == ["yes_no_question", "agreement", "wh_question"]
    assert result.floor_transfers[0].after_question
    s = result.summary()
    assert s["B"]["n_responses_after_question"] == 1
    assert s["B"]["response_latency_after_question_mean_s"] == pytest.approx(0.3)
    assert s["A"]["n_responses_after_question"] == 0
    assert s["A"]["response_latency_after_question_mean_s"] is None


def test_speaker_summary_turn_metrics() -> None:
    """Speaking time, share, turn counts and turn-length statistics."""
    words = (
        utterance("A", 0.0, 2.0, "one two three four")
        + utterance("B", 3.0, 4.0, "five six seven eight")
        + utterance("A", 5.0, 9.0, "nine ten eleven twelve thirteen fourteen")
    )
    result = analyze_conversation(words)
    s = result.summary()
    assert s["A"]["speaking_time_s"] == pytest.approx(6.0)
    assert s["A"]["speaking_share"] == pytest.approx(6.0 / 7.0)
    assert s["A"]["n_turns"] == 2
    assert s["A"]["turns_per_min"] == pytest.approx(2 / (9.0 / 60))
    assert s["A"]["turn_duration_mean_s"] == pytest.approx(3.0)
    assert s["A"]["turn_duration_median_s"] == pytest.approx(3.0)
    assert s["A"]["turn_duration_sd_s"] == pytest.approx(2**0.5)
    assert s["A"]["turn_words_mean"] == pytest.approx(5.0)
    assert s["A"]["longest_monologue_s"] == pytest.approx(4.0)
    assert s["A"]["longest_monologue_words"] == 6
    assert s["B"]["turn_duration_sd_s"] is None
    assert s["B"]["fto_mean_s"] == pytest.approx(1.0)
    assert s["A"]["fto_mean_s"] == pytest.approx(1.0)


def test_config_threshold_override() -> None:
    """A larger ``ipu_gap_s`` merges what the default splits."""
    words = [
        {"speaker": "A", "start": 0.0, "end": 0.5, "text": "first"},
        {"speaker": "A", "start": 0.8, "end": 1.0, "text": "second"},
    ]
    assert len(analyze_conversation(words).ipus) == 2
    config = ConversationAnalysisConfig(ipu_gap_s=0.5)
    assert len(analyze_conversation(words, config=config).ipus) == 1
    assert ConversationAnalysisConfig().ipu_gap_s == pytest.approx(0.2)
