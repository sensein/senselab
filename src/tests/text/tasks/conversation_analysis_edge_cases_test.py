"""Edge cases: empty input, one speaker, three speakers, zero-duration words, unsorted input, attribution."""

import random
from typing import Any, Dict, List

import pytest

from senselab.text.tasks.conversation_analysis import analyze_conversation
from senselab.utils.data_structures import ScriptLine

TABLES = {
    "words",
    "ipus",
    "turns",
    "overlaps",
    "floor_transfers",
    "disfluencies",
    "dialogue_acts",
    "markers",
    "summary",
}


def utterance(speaker: str, start: float, end: float, text: str) -> List[Dict[str, Any]]:
    """Contiguous words splitting ``[start, end]`` evenly."""
    tokens = text.split()
    step = (end - start) / len(tokens)
    return [
        {"speaker": speaker, "start": start + i * step, "end": start + (i + 1) * step, "text": tok}
        for i, tok in enumerate(tokens)
    ]


def test_empty_input() -> None:
    """No words gives empty tables with their columns."""
    result = analyze_conversation([])
    assert result.speakers == [] and result.turns == [] and result.summary() == {}
    frames = result.to_dataframes()
    assert set(frames) == TABLES
    assert "fto" in frames["floor_transfers"].columns
    assert frames["turns"].empty


def test_single_speaker() -> None:
    """One speaker: no transfers, overlaps or backchannels; share 1."""
    result = analyze_conversation(
        utterance("A", 0.0, 3.0, "just me talking um about things") + utterance("A", 4.0, 5.0, "okay")
    )
    assert result.floor_transfers == [] and result.overlaps == []
    assert not any(i.is_backchannel for i in result.ipus)
    s = result.summary()["A"]
    assert s["speaking_share"] == pytest.approx(1.0)
    assert s["n_turns"] == 1
    assert s["fto_mean_s"] is None
    assert s["floor_holding_index"] is None
    assert s["n_filled_pauses"] == 1


def test_three_speakers() -> None:
    """Three parties: transfers between each consecutive floor holder."""
    words = (
        utterance("A", 0.0, 2.0, "let us begin the meeting now")
        + utterance("B", 2.3, 4.0, "I have the first agenda item")
        + utterance("C", 3.9, 6.0, "and I have the second agenda item")
        + utterance("A", 6.5, 7.5, "great thanks both of you")
    )
    result = analyze_conversation(words)
    assert result.speakers == ["A", "B", "C"]
    assert [(f.from_speaker, f.to_speaker) for f in result.floor_transfers] == [("A", "B"), ("B", "C"), ("C", "A")]
    assert [f.fto for f in result.floor_transfers] == pytest.approx([0.3, -0.1, 0.5])
    assert [f.transfer_type for f in result.floor_transfers] == ["gap", "overlap", "gap"]
    assert set(result.summary()) == {"A", "B", "C"}
    assert sum(s["speaking_share"] for s in result.summary().values()) == pytest.approx(1.0)


def test_zero_duration_words() -> None:
    """Zero-duration words are kept, merged into IPUs and excluded from prolongation statistics."""
    words = [{"speaker": "A", "start": 0.1 * i, "end": 0.1 * i, "text": f"w{i}"} for i in range(12)]
    words += [
        {"speaker": "B", "start": 2.0, "end": 2.0, "text": "uh"},
        {"speaker": "B", "start": 2.0, "end": 2.5, "text": "right"},
    ]
    result = analyze_conversation(words)
    assert len(result.words) == 14
    assert [i.n_words for i in result.ipus if i.speaker == "A"] == [12]
    assert not [d for d in result.disfluencies if d.disfluency_type == "prolongation"]
    assert [d.text for d in result.disfluencies if d.disfluency_type == "filled_pause"] == ["uh"]


def test_unsorted_input_matches_sorted() -> None:
    """Input order does not change any table."""
    words = (
        utterance("A", 0.0, 2.0, "we started um the project")
        + utterance("B", 1.0, 1.3, "mm-hm")
        + utterance("B", 2.4, 4.0, "how did it go?")
        + utterance("A", 4.2, 6.0, "really well I I think")
    )
    shuffled = list(words)
    random.Random(0).shuffle(shuffled)
    a = analyze_conversation(words).to_dataframes()
    b = analyze_conversation(shuffled).to_dataframes()
    for name in TABLES - {"summary"}:
        assert a[name].astype(str).equals(b[name].astype(str)), name
    assert a["summary"].astype(str).equals(b["summary"].astype(str))


def test_speaker_from_diarization() -> None:
    """Words without a speaker are attributed by the diarization segment overlapping them most."""
    words = [
        {"start": 0.0, "end": 0.5, "text": "hello"},
        {"start": 0.5, "end": 1.0, "text": "there."},
        {"start": 1.4, "end": 1.8, "text": "hi"},
    ]
    diarization = [
        ScriptLine(speaker="spk0", start=0.0, end=1.1),
        ScriptLine(speaker="spk1", start=1.3, end=2.0),
    ]
    result = analyze_conversation(words, diarization=diarization)
    assert [w.speaker for w in result.words] == ["spk0", "spk0", "spk1"]
    assert result.summary()["spk0"]["diarized_speaking_time_s"] == pytest.approx(1.1)


def test_invalid_words_raise() -> None:
    """Missing speaker without diarization, missing times and reversed times are errors."""
    with pytest.raises(ValueError, match="no speaker"):
        analyze_conversation([{"start": 0.0, "end": 1.0, "text": "hi"}])
    with pytest.raises(ValueError, match="no start/end"):
        analyze_conversation([{"speaker": "A", "start": None, "end": 1.0, "text": "hi"}])
    with pytest.raises(ValueError, match="ends before"):
        analyze_conversation([{"speaker": "A", "start": 2.0, "end": 1.0, "text": "hi"}])


def test_scriptline_words_and_confidence() -> None:
    """``ScriptLine`` words carry their score through as confidence."""
    lines = [ScriptLine(speaker="A", start=0.0, end=0.4, text="hello", score=0.9)]
    result = analyze_conversation(lines)
    assert result.words[0].confidence == pytest.approx(0.9)
