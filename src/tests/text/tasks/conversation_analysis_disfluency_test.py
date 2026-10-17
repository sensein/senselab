"""Filled pauses, repetitions, cut-offs, prolongations, and fillers surviving end to end."""

from typing import Any, Dict, List

import numpy as np
import pytest

from senselab.text.tasks.conversation_analysis import analyze_conversation, load_default_lexicons
from senselab.text.tasks.conversation_analysis.disfluency import filler_family, is_cutoff
from senselab.text.tasks.conversation_analysis.text_utils import normalize_token
from senselab.utils.data_structures import ScriptLine


def w(speaker: str, start: float, end: float, text: str) -> Dict[str, Any]:
    """One word."""
    return {"speaker": speaker, "start": start, "end": end, "text": text}


def filler_conversation() -> List[Dict[str, Any]]:
    """A's turn with three fillers, a repetition and a cut-off, then B's turn."""
    return [
        w("A", 0.0, 0.3, "[UH]"),
        w("A", 0.3, 0.4, "I"),
        w("A", 0.4, 0.5, "I"),
        w("A", 0.5, 0.8, "went"),
        w("A", 0.8, 0.9, "to"),
        w("A", 0.9, 1.0, "the"),
        w("A", 1.0, 1.4, "um,"),
        w("A", 1.9, 2.3, "store"),
        w("A", 2.3, 2.4, "wh-"),
        w("A", 2.4, 2.7, "where"),
        w("A", 2.7, 3.0, "uh"),
        w("B", 3.5, 3.7, "Okay"),
        w("B", 3.7, 3.9, "so"),
        w("B", 3.9, 4.1, "tell"),
        w("B", 4.1, 4.3, "me"),
        w("B", 4.3, 4.5, "more."),
    ]


@pytest.mark.parametrize(
    "text,family",
    [("[UH]", "uh"), ("[um]", "um"), ("Um,", "um"), ("UHM", "um"), ("er", "uh"), ("hmm", "other"), ("umbrella", None)],
)
def test_filler_family_is_case_and_bracket_insensitive(text: str, family: str) -> None:
    """CrisperWhisper bracketed forms and plain forms map to the same family."""
    assert filler_family(normalize_token(text), load_default_lexicons()) == family


def test_cutoff_detection() -> None:
    """A trailing hyphen after letters is a cut-off; an internal hyphen is not."""
    assert is_cutoff(normalize_token("wh-"))
    assert is_cutoff(normalize_token("basi—"))
    assert not is_cutoff(normalize_token("mm-hm"))
    assert not is_cutoff(normalize_token("-"))


def test_filler_positions_pauses_and_transitions() -> None:
    """Exact position, surrounding pauses and next event of each filler."""
    result = analyze_conversation(filler_conversation())
    fillers = [d for d in result.disfluencies if d.disfluency_type == "filled_pause"]
    assert [d.text for d in fillers] == ["[UH]", "um,", "uh"]
    assert [d.filler_family for d in fillers] == ["uh", "um", "uh"]
    assert [d.position for d in fillers] == ["turn_initial", "turn_medial", "turn_final"]
    assert fillers[0].pause_before is None
    assert fillers[0].pause_after == pytest.approx(0.0)
    assert fillers[1].pause_after == pytest.approx(0.5)
    assert fillers[1].pre_pause is True
    assert fillers[1].post_pause is False
    assert fillers[2].pause_after is None
    assert [d.next_event for d in fillers] == ["word", "pause", "speaker_change"]
    assert [d.duration for d in fillers] == pytest.approx([0.3, 0.4, 0.3])

    s = result.summary()["A"]
    assert s["n_filled_pauses"] == 3
    assert s["n_turn_words"] == 11
    assert s["filled_pauses_per_100_words"] == pytest.approx(300 / 11)
    assert s["filled_pauses_per_min"] == pytest.approx(3 / (2.5 / 60))
    assert s["n_uh"] == 2 and s["n_um"] == 1
    assert s["uh_um_ratio"] == pytest.approx(2.0)
    assert s["filler_duration_mean_s"] == pytest.approx(1.0 / 3)
    assert s["fillers_turn_initial"] == 1
    assert s["fillers_turn_medial"] == 1
    assert s["fillers_turn_final"] == 1
    assert s["fillers_pre_pause"] == 1
    assert s["p_filler_to_pause"] == pytest.approx(1 / 3)
    assert s["p_filler_to_speaker_change"] == pytest.approx(1 / 3)


def test_repetition_and_cutoff_counts() -> None:
    """One repetition for "I I"; "where" after "wh-" is not a repetition."""
    result = analyze_conversation(filler_conversation())
    reps = [d for d in result.disfluencies if d.disfluency_type == "repetition"]
    cuts = [d for d in result.disfluencies if d.disfluency_type == "cutoff"]
    assert [d.text for d in reps] == ["I I"]
    assert [d.word_ids for d in reps] == [[1, 2]]
    assert [d.text for d in cuts] == ["wh-"]
    s = result.summary()["A"]
    assert s["n_repetitions"] == 1
    assert s["n_cutoffs"] == 1
    assert s["n_disfluencies"] == 5
    assert s["disfluencies_per_100_words"] == pytest.approx(500 / 11)


def test_multiword_repetition_across_filler_and_exempt_repeats() -> None:
    """One 2-gram repetition in "I want uh I want"; "yeah yeah" and "very very" are not disfluent."""
    texts = "I want uh I want to go yeah yeah very very much".split()
    words = [w("A", i * 0.3, i * 0.3 + 0.3, t) for i, t in enumerate(texts)]
    result = analyze_conversation(words)
    reps = [d.text for d in result.disfluencies if d.disfluency_type == "repetition"]
    assert reps == ["I want I want"]


def test_prolongation_zscore() -> None:
    """One word four times longer per character than the speaker's other eleven is a prolongation."""
    texts = "big red cat ran far and hid now saw two old dog".split()
    words = []
    t = 0.0
    for i, text in enumerate(texts):
        duration = 1.2 if i == len(texts) - 1 else 0.3
        words.append(w("A", t, t + duration, text))
        t += duration
    result = analyze_conversation(words)
    pro = [d for d in result.disfluencies if d.disfluency_type == "prolongation"]
    per_char = np.array([0.1] * 11 + [0.4])
    expected = (0.4 - per_char.mean()) / per_char.std()
    assert [d.text for d in pro] == ["dog"]
    assert pro[0].zscore == pytest.approx(expected)


def test_prolongation_needs_enough_words() -> None:
    """Below ``prolongation_min_words`` no prolongation is reported."""
    words = [w("A", 0.0, 0.3, "big"), w("A", 0.3, 0.6, "red"), w("A", 0.6, 3.0, "dog")]
    result = analyze_conversation(words)
    assert not [d for d in result.disfluencies if d.disfluency_type == "prolongation"]


def test_backchannel_hm_is_not_a_filler() -> None:
    """A listener's "hm" backchannel is not counted as that listener's filled pause."""
    words = [w("A", i * 0.3, i * 0.3 + 0.3, t) for i, t in enumerate("we went there in the summer".split())]
    words.append(w("B", 0.5, 0.8, "hm"))
    result = analyze_conversation(words)
    assert result.summary()["B"]["n_filled_pauses"] == 0
    assert result.summary()["B"]["n_backchannels"] == 1


def test_fillers_survive_end_to_end() -> None:
    """Every filler reaches the words, IPUs, turns, dialogue-act segments, disfluencies and every table."""
    lines = [
        ScriptLine(
            speaker="A",
            start=0.0,
            end=3.0,
            text="[UH] I I went to the um, store wh- where uh",
            chunks=[ScriptLine(text=d["text"], start=d["start"], end=d["end"]) for d in filler_conversation()[:11]],
        ),
        *[ScriptLine(speaker="B", text=d["text"], start=d["start"], end=d["end"]) for d in filler_conversation()[11:]],
    ]
    result = analyze_conversation(lines)
    fillers = {"[UH]", "um,", "uh"}
    assert fillers <= {x.text for x in result.words}
    turn_a = result.turns[0]
    assert fillers <= set(turn_a.words)
    assert turn_a.text == "[UH] I I went to the um, store wh- where uh"
    ipu_text = " ".join(i.text for i in result.ipus if i.speaker == "A")
    assert all(f in ipu_text for f in fillers)
    assert all(f in " ".join(s.text for s in result.dialogue_acts) for f in fillers)
    assert {d.text for d in result.disfluencies if d.disfluency_type == "filled_pause"} == fillers
    frames = result.to_dataframes()
    assert fillers <= set(frames["words"]["text"])
    assert fillers <= set(frames["disfluencies"]["text"])
    assert "[UH]" in frames["turns"].loc[0, "text"]
