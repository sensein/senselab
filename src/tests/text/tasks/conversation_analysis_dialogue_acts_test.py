"""Rule-based dialogue acts, the pluggable labeller, and politeness/engagement/alignment markers."""

from typing import Any, Dict, List, Sequence

import pytest

from senselab.text.tasks.conversation_analysis import (
    ConversationAnalysisConfig,
    DialogueActLabel,
    DialogueActSegment,
    analyze_conversation,
    content_words,
    load_default_lexicons,
    politeness_counts,
    primary_act,
    rule_act,
)


def utterance(speaker: str, start: float, end: float, text: str) -> List[Dict[str, Any]]:
    """Contiguous words splitting ``[start, end]`` evenly."""
    tokens = text.split()
    step = (end - start) / len(tokens)
    return [
        {"speaker": speaker, "start": start + i * step, "end": start + (i + 1) * step, "text": tok}
        for i, tok in enumerate(tokens)
    ]


@pytest.mark.parametrize(
    "text,act",
    [
        ("What did you do yesterday?", "wh_question"),
        ("So what do you think?", "wh_question"),
        ("Hello, how are you?", "wh_question"),
        ("What I mean is fine.", "statement"),
        ("Do you like it?", "yes_no_question"),
        ("Is that right?", "yes_no_question"),
        ("You went there?", "declarative_question"),
        ("It was fun, right?", "declarative_question"),
        ("Could you tell me more?", "directive"),
        ("Can you describe your day?", "directive"),
        ("Please describe your day.", "directive"),
        ("Tell me about your work.", "directive"),
        ("Okay, so tell me more.", "directive"),
        ("I'd like you to read this sentence.", "directive"),
        ("Let's start with your name.", "directive"),
        ("Mm-hm.", "backchannel"),
        ("Hmm.", "backchannel"),
        ("Thank you so much.", "thanking"),
        ("Sorry about that.", "apology"),
        ("Hello there.", "greeting_closing"),
        ("Okay, bye!", "greeting_closing"),
        ("I disagree with that.", "disagreement"),
        ("No.", "disagreement"),
        ("Exactly.", "agreement"),
        ("That's right, I agree.", "agreement"),
        ("I went to the store.", "statement"),
        ("Yeah, I think so.", "statement"),
    ],
)
def test_rule_acts_on_canned_sentences(text: str, act: str) -> None:
    """Each canned sentence gets its rule-layer act."""
    label, confidence = rule_act(text, load_default_lexicons())
    assert label == act
    assert 0.0 < confidence <= 1.0


def test_bare_yes_after_question_is_agreement() -> None:
    """Context: "Yes." after a yes-no question is agreement, otherwise a backchannel."""
    lex = load_default_lexicons()
    assert rule_act("Yes.", lex, previous_other_act="yes_no_question")[0] == "agreement"
    assert rule_act("Yes.", lex)[0] == "backchannel"


def test_turns_split_into_segments_and_get_primary_act() -> None:
    """A turn splits at sentence-final punctuation; its primary act is the highest-priority segment act."""
    words = utterance("A", 0.0, 3.0, "Thanks for coming. What brings you here today?") + utterance(
        "B", 3.5, 5.0, "I wanted to talk about work."
    )
    result = analyze_conversation(words)
    segs = [(s.turn_id, s.segment_index, s.act) for s in result.dialogue_acts]
    assert segs == [(0, 0, "thanking"), (0, 1, "wh_question"), (1, 0, "statement")]
    assert [t.dialogue_act for t in result.turns] == ["wh_question", "statement"]
    assert all(s.source == "rules" for s in result.dialogue_acts)
    assert result.summary()["A"]["questions_asked"] == 1
    assert result.summary()["A"]["act_thanking"] == 1
    assert result.summary()["B"]["act_wh_question"] == 0
    frame = result.to_dataframes()["dialogue_acts"]
    assert list(frame["act"]) == ["thanking", "wh_question", "statement"]


def segments(*labelled: tuple) -> List[DialogueActSegment]:
    """Labelled segments of one turn from ``(text, act)`` pairs."""
    return [
        DialogueActSegment(id=i, speaker="B", start=float(i), end=i + 1.0, text=text, word_ids=[], turn_id=0, act=act)
        for i, (text, act) in enumerate(labelled)
    ]


def test_content_words_skip_stopwords_and_fillers() -> None:
    """Stopwords and fillers are not content words."""
    lex = load_default_lexicons()
    assert content_words("Yes I do.", lex) == 0
    assert content_words("Um, I work as a nurse.", lex) == 2


@pytest.mark.parametrize(
    "labelled,expected",
    [
        ((("Sure.", "backchannel"), ("I work as a nurse.", "statement")), "statement"),
        ((("Yeah.", "agreement"), ("I work as a nurse.", "statement")), "statement"),
        ((("Mm-hm.", "backchannel"), ("Where do you work?", "wh_question")), "wh_question"),
        ((("Right.", "agreement"), ("Tell me about your job.", "directive")), "directive"),
        ((("Sure.", "backchannel"), ("Yes I do.", "statement")), "backchannel"),
        ((("Exactly.", "agreement"), ("I see.", "statement")), "agreement"),
        ((("Thanks.", "thanking"), ("I work as a nurse.", "statement")), "thanking"),
        ((("No.", "disagreement"), ("I work as a nurse.", "statement")), "disagreement"),
        ((("Sure.", "backchannel"),), "backchannel"),
        ((("Hmm.", "backchannel"), ("Okay.", "agreement")), "agreement"),
        ((), None),
    ],
)
def test_primary_act_precedence(labelled: tuple, expected: str) -> None:
    """Backchannel/agreement never head a turn that also holds a substantive question/directive/statement."""
    assert primary_act(segments(*labelled)) == expected


def test_primary_act_content_threshold_is_configurable() -> None:
    """A one-content-word statement is substantive only when the threshold is lowered to 1."""
    turn = segments(("Sure.", "backchannel"), ("Nursing.", "statement"))
    assert primary_act(turn) == "backchannel"
    assert primary_act(turn, ConversationAnalysisConfig(substantive_min_content_words=1)) == "statement"


def test_backchannel_led_answer_is_a_statement_turn() -> None:
    """End to end: "Sure. I work as a nurse at the hospital." answering a question is a statement turn."""
    words = utterance("A", 0.0, 2.0, "What do you do for work?") + utterance(
        "B", 2.5, 5.0, "Sure. I work as a nurse at the hospital."
    )
    result = analyze_conversation(words)
    assert [s.act for s in result.dialogue_acts if s.turn_id == 1][0] in ("backchannel", "agreement")
    assert [t.dialogue_act for t in result.turns] == ["wh_question", "statement"]


def test_custom_labeller_is_used() -> None:
    """A model-based labeller plugs in through ``labeller=``."""

    class Constant:
        def __call__(self, segments: Sequence[DialogueActSegment]) -> List[DialogueActLabel]:
            return [DialogueActLabel(act="inform", confidence=0.42, source="llm") for _ in segments]

    words = utterance("A", 0.0, 2.0, "Hello there friend.") + utterance("B", 2.5, 4.0, "Nice to meet you.")
    result = analyze_conversation(words, labeller=Constant())
    assert {(s.act, s.confidence, s.source) for s in result.dialogue_acts} == {("inform", 0.42, "llm")}
    assert [t.dialogue_act for t in result.turns] == ["inform", "inform"]
    assert result.summary()["A"]["act_inform"] == 1


def test_labeller_returning_wrong_count_raises() -> None:
    """A labeller must return one label per segment."""

    def broken(segments: Sequence[DialogueActSegment]) -> List[DialogueActLabel]:
        return []

    with pytest.raises(ValueError, match="labels for"):
        analyze_conversation(utterance("A", 0.0, 1.0, "hello there"), labeller=broken)


def test_politeness_counts() -> None:
    """Strategy counts on a request with gratitude, please, modals and hedges."""
    counts = politeness_counts(
        ["Could you please send it, thanks.", "I think maybe we can talk by the way.", "So can you do it?"],
        load_default_lexicons(),
    )
    assert counts["counterfactual_modal"] == 1
    assert counts["indicative_modal"] == 1
    assert counts["please"] == 1
    assert counts["please_start"] == 0
    assert counts["gratitude"] == 1
    assert counts["hedges"] == 2
    assert counts["indirect_btw"] == 1
    assert counts["first_person_start"] == 1
    assert counts["first_person_plural"] == 1
    assert counts["direct_start"] == 1
    assert counts["second_person"] == 2


def test_markers_alignment_and_pronouns() -> None:
    """Lexical alignment is the share of a turn's content tokens repeated from the partner's last turn."""
    words = utterance("A", 0.0, 2.0, "I love hiking in the mountains.") + utterance(
        "B", 2.5, 4.0, "Mountains are great for hiking, you know."
    )
    result = analyze_conversation(words)
    a, b = result.markers
    assert a.lexical_alignment is None
    assert b.n_content_tokens == 4  # mountains, great, hiking, know
    assert b.lexical_alignment == pytest.approx(2 / 4)
    assert a.pronoun_i == 1 and b.pronoun_you == 1
    assert b.discourse_markers == 1  # "you know"
    s = result.summary()
    assert s["A"]["pronoun_i_per_100_words"] == pytest.approx(100 / 6)
    assert s["B"]["lexical_alignment_mean"] == pytest.approx(0.5)
    assert s["B"]["discourse_markers_per_100_words"] == pytest.approx(100 / 7)
    markers = result.to_dataframes()["markers"]
    assert "politeness_gratitude" in markers.columns


def test_yaml_boolean_words_load_as_strings() -> None:
    """The repo's YAML formatter unquotes yes/no/on/off; the loader must still read them as words."""
    lex = load_default_lexicons()
    assert ("yes",) in lex.backchannel
    assert ("no",) in lex.disagreement_phrases
    assert {"no", "yes", "on", "off"} <= lex.stopwords
