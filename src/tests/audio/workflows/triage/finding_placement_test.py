"""Where a detector finding sits among the transcript's words."""

from __future__ import annotations

from senselab.audio.workflows.triage.finding_placement import cut, is_temporal, locate, match_key


def test_a_key_drops_edges_possessives_and_hyphens() -> None:
    """``Alan's,`` is ``alan``; ``ninety-three`` is ``ninetythree``."""
    assert match_key("Alan's,") == "alan"
    assert match_key("ninety-three") == "ninetythree"
    assert match_key("[") == ""


def test_a_split_or_joined_spelling_places_and_a_part_of_a_word_does_not() -> None:
    """Token runs compared by their joined keys."""
    assert locate("ninety-three", ["he", "is", "ninety", "three"]) == [(2, 3)]
    assert locate("ninety three", ["he", "is", "ninetythree"]) == [(2, 2)]
    assert locate("nine", ["ninety"]) == []


def test_every_occurrence_places() -> None:
    """A name said twice is two runs."""
    assert locate("alice", ["alice", "met", "Alice."]) == [(0, 0), (2, 2)]


def test_a_date_keeps_its_temporal_words() -> None:
    """The r6 overrun: only "week" is a date word."""
    tokens = "Australia, my brother Alan's wife, had died that week".split()
    runs, did = cut((0, len(tokens) - 1), tokens, "DATE", 3)
    assert did and runs == [(len(tokens) - 1, len(tokens) - 1)]
    tokens = "a couple of weeks ago".split()
    assert cut((0, 4), tokens, "DATE_TIME", 3) == ([(1, 4)], True)


def test_a_date_with_no_temporal_word_keeps_its_run() -> None:
    """Nothing to cut to: the finding stands."""
    assert cut((0, 1), ["the", "festival"], "DATE", 3) == ([(0, 1)], False)


def test_a_long_name_keeps_its_proper_nouns_and_a_short_one_is_left() -> None:
    """A person span past ``name_words_max`` is cut; "my brother Alan" is not."""
    tokens = "and then my brother Alan came".split()
    assert cut((0, 5), tokens, "PERSON", 3) == ([(4, 4)], True)
    assert cut((2, 4), tokens, "PERSON", 3) == ([(2, 4)], False)


def test_temporal_words() -> None:
    """Digits, number words and their compounds, calendar and unit words."""
    assert all(is_temporal(t) for t in ["1998", "ninetythree", "Tuesday", "weeks", "ago"])
    assert not any(is_temporal(t) for t in ["brother", "Alan", "died"])
