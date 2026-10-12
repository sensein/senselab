"""Word timing arbitration (r37 B1): a word the recognizers place far apart takes the time on sound; its text stays."""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.workflows.triage.consensus import (
    SourceHypothesis,
    align_sources,
    arbitrate_time,
    word_attributes,
    word_from_attributes,
)
from senselab.audio.workflows.triage.nodes.common import consensus_words, find_measurement, software_agent, word_hull
from senselab.audio.workflows.triage.word_arbitration import (
    WORD_ARBITRATION,
    arbitrate,
    arbitrate_words,
    held_voicing,
)
from senselab.utils.prov_store import ProvStore

CRISPER = "asr_crisperwhisper"
QWEN = "asr_qwen"


def _words(qwen: list[tuple[str, float]], crisper: list[tuple[str, float]]) -> list:
    """Two recognizers' words, each 0.3 s long, aligned into one stream."""
    sources = [
        SourceHypothesis(QWEN, tuple((t, s, s + 0.3) for t, s in qwen), "bundled_aligner", None),
        SourceHypothesis(CRISPER, tuple((t, s, s + 0.3) for t, s in crisper), "native", None),
    ]
    return list(align_sources(sources, onomatopoeic=set(), duration_s=60.0).words)


def test_the_word_takes_the_time_of_the_recognizer_on_sound_and_keeps_its_text() -> None:
    """SUBJ01 stroop: Qwen's text is kept, its time 12 s off is replaced by CrisperWhisper's on sound."""
    [word] = _words([("red", 14.0)], [("read", 2.0)])
    assert word.text in ("red", "read")
    timed = arbitrate_time(word, on_sound=[(1.8, 2.6)], disagreement_s=1.0, duration_s=60.0)
    assert timed.extent == (2.0, 2.3) and timed.arbitrated_source == CRISPER and not timed.unlocated
    assert timed.text == word.text and timed.readings == word.readings and timed.timings == word.timings


def test_a_word_no_recognizer_places_on_sound_is_unlocated() -> None:
    """Neither time lies on sound: no guess, no replacement; a zero-length extent at the fitted onset."""
    [word] = _words([("red", 14.0)], [("red", 2.0)])
    timed = arbitrate_time(word, on_sound=[(30.0, 31.0)], disagreement_s=1.0, duration_s=60.0)
    assert timed.unlocated and timed.arbitrated_source is None
    assert timed.extent[0] == timed.extent[1] == word.extent[0]


def test_close_times_or_two_times_on_sound_leave_the_fit() -> None:
    """Under the bound nothing is disagreed; both on sound is ambiguous and the fit stands."""
    [near] = _words([("red", 2.5)], [("red", 2.0)])
    assert arbitrate_time(near, on_sound=[(0.0, 1.0)], disagreement_s=1.0) == near
    [far] = _words([("red", 14.0)], [("red", 2.0)])
    assert arbitrate_time(far, on_sound=[(1.0, 20.0)], disagreement_s=1.0) == far


def test_in_a_voice_task_the_held_phonation_is_not_where_a_word_is() -> None:
    """061cec48: Qwen's first word inside the held vowel; the count-in time outside it wins."""
    [word] = _words([("one", 1.12)], [("one", 0.1)])
    timed = arbitrate_time(word, on_sound=[(0.0, 9.0)], off_limits=[(1.0, 9.0)], disagreement_s=1.0)
    assert timed.arbitrated_source == CRISPER and timed.extent == (0.1, 0.4)


def test_the_flags_round_trip_through_the_store_attributes() -> None:
    """An arbitrated or unlocated word reads back as written."""
    [word] = _words([("red", 14.0)], [("red", 2.0)])
    for timed in (
        arbitrate_time(word, on_sound=[(1.8, 2.6)], disagreement_s=1.0),
        arbitrate_time(word, on_sound=[], disagreement_s=1.0),
    ):
        assert word_from_attributes(word_attributes(timed), timed.extent) == timed


def test_the_hull_of_an_arbitrated_word_is_its_own_time() -> None:
    """A mask over the word covers the arbitrated time, not every recognizer's placement."""
    [word] = _words([("red", 14.0)], [("red", 2.0)])
    store = ProvStore(run_id="hull")
    raw = store.entity(prov_type="word", extent=word.extent, attributes=word_attributes(word))
    timed = arbitrate_time(word, on_sound=[(1.8, 2.6)], disagreement_s=1.0)
    moved = store.entity(prov_type="word", extent=timed.extent, attributes=word_attributes(timed))
    assert word_hull(store.get_entity(raw)) == (2.0, 14.3)
    assert word_hull(store.get_entity(moved)) == (2.0, 2.3)


def test_the_counts_name_what_moved() -> None:
    """One re-timed, one unlocated, one agreed."""
    words = _words([("red", 14.0), ("blue", 20.0), ("green", 25.0)], [("red", 2.0), ("blue", 8.0), ("green", 25.1)])
    result = arbitrate(words, on_sound=[(1.8, 2.6), (24.0, 26.0)])
    assert result.record["disagreed_n"] == 2 and result.record["retimed_n"] == 1
    assert result.record["unlocated_n"] == 1 and result.record["won_by_source"] == {CRISPER: 1}
    assert result.moved == (0, 1)


def test_an_unlocated_word_is_masked_over_the_gap_between_its_timed_neighbours() -> None:
    """r37 owner: mask the gap between the timed neighbours rather than withhold; the hull reads the gap."""
    words = _words([("red", 2.0), ("blue", 20.0), ("green", 25.0)], [("red", 2.0), ("blue", 8.0), ("green", 25.1)])
    result = arbitrate(words, on_sound=[(1.8, 2.6), (24.0, 26.0)], duration_s=60.0)
    first, middle, last = result.words
    assert middle.unlocated and middle.mask_gap == (round(first.extent[1], 3), round(last.extent[0], 3))
    assert first.mask_gap is None and last.mask_gap is None
    store = ProvStore(run_id="gap")
    stored = store.entity(prov_type="word", extent=middle.extent, attributes=word_attributes(middle))
    assert word_hull(store.get_entity(stored)) == middle.mask_gap
    assert word_from_attributes(word_attributes(middle), middle.extent) == middle


def test_an_unlocated_word_with_no_neighbour_after_runs_to_the_end() -> None:
    """No timed word after it: the gap closes at the recording's end."""
    words = _words([("red", 2.0), ("blue", 40.0)], [("red", 2.0), ("blue", 8.0)])
    result = arbitrate(words, on_sound=[(1.8, 2.6)], duration_s=60.0)
    assert result.words[1].mask_gap == (round(result.words[0].extent[1], 3), 60.0)


def test_held_voicing_is_a_long_voiced_run_bridged_over_short_gaps() -> None:
    """A two-second vowel with a 0.1 s break is one held run; a 0.4 s word is none."""
    times = np.arange(0.0, 5.0, 0.01)
    voiced = ((times >= 1.0) & (times < 3.0) & ~((times >= 2.0) & (times < 2.1))) | ((times >= 4.0) & (times < 4.4))
    [(start, end)] = held_voicing(times, voiced, min_s=1.5, bridge_s=0.25)
    assert start == pytest.approx(1.0) and end == pytest.approx(3.0)


def test_a_store_s_moved_words_are_superseded_and_the_transcript_rewritten() -> None:
    """The moved word is a new entity; the others keep their ids; the arbitration is recorded."""
    store = ProvStore(run_id="arbitrate")
    software_agent(store)
    store.entity(prov_type="stream", extent=(0.0, 60.0), attributes={"name": "recording", "path": "r.wav"})
    words = _words([("red", 14.0), ("green", 25.0)], [("red", 2.0), ("green", 25.1)])
    ids = [store.entity(prov_type="word", extent=w.extent, attributes=word_attributes(w)) for w in words]
    store.entity(
        prov_type="measurement",
        extent=None,
        attributes={"name": "consensus_transcript", "signal": "plain", "word_ids": ids, "text": "x"},
    )
    for extent in ((1.8, 2.6), (24.0, 26.0)):
        store.entity(prov_type="span", extent=extent, attributes={"measure": "amplitude", "signal": "preemphasised"})
    measurement = arbitrate_words(store, run_dir=None, family="word-color-stroop")
    assert measurement is not None
    found = find_measurement(store, WORD_ARBITRATION)
    assert found is not None
    record = found.attributes
    assert record["retimed_n"] == 1 and record["unlocated_n"] == 0 and record["moved_n"] == 1
    live = consensus_words(store)
    assert [w.extent for w in live] == [(2.0, 2.3), words[1].extent]
    assert live[1].id == ids[1] and live[0].id != ids[0]
    assert store.is_invalidated(ids[0])
