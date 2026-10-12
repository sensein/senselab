"""Tests for where a task's extent begins and ends, and whether it holds the whole task (task_extent.py)."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from senselab.audio.data_structures.audio_hints import ExpectedSpeech
from senselab.audio.workflows.triage.ddk_task import ddk_extent_check
from senselab.audio.workflows.triage.nodes.speech import _read_as_stimulus
from senselab.audio.workflows.triage.nodes.verdict import atypical_readings, extent_reading
from senselab.audio.workflows.triage.stimulus import LexicalWord, align_stimulus
from senselab.audio.workflows.triage.task_events import TaskEvent, evidence_of, generic_view
from senselab.audio.workflows.triage.task_extent import (
    BOTH,
    COMPLETE,
    NO_TASK,
    STOP,
    TOO_NARROW,
    TOO_WIDE,
    extent_check,
    grow,
    median_interval_s,
    modulation_events,
    reachable_outside,
    task_extent_parameters,
    trim_train,
)
from senselab.audio.workflows.triage.voice_phonation import (
    PhonationReading,
    continues,
    voice_task_extent,
    voiced_onset,
)

RATE = 16000


def _on(**sections: bool) -> dict[str, Any]:
    p = copy.deepcopy(task_extent_parameters())
    for name, value in sections.items():
        p[name]["enabled"] = value
    return p


class TestGrowth:
    """A1: bounded growth from the task cluster."""

    def test_growth_takes_events_within_k_median_gaps_and_stops_beyond(self) -> None:
        """Events 2 s apart grow in; one 10 s further does not."""
        core = [(10.0, 10.5), (12.5, 13.0), (14.5, 15.0)]
        gap = median_interval_s(core, floor_s=0.25, single_s=3.0)
        grew = grow(core, [(6.0, 6.5), (8.0, 8.5), (17.0, 17.5), (28.0, 28.5)], limit_s=2.5 * gap, stops=())
        assert grew.added == ((6.0, 6.5), (8.0, 8.5), (17.0, 17.5))
        assert grew.stopped == ("edge", "gap")
        assert grew.before_s == pytest.approx(4.0) and grew.after_s == pytest.approx(2.5)

    def test_growth_stops_at_speech_and_keeps_a_background_talkers_tail_out(self) -> None:
        """A speech span between the cluster and the next event stops growth there."""
        core = [(5.3, 5.75), (7.1, 7.65), (9.35, 10.05), (11.4, 11.9)]
        tail = [(13.0, 13.4), (20.0, 20.5), (60.0, 61.0)]
        grew = grow(core, tail, limit_s=10.0, stops=[(16.8, 30.0)])
        assert grew.added == ((13.0, 13.4),)
        assert grew.stopped[1] == STOP

    def test_a_one_event_cluster_grows_by_its_family_gap(self) -> None:
        """With one event the clustering gap is the base."""
        assert median_interval_s([(13.9, 14.8)], floor_s=0.25, single_s=5.0) == 5.0

    def test_growth_through_evidence_is_off_by_default_and_never_moves_the_decision(self) -> None:
        """Switched off the extent is the cluster's; switched on it grows and the decision stands."""
        rng = np.random.default_rng(0)
        signal = rng.normal(0.0, 0.001, 20 * RATE)
        bursts = [1.0, 3.0, 5.0, 8.9, 10.9]
        for start in bursts:
            a = int(start * RATE)
            signal[a : a + int(0.6 * RATE)] += rng.normal(0.0, 0.05, int(0.6 * RATE)) * np.hanning(int(0.6 * RATE))
        view = generic_view((signal.astype(np.float32), RATE), None)
        spans = [(s, s + 0.6) for s in bursts]
        off = evidence_of(view, spans, gap_s=3.0, stops=[], grow_kind="breath", extent_p=_on(growth=False))
        on = evidence_of(
            view,
            spans,
            gap_s=3.0,
            stops=[],
            grow_kind="breath",
            growth_candidates=[(7.0, 7.4)],
            extent_p=_on(growth=True),
        )
        assert off.extent is not None and off.extent[1] < 6.0
        assert off.check["status"] == TOO_NARROW and off.check["outside_n"] == 2
        assert on.extent is not None and on.extent[1] > 11.0
        assert on.growth["added_n"] == 2 and on.check["status"] == COMPLETE
        assert on.decision == off.decision and on.inputs == off.inputs

    def test_an_event_behind_speech_is_not_reachable(self) -> None:
        """An event outside the extent with speech between is excluded for cause, not missed."""
        assert reachable_outside((5.0, 10.0), [(12.0, 12.5), (20.0, 20.5)], [(15.0, 18.0)]) == [(12.0, 12.5)]


class TestModulation:
    """A1: the envelope-modulation event detector."""

    def test_slow_breathing_lobes_are_found_in_the_envelope(self) -> None:
        """A 0.3 Hz envelope with lobes 20 dB over the floor reads one event per cycle."""
        hop = 0.01
        times = np.arange(0, 30.0, hop)
        over = np.clip(20.0 * np.sin(2 * np.pi * 0.3 * times), 0.0, None)
        events = modulation_events(times, over, hop, task_extent_parameters()["modulation"])
        assert 7 <= len(events) <= 10
        assert all(0.3 <= b - a <= 4.0 for a, b in events)

    def test_a_flat_envelope_holds_no_event(self) -> None:
        """No modulation, no event."""
        times = np.arange(0, 10.0, 0.01)
        assert modulation_events(times, np.zeros_like(times), 0.01) == []


class TestCompleteness:
    """A5: the status from the signals that fired."""

    @pytest.mark.parametrize(
        ("narrow", "wide", "has", "status"),
        [
            ({"a": 0}, {"b": 0}, True, COMPLETE),
            ({"a": 2}, {"b": 0}, True, TOO_NARROW),
            ({"a": 0}, {"b": True}, True, TOO_WIDE),
            ({"a": 1}, {"b": 0.4}, True, BOTH),
            ({}, {}, False, NO_TASK),
        ],
    )
    def test_status(self, narrow: dict[str, Any], wide: dict[str, Any], has: bool, status: str) -> None:
        """Each combination of signals reads its status, and names what fired."""
        check = extent_check(has_extent=has, narrow=narrow, wide=wide)
        assert check.status == status
        if status in (TOO_NARROW, BOTH):
            assert "narrow:a" in check.checks


class TestDdkTrim:
    """A2: a syllable train trimmed at agreed non-task speech and over-long edge events."""

    def test_speech_inserted_mid_train_keeps_the_larger_piece_before_it(self) -> None:
        """07929b7b's shape: nine cycles, speech over two more, one cycle after; the extent ends before it."""
        events = [(0.2 + i * 1.2, 1.0 + i * 1.2) for i in range(9)] + [(12.0, 12.7), (12.7, 13.3), (14.3, 15.4)]
        trimmed = trim_train(events, [(12.0, 13.1)], ratio=2.5)
        assert trimmed.kept == tuple(range(9))
        assert trimmed.speech_n == 3

    def test_an_overlong_opening_event_is_not_the_train(self) -> None:
        """119daa75's shape: a 0.5 s opening before 0.12 s syllables."""
        events = [(0.015, 0.54)] + [(0.54 + 0.12 * i, 0.66 + 0.12 * i) for i in range(11)]
        trimmed = trim_train(events, [], ratio=2.5)
        assert trimmed.kept[0] == 1 and trimmed.edge_n == 1

    def test_a_regular_train_is_untouched(self) -> None:
        """No speech and even events: nothing is trimmed (buttercup cycles among them)."""
        events = [(0.5 + 0.6 * i, 1.0 + 0.6 * i) for i in range(10)]
        assert trim_train(events, [], ratio=2.5).kept == tuple(range(10))

    def test_the_check_reads_speech_inside_the_extent_as_too_wide(self) -> None:
        """Untrimmed, an agreed run inside the extent is over-inclusion."""
        cluster = [TaskEvent(0.5 + 0.6 * i, 1.0 + 0.6 * i, 20.0) for i in range(10)]
        check = ddk_extent_check((0.5, 6.4), cluster, cluster, [(5.4, 6.1)], ratio=2.5)
        assert check.status == TOO_WIDE and "wide:speech_in_extent" in check.checks


class TestVoice:
    """A2 and A3: the voice extent's onset, its count-in, and an interrupted vowel."""

    def test_the_onset_is_the_first_sustained_voiced_stretch(self) -> None:
        """Unvoiced breath noise before the vowel is not its onset."""
        voiced = np.zeros(400, dtype=bool)
        voiced[10:12] = True
        voiced[120:400] = True
        assert voiced_onset(voiced, 0, 400, 20) == 120

    def test_a_run_at_the_vowels_pitch_continues_it(self) -> None:
        """A run within the semitone bound continues, an octave away breaks, a fragment is passed over."""
        voiced = np.ones(100, dtype=bool)
        f0 = np.full(100, 200.0)
        q = {"semitones": 3.0, "voiced_min_s": 0.1}
        assert continues((0, 50), 210.0, voiced, f0, q, 0.01) == "continues"
        assert continues((0, 50), 400.0, voiced, f0, q, 0.01) == "breaks"
        assert continues((0, 5), 210.0, voiced, f0, q, 0.01) == "unread"

    def _reading(self, **kw: Any) -> PhonationReading:  # noqa: ANN401
        extent = {"start_s": 3.4, "end_s": 11.6, "onset_s": 3.4, "offset_s": 11.6, "voiced_onset_s": 3.4}
        return PhonationReading(extent, **kw)

    def test_the_count_in_joins_the_extent_when_switched_on(self) -> None:
        """d2687d53's shape: a count-in ending at the vowel joins the extent; off, it is too narrow."""
        on = voice_task_extent(self._reading(), (0.88, 3.39), _on(count_in=True))
        off = voice_task_extent(self._reading(), (0.88, 3.39), _on(count_in=False))
        assert on.task_extent == (0.88, 11.6) and on.check["status"] == COMPLETE
        assert off.task_extent is None and off.check["status"] == TOO_NARROW

    def test_a_count_in_inside_the_extent_stays_out_of_the_speaker_comparison(self) -> None:
        """A3: a voice family's target pool is empty, so a count-in inside its extent is compared with nothing."""
        from senselab.audio.workflows.triage.cohort_stage import TARGET_NONE, target_rule

        assert target_rule("prolonged-vowel") == TARGET_NONE
        assert target_rule("maximum-phonation-time") == TARGET_NONE

    def test_a_head_before_the_onset_is_too_wide(self) -> None:
        """43bc9a51's shape: phonation-level noise from 0.3 s, the vowel from 1.15 s."""
        extent = {"start_s": 0.3, "end_s": 4.4, "onset_s": 0.3, "offset_s": 4.4, "voiced_onset_s": 1.15}
        check = voice_task_extent(PhonationReading(extent), None, _on()).check
        assert check["status"] == TOO_WIDE

    def test_a_reachable_hold_at_the_vowels_pitch_is_too_narrow(self) -> None:
        """e8044ea4's shape: the vowel's start left outside the extent."""
        hold = {"start_s": 3.35, "end_s": 4.84, "voiced_s": 0.9, "semitones": 0.4, "reachable": True}
        check = voice_task_extent(self._reading(holds_outside=(hold,)), None, _on()).check
        assert check["status"] == TOO_NARROW and check["outside_n"] == 1


class TestVerdictReads:
    """A5 and E2: the fold's inputs."""

    def test_the_breath_reading_carries_its_check_and_growth(self) -> None:
        """The owning reading's check and growth reach the task evidence."""
        breath = {"reading": {"evidence": {"extent_check": {"status": TOO_NARROW}, "growth": {"seconds": 4.5}}}}
        store = SimpleNamespace()
        check, growth = extent_reading(store, breath, None, None, None)  # type: ignore[arg-type]
        assert check["status"] == TOO_NARROW and growth == 4.5

    def test_a_long_effortful_cough_and_a_slow_train_are_atypical_attempts(self) -> None:
        """Median cough events beyond the corpus p97.5, or a syllable rate outside p2.5-p97.5."""
        cough = {"reading": {"evidence": {"events": [{"start_s": 0.0, "end_s": 2.3}, {"start_s": 3.0, "end_s": 5.1}]}}}
        ddk = {"decision": "present", "annotations": {"syllable_rate_hz": 2.0}}
        assert atypical_readings(cough, None) == ("cough_event_s=2.20",)
        assert atypical_readings(None, ddk) == ("ddk_rate_hz=2.00",)
        assert atypical_readings(None, {"decision": "present", "annotations": {"syllable_rate_hz": 5.0}}) == ()


def _word(index: int, text: str, start: float, end: float, **readings: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=f"w{index}",
        extent=(start, end),
        attributes={"index": index, "text": text, "readings": readings or {"a": text, "b": text}},
    )


def test_an_edge_token_one_recogniser_read_as_the_stimulus_is_read_as_it() -> None:
    """A4, c1807e86's shape: the last token substituted, one recogniser reading it exactly; an aside is not."""
    words = [
        _word(0, "the", 0.5, 0.7),
        _word(1, "birch", 0.7, 1.1),
        _word(2, "canoe", 1.1, 1.6),
        _word(3, "slid", 1.6, 2.0),
        _word(4, "planx", 2.0, 2.6, a="planks", b="planx"),
        _word(5, "okay", 4.0, 4.3),
    ]
    alignment = align_stimulus(
        [ExpectedSpeech(text="The birch canoe slid planks.")],
        [LexicalWord(int(w.attributes["index"]), w.attributes["text"], w.extent, None) for w in words],
        terminators=".?!",
    )
    bounds = {"exact_below": 5, "two_edits_from": 8}
    params = SimpleNamespace(setting=lambda path, coerce=str: bounds[path.split(".")[-1]])
    read = _read_as_stimulus(alignment, words, params)  # type: ignore[arg-type]
    assert [w.id for w in read] == ["w0", "w1", "w2", "w3", "w4"]
