"""Tests for the r37 round-2 extent rules: crossing speech, DDK margins, edges, parts, background."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from senselab.audio.workflows.triage.discard_replaced_task import ReplacedTaskInputs, background_drowns
from senselab.audio.workflows.triage.task_events import (
    TaskEvent,
    background_reading,
    envelope_edge,
    evidence_of,
    generic_view,
)
from senselab.audio.workflows.triage.task_extent import grow, margined_extent, task_extent_parameters
from senselab.audio.workflows.triage.voice_phonation import Frames, PhonationReading, extent_of, voice_task_extent

RATE = 16000


def _params(**changes: Any) -> dict[str, Any]:  # noqa: ANN401
    p = copy.deepcopy(task_extent_parameters())
    for path, value in changes.items():
        section, key = path.split("__")
        p[section][key] = value
    return p


def _bursts(starts: list[float], seconds: float, level: float, noise: float = 0.001) -> np.ndarray:
    rng = np.random.default_rng(0)
    signal = rng.normal(0.0, noise, int(seconds * RATE))
    n = int(0.6 * RATE)
    for start in starts:
        a = int(start * RATE)
        signal[a : a + n] += rng.normal(0.0, level, n) * np.hanning(n)
    return signal.astype(np.float32)


def test_growth_crosses_speech_where_breathing_continues_and_still_stops_at_a_long_gap() -> None:
    """Change 4: speech between breaths no longer stops growth; a long run with no breath still does."""
    core = [(1.0, 1.5), (3.0, 3.5), (5.0, 5.5)]
    candidates = [(7.0, 7.5), (9.0, 9.5), (40.0, 40.5)]
    stopped = grow(core, candidates, limit_s=5.0, stops=[(6.0, 6.8)])
    crossed = grow(core, candidates, limit_s=5.0, stops=[(6.0, 6.8)], cross_stops=True)
    assert stopped.added == ()
    assert crossed.added == ((7.0, 7.5), (9.0, 9.5)) and crossed.crossed_n == 1
    assert crossed.stopped[1] == "gap"
    assert grow(core, [(7.0, 7.5)], limit_s=5.0, stops=[(7.2, 7.4)], cross_stops=True).added == ()


def test_a_trimmed_train_keeps_one_onset_interval_beyond_its_trimmed_edge() -> None:
    """Change 1: the margin is the train's own median onset interval, never into speech or past the original."""
    kept = [(1.0, 1.1), (1.2, 1.3), (1.4, 1.5), (1.6, 1.7)]
    assert margined_extent((0.0, 3.0), kept, head=True, tail=False, intervals=1.0, runs=[]) == (
        0.8,
        1.7,
    )
    start, end = margined_extent((0.0, 3.0), kept, head=True, tail=True, intervals=1.0, runs=[(0.0, 0.95), (1.75, 2)])
    assert start == 0.95 and end == 1.75


def test_buttercup_is_never_trimmed() -> None:
    """Change 2: the family's task word spans the recording, so no edge or speech trim applies."""
    assert "diadochokinesis-buttercup" in task_extent_parameters()["trim"]["ddk"]["exclude_families"]
    assert "diadochokinesis-v2-buttercup" in task_extent_parameters()["trim"]["ddk"]["exclude_families"]


def test_a_weak_breaths_edges_run_out_to_its_envelope() -> None:
    """Change 7: the extent takes the breath's whole envelope over the floor, not its detected core."""
    signal = _bursts([2.0], 6.0, 0.02)
    view = generic_view((signal, RATE), None)
    start = envelope_edge(view, 2.25, -1, margin_db=3.0, bridge_s=0.1, max_s=3.0)
    end = envelope_edge(view, 2.35, 1, margin_db=3.0, bridge_s=0.1, max_s=3.0)
    assert 1.9 <= start < 2.25 and 2.35 < end <= 2.7
    assert envelope_edge(view, 2.35, 1, margin_db=3.0, bridge_s=0.1, max_s=3.0, stops=[(2.36, 3.0)]) == 2.35


def test_edges_switched_on_widen_a_breath_extent_to_the_envelope() -> None:
    """The edges switch moves only the extent; the events and the decision stand."""
    signal = _bursts([2.0, 4.0], 8.0, 0.02)
    view = generic_view((signal, RATE), None)
    spans = [(2.2, 2.4), (4.2, 4.4)]
    off = evidence_of(
        view, spans, gap_s=3.0, inhale=False, stops=[], grow_kind="breath", extent_p=_params(edges__enabled=False)
    )
    on = evidence_of(
        view, spans, gap_s=3.0, inhale=False, stops=[], grow_kind="breath", extent_p=_params(edges__enabled=True)
    )
    assert on.extent is not None and off.extent is not None
    assert on.extent[0] < off.extent[0] and on.extent[1] > off.extent[1]
    assert on.decision == off.decision and on.events == off.events


def test_events_drowned_by_a_loud_background_read_low_against_it() -> None:
    """Change 8: the same breaths read high over a quiet background and low over a loud one."""
    quiet = generic_view((_bursts([2.0, 4.0], 8.0, 0.02, noise=0.0005), RATE), None)
    loud = generic_view((_bursts([2.0, 4.0], 8.0, 0.02, noise=0.01), RATE), None)
    events = [TaskEvent(2.0, 2.6, 0.0), TaskEvent(4.0, 4.6, 0.0)]
    high = background_reading(quiet, events, events)["event_over_background_db"]
    low = background_reading(loud, events, events)["event_over_background_db"]
    assert high > low + 10.0


def test_the_background_rule_reads_only_when_switched_on_and_fitted() -> None:
    """The bound is read only with the block on; off, nothing discards."""
    inputs = ReplacedTaskInputs(family="respiration-and-cough-breath", event_over_background_db=2.0)
    off = {"background": {"enabled": False, "kinds": ["airway"], "event_over_background_db_max": 6.0}}
    on = {"background": {"enabled": True, "kinds": ["airway"], "event_over_background_db_max": 6.0}}
    assert background_drowns(inputs, off) is None
    drowned = background_drowns(inputs, on)
    assert drowned is not None and drowned["name"] == "event_over_background_db"
    assert (
        background_drowns(ReplacedTaskInputs(family="respiration-and-cough-breath", event_over_background_db=9.0), on)
        is None
    )


def _frames(level: np.ndarray, hop: float = 0.01) -> Frames:
    times = np.arange(len(level)) * hop + hop / 2
    return Frames(times, level, np.zeros((len(level), 2)), 50.0)


def _voice_params() -> dict[str, Any]:
    from senselab.audio.workflows.triage.voice_phonation import voice_phonation_parameters

    return voice_phonation_parameters()


def test_a_task_done_in_three_parts_is_spanned_whole() -> None:
    """Change 5: three separate phonations 3 s apart are one extent where parts are on."""
    voiced = np.zeros(1600, dtype=bool)
    for a in (100, 600, 1100):
        voiced[a : a + 200] = True
    level = np.where(voiced, -20.0, -80.0)
    frames = _frames(level)
    p = _voice_params()
    blocked = np.zeros_like(voiced)
    alone = extent_of(frames, voiced, -80.0, [], blocked, p)
    whole = extent_of(frames, voiced, -80.0, [], blocked, p, parts=dict(task_extent_parameters()["parts"]))
    assert alone is not None and whole is not None
    assert (alone.offset - alone.onset) == 200
    assert whole.onset == 100 and whole.offset == 1300


def test_continuity_ends_right_after_the_last_burst() -> None:
    """Change 3: a run taken in by continuity ends tail_s after its last voiced frame, not at its level end."""
    voiced = np.zeros(800, dtype=bool)
    voiced[100:400] = True
    voiced[450:480] = True
    level = np.full(800, -80.0)
    level[100:400] = -20.0
    level[450:700] = -65.0
    frames = _frames(level)
    f0 = np.where(voiced, 200.0, np.nan)
    q = dict(task_extent_parameters()["continuity"])
    extent = extent_of(frames, voiced, -80.0, [], np.zeros_like(voiced), _voice_params(), f0_hz=f0, continuity=q)
    assert extent is not None
    assert extent.offset == 480 + int(round(q["tail_s"] / 0.01))


def test_a_count_in_overlapping_the_vowel_is_taken_in() -> None:
    """Change 6: 282b529e's shape, the last count word timed past the vowel's start."""
    extent = {"start_s": 2.585, "end_s": 12.0, "onset_s": 2.585, "offset_s": 12.0, "voiced_onset_s": 2.585}
    read = voice_task_extent(PhonationReading(extent), (1.58, 2.94), _params())
    assert read.task_extent == (1.58, 12.0)


def test_a_speech_split_leaving_under_half_the_train_does_not_stand() -> None:
    """Round 2 (2ca44133's shape): agreed 'words' all through a train are the task misread, not speech."""
    from senselab.audio.workflows.triage.task_extent import trim_train

    events = [(0.5 * i, 0.5 * i + 0.4) for i in range(10)]
    runs = [(1.45, 1.55), (2.95, 3.05)]
    assert trim_train(events, runs, ratio=2.5, kept_share_min=0.5).kept == tuple(range(10))
    assert len(trim_train(events, runs, ratio=2.5).kept) < 10
