"""Tests for the task layer's shared reading (task_events.py) on synthetic audio."""

from __future__ import annotations

import numpy as np

from senselab.audio.workflows.triage.task_events import (
    ABSENT,
    PRESENT,
    REVIEW,
    TaskEvent,
    decide,
    dominant_cluster,
    event_snr_db,
    evidence_of,
    generic_view,
    rhythm_of,
    task_cluster,
    task_events_parameters,
)

RATE = 16000
P = {"snr_low_db": 6.0, "snr_high_db": 12.0}


def _noise(seconds: float, level: float, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, level, int(seconds * RATE))


def _with_bursts(base: np.ndarray, starts: list[float], length_s: float, level: float) -> np.ndarray:
    out = base.copy()
    rng = np.random.default_rng(1)
    n = int(length_s * RATE)
    envelope = np.hanning(n)
    for start in starts:
        a = int(start * RATE)
        out[a : a + n] += rng.normal(0.0, level, n) * envelope
    return out


def test_a_clear_event_is_present_a_weak_one_review_and_none_absent() -> None:
    """C5: a clear event decides present, a weak one review, and none over the low cut absent."""
    assert decide([TaskEvent(0.0, 1.0, 20.0)], P)[0] == PRESENT
    assert decide([TaskEvent(0.0, 1.0, 8.0)], P)[0] == REVIEW
    assert decide([TaskEvent(0.0, 1.0, 3.0)], P)[0] == ABSENT
    assert decide([], P)[0] == ABSENT


def test_a_clear_event_an_impulse_abuts_is_left_for_review() -> None:
    """Only background temporally entangled with the deciding event sends it to review."""
    assert decide([TaskEvent(0.0, 1.0, 20.0, entangled=True)], P) == (REVIEW, "entangled")


def test_the_dominant_cluster_is_the_largest_and_ignores_a_far_event() -> None:
    """The cluster is the largest run of events within the gap."""
    events = [TaskEvent(t, t + 0.5, 15.0) for t in (1.0, 2.5, 4.0)] + [TaskEvent(20.0, 20.5, 15.0)]
    cluster = dominant_cluster(events, gap_s=3.0)
    assert [e.start_s for e in cluster] == [1.0, 2.5, 4.0]


def test_breath_bursts_over_quiet_noise_read_clear_and_their_extent_stays_off_the_quiet_tail() -> None:
    """Bursts well over the floor are present, and the extent ends at the last event."""
    signal = _with_bursts(_noise(12.0, 0.001), [1.0, 3.0, 5.0], 0.8, 0.05)
    view = generic_view((signal.astype(np.float32), RATE), None)
    evidence = evidence_of(view, [(1.0, 1.8), (3.0, 3.8), (5.0, 5.8)], gap_s=3.0)
    assert evidence.decision == PRESENT
    assert evidence.extent is not None and evidence.extent[1] <= 6.0
    assert event_snr_db(view, 8.0, 11.0) < 6.0


def test_a_click_is_never_a_task_event() -> None:
    """A candidate an impulse explains is dropped."""
    signal = _noise(4.0, 0.001)
    signal[int(2.0 * RATE) : int(2.0 * RATE) + 16] += 0.5
    view = generic_view((signal.astype(np.float32), RATE), None)
    assert view.impulses
    evidence = evidence_of(view, [(1.95, 2.05)], gap_s=3.0)
    assert evidence.events_found_n == 0 and evidence.decision == ABSENT


def test_an_impulsive_onset_inside_its_own_event_does_not_entangle_a_cough() -> None:
    """A click inside a breath entangles it; inside a cough, whose onset is impulsive, it does not."""
    signal = _with_bursts(_noise(4.0, 0.0005), [1.0], 1.0, 0.01)
    signal[int(1.5 * RATE) : int(1.5 * RATE) + 16] += 0.8
    view = generic_view((signal.astype(np.float32), RATE), None)
    assert view.impulses
    breath = evidence_of(view, [(1.0, 2.0)], gap_s=3.0)
    cough = evidence_of(view, [(1.0, 2.0)], gap_s=3.0, entangle_inside=False)
    assert breath.found[0].entangled and not cough.found[0].entangled


def test_a_regular_burst_train_gives_a_breathing_band_rhythm() -> None:
    """Bursts every 2.5 s read as a 0.4 Hz breathing-band rhythm."""
    signal = _with_bursts(_noise(30.0, 0.001), [float(t) for t in np.arange(1.0, 29.0, 2.5)], 1.0, 0.05)
    view = generic_view((signal.astype(np.float32), RATE), None)
    rhythm = rhythm_of(view, (0.1, 1.2), 3.0)
    assert rhythm is not None and abs(rhythm.hz - 0.4) < 0.08


WEIGHTS = task_events_parameters()["extent"]


def test_a_long_file_takes_the_early_breaths_over_a_late_pause_run() -> None:
    """On a 130 s file the instructed breaths at the start are the task, not a larger run of sounds at 101-122 s."""
    early = [TaskEvent(t, t + 0.8, 20.0) for t in (0.85, 2.6, 4.4, 6.5, 8.4, 11.1)]
    late = [TaskEvent(t, t + 0.6, 20.0) for t in np.arange(101.0, 122.0, 2.5)]
    for instructed in (6, None):
        cluster = task_cluster([*early, *late], 3.0, instructed_n=instructed, duration_s=130.0, weights=WEIGHTS)
        assert cluster[0].start_s == 0.85 and cluster[-1].start_s == 11.1


def test_a_cluster_touching_speech_is_set_aside_unless_every_cluster_does() -> None:
    """A run of events inside a conversation is not the task; where speech touches every run, one is still chosen."""
    first = [TaskEvent(t, t + 0.5, 20.0) for t in (2.0, 4.0, 6.0)]
    second = [TaskEvent(t, t + 0.5, 20.0) for t in (30.0, 32.0, 34.0)]
    talk = [(1.5, 7.0)]
    cluster = task_cluster([*first, *second], 3.0, speech=talk, instructed_n=6, duration_s=40.0, weights=WEIGHTS)
    assert cluster[0].start_s == 30.0
    everywhere = [(0.0, 40.0)]
    cluster = task_cluster([*first, *second], 3.0, speech=everywhere, instructed_n=6, duration_s=40.0, weights=WEIGHTS)
    assert cluster[0].start_s == 2.0


def test_an_event_outside_the_extent_decides_nothing() -> None:
    """A clear event outside the cluster cannot pass a recording whose own events are weak; it is reported."""
    signal = _noise(60.0, 0.001)
    signal = _with_bursts(signal, [2.0, 4.0, 6.0], 0.8, 0.004)
    signal = _with_bursts(signal, [50.0], 0.8, 0.2)
    view = generic_view((signal.astype(np.float32), RATE), None)
    evidence = evidence_of(view, [(2.0, 2.8), (4.0, 4.8), (6.0, 6.8), (50.0, 50.8)], gap_s=3.0, instructed_n=3)
    assert [round(e.start_s, 1) for e in evidence.events] == [2.0, 4.0, 6.0]
    assert evidence.decision != PRESENT
    assert evidence.inputs["local_db"] < evidence.inputs["snr_high_db"]
    assert evidence.record()["outside"]["standing_n"] == 1
