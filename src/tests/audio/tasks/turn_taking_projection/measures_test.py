"""Derived dyadic measures, on VAP frames constructed so every expected value is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from senselab.audio.tasks.turn_taking_projection import summarize_projection
from senselab.audio.tasks.turn_taking_projection.measures import (
    derive_backchannels,
    find_peaks,
    pause_events,
    shift_events,
)
from senselab.audio.tasks.turn_taking_projection.result import VAPResult
from senselab.utils.data_structures import ScriptLine

RATE = 10.0


def frames(duration: float, p_now_a: object = 0.5, p_bc_a: object = 0.0, p_bc_b: object = 0.0) -> VAPResult:
    """A VAPResult whose columns are built from callables of time (or constants)."""
    n = int(round(duration * RATE))
    t = (np.arange(n) + 1.0) / RATE

    def col(v: object) -> np.ndarray:
        return np.asarray([v(x) for x in t]) if callable(v) else np.full(n, float(v))  # type: ignore[arg-type]

    pa = col(p_now_a)
    df = pd.DataFrame(
        {
            "time": t,
            "p_now_a": pa,
            "p_now_b": 1 - pa,
            "p_future_a": pa,
            "p_future_b": 1 - pa,
            "vad_a": np.zeros(n),
            "vad_b": np.zeros(n),
            "p_bc_a": col(p_bc_a),
            "p_bc_b": col(p_bc_b),
        }
    )
    return VAPResult(frame_rate=RATE, speakers=("A", "B"), frames=df)


def line(speaker: str, start: float, end: float) -> ScriptLine:
    """A labelled segment."""
    return ScriptLine(speaker=speaker, start=start, end=end)


def test_a_shift_the_model_saw_coming_is_projected_with_its_lead() -> None:
    """B's p_now is high for the last 0.8 s of A's turn: projected, lead ~0.8 s."""
    vap = frames(10.0, p_now_a=lambda t: 0.1 if 2.2 < t <= 4.0 else 0.9)
    [row] = shift_events(vap, [line("A", 0.0, 3.0), line("B", 3.3, 6.0)])
    assert row["from"] == "A" and row["to"] == "B"
    assert row["gap_s"] == pytest.approx(0.3)
    assert row["p_now_next_mean"] == pytest.approx(0.9)
    assert row["projected"] is True
    assert row["lead_s"] == pytest.approx(0.8, abs=0.11)


def test_an_unforeseen_shift_is_not_projected() -> None:
    """A's p_now stays high up to the change: the shift was not projected and has no lead."""
    vap = frames(10.0, p_now_a=0.8)
    [row] = shift_events(vap, [line("A", 0.0, 3.0), line("B", 3.3, 6.0)])
    assert row["p_now_next_mean"] == pytest.approx(0.2)
    assert row["projected"] is False
    assert row["lead_s"] == 0.0


def test_an_overlapping_shift_is_read_before_the_next_onset() -> None:
    """In overlap the floor changes at B's onset, so the window ends there, not at A's end."""
    vap = frames(10.0, p_now_a=lambda t: 0.9 if t <= 2.5 else 0.1)
    [row] = shift_events(vap, [line("A", 0.0, 3.0), line("B", 2.5, 6.0)])
    assert row["gap_s"] == pytest.approx(-0.5)
    assert row["p_now_next_mean"] == pytest.approx(0.1)


def test_same_speaker_consecutive_turns_are_not_shifts() -> None:
    """Only speaker changes are scored."""
    vap = frames(10.0)
    assert shift_events(vap, [line("A", 0, 1), line("A", 2, 3), line("B", 4, 5)])[0]["from"] == "A"
    assert len(shift_events(vap, [line("A", 0, 1), line("A", 2, 3)])) == 0


def test_pauses_are_called_hold_or_shift_and_scored() -> None:
    """A pause the model reads as B's is a correct shift call; one it reads as A's is a correct hold."""
    vap = frames(20.0, p_now_a=lambda t: 0.2 if 5.0 < t <= 6.0 else 0.8)
    ipus = [line("A", 0.0, 2.0), line("A", 2.6, 5.0), line("B", 5.6, 8.0)]
    rows = pause_events(vap, ipus, [])
    assert [r["outcome"] for r in rows] == ["hold", "shift"]
    assert [r["predicted"] for r in rows] == ["hold", "shift"]
    assert all(r["correct"] for r in rows)
    assert rows[0]["p_shift"] == pytest.approx(0.2)
    assert rows[1]["p_shift"] == pytest.approx(0.8)


def test_short_gaps_and_backchannels_do_not_open_pauses() -> None:
    """A 0.1 s gap is below min_pause; a backchannel inside a pause does not split or end it."""
    vap = frames(20.0)
    ipus = [line("A", 0.0, 2.0), line("A", 2.1, 4.0), line("B", 4.3, 4.6), line("A", 5.0, 7.0)]
    rows = pause_events(vap, ipus, [("B", 4.3, 4.6)])
    assert len(rows) == 1
    assert rows[0]["outcome"] == "hold"
    assert rows[0]["start"] == pytest.approx(4.0) and rows[0]["end"] == pytest.approx(5.0)


def test_an_overlapped_ipu_does_not_make_the_shorter_speaker_last() -> None:
    """The last speaker is whoever ended last, even if another IPU started later."""
    vap = frames(20.0)
    rows = pause_events(vap, [line("A", 0.0, 5.0), line("B", 1.0, 2.0), line("B", 6.0, 8.0)], [])
    assert len(rows) == 1 and rows[0]["last"] == "A" and rows[0]["outcome"] == "shift"


def test_backchannels_are_derived_from_short_listener_ipus_inside_the_other_turn() -> None:
    """Short and inside the other's turn: a backchannel. Long, or in one's own turn: not."""
    turns = [line("A", 0.0, 10.0), line("B", 11.0, 15.0)]
    ipus = [line("B", 3.0, 3.4), line("B", 5.0, 7.0), line("B", 12.0, 12.3), line("A", 13.0, 13.2)]
    assert derive_backchannels(ipus, turns, ("A", "B"), 1.0) == [("B", 3.0, 3.4), ("A", 13.0, 13.2)]


def test_peaks_respect_threshold_and_separation() -> None:
    """Tallest first, so of two peaks 0.5 s apart the higher survives."""
    t = (np.arange(50) + 1) / RATE
    v = np.zeros(50)
    v[10], v[15], v[30], v[40] = 0.7, 0.9, 0.4, 0.6
    assert find_peaks(t, v, 0.5, 1.0) == [15, 40]


def test_backchannel_responsiveness_counts_answered_opportunities() -> None:
    """Two B opportunities, one answered within the window; one B backchannel off any opportunity."""
    vap = frames(30.0, p_bc_b=lambda t: 0.9 if abs(t - 5.0) < 0.05 or abs(t - 15.0) < 0.05 else 0.1)
    turns = [line("A", 0.0, 25.0)]
    ipus = [line("A", 0.0, 25.0), line("B", 5.3, 5.6), line("B", 20.0, 20.3)]
    summary = summarize_projection(vap, turns, ipus)
    bc = summary["speakers"]["B"]["backchannel"]
    assert summary["overall"]["backchannel_source"] == "derived"
    assert bc["n_produced"] == 2
    assert bc["n_opportunities"] == 2
    assert bc["n_answered"] == 1
    assert bc["responsiveness"] == pytest.approx(0.5)
    assert bc["timed_fraction"] == pytest.approx(0.5)
    assert summary["speakers"]["A"]["backchannel"]["n_opportunities"] == 0


def test_summary_aggregates_per_speaker_and_is_json_shaped() -> None:
    """Per-speaker turn ends, takes and pause summaries line up with the events."""
    import json

    vap = frames(20.0, p_now_a=lambda t: 0.2 if 2.5 < t <= 4.0 else (0.8 if 7.5 < t <= 9.0 else 0.5))
    turns = [line("A", 0.0, 3.0), line("B", 3.2, 8.0), line("A", 8.5, 12.0)]
    summary = summarize_projection(vap, turns, turns, backchannels=[])
    a, b = summary["speakers"]["A"], summary["speakers"]["B"]
    assert a["turn_ends"]["n"] == 1 and b["turn_ends"]["n"] == 1
    assert a["turn_ends"]["mean_p_now_next"] == pytest.approx(0.8)
    assert b["turn_ends"]["mean_p_now_next"] == pytest.approx(0.8)
    assert b["turn_takes"]["mean_gap_s"] == pytest.approx(0.2)
    assert summary["overall"]["pauses"]["n_shift"] == 1
    assert summary["overall"]["backchannel_source"] == "given"
    json.dumps(summary)


def test_auc_and_balanced_accuracy_need_both_outcomes() -> None:
    """With only holds there is nothing to rank against."""
    vap = frames(20.0, p_now_a=0.9)
    summary = summarize_projection(vap, [line("A", 0, 10)], [line("A", 0, 2), line("A", 3, 5)], [])
    pauses = summary["overall"]["pauses"]
    assert pauses["n_hold"] == 1 and pauses["n_shift"] == 0
    assert pauses["auc"] is None and pauses["balanced_accuracy"] is None
    assert pauses["accuracy"] == 1.0


def test_summary_without_backchannel_columns_reports_none() -> None:
    """A projection run without a backchannel checkpoint has no backchannel block."""
    vap = frames(10.0)
    bare = VAPResult(frame_rate=RATE, speakers=("A", "B"), frames=vap.frames.drop(columns=["p_bc_a", "p_bc_b"]))
    summary = summarize_projection(bare, [line("A", 0, 3), line("B", 4, 6)], [line("A", 0, 3), line("B", 4, 6)])
    assert summary["speakers"]["A"]["backchannel"] is None


def test_vapresult_rejects_missing_columns_and_unknown_speakers() -> None:
    """The frame table contract is checked on construction."""
    with pytest.raises(ValueError):
        VAPResult(frame_rate=RATE, speakers=("A", "B"), frames=pd.DataFrame({"time": [0.1]}))
    with pytest.raises(KeyError):
        frames(1.0).series("p_now", "C")
