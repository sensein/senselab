"""Metrics and the window orchestration."""

from typing import Any

import pytest

from senselab.video.tasks.event_localization import (
    WindowTooLarge,
    binary_map,
    event_prf,
    localize_events,
    map_tal,
    presence,
    seg_prf,
    shuffle_floor,
    variant_confusion,
    whole_video,
)

G: list[dict[str, Any]] = [
    {"video": "a", "label": "x", "start_s": 10.0, "end_s": 20.0},
    {"video": "a", "label": "x", "start_s": 40.0, "end_s": 50.0},
    {"video": "b", "label": "y", "start_s": 0.0, "end_s": 5.0},
]
PERFECT = [dict(x, score=1.0) for x in G]


def test_map_and_prf() -> None:
    """Mean AP is 1 for a perfect list, AP uses the recall-precision area, tIoU thresholds are inclusive."""
    assert map_tal(PERFECT, G)["avg_map"] == pytest.approx(1.0)
    assert map_tal([], G)["avg_map"] == 0.0
    gx = [x for x in G if x["label"] == "x"]
    p = [
        {"video": "a", "label": "x", "start_s": 10.0, "end_s": 20.0, "score": 0.9},
        {"video": "a", "label": "x", "start_s": 70.0, "end_s": 80.0, "score": 0.5},
    ]
    assert map_tal(p, gx)["ap"]["x"][0.5] == pytest.approx(0.5)
    q = [{"video": "a", "label": "x", "start_s": 10.0, "end_s": 15.0, "score": 1.0}]
    assert map_tal(q, gx[:1])["map"][0.5] == 1.0 and map_tal(q, gx[:1])["map"][0.6] == 0.0
    assert seg_prf(PERFECT, G, 0.5)["f1"] == 1.0 and binary_map(PERFECT, G)["avg_map"] == 1.0


def test_events_presence_confusion_floors() -> None:
    """Point events, presence with prevalence and count error, variant confusion, shuffled and whole-video floors."""
    one = {"video": "a", "label": "n", "start_s": 12.0, "end_s": 12.0}
    assert event_prf([one], [dict(one, start_s=14.0, end_s=14.0)], 2.5)["tp"] == 1
    miss = event_prf([one], [dict(one, start_s=15.0, end_s=15.0)], 2.5)
    assert (miss["tp"], miss["fp"], miss["fn"]) == (0, 1, 1)
    pr = presence(PERFECT, G, ["a", "b", "c"], ["x", "y"])
    assert pr["x"]["ba"] == 1.0 and pr["x"]["count_exact"] == 1.0 and pr["x"]["prevalence"] == pytest.approx(1 / 3)
    pr = presence(PERFECT[:1], G, ["a", "b", "c"], ["x", "y"])
    assert pr["x"]["count_bias"] == -1.0 and pr["y"]["recall"] == 0.0
    cf = variant_confusion([{"video": "a", "label": "y", "start_s": 10.0, "end_s": 20.0, "score": 1.0}], G, ["x", "y"])
    assert cf["x"]["y"] == 1 and cf["x"]["none"] == 1 and cf["y"]["none"] == 1
    fl = shuffle_floor(PERFECT, {"a": 60.0, "b": 10.0}, seed=1, draws=3)
    assert len(fl) == 3 and all(
        abs((s["end_s"] - s["start_s"]) - (float(t["end_s"]) - float(t["start_s"]))) < 1e-9 for s, t in zip(fl[0], G)
    )
    assert len(whole_video(["a", "b"], {"a": 60.0, "b": 10.0}, ["x", "y"])) == 4


def test_localize_events_absolute_times_and_split() -> None:
    """The model sees window-relative time; results are absolute; a refused window is halved; overlaps merge."""
    seen = []

    def ask(w: tuple) -> str:
        seen.append(w)
        if w[1] - w[0] > 25 and w[0] == 20.0:
            raise WindowTooLarge()
        if w[0] == 0.0:
            return '[{"label": "x", "start_s": 25.0, "end_s": 29.0}]'
        return '[{"label": "x", "start_s": 5.0, "end_s": 6.0}]'

    r = localize_events(60.0, ask, ["x"], win_s=30, stride_s=20)
    assert r.n_split == 1 and r.n_unfit == 0 and len(r.windows) >= 3
    assert any(abs(s["start_s"] - 25.0) < 1e-9 for s in r.segments)
    assert all(0.0 <= s["start_s"] <= s["end_s"] <= 60.0 for s in r.segments)
    assert r.counts()["bad_label"] == 0
