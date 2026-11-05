"""Window planning, window-to-absolute time conversion, cross-window merge and rasterization."""

import pytest

from senselab.video.tasks.event_localization import merge_windows, plan_windows, rasterize, shift_segments, split_window


def test_plan_windows_cover_and_bound() -> None:
    """Short recordings are one window; long ones are covered with the last window pulled back to the end."""
    assert plan_windows(0) == []
    assert plan_windows(30, 30, 20) == [(0.0, 30.0)]
    assert plan_windows(34, 30, 20) == [(0.0, 34.0)]
    w = plan_windows(100, 30, 20)
    assert w[0] == (0.0, 30.0) and w[-1] == (70.0, 100.0)
    for t in (0.0, 17.3, 50.0, 99.9):
        assert any(a <= t <= b for a, b in w)
    assert all(b - a <= 30.0 + 1e-9 for a, b in plan_windows(1037, 30, 20))
    with pytest.raises(ValueError):
        plan_windows(100, 20, 30)


def test_split_and_shift() -> None:
    """A refused window halves; window-relative times become absolute."""
    assert split_window((10.0, 30.0)) == [(10.0, 20.0), (20.0, 30.0)]
    out = shift_segments([{"label": "x", "start_s": 1.0, "end_s": 2.5, "extra": {"k": "v"}}], 20.0)
    assert out == [{"label": "x", "start_s": 21.0, "end_s": 22.5, "extra": {"k": "v"}}]


def test_merge_windows() -> None:
    """Overlap and short gaps chain per label; the score is the fraction of covering windows that saw it."""
    w1, w2 = (0.0, 30.0), (20.0, 50.0)
    a = [{"label": "x", "start_s": 25.0, "end_s": 30.0}]
    b = shift_segments(
        [{"label": "x", "start_s": 5.0, "end_s": 9.0}, {"label": "y", "start_s": 20.0, "end_s": 22.0}], 20.0
    )
    m = merge_windows([(w1, a), (w2, b)], gap_s=1.0)
    x = [s for s in m if s["label"] == "x"]
    assert len(x) == 1 and (x[0]["start_s"], x[0]["end_s"], x[0]["n_windows"], x[0]["score"]) == (25.0, 30.0, 2, 1.0)
    y = [s for s in m if s["label"] == "y"][0]
    assert y["start_s"] == 40.0 and y["n_windows"] == 1
    two = merge_windows(
        [(w1, [{"label": "x", "start_s": 1.0, "end_s": 2.0}, {"label": "x", "start_s": 4.0, "end_s": 5.0}])]
    )
    assert len(two) == 2
    one = merge_windows(
        [(w1, [{"label": "x", "start_s": 1.0, "end_s": 2.0}, {"label": "x", "start_s": 2.8, "end_s": 5.0}])]
    )
    assert len(one) == 1 and one[0]["end_s"] == 5.0
    ex = merge_windows(
        [
            (w1, [{"label": "p", "start_s": 3.0, "end_s": 3.0, "extra": {"c": "yes"}}]),
            (w2, [{"label": "p", "start_s": 23.5, "end_s": 23.5, "extra": {"c": "no"}}]),
        ]
    )
    assert len(ex) == 2 and merge_windows([(w1, [])]) == []


def test_rasterize_overlap_by_score() -> None:
    """A frame takes the label of the containing segment; the higher score wins an overlap."""
    r = rasterize(
        [{"label": "a", "start_s": 0.2, "end_s": 0.5}, {"label": "b", "start_s": 0.4, "end_s": 0.6, "score": 2.0}],
        10,
        15.0,
    )
    assert r[3] == "a" and r[6] == "b" and r[0] == "background" and len(r) == 10
