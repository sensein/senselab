"""Each axis declares what its rows are indexed by (D-24).

`AxisKey = (Axis, Bucket)` assumed one row index for every axis. There is no universal bucket: three
axes are estimated on a 0.1 s time grid and `asr` is estimated on the word grid, because a transcript
has no natural per-bucket value and bucketing it performs the `REDUCE` that `GridRelation` names — on
the finest-grained evidence in the run, to match three coarser axes.

Declared rather than assumed, so a consumer that took time buckets for granted cannot silently
mis-join `asr` against the other three.
"""

from __future__ import annotations

from senselab.audio.workflows.audio_analysis.axes import AXES, AXIS_GRIDS, AXIS_NAMES, axis


def test_asr_is_on_the_word_grid_and_the_others_are_on_time() -> None:
    """The one asymmetry, and it follows from what the evidence is rather than from convenience."""
    assert axis("asr").grid == "word"
    assert {AXIS_GRIDS[name] for name in AXIS_NAMES if name != "asr"} == {"time_0.1s"}


def test_every_active_axis_declares_a_grid() -> None:
    """A missing grid would default to time and mis-join whichever axis forgot to say."""
    assert all(a.grid for a in AXES if a.active)


def test_the_grid_map_covers_exactly_the_active_axes() -> None:
    """A declared-but-inactive axis has no rows to index, and must not appear as if it did."""
    assert set(AXIS_GRIDS) == set(AXIS_NAMES)


def test_joining_asr_to_a_time_axis_is_detectable_from_the_declaration_alone() -> None:
    """The point of declaring it: a consumer can tell a trivial join from a projection.

    Without this, code that zipped two axes' rows by index would produce a plausible, wrong
    alignment — the failure mode this design keeps finding, where a name resolves to something and
    nothing says it was the wrong something.
    """
    assert AXIS_GRIDS["asr"] != AXIS_GRIDS["speech_presence"]
    assert AXIS_GRIDS["speaker"] == AXIS_GRIDS["background_mask"]
