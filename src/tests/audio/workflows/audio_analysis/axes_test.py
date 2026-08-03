"""One authoritative axis set, and the fourth axis on the same terms as the other three (D-17).

``background_mask`` was fused, written to ``estimates/`` and drawn on the timeline, and then
skipped by everything that iterated ``("speech_presence", "speaker", "asr")`` — region proposal,
convergence marking, the convergence report. A run could report "nothing left to do" having never
asked the fourth axis anything. These are the tests that make a list of three fail.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from senselab.audio.workflows.audio_analysis.axes import (
    ATTENUATED_AXES,
    AXES,
    AXIS_NAMES,
    AXIS_PRIORITY,
    HARVESTED_AXES,
    Axis,
    axis,
)

WORKFLOW_DIR = Path(__file__).resolve().parents[5] / "src" / "senselab" / "audio" / "workflows" / "audio_analysis"
SCRIPTS_DIR = Path(__file__).resolve().parents[5] / "scripts"

THE_THREE = frozenset({"speech_presence", "speaker", "asr"})


def _sources() -> list[Path]:
    return sorted(WORKFLOW_DIR.rglob("*.py")) + [SCRIPTS_DIR / "analyze_audio.py", SCRIPTS_DIR / "adaptive_loop.py"]


# ── the set itself ───────────────────────────────────────────────────────────


def test_the_active_axis_set_is_four_and_the_fourth_is_the_mask() -> None:
    """Any list of three axes is wrong."""
    assert AXIS_NAMES == ("speech_presence", "speaker", "asr", "background_mask")


def test_the_punted_fifth_is_declared_and_inactive() -> None:
    """Declared-and-not-yet-built and not-thought-of are different states."""
    task = axis("task")
    assert task.active is False
    assert "task" not in AXIS_NAMES


def test_every_subset_is_derived_from_a_property_not_written_out() -> None:
    """A subset written by hand is a second list that has to agree with the first, and did not."""
    assert HARVESTED_AXES == tuple(a.name for a in AXES if a.active and a.harvested)
    assert ATTENUATED_AXES == tuple(a.name for a in AXES if a.active and a.attenuable)
    assert set(AXIS_PRIORITY) == set(AXIS_NAMES)


def test_the_attenuated_set_excludes_exactly_what_its_axes_say_it_does() -> None:
    """``ATTENUATED_AXES`` is justified per axis, on the axis, or it is not justified.

    The exclusions are claims about evidence: "nobody spoke here" says nothing about *which*
    speaker it was, and nothing about whether a region is free of target activity. Both are
    recorded as ``attenuable=False`` on the axis they exclude, so the reason cannot drift away
    from the decision the way a comment beside a hand-written tuple can.
    """
    assert ATTENUATED_AXES == ("speech_presence", "asr")
    assert axis("speaker").attenuable is False
    assert axis("background_mask").attenuable is False


def test_the_mask_axis_is_harvested_like_every_other() -> None:
    """VAD, ASR words and speaker occupancy all bear on whether the target was active.

    It was ``harvested=False`` while a single derived judgement produced the mask, which read as a
    property of the mask when it was a property of there being one producer. The flag decides two
    things at once, and the second is why this matters: the disagreements index builds from
    ``HARVESTED_AXES``, so while the flag was ``False`` the axis was fused, written to ``estimates/``
    and drawn on the timeline while absent from the ranking that decides what a reader looks at.
    """
    assert axis("background_mask").harvested is True
    assert "background_mask" in HARVESTED_AXES
    assert "background_mask" in AXIS_NAMES


def test_every_active_axis_is_harvested_and_that_is_checked_not_assumed() -> None:
    """A future axis may genuinely have one producer, so the property stays declared."""
    assert set(HARVESTED_AXES) == set(AXIS_NAMES)


def test_an_undeclared_axis_raises_rather_than_returning_nothing() -> None:
    """A typo used to produce an empty result set, which reads as "this axis had nothing to say"."""
    with pytest.raises(KeyError, match="declare it in axes.AXES"):
        axis("utterance")


# ── adding the fifth is one edit ─────────────────────────────────────────────


def test_adding_an_axis_is_one_edit() -> None:
    """Activating ``task`` must move every derived set, with no second list to update.

    Simulated by rebuilding the derivations over an extended declaration: if any consumer had its
    own copy of the set, this would pass here and the pipeline would still skip the new axis —
    which is why the source scan below exists alongside it.
    """
    extended = (
        *AXES,
        Axis(
            name="prosody",
            question="how was it said?",
            harvested=True,
            attenuable=True,
            overlap_informed=False,
            calibrated=True,
            rank=5,
        ),
    )
    names = tuple(a.name for a in extended if a.active)
    assert names[-1] == "prosody"
    assert tuple(a.name for a in extended if a.active and a.harvested)[-1] == "prosody"
    assert tuple(a.name for a in extended if a.active and a.attenuable)[-1] == "prosody"
    assert {a.name: a.rank for a in extended if a.active}["prosody"] == 5
    assert tuple(a.name for a in extended if a.active and a.calibrated)[-1] == "prosody"


def test_no_pipeline_module_writes_the_axis_set_out_by_hand() -> None:
    """The axis set may not be spelled out anywhere but its declaration.

    Twenty-two literal ``("speech_presence", "speaker", "asr")`` tuples is twenty-two chances for
    the fourth axis to be missed, and it was missed in all of them. A genuine *subset* is fine —
    but it has to come from a property on the axis, so the reason for the exclusion lives on the
    thing excluded.
    """
    offenders: list[str] = []
    for path in _sources():
        if not path.exists() or path.name in {"axes.py", "contracts.py"}:
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                continue
            values = {e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)}
            if values and values <= THE_THREE and len(values) >= 2:
                offenders.append(f"{path.name}:{node.lineno}")
    assert offenders == [], f"the axis set is written out by hand in {offenders}"


def test_the_type_aliases_come_from_the_declaration() -> None:
    """Three modules each declared a three-member ``Literal``, so they could disagree — and did."""
    import typing

    from senselab.audio.workflows.audio_analysis import types as workflow_types
    from senselab.audio.workflows.audio_analysis.adaptive import types as adaptive_types
    from senselab.audio.workflows.audio_analysis.axes import AxisName

    assert workflow_types.UncertaintyAxis is AxisName
    assert adaptive_types.AxisName is AxisName
    assert typing.get_origin(AxisName) is None and AxisName is str, "the axis set is open"


# ── the fourth axis participates on the same terms ───────────────────────────


def test_the_loop_iterates_the_declared_set_not_a_literal() -> None:
    """``belief.AXES`` is the re-export half the loop imports; it must not be its own list."""
    from senselab.audio.workflows.audio_analysis.adaptive.belief import AXES as LOOP_AXES

    assert LOOP_AXES == AXIS_NAMES
    assert "background_mask" in LOOP_AXES


def test_convergence_marking_and_the_report_cover_every_axis() -> None:
    """Both read ``AXES``, so the fourth axis is marked and reported like the others.

    Checked on the *source*, because the alternative — running a whole loop — would pass with an
    axis silently absent from the state and prove only that nothing crashed.
    """
    source = (WORKFLOW_DIR / "adaptive" / "convergence.py").read_text()
    loops = [n for n in ast.walk(ast.parse(source)) if isinstance(n, ast.For)]
    over_axes = [n for n in loops if isinstance(n.iter, ast.Name) and n.iter.id == "AXES"]
    assert len(over_axes) >= 2, "convergence marking and the report must each iterate the declared set"


def test_the_disagreements_tiebreak_ranks_every_axis_by_decision() -> None:
    """The fourth axis fell to a ``.get(axis, 99)`` default and sorted last by accident."""
    from senselab.audio.workflows.audio_analysis import disagreements

    assert disagreements._AXIS_PRIORITY is AXIS_PRIORITY
    assert AXIS_PRIORITY["asr"] < AXIS_PRIORITY["speaker"] < AXIS_PRIORITY["speech_presence"]
    assert AXIS_PRIORITY["background_mask"] == 3


def test_the_mask_axis_votes_are_written_where_the_others_are(tmp_path: Path) -> None:
    """Its evidence has to be in the store, or iterating four axes still proposes over three."""
    import pyarrow.parquet as pq

    from senselab.audio.workflows.audio_analysis.fuse import mask_axis_votes
    from senselab.audio.workflows.audio_analysis.io import write_linked_votes
    from senselab.audio.workflows.audio_analysis.layout import derivatives_dir

    regions = [
        {"start": 0.0, "end": 1.0, "state": "target_free", "confidence": 0.9},
        {"start": 1.0, "end": 2.0, "state": "indeterminate", "confidence": 0.5},
    ]
    votes = mask_axis_votes(regions)
    # "I cannot tell" is the absence of a claim, not a confident maximum.
    assert [(v["start"], v["end"]) for v in votes] == [(0.0, 1.0)]

    dest = write_linked_votes(
        {"mask": votes}, "background_mask", derivatives_dir(tmp_path, 0) / "votes" / "background_mask.parquet"
    )
    assert dest.parent == tmp_path / "L2" / "round" / "0" / "derivatives" / "votes"
    assert pq.read_table(dest).column("axis").to_pylist() == ["background_mask"]
