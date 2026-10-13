"""One authoritative axis set, and the fourth axis on the same terms as the other three (D-17).

``background_mask`` was fused, written to ``estimates/`` and drawn on the timeline, and then
skipped by everything that iterated ``("speech_presence", "speaker", "asr")`` — region proposal,
convergence marking, the convergence report. A run could report "nothing left to do" having never
asked the fourth axis anything. These are the tests that make a list of three fail.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

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

AXES_MODULE = "senselab.audio.workflows.audio_analysis.axes"


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


def test_no_module_narrows_the_axis_alias_to_an_enumeration() -> None:
    """Three modules each declared a three-member ``Literal``, so they could disagree — and did.

    Checked at the **source** level, not by identity. ``types.UncertaintyAxis`` is declared in
    ``types.py`` rather than imported from ``axes``, so extraction-layer code that reaches
    ``types`` does not thereby reach the axis vocabulary — and the obvious guard,
    ``types.UncertaintyAxis is axes.AxisName``, is then a test that cannot fail, because both sides
    are the builtin ``str`` whatever either module did. What actually has to hold is that no
    module re-narrows the alias, and only the source says that: a ``Literal[...]``, an ``Enum``, or
    anything other than a bare ``str`` on the right-hand side is the defect.
    """
    aliases = {
        "axes.py": "AxisName",
        "types.py": "UncertaintyAxis",
        "adaptive/types.py": "AxisName",
    }
    seen: dict[str, str] = {}
    for rel, name in aliases.items():
        tree = ast.parse((WORKFLOW_DIR / rel).read_text())
        assignments = [
            node
            for node in tree.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
            if isinstance(target, ast.Name) and target.id == name
        ]
        imported = [
            f"{node.module}.{a.name}"
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for a in node.names
            if (a.asname or a.name) == name
        ]
        assert len(assignments) + len(imported) == 1, f"{rel}: {name} must be declared exactly once"
        if imported:
            seen[rel] = f"import {imported[0]}"
            continue
        value = assignments[0].value
        assert isinstance(value, ast.Name) and value.id == "str", (
            f"{rel}: {name} must be a bare `str` — the axis set is open, and a type that enumerates "
            f"its members is what made `background_mask` unrepresentable"
        )
        seen[rel] = "str"

    assert seen["types.py"] == "str", "types.py declares the alias itself, so extraction cannot reach axes through it"
    assert seen["axes.py"] == "str"
    # Not merely "imported from somewhere": an import of an `AxisName` that a third module declared
    # as a `Literal` satisfies the loop above and re-narrows the alias anyway.
    assert seen["adaptive/types.py"] == f"import {AXES_MODULE}.AxisName", (
        f"adaptive/types.py must take AxisName from {AXES_MODULE}, the declaration this test checks; "
        f"it takes it from {seen['adaptive/types.py']}"
    )


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


def test_every_harvested_axis_declares_where_its_evidence_lives() -> None:
    """``harvested=True`` with no ``HarvestSource`` is a claim no reader can act on.

    The two sets are kept separate on purpose — ``HARVESTED_AXES`` is what an axis *claims*,
    ``HARVEST_SOURCES`` is what a reader can *find* — and their agreeing is exactly the property
    that failed. ``background_mask`` was flagged harvested while the adaptive store's ingest
    enumerated three axes in a literal tuple, and the store's own accessor set was built from the
    flag, so it reported the mask as covered and the guard that exists to catch an unreadable axis
    could not fire.
    """
    from senselab.audio.workflows.audio_analysis.axes import HARVEST_SOURCES

    assert set(HARVEST_SOURCES) == set(HARVESTED_AXES)
    assert HARVEST_SOURCES["background_mask"].field == "background_mask_evidence"
    assert HARVEST_SOURCES["background_mask"].holds == "votes"
    # Presence is the one axis whose harvest holds measurements: every threshold that turns them
    # into a statement is L2's, so a reader must link rather than fold them as they are.
    assert HARVEST_SOURCES["speech_presence"].holds == "measurements"


def test_the_mask_axis_is_read_from_the_harvest_one_row_per_bucket(tmp_path: Path) -> None:
    """The bug this file's fourth-axis theme exists for, in its last hiding place.

    ``fuse`` and ``reliability`` read the mask's per-bucket harvest; the adaptive store did not, and
    the driver made up the difference by handing it ``mask_axis_votes(mask_regions)`` — one vote per
    mask *region*. A run whose mask found a single region therefore carried 1070 mask buckets at
    round 0 and **one** by round 4, and an axis with one bucket has nowhere to be uncertain, so the
    convergence report read it as settled. Both the store and the votes file now come from the same
    per-bucket harvest.
    """
    import pyarrow.parquet as pq

    from senselab.audio.workflows.audio_analysis.adaptive.belief import VoteStore
    from senselab.audio.workflows.audio_analysis.io import write_linked_votes
    from senselab.audio.workflows.audio_analysis.layout import derivatives_dir
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest, link_pass

    spans = [(round(s / 10, 6), round((s + 1) / 10, 6)) for s in range(6)]
    evidence: list[dict[str, Any]] = [
        {"start": start, "end": end, "task_type": "speech", "votes": {"speech": {"reading": 0.4 + i / 100}}}
        for i, (start, end) in enumerate(spans)
    ]
    harvest = PassHarvest(perturbation="raw", background_mask_evidence=evidence)

    store = VoteStore.from_harvests({"raw": harvest})
    assert store.buckets("background_mask") == spans

    linked = link_pass(harvest, params={})
    assert linked.buckets_by_axis["background_mask"] == evidence
    dest = write_linked_votes(
        {"raw": linked.buckets_by_axis["background_mask"]},
        "background_mask",
        derivatives_dir(tmp_path, 0) / "votes" / "background_mask.parquet",
    )
    assert dest.parent == tmp_path / "L2" / "round" / "0" / "derivatives" / "votes"
    table = pq.read_table(dest)
    assert table.column("axis").to_pylist() == ["background_mask"] * len(evidence)
    # Keyed by the perturbation it was measured under. The per-region write named a fabricated
    # perturbation ("mask"), which is in no run's perturbation set, so the artifact ingest path —
    # which skips a row naming a perturbation the run did not take — dropped every one of them.
    assert set(table.column("perturbation").to_pylist()) == {"raw"}


def test_an_axis_no_reader_can_find_raises_rather_than_folding_to_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard has to stay able to fail, now that no active axis needs ``unharvested``.

    All four axes declare a harvest source today, so the ``unharvested`` mapping is empty on every
    real call — which is precisely when a guard rots into a formality. Simulated with a fifth active
    axis that declares no source: it must raise, because an axis nobody hands in carries no belief
    through any round and is then reported as ``0 buckets, residual mass 0.0`` — settled rather than
    never asked.
    """
    from senselab.audio.workflows.audio_analysis.adaptive import belief as belief_module
    from senselab.audio.workflows.audio_analysis.votes import PassHarvest

    monkeypatch.setattr(belief_module, "AXES", (*AXIS_NAMES, "prosody"))
    with pytest.raises(ValueError, match="prosody"):
        belief_module.VoteStore.from_harvests({"raw": PassHarvest(perturbation="raw")})
    # ...and an *empty* entry is accepted, because "nothing was found" is a measurement.
    store = belief_module.VoteStore.from_harvests({"raw": PassHarvest(perturbation="raw")}, unharvested={"prosody": {}})
    assert store.buckets("prosody") == []
