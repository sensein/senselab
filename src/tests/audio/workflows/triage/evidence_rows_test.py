"""Evidence items are one scalar row each, compared by their value's type.

``conftest.py`` applies :func:`row_problems` to every fold any triage test runs (breath, cough, voice,
DDK, speech and the join); this module pins the rule itself.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.triage import vocabulary
from senselab.audio.workflows.triage.decision import (
    ANNOTATION,
    DISCARD,
    PASS,
    REVIEW,
    EvidenceItem,
    item,
    row_problems,
)


@pytest.mark.parametrize(
    "entry",
    [
        item("capture.plain_active_s", 0.0, DISCARD, unit="s", comparison=">", threshold=0.0),
        item("capture.route_state", "empty", DISCARD, comparison="not in", threshold=["empty"]),
        item("breath_events_entangled", False, PASS, comparison="==", threshold=False),
        item("ddk_identity", None, REVIEW, comparison=">=", threshold=0.3),
        item("breath_rhythm_hz", 0.25, ANNOTATION, unit="Hz"),
        item("breaths_found", 3, ANNOTATION, comparison=">=", threshold=None),
    ],
)
def test_a_scalar_row_with_a_comparison_of_its_type_is_well_formed(entry: EvidenceItem) -> None:
    """Numbers against numbers, flags against flags, categories against a list, or no comparison."""
    assert row_problems(entry) == []


@pytest.mark.parametrize(
    "entry",
    [
        item("nothing_captured", {"route_state": "empty"}, DISCARD, comparison="==", threshold=False),
        item("owning_branch_input_absent", ["AIRWAY:x"], REVIEW),
        item("capture.plain_active_s", 0.0, DISCARD, comparison="==", threshold=False),
        item("capture.route_state", "empty", DISCARD, comparison=">=", threshold=1),
        item("phonation_found", True, PASS, comparison=">=", threshold=True),
        item("capture.route_state", "empty", DISCARD, comparison="in", threshold="empty"),
    ],
)
def test_a_dict_a_list_or_a_mismatched_comparison_is_a_problem(entry: EvidenceItem) -> None:
    """A dict or list value, or a comparison of another type, is reported."""
    assert row_problems(entry)


def test_the_fold_check_is_installed() -> None:
    """The conftest wrapper is what every fold in the suite runs through."""
    assert vocabulary._decision_evidence.__name__ == "checked"


def test_capture_is_one_row_per_reading_and_the_empty_ones_decide_the_discard() -> None:
    """Nothing captured is four rows; on the discard the failing ones discard, one silent stream annotates."""
    quality = {
        "plain_active_s": 0.0,
        "enhanced_active_s": 0.0,
        "no_activity": True,
        "level_rel_db": -41.0,
        "quiet_vs_session": True,
        "level_rel_db_max": -30.0,
    }
    rows = vocabulary._capture_items("empty", quality, True)
    assert [row.name for row in rows] == [
        "capture.route_state",
        "capture.plain_active_s",
        "capture.enhanced_active_s",
        "capture.level_rel_db",
    ]
    assert all(row.effect == DISCARD and not row_problems(row) for row in rows)
    heard = vocabulary._capture_items("speech", {**quality, "plain_active_s": 2.0, "no_activity": False}, False)
    assert {row.name: row.effect for row in heard} == {
        "capture.route_state": PASS,
        "capture.plain_active_s": PASS,
        "capture.enhanced_active_s": ANNOTATION,
        "capture.level_rel_db": ANNOTATION,
    }
