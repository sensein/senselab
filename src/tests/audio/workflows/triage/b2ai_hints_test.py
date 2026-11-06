"""Tests of ``b2ai_hints``: the source dataset's file-name reader and hint builder, which plan step 7 moves out.

Collected here from the tests that used to read identity off file names (plan step 8), so they move as one file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.audio.workflows.triage.b2ai_hints import identity_from_name, task_id_of
from senselab.audio.workflows.triage.identity import FROM_STEM
from senselab.audio.workflows.triage.routing_analysis.families import UNKNOWN_TASK, task_family
from senselab.audio.workflows.triage.run import entity_subdir

STEM = "sub-a_ses-b_task-respiration-and-cough-fivebreaths-2"


class TestFromTheStem:
    """With no hint, the identity is exactly what the stem parsers read."""

    def test_a_bids_stem_names_participant_session_and_task(self) -> None:
        """Every field comes from the stem, and the family drops the trailing index."""
        identity = identity_from_name(STEM)
        assert (identity.participant, identity.session) == ("sub-a", "ses-b")
        assert identity.task_id == "respiration-and-cough-fivebreaths-2"
        assert identity.task_family == "respiration-and-cough-fivebreaths"
        assert set(identity.sources.values()) == {FROM_STEM}

    def test_each_entity_is_read_on_its_own(self) -> None:
        """``sub-`` and ``ses-`` are read wherever they sit; a lone participant keeps its folder and group."""
        cases = {
            STEM: ("sub-a", "ses-b"),
            "sub-a_acq-x_ses-b_task-loudness": ("sub-a", "ses-b"),
            "sub-a_task-rainbow-passage": ("sub-a", None),
            "ses-b_task-x": (None, "ses-b"),
            "sub-_ses-_task-x": (None, None),
            "plain-run-id": (None, None),
            "": (None, None),
        }
        for stem, expected in cases.items():
            identity = identity_from_name(stem)
            assert (identity.participant, identity.session) == expected
            assert identity.task_id == task_id_of(stem)
            assert identity.task_family == task_family(task_id_of(stem))
        assert identity_from_name("sub-a_task-x").session_key is None

    def test_a_stem_with_no_task_entity_declares_none(self) -> None:
        """The task id is the parsers' own placeholder."""
        assert identity_from_name("plain-run-id").task_id == UNKNOWN_TASK


def test_task_id_reads_the_bids_entity() -> None:
    """The task id comes from the stem's trailing ``task-`` entity, lowercased."""
    assert task_id_of("sub-A_ses-B_task-Prolonged-Vowel") == "prolonged-vowel"
    assert task_id_of("sub-A_ses-B") == "unknown"


@pytest.mark.parametrize(
    ("stem", "expected"),
    [
        ("sub-01_ses-02_task-vowel", ("sub-01", "ses-02")),
        ("sub-01_task-vowel", ("sub-01",)),
        ("recording", ()),
        ("nothing_bids_here", ()),
        ("sub-SUBJ01_ses-SES01_task-diadochokinesis-ka", ("sub-SUBJ01", "ses-SES01")),
        ("ses-02_task-vowel", ("ses-02",)),
        ("sub-_ses-_task-vowel", ()),
        ("sub-01_ses-_task-vowel", ("sub-01",)),
    ],
)
def test_the_name_readers_entities_place_the_run(stem: str, expected: tuple[str, ...]) -> None:
    """The reader's participant and session place a run: a label may contain hyphens, an empty one is none."""
    assert entity_subdir(identity_from_name(stem)) == (Path(*expected) if expected else Path("."))
