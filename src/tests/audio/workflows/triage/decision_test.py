"""Tests for the decision table's vocabulary: every ground key maps to one reason, in one precedence."""

from __future__ import annotations

import pytest

from senselab.audio.workflows.triage.decision import (
    DISCARD,
    EFFECTS,
    PASS,
    from_records,
    item,
    reason_of,
    reason_vocabulary,
    reasons_of,
    records,
)
from senselab.audio.workflows.triage.vocabulary import (
    EVENT_ABSENT_GROUNDS,
    GROUND_KEY_PREFIXES,
    GROUND_KEYS,
    RELEASE_GROUND_KEYS,
    RELEASE_UNKNOWN_GROUNDS,
    RELEASE_WITHHELD_GROUNDS,
)


def test_every_ground_key_maps_to_a_reason() -> None:
    """A ground key the vocabulary does not name fails here, printed, before it reaches a table."""
    keys = [*GROUND_KEYS, *(f"{prefix}:SPEECH" for prefix in GROUND_KEY_PREFIXES), *EVENT_ABSENT_GROUNDS.values()]
    unmapped = sorted(key for key in keys if reason_of(key) is None)
    print("unmapped ground keys:", unmapped)
    assert unmapped == []


def test_every_release_ground_that_holds_or_is_unassessed_maps_to_a_reason() -> None:
    """A withholding ground is a redaction hold; a release the graph could not assess is not measured."""
    held = {RELEASE_GROUND_KEYS[ground] for ground in RELEASE_WITHHELD_GROUNDS}
    unknown = {RELEASE_GROUND_KEYS[ground] for ground in RELEASE_UNKNOWN_GROUNDS}
    assert {reason_of(key, release=True) for key in held} == {"redaction_hold"}
    assert {reason_of(key, release=True) for key in unknown} == {"not_measured"}


def test_every_reason_has_a_place_in_the_precedence_and_a_meaning() -> None:
    """The precedence orders every reason a key can map to, and says what each means."""
    vocabulary = reason_vocabulary()
    named = {*vocabulary["keys"].values(), *vocabulary["release_keys"].values(), *vocabulary["prefixes"].values()}
    assert named <= set(vocabulary["precedence"])
    assert set(vocabulary["precedence"]) == set(vocabulary["meaning"])


def test_discard_reasons_come_first() -> None:
    """A discard's reason outranks every review reason the same recording carries."""
    assert reasons_of(["second_opinion_disagreement", "too_short_for_task", "conformance:SPEECH"]) == [
        "task_too_short",
        "task_not_conforming",
        "second_opinion_disagrees",
    ]
    assert reasons_of(["person_name_review"], "reviewer_proposed_redaction") == [
        "identifying_content",
        "redaction_hold",
    ]


def test_a_settled_decision_names_no_owed_measurement() -> None:
    """``owed_counts=False`` drops ``not_measured`` and keeps every other reason."""
    keys = ["too_short_for_task", "owning_branch_input_absent", "conformance:SPEECH"]
    assert reasons_of(keys) == ["task_too_short", "not_measured", "task_not_conforming"]
    assert reasons_of(keys, "no_transcript", owed_counts=False) == ["task_too_short", "task_not_conforming"]


def test_an_unmapped_key_is_left_out_not_invented() -> None:
    """An unknown key yields no reason rather than a guessed one."""
    assert reasons_of(["no_such_key"]) == []


def test_an_evidence_item_round_trips_and_its_effect_is_checked() -> None:
    """An item writes JSON-ready and reads back; an effect outside the vocabulary raises."""
    entry = item("task_duration_s", 0.4, DISCARD, unit="s", comparison=">=", threshold=1.0, decisive=True)
    assert from_records(records([entry])) == [entry]
    assert PASS in EFFECTS
    with pytest.raises(ValueError):
        item("x", 1, "flag")
