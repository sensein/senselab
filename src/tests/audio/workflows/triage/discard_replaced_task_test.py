"""Discard where the task is absent or weak and talk outside it takes its place."""

from __future__ import annotations

from typing import Any

import pytest

from senselab.audio.workflows.triage import discard_replaced_task as rule
from senselab.audio.workflows.triage.discard_replaced_task import (
    RULE_LEXICAL,
    RULE_NON_LEXICAL,
    RULE_SHORT_RESPONSE,
    TASK_REPLACED_BY_TALK,
    ReplacedTaskInputs,
    replaced_task,
    rule_family,
)

BREATH = "respiration-and-cough-v2-breath"


def _rule(inputs: ReplacedTaskInputs) -> str:
    """The rule that holds for the inputs; fails where none does."""
    hit = replaced_task(inputs)
    assert hit is not None
    return str(hit["rule"])


def _rate() -> float:
    return float(rule.parameters()[RULE_NON_LEXICAL]["free_speech_words_per_s"])


def test_the_packaged_rule_is_off() -> None:
    """The packaged rule is off."""
    assert rule.enabled() is False


def test_each_family_is_read_by_its_rule() -> None:
    """Each family is read by its rule."""
    assert rule_family(BREATH) == RULE_NON_LEXICAL
    assert rule_family("prolonged-vowel") == RULE_NON_LEXICAL
    assert rule_family("loudness") == RULE_NON_LEXICAL
    assert rule_family("word-color-stroop") == RULE_LEXICAL
    assert rule_family("diadochokinesis-pa") is None
    assert rule_family(None) is None


def test_non_lexical_talk_over_a_weak_task_holds() -> None:
    """Non lexical talk over a weak task holds."""
    words = 0.3 * _rate() * 10.0
    hit = replaced_task(ReplacedTaskInputs(BREATH, duration_s=10.0, talk_words_n=words, task_mismatch=True))
    assert hit is not None and hit["rule"] == RULE_NON_LEXICAL and hit["value"] == pytest.approx(0.3)
    short_extent = ReplacedTaskInputs(BREATH, duration_s=10.0, talk_words_n=words, extent_fraction=0.2)
    assert replaced_task(short_extent) is not None


def test_non_lexical_talk_over_a_strong_task_or_little_talk_does_not() -> None:
    """Non lexical talk over a strong task or little talk does not."""
    words = 0.3 * _rate() * 10.0
    assert replaced_task(ReplacedTaskInputs(BREATH, duration_s=10.0, talk_words_n=words, extent_fraction=0.8)) is None
    little = 0.1 * _rate() * 10.0
    assert replaced_task(ReplacedTaskInputs(BREATH, duration_s=10.0, talk_words_n=little, task_mismatch=True)) is None
    assert replaced_task(ReplacedTaskInputs(BREATH, duration_s=None, talk_words_n=words, task_mismatch=True)) is None


def test_lexical_absent_task_needs_the_reviewer() -> None:
    """Lexical absent task needs the reviewer."""
    absent = ReplacedTaskInputs("word-color-stroop", expected_tokens_matched=0, reviewer_off_task=True)
    assert _rule(absent) == RULE_LEXICAL
    no_extent = ReplacedTaskInputs("word-color-stroop", extent_fraction=0.0, reviewer_more_than_one=True)
    assert _rule(no_extent) == RULE_LEXICAL
    assert replaced_task(ReplacedTaskInputs("word-color-stroop", expected_tokens_matched=0)) is None
    assert (
        replaced_task(ReplacedTaskInputs("word-color-stroop", expected_tokens_matched=3, reviewer_off_task=True))
        is None
    )


def test_a_short_free_response_with_off_task_talk_holds() -> None:
    """A short free response with off task talk holds."""
    median = float(rule.parameters()[RULE_SHORT_RESPONSE]["family_median_s"]["story-recall"])
    short = ReplacedTaskInputs("story-recall", response_s=0.05 * median, reviewer_off_task=True)
    assert _rule(short) == RULE_SHORT_RESPONSE
    by_other = ReplacedTaskInputs("story-recall", response_s=0.05 * median, instructions_by_other=True)
    assert _rule(by_other) == RULE_SHORT_RESPONSE
    assert replaced_task(ReplacedTaskInputs("story-recall", response_s=0.5 * median, reviewer_off_task=True)) is None
    assert replaced_task(ReplacedTaskInputs("story-recall", response_s=0.05 * median)) is None


def _fold(enabled: bool, monkeypatch: pytest.MonkeyPatch) -> Any:  # noqa: ANN401
    from senselab.audio.workflows.triage import vocabulary
    from senselab.audio.workflows.triage.vocabulary import TaskEvidence, fold_file_verdict

    monkeypatch.setattr(vocabulary, "replaced_task_enabled", lambda: enabled)
    task = TaskEvidence(
        owning_branches=("AIRWAY",),
        duration_s=10.0,
        task_speech={"words_n": 10, "runs_n": 2},
        task_speech_words_min=3,
    )
    cohort = {"task_extent_fraction": 0.1}
    return fold_file_verdict(
        [],
        branch_decisions={},
        ran={},
        hint_claims={},
        route_state=None,
        declared_family=BREATH,
        task=task,
        cohort=cohort,
    )


def test_the_fold_discards_on_the_rule_only_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fold discards on the rule only when enabled."""
    from senselab.audio.workflows.triage.vocabulary import Triage

    off = _fold(False, monkeypatch)
    assert off.triage is Triage.REVIEW and TASK_REPLACED_BY_TALK not in off.ground_keys
    on = _fold(True, monkeypatch)
    assert on.triage is Triage.DISCARD and on.discard_ground == TASK_REPLACED_BY_TALK
    assert on.reason == "task_not_found"
    decisive = [entry for entry in on.evidence if entry.decisive]
    assert [entry.name for entry in decisive] == [TASK_REPLACED_BY_TALK]
    assert decisive[0].value == pytest.approx(10 / 10.0 / _rate(), abs=1e-4)
