"""The one membership rule PREPROCESS stamps and the routing analysis re-derives."""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.label_membership import (
    LabelMembership,
    load_label_membership,
    optional_label_membership,
)


class TestMembershipIsAConjunction:
    """A label joins on being both ranked and confident; either alone is not enough."""

    def test_the_top_k_and_the_floor_both_bind(self) -> None:
        """Four over the floor, ranked first, are members; the fifth over the floor is not."""
        rule = LabelMembership(top_k=4, floor=0.2, label_floors={})
        scores = {"a": 0.9, "b": 0.8, "c": 0.7, "d": 0.6, "e": 0.5}
        assert list(rule.members(scores)) == ["a", "b", "c", "d"]

    def test_a_top_ranked_label_under_the_floor_is_not_a_member(self) -> None:
        """Being the best of a quiet window is not confidence."""
        rule = LabelMembership(top_k=4, floor=0.2, label_floors={})
        assert rule.members({"a": 0.19, "b": 0.01}) == {}

    def test_a_per_label_floor_overrides_the_default(self) -> None:
        """The override refines the floor and leaves the ranking alone."""
        rule = LabelMembership(top_k=4, floor=0.5, label_floors={"a": 0.1})
        assert list(rule.members({"a": 0.2, "b": 0.4})) == ["a"]

    def test_the_ranking_breaks_ties_by_label(self) -> None:
        """Two labels at one score must cut the same way on every run of the corpus."""
        rule = LabelMembership(top_k=1, floor=0.0, label_floors={})
        assert list(rule.members({"b": 0.5, "a": 0.5})) == ["a"]

    def test_the_members_come_back_in_descending_score_order(self) -> None:
        """The order is the reader's, so a consumer taking the head takes the strongest."""
        rule = LabelMembership(top_k=4, floor=0.2, label_floors={})
        assert list(rule.members({"a": 0.3, "b": 0.9})) == ["b", "a"]


class TestTheRuleComesOffTheConfiguration:
    """No number here is a literal; both come from ``windows.<classifier>``."""

    def test_the_packaged_pair_is_shared_by_both_span_classifiers(self) -> None:
        """One pair of values, so the two readers of it cannot drift."""
        config = load_triage_config()
        yamnet = optional_label_membership(config, "yamnet")
        hear = optional_label_membership(config, "hear")
        assert yamnet is not None and hear is not None
        assert (yamnet.top_k, yamnet.floor) == (4, 0.2)
        assert (hear.top_k, hear.floor) == (yamnet.top_k, yamnet.floor)

    def test_a_null_floor_reads_as_no_rule_rather_than_as_zero(self) -> None:
        """AST's floor is unmeasured, and a rule admitting everything is not the honest stand-in."""
        assert optional_label_membership(load_triage_config(), "ast") is None

    def test_the_strict_reader_refuses_a_null_override_map(self) -> None:
        """``label_thresholds`` is still an open key, which is what keeps the whole-file fold absent."""
        with pytest.raises(ValueError, match="has no value"):
            load_label_membership(load_triage_config(), "yamnet")

    def test_an_override_changes_the_rule(self, tmp_path: Path) -> None:
        """The pair is configuration, so a partial YAML moves both readers at once."""
        override = tmp_path / "membership.yaml"
        override.write_text("windows:\n  yamnet:\n    label_top_k: 2\n    default_threshold: 0.7\n")
        rule = optional_label_membership(load_triage_config(override), "yamnet")
        assert rule is not None
        assert (rule.top_k, rule.floor) == (2, 0.7)
