"""The near match between a transcript token and the stimulus token it is a spelling of.

The bound is fitted in ``specs/20260923-pii-near-match-and-expected-names/near-match-and-expected-names.md``
over the 62,548-run ``triage_rerun_20260923`` corpus; this file pins the shape and the config
contract, not the numbers, except where a number *is* the contract.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.stimulus import NearMatch, near_match


class TestTheBoundIsConfiguredNotLiteral:
    """A tolerance nobody fitted is worse than none; the packaged config carries the fit."""

    def test_the_packaged_config_carries_both_knobs(self) -> None:
        """Two keys, read together, so an override cannot leave half a rule in place."""
        config = load_triage_config()
        assert config.require("stimulus.near_match.exact_below") == 5
        assert config.require("stimulus.near_match.two_edits_from") == 8

    def test_the_node_reads_the_bound_from_the_config(self) -> None:
        """No literal in the matcher; the run's own config is what decides."""
        assert near_match(load_triage_config()) == NearMatch(exact_below=5, two_edits_from=8)

    def test_a_two_edits_bound_below_the_exact_bound_is_refused(self) -> None:
        """An inverted pair would silently make the short tokens the loosest ones."""
        with pytest.raises(ValueError, match="two_edits_from"):
            NearMatch(exact_below=9, two_edits_from=8).tolerance(4)


class TestTheToleranceGrowsWithTheToken:
    """A one-edit slack on a three-letter word is a different word; on a nine-letter one it is a typo."""

    def test_a_short_token_must_match_exactly(self) -> None:
        """`cat`/`cap`/`bat` are one edit apart and are three different words."""
        near = NearMatch(exact_below=5, two_edits_from=8)
        assert near.tolerance(1) == 0
        assert near.tolerance(4) == 0
        assert not near.matches("cat", "cap")

    def test_a_middling_token_may_differ_by_one(self) -> None:
        """The recogniser's own spelling variation on a five-to-seven letter word."""
        near = NearMatch(exact_below=5, two_edits_from=8)
        assert near.tolerance(5) == 1
        assert near.tolerance(7) == 1
        assert near.matches("rainbow", "rainbo")
        assert not near.matches("rainbow", "rainy")

    def test_a_long_token_may_differ_by_two(self) -> None:
        """Length is what makes two edits still the same word."""
        near = NearMatch(exact_below=5, two_edits_from=8)
        assert near.tolerance(8) == 2
        assert near.matches("cinderella", "cindarela")

    def test_the_longer_of_the_pair_sets_the_tolerance(self) -> None:
        """Otherwise a deletion down to four characters would be judged as a short token."""
        near = NearMatch(exact_below=5, two_edits_from=8)
        assert near.tolerance(max(len("cats"), len("catsup"))) == 1


class TestTheRunIsStillContiguousAndOrdered:
    """Widening the comparison must not widen the predicate around it."""

    def test_a_run_of_near_matches_is_found(self) -> None:
        """Every position near its counterpart, in order, with no gap."""
        near = NearMatch(exact_below=5, two_edits_from=8)
        assert near.run_offset(["the", "rainbow", "passage"], ["rainbo", "passag"]) == 1

    def test_a_run_that_is_out_of_order_is_not_found(self) -> None:
        """Order is what makes it the stimulus rather than its vocabulary."""
        near = NearMatch(exact_below=5, two_edits_from=8)
        assert near.run_offset(["the", "rainbow", "passage"], ["passag", "rainbo"]) is None

    def test_a_run_with_a_gap_is_not_found(self) -> None:
        """A subsequence is not a run, and never was."""
        near = NearMatch(exact_below=5, two_edits_from=8)
        assert near.run_offset(["the", "rainbow", "of", "passage"], ["rainbo", "passag"]) is None

    def test_an_empty_needle_matches_nothing(self) -> None:
        """A detector that returned nothing found nothing; it did not find everything."""
        near = NearMatch(exact_below=5, two_edits_from=8)
        assert near.run_offset(["the", "rainbow"], []) is None

    def test_an_exact_rule_reproduces_the_shipped_behaviour(self) -> None:
        """The bound that admits nothing extra is expressible, which is what makes it a fit."""
        exact = NearMatch(exact_below=99, two_edits_from=99)
        assert exact.run_offset(["rainbow"], ["rainbo"]) is None
        assert exact.run_offset(["rainbow"], ["rainbow"]) == 0
