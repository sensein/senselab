"""The family x gate matrix and the disagreement qualification, over synthetic feature records.

The aggregation is what is under test here, not the gates: every fixture pins the readings a gate
sees so each expected count and percentile can be worked out by hand from the docstring.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.routing_analysis.features import RecordingFeatures
from senselab.audio.workflows.triage.routing_analysis.gate_matrix import (
    DECIDING_CLASSES,
    EXTRA,
    MISSED,
    PROFILE_DIR,
    Deciding,
    GateCell,
    gate_matrix,
    gate_order,
    group_disagreements,
    load_disagreement_profile,
    qualify_disagreements,
    relative_margin,
    routing_branch_of,
    unassigned_families,
)
from senselab.audio.workflows.triage.routing_analysis.gate_plot import (
    FIRED_PANEL,
    UNAVAILABLE_PANEL,
    HeatmapStyle,
    branch_boundaries,
    draw_gate_matrix,
    ordered_families,
    panel_values,
)
from senselab.audio.workflows.triage.routing_analysis.report import Confusion
from senselab.audio.workflows.triage.routing_analysis.ruleset import Ruleset, load_ruleset
from senselab.audio.workflows.triage.vocabulary import BRANCHES

COUGH_KEY = "yamnet.cough_labels.peak_over_floor_db_max"
"""Where the ``airway.cough`` gate reads its dB from, and the one key a fixture drops to unread it."""


@pytest.fixture(scope="module")
def ruleset() -> Ruleset:
    """The packaged ruleset.

    Returns:
        The ruleset as ``data/config/default.yaml`` declares it.
    """
    return load_ruleset(load_triage_config())


def _features(family: str, stem: str = "", **overrides: object) -> RecordingFeatures:
    """A feature record whose every gate is readable and silent, before the overrides.

    Args:
        family: The task family.
        stem: The stem, defaulting to one built from the family.
        overrides: Fields to replace wholesale.

    Returns:
        The record.
    """
    record = RecordingFeatures(
        stem=stem or f"sub-1_ses-1_task-{family}",
        run_root="run",
        task_id=family,
        family=family,
        duration_s=10.0,
        words={"agreement": 0, "total": 0, "lexical": 0},
        consensus_present=True,
        transcript="",
        residual={"energy_fraction": 0.0},
        span_longest_s={"amplitude": 0.0},
        span_stats={"all.peak_over_floor_db_max": 0.0},
        span_label_set_stats={COUGH_KEY: 0.0},
        ppg={"silent_fraction": 0.0, "segment_rate_per_s": 0.0},
        classifier_streams=["plain|yamnet"],
    )
    for name, value in overrides.items():
        setattr(record, name, value)
    return record


def _sustained(family: str, seconds: float, stem: str = "") -> RecordingFeatures:
    """A record whose only non-silent reading is ``voice.sustained``, at the given length.

    Args:
        family: The task family.
        seconds: The longest live amplitude span, which is what ``voice.sustained`` reads.
        stem: The stem, defaulting to one built from the family and the length.

    Returns:
        The record.
    """
    return _features(family, stem or f"sub-1_ses-1_task-{family}-{seconds}", span_longest_s={"amplitude": seconds})


class TestUnavailableIsNeverFoldedIntoNotFired:
    """A gate whose evidence was never written did not decline to fire."""

    def test_an_unreadable_gate_counts_as_unavailable_and_not_as_silent(self, ruleset: Ruleset) -> None:
        """Two recordings, one with the cough dB and one without: 1 silent and 1 unavailable.

        Not 2 silent, which is what folding the two together would report.
        """
        records = [
            _features("voluntary-cough", "with", span_label_set_stats={COUGH_KEY: 10.0}),
            _features("voluntary-cough", "without", span_label_set_stats={}),
        ]
        cell = gate_matrix(records, ruleset).cell("voluntary-cough", "airway.cough")
        assert (cell.n, cell.fired, cell.silent, cell.unavailable) == (2, 0, 1, 1)
        assert cell.n_evaluable == 1

    def test_the_two_fired_rates_have_different_denominators(self, ruleset: Ruleset) -> None:
        """One firing of three recordings, one of which was unreadable: 1/3 over all, 1/2 readable.

        Reporting only the first hides the missing evidence; only the second hides its cost.
        """
        records = [
            _features("voluntary-cough", "fires", span_label_set_stats={COUGH_KEY: 99.0}),
            _features("voluntary-cough", "silent", span_label_set_stats={COUGH_KEY: 1.0}),
            _features("voluntary-cough", "unread", span_label_set_stats={}),
        ]
        cell = gate_matrix(records, ruleset).cell("voluntary-cough", "airway.cough")
        assert cell.fired_rate == pytest.approx(1 / 3)
        assert cell.fired_rate_evaluable == pytest.approx(1 / 2)
        assert cell.unavailable_rate == pytest.approx(1 / 3)

    def test_an_unreadable_recording_contributes_no_reading_to_the_distribution(self, ruleset: Ruleset) -> None:
        """The distribution is over the readable recordings only, so its ``n`` is 1 and not 2."""
        records = [
            _features("voluntary-cough", "with", span_label_set_stats={COUGH_KEY: 42.0}),
            _features("voluntary-cough", "without", span_label_set_stats={}),
        ]
        cell = gate_matrix(records, ruleset).cell("voluntary-cough", "airway.cough")
        assert cell.values["n"] == 1.0
        assert cell.median == pytest.approx(42.0)


class TestAFamilyWithNoEvaluableRecordingRendersMissing:
    """A cell nothing could evaluate has no rate, and None is not zero."""

    def test_the_evaluable_fired_rate_is_none_rather_than_zero(self, ruleset: Ruleset) -> None:
        """Every recording of the family is unreadable, so there is no rate over the readable ones.

        A 0.0 here would read as "the gate never fired", which is a claim this cell cannot make.
        """
        records = [_features("voluntary-cough", str(index), span_label_set_stats={}) for index in range(3)]
        cell = gate_matrix(records, ruleset).cell("voluntary-cough", "airway.cough")
        assert cell.unavailable == 3
        assert cell.n_evaluable == 0
        assert cell.fired_rate_evaluable is None
        assert cell.values == {}
        assert cell.median is None and cell.p90 is None

    def test_a_gate_that_never_fires_is_zero_and_not_none(self, ruleset: Ruleset) -> None:
        """The other half of the distinction: readable and silent throughout is a real 0.0."""
        records = [
            _features("voluntary-cough", str(index), span_label_set_stats={COUGH_KEY: 1.0}) for index in range(3)
        ]
        cell = gate_matrix(records, ruleset).cell("voluntary-cough", "airway.cough")
        assert cell.fired_rate_evaluable == 0.0
        assert cell.fired_rate_evaluable is not None

    def test_the_panel_holds_nan_for_the_missing_cell_and_a_number_for_the_never_fired_one(
        self, ruleset: Ruleset
    ) -> None:
        """What the figure is handed keeps the two apart, so the drawing cannot conflate them."""
        records = [
            _features("voluntary-cough", "unread", span_label_set_stats={}),
            _features("breath-sounds", "silent", span_label_set_stats={COUGH_KEY: 1.0}),
        ]
        matrix = gate_matrix(records, ruleset)
        column = matrix.gates.index("airway.cough")
        families = ordered_families(matrix, HeatmapStyle(sort_families_by_n=False))
        grid = panel_values(matrix, families, FIRED_PANEL)
        assert np.isnan(grid[families.index("voluntary-cough"), column])
        assert grid[families.index("breath-sounds"), column] == 0.0

    def test_the_unavailable_panel_is_a_real_rate_where_the_fired_panel_is_missing(self, ruleset: Ruleset) -> None:
        """The cell has no fired rate and does have an unavailable rate of 1.0."""
        records = [_features("voluntary-cough", "unread", span_label_set_stats={})]
        matrix = gate_matrix(records, ruleset)
        grid = panel_values(matrix, matrix.families, UNAVAILABLE_PANEL)
        assert grid[0, matrix.gates.index("airway.cough")] == 1.0

    def test_a_figure_is_drawn_and_hatches_every_missing_cell(self, ruleset: Ruleset) -> None:
        """One hatched patch per cell with no rate, so the legend's count is the drawn count."""
        records = [_features("voluntary-cough", "unread", span_label_set_stats={})]
        matrix = gate_matrix(records, ruleset)
        grid = panel_values(matrix, matrix.families, FIRED_PANEL)
        figure = draw_gate_matrix(matrix, ruleset, panel=FIRED_PANEL)
        patches = figure.axes[0].patches
        assert len(patches) == int(np.count_nonzero(np.isnan(grid))) == 1
        assert patches[0].get_hatch() == HeatmapStyle().missing_hatch


class TestTheValueDistributionIsRight:
    """The percentiles, on a sample small enough to check by hand."""

    def test_the_median_and_p90_of_a_hand_checked_sample(self, ruleset: Ruleset) -> None:
        """``voice.sustained`` reads 1..10 s over ten recordings of one family.

        Sorted: 1 2 3 4 5 6 7 8 9 10. The median of an even sample is the mean of the middle pair,
        (5 + 6) / 2 = 5.5. The p90 is the 9th decile cut by the inclusive method, which places it
        at fractional index 0.9 * (10 - 1) = 8.1 and interpolates from there: the value at index 8
        is 9 and at index 9 is 10, so 9 + 0.1 * (10 - 9) = 9.1. min 1, max 10, and the mean is
        55 / 10 = 5.5.
        """
        records = [_sustained("prolonged-vowel", float(seconds)) for seconds in range(1, 11)]
        cell = gate_matrix(records, ruleset).cell("prolonged-vowel", "voice.sustained")
        assert cell.values["n"] == 10.0
        assert cell.median == pytest.approx(5.5)
        assert cell.p90 == pytest.approx(9.1)
        assert cell.values["min"] == pytest.approx(1.0)
        assert cell.values["max"] == pytest.approx(10.0)
        assert cell.values["mean"] == pytest.approx(5.5)

    def test_a_single_reading_reports_itself_for_every_percentile(self, ruleset: Ruleset) -> None:
        """A sample too small to interpolate reports its own value rather than extrapolating."""
        cell = gate_matrix([_sustained("prolonged-vowel", 4.0)], ruleset).cell("prolonged-vowel", "voice.sustained")
        assert cell.median == pytest.approx(4.0)
        assert cell.p90 == pytest.approx(4.0)

    def test_the_distribution_counts_the_firings_and_the_silences_alike(self, ruleset: Ruleset) -> None:
        """It is a reading distribution, not a firing distribution: 1 s and 9 s are both in it.

        ``voice.sustained`` fires at 3.0 s, so 1 s is silent and 9 s fires, and the median of the
        two readings is 5.0.
        """
        records = [_sustained("prolonged-vowel", 1.0), _sustained("prolonged-vowel", 9.0)]
        cell = gate_matrix(records, ruleset).cell("prolonged-vowel", "voice.sustained")
        assert (cell.fired, cell.silent) == (1, 1)
        assert cell.median == pytest.approx(5.0)


class TestTheMatrixAxes:
    """Which rows and columns exist, and what a cell that was never seen reports."""

    def test_every_family_and_gate_pair_is_keyed(self, ruleset: Ruleset) -> None:
        """A cell absent from the mapping would render as a hole rather than as a measurement."""
        records = [_features("prolonged-vowel"), _features("breath-sounds")]
        matrix = gate_matrix(records, ruleset)
        assert matrix.families == ("breath-sounds", "prolonged-vowel")
        assert set(matrix.gates) == set(ruleset.gates)
        assert len(matrix.cells) == len(matrix.families) * len(matrix.gates)
        assert len(matrix.rows()) == len(matrix.cells)

    def test_the_columns_are_the_routing_gates_in_branch_order_then_the_flags(self, ruleset: Ruleset) -> None:
        """A reader scanning a branch's gates finds them adjacent, and flags are not among them."""
        order = gate_order(ruleset)
        routing = [name for name in order if routing_branch_of(ruleset, name) is not None]
        flags = [name for name in order if routing_branch_of(ruleset, name) is None]
        assert order == tuple(routing + flags)
        assert [routing_branch_of(ruleset, name) for name in routing] == sorted(
            (routing_branch_of(ruleset, name) for name in routing),
            key=lambda branch: BRANCHES.index(str(branch)),
        )

    def test_a_flag_gate_is_measured_although_it_routes_nothing(self, ruleset: Ruleset) -> None:
        """Leaving it out would make the matrix silent about a gate the run evaluates."""
        matrix = gate_matrix([_features("prolonged-vowel")], ruleset)
        assert "speech.transcript_agreement" in matrix.gates
        assert routing_branch_of(ruleset, "speech.transcript_agreement") is None

    def test_a_gate_no_branch_names_is_still_a_column(self, ruleset: Ruleset) -> None:
        """A gate the config defines and no branch references would otherwise go unmeasured.

        The packaged config has no such gate, so nothing here can be read off it: the ruleset is
        rebuilt with one added. Without the fallback that gate is silently absent from the matrix
        and the run reports success having measured 11 of 12 gates.
        """
        from dataclasses import replace

        from senselab.audio.workflows.triage.routing_analysis.ruleset import AT_LEAST, Gate

        orphan = Gate(name="zz.orphan", feature=("words", "total"), op=AT_LEAST, threshold=1.0)
        widened = replace(ruleset, gates={**ruleset.gates, orphan.name: orphan})
        assert orphan.name not in {
            name for names in (*widened.branch_gates.values(), *widened.branch_flags.values()) for name in names
        }
        order = gate_order(widened)
        assert orphan.name in order
        assert order[-1] == orphan.name
        assert orphan.name in gate_matrix([_features("prolonged-vowel")], widened).gates

    def test_the_branch_rules_fall_between_branches(self, ruleset: Ruleset) -> None:
        """One rule per change of branch along the columns, and none inside a branch's run."""
        matrix = gate_matrix([_features("prolonged-vowel")], ruleset)
        boundaries = branch_boundaries(matrix, ruleset)
        for index in boundaries:
            assert routing_branch_of(ruleset, matrix.gates[index]) != routing_branch_of(
                ruleset, matrix.gates[index - 1]
            )
        assert len(boundaries) == len({routing_branch_of(ruleset, gate) for gate in matrix.gates}) - 1

    def test_the_per_family_count_is_the_recording_count(self, ruleset: Ruleset) -> None:
        """Every cell of a row shares the row's ``n``, whatever each gate could read."""
        records = [_features("prolonged-vowel", str(index)) for index in range(4)]
        records[0].span_label_set_stats = {}
        matrix = gate_matrix(records, ruleset)
        assert matrix.recordings["prolonged-vowel"] == 4
        assert {matrix.cell("prolonged-vowel", gate).n for gate in matrix.gates} == {4}

    def test_an_empty_matrix_refuses_to_draw(self, ruleset: Ruleset) -> None:
        """A figure with no rows is a corpus that read nothing, not a blank page to publish."""
        matrix = gate_matrix([], ruleset)
        assert matrix.families == ()
        with pytest.raises(ValueError, match="nothing to draw"):
            draw_gate_matrix(matrix, ruleset)


class TestTheRelativeMargin:
    """One band has to cover gates in seconds, dB, counts and probabilities."""

    def test_it_is_positive_when_the_gate_fires_and_negative_when_it_does_not(self, ruleset: Ruleset) -> None:
        """``voice.sustained`` cuts at 3.0 s: 3.75 s is +0.25 and 2.25 s is -0.25."""
        gate = ruleset.gates["voice.sustained"]
        assert relative_margin(3.75, gate) == pytest.approx(0.25)
        assert relative_margin(2.25, gate) == pytest.approx(-0.25)
        assert relative_margin(3.0, gate) == pytest.approx(0.0)

    def test_an_at_most_gate_is_signed_the_same_way(self, ruleset: Ruleset) -> None:
        """Firing is positive however the comparison points, or one band would mean two things."""
        from senselab.audio.workflows.triage.routing_analysis.ruleset import AT_MOST, Gate

        gate = Gate(name="test", feature=("words", "lexical"), op=AT_MOST, threshold=4.0)
        assert relative_margin(3.0, gate) == pytest.approx(0.25)
        assert relative_margin(5.0, gate) == pytest.approx(-0.25)

    def test_a_zero_threshold_does_not_divide_by_zero(self) -> None:
        """It is scaled by 1 instead, leaving the margin in the gate's own units."""
        from senselab.audio.workflows.triage.routing_analysis.ruleset import AT_LEAST, Gate

        gate = Gate(name="test", feature=("words", "lexical"), op=AT_LEAST, threshold=0.0)
        assert relative_margin(2.0, gate) == pytest.approx(2.0)


class TestTheThreeFindingsAreDistinct:
    """A threshold question, an evidence question and a missing-feature question are not one thing."""

    def test_a_miss_just_short_of_the_cut_is_a_threshold_question(self, ruleset: Ruleset) -> None:
        """``prolonged-vowel`` declares VOICE. 2.9 s of sustained phonation is -0.033, inside 0.25."""
        record = _sustained("prolonged-vowel", 2.9)
        [entry] = qualify_disagreements([record], ruleset, near_threshold_band=0.25)
        assert (entry.kind, entry.branch) == (MISSED, "VOICE")
        assert entry.deciding_gate == "voice.sustained"
        assert entry.value == pytest.approx(2.9)
        assert entry.margin == pytest.approx(-0.1 / 3.0)
        assert entry.finding is Deciding.NEAR_THRESHOLD

    def test_the_same_miss_far_from_the_cut_is_an_evidence_question(self, ruleset: Ruleset) -> None:
        """0.3 s is -0.9, outside the band: no threshold move of that size recovers the branch."""
        [entry] = qualify_disagreements([_sustained("prolonged-vowel", 0.3)], ruleset, near_threshold_band=0.25)
        assert entry.finding is Deciding.FAR
        assert entry.margin == pytest.approx(-0.9)

    def test_a_miss_with_every_gate_unreadable_is_a_missing_feature_question(self, ruleset: Ruleset) -> None:
        """No gate of the branch could be read, so there is no threshold to question.

        AIRWAY has four gates; a record that unreads all four is declared AIRWAY, missed, and
        carries no deciding gate at all.
        """
        record = _features("voluntary-cough", residual={}, span_label_set_stats={}, ppg={}, words={})
        record.consensus_present = False
        entries = {entry.branch: entry for entry in qualify_disagreements([record], ruleset, near_threshold_band=0.25)}
        entry = entries["AIRWAY"]
        assert entry.kind == MISSED
        assert entry.finding is Deciding.UNAVAILABLE
        assert entry.deciding_gate is None
        assert entry.value is None and entry.margin is None
        assert entry.n_unavailable_gates == entry.n_gates == 4

    def test_the_band_is_the_only_thing_separating_near_from_far(self, ruleset: Ruleset) -> None:
        """The same recording reads near under a wide band and far under a narrow one.

        Which is what makes the band a reporting convention rather than a measurement: a reader
        who moves it sees the table move.
        """
        record = _sustained("prolonged-vowel", 2.5)
        [wide] = qualify_disagreements([record], ruleset, near_threshold_band=0.25)
        [narrow] = qualify_disagreements([record], ruleset, near_threshold_band=0.01)
        assert wide.finding is Deciding.NEAR_THRESHOLD
        assert narrow.finding is Deciding.FAR
        assert wide.margin == narrow.margin

    def test_an_extra_turns_on_the_firing_gate_that_cleared_its_cut_by_least(self, ruleset: Ruleset) -> None:
        """A breath-sounds recording routed to VOICE by two gates is reported under the closer one.

        ``voice.sustained`` at 3.3 s is +0.10; ``voice.chant`` at 0.9 is +44. The threshold
        question is about the first, so that is the deciding gate.
        """
        record = _features(
            "breath-sounds",
            span_longest_s={"amplitude": 3.3},
            peaks={"plain|yamnet|Chant": 0.9},
        )
        extras = [
            entry for entry in qualify_disagreements([record], ruleset, near_threshold_band=0.25) if entry.kind == EXTRA
        ]
        voice = [entry for entry in extras if entry.branch == "VOICE"]
        assert len(voice) == 1
        assert voice[0].deciding_gate == "voice.sustained"
        assert voice[0].margin == pytest.approx(0.1)
        assert voice[0].finding is Deciding.NEAR_THRESHOLD

    def test_an_extra_is_never_a_missing_feature_question(self, ruleset: Ruleset) -> None:
        """Something fired, so the branch was routed; unread gates beside it are reported apart.

        The AIRWAY extra here fires on the residual fraction while the cough dB is unreadable:
        the finding is about the firing gate, and ``n_unavailable_gates`` carries the other fact.
        """
        record = _features("prolonged-vowel", residual={"energy_fraction": 0.5}, span_label_set_stats={})
        extras = [
            entry for entry in qualify_disagreements([record], ruleset, near_threshold_band=0.25) if entry.kind == EXTRA
        ]
        [airway] = [entry for entry in extras if entry.branch == "AIRWAY"]
        assert airway.finding is not Deciding.UNAVAILABLE
        assert airway.deciding_gate == "airway.breath"
        assert airway.n_unavailable_gates == 1

    def test_every_disagreement_lands_in_exactly_one_finding(self, ruleset: Ruleset) -> None:
        """The three classes partition the disagreements; a group's findings sum to its ``n``."""
        records = [
            _sustained("prolonged-vowel", 2.9),
            _sustained("prolonged-vowel", 0.3),
            _features("prolonged-vowel", "extra", residual={"energy_fraction": 0.5}),
        ]
        groups = group_disagreements(qualify_disagreements(records, ruleset, near_threshold_band=0.25))
        assert groups
        for group in groups:
            assert set(group.findings) == set(DECIDING_CLASSES)
            assert sum(group.findings.values()) == group.n


class TestTheGroupsAggregateByTaskFamily:
    """The misses are asked about per task, which is the grouping the register needs."""

    def test_two_families_missing_the_same_branch_stay_two_groups(self, ruleset: Ruleset) -> None:
        """Folding them would lose which task the threshold question is about."""
        records = [_sustained("prolonged-vowel", 2.9), _sustained("glides-low-to-high", 2.9)]
        groups = group_disagreements(qualify_disagreements(records, ruleset, near_threshold_band=0.25))
        misses = {group.family for group in groups if group.kind == MISSED and group.branch == "VOICE"}
        assert misses == {"prolonged-vowel", "glides-low-to-high"}

    def test_a_group_names_its_deciding_gates_and_counts_them(self, ruleset: Ruleset) -> None:
        """Two recordings of one family missing VOICE on the same gate report it once, with 2."""
        records = [_sustained("prolonged-vowel", 2.9, "a"), _sustained("prolonged-vowel", 2.8, "b")]
        groups = group_disagreements(qualify_disagreements(records, ruleset, near_threshold_band=0.25))
        [group] = [group for group in groups if group.kind == MISSED and group.branch == "VOICE"]
        assert group.n == 2
        assert group.gates == {"voice.sustained": 2}
        assert group.gates_summary() == "voice.sustained:2"

    def test_a_group_with_no_readable_gate_reports_a_dash_and_no_margins(self, ruleset: Ruleset) -> None:
        """There is no margin to summarise, which is the point of the unavailable class."""
        record = _features("voluntary-cough", residual={}, span_label_set_stats={}, ppg={}, words={})
        record.consensus_present = False
        groups = group_disagreements(qualify_disagreements([record], ruleset, near_threshold_band=0.25))
        [group] = [group for group in groups if group.branch == "AIRWAY" and group.kind == MISSED]
        assert group.findings["unavailable"] == 1
        assert group.margins == {}
        assert group.gates_summary() == "-"

    def test_the_margin_distribution_of_a_group_is_hand_checkable(self, ruleset: Ruleset) -> None:
        """Three misses at 1.5, 2.0 and 2.5 s against a 3.0 s cut: margins -0.5, -1/3, -1/6.

        Sorted that is -0.5, -0.333, -0.167, whose median is -1/3.
        """
        records = [_sustained("prolonged-vowel", seconds, str(seconds)) for seconds in (1.5, 2.0, 2.5)]
        groups = group_disagreements(qualify_disagreements(records, ruleset, near_threshold_band=1.0))
        [group] = [group for group in groups if group.kind == MISSED and group.branch == "VOICE"]
        assert group.n == 3
        assert group.margins["median"] == pytest.approx(-1 / 3)
        assert group.margins["min"] == pytest.approx(-0.5)
        assert group.margins["max"] == pytest.approx(-1 / 6)

    def test_misses_are_reported_before_extras(self, ruleset: Ruleset) -> None:
        """The under-routing side is the one that loses a branch, so it leads the table."""
        records = [
            _sustained("prolonged-vowel", 2.9),
            _features("prolonged-vowel", "extra", residual={"energy_fraction": 0.5}),
        ]
        kinds = [
            group.kind
            for group in group_disagreements(qualify_disagreements(records, ruleset, near_threshold_band=0.25))
        ]
        assert kinds.index(MISSED) < kinds.index(EXTRA)


class TestTheBranchAgreementRates:
    """Recall, precision and specificity, each with its own denominator."""

    def test_precision_is_over_the_routings_and_specificity_over_the_negatives(self) -> None:
        """A 2x2 of tp 3, fp 7, tn 13, fn 2 by hand.

        recall 3/(3+2) = 0.6, precision 3/(3+7) = 0.3, specificity 13/(13+7) = 0.65.
        """
        table = Confusion(tp=3, fp=7, tn=13, fn=2)
        assert table.sensitivity == pytest.approx(0.6)
        assert table.precision == pytest.approx(0.3)
        assert table.specificity == pytest.approx(0.65)

    def test_precision_is_none_when_the_branch_never_routed(self) -> None:
        """No firings means no fraction of firings, which is not a precision of zero."""
        assert Confusion(tp=0, fp=0, tn=9, fn=4).precision is None

    def test_precision_reaches_the_machine_readable_output(self) -> None:
        """The over-routing side has to be readable off the JSON, not only off the table."""
        assert Confusion(tp=1, fp=1, tn=1, fn=1).as_json()["precision"] == pytest.approx(0.5)


class TestUnassignedFamiliesAreReportedNotAbsorbed:
    """A family in no reference set is in no denominator, and must not sit silently in a negative."""

    def test_a_family_no_reference_set_carries_is_listed_with_its_count(self, ruleset: Ruleset) -> None:
        """``word-color-stroop`` or any unrecognised id belongs in its own row."""
        records = [_features("not-a-real-family", str(index)) for index in range(3)]
        assert unassigned_families(records, ruleset) == {"not-a-real-family": 3}

    def test_an_assigned_family_is_not_listed(self, ruleset: Ruleset) -> None:
        """Every branch-assigned family is scored, so listing it here would double-report it."""
        records = [_features("prolonged-vowel"), _features("voluntary-cough")]
        assert unassigned_families(records, ruleset) == {}

    def test_the_packaged_config_assigns_each_named_family_set_it_references(self, ruleset: Ruleset) -> None:
        """The reference mapping is read from the config, never restated in the analysis."""
        assert set(ruleset.reference_family_set) == set(BRANCHES)
        assert ruleset.reference_family_set["VOICE"] == "voice"


class TestTheDisagreementProfile:
    """The band is a declared value in ``data/``, refused rather than defaulted in code."""

    def test_the_packaged_profile_loads_and_carries_a_positive_band(self) -> None:
        """A band nobody wrote down would make the three-way split unreproducible."""
        profile = load_disagreement_profile()
        assert profile["near_threshold_band"] > 0.0
        assert profile["source"].endswith(".yaml")

    def test_the_profile_carries_no_prose_value(self) -> None:
        """Its reasoning is in ``specs/``, matching the rule the triage config is held to."""
        profile = load_disagreement_profile()
        for key, value in profile.items():
            assert not (isinstance(value, str) and "\n" in value), f"{key} carries prose"
        assert "derivation" not in profile

    def test_a_named_profile_that_is_absent_is_refused(self, tmp_path: Path) -> None:
        """An operator naming a missing file is an error, not a reason to use the bundled one."""
        with pytest.raises(FileNotFoundError):
            load_disagreement_profile(str(tmp_path / "nope.yaml"))

    def test_an_unknown_schema_version_is_refused(self, tmp_path: Path) -> None:
        """A profile of a shape this reader does not know is not read optimistically."""
        path = tmp_path / "bad.yaml"
        path.write_text('schema_version: "99"\nnear_threshold_band: 0.5\n')
        with pytest.raises(ValueError, match="schema_version"):
            load_disagreement_profile(str(path))

    @pytest.mark.parametrize("band", ["0", "-0.2", "null", "true", '"0.25"'])
    def test_a_band_that_is_not_a_positive_number_is_refused(self, tmp_path: Path, band: str) -> None:
        """Zero, negative, absent, boolean and string bands are all unusable as a margin cut.

        Args:
            tmp_path: Where the profile is written.
            band: The unusable value, as YAML.
        """
        path = tmp_path / f"band-{band}.yaml"
        path.write_text(f'schema_version: "1"\nnear_threshold_band: {band}\n')
        with pytest.raises(ValueError, match="near_threshold_band"):
            load_disagreement_profile(str(path))

    def test_the_bundled_profile_is_the_newest_dated_one(self) -> None:
        """Dated files, newest wins, matching the classifier-ontology idiom."""
        bundled = sorted(PROFILE_DIR.glob("*.yaml"))
        assert bundled
        assert load_disagreement_profile()["source"] == str(bundled[-1])


class TestThePlotCarriesNoAnalysisKnob:
    """The style governs the drawing; nothing in it can change a rate the figure shows."""

    def test_no_style_field_changes_any_cell_of_the_matrix(self, ruleset: Ruleset) -> None:
        """Two very different styles draw the same numbers, because the numbers are not theirs."""
        records = [_sustained("prolonged-vowel", 4.0), _features("voluntary-cough", span_label_set_stats={})]
        matrix = gate_matrix(records, ruleset)
        plain = HeatmapStyle()
        loud = HeatmapStyle(fired_cmap="plasma", annotate=False, cell_width_in=1.5, dpi=72, sort_families_by_n=False)
        families = ordered_families(matrix, plain)
        assert np.array_equal(
            panel_values(matrix, families, FIRED_PANEL),
            panel_values(matrix, families, FIRED_PANEL),
            equal_nan=True,
        )
        for style in (plain, loud):
            grid = panel_values(matrix, ordered_families(matrix, style), FIRED_PANEL)
            assert np.nansum(grid) == pytest.approx(
                sum(
                    cell.fired_rate_evaluable
                    for cell in (matrix.cell(family, gate) for family in matrix.families for gate in matrix.gates)
                    if cell.fired_rate_evaluable is not None
                )
            )

    def test_sorting_reorders_rows_without_changing_a_row(self, ruleset: Ruleset) -> None:
        """Row order is a drawing choice; each family keeps its own readings either way."""
        records = [_features("prolonged-vowel", "a"), _features("prolonged-vowel", "b"), _features("breath-sounds")]
        matrix = gate_matrix(records, ruleset)
        by_name = ordered_families(matrix, HeatmapStyle(sort_families_by_n=False))
        by_count = ordered_families(matrix, HeatmapStyle(sort_families_by_n=True))
        assert by_name == ("breath-sounds", "prolonged-vowel")
        assert by_count == ("prolonged-vowel", "breath-sounds")
        assert set(by_name) == set(by_count)

    def test_an_unknown_panel_is_refused(self, ruleset: Ruleset) -> None:
        """A typo'd panel name would otherwise draw a blank figure and report success."""
        matrix = gate_matrix([_features("prolonged-vowel")], ruleset)
        with pytest.raises(ValueError, match="panel is not one of"):
            draw_gate_matrix(matrix, ruleset, panel="fired_rate")

    def test_a_cell_with_no_rate_carries_no_annotation(self, ruleset: Ruleset) -> None:
        """Writing a number over a hatched cell would undo the distinction the hatch makes."""
        records = [_features("voluntary-cough", span_label_set_stats={})]
        matrix = gate_matrix(records, ruleset)
        grid = panel_values(matrix, matrix.families, FIRED_PANEL)
        figure = draw_gate_matrix(matrix, ruleset, panel=FIRED_PANEL, style=HeatmapStyle(annotate=True))
        drawn = [text for text in figure.axes[0].texts if text.get_text()]
        assert len(drawn) == int(np.count_nonzero(~np.isnan(grid)))


class TestGateCellArithmetic:
    """The rates, on a cell built by hand rather than reduced from records."""

    def test_the_rates_of_a_hand_built_cell(self) -> None:
        """A cell of 10 recordings: 2 fired, 3 silent, 5 unavailable.

        fired_rate 2/10 = 0.2, fired_rate_evaluable 2/5 = 0.4, unavailable_rate 5/10 = 0.5.
        """
        cell = GateCell(family="f", gate="g", n=10, fired=2, silent=3, unavailable=5, values={})
        assert cell.n_evaluable == 5
        assert cell.fired_rate == pytest.approx(0.2)
        assert cell.fired_rate_evaluable == pytest.approx(0.4)
        assert cell.unavailable_rate == pytest.approx(0.5)

    def test_an_empty_cell_has_no_rates_at_all(self) -> None:
        """No recordings means no denominator anywhere, and None rather than zero throughout."""
        cell = GateCell(family="f", gate="g", n=0, fired=0, silent=0, unavailable=0, values={})
        assert cell.fired_rate is None
        assert cell.fired_rate_evaluable is None
        assert cell.unavailable_rate is None

    def test_the_json_row_keeps_a_none_rate_as_none(self) -> None:
        """A null in the parquet column is an unmeasured rate; a 0.0 would be a measured one."""
        row = GateCell(family="f", gate="g", n=3, fired=0, silent=0, unavailable=3, values={}).as_json()
        assert row["fired_rate_evaluable"] is None
        assert row["unavailable_rate"] == pytest.approx(1.0)
        assert row["fired_rate"] == pytest.approx(0.0)
