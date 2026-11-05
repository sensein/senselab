"""Tests for ``fit.py``: each corpus statistic computed by its written method (plan step 10)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from senselab.audio.workflows.triage.dataset_profile import FITTED_KIND, profile_values
from senselab.audio.workflows.triage.fit import (
    columns_read,
    fit_methods,
    fit_value,
    fit_values,
    fitted_values,
    select,
    statistic,
    task_families,
)
from senselab.audio.workflows.triage.recording_vectors import FIT_COLUMNS, schema

TASKS = {
    "chat": {"branch": "SPEECH", "expects": {"pattern": "free_response"}},
    "story": {"branch": "SPEECH", "expects": {"pattern": "free_response", "connected": True}},
    "read": {"branch": "SPEECH", "expects": {"pattern": "read_text"}},
}


def _no_setting(key: Any) -> Any:  # noqa: ANN401
    raise AssertionError(f"no setting expected, got {key}")


class TestTheStatistics:
    """Each statistic, rounded as its method says."""

    def test_a_tukey_fence_is_a_quartile_and_a_half_interquartile_range_beyond(self) -> None:
        """Q1 = 2, Q3 = 4 over 1..5 (linear quantiles): the upper fence 7, the lower -1."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert statistic(values, {"statistic": "tukey_upper", "decimals": 3}) == 7.0
        assert statistic(values, {"statistic": "tukey_lower", "decimals": 3}) == -1.0

    def test_a_fence_is_clamped_where_its_method_says(self) -> None:
        """A share cannot fence above one."""
        values = np.array([0.1, 0.5, 0.9, 0.95])
        assert statistic(values, {"statistic": "tukey_upper", "decimals": 3, "clamp": [0, 1]}) == 1.0

    def test_percentiles_interpolate_linearly_and_round(self) -> None:
        """The 97.5th percentile of 0..100 is 97.5; the band [2.5, 97.5] its two ends."""
        values = np.arange(101, dtype=float)
        assert statistic(values, {"statistic": "percentile", "q": 97.5, "decimals": 3}) == 97.5
        assert statistic(values, {"statistic": "percentile_band", "q": [2.5, 97.5], "decimals": 1}) == [2.5, 97.5]
        assert statistic(np.array([1.0, 2.0, 4.0]), {"statistic": "median", "decimals": 2}) == 2.0

    def test_no_values_fit_nothing(self) -> None:
        """An empty column leaves the value unfitted rather than inventing one."""
        assert statistic(np.array([]), {"statistic": "median"}) is None


class TestWhichRowsAMethodReads:
    """One value per group, filtered by the method."""

    def test_a_participant_reading_counts_once_however_many_recordings_repeat_it(self) -> None:
        """COHORT repeats a participant's reading on each recording; the fence takes one per participant."""
        rows = [
            {"participant": "p1", "session": "s1", "x": 1.0},
            {"participant": "p1", "session": "s2", "x": 1.0},
            {"participant": "p2", "session": "s1", "x": 3.0},
            {"participant": None, "session": None, "x": 9.0},
        ]
        assert [
            r["participant"] for r in select(rows, {"column": "x", "per": "participant"}, tasks={}, setting=_no_setting)
        ] == [
            "p1",
            "p2",
        ]
        assert len(select(rows, {"column": "x", "per": "session"}, tasks={}, setting=_no_setting)) == 3
        assert len(select(rows, {"column": "x"}, tasks={}, setting=_no_setting)) == 4

    def test_a_filter_may_read_another_value_of_the_profile(self) -> None:
        """The shared non-match fence reads only participants compared ``recordings_min`` times or more."""
        rows = [{"participant": f"p{n}", "x": 0.5, "compared": n} for n in range(1, 6)]
        method = {"column": "x", "per": "participant", "where": [{"column": "compared", "op": "ge", "setting": ["m"]}]}
        chosen = select(rows, method, tasks={}, setting=lambda key: 3 if list(key) == ["m"] else None)
        assert [r["compared"] for r in chosen] == [3, 4, 5]

    def test_a_task_filter_reads_the_task_list(self) -> None:
        """``tasks: {pattern: free_response}`` keeps the families whose ``expects`` say so."""
        assert task_families(TASKS, {"pattern": "free_response"}) == {"chat", "story"}
        rows = [{"declared_family": family, "x": 1.0} for family in ("chat", "story", "read")]
        method = {"column": "x", "tasks": {"pattern": "free_response"}}
        assert [r["declared_family"] for r in select(rows, method, tasks=TASKS, setting=_no_setting)] == [
            "chat",
            "story",
        ]

    def test_a_value_by_family_is_one_median_per_family_with_its_count(self) -> None:
        """The per-task response medians."""
        rows = [{"declared_family": f, "x": x} for f, x in (("chat", 10.0), ("chat", 20.0), ("story", 40.0))]
        method = {"column": "x", "statistic": "median", "decimals": 2, "by": "declared_family"}
        fit = fit_value(rows, "f.yaml", ("a",), method, tasks=TASKS, setting=_no_setting)
        assert (fit.value, fit.n) == ({"chat": 15.0, "story": 40.0}, {"chat": 2, "story": 1})


class TestTheMethodsSenselabShips:
    """Every corpus statistic carries a method over the table's columns."""

    def test_every_corpus_statistic_has_a_method(self) -> None:
        """The ten of step 5 and the per-task response medians."""
        kind_b = [
            (relative, tuple(entry["path"]))
            for relative, entries in profile_values().items()
            for entry in entries
            if entry.get("kind") == FITTED_KIND
        ]
        assert sorted(kind_b) == sorted((relative, path) for relative, path, _ in fit_methods())
        assert len(kind_b) == 11

    def test_every_column_a_method_reads_is_in_the_corpus_table(self) -> None:
        """Including the fit columns the table carries from schema 32."""
        assert set(columns_read()) <= set(schema().names)
        assert {method["column"] for _, _, method in fit_methods()} - {"gate_response_min_s"} <= set(FIT_COLUMNS)

    def test_fitting_a_table_nests_the_values_by_file_and_leaves_out_what_it_could_not_fit(self) -> None:
        """A table of one participant fits the participant fences and nothing that needs other readings."""
        rows = [
            {
                "participant": "p1",
                "session": "s1",
                "declared_family": "chat",
                "cohort_participant_rms_dbfs_median": -20.0,
            }
        ]
        rows = [{**{name: None for name in columns_read()}, **row} for row in rows]
        settings = {("cohort_stage.yaml", ("shared_nonmatch", "recordings_min")): 3}
        fits = fit_values(rows, tasks=TASKS, setting=lambda relative, key: settings.get((relative, tuple(key))))
        values = fitted_values(fits)
        assert values == {"cohort_stage.yaml": {"condition": {"hot_rms_dbfs_min": -20.0}}}
        assert {fit.key: fit.n for fit in fits}["cohort_stage.yaml condition.hot_rms_dbfs_min"] == 1


@pytest.mark.parametrize("statistic_name", ["tukey_upper", "tukey_lower", "percentile", "percentile_band", "median"])
def test_every_statistic_a_method_may_name_is_computed(statistic_name: str) -> None:
    """No method names a statistic the module cannot compute."""
    method: dict[str, Any] = {
        "statistic": statistic_name,
        "decimals": 3,
        "q": [2.5, 97.5] if statistic_name == "percentile_band" else 50,
    }
    assert statistic(np.array([1.0, 2.0, 3.0]), method) is not None
