"""Tests for ``fit.py``: each corpus statistic computed by its written method (plan step 10)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from senselab.audio.workflows.triage.dataset_profile import FITTED_KIND, profile_values
from senselab.audio.workflows.triage.fit import (
    STATISTICS,
    WHOLE,
    columns_read,
    fit_methods,
    fit_value,
    fit_values,
    fitted_values,
    n_min_defaults,
    n_min_of,
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
        rows: list[dict[str, Any]] = [
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

    def test_a_target_speaker_is_its_own_group_as_cohort_reads_it(self) -> None:
        """Review J1: two target speakers of one participant are two groups, as COHORT groups them, not one."""
        rows = [
            {"participant": "p1", "session": "s1", "target_speaker": "p1:child", "x": 1.0},
            {"participant": "p1", "session": "s1", "target_speaker": "p1:parent", "x": 2.0},
            {"participant": "p1", "session": "s1", "target_speaker": "p1:parent", "x": 2.0},
            {"participant": "p2", "session": "s1", "target_speaker": None, "x": 3.0},
        ]
        by_participant = select(rows, {"column": "x", "per": "participant"}, tasks={}, setting=_no_setting)
        assert [r["x"] for r in by_participant] == [1.0, 2.0, 3.0]
        by_session = select(rows, {"column": "x", "per": "session"}, tasks={}, setting=_no_setting)
        assert [r["x"] for r in by_session] == [1.0, 2.0, 3.0]

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
        method = {"column": "x", "statistic": "median", "decimals": 2, "by": "declared_family", "n_min": 1}
        fit = fit_value(rows, "f.yaml", ("a",), method, tasks=TASKS, setting=_no_setting)
        assert (fit.value, fit.n) == ({"chat": 15.0, "story": 40.0}, {"chat": 2, "story": 1})


class TestTheMinimumCount:
    """Satra review 2026-11-14 (#637-1): a statistic over too few values is not fitted unless allowed."""

    def test_every_statistic_has_a_default_and_a_method_may_raise_it(self) -> None:
        """The defaults cover every statistic; the 99th-percentile count-in gap sets its own, higher one."""
        assert set(n_min_defaults()) == set(STATISTICS)
        assert n_min_of({"statistic": "median"}) == n_min_defaults()["median"]
        assert n_min_of({"statistic": "median", "n_min": 7}) == 7
        gap = next(method for relative, path, method in fit_methods() if path == ("count_in", "gap_max_s"))
        assert n_min_of(gap) > n_min_defaults()["percentile"]

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "ten", True])
    def test_a_minimum_that_is_not_a_count_is_refused(self, bad: Any) -> None:  # noqa: ANN401
        """A method's own n_min must be a whole number of at least one."""
        with pytest.raises(ValueError, match="n_min"):
            n_min_of({"statistic": "median", "n_min": bad})

    def test_every_shipped_percentile_leaves_values_beyond_its_cut(self) -> None:
        """At n_min, five or more values lie beyond each percentile a method fits.

        The derivation is in config-derivations.md, "profile fit: n_min": the cut is then not the sample's extreme.
        """
        for _, path, method in fit_methods():
            if method["statistic"] not in ("percentile", "percentile_band"):
                continue
            qs = method["q"] if isinstance(method["q"], list) else [method["q"]]
            tail = min(min(float(q), 100.0 - float(q)) for q in qs) / 100.0
            assert n_min_of(method) * tail >= 5, path

    def test_a_value_below_its_minimum_is_not_fitted_unless_allowed(self) -> None:
        """Twenty recordings are too few for a median at n_min 30; allowed, the median is written and still named."""
        rows = [{"x": float(i)} for i in range(20)]
        method = {"column": "x", "statistic": "median", "decimals": 2, "n_min": 30}
        refused = fit_value(rows, "f.yaml", ("a",), method, tasks={}, setting=_no_setting)
        assert (refused.value, refused.n, refused.n_min, refused.short) == (None, 20, 30, (WHOLE,))
        allowed = fit_value(rows, "f.yaml", ("a",), method, tasks={}, setting=_no_setting, allow_short=True)
        assert (allowed.value, allowed.short) == (9.5, (WHOLE,))
        enough = fit_value(rows * 2, "f.yaml", ("a",), method, tasks={}, setting=_no_setting)
        assert (enough.value, enough.short) == (9.5, ())

    def test_no_values_at_all_stay_unfitted_and_are_not_short(self) -> None:
        """A column the table never fills is left unfitted, as before, not counted as too few."""
        fit = fit_value(
            [{"x": None}], "f.yaml", ("a",), {"column": "x", "statistic": "median"}, tasks={}, setting=_no_setting
        )
        assert (fit.value, fit.n, fit.short) == (None, 0, ())

    def test_each_family_of_a_value_by_family_needs_its_own_minimum(self) -> None:
        """A family below n_min is left out while the others are fitted; its count is kept."""
        rows = [{"declared_family": "chat", "x": float(i)} for i in range(5)] + [{"declared_family": "story", "x": 1.0}]
        method = {"column": "x", "statistic": "median", "decimals": 2, "by": "declared_family", "n_min": 5}
        fit = fit_value(rows, "f.yaml", ("a",), method, tasks=TASKS, setting=_no_setting)
        assert (fit.value, fit.n, fit.short) == ({"chat": 2.0}, {"chat": 5, "story": 1}, ("story",))
        allowed = fit_value(rows, "f.yaml", ("a",), method, tasks=TASKS, setting=_no_setting, allow_short=True)
        assert allowed.value == {"chat": 2.0, "story": 1.0}


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
        """A table of one participant fits nothing: the participant fence is below its n_min.

        Allowed, it fits that fence and nothing that needs other readings.
        """
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

        def setting(relative: str, key: Any) -> Any:  # noqa: ANN401
            return settings.get((relative, tuple(key)))

        refused = fit_values(rows, tasks=TASKS, setting=setting)
        assert fitted_values(refused) == {}
        assert [fit.key for fit in refused if fit.short] == ["cohort_stage.yaml condition.hot_rms_dbfs_min"]
        fits = fit_values(rows, tasks=TASKS, setting=setting, allow_short=True)
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
