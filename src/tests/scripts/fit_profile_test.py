"""Tests for ``scripts/fit_profile.py``: a pilot table's corpus statistics written into the profile (plan step 10)."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml  # type: ignore[import-untyped]

from senselab.audio.workflows.triage.dataset_profile import deactivate_profile, load_profile
from senselab.audio.workflows.triage.fit import columns_read

cli = importlib.import_module("senselab.audio.workflows.triage.drivers.fit_profile")

TASKS = "version: 1\ntasks:\n  chat:\n    branch: SPEECH\n    expects: {pattern: free_response}\n"


def _profile(root: Path, own: dict[str, str] | None = None) -> Path:
    """A pilot profile extending the reference profile, with its task list and any files of its own."""
    root.mkdir(parents=True)
    (root / "tasks.yaml").write_text(TASKS)
    files = {"tasks.yaml": "tasks.yaml"}
    for relative, text in (own or {}).items():
        (root / relative).write_text(text)
        files[relative] = relative
    manifest = {
        "name": "lab",
        "version": "0",
        "pilot": True,
        "extends": "senselab:adult_clinical_speech",
        "files": files,
    }
    (root / "profile.yaml").write_text(yaml.safe_dump(manifest))
    return root


def _table(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> Path:
    """A corpus table holding ``columns`` (every column a fit reads by default), null where a row leaves one out."""
    names = columns if columns is not None else columns_read()
    data = {name: [row.get(name) for row in rows] for name in names}
    pq.write_table(pa.table(data), path)
    return path


def _rows() -> list[dict[str, Any]]:
    """Twelve recordings of four participants: chat responses, cough medians and count-in gaps."""
    return [
        {
            "participant": f"p{i % 4}",
            "session": f"s{i % 2}",
            "declared_family": "chat",
            "gate_response_min_s": 20.0 + i,
            "lexical_words_per_s": 2.0 + 0.1 * i,
            "cough_event_s_median": 0.3 + 0.05 * i,
            "count_in_gap_s": 0.5 + 0.1 * i,
            "cohort_participant_rms_dbfs_median": -30.0 + (i % 4),
        }
        for i in range(12)
    ]


@pytest.fixture
def no_profile(without_triage_profile: None) -> Any:  # noqa: ANN401
    """No profile active before the test, and none after it."""
    yield
    deactivate_profile()


def test_a_pilot_table_is_fitted_into_the_profile(no_profile: None, tmp_path: Path) -> None:
    """The fitted values land in fitted.yaml with their counts, and the profile then reads them."""
    root = _profile(tmp_path / "lab")
    table = _table(tmp_path / "rv.parquet", _rows())
    assert cli.main([str(table), "--profile", str(root)]) == 0
    written = yaml.safe_load((root / "fitted.yaml").read_text())
    assert written["fitted_on"]["rows"] == 12
    fits = {entry["value"]: entry for entry in written["fits"]}
    assert fits["cohort_stage.yaml condition.hot_rms_dbfs_min"]["n"] == 4, "one per participant"
    assert fits["discard_replaced_task.yaml short_response.family_median_s"]["fitted"] == {"chat": 25.5}
    deactivate_profile()
    values = load_profile(root).fitted
    assert (
        values["task_extent.yaml"]["count_in"]["gap_max_s"]
        == written["fits"][[entry["value"] for entry in written["fits"]].index("task_extent.yaml count_in.gap_max_s")][
            "fitted"
        ]
    )
    assert "ddk_rate_hz" not in values["task_extent.yaml"]["atypical"], "no syllable task: left unfitted"


def test_a_value_the_profile_sets_itself_is_named_as_winning(
    no_profile: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dataset's count-in gap: fitted at the percentile, kept at the profile's own value."""
    root = _profile(tmp_path / "lab", {"task_extent.yaml": "count_in:\n  gap_max_s: 3.0\n"})
    assert cli.main([str(_table(tmp_path / "rv.parquet", _rows())), "--profile", str(root)]) == 0
    line = next(line for line in capsys.readouterr().out.splitlines() if "count_in.gap_max_s" in line)
    assert "the profile sets 3.0 itself, which wins" in line


def test_a_fit_is_refused_over_an_earlier_one_and_over_an_older_table(no_profile: None, tmp_path: Path) -> None:
    """A second fit needs --force; a table without the fit columns is rebuilt first."""
    root = _profile(tmp_path / "lab")
    table = _table(tmp_path / "rv.parquet", _rows())
    assert cli.main([str(table), "--profile", str(root)]) == 0
    deactivate_profile()
    assert cli.main([str(table), "--profile", str(root)]) == 2
    deactivate_profile()
    assert cli.main([str(table), "--profile", str(root), "--force"]) == 0
    deactivate_profile()
    old = _table(tmp_path / "old.parquet", _rows(), ["participant", "session", "declared_family"])
    assert cli.main([str(old), "--profile", str(root), "--out", str(tmp_path / "other.yaml")]) == 2


def test_a_blank_the_profile_writes_is_named_as_hiding_the_fit(
    no_profile: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Review of step 10: an explicit blank in the profile's own file wins over the fit, and the driver says so."""
    root = _profile(tmp_path / "lab", {"task_extent.yaml": "count_in:\n  gap_max_s:\n"})
    assert cli.main([str(_table(tmp_path / "rv.parquet", _rows())), "--profile", str(root)]) == 0
    line = next(line for line in capsys.readouterr().out.splitlines() if "count_in.gap_max_s" in line)
    assert "hides the fit" in line


def test_a_fit_written_elsewhere_is_named_as_not_read(
    no_profile: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--out`` somewhere else writes a file the profile does not load, and the driver says so."""
    root = _profile(tmp_path / "lab")
    out = tmp_path / "elsewhere.yaml"
    assert cli.main([str(_table(tmp_path / "rv.parquet", _rows())), "--profile", str(root), "--out", str(out)]) == 0
    assert out.is_file() and not (root / "fitted.yaml").exists()
    assert "the profile reads none of these" in capsys.readouterr().out


def test_an_unreadable_table_is_refused_not_raised(no_profile: None, tmp_path: Path) -> None:
    """Review of step 10: a missing table, or one that is not parquet, exits 2."""
    root = _profile(tmp_path / "lab")
    assert cli.main([str(tmp_path / "absent.parquet"), "--profile", str(root)]) == 2
    deactivate_profile()
    junk = tmp_path / "junk.parquet"
    junk.write_text("not a table")
    assert cli.main([str(junk), "--profile", str(root)]) == 2


def test_a_broken_fit_is_replaced_with_force_and_kept_aside(no_profile: None, tmp_path: Path) -> None:
    """Review of step 10: --force refits over a fitted.yaml the profile cannot load, keeping it as .previous."""
    root = _profile(tmp_path / "lab")
    (root / "fitted.yaml").write_text("values: [not, a, mapping\n")
    assert cli.main([str(_table(tmp_path / "rv.parquet", _rows())), "--profile", str(root), "--force"]) == 0
    assert (root / "fitted.yaml.previous").read_text().startswith("values: [not")
    deactivate_profile()
    assert load_profile(root).fitted["task_extent.yaml"]["count_in"]["gap_max_s"] is not None


def test_any_failure_of_a_refit_puts_the_earlier_fit_back(
    no_profile: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review K9: the earlier fitted.yaml is restored whatever stops the refit, an interrupt or an unexpected error."""
    root = _profile(tmp_path / "lab")
    table = _table(tmp_path / "rv.parquet", _rows())
    assert cli.main([str(table), "--profile", str(root)]) == 0
    earlier = (root / "fitted.yaml").read_bytes()
    for stop in (KeyboardInterrupt, TypeError, RuntimeError):

        def _stops(args: object, stop: type[BaseException] = stop) -> None:
            raise stop("stopped")

        monkeypatch.setattr(cli, "_fit", _stops)
        if stop is RuntimeError:
            assert cli.main([str(table), "--profile", str(root), "--force"]) == 2
        else:
            with pytest.raises(stop):
                cli.main([str(table), "--profile", str(root), "--force"])
        assert (root / "fitted.yaml").read_bytes() == earlier, stop.__name__
        assert not (root / "fitted.yaml.previous").exists()
        deactivate_profile()
