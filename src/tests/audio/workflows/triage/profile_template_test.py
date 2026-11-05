"""A new dataset's profile template: every value it must supply blank, beside what the value means (plan step 5d)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.dataset_profile import (
    REQUIRED_DATASET_PATHS,
    load_profile,
    missing_dataset_values,
    profile_values,
    unfitted_values,
)
from senselab.audio.workflows.triage.profile_template import REFERENCE_PROFILE, values_to_supply, write_template
from tests.conftest import B2AI_TRIAGE_PROFILE


@pytest.mark.usefixtures("without_triage_profile")
class TestTheTemplate:
    """What a template asks a dataset for, and that a run names whatever it leaves blank."""

    def test_extending_the_reference_profile_asks_only_for_a_datasets_own_values(self) -> None:
        """Policy, the dataset's own statistics and per-task values; never a detector setting."""
        asked = values_to_supply()
        kinds = {entry["kind"] for entries in asked.values() for entry in entries}
        assert kinds <= {"a", "b", "d"}
        assert sum(len(entries) for entries in asked.values()) == 39

    def test_without_a_base_every_required_value_is_asked_for(self) -> None:
        """A dataset that builds on no profile supplies every value senselab leaves to one."""
        required = sum(1 for entries in profile_values().values() for entry in entries if entry["required"])
        assert sum(len(entries) for entries in values_to_supply(None).values()) == required

    def test_the_template_loads_and_a_run_names_what_is_blank(self, tmp_path: Path) -> None:
        """It is a profile that extends the reference one; until it is filled in, the start check lists the blanks."""
        root = write_template(tmp_path / "new", name="new-dataset")
        profile = load_profile(root)
        assert profile.name == "new-dataset" and profile.base is not None
        manifest = yaml.safe_load((root / "profile.yaml").read_text())
        assert manifest["extends"] == REFERENCE_PROFILE
        files = set(manifest["files"])
        assert {
            "tasks.yaml",
            "airway_event_tokens.yaml",
            "time_release.yaml",
            "review_policy.txt",
            "task_guidance.yaml",
        } <= files
        inherited = {"discard_contested.yaml", "redaction_policy.yaml"}
        assert not inherited & files, "the reference profile supplies them, so they are inherited"
        assert set(REQUIRED_DATASET_PATHS) <= files | inherited
        missing = missing_dataset_values(load_triage_config(profile=root).values)
        assert "tasks.yaml tasks (the task list names no task)" in missing
        assert "review_policy.txt (still the template: write your own)" in missing
        values = [item for item in missing if item.split()[0] not in {"tasks.yaml", "review_policy.txt"}]
        unfitted = unfitted_values()
        assert len(values) + len(unfitted) == 39, "the dataset files' values too"
        assert manifest["pilot"] is True and len(unfitted) == 10, "a pilot is not asked for its corpus statistics"

    def test_each_blank_carries_its_kind(self, tmp_path: Path) -> None:
        """A dataset reads, beside each blank, how it is set."""
        root = write_template(tmp_path / "new", name="new-dataset")
        text = (root / "cohort_stage.yaml").read_text()
        assert (
            "[compute from your own recordings: left commented out; run a pilot, then scripts/fit_profile.py fits it]"
            in text
        )
        assert "clipped_fraction_min:" in text

    def test_a_fit_fills_what_the_template_left_for_it(self, tmp_path: Path) -> None:
        """Review of step 10: the template leaves the corpus statistics commented out, so fitted.yaml supplies them."""
        from senselab.audio.workflows.triage.dataset_profile import data_values, deactivate_profile

        root = write_template(tmp_path / "new", name="new-dataset")
        assert "count_in" not in (yaml.safe_load((root / "task_extent.yaml").read_text()) or {})
        fitted = {"values": {"task_extent.yaml": {"count_in": {"gap_max_s": 2.4}}}}
        (root / "fitted.yaml").write_text(yaml.safe_dump(fitted))
        load_triage_config(profile=root)
        assert data_values("task_extent.yaml")["count_in"]["gap_max_s"] == 2.4
        assert "task_extent.yaml count_in.gap_max_s" not in unfitted_values()
        deactivate_profile()

    def test_an_existing_directory_is_refused(self, tmp_path: Path) -> None:
        """A template never writes over a profile."""
        (tmp_path / "taken").mkdir()
        with pytest.raises(FileExistsError):
            write_template(tmp_path / "taken", name="x")

    def test_starting_from_another_profile_copies_it_and_says_so(self, tmp_path: Path) -> None:
        """The copy is renamed and records which profile it came from."""
        root = write_template(tmp_path / "copy", name="my-adult-study", start_from=B2AI_TRIAGE_PROFILE)
        manifest = yaml.safe_load((root / "profile.yaml").read_text())
        assert manifest["name"] == "my-adult-study"
        assert manifest["provenance"]["copied_from_profile"] == "bridge2ai-voice-adult"
        assert (root / "tasks.yaml").is_file()

    def test_extending_a_complete_profile_asks_for_nothing_and_hides_none_of_its_files(self, tmp_path: Path) -> None:
        """No skeleton overrides a dataset file the base already supplies (review finding 2)."""
        root = write_template(tmp_path / "on-b2ai", name="b2ai-variant", extends=B2AI_TRIAGE_PROFILE)
        assert not (root / "tasks.yaml").exists()
        assert values_to_supply(B2AI_TRIAGE_PROFILE) == {}
        assert missing_dataset_values(load_triage_config(profile=root).values) == []

    def test_a_relative_base_is_recorded_so_the_profile_loads_from_anywhere(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A path given relative to the caller's directory is written absolute (review finding 3)."""
        base = write_template(tmp_path / "base", name="base")
        monkeypatch.chdir(tmp_path)
        root = write_template(tmp_path / "sub" / "child", name="child", extends="base")
        assert Path(yaml.safe_load((root / "profile.yaml").read_text())["extends"]).is_absolute()
        monkeypatch.chdir(tmp_path / "sub")
        assert load_profile(root).base is not None and load_profile(root).base.root == base.resolve()

    def test_a_bad_base_writes_nothing(self, tmp_path: Path) -> None:
        """An unknown base is refused before the directory is made (review finding 9)."""
        with pytest.raises(ValueError, match="ships no profile"):
            write_template(tmp_path / "new", name="x", extends="senselab:children")
        assert not (tmp_path / "new").exists()

    def test_stages_left_off_are_not_asked_for(self) -> None:
        """A lab corpus without breathing or phonation tasks, cohorts or a quality join supplies fewer values."""
        off = ("AIRWAY", "VOICE", "COHORT", "QUALITY")
        asked = values_to_supply(off=off)
        stages = {stage for entries in asked.values() for entry in entries for stage in entry.get("stages") or ()}
        assert stages.isdisjoint(off)
        assert sum(len(entries) for entries in asked.values()) < 39

    def test_the_written_workflow_loads_and_the_start_check_asks_only_for_what_runs(self, tmp_path: Path) -> None:
        """The template's config lists the stages that run; the run is asked for no value of the others."""
        off = ("AIRWAY", "VOICE", "COHORT", "QUALITY")
        root = write_template(tmp_path / "lab", name="lab", off=off)
        values = load_triage_config(profile=root).values
        assert set(off).isdisjoint(values["workflow"]["stages"])
        missing = missing_dataset_values(values)
        asked = sum(1 for entries in values_to_supply(off=off).values() for entry in entries if entry["kind"] != "b")
        assert len([item for item in missing if item.split()[0] not in {"tasks.yaml", "review_policy.txt"}]) == asked

    def test_without_review_the_reviewers_files_are_neither_written_nor_asked_for(self, tmp_path: Path) -> None:
        """REVIEW and SECOND_OPINION off: no review policy or task guidance (plan step 6b, e)."""
        root = write_template(tmp_path / "lab", name="lab", off=("REVIEW", "SECOND_OPINION"))
        files = set(yaml.safe_load((root / "profile.yaml").read_text())["files"])
        assert not {"review_policy.txt", "task_guidance.yaml"} & files
        missing = missing_dataset_values(load_triage_config(profile=root).values)
        assert not [item for item in missing if item.split()[0] in {"review_policy.txt", "task_guidance.yaml"}]

    def test_a_copy_keeps_its_own_workflow(self, tmp_path: Path) -> None:
        """Copying a profile and leaving stages off at once is refused; the copy is edited instead."""
        with pytest.raises(ValueError, match="start_from"):
            write_template(tmp_path / "copy", name="copy", start_from=B2AI_TRIAGE_PROFILE, off=("AIRWAY",))
