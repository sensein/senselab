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
        assert set(REQUIRED_DATASET_PATHS) <= set(manifest["files"])
        missing = missing_dataset_values(load_triage_config(profile=root).values)
        assert len([item for item in missing if " " in item]) == 39

    def test_each_blank_carries_its_kind(self, tmp_path: Path) -> None:
        """A dataset reads, beside each blank, how it is set."""
        root = write_template(tmp_path / "new", name="new-dataset")
        text = (root / "cohort_stage.yaml").read_text()
        assert "[compute from your own recordings (a pilot run)]" in text
        assert "clipped_fraction_min:" in text

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
