"""A dataset profile: what it may supply, where a data path resolves, and what a run needs from it."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from senselab.audio.workflows.triage import task_extent
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.dataset_profile import (
    DATASET_PATHS,
    PACKAGED_DATA,
    DatasetDataMissing,
    activate_profile,
    active_profile,
    data_path,
    deactivate_profile,
    load_profile,
    missing_dataset_values,
    require_dataset_values,
)
from tests.conftest import B2AI_TRIAGE_PROFILE


def _profile(root: Path, manifest: dict, files: dict[str, str] | None = None) -> Path:
    """Write a profile directory: its manifest and the files it names."""
    root.mkdir(parents=True, exist_ok=True)
    for relative, text in (files or {}).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (root / "profile.yaml").write_text(yaml.safe_dump(manifest))
    return root


class TestSenselabShipsNoDatasetData:
    """The packaged data directory holds engine and defaulted files only."""

    def test_no_dataset_path_is_packaged(self) -> None:
        """Every dataset path lives in a profile, never in the package."""
        assert not [path for path in DATASET_PATHS if (PACKAGED_DATA / path).exists()]

    def test_without_a_profile_a_dataset_path_raises(self, without_triage_profile: None) -> None:
        """A loader asking for dataset data with no profile gets an error naming the path."""
        with pytest.raises(DatasetDataMissing, match="cohort_conditions"):
            data_path("cohort_conditions/anything.yaml")

    def test_without_a_profile_a_defaulted_file_is_packaged(self, without_triage_profile: None) -> None:
        """A defaulted file falls back to senselab's copy."""
        assert data_path("task_extent.yaml") == PACKAGED_DATA / "task_extent.yaml"

    def test_without_a_profile_a_run_is_refused_naming_what_is_missing(self, without_triage_profile: None) -> None:
        """Every missing dataset value is listed at once, before any node runs."""
        config = load_triage_config()
        missing = missing_dataset_values(config.values)
        assert "time_release.yaml" in missing
        assert any("voice.glide" in item for item in missing)
        assert any("task_extent.yaml atypical.ddk_rate_hz" == item for item in missing)
        with pytest.raises(DatasetDataMissing, match="no dataset profile"):
            require_dataset_values(config.values)

    def test_the_b2ai_profile_supplies_everything_a_run_needs(self) -> None:
        """The suite's own profile leaves nothing missing."""
        assert missing_dataset_values(load_triage_config().values) == []


class TestLoading:
    """The manifest may list only what a profile can supply."""

    def test_a_directory_and_a_file_are_supplied(self, without_triage_profile: None, tmp_path: Path) -> None:
        """A supplied directory resolves inside the profile; so does a supplied file."""
        root = _profile(
            tmp_path / "p",
            {"name": "demo", "version": 1, "files": {"task_extent.yaml": "te.yaml", "cohort_conditions": "cc"}},
            {"te.yaml": "a: 1\n", "cc/demo.yaml": "conditions: {}\n"},
        )
        profile = load_profile(root)
        assert (profile.name, profile.version) == ("demo", "1")
        assert profile.resolve("task_extent.yaml") == root / "te.yaml"
        assert profile.resolve("cohort_conditions/demo.yaml") == root / "cc/demo.yaml"
        assert profile.resolve("task_speech.yaml") is None

    def test_the_digest_follows_every_supplied_byte(self, without_triage_profile: None, tmp_path: Path) -> None:
        """Changing a supplied file changes the digest; loading twice does not."""
        root = _profile(
            tmp_path / "p", {"name": "d", "version": "1", "files": {"task_lexicon": "lex"}}, {"lex/x.yaml": "a\n"}
        )
        first = load_profile(root).digest
        assert load_profile(root).digest == first
        (root / "lex/x.yaml").write_text("b\n")
        assert load_profile(root).digest != first

    @pytest.mark.parametrize(
        "files",
        [
            {"triage_table.yaml": "t.yaml"},  # an engine file
            {"config/default.yaml": "t.yaml"},  # the packaged config: a profile overrides it through `config`
            {"task_lexicon/cinderella-story.yaml": "t.yaml"},  # a file inside a dataset directory
            {"task_extnet.yaml": "t.yaml"},  # a typo
            {".": "t.yaml"},
            {"task_extent.yaml": "../t.yaml"},
            {"task_extent.yaml": "missing.yaml"},
            {"cohort_conditions": "t.yaml"},  # a file where a directory belongs
        ],
    )
    def test_a_path_a_profile_cannot_supply_is_refused(
        self, without_triage_profile: None, tmp_path: Path, files: dict
    ) -> None:
        """Engine files, files inside directories, typos, escapes, missing targets and the wrong kind."""
        root = _profile(tmp_path / "p", {"name": "d", "version": "1", "files": files}, {"t.yaml": "a: 1\n"})
        with pytest.raises(ValueError):
            load_profile(root)

    @pytest.mark.parametrize(
        "manifest", [{"version": "1"}, {"name": "d"}, {"name": "d", "version": "1", "config": "x.yaml"}]
    )
    def test_a_malformed_manifest_is_refused(
        self, without_triage_profile: None, tmp_path: Path, manifest: dict
    ) -> None:
        """No name, no version, or a config that does not exist."""
        with pytest.raises(ValueError):
            load_profile(_profile(tmp_path / "p", manifest))


class TestActivation:
    """What a loader reads follows the active profile, and only one is active at a time."""

    def test_a_loader_reads_the_profiles_copy_and_forgets_it_after(
        self, without_triage_profile: None, tmp_path: Path
    ) -> None:
        """The per-profile cache serves each profile its own file."""
        packaged = task_extent.task_extent_parameters()
        root = _profile(
            tmp_path / "p",
            {"name": "d", "version": "1", "files": {"task_extent.yaml": "te.yaml"}},
            {"te.yaml": "marker: 7\n"},
        )
        activate_profile(load_profile(root))
        assert task_extent.task_extent_parameters() == {"marker": 7}
        deactivate_profile()
        assert task_extent.task_extent_parameters() == packaged

    def test_activating_a_different_profile_raises(self, without_triage_profile: None, tmp_path: Path) -> None:
        """A second profile is refused until the first is deactivated."""
        activate_profile(load_profile(_profile(tmp_path / "a", {"name": "a", "version": "1"})))
        activate_profile(load_profile(_profile(tmp_path / "a", {"name": "a", "version": "1"})))  # the same: allowed
        with pytest.raises(RuntimeError, match="already active"):
            activate_profile(load_profile(_profile(tmp_path / "b", {"name": "b", "version": "1"})))

    def test_a_config_loaded_without_a_profile_names_the_active_one(self) -> None:
        """The configuration always names the profile its data files are read from."""
        profile = active_profile()
        assert profile is not None and profile.root == B2AI_TRIAGE_PROFILE.resolve()
        assert load_triage_config().profile == profile

    def test_a_failed_load_leaves_no_profile_active(self, without_triage_profile: None, tmp_path: Path) -> None:
        """The profile is activated only once its merged configuration has validated."""
        root = _profile(
            tmp_path / "p", {"name": "d", "version": "1", "config": "c.yaml"}, {"c.yaml": "spans:\n  k_bd: 1\n"}
        )
        with pytest.raises(ValueError, match="k_bd"):
            load_triage_config(profile=root)
        assert active_profile() is None
