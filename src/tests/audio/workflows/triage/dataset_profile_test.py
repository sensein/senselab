"""A dataset profile: what it may supply, where a data path resolves, and what a run needs from it."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from senselab.audio.workflows.triage import task_extent
from senselab.audio.workflows.triage.config import load_triage_config
from senselab.audio.workflows.triage.dataset_profile import (
    DATASET_PATHS,
    MERGED_PATHS,
    PACKAGED_DATA,
    DatasetDataMissing,
    activate_profile,
    active_profile,
    data_path,
    data_values,
    deactivate_profile,
    load_profile,
    missing_dataset_values,
    needed,
    profile_values,
    require_dataset_values,
    run_data,
)
from senselab.audio.workflows.triage.workflow import OPTIONAL_STAGES, STAGE_NAMES
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
        """The per-profile cache serves each profile its own values: its copy changes one, the rest are senselab's."""
        packaged = task_extent.task_extent_parameters()
        root = _profile(
            tmp_path / "p",
            {"name": "d", "version": "1", "files": {"task_extent.yaml": "te.yaml"}},
            {"te.yaml": "trim:\n  ddk:\n    kept_share_min: 0.9\n"},
        )
        activate_profile(load_profile(root))
        changed = task_extent.task_extent_parameters()
        assert changed["trim"]["ddk"]["kept_share_min"] == 0.9
        assert changed["trim"]["ddk"]["agreed_run_min"] == packaged["trim"]["ddk"]["agreed_run_min"]
        assert {key: value for key, value in changed.items() if key != "trim"} == {
            key: value for key, value in packaged.items() if key != "trim"
        }
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


class TestEveryValueIsOverridable:
    """Plan step 5b: a profile or a run changes any packaged value, one at a time; output formats and 16 kHz stay."""

    def _profile_with(self, tmp_path: Path, relative: str, text: str) -> Path:
        return _profile(
            tmp_path / "p", {"name": "d", "version": "1", "files": {relative: "copy.yaml"}}, {"copy.yaml": text}
        )

    def test_a_key_senselab_does_not_have_is_refused(self, without_triage_profile: None, tmp_path: Path) -> None:
        """A typo is caught, naming the file and the key."""
        activate_profile(load_profile(self._profile_with(tmp_path, "task_extent.yaml", "trim:\n  dkk: {}\n")))
        with pytest.raises(ValueError, match=r"task_extent\.yaml:trim\.dkk"):
            data_values("task_extent.yaml")

    def test_a_mapping_senselab_leaves_empty_takes_the_profiles_entries(
        self, without_triage_profile: None, tmp_path: Path
    ) -> None:
        """A per-task table senselab ships empty is the profile's to fill."""
        text = "short_response:\n  family_median_s: {story-retell: 40.0}\n"
        activate_profile(load_profile(self._profile_with(tmp_path, "discard_replaced_task.yaml", text)))
        assert data_values("discard_replaced_task.yaml")["short_response"]["family_median_s"] == {"story-retell": 40.0}

    @pytest.mark.parametrize(
        ("relative", "text", "message"),
        [
            ("decision_reasons.yaml", "precedence: [unreadable]\n", "precedence is fixed"),
            ("task_extent.yaml", "version: 99\n", "version is fixed"),
        ],
    )
    def test_an_output_vocabulary_or_a_version_cannot_change(
        self, without_triage_profile: None, tmp_path: Path, relative: str, text: str, message: str
    ) -> None:
        """The reason vocabulary and its order are senselab's output; a version restated must match."""
        activate_profile(load_profile(self._profile_with(tmp_path, relative, text)))
        with pytest.raises(ValueError, match=message):
            data_values(relative)

    def test_a_word_list_is_replaced_whole(self, without_triage_profile: None, tmp_path: Path) -> None:
        """Another language's temporal words replace senselab's, list by list."""
        activate_profile(load_profile(self._profile_with(tmp_path, "temporal_words.yaml", "units: [segundo]\n")))
        assert data_values("temporal_words.yaml") == {"units": ["segundo"]}

    def test_the_four_files_a_profile_could_not_replace_now_can(self) -> None:
        """voice_phonation, background_model, background_speech and task_squim are value files."""
        assert {"voice_phonation.yaml", "background_model.yaml", "background_speech.yaml", "task_squim.yaml"} <= (
            MERGED_PATHS
        )

    def test_a_run_changes_one_value_over_the_profile(self, tmp_path: Path) -> None:
        """The run's data section is applied after the profile's copy, and the config hash covers it."""
        before = load_triage_config()
        packaged = task_extent.task_extent_parameters()
        override = tmp_path / "run.yaml"
        override.write_text("data:\n  task_extent.yaml:\n    trim:\n      ddk:\n        kept_share_min: 0.75\n")
        config = load_triage_config(override)
        try:
            assert task_extent.task_extent_parameters()["trim"]["ddk"]["kept_share_min"] == 0.75
            assert config.config_hash != before.config_hash
        finally:
            load_triage_config()
        assert task_extent.task_extent_parameters() == packaged

    def test_a_run_may_not_name_a_fixed_file(self, tmp_path: Path) -> None:
        """The corpus-table dictionary is senselab's output format."""
        override = tmp_path / "run.yaml"
        override.write_text("data:\n  recording_vectors/dictionary.yaml: {}\n")
        with pytest.raises(ValueError, match="dictionary"):
            load_triage_config(override)
        load_triage_config()

    def test_the_working_rate_is_fixed(self, tmp_path: Path) -> None:
        """Every model triage runs is native at 16 kHz."""
        override = tmp_path / "run.yaml"
        override.write_text("resample:\n  target_hz: 22050\n")
        with pytest.raises(ValueError, match="fixed"):
            load_triage_config(override)


class TestSenselabShipsNoFittedValue:
    """Plan step 5c: every value fitted on a dataset or chosen by its owner is a profile's, and senselab ships none."""

    @staticmethod
    def _value(document: Any, path: list[str]) -> Any:  # noqa: ANN401 -- a YAML value
        for part in path:
            if not isinstance(document, dict) or part not in document:
                return None
            document = document[part]
        return document

    def test_every_profile_value_is_empty_in_senselab(self) -> None:
        """Senselab's packaged files and config leave each listed value empty, so no run inherits one unknowingly."""
        for relative, entries in profile_values().items():
            if not (PACKAGED_DATA / relative).is_file():
                continue  # a dataset file senselab does not package at all
            document = yaml.safe_load((PACKAGED_DATA / relative).read_text()) or {}
            shipped = [entry["path"] for entry in entries if self._value(document, entry["path"]) is not None]
            assert shipped == [], relative

    def test_senselab_data_names_no_dataset_task(self) -> None:
        """No key or value of a packaged data file is a B2AI task name (the vendored registry moves in step 7)."""
        tasks = set(yaml.safe_load((B2AI_TRIAGE_PROFILE / "tasks.yaml").read_text())["tasks"])

        def strings(node: Any) -> list[str]:  # noqa: ANN401 -- a YAML value
            if isinstance(node, dict):
                return [str(key) for key in node] + [text for value in node.values() for text in strings(value)]
            if isinstance(node, list):
                return [text for value in node for text in strings(value)]
            return [node] if isinstance(node, str) else []

        named = {
            str(path.relative_to(PACKAGED_DATA)): sorted(set(strings(yaml.safe_load(path.read_text()))) & tasks)
            for path in sorted(PACKAGED_DATA.rglob("*.yaml"))
            if "b2ai_task_registry" not in path.parts
        }
        assert {path: names for path, names in named.items() if names} == {}

    def test_the_reference_and_b2ai_profiles_supply_every_required_value(self) -> None:
        """With B2AI's profile (which extends the reference profile) active, the start check finds nothing missing."""
        assert missing_dataset_values(load_triage_config().values) == []

    def test_the_reference_profile_alone_leaves_only_a_datasets_own_values(self, without_triage_profile: None) -> None:
        """Extending the reference profile, a dataset supplies its task list, policy, statistics and per-task values."""
        config = load_triage_config(profile="senselab:adult_clinical_speech")
        missing = missing_dataset_values(config.values)
        kinds = {
            (relative, tuple(entry["path"])): entry["kind"]
            for relative, entries in profile_values().items()
            for entry in entries
        }
        values = [item for item in missing if " " in item]
        assert {item.split()[0] for item in missing if " " not in item} == {
            "airway_event_tokens.yaml",
            "tasks.yaml",
            "time_release.yaml",
        }
        for item in values:
            where, dotted = item.split(" ", 1)
            relative = "config/default.yaml" if where == "config" else where
            kind = next(k for (r, path), k in kinds.items() if r == relative and ".".join(path) == dotted)
            assert kind in {"a", "b", "d"}, item
        deactivate_profile()

    def test_a_profile_may_extend_a_shipped_one_and_override_it(
        self, without_triage_profile: None, tmp_path: Path
    ) -> None:
        """The extending profile's value wins over the reference's, and the digest covers the chain."""
        root = _profile(
            tmp_path / "p",
            {
                "name": "d",
                "version": "1",
                "extends": "senselab:adult_clinical_speech",
                "files": {"cough_pattern.yaml": "c.yaml"},
            },
            {"c.yaml": "measure:\n  rise_db: 9.0\n"},
        )
        profile = load_profile(root)
        assert profile.base is not None and profile.base.name == "adult-clinical-speech"
        activate_profile(profile)
        values = data_values("cough_pattern.yaml")["measure"]
        assert values["rise_db"] == 9.0
        assert values["peak_db"] is not None, "inherited from the reference profile"

    def test_a_profile_extending_itself_is_refused(self, without_triage_profile: None, tmp_path: Path) -> None:
        """A cycle in `extends` names the chain."""
        root = _profile(tmp_path / "loop", {"name": "loop", "version": "1", "extends": "."})
        with pytest.raises(ValueError, match="extends itself"):
            load_profile(root)

    def test_an_unknown_shipped_profile_is_refused(self, without_triage_profile: None, tmp_path: Path) -> None:
        """`senselab:<name>` must name a profile senselab ships."""
        root = _profile(tmp_path / "p", {"name": "d", "version": "1", "extends": "senselab:children"})
        with pytest.raises(ValueError, match="ships no profile"):
            load_profile(root)


class TestTheReviewFixes:
    """Step-5 review findings 1, 5, 6, 7 and 8."""

    def test_a_run_may_not_change_a_file_no_loader_merges(self, tmp_path: Path) -> None:
        """The task list is read whole; a run's change to it would be dropped while the hash claimed it."""
        override = tmp_path / "run.yaml"
        override.write_text("data:\n  tasks.yaml:\n    tasks: {}\n")
        with pytest.raises(ValueError, match="no loader merges"):
            load_triage_config(override)
        load_triage_config()

    def test_a_profile_config_may_not_carry_a_data_section(self, without_triage_profile: None, tmp_path: Path) -> None:
        """A profile changes data files by supplying them, so the chain's order holds."""
        root = _profile(
            tmp_path / "p",
            {"name": "d", "version": "1", "config": "config.yaml"},
            {"config.yaml": "data:\n  task_speech.yaml:\n    words_min: 1\n"},
        )
        with pytest.raises(ValueError, match="not through a `data:` section"):
            load_triage_config(profile=root)

    def test_a_refused_profile_switch_leaves_the_run_data_as_it_was(self, tmp_path: Path) -> None:
        """Activating another profile fails before the new run data is installed."""
        before = run_data()
        other = _profile(tmp_path / "other", {"name": "other", "version": "1"})
        override = tmp_path / "run.yaml"
        override.write_text("data:\n  task_extent.yaml:\n    trim:\n      ddk:\n        kept_share_min: 0.7\n")
        with pytest.raises(RuntimeError, match="already active"):
            load_triage_config(override, profile=other)
        assert run_data() == before

    def test_an_empty_task_list_is_refused(self, without_triage_profile: None, tmp_path: Path) -> None:
        """Every recording's task would be unknown (review finding 1)."""
        from senselab.audio.workflows.triage.profile_template import write_template  # noqa: PLC0415

        root = write_template(tmp_path / "new", name="new")
        assert "tasks.yaml tasks (the task list names no task)" in missing_dataset_values(
            load_triage_config(profile=root).values
        )

    def test_a_datasets_newer_disagreement_profile_is_read(self, without_triage_profile: None, tmp_path: Path) -> None:
        """The newest dated file across senselab and the chain wins, merged over senselab's of that date if any."""
        from senselab.audio.workflows.triage.routing_analysis.gate_matrix import (
            load_disagreement_profile,  # noqa: PLC0415
        )

        root = _profile(
            tmp_path / "p",
            {"name": "d", "version": "1", "files": {"disagreement_profile": "dp"}},
            {"dp/2099-01-01.yaml": "schema_version: '1'\nnear_threshold_band: 0.4\n"},
        )
        activate_profile(load_profile(root))
        load_disagreement_profile.cache_clear()
        assert load_disagreement_profile()["near_threshold_band"] == 0.4


class TestValuesFollowTheWorkflow:
    """A value only an off stage reads is not asked for (plan step 11d)."""

    def test_every_stage_a_value_names_may_be_off(self) -> None:
        """A value of a stage that always runs carries no ``stages``: it is always needed."""
        named = {
            stage for entries in profile_values().values() for entry in entries for stage in entry.get("stages") or ()
        }
        assert named and named <= OPTIONAL_STAGES

    def test_a_value_is_needed_only_when_one_of_its_stages_runs(self) -> None:
        """No stages: whatever runs; with stages: only where one of them does."""
        entry = {"path": ["x"], "required": True, "stages": ["AIRWAY", "VOICE"]}
        assert needed(entry, None) and needed(entry, {"VOICE"}) and not needed(entry, {"SPEECH"})
        assert needed({"path": ["x"], "required": True}, frozenset())
        assert not needed({"path": ["x"], "required": False}, None)

    def test_a_task_whose_branch_is_off_is_refused_by_name(self, tmp_path: Path) -> None:
        """B2AI's breathing tasks under a workflow without AIRWAY: the start check names each one."""
        override = tmp_path / "run.yaml"
        stages = [name for name in STAGE_NAMES if name != "AIRWAY"]
        override.write_text(yaml.safe_dump({"workflow": {"stages": stages}}))
        missing = missing_dataset_values(load_triage_config(override).values)
        refused = [item for item in missing if item.startswith("tasks.yaml task ")]
        assert refused and all("its branch AIRWAY is off in this workflow" in item for item in refused)
        assert not missing_dataset_values(load_triage_config().values)

    def test_a_named_cohort_list_the_profile_lacks_is_refused_before_the_run(self, tmp_path: Path) -> None:
        """Found by the step-11 speech-only run: VERDICT failed on every recording instead of the run refusing."""
        override = tmp_path / "run.yaml"
        override.write_text("verdict:\n  cohort_conditions: no_such_list\n")
        assert "cohort_conditions/no_such_list.yaml (config verdict.cohort_conditions names it)" in (
            missing_dataset_values(load_triage_config(override).values)
        )

    def test_a_value_every_run_reads_names_no_stage(self) -> None:
        """redaction.fill is read when the release is settled, after VERDICT, whatever runs (step-11 review 1)."""
        [fill] = [e for e in profile_values()["config/default.yaml"] if e["path"] == ["redaction", "fill"]]
        assert not fill.get("stages")

    def test_a_listed_task_list_that_is_absent_is_named_not_raised(
        self, without_triage_profile: None, tmp_path: Path
    ) -> None:
        """tasks.yaml goes after the profile loaded (loading refuses it sooner): the start check names it (review 6)."""
        root = tmp_path / "p"
        shutil.copytree(B2AI_TRIAGE_PROFILE, root)
        config = load_triage_config(profile=root)
        (root / "tasks.yaml").unlink()
        missing = missing_dataset_values(config.values)
        assert "tasks.yaml (the profile lists it, but the file is absent)" in missing
