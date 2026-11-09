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
    PACKAGED_PROFILES,
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
from tests.conftest import FIXTURE_PROFILE


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
        with pytest.raises(DatasetDataMissing, match="recruited_conditions"):
            data_path("recruited_conditions/anything.yaml")

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

    def test_the_fixture_profile_supplies_everything_a_run_needs(self) -> None:
        """The suite's own profile leaves nothing missing."""
        assert missing_dataset_values(load_triage_config().values) == []


class TestLoading:
    """The manifest may list only what a profile can supply."""

    def test_a_directory_and_a_file_are_supplied(self, without_triage_profile: None, tmp_path: Path) -> None:
        """A supplied directory resolves inside the profile; so does a supplied file."""
        root = _profile(
            tmp_path / "p",
            {"name": "demo", "version": 1, "files": {"task_extent.yaml": "te.yaml", "recruited_conditions": "cc"}},
            {"te.yaml": "a: 1\n", "cc/demo.yaml": "conditions: {}\n"},
        )
        profile = load_profile(root)
        assert (profile.name, profile.version) == ("demo", "1")
        assert profile.resolve("task_extent.yaml") == root / "te.yaml"
        assert profile.resolve("recruited_conditions/demo.yaml") == root / "cc/demo.yaml"
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
            {"recruited_conditions": "t.yaml"},  # a file where a directory belongs
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
        assert profile is not None and profile.root == FIXTURE_PROFILE.resolve()
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
        """No key or value of a packaged data file, nor of the reference profile senselab ships, is a fixture task."""
        tasks = set(yaml.safe_load((FIXTURE_PROFILE / "tasks.yaml").read_text())["tasks"])

        def strings(node: Any) -> list[str]:  # noqa: ANN401 -- a YAML value
            if isinstance(node, dict):
                return [str(key) for key in node] + [text for value in node.values() for text in strings(value)]
            if isinstance(node, list):
                return [text for value in node for text in strings(value)]
            return [node] if isinstance(node, str) else []

        reference = PACKAGED_PROFILES / "adult_clinical_speech"
        named = {
            str(path.relative_to(root.parent)): sorted(set(strings(yaml.safe_load(path.read_text()))) & tasks)
            for root in (PACKAGED_DATA, reference)
            for path in sorted(root.rglob("*.yaml"))
        }
        assert {path: names for path, names in named.items() if names} == {}

    def test_the_reference_and_fixture_profiles_supply_every_required_value(self) -> None:
        """With the fixture profile (extending the reference profile) active, the start check finds nothing missing."""
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
            "review_policy.txt",
            "task_guidance.yaml",
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
        """The fixture's breathing tasks under a workflow without AIRWAY: the start check names each one."""
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
        override.write_text("verdict:\n  recruited_conditions: no_such_list\n")
        assert "recruited_conditions/no_such_list.yaml (config verdict.recruited_conditions names it)" in (
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
        shutil.copytree(FIXTURE_PROFILE, root)
        config = load_triage_config(profile=root)
        (root / "tasks.yaml").unlink()
        missing = missing_dataset_values(config.values)
        assert "tasks.yaml (the profile lists it, but the file is absent)" in missing


class TestAPilotAndItsFit:
    """Plan step 10: a pilot leaves its corpus statistics empty; fitted.yaml supplies them, under its own files."""

    def _reference(self) -> list[str]:
        config = load_triage_config(profile="senselab:adult_clinical_speech")
        missing = missing_dataset_values(config.values)
        deactivate_profile()
        return missing

    def test_a_pilot_is_not_asked_for_its_corpus_statistics_and_lists_them_unfitted(
        self, without_triage_profile: None, tmp_path: Path
    ) -> None:
        """The same profile, marked a pilot, is asked for every value but the kind-b ones, which it names."""
        from senselab.audio.workflows.triage.dataset_profile import FITTED_KIND, unfitted_values

        kind_b = {
            f"{relative} {'.'.join(entry['path'])}"
            for relative, entries in profile_values().items()
            for entry in entries
            if entry.get("kind") == FITTED_KIND and entry.get("required")
        }
        asked = set(self._reference())
        assert kind_b <= asked
        root = _profile(tmp_path / "p", {"name": "lab", "version": "0", "extends": "senselab:adult_clinical_speech"})
        manifest = yaml.safe_load((root / "profile.yaml").read_text())
        (root / "profile.yaml").write_text(yaml.safe_dump({**manifest, "pilot": True}))
        config = load_triage_config(profile=root)
        assert set(missing_dataset_values(config.values)) == asked - kind_b
        assert set(unfitted_values()) == kind_b
        deactivate_profile()

    def test_a_pilot_flag_that_is_not_true_or_false_is_refused(
        self, without_triage_profile: None, tmp_path: Path
    ) -> None:
        """``pilot: yes please`` is a typo, not a pilot."""
        root = _profile(tmp_path / "p", {"name": "lab", "version": "0", "pilot": "yes please"})
        with pytest.raises(ValueError, match="pilot is true or false"):
            load_profile(root)

    def test_fitted_values_sit_over_the_base_and_under_the_profiles_own_files(
        self, without_triage_profile: None, tmp_path: Path
    ) -> None:
        """A fitted count-in gap is read; a value the profile writes itself wins over its fitted one."""
        fitted = {
            "values": {"task_extent.yaml": {"count_in": {"gap_max_s": 2.9}, "atypical": {"cough_event_s_max": 1.5}}}
        }
        root = _profile(
            tmp_path / "p",
            {
                "name": "lab",
                "version": "0",
                "extends": "senselab:adult_clinical_speech",
                "files": {"task_extent.yaml": "task_extent.yaml"},
            },
            {"task_extent.yaml": "count_in:\n  gap_max_s: 3.0\n", "fitted.yaml": yaml.safe_dump(fitted)},
        )
        profile = load_profile(root)
        before = profile.digest
        activate_profile(profile)
        values = data_values("task_extent.yaml")
        assert values["count_in"]["gap_max_s"] == 3.0, "the profile's own value wins"
        assert values["atypical"]["cough_event_s_max"] == 1.5, "the fitted value fills what the profile leaves"
        deactivate_profile()
        (root / "fitted.yaml").write_text(
            yaml.safe_dump({"values": {"task_extent.yaml": {"atypical": {"cough_event_s_max": 1.6}}}})
        )
        assert load_profile(root).digest != before, "a refit changes the profile's digest"

    def test_a_fitted_family_the_profile_below_lacks_is_added(
        self, without_triage_profile: None, tmp_path: Path
    ) -> None:
        """Review of step 10: a per-family median a base profile does not list is a new family, not a typo."""
        base = _profile(
            tmp_path / "base",
            {"name": "base", "version": "0", "files": {"discard_replaced_task.yaml": "d.yaml"}},
            {"d.yaml": "short_response:\n  family_median_s:\n    story: 40.0\n"},
        )
        fitted = {"values": {"discard_replaced_task.yaml": {"short_response": {"family_median_s": {"chat": 20.0}}}}}
        child = _profile(
            tmp_path / "child",
            {"name": "child", "version": "0", "extends": str(base)},
            {"fitted.yaml": yaml.safe_dump(fitted)},
        )
        activate_profile(load_profile(child))
        medians = data_values("discard_replaced_task.yaml")["short_response"]["family_median_s"]
        assert medians == {"story": 40.0, "chat": 20.0}
        deactivate_profile()

    def test_a_fitted_file_naming_no_value_file_is_refused(self, without_triage_profile: None, tmp_path: Path) -> None:
        """A fit writes only senselab's value files, never the task list."""
        root = _profile(
            tmp_path / "p",
            {"name": "lab", "version": "0"},
            {"fitted.yaml": yaml.safe_dump({"values": {"tasks.yaml": {}}})},
        )
        with pytest.raises(ValueError, match="not a value file a fit writes"):
            load_profile(root)


class TestMergingAcrossLayers:
    """How a value file's layers combine: the profile's own over its fit, directories, empty mappings (review H)."""

    def test_a_profiles_own_family_mapping_wins_over_its_fit(
        self, without_triage_profile: None, tmp_path: Path
    ) -> None:
        """A family the profile's own medians leave out is not brought back by its fit (H1)."""
        fitted = {"discard_replaced_task.yaml": {"short_response": {"family_median_s": {"story": 40.0, "chat": 20.0}}}}
        root = _profile(
            tmp_path / "p",
            {"name": "p", "version": "0", "files": {"discard_replaced_task.yaml": "d.yaml"}},
            {
                "d.yaml": "short_response:\n  family_median_s:\n    story: 45.0\n",
                "fitted.yaml": yaml.safe_dump({"values": fitted}),
            },
        )
        activate_profile(load_profile(root))
        try:
            assert data_values("discard_replaced_task.yaml")["short_response"]["family_median_s"] == {"story": 45.0}
        finally:
            deactivate_profile()

    def test_two_layers_may_each_supply_a_value_directory(self, without_triage_profile: None, tmp_path: Path) -> None:
        """A dated file only the base's directory holds is the base's; the newest across the layers is found (H2)."""
        from senselab.audio.workflows.triage.dataset_profile import newest_member

        base = _profile(
            tmp_path / "base",
            {"name": "base", "version": "0", "files": {"disagreement_profile": "dp"}},
            {"dp/2099-01-01.yaml": "schema_version: '1'\nnear_threshold_band: 0.4\n"},
        )
        child = _profile(
            tmp_path / "child",
            {"name": "child", "version": "0", "extends": str(base), "files": {"disagreement_profile": "dp"}},
            {"dp/2098-01-01.yaml": "schema_version: '1'\nnear_threshold_band: 0.3\n"},
        )
        activate_profile(load_profile(child))
        try:
            assert newest_member("disagreement_profile") == "disagreement_profile/2099-01-01.yaml"
            assert data_values("disagreement_profile/2099-01-01.yaml")["near_threshold_band"] == 0.4
            assert data_values("disagreement_profile/2098-01-01.yaml")["near_threshold_band"] == 0.3
            assert data_path("disagreement_profile/2099-01-01.yaml") == base.resolve() / "dp" / "2099-01-01.yaml"
        finally:
            deactivate_profile()

    def test_a_mapping_senselab_leaves_empty_takes_keys_from_every_layer(
        self, without_triage_profile: None, tmp_path: Path
    ) -> None:
        """The base fills senselab's empty per-task floors; the extending profile and the run each add one (H3)."""
        from senselab.audio.workflows.triage.dataset_profile import set_run_data

        base = _profile(
            tmp_path / "base",
            {"name": "base", "version": "0", "files": {"task_minimum_duration.yaml": "m.yaml"}},
            {"m.yaml": "families:\n  cough: 0.5\n"},
        )
        child = _profile(
            tmp_path / "child",
            {"name": "child", "version": "0", "extends": str(base), "files": {"task_minimum_duration.yaml": "m.yaml"}},
            {"m.yaml": "families:\n  sniff: 0.3\n"},
        )
        activate_profile(load_profile(child))
        try:
            set_run_data({"task_minimum_duration.yaml": {"families": {"hum": 0.2}}})
            assert data_values("task_minimum_duration.yaml")["families"] == {"cough": 0.5, "sniff": 0.3, "hum": 0.2}
            set_run_data({"task_minimum_duration.yaml": {"familes": {"hum": 0.2}}})
            with pytest.raises(ValueError, match="familes"):
                data_values("task_minimum_duration.yaml")
        finally:
            set_run_data(None)
            deactivate_profile()


class TestTheStartCheckReadsWhatARunWouldFailOn:
    """The start check refuses, by name, what would otherwise fail only once a recording reached it (review I)."""

    def _fixture_copy(self, tmp_path: Path) -> Path:
        root = tmp_path / "p"
        shutil.copytree(FIXTURE_PROFILE, root)
        return root

    def _missing(self, root: Path, override: Path | None = None) -> list[str]:
        try:
            return missing_dataset_values(load_triage_config(override, profile=root).values)
        finally:
            deactivate_profile()

    def test_the_fixture_and_the_reference_pass(self, without_triage_profile: None, tmp_path: Path) -> None:
        """The strengthened check leaves the fixture's profile, and a lab corpus written from the template, whole."""
        assert self._missing(FIXTURE_PROFILE) == []
        assert self._missing(Path(__file__).parent / "profiles" / "lab_corpus") == []

    def test_a_word_list_file_missing_a_key_is_refused(self, without_triage_profile: None, tmp_path: Path) -> None:
        """A redaction word list without months, or with a typo, is named before REDACT or VERDICT reads it (I1)."""
        root = self._fixture_copy(tmp_path)
        source = Path(load_profile("senselab:adult_clinical_speech").files["redaction_policy.yaml"])
        policy = yaml.safe_load(source.read_text())
        policy["month"] = policy.pop("months")
        (root / "redaction_policy.yaml").write_text(yaml.safe_dump(policy))
        manifest = yaml.safe_load((root / "profile.yaml").read_text())
        manifest["files"]["redaction_policy.yaml"] = "redaction_policy.yaml"
        (root / "profile.yaml").write_text(yaml.safe_dump(manifest))
        problems = [item for item in self._missing(root) if item.startswith("redaction_policy.yaml")]
        assert problems == [
            "redaction_policy.yaml (it lacks the keys ['months'])",
            "redaction_policy.yaml (no rule reads its keys ['month'])",
        ]

    def test_a_yaml_template_left_unwritten_is_refused(self, without_triage_profile: None, tmp_path: Path) -> None:
        """Each policy template carries the marker, as review_policy.txt does (I2)."""
        from senselab.audio.workflows.triage.dataset_profile import is_template
        from senselab.audio.workflows.triage.profile_template import TEMPLATE_DIR

        for name in ("review_policy.txt", "time_release.yaml", "airway_event_tokens.yaml", "task_guidance.yaml"):
            assert is_template(TEMPLATE_DIR / name), name
        root = self._fixture_copy(tmp_path)
        shutil.copy(TEMPLATE_DIR / "time_release.yaml", root / "time_release.yaml")
        assert "time_release.yaml (still the template: write your own)" in self._missing(root)

    def test_a_task_list_that_fails_only_when_read_is_refused(
        self, without_triage_profile: None, tmp_path: Path
    ) -> None:
        """A missing or shared lexicon, a missing category list and a fixed category the task lacks (I4)."""
        root = self._fixture_copy(tmp_path)
        tasks = yaml.safe_load((root / "tasks.yaml").read_text())
        story = next(name for name, entry in tasks["tasks"].items() if entry.get("lexicon"))
        other = next(name for name in tasks["tasks"] if name != story)
        tasks["tasks"][other]["lexicon"] = tasks["tasks"][story]["lexicon"]
        animals = next(
            name for name, entry in tasks["tasks"].items() if (entry.get("expects") or {}).get("item_category")
        )
        tasks["tasks"][animals]["categories"] = {
            "Fruits": {"rule": "list", "words": "task_lexicon/categories/none.yaml"}
        }
        (root / "tasks.yaml").write_text(yaml.safe_dump(tasks))
        missing = self._missing(root)
        assert any(
            item.startswith(f"tasks.yaml task {other} lexicon:") and "declares family" in item for item in missing
        )
        assert any("task_lexicon/categories/none.yaml (the file is absent)" in item for item in missing)
        assert f"tasks.yaml task {animals}: expects.item_category 'Animals' is not one of its categories" in missing

    def test_settings_keyed_by_a_task_or_kind_triage_lacks_are_refused(
        self, without_triage_profile: None, tmp_path: Path
    ) -> None:
        """A per-task gate naming no task or no gate, and a task-content kind triage does not know (I5)."""
        root = self._fixture_copy(tmp_path)
        override = tmp_path / "run.yaml"
        override.write_text(
            yaml.safe_dump(
                {
                    "verdict": {"gates": {"by_family": {"no-such-task": {"content_omission_fraction_max": 0.1}}}},
                    "data": {"task_content.yaml": {"any_word_families": ["no_such_kind"]}},
                }
            )
        )
        missing = self._missing(root, override)
        assert "config verdict.gates.by_family names tasks the task list lacks: ['no-such-task']" in missing
        assert any(item.startswith("task_content.yaml any_word_families names kinds") for item in missing)
        override.write_text(yaml.safe_dump({"verdict": {"gates": {"by_family": {"rainbow-passage": {"no_gate": 1}}}}}))
        assert "config verdict.gates.by_family.rainbow-passage names gates triage does not have: ['no_gate']" in (
            self._missing(root, override)
        )

    def test_a_value_file_that_does_not_merge_is_refused(self, without_triage_profile: None, tmp_path: Path) -> None:
        """A run's data section, or a profile's own file, with a key senselab's file lacks (I6)."""
        root = self._fixture_copy(tmp_path)
        override = tmp_path / "run.yaml"
        override.write_text(yaml.safe_dump({"data": {"breath_pattern.yaml": {"trian": {"distance_s": 0.5}}}}))
        assert any(
            item.startswith("breath_pattern.yaml (") and "trian" in item for item in self._missing(root, override)
        )
        (root / "task_speech.yaml").write_text((root / "task_speech.yaml").read_text() + "no_such_key: 1\n")
        assert any(item.startswith("task_speech.yaml (") for item in self._missing(root))


def test_a_run_records_the_files_beside_the_profiles_data(without_triage_profile: None, tmp_path: Path) -> None:
    """A hint builder beside the data is digested on its own; the profile's digest leaves it out (review O4)."""
    from senselab.audio.workflows.triage.dataset_profile import unlisted_files

    root = _profile(tmp_path / "p", {"name": "p", "version": "0"}, {"hints.py": "def build_hint(wav): ...\n"})
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "hints.cpython-312.pyc").write_bytes(b"\0")
    before = load_profile(root)
    recorded = unlisted_files(before)
    assert recorded["files"] == ["p/hints.py"]
    (root / "hints.py").write_text("def build_hint(wav): return None\n")
    after = load_profile(root)
    assert after.digest == before.digest and unlisted_files(after)["digest"] != recorded["digest"]


def test_the_per_task_response_medians_are_asked_for() -> None:
    """Plan step 10c: a new dataset's template asks for its free-response medians, a pilot's fit fills them (N2)."""
    [entry] = [
        e for e in profile_values()["discard_replaced_task.yaml"] if e["path"] == ["short_response", "family_median_s"]
    ]
    assert entry["required"] and entry["kind"] == "b"
