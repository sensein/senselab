"""The triage configuration: every number, its derivation, and what happens when one is unset."""

from __future__ import annotations

import inspect
import re
from dataclasses import fields
from pathlib import Path

import pytest

import senselab.audio.tasks
from senselab.audio.tasks.phonation import derive_f0_range
from senselab.audio.workflows.triage import config as config_module
from senselab.audio.workflows.triage.config import DATA_MAP_PATHS, TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.common import (
    CPPS_SCALAR_KEYS,
    F0_RANGE_PARAMETER_NAMES,
    cpps_settings,
    f0_range_parameters,
)
from senselab.text.tasks.pii_detection.api import default_detectors

_MAX_VALUE_CHARS = 80
"""Longest a string value may be. A name, a format or a label fits; a sentence does not."""


def _leaves(node: object, path: str = "") -> dict[str, object]:
    """Flatten a nested mapping to dotted path -> scalar.

    Args:
        node: A mapping, sequence or scalar.
        path: The dotted path accumulated so far.

    Returns:
        Every leaf keyed by its dotted path. List elements are indexed.
    """
    if isinstance(node, dict):
        out: dict[str, object] = {}
        for key, value in node.items():
            out.update(_leaves(value, f"{path}.{key}" if path else str(key)))
        return out
    if isinstance(node, list):
        out = {}
        for index, value in enumerate(node):
            out.update(_leaves(value, f"{path}[{index}]"))
        return out
    return {path: node}


class TestMeasuredValues:
    """A value with a derivation is readable."""

    def test_the_measured_values_are_present(self) -> None:
        """Each measured value reads back exactly as the file states it."""
        cfg = load_triage_config()
        assert cfg.require("envelope.lowpass_hz") == 40.0
        assert cfg.require("spans.transition_window_ms") == 30
        assert cfg.require("spans.k_db") == 6.0
        assert cfg.require("preemphasis.coefficient") == 0.97
        assert cfg.require("floor.percentile") == 5.0
        assert cfg.require("phonation.periods_per_window") == 4.5

    def test_the_required_detectors_are_the_pii_modules_own_inventory(self) -> None:
        """``pii.required_detectors`` is a vocabulary read off the module, not a fitted subset.

        Drifting from ``default_detectors()`` in either direction is a defect: a detector the scan
        runs but the config does not require would stop being missed when it silently stops running,
        and one the config requires but the scan never runs would make every scan incomplete.
        """
        cfg = load_triage_config()
        assert cfg.require("pii.required_detectors") == default_detectors()

    def test_identity_travels_with_the_config(self) -> None:
        """The config carries its name, version and hash."""
        cfg = load_triage_config()
        assert cfg.name == "senselab-triage/default"
        assert isinstance(cfg.version, int)
        assert len(cfg.config_hash) == 16


class TestUnsetValues:
    """A number nobody measured must be impossible to use by accident."""

    def test_reading_an_unset_value_raises_and_names_it(self) -> None:
        """Requiring a null value raises, naming the parameter."""
        cfg = load_triage_config()
        with pytest.raises(ValueError, match="phonation.hnr_floor_interval_db"):
            cfg.require("phonation.hnr_floor_interval_db")

    def test_a_typo_is_an_unknown_key_not_an_unmeasured_value(self) -> None:
        """An absent key is a typo, and the error says so instead of citing open.md."""
        cfg = load_triage_config()
        with pytest.raises(ValueError, match="unknown configuration key"):
            cfg.require("phonation.hnr_flor_interval_db")

    def test_a_null_key_is_unmeasured_not_unknown(self) -> None:
        """A present-null key still points at open.md, never at the typo message."""
        cfg = load_triage_config()
        with pytest.raises(ValueError, match="benchmarks/open.md"):
            cfg.require("phonation.rms_floor_interval")

    def test_the_error_points_at_what_would_settle_it(self) -> None:
        """The error names the open-questions file."""
        cfg = load_triage_config()
        with pytest.raises(ValueError, match="benchmarks/open.md"):
            cfg.require("redaction.padding_ms")

    def test_every_unset_value_is_null_rather_than_absent(self) -> None:
        """Absent is a typo; null is a decision not yet taken."""
        cfg = load_triage_config()
        for path in (
            "phonation.hnr_floor_interval_db",
            "phonation.rms_floor_interval",
            "redaction.padding_ms",
            "speech.second_diarizer",
            "quality.stoi_floor",
            "taxonomy.speech_labels",
        ):
            node: object = cfg.values
            for part in path.split("."):
                assert isinstance(node, dict) and part in node, f"{path} must be present and null"
                node = node[part]
            assert node is None, f"{path} must be present and null"

    def test_get_returns_a_default_instead_of_raising(self) -> None:
        """A caller that can proceed without the value may ask politely."""
        cfg = load_triage_config()
        assert cfg.get("phonation.hnr_floor_interval_db", 8.0) == 8.0


class TestOverrides:
    """Whole-file overrides, and the hash follows the merged mapping."""

    def test_an_override_supplies_an_unset_value(self, tmp_path: Path) -> None:
        """An override can supply what nobody had measured."""
        override = tmp_path / "o.yaml"
        override.write_text("redaction:\n  padding_ms: 250\n")
        cfg = load_triage_config(override)
        assert cfg.require("redaction.padding_ms") == 250

    def test_an_override_changes_the_hash(self, tmp_path: Path) -> None:
        """Two different merged mappings never share a hash."""
        override = tmp_path / "o.yaml"
        override.write_text("spans:\n  k_db: 8.0\n")
        assert load_triage_config(override).config_hash != load_triage_config().config_hash

    def test_an_unknown_key_is_refused_rather_than_ignored(self, tmp_path: Path) -> None:
        """A typo in an override key is an error, not a no-op."""
        override = tmp_path / "o.yaml"
        override.write_text("spans:\n  k_bd: 8.0\n")
        with pytest.raises(ValueError, match="k_bd"):
            load_triage_config(override)


class TestOverridesMayExtendADataMap:
    """A schema key is a name the code reads; a data-map key is a value the data supplies."""

    def test_a_new_route_index_entry_is_accepted(self, tmp_path: Path) -> None:
        """A campaign numbering its breathing trials differently must not have to edit the package."""
        override = tmp_path / "o.yaml"
        override.write_text("airway:\n  route_by_task_index:\n    5: nose\n")
        cfg = load_triage_config(override)
        assert cfg.require("airway.route_by_task_index")[5] == "nose"

    def test_the_packaged_entries_of_a_non_null_data_map_survive_the_addition(self, tmp_path: Path) -> None:
        """An additive override that silently dropped a packaged entry would disable it."""
        override = tmp_path / "o.yaml"
        override.write_text("airway:\n  route_by_task_index:\n    5: nose\n")
        resolved = load_triage_config(override).require("airway.route_by_task_index")
        assert resolved == {1: "nose", 2: "mouth", 3: "nose", 4: "mouth", 5: "nose"}

    def test_an_existing_entry_is_replaced_not_merged(self, tmp_path: Path) -> None:
        """The value under a data-map key is data; the override replaces what it names."""
        override = tmp_path / "o.yaml"
        override.write_text("airway:\n  route_by_task_index:\n    1: mouth\n")
        resolved = load_triage_config(override).require("airway.route_by_task_index")
        assert resolved[1] == "mouth"

    def test_a_null_data_map_still_takes_a_whole_mapping(self, tmp_path: Path) -> None:
        """The control: the paths that ship null must keep accepting the mapping that fills them."""
        override = tmp_path / "o.yaml"
        override.write_text("routing:\n  hint_branch_map:\n    cough: AIRWAY\n")
        assert load_triage_config(override).require("routing.hint_branch_map") == {"cough": "AIRWAY"}

    def test_a_schema_key_is_still_refused(self, tmp_path: Path) -> None:
        """The whole point of the refusal: a section the code reads by name cannot grow a key."""
        override = tmp_path / "o.yaml"
        override.write_text("taxonomy:\n  nonsense: 1\n")
        with pytest.raises(ValueError, match="nonsense"):
            load_triage_config(override)

    def test_a_schema_key_inside_a_section_holding_a_data_map_is_still_refused(self, tmp_path: Path) -> None:
        """The exemption is the map, not the section it sits in."""
        override = tmp_path / "o.yaml"
        override.write_text("airway:\n  corroboratoin_overrides:\n    Sneeze: [Sneeze]\n")
        with pytest.raises(ValueError, match="corroboratoin_overrides"):
            load_triage_config(override)

    def test_every_declared_data_map_path_exists_in_the_packaged_file(self) -> None:
        """A path that has been renamed away would exempt nothing and refuse silently."""
        cfg = load_triage_config()
        for path in DATA_MAP_PATHS:
            node: object = cfg.values
            for part in path.split("."):
                assert isinstance(node, dict) and part in node, f"{path} is not in the packaged config"
                node = node[part]
            assert node is None or isinstance(node, dict), f"{path} is neither null nor a mapping"


_KEY_PATTERN = re.compile(r"`+([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+)`+")
_TASK_MODULES = ("envelope", "spans", "gammatone", "phonation", "redaction", "disruptions")


def _docstring_config_keys() -> dict[str, set[str]]:
    """Collect every ``section.key`` a task api docstring names, per module."""
    keys: dict[str, set[str]] = {}
    for name in _TASK_MODULES:
        source = (Path(senselab.audio.tasks.__file__).parent / name / "api.py").read_text()
        keys[name] = set(_KEY_PATTERN.findall(source))
    return keys


class TestDocstringKeysResolve:
    """Every config key a task docstring names must exist in the default configuration."""

    def test_each_module_names_at_least_one_key(self) -> None:
        """An empty extraction means the pattern broke, not that a module has no keys."""
        for name, keys in _docstring_config_keys().items():
            assert keys, f"no `section.key` references extracted from {name}/api.py"

    def test_every_docstring_key_resolves_in_the_default_config(self) -> None:
        """A key a docstring tells the caller to read must be present in default.yaml, even if null."""
        cfg = load_triage_config()
        for name, keys in _docstring_config_keys().items():
            for key in sorted(keys):
                node: object = cfg.values
                for part in key.split("."):
                    assert isinstance(node, dict) and part in node, (
                        f"{name}/api.py names `{key}` but it does not resolve in default.yaml"
                    )
                    node = node[part]


class TestTheV2OpenKeys:
    """Every key the v2 specs owe a measurement exists and is null."""

    OPEN_KEYS = (
        "windows.yamnet.label_thresholds",
        "windows.ast.default_threshold",
        "windows.ast.label_thresholds",
        "windows.hear.label_thresholds",
        "taxonomy.speech_labels",
        "routing.hint_branch_map",
        "speech.enrollment_model",
        "speech.separation_backend",
        "speech.separation_sound_class",
        "speech.nontarget.level_db",
        "speech.nontarget.tilt_db_per_octave",
        "speech.nontarget.d_to_r_db",
    )

    def test_every_open_key_exists_and_is_null(self) -> None:
        """A key that does not exist is a typo; a key with a value is an unmeasured decision shipped.

        Both halves are checked through the public API: ``require`` distinguishes the two failures by
        message — "unknown configuration key" for a typo, "has no value" for a null — so asserting on
        which message fires is what tells "the key is missing" from "the key is null".
        """
        config = load_triage_config()
        for path in self.OPEN_KEYS:
            with pytest.raises(ValueError, match="has no value") as raised:
                config.require(path)
            assert "unknown configuration key" not in str(raised.value), path
            assert config.get(path, "SENTINEL") == "SENTINEL", path

    def test_the_v1_keys_the_v2_specs_replaced_are_gone(self) -> None:
        """Pre-alpha: a replaced key is deleted, not left beside its replacement."""
        config = load_triage_config()
        for path in (
            "phonation.f0_min_hz",
            "phonation.f0_max_hz",
            "voice.f0_range_hz",
            "taxonomy.audioset_speech_labels",
            "taxonomy.audioset_airway_labels",
            "taxonomy.hear_airway_labels",
            "taxonomy.min_families",
            "taxonomy.ast_frame_s",
            "taxonomy.lexical_airway_tokens",
            "taxonomy.presence_floor.yamnet",
            "taxonomy.presence_floor.speech.lexical",
            "taxonomy.voice_min_duration_s",
            "taxonomy.voice_uncertain_duration_s",
            "routing.hint_kind_map",
            "hear.label_floor",
            "airway.contest_labels",
            "airway.corroboration_overrides",
        ):
            with pytest.raises(ValueError, match="unknown configuration key"):
                config.require(path)

    def test_the_window_hops_are_declared_defaults_not_open_keys(self) -> None:
        """A null hop is not the honest state here: it stopped the classifier running at all.

        ``require`` raises on a null, and both hops are read inside the *scores* block, so while they
        were null AST and HeAR never ran under the packaged config -- the expensive model output was
        lost along with the threshold fold V3 exists to let it survive. Both now ship non-overlapping,
        which is a declared choice the config_hash names.
        """
        config = load_triage_config()
        assert config.require("windows.ast.hop_s") == 10.24
        assert config.require("windows.ast.win_length_s") == 10.24
        assert config.require("windows.hear.hop_s") == 2.0
        with pytest.raises(ValueError, match="has no value"):
            config.require("windows.ast.default_threshold")

    def test_the_span_label_membership_pair_is_declared_for_both_span_classifiers(self) -> None:
        """One pair of values, so PREPROCESS and the routing analysis cannot drift apart."""
        config = load_triage_config()
        for classifier in ("yamnet", "hear"):
            assert config.require(f"windows.{classifier}.default_threshold") == 0.2
            assert config.require(f"windows.{classifier}.label_top_k") == 4

    def test_ast_hop_cannot_imply_finer_temporal_evidence_than_the_model_has(self, tmp_path: Path) -> None:
        """AST's 10.24-second context is summary evidence, so its hop has an explicit lower bound."""
        override = tmp_path / "ast-hop.yaml"
        override.write_text("windows:\n  ast:\n    hop_s: 7.99\n")
        with pytest.raises(ValueError, match=r"windows\.ast\.hop_s must be at least 8 s"):
            load_triage_config(override)

        override.write_text("windows:\n  ast:\n    hop_s: 8.0\n")
        assert load_triage_config(override).require("windows.ast.hop_s") == 8.0

    def test_the_wide_search_replaces_the_declared_range(self) -> None:
        """One wide search, read by PREPROCESS and VOICE alike, each narrowing it the same way."""
        config = load_triage_config()
        assert config.require("voice.f0_search_range_hz") == [50.0, 600.0]
        with pytest.raises(ValueError, match="unknown configuration key"):
            config.require("voice.f0_range_hz")

    def test_every_f0_range_parameter_comes_from_the_config(self) -> None:
        """``derive_f0_range`` has no defaults, so each of its parameters must resolve to a key."""
        resolved = f0_range_parameters(load_triage_config())
        assert set(resolved) == set(F0_RANGE_PARAMETER_NAMES)
        assert (resolved["search_floor_hz"], resolved["search_ceiling_hz"]) == (50.0, 600.0)
        assert resolved["pitch_floor_divisor"] == 1.5
        assert resolved["pitch_ceiling_quartile_multiplier"] == 2.5
        assert resolved["pitch_pinned_percentile"] == 95.0
        assert resolved["pitch_excursion_multiplier"] == 1.5
        assert resolved["pitch_pinned_octave_ratio"] == 2.0

    def test_a_narrowing_coefficient_override_reaches_the_helper(self, tmp_path: Path) -> None:
        """The helper reads the merged mapping, so an override is what the three call sites get."""
        override = tmp_path / "pinned.yaml"
        override.write_text("praat_features:\n  pitch_pinned_octave_ratio: 2.5\n")
        assert f0_range_parameters(load_triage_config(override))["pitch_pinned_octave_ratio"] == 2.5

    def test_the_helper_names_exactly_the_parameters_derive_f0_range_requires(self) -> None:
        """A parameter added to the signature without a config key fails here, not in a corpus pass."""
        required = {
            name
            for name, parameter in inspect.signature(derive_f0_range).parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        }
        assert required == set(F0_RANGE_PARAMETER_NAMES)
        assert all(
            parameter.default is inspect.Parameter.empty
            for name, parameter in inspect.signature(derive_f0_range).parameters.items()
            if name in required
        ), "derive_f0_range is the triage-facing wrapper; no default may stand in for a config key"


class TestEverySettingCppsIsComputedUnderIsAKey:
    """CPPS has no coefficient left in code: a value that moves the number moves a config key.

    Their derivations are in ``specs/20260817-triage-workflow-dag/config-derivations.md``.
    """

    def test_every_field_of_the_settings_resolves_to_a_key(self) -> None:
        """A field added without a key would silently take the library default on every run."""
        settings = cpps_settings(load_triage_config())
        packaged = load_triage_config().require("praat_features.cpps")
        named = set(CPPS_SCALAR_KEYS) | {
            "peak_search_range_hz",
            "trend_range_s",
            "subtract_tilt_before_smoothing",
            "tilt_line_type",
            "peak_interpolation",
        }
        assert set(packaged) == named, "a cpps key nothing reads, or a field with no key"
        assert len({field.name for field in fields(settings)}) == len(named) + 2, (
            "the two range keys become four fields; any other count means a field lost its key"
        )

    def test_the_packaged_values_are_the_ones_step_4_declares(self) -> None:
        """The band is 60-700, the two smoothing windows are the incumbent's, and nothing is derived."""
        settings = cpps_settings(load_triage_config())
        assert (settings.peak_search_floor_hz, settings.peak_search_ceiling_hz) == (60.0, 700.0)
        assert (settings.time_averaging_s, settings.quefrency_averaging_s) == (0.01, 0.001)
        assert (settings.trend_start_s, settings.trend_end_s) == (0.001, 0.0)
        assert settings.max_frequency_hz == 5000.0
        assert settings.robust_tolerance == 0.05
        assert settings.subtract_tilt_before_smoothing is False
        assert (settings.tilt_line_type, settings.peak_interpolation) == ("straight", "parabolic")

    def test_an_override_reaches_the_settings(self, tmp_path: Path) -> None:
        """The helper reads the merged mapping, so an override is what the extractor gets."""
        override = tmp_path / "cpps.yaml"
        override.write_text("praat_features:\n  cpps:\n    peak_search_range_hz: [60.0, 500.0]\n")
        assert cpps_settings(load_triage_config(override)).peak_search_ceiling_hz == 500.0

    def test_the_trend_range_and_the_peak_band_are_separate_keys(self) -> None:
        """Conflating them is the error step 4 names: the fit range is not the search band."""
        settings = cpps_settings(load_triage_config())
        assert settings.trend_start_s < 1.0 / settings.peak_search_ceiling_hz
        assert settings.trend_end_s == 0.0, "the fit runs to the end of the axis, past the peak band"


class TestTheHashNamesParametersOnly:
    """``config_hash`` is over the merged mapping, so prose in a value changes a run's identity."""

    def test_no_value_carries_prose(self) -> None:
        """Every string value is a short token, not a sentence, and none spans lines.

        The file once carried a 50 kB ``derivation`` string, which put the reasoning inside
        ``config_hash``: correcting a word made two behaviourally identical runs report different
        identities. Descriptions are ``#`` comments now, which the loader never reads. This fails
        if prose returns to a value.
        """
        offenders = []
        for path, value in _leaves(load_triage_config().values).items():
            if isinstance(value, str) and ("\n" in value or len(value) > _MAX_VALUE_CHARS):
                offenders.append(f"{path} ({len(value)} chars)")
        assert not offenders, (
            "config values must not carry prose — put it in "
            "specs/20260817-triage-workflow-dag/config-derivations.md and leave a `#` comment: " + ", ".join(offenders)
        )

    def test_the_derivation_key_is_gone(self) -> None:
        """Its prose moved to the spec; nothing in the package reads the key."""
        assert "derivation" not in load_triage_config().values
