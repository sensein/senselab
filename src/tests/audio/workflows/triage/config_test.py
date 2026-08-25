"""The triage configuration: every number, its derivation, and what happens when one is unset."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import senselab.audio.tasks
from senselab.audio.workflows.triage.config import DATA_MAP_PATHS, TriageConfig, load_triage_config
from senselab.text.tasks.pii_detection.api import default_detectors


class TestMeasuredValues:
    """A value with a derivation is readable."""

    def test_the_measured_values_are_present(self) -> None:
        """Each measured value reads back exactly as the file states it."""
        cfg = load_triage_config()
        assert cfg.require("envelope.lowpass_hz") == 40.0
        assert cfg.require("spans.onset_drop_db") == 15.0
        assert cfg.require("spans.offset_fraction") == 0.7
        assert cfg.require("spans.k_db.airway") == 18.0
        assert cfg.require("preemphasis.coefficient") == 0.97
        assert cfg.require("floor.eval_grid_s") == 0.1
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
            "speech.word_gap_ms",
            "quality.stoi_floor",
            "taxonomy.voice_min_duration_s",
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
        override.write_text("spans:\n  onset_drop_db: 12.0\n")
        assert load_triage_config(override).config_hash != load_triage_config().config_hash

    def test_an_unknown_key_is_refused_rather_than_ignored(self, tmp_path: Path) -> None:
        """A typo in an override key is an error, not a no-op."""
        override = tmp_path / "o.yaml"
        override.write_text("spans:\n  onset_drpo_db: 12.0\n")
        with pytest.raises(ValueError, match="onset_drpo_db"):
            load_triage_config(override)


class TestOverridesMayExtendADataMap:
    """A schema key is a name the code reads; a data-map key is a value the data supplies."""

    def test_a_new_confirmation_map_entry_is_accepted(self, tmp_path: Path) -> None:
        """A campaign screening for sneezes must be able to say so without editing the package."""
        override = tmp_path / "o.yaml"
        override.write_text("airway:\n  confirmation_map:\n    Sneeze: [Sneeze]\n")
        cfg = load_triage_config(override)
        assert cfg.require("airway.confirmation_map")["Sneeze"] == ["Sneeze"]

    def test_the_packaged_entries_survive_the_addition(self, tmp_path: Path) -> None:
        """An additive override that silently dropped Cough would disable the branch it extended."""
        override = tmp_path / "o.yaml"
        override.write_text("airway:\n  confirmation_map:\n    Sneeze: [Sneeze]\n")
        confirmation = load_triage_config(override).require("airway.confirmation_map")
        assert confirmation["Cough"] == ["Cough"]
        assert confirmation["Breathe"] == ["Breathing", "Sigh", "Gasp"]

    def test_an_existing_entry_is_replaced_not_merged(self, tmp_path: Path) -> None:
        """The value under a data-map key is data; two lists do not deep-merge into one."""
        override = tmp_path / "o.yaml"
        override.write_text("airway:\n  confirmation_map:\n    Breathe: [Breathing]\n")
        assert load_triage_config(override).require("airway.confirmation_map")["Breathe"] == ["Breathing"]

    def test_a_new_span_gate_kind_is_accepted(self, tmp_path: Path) -> None:
        """``spans.k_db`` is keyed by kind, and a kind is data the caller supplies."""
        override = tmp_path / "o.yaml"
        override.write_text("spans:\n  k_db:\n    speech: 12.0\n")
        cfg = load_triage_config(override)
        assert cfg.require("spans.k_db") == {"airway": 18.0, "speech": 12.0}

    def test_a_null_data_map_still_takes_a_whole_mapping(self, tmp_path: Path) -> None:
        """The control: the paths that ship null must keep accepting the mapping that fills them."""
        override = tmp_path / "o.yaml"
        override.write_text("routing:\n  hint_kind_map:\n    cough: airway\n")
        assert load_triage_config(override).require("routing.hint_kind_map") == {"cough": "airway"}

    def test_a_schema_key_is_still_refused(self, tmp_path: Path) -> None:
        """The whole point of the refusal: a section the code reads by name cannot grow a key."""
        override = tmp_path / "o.yaml"
        override.write_text("taxonomy:\n  nonsense: 1\n")
        with pytest.raises(ValueError, match="nonsense"):
            load_triage_config(override)

    def test_a_schema_key_inside_a_section_holding_a_data_map_is_still_refused(self, tmp_path: Path) -> None:
        """The exemption is the map, not the section it sits in."""
        override = tmp_path / "o.yaml"
        override.write_text("airway:\n  confirmatoin_map:\n    Sneeze: [Sneeze]\n")
        with pytest.raises(ValueError, match="confirmatoin_map"):
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
        "windows.yamnet.default_threshold",
        "windows.yamnet.label_thresholds",
        "windows.ast.default_threshold",
        "windows.ast.label_thresholds",
        "windows.hear.default_threshold",
        "windows.hear.label_thresholds",
        "phonation_spans.f0_stability_cents",
        "phonation_spans.formant_stability_hz",
        "phonation_spans.glide_min_excursion_cents",
        "phonation_spans.hangover_ms",
        "phonation_spans.voicing_strength_floor",
        "phonation_spans.mixed_voiced_fraction",
        "words.onomatopoeic_tokens",
        "taxonomy.presence_floor.speech.acoustic",
        "taxonomy.presence_floor.speech.lexical",
        "taxonomy.presence_floor.airway.health_acoustic",
        "taxonomy.presence_floor.airway.acoustic",
        "taxonomy.voice_min_duration_s",
        "taxonomy.voice_uncertain_duration_s",
        "taxonomy.speech_labels",
        "routing.hint_kind_map",
        "airway.k_db",
        "airway.k_db_by_task",
        "airway.k_margin_db",
        "airway.contest_labels",
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
            "taxonomy.audioset_speech_labels",
            "taxonomy.min_families",
            "taxonomy.ast_frame_s",
            "taxonomy.lexical_airway_tokens",
            "taxonomy.presence_floor.yamnet",
            "hear.label_floor",
        ):
            with pytest.raises(ValueError, match="unknown configuration key"):
                config.require(path)

    def test_the_window_hops_are_declared_defaults_not_open_keys(self) -> None:
        """A null hop is not the honest state here: it stopped the classifier running at all.

        ``require`` raises on a null, and both hops are read inside the *scores* block, so while they
        were null AST and HeAR never ran under the packaged config -- the expensive model output was
        lost along with the threshold fold V3 exists to let it survive. Both now ship non-overlapping,
        which is a declared choice the config_hash names, while the thresholds stay null.
        """
        config = load_triage_config()
        assert config.require("windows.ast.hop_s") == 10.24
        assert config.require("windows.ast.win_length_s") == 10.24
        assert config.require("windows.hear.hop_s") == 2.0
        for path in ("windows.ast.default_threshold", "windows.hear.default_threshold"):
            with pytest.raises(ValueError, match="has no value"):
                config.require(path)

    def test_the_f0_range_replaces_the_two_scalar_keys(self) -> None:
        """One range, read by PREPROCESS and VOICE alike, so the two cannot drift."""
        config = load_triage_config()
        with pytest.raises(ValueError, match="has no value"):
            config.require("voice.f0_range_hz")
        assert config.get("voice.f0_range_hz", "SENTINEL") == "SENTINEL"
