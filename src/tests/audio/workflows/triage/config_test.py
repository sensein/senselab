"""The triage configuration: every number, its derivation, and what happens when one is unset."""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config


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
        with pytest.raises(ValueError, match="phonation.hnr_floor_db"):
            cfg.require("phonation.hnr_floor_db")

    def test_a_typo_is_an_unknown_key_not_an_unmeasured_value(self) -> None:
        """An absent key is a typo, and the error says so instead of citing open.md."""
        cfg = load_triage_config()
        with pytest.raises(ValueError, match="unknown configuration key"):
            cfg.require("phonation.hnr_flor_db")

    def test_a_null_key_is_unmeasured_not_unknown(self) -> None:
        """A present-null key still points at open.md, never at the typo message."""
        cfg = load_triage_config()
        with pytest.raises(ValueError, match="benchmarks/open.md"):
            cfg.require("phonation.rms_floor")

    def test_the_error_points_at_what_would_settle_it(self) -> None:
        """The error names the open-questions file."""
        cfg = load_triage_config()
        with pytest.raises(ValueError, match="benchmarks/open.md"):
            cfg.require("redaction.padding_ms")

    def test_every_unset_value_is_null_rather_than_absent(self) -> None:
        """Absent is a typo; null is a decision not yet taken."""
        cfg = load_triage_config()
        for path in (
            "phonation.hnr_floor_db",
            "phonation.rms_floor",
            "redaction.padding_ms",
            "speech.word_gap_ms",
            "quality.stoi_floor",
            "taxonomy.min_families.airway",
        ):
            node: object = cfg.values
            for part in path.split("."):
                assert isinstance(node, dict) and part in node, f"{path} must be present and null"
                node = node[part]
            assert node is None, f"{path} must be present and null"

    def test_get_returns_a_default_instead_of_raising(self) -> None:
        """A caller that can proceed without the value may ask politely."""
        cfg = load_triage_config()
        assert cfg.get("phonation.hnr_floor_db", 8.0) == 8.0


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
