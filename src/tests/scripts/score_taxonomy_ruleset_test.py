"""Tests for ``scripts/score_taxonomy_ruleset.py``."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

cli = importlib.import_module("senselab.audio.workflows.triage.drivers.score_taxonomy_ruleset")


def test_without_a_profile_the_driver_exits_2_naming_it(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Review A1/A2: the ruleset is the profile's; with none it exits 2, before reading the shard, writing nothing."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")
    assert cli.main([str(tmp_path), str(tmp_path / "out")]) == 2
    assert "dataset profile" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
