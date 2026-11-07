"""The config-derivations check: config files hold values, and each key's derivation is a spec section."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.logic

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "check_config_derivations.py"
_spec = importlib.util.spec_from_file_location("check_config_derivations", _SCRIPT)
assert _spec and _spec.loader
cd = importlib.util.module_from_spec(_spec)
sys.modules["check_config_derivations"] = cd
_spec.loader.exec_module(cd)

PATH = "scripts/data/x.yaml"
SPEC = f"# Config derivations\n\n## other.yaml\n\n### a\n\n## {PATH}\n\n### limit\n\nWhy.\n\n### sets\n\nWhy.\n"


@pytest.mark.parametrize(
    "raw",
    [
        {"limit": {"value": 1, "derivation": "d"}},
        {"sets": {"a": {"ids": [1], "derivations": "d"}}},
        {"sets": [{"derivation": "d"}]},
        {"derivations": {"limit": "d"}},
    ],
)
def test_an_inline_derivation_fails_in_either_spelling(raw: dict[str, Any]) -> None:
    """A `derivation` or `derivations` key at any depth is refused, with or without the spec."""
    assert cd.problems(PATH, raw, None)
    assert cd.problems(PATH, raw, cd.sections(SPEC))


def test_a_key_without_a_section_fails() -> None:
    """Each top-level key but `format` needs a section under its file's heading; another file's does not count."""
    spec = cd.sections(SPEC)
    assert cd.problems(PATH, {"format": "x@1.0", "limit": 1, "sets": {}}, spec) == []
    assert cd.problems(PATH, {"limit": 1, "a": 2}, spec) == [f"{PATH}: a has no '### a' under '## {PATH}'"]
    assert cd.problems("scripts/data/y.yaml", {"limit": 1}, spec)


def test_the_tree_passes(tmp_path: Path) -> None:
    """The tracked configs hold no inline derivation; without the spec only that is checked."""
    assert cd.main(spec_path=tmp_path / "absent.md") == 0


def test_a_new_data_file_is_checked_without_being_listed(tmp_path: Path) -> None:
    """Any tracked `data/*.yaml` is checked: a new one with an inline derivation fails, and so do stale exemptions."""
    new = tmp_path / "packages" / "p" / "src" / "p" / "data" / "new.yaml"
    new.parent.mkdir(parents=True)
    new.write_text("limit:\n  value: 1\n  derivation: d\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    assert cd.tracked_data(tmp_path) == ["packages/p/src/p/data/new.yaml"]
    found = cd.findings(tmp_path, tmp_path / "absent.md")
    assert "packages/p/src/p/data/new.yaml: limit.derivation is inline" in " ".join(found)
    assert any("is exempt but not tracked" in line for line in found)
