"""The config-derivations check: config files hold values, and each key's derivation is a spec section."""

import importlib.util
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
