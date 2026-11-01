"""No ruling id, finding id or review-round name under ``packages/``; they live in ``specs/``."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "check_spec_ids.py"
_spec = importlib.util.spec_from_file_location("check_spec_ids", _SCRIPT)
assert _spec and _spec.loader
ids = importlib.util.module_from_spec(_spec)
sys.modules["check_spec_ids"] = ids
_spec.loader.exec_module(ids)


def _kinds(*parts: str) -> list[str]:
    """The kinds found in a line assembled from `parts`, so this file holds no id itself."""
    found: list[str] = ids.kinds("".join(parts))
    return found


def test_the_tree_is_free_of_ids() -> None:
    """The script exits 0 over the committed index."""
    done = subprocess.run([sys.executable, str(_SCRIPT)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout


@pytest.mark.parametrize(
    ("parts", "kind"),
    [
        (("see ", "H-", "R", "12"), "ruling id"),
        (("ruling ", "R", "9"), "ruling id"),
        (("# ", "R-", "37"), "register or finding id"),
        (("# ", "F-", "3"), "register or finding id"),
        (("# ", "A-", "14"), "register or finding id"),
        (("# ", "X", "K-", "9"), "register or finding id"),
        (("round ", "r", "07"), "review-round name"),
        (("task ", "H", "9c"), "review-round name"),
        (("the ", "full", "-review"), "review-round name"),
        (("the ", "re", "attack"), "review-round name"),
        (("a ", "Re", "-Review"), "review-round name"),
    ],
)
def test_each_id_family_is_found(parts: tuple[str, ...], kind: str) -> None:
    """A planted id of each family is reported under its kind."""
    assert kind in _kinds(*parts)


@pytest.mark.parametrize(
    "line",
    ["GPU H100 80GB", "UTF-8 and ISO-8601", "ruff F401", "P-23 and ADR-002", "order"],
)
def test_other_text_is_not_flagged(line: str) -> None:
    """GPU names, encodings, lint codes and principle or ADR pointers are not ids of a review."""
    assert ids.kinds(line) == []


def test_findings_name_file_line_and_kind_and_skip_other_trees(tmp_path: Path) -> None:
    """A hit is `path:line: kind`; a file outside `packages/` or of another suffix is not read."""
    inside = tmp_path / "packages" / "a" / "m.py"
    inside.parent.mkdir(parents=True)
    inside.write_text("x = 1\n# " + "H-" + "R" + "5" + "\n")
    outside = tmp_path / "specs" / "n.md"
    outside.parent.mkdir()
    outside.write_text("H-" + "R" + "5\n")
    found = ids.findings(["packages/a/m.py", "specs/n.md"], tmp_path)
    assert len(found) == 1 and found[0].startswith("packages/a/m.py:2: ruling id")
    assert ids.main(["packages/a/m.py"], tmp_path) == 1
