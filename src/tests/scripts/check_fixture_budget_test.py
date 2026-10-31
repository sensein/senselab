"""The fixture budget flags new large files and growth, spares grandfathered ones, and scans every test tree."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check_fixture_budget.py"
_spec = importlib.util.spec_from_file_location("check_fixture_budget", _SCRIPT)
assert _spec and _spec.loader
budget = importlib.util.module_from_spec(_spec)
sys.modules["check_fixture_budget"] = budget
_spec.loader.exec_module(budget)


def _file(root: Path, name: str, size: int) -> Path:
    """Write `size` zero bytes at `root/name`."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


def _repo(root: Path, files: dict[str, int], allowlist: str) -> Path:
    """Make a git repo at `root` tracking `files` (name to size), and write its allowlist; return the path."""
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    for name, size in files.items():
        _file(root, name, size)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    path = root / "allow.txt"
    path.write_text(allowlist)
    return path


def test_budget_flags_large_new_file(tmp_path: Path) -> None:
    """A new file over the per-file limit yields one message naming it, its size and the fix."""
    big = _file(tmp_path, "tests/data/big.wav", 300 * 1024)
    msgs = budget.check([big], allow={}, root=tmp_path)
    assert len(msgs) == 1 and "tests/data/big.wav" in msgs[0] and "262144" in msgs[0] and "generate" in msgs[0]


def test_budget_allows_grandfathered(tmp_path: Path) -> None:
    """An allowlisted file is spared at its recorded size and flagged once it grows."""
    big = _file(tmp_path, "tests/data/big.wav", 300 * 1024)
    assert budget.check([big], allow={"tests/data/big.wav": 300 * 1024}, root=tmp_path) == []
    grown = budget.check([big], allow={"tests/data/big.wav": 1000}, root=tmp_path)
    assert len(grown) == 1 and "grew" in grown[0]


def test_budget_total_counts_only_non_allowlisted(tmp_path: Path) -> None:
    """Only non-allowlisted files count toward the total."""
    files = [_file(tmp_path, f"d/{i}.bin", 200 * 1024) for i in range(11)]
    assert any("total" in m for m in budget.check(files, allow={}, root=tmp_path))
    allow = {f"d/{i}.bin": 200 * 1024 for i in range(11)}
    assert budget.check(files, allow=allow, root=tmp_path) == []


def test_allowlist_parser_rejects_missing_and_malformed_sizes() -> None:
    """A line without a size, or with a non-numeric one, is an error naming the line."""
    assert budget.parse_allowlist("# c\n\na/b.wav 12\n") == {"a/b.wav": 12}
    for text, line in (("a.wav 1\nb.wav\n", "line 2"), ("a.wav x\n", "line 1")):
        with pytest.raises(ValueError, match=line):
            budget.parse_allowlist(text)


def test_main_catches_grown_allowlisted_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main() fails on an allowlisted file larger than its recorded size, with no dependence on git history."""
    name = "src/tests/data_for_testing/a.wav"
    ok = _repo(tmp_path, {name: 500}, f"{name} 500\n")
    assert budget.main(tmp_path, ok) == 0
    ok.write_text(f"{name} 100\n")
    assert budget.main(tmp_path, ok) == 1 and "grew" in capsys.readouterr().out


def test_main_errors_on_missing_size_or_malformed_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """main() fails, naming the line, on an allowlist entry without a size."""
    path = _repo(tmp_path, {}, "src/tests/data_for_testing/a.wav\n")
    assert budget.main(tmp_path, path) == 1 and "line 1" in capsys.readouterr().out


@pytest.mark.parametrize("where", ["src/tests/some/other/new.wav", "packages/senselab-core/tests/new.wav"])
def test_main_catches_new_large_wav_anywhere_in_test_trees(tmp_path: Path, where: str) -> None:
    """A new oversized `.wav` outside data_for_testing is caught; a large `.py` source is not a fixture."""
    path = _repo(tmp_path, {where: 300 * 1024, "src/tests/big_test.py": 300 * 1024}, "")
    assert budget.main(tmp_path, path) == 1
    assert budget.is_fixture(where) and not budget.is_fixture("src/tests/big_test.py")


def test_large_json_is_caught_and_py_is_not_counted(tmp_path: Path) -> None:
    """A new large `.json` (or yaml, md, toml) under src/tests counts toward the budget; code does not."""
    for ext in ("json", "yaml", "md", "toml"):
        assert budget.is_fixture(f"src/tests/x/data.{ext}")
    path = _repo(tmp_path, {"src/tests/x/data.json": 300 * 1024, "src/tests/x/big_test.py": 3 * 1024 * 1024}, "")
    assert budget.main(tmp_path, path) == 1
    path.write_text("src/tests/x/data.json 307200\n")
    assert budget.main(tmp_path, path) == 0


def test_allowlist_has_no_dead_entries() -> None:
    """Every allowlist entry names a tracked file the check scans."""
    allow = budget.parse_allowlist(budget.ALLOWLIST.read_text())
    tracked = set(budget._tracked(budget.ROOT))
    assert allow and set(allow) <= tracked


def test_budget_passes_on_current_tree() -> None:
    """The script exits 0 on the committed tree."""
    assert subprocess.run([sys.executable, str(_SCRIPT)], capture_output=True).returncode == 0
