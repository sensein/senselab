"""The fixture budget flags new large files and growth, and spares grandfathered ones."""

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


def test_budget_flags_large_new_file(tmp_path: Path) -> None:
    """A new file over the per-file limit yields one message naming it, its size and the fix."""
    big = _file(tmp_path, "tests/data/big.wav", 300 * 1024)
    msgs = budget.check([big], allow=set(), root=tmp_path)
    assert len(msgs) == 1 and "tests/data/big.wav" in msgs[0] and "262144" in msgs[0] and "generate" in msgs[0]


def test_budget_allows_grandfathered(tmp_path: Path) -> None:
    """An allowlisted file is spared, unless it grew past its recorded size."""
    big = _file(tmp_path, "tests/data/big.wav", 300 * 1024)
    assert budget.check([big], allow={"tests/data/big.wav"}, root=tmp_path) == []
    grown = budget.check([big], allow={"tests/data/big.wav"}, root=tmp_path, sizes={"tests/data/big.wav": 1000})
    assert len(grown) == 1 and "grew" in grown[0]


def test_budget_total_counts_only_non_allowlisted(tmp_path: Path) -> None:
    """Only non-allowlisted files count toward the total."""
    files = [_file(tmp_path, f"d/{i}.bin", 200 * 1024) for i in range(11)]
    assert any("total" in m for m in budget.check(files, allow=set(), root=tmp_path))
    assert budget.check(files, allow={f"d/{i}.bin" for i in range(11)}, root=tmp_path) == []


def test_budget_passes_on_current_tree() -> None:
    """The script exits 0 on the committed tree."""
    assert subprocess.run([sys.executable, str(_SCRIPT)], capture_output=True).returncode == 0
