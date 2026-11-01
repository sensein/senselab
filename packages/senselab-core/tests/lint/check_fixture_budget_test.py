"""The fixture budget: 256 KB per file and 2 MB of data under ``packages/``, and no symlink or submodule."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.logic

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "check_fixture_budget.py"
_spec = importlib.util.spec_from_file_location("check_fixture_budget", _SCRIPT)
assert _spec and _spec.loader
budget = importlib.util.module_from_spec(_spec)
sys.modules["check_fixture_budget"] = budget
_spec.loader.exec_module(budget)

Obj = budget.TrackedObject


def _obj(path: str, size: int, mode: str = "100644") -> object:
    return Obj(path, mode, size)


def test_a_file_over_the_per_file_limit_is_refused() -> None:
    """A file of 256 KB passes; one byte more is named with its size."""
    assert budget.check([_obj("packages/a/data/x.bin", budget.FILE_MAX_BYTES)]) == []
    (message,) = budget.check([_obj("packages/a/data/x.bin", budget.FILE_MAX_BYTES + 1)])
    assert "packages/a/data/x.bin" in message and str(budget.FILE_MAX_BYTES + 1) in message


def test_code_over_the_per_file_limit_is_refused_but_not_totalled() -> None:
    """A code file counts against the per-file limit only; many small ones do not reach the data total."""
    assert budget.check([_obj("packages/a/big.py", budget.FILE_MAX_BYTES + 1)])
    assert budget.check([_obj(f"packages/a/m{i}.py", budget.FILE_MAX_BYTES) for i in range(20)]) == []


def test_data_files_are_capped_in_total() -> None:
    """Data files each under the limit are refused once their total passes 2 MB."""
    count = budget.DATA_TOTAL_MAX_BYTES // 200_000
    assert budget.check([_obj(f"packages/a/d{i}.bin", 200_000) for i in range(count)]) == []
    assert budget.check([_obj(f"packages/a/d{i}.bin", 200_000) for i in range(count + 1)])


def test_files_outside_packages_are_not_budgeted() -> None:
    """The 1.x pool and its data are out of scope until a file moves under ``packages/``."""
    assert budget.check([_obj("src/tests/data_for_testing/video.mp4", 10_000_000)]) == []


@pytest.mark.parametrize("mode", ["120000", "160000", "100664"])
def test_symlinks_submodules_and_odd_modes_are_refused_anywhere(mode: str) -> None:
    """Only regular files may be tracked, wherever they sit."""
    assert budget.check([_obj("src/anything", 0, mode)])


def test_budget_passes_on_current_tree() -> None:
    """The script exits 0 on the committed index."""
    done = subprocess.run([sys.executable, str(_SCRIPT)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout


@pytest.mark.parametrize(
    "entry",
    ["  value: 256000\n", "  value: 0\n  derivation: d\n", "  value: true\n  derivation: d\n"],
)
def test_a_limit_without_a_derivation_or_a_positive_value_is_refused(tmp_path: Path, entry: str) -> None:
    """The budget file is read only when each limit is a positive integer with a derivation."""
    good = "  value: 2000000\n  derivation: d\n"
    path = tmp_path / "budget.yaml"
    path.write_text(f"format: {budget.BUDGET_FORMAT}\nfile_max_bytes:\n{entry}data_total_max_bytes:\n{good}")
    with pytest.raises(ValueError, match="file_max_bytes"):
        budget.load_budget(path)
    path.write_text(f"format: {budget.BUDGET_FORMAT}\nfile_max_bytes:\n{good}data_total_max_bytes:\n{good}")
    assert budget.load_budget(path) == {"file_max_bytes": 2000000, "data_total_max_bytes": 2000000}
