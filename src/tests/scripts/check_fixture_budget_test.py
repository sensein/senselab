"""The fixture budget flags new large files and growth, binds its allowlist to the freeze, and scans every path."""

import hashlib
import importlib.util
import os
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

FREEZE = budget.FREEZE_BRANCH
SHA0 = "0" * 64


def _file(root: Path, name: str, size: int) -> Path:
    """Write `size` zero bytes at `root/name`."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0" * size)
    return path


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _line(name: str, size: int, *, fill: bytes = b"\0") -> str:
    return f"{name}\t{hashlib.sha256(fill * size).hexdigest()}\t{size}\n"


def _repo(root: Path, frozen: dict[str, int], later: dict[str, int], allowlist: str) -> Path:
    """A git repo whose freeze branch holds `frozen` (name to zero-byte size) and whose index adds `later`."""
    _git(root, "init", "-q", "-b", "main")
    for name, size in frozen.items():
        _file(root, name, size)
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "--allow-empty", "-m", "freeze")
    _git(root, "branch", FREEZE)
    for name, size in later.items():
        _file(root, name, size)
    _git(root, "add", "-A")
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
    """Only non-allowlisted test-tree fixtures count toward the total."""
    names = [f"src/tests/d/{i}.bin" for i in range(11)]
    files = [_file(tmp_path, n, 200 * 1024) for n in names]
    assert any("total" in m for m in budget.check(files, allow={}, root=tmp_path))
    assert budget.check(files, allow={n: 200 * 1024 for n in names}, root=tmp_path) == []


def test_allowlist_parser_rejects_missing_and_malformed_columns() -> None:
    """A line without a hash or size, or with a malformed one, is an error naming the line."""
    assert budget.parse_allowlist(f"# c\n\na/b.wav\t{SHA0}\t12\n") == {"a/b.wav": budget.Entry(SHA0, 12)}
    bad = (
        (f"a.wav\t{SHA0}\t1\nb.wav\n", "line 2"),
        (f"a.wav\t{SHA0}\tx\n", "line 1"),
        ("a.wav 12\n", "line 1"),
        (f"a.wav\t{'g' * 64}\t1\n", "line 1"),
    )
    for text, line in bad:
        with pytest.raises(ValueError, match=line):
            budget.parse_allowlist(text)


def test_main_passes_on_a_frozen_entry_and_catches_growth(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A grandfathered file passes at its frozen bytes; one that grows or changes fails."""
    name = "src/tests/data_for_testing/a.wav"
    allow = _repo(tmp_path, {name: 500}, {}, _line(name, 500))
    assert budget.main(tmp_path, allow) == 0
    (tmp_path / name).write_bytes(b"\0" * 600)
    assert budget.main(tmp_path, allow) == 1 and "grew" in capsys.readouterr().out
    (tmp_path / name).write_bytes(b"\1" * 500)
    assert budget.main(tmp_path, allow) == 1 and "content differs" in capsys.readouterr().out


def test_new_allowlist_entry_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An entry for a path absent from the freeze fails, however large or well hashed the file."""
    name = "src/tests/data_for_testing/new.wav"
    allow = _repo(tmp_path, {}, {name: 9 * 1024 * 1024}, _line(name, 9 * 1024 * 1024))
    assert budget.main(tmp_path, allow) == 1
    assert f"{name}: not in" in capsys.readouterr().out


def test_entry_with_other_hash_or_larger_size_than_the_freeze_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An entry must record the frozen content's hash and no more than its size."""
    name = "src/tests/data_for_testing/a.wav"
    allow = _repo(tmp_path, {name: 500}, {}, _line(name, 500, fill=b"\1"))
    assert budget.main(tmp_path, allow) == 1 and "not the content" in capsys.readouterr().out
    allow.write_text(f"{name}\t{hashlib.sha256(bytes(500)).hexdigest()}\t900\n")
    assert budget.main(tmp_path, allow) == 1 and "larger" in capsys.readouterr().out


def test_grown_fixture_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A frozen fixture that grew past its frozen size fails even when the allowlist is edited to its new size."""
    name = "src/tests/data_for_testing/a.wav"
    allow = _repo(tmp_path, {name: 500}, {}, _line(name, 500))
    _file(tmp_path, name, 700)
    allow.write_text(_line(name, 700))
    assert budget.main(tmp_path, allow) == 1
    assert capsys.readouterr().out


def test_large_code_file_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A code file over the policy's cap fails in a non-test tree; a small one passes."""
    cap = budget.load_policy().code_file_max_bytes
    for suffix in ("py", "pyi", "ipynb"):
        (tmp_path / suffix).mkdir()
        allow = _repo(tmp_path / suffix, {}, {f"src/pkg/blob.{suffix}": cap + 1, "src/pkg/ok.py": 100}, "")
        assert budget.main(tmp_path / suffix, allow) == 1
        out = capsys.readouterr().out
        assert f"src/pkg/blob.{suffix}" in out and "code-file limit" in out and "ok.py" not in out


def test_symlink_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A tracked symlink to a file or to a directory fails."""
    allow = _repo(tmp_path, {"real/a.txt": 10}, {}, "")
    os.symlink("real/a.txt", tmp_path / "link.txt")
    os.symlink("real", tmp_path / "linkdir")
    _git(tmp_path, "add", "-A")
    assert budget.main(tmp_path, allow) == 1
    out = capsys.readouterr().out
    assert "link.txt: tracked symlink" in out and "linkdir: tracked symlink" in out


def test_no_symlink_is_exempt(tmp_path: Path) -> None:
    """A symlink fails whatever its name and target, including the one the hatch hook used to be."""
    for name, target in {
        "packages/senselab-core/hatch_build_pins.py": "../../hatch_build_pins.py",
        "a.txt": "b",
    }.items():
        link = tmp_path / name
        link.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, link)
        assert "tracked symlink" in budget.check([link], allow={}, root=tmp_path)[0]


def test_repository_tracks_no_symlink() -> None:
    """Nothing in this repository's index has mode 120000."""
    assert [path for mode, path in budget._tracked(budget.ROOT) if mode == budget.SYMLINK_MODE] == []


@pytest.mark.parametrize("where", ["docs/new.png", "specs/x/new.json", "src/senselab/data/new.bin", "new.bin"])
def test_non_test_tree_is_scanned(tmp_path: Path, where: str) -> None:
    """An oversized non-code file is caught under any tree, not only the test trees."""
    allow = _repo(tmp_path, {}, {where: 300 * 1024}, "")
    assert budget.main(tmp_path, allow) == 1


def test_generated_files_are_exempt_from_the_per_file_limit(tmp_path: Path) -> None:
    """A file matching the policy's generated globs is not capped; the same bytes elsewhere are."""
    gen = _file(tmp_path, "uv.lock", 300 * 1024)
    other = _file(tmp_path, "other.lock", 300 * 1024)
    assert budget.check([gen], allow={}, root=tmp_path, generated=("uv.lock",)) == []
    assert len(budget.check([other], allow={}, root=tmp_path, generated=("uv.lock",))) == 1


@pytest.mark.parametrize("where", ["src/tests/some/other/new.wav", "packages/senselab-core/tests/new.wav"])
def test_main_catches_new_large_wav_anywhere_in_test_trees(tmp_path: Path, where: str) -> None:
    """A new oversized `.wav` outside data_for_testing is caught; a large `.py` source is not a fixture."""
    allow = _repo(tmp_path, {}, {where: 300 * 1024, "src/tests/big_test.py": 300 * 1024}, "")
    assert budget.main(tmp_path, allow) == 1
    assert budget.is_fixture(where) and not budget.is_fixture("src/tests/big_test.py")


def test_a_local_freeze_branch_never_takes_precedence(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """With an `origin/` freeze ref present, entries are bound to it even when the local branch holds more."""
    name = "src/tests/data_for_testing/added_locally.wav"
    allow = _repo(tmp_path, {}, {}, _line(name, 500))
    _git(tmp_path, "update-ref", f"refs/remotes/origin/{FREEZE}", FREEZE)
    _git(tmp_path, "checkout", "-q", FREEZE)
    _file(tmp_path, name, 500)
    _git(tmp_path, "add", name)
    _git(tmp_path, "commit", "-q", "-m", "advance the local branch")
    _git(tmp_path, "checkout", "-q", "main")
    allow.write_text(_line(name, 500))
    assert budget.resolve_freeze_ref(tmp_path) == f"origin/{FREEZE}"
    assert budget.main(tmp_path, allow) == 1 and f"{name}: not in origin/{FREEZE}" in capsys.readouterr().out


def test_main_errors_without_the_freeze_branch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """With no freeze ref the check fails naming the fetch, rather than passing unbound."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "commit", "-q", "--allow-empty", "-m", "x")
    allow = tmp_path / "allow.txt"
    allow.write_text("")
    assert budget.main(tmp_path, allow) == 1 and "git fetch origin" in capsys.readouterr().out


def test_allowlist_has_no_dead_entries() -> None:
    """Every allowlist entry names a tracked file."""
    allow = budget.parse_allowlist(budget.ALLOWLIST.read_text())
    tracked = {p for _mode, p in budget._tracked(budget.ROOT)}
    assert allow and set(allow) <= tracked


def test_budget_passes_on_current_tree() -> None:
    """The script exits 0 on the committed tree, which also binds the allowlist to the freeze branch."""
    try:
        budget.resolve_freeze_ref(budget.ROOT)
    except LookupError:
        pytest.skip("freeze branch not fetched here; the pre-commit job runs the script with it")
    assert subprocess.run([sys.executable, str(_SCRIPT)], capture_output=True).returncode == 0
