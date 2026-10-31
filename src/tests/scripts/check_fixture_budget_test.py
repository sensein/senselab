"""The fixture budget counts every tracked object, binds its allowlist and its policy to the freeze, and caps totals."""

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.logic

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check_fixture_budget.py"
_spec = importlib.util.spec_from_file_location("check_fixture_budget", _SCRIPT)
assert _spec and _spec.loader
budget = importlib.util.module_from_spec(_spec)
sys.modules["check_fixture_budget"] = budget
_spec.loader.exec_module(budget)

FREEZE = budget.FREEZE_BRANCH
SHA0 = "0" * 64
LIMITS = {
    "code_file_max_bytes": 1000,
    "code_total_max_bytes": 5000,
    "data_file_max_bytes": 500,
    "data_total_max_bytes": 4000,
    "test_data_total_max_bytes": 2000,
}
Obj = budget.TrackedObject


def _file(root: Path, name: str, size: int, fill: bytes = b"\0") -> Path:
    """Write `size` bytes of `fill` at `root/name`."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fill * size)
    return path


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _line(name: str, size: int, *, fill: bytes = b"\0") -> str:
    return f"{name}\t{hashlib.sha256(fill * size).hexdigest()}\t{size}\n"


def _policy(root: Path, *, generated: dict[str, str] | None = None, measured: dict[str, int] | None = None) -> Path:
    """A policy for the throwaway repo `root`: each cap's `measured` is what its freeze measures, unless overridden."""
    allow_path = root / "allow.txt"
    allow = budget.parse_allowlist(allow_path.read_text()) if allow_path.exists() else {}
    generated = generated or {}
    actual = budget.measure(budget.frozen_objects(root, FREEZE), allow, generated)
    caps = {n: {"measured": (measured or {}).get(n, actual[n]), "headroom": 2} for n in budget.CAPS}
    path = root / "policy.yaml"
    path.write_text(yaml.safe_dump({**caps, "generated_files": generated, "derivation": "test"}))
    return path


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


def _main(root: Path, allow: Path, **measured: int) -> int:
    return budget.main(root, allow, _policy(root, measured=measured))


def test_budget_flags_large_new_file() -> None:
    """A new data file over the per-file limit yields one message naming it, its size, the limit and the fix."""
    msgs = budget.check([Obj("tests/data/big.wav", "100644", "x", 600)], allow={}, limits=LIMITS)
    assert len(msgs) == 1 and "tests/data/big.wav" in msgs[0] and "500" in msgs[0] and "generate" in msgs[0]


def test_budget_allows_grandfathered() -> None:
    """An allowlisted file is spared at its recorded size, left out of the totals, and flagged once it grows."""
    big = Obj("src/tests/data/big.wav", "100644", "x", 9000)
    assert budget.check([big], allow={big.path: 9000}, limits=LIMITS) == []
    grown = budget.check([big], allow={big.path: 1000}, limits=LIMITS)
    assert len(grown) == 1 and "grew" in grown[0]


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
    """A grandfathered file passes at its frozen bytes; one whose indexed bytes grow or change fails."""
    name = "src/tests/data_for_testing/a.wav"
    allow = _repo(tmp_path, {name: 500}, {}, _line(name, 500))
    assert _main(tmp_path, allow) == 0
    _file(tmp_path, name, 600)
    _git(tmp_path, "add", name)
    assert _main(tmp_path, allow) == 1 and "grew" in capsys.readouterr().out
    _file(tmp_path, name, 500, b"\1")
    _git(tmp_path, "add", name)
    assert _main(tmp_path, allow) == 1 and "content differs" in capsys.readouterr().out


def test_new_allowlist_entry_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An entry for a path absent from the freeze fails, however large or well hashed the file."""
    name = "src/tests/data_for_testing/new.wav"
    allow = _repo(tmp_path, {}, {name: 9 * 1024}, _line(name, 9 * 1024))
    assert _main(tmp_path, allow) == 1
    assert f"{name}: not in" in capsys.readouterr().out


def test_entry_with_other_hash_or_larger_size_than_the_freeze_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An entry must record the frozen content's hash and no more than its size."""
    name = "src/tests/data_for_testing/a.wav"
    allow = _repo(tmp_path, {name: 500}, {}, _line(name, 500, fill=b"\1"))
    assert _main(tmp_path, allow) == 1 and "not the content" in capsys.readouterr().out
    allow.write_text(f"{name}\t{hashlib.sha256(bytes(500)).hexdigest()}\t900\n")
    assert _main(tmp_path, allow) == 1 and "larger" in capsys.readouterr().out


def test_large_code_file_is_refused() -> None:
    """A code file over the code-file cap fails wherever it is; a small one passes."""
    for suffix in ("py", "pyi", "ipynb"):
        msgs = budget.check(
            [Obj(f"src/pkg/blob.{suffix}", "100644", "x", 1001), Obj("src/pkg/ok.py", "100644", "y", 100)],
            allow={},
            limits=LIMITS,
        )
        assert len(msgs) == 1 and f"src/pkg/blob.{suffix}" in msgs[0] and "code-file limit" in msgs[0]


@pytest.mark.parametrize("where", ["src/tests/x", "packages/senselab-core/tests/x", "src/senselab/x", "docs"])
def test_code_files_are_capped_in_total_everywhere(where: str) -> None:
    """Code files each under the per-file cap are refused once their sum across the whole tree passes the total."""
    objects = [Obj(f"{where}/m{i}.py", "100644", str(i), 900) for i in range(6)]
    msgs = budget.check(objects, allow={}, limits=LIMITS)
    assert len(msgs) == 1 and "total of code files: 5400 bytes exceeds the 5000 byte limit" in msgs[0]


def test_xg10_mjs_files_under_the_test_trees_count_as_data() -> None:
    """A suffix that is not code is data: many `.mjs` files under the per-file cap exceed the test-tree total."""
    objects = [Obj(f"src/tests/js/m{i}.mjs", "100644", str(i), 450) for i in range(5)]
    msgs = budget.check(objects, allow={}, limits=LIMITS)
    assert any("under the test trees: 2250 bytes" in m for m in msgs)


def test_xg10_data_outside_the_test_trees_is_totalled() -> None:
    """Data files under a package's data directory, each under the per-file cap, are capped in total."""
    objects = [Obj(f"src/senselab/audio/data/f{i}.bin", "100644", str(i), 450) for i in range(10)]
    msgs = budget.check(objects, allow={}, limits=LIMITS)
    assert msgs and "total of non-allowlisted data files: 4500 bytes" in msgs[0]


@pytest.mark.parametrize(
    "path",
    [
        "src/senselab/utils/data/venv_locks/sub/payload.txt",
        "src/senselab/utils/data/venv_locks/payload.txt",
        "src/senselab/utils/data/venv_locks/a.txt.bak",
        "uv.lock.old",
    ],
)
def test_xg10_an_exemption_covers_exactly_its_path(path: str) -> None:
    """An exemption is one exact path: a sibling, a nested file or a similar name under it is capped."""
    generated = {"uv.lock": "uv-lock", "src/senselab/utils/data/venv_locks/a.txt": "venv-lock"}
    msgs = budget.check([Obj(path, "100644", "x", 900)], allow={}, limits=LIMITS, generated=generated)
    assert len(msgs) == 1 and "per-file limit" in msgs[0]
    assert budget.check([Obj("uv.lock", "100644", "x", 900)], allow={}, limits=LIMITS, generated=generated) == []


@pytest.mark.parametrize(
    "glob", ["src/senselab/utils/data/venv_locks/*.txt", "uv.lo?k", "[u]v.lock", "/uv.lock", "a/../b"]
)
def test_xg10_a_glob_or_unanchored_exemption_is_refused(glob: str, tmp_path: Path) -> None:
    """The policy names exemptions by exact path; a glob, an absolute or a `..` path fails to load."""
    path = tmp_path / "p.yaml"
    data = yaml.safe_load(budget.POLICY.read_text())
    data["generated_files"] = {glob: "venv-lock"}
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="exact"):
        budget.load_policy(path)


def test_xg10_a_tracked_submodule_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A gitlink (mode 160000) is refused, whatever it holds."""
    allow = _repo(tmp_path, {"a.txt": 10}, {}, "")
    commit = _git(tmp_path, "rev-parse", "HEAD").strip()
    _git(tmp_path, "update-index", "--add", "--cacheinfo", f"160000,{commit},src/tests/data/bigsubmodule")
    assert _main(tmp_path, allow) == 1
    assert "src/tests/data/bigsubmodule: tracked submodule" in capsys.readouterr().out


def test_xg10_an_indexed_symlink_is_refused_with_no_link_in_the_work_tree(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Mode 120000 is read from the index: a symlink absent from the checkout, or checked out as a file, fails."""
    allow = _repo(tmp_path, {"real/a.txt": 10}, {}, "")
    hashed = subprocess.run(
        ["git", "hash-object", "-w", "--stdin"], cwd=tmp_path, input=b"real/a.txt", capture_output=True, check=True
    )
    blob = hashed.stdout.decode().strip()
    _git(tmp_path, "update-index", "--add", "--cacheinfo", f"120000,{blob},absent-link")
    _git(tmp_path, "update-index", "--add", "--cacheinfo", f"120000,{blob},as-file")
    _file(tmp_path, "as-file", 10)
    assert not (tmp_path / "absent-link").exists() and not (tmp_path / "as-file").is_symlink()
    assert _main(tmp_path, allow) == 1
    out = capsys.readouterr().out
    assert "absent-link: tracked symlink" in out and "as-file: tracked symlink" in out


def test_symlink_is_refused(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A tracked symlink to a file or to a directory fails."""
    allow = _repo(tmp_path, {"real/a.txt": 10}, {}, "")
    (tmp_path / "link.txt").symlink_to("real/a.txt")
    (tmp_path / "linkdir").symlink_to("real")
    _git(tmp_path, "add", "-A")
    assert _main(tmp_path, allow) == 1
    out = capsys.readouterr().out
    assert "link.txt: tracked symlink" in out and "linkdir: tracked symlink" in out


def test_xg10_the_index_is_measured_not_the_work_tree(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A large indexed file deleted from the checkout still counts; an untracked large file does not."""
    allow = _repo(tmp_path, {"docs/small.bin": 400}, {"docs/big.bin": 9000}, "")
    (tmp_path / "docs/big.bin").unlink()
    _file(tmp_path, "untracked.bin", 9000)
    assert _main(tmp_path, allow) == 1
    out = capsys.readouterr().out
    assert "docs/big.bin: 9000 bytes" in out and "untracked.bin" not in out


def test_xg10_the_policy_measurements_are_bound_to_the_freeze(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cap whose recorded measurement is not what the freeze measures fails, so a cap moves only by its headroom."""
    allow = _repo(tmp_path, {"src/tests/a.py": 300, "docs/x.bin": 200}, {}, "")
    assert _main(tmp_path, allow) == 0
    assert _main(tmp_path, allow, code_file_max_bytes=10_000_000) == 1
    assert "code_file_max_bytes: records 10000000 as measured at the freeze" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("frozen", "content", "message"),
    [
        (False, "# senselab subprocess-venv lock: x\n# requirements-sha256: y\nz==1\n", "not in"),
        (True, "payload\n" * 100, "is not a venv-lock"),
    ],
)
def test_xg10_an_exemption_must_be_a_frozen_lock(
    frozen: bool, content: str, message: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An exempt path must exist in the freeze and parse as its lock kind."""
    rel = "src/senselab/utils/data/venv_locks/a.txt"
    allow = _repo(tmp_path, {rel: 10} if frozen else {"other.txt": 10}, {}, "")
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_text(content)
    _git(tmp_path, "add", "-A")
    assert budget.main(tmp_path, allow, _policy(tmp_path, generated={rel: "venv-lock"})) == 1
    assert message in capsys.readouterr().out


@pytest.mark.parametrize(
    ("kind", "content", "ok"),
    [
        ("uv-lock", 'version = 1\n[[package]]\nname = "a"\n', True),
        ("uv-lock", "version = 1\n", False),
        ("uv-lock", "not toml [", False),
        ("venv-lock", "# senselab subprocess-venv lock: a\n# requirements-sha256: b\nc==1\n", True),
        ("venv-lock", "a==1\nb==2\nc==3\n", False),
    ],
)
def test_lock_kinds_are_recognised_by_content(kind: str, content: str, ok: bool) -> None:
    """A uv lock is TOML with a version and packages; a venv lock starts with its two header lines."""
    assert budget.is_lock(content.encode(), kind) is ok


def test_policy_caps_are_measured_quantities_with_a_headroom(tmp_path: Path) -> None:
    """Each cap is `{measured: <int >= 0>, headroom: <number >= 1>}`; unknown keys fail; the shipped one loads."""
    policy = budget.load_policy()
    assert policy.caps["code_total_max_bytes"].limit > policy.caps["code_file_max_bytes"].limit
    data = yaml.safe_load(budget.POLICY.read_text())
    path = tmp_path / "p.yaml"
    for bad in (-1, True, 1.5, "x", None):
        path.write_text(yaml.safe_dump({**data, "code_total_max_bytes": {"measured": bad, "headroom": 1}}))
        with pytest.raises(ValueError, match="code_total_max_bytes"):
            budget.load_policy(path)
    for bad in (0.5, "2", True, None):
        path.write_text(yaml.safe_dump({**data, "code_total_max_bytes": {"measured": 10, "headroom": bad}}))
        with pytest.raises(ValueError, match="code_total_max_bytes"):
            budget.load_policy(path)
    path.write_text(yaml.safe_dump({**data, "code_total_max_bytes": 12_000_000}))
    with pytest.raises(ValueError, match="code_total_max_bytes"):
        budget.load_policy(path)
    path.write_text(yaml.safe_dump({**data, "extra_cap": 1}))
    with pytest.raises(ValueError, match="unknown"):
        budget.load_policy(path)


_PLACES = ["src/tests/d", "packages/senselab-core/tests", "src/senselab/data", "docs", "specs/x", ""]
_SUFFIXES = [".py", ".pyi", ".ipynb", ".mjs", ".json", ".wav", ".txt", ".lock", ""]
_MODES = ["100644", "100755", "120000", "160000"]


@settings(max_examples=200, deadline=None)
@given(
    st.lists(
        st.tuples(st.sampled_from(_PLACES), st.sampled_from(_SUFFIXES), st.sampled_from(_MODES), st.integers(0, 1200)),
        max_size=12,
    )
)
def test_xg10_every_tracked_object_is_counted(rows: list[tuple[str, str, str, int]]) -> None:
    """Whatever the place, suffix and mode, an object is refused when non-regular, over its cap or over a total."""
    objects = [Obj(f"{p}/f{i}{s}".lstrip("/"), mode, str(i), size) for i, (p, s, mode, size) in enumerate(rows)]
    regular = [o for o in objects if o.mode in ("100644", "100755")]
    code = [o for o in regular if o.path.endswith((".py", ".pyi", ".ipynb"))]
    data = [o for o in regular if o not in code]
    tests = [o for o in data if o.path.startswith(("src/tests/", "packages/senselab-core/tests/"))]
    passes = (
        len(regular) == len(objects)
        and all(o.size <= LIMITS["code_file_max_bytes"] for o in code)
        and all(o.size <= LIMITS["data_file_max_bytes"] for o in data)
        and sum(o.size for o in code) <= LIMITS["code_total_max_bytes"]
        and sum(o.size for o in data) <= LIMITS["data_total_max_bytes"]
        and sum(o.size for o in tests) <= LIMITS["test_data_total_max_bytes"]
    )
    assert (budget.check(objects, allow={}, limits=LIMITS) == []) is passes


def test_repository_tracks_no_symlink_or_submodule() -> None:
    """Nothing in this repository's index has mode 120000 or 160000."""
    modes = {o.mode for o in budget.indexed_objects(budget.ROOT)}
    assert modes <= budget.REGULAR_MODES


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
    policy = tmp_path / "policy.yaml"
    policy.write_text(budget.POLICY.read_text())
    assert budget.main(tmp_path, allow, policy) == 1 and f"{name}: not in origin/{FREEZE}" in capsys.readouterr().out


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
    tracked = {o.path for o in budget.indexed_objects(budget.ROOT)}
    assert allow and set(allow) <= tracked


def test_budget_passes_on_current_tree() -> None:
    """The script exits 0 on the committed tree, which also binds the allowlist and the policy to the freeze branch."""
    try:
        budget.resolve_freeze_ref(budget.ROOT)
    except LookupError:
        pytest.skip("freeze branch not fetched here; the pre-commit job runs the script with it")
    done = subprocess.run([sys.executable, str(_SCRIPT)], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout
