"""The limits selector: marked tests and their dependencies come from the tree, the phase branch from the remote."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.logic

REPO = Path(__file__).resolve().parents[4]
SRC = "packages/senselab-core/src/senselab_core"
TESTS = "packages/senselab-core/tests"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("limits_scope", REPO / "scripts" / "limits_scope.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["limits_scope"] = module
    spec.loader.exec_module(module)
    return module


ls = _load()


def _write(root: Path, rel: str, text: str = "") -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text)


def _tree(root: Path) -> None:
    """A decorated limits test, a module-marked one with a test-side helper, an `@mark.limits` one and decoys."""
    _write(root, f"{SRC}/__init__.py")
    for module in ("fast", "slow", "other", "deep", "pkg/__init__", "pkg/inner"):
        _write(root, f"{SRC}/{module}.py")
    _write(root, f"{SRC}/slow.py", "from senselab_core import deep\n")
    _write(root, "packages/senselab-worker/src/senselab_worker/wire.py")
    _write(
        root,
        f"{TESTS}/a_test.py",
        "import pytest\nfrom senselab_core import fast\n@pytest.mark.limits\ndef test(): ...\n",
    )
    _write(
        root,
        f"{TESTS}/sub/b_test.py",
        "import pytest\nimport senselab_core.pkg\nfrom senselab_core.pkg import inner\nimport helper_fixture\n"
        "from senselab_worker import wire\npytestmark = [pytest.mark.limits]\n",
    )
    _write(root, f"{TESTS}/sub/helper_fixture.py", "from senselab_core import slow\nfrom . import sibling_fixture\n")
    _write(root, f"{TESTS}/sub/sibling_fixture.py")
    _write(root, f"{TESTS}/d_test.py", "from pytest import mark\n@mark.limits(1)\ndef test(): ...\n")
    _write(
        root,
        f"{TESTS}/c_test.py",
        "import pytest\nfrom senselab_core import other\nTEXT = 'pytest.mark.limits'\n"
        "@pytest.mark.logic\ndef test(): ...\n",
    )
    _write(root, f"{TESTS}/conftest.py")
    _write(root, f"{TESTS}/data/profile.yaml")


def test_the_limits_tests_are_found_by_syntax_not_text(tmp_path: Path) -> None:
    """A decorator, a module-level mark and `@mark.limits` select a file; a string mentioning the mark does not."""
    _tree(tmp_path)
    assert ls.limits_tests(tmp_path) == [f"{TESTS}/a_test.py", f"{TESTS}/d_test.py", f"{TESTS}/sub/b_test.py"]


def test_the_subjects_follow_imports_to_source_and_test_side_modules_transitively(tmp_path: Path) -> None:
    """Source modules, a top-level test helper, relative imports and the imports of each resolve."""
    _tree(tmp_path)
    assert ls.subjects(tmp_path / TESTS / "a_test.py", tmp_path) == {f"{SRC}/__init__.py", f"{SRC}/fast.py"}
    assert ls.subjects(tmp_path / TESTS / "sub" / "b_test.py", tmp_path) == {
        f"{SRC}/__init__.py",
        f"{SRC}/pkg/__init__.py",
        f"{SRC}/pkg/inner.py",
        f"{SRC}/slow.py",
        f"{SRC}/deep.py",
        f"{TESTS}/sub/helper_fixture.py",
        f"{TESTS}/sub/sibling_fixture.py",
        "packages/senselab-worker/src/senselab_worker/wire.py",
    }
    assert f"{SRC}/other.py" not in ls.scope(tmp_path)


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ([f"{TESTS}/a_test.py"], True),
        ([f"{SRC}/fast.py"], True),
        ([f"{SRC}/deep.py"], True),
        ([f"{TESTS}/sub/helper_fixture.py"], True),
        ([f"{TESTS}/conftest.py"], True),
        ([f"{TESTS}/data/profile.yaml"], True),
        (["scripts/limits_scope.py"], True),
        ([f"{SRC}/other.py", f"{TESTS}/c_test.py", "README.md"], False),
        ([], False),
    ],
)
def test_a_change_is_in_scope_when_it_touches_a_marked_test_or_a_dependency(
    tmp_path: Path, changed: list[str], expected: bool
) -> None:
    """A marked test, anything it imports through any depth, a conftest or the data selects; other files do not."""
    _tree(tmp_path)
    assert ls.affected(changed, ls.scope(tmp_path)) is expected


def test_the_real_tree_selects_the_files_the_old_scope_missed() -> None:
    """A fixture-only change and the pipe module the fixture imports are in scope, like every marked test."""
    marked = ls.limits_tests(REPO)
    assert marked and f"{TESTS}/runtime/fetch_test.py" in marked
    covered = ls.scope(REPO)
    assert set(marked) <= covered
    assert ls.affected([f"{TESTS}/runtime/shutdown_schedule_fixture.py"], covered)
    assert ls.affected([f"{SRC}/runtime/worker_pipes.py"], covered)
    assert ls.affected([f"{TESTS}/data/test_slack.yaml"], covered)


LS_REMOTE = "\n".join(
    f"{'a' * 40}\trefs/heads/{name}"
    for name in (
        "2x/phase-3",
        "2x/phase-10",
        "2x/phase-9-spec",
        "2x/phase-12-spec",
        "2x/phase-2",
        "2x/phase-3-x",
        "2.x",
    )
)


def test_the_phase_branch_is_the_highest_phase_head_by_number() -> None:
    """Numeric order (10 beats 9 and 3), `-spec` and suffixed branches ignored."""
    assert ls.phase_branch_from(LS_REMOTE) == "2x/phase-10"
    assert ls.phase_branch_from(f"{'b' * 40}\trefs/heads/2x/phase-4\n") == "2x/phase-4"
    with pytest.raises(ValueError, match="phase"):
        ls.phase_branch_from(f"{'b' * 40}\trefs/heads/2x/phase-4-spec\n")


def _git(repo: Path, *args: str) -> str:
    cmd = ["git", "-C", str(repo), "-c", "user.name=user", "-c", "user.email=user@example.invalid", *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60).stdout.strip()


def _commit(repo: Path, rel: str, text: str | None = None) -> str:
    _write(repo, rel, rel if text is None else text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", rel)
    return _git(repo, "rev-parse", "HEAD")


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _tree(repo)
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "tree")
    return repo


def test_the_job_runs_on_the_phase_branch_always_and_elsewhere_on_a_change_in_scope(tmp_path: Path) -> None:
    """Every push to the phase branch runs; another branch runs on a scoped change, a missing base or a PR only."""
    repo = _repo(tmp_path)
    first = _commit(repo, "README.md")
    phase = "2x/phase-9"
    assert not ls.runs("push", "2x/other", first, repo, phase)
    second = _commit(repo, f"{TESTS}/sub/helper_fixture.py", "from senselab_core import slow\n")
    assert ls.runs("push", "2x/other", first, repo, phase)
    assert not ls.runs("push", "2x/other", second, repo, phase)
    assert ls.runs("push", phase, second, repo, phase)
    assert ls.runs("push", "2x/other", "0" * 40, repo, phase)
    assert ls.runs("pull_request", "2x/other", second, repo, phase)


def test_a_deleted_dependency_and_a_removed_mark_are_seen_through_the_scope_before_the_push(tmp_path: Path) -> None:
    """Deleting a subject, or dropping a test's mark, is a change in scope although HEAD no longer holds the file."""
    repo = _repo(tmp_path)
    base = _git(repo, "rev-parse", "HEAD")
    _git(repo, "rm", "-q", f"{SRC}/fast.py")
    _git(repo, "commit", "-q", "-m", "delete the subject")
    assert ls.runs("push", "2x/other", base, repo, "2x/phase-9")
    middle = _git(repo, "rev-parse", "HEAD")
    _commit(repo, f"{TESTS}/d_test.py", "def test(): ...\n")
    assert ls.runs("push", "2x/other", middle, repo, "2x/phase-9")


def test_the_command_line_reads_the_phase_branch_from_the_remote(tmp_path: Path) -> None:
    """`phase-branch`, `refs` and `scope` use the highest phase head of `origin`, including one cut after 2.x's tree."""
    repo = _repo(tmp_path)
    remote = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True, timeout=60)
    _git(repo, "remote", "add", "origin", str(remote))
    for name in ("2x/phase-9", "2x/phase-10", "2x/phase-11-spec"):
        _git(repo, "push", "-q", "origin", f"HEAD:refs/heads/{name}")
    script = str(REPO / "scripts" / "limits_scope.py")
    env = {**os.environ, "GITHUB_EVENT_NAME": "push", "GITHUB_REF_NAME": "2x/phase-10", "BEFORE": "0" * 40}

    def run(*args: str) -> str:
        done = subprocess.run(
            [sys.executable, script, *args, "--root", str(repo)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return done.stdout.strip()

    assert run("phase-branch") == "2x/phase-10"
    assert json.loads(run("refs", "2.x")) == ["2.x", "2x/phase-10"]
    assert run("scope") == "run=true"


def test_the_script_does_not_read_state_md() -> None:
    """The phase branch has one source, the remote's branches; STATE.md lags on 2.x by a phase."""
    assert "STATE" not in (REPO / "scripts" / "limits_scope.py").read_text()
