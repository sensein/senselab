"""The limits selector: the marked tests and their subjects come from the tree, the phase branch from STATE.md."""

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
STATE = "specs/20261030-140022-senselab-2x-api/STATE.md"


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


def _tree(root: Path, branch: str = "2x/phase-9") -> None:
    """A tree with a decorated limits test, a module-marked one, an unmarked one and their subjects."""
    _write(root, f"{SRC}/__init__.py")
    for module in ("fast", "slow", "other", "pkg/__init__", "pkg/inner"):
        _write(root, f"{SRC}/{module}.py")
    _write(root, "packages/senselab-worker/src/senselab_worker/wire.py")
    _write(
        root,
        f"{TESTS}/a_test.py",
        "import pytest\nfrom senselab_core import fast\n@pytest.mark.limits\ndef test(): ...\n",
    )
    _write(
        root,
        f"{TESTS}/sub/b_test.py",
        "import pytest\nimport senselab_core.pkg\nfrom senselab_core.pkg import inner\n"
        "from senselab_worker import wire\npytestmark = [pytest.mark.limits]\n",
    )
    _write(
        root,
        f"{TESTS}/c_test.py",
        "import pytest\nfrom senselab_core import other\n@pytest.mark.logic\ndef test(): ...\n",
    )
    _write(root, f"{TESTS}/conftest.py")
    _write(root, f"{TESTS}/data/profile.yaml")
    _write(
        root,
        STATE,
        f"| What | Where |\n|---|---|\n| Code (current) | `{branch}`, cut from `2x/phase-2` |\n"
        "| Phase 1 | `2x/phase-1` |\n",
    )


def test_the_limits_tests_are_the_marked_files_in_the_tree(tmp_path: Path) -> None:
    """A decorator and a module-level mark select a file; a test of another tier does not."""
    _tree(tmp_path)
    assert ls.limits_tests(tmp_path) == [f"{TESTS}/a_test.py", f"{TESTS}/sub/b_test.py"]


def test_the_subjects_are_the_source_files_a_test_imports(tmp_path: Path) -> None:
    """Modules, the packages imported through, submodules from-imported and other distributions' modules resolve."""
    _tree(tmp_path)
    assert ls.subjects(tmp_path / TESTS / "a_test.py", tmp_path) == {f"{SRC}/__init__.py", f"{SRC}/fast.py"}
    assert ls.subjects(tmp_path / TESTS / "sub" / "b_test.py", tmp_path) == {
        f"{SRC}/__init__.py",
        f"{SRC}/pkg/__init__.py",
        f"{SRC}/pkg/inner.py",
        "packages/senselab-worker/src/senselab_worker/wire.py",
    }
    assert f"{SRC}/other.py" not in ls.scope(tmp_path)


@pytest.mark.parametrize(
    ("changed", "expected"),
    [
        ([f"{TESTS}/a_test.py"], True),
        ([f"{SRC}/fast.py"], True),
        ([f"{SRC}/pkg/inner.py"], True),
        ([f"{TESTS}/conftest.py"], True),
        ([f"{TESTS}/data/profile.yaml"], True),
        (["scripts/limits_scope.py"], True),
        ([f"{SRC}/other.py", f"{TESTS}/c_test.py", "README.md"], False),
        ([], False),
    ],
)
def test_a_change_is_in_scope_when_it_touches_a_marked_test_or_its_subject(
    tmp_path: Path, changed: list[str], expected: bool
) -> None:
    """A marked test, a module it imports, a conftest, the data or the selector selects; other files do not."""
    _tree(tmp_path)
    assert ls.affected(changed, tmp_path) is expected


def test_a_new_limits_test_extends_the_scope_with_no_list_edit(tmp_path: Path) -> None:
    """Marking one more file, as T-TIME-3's test did, puts it and its subject in scope."""
    _tree(tmp_path)
    assert not ls.affected([f"{SRC}/other.py"], tmp_path)
    _write(
        tmp_path,
        f"{TESTS}/c_test.py",
        "import pytest\nfrom senselab_core import other\n@pytest.mark.limits\ndef t(): ...\n",
    )
    assert ls.affected([f"{SRC}/other.py"], tmp_path)


def test_the_real_tree_selects_every_marked_file_and_what_it_imports() -> None:
    """Every core limits test is in the real scope, with its subject; the old hand list's files are covered."""
    marked = ls.limits_tests(REPO)
    assert marked and f"{TESTS}/runtime/fetch_test.py" in marked
    covered = ls.scope(REPO)
    assert set(marked) <= covered
    assert f"{SRC}/runtime/fetch.py" in covered
    assert ls.affected([f"{TESTS}/data/test_slack.yaml"], REPO)


def test_the_phase_branch_is_the_first_branch_of_the_code_row(tmp_path: Path) -> None:
    """The branch is read from the `Code (current)` row, not from a later row or a literal; a missing row raises."""
    _tree(tmp_path, branch="2x/phase-9")
    assert ls.phase_branch(tmp_path) == "2x/phase-9"
    assert ls.phase_branch(REPO).startswith("2x/")
    _write(tmp_path, STATE, "| What | Where |\n")
    with pytest.raises(ValueError, match="Code"):
        ls.phase_branch(tmp_path)


def _git(repo: Path, *args: str) -> str:
    cmd = ["git", "-C", str(repo), "-c", "user.name=user", "-c", "user.email=user@example.invalid", *args]
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60).stdout.strip()


def _commit(repo: Path, rel: str) -> str:
    _write(repo, rel, rel)
    _git(repo, "add", rel)
    _git(repo, "commit", "-q", "-m", rel)
    return _git(repo, "rev-parse", "HEAD")


def test_the_job_runs_on_the_phase_branch_always_and_elsewhere_on_a_change_in_scope(tmp_path: Path) -> None:
    """Every push to the phase branch runs; another branch runs on a scoped change, a missing base or a PR only."""
    _tree(tmp_path)
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "tree")
    first = _commit(tmp_path, "README.md")
    other = "2x/other"
    assert not ls.runs("push", other, first, tmp_path)
    second = _commit(tmp_path, f"{SRC}/fast.py")
    assert ls.runs("push", other, first, tmp_path)
    assert not ls.runs("push", other, second, tmp_path)
    assert ls.runs("push", "2x/phase-9", second, tmp_path)
    assert ls.runs("push", other, "0" * 40, tmp_path)
    assert ls.runs("pull_request", other, second, tmp_path)


def test_the_command_line_prints_the_answers_the_workflow_reads(tmp_path: Path) -> None:
    """`scope`, `phase-branch` and `refs` print what the workflow appends to its output and reads."""
    _tree(tmp_path)
    script = str(REPO / "scripts" / "limits_scope.py")
    env = {**os.environ, "GITHUB_EVENT_NAME": "push", "GITHUB_REF_NAME": "2x/phase-9", "BEFORE": "0" * 40}

    def run(*args: str) -> str:
        done = subprocess.run(
            [sys.executable, script, *args, "--root", str(tmp_path)],
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        return done.stdout.strip()

    assert run("scope") == "run=true"
    assert run("phase-branch") == "2x/phase-9"
    assert json.loads(run("refs", "2.x")) == ["2.x", "2x/phase-9"]
