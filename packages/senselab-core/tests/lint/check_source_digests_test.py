"""The source-digest check: a task or backend whose source changed needs a new declared version."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "check_source_digests.py"
_spec = importlib.util.spec_from_file_location("check_source_digests", _SCRIPT)
assert _spec and _spec.loader
digests = importlib.util.module_from_spec(_spec)
sys.modules["check_source_digests"] = digests
_spec.loader.exec_module(digests)

MODULE = "packages/p/src/p/m.py"
WORKER = "packages/p/src/p/w.py"
PARAMS = "packages/p/src/p/q.py"
SOURCE = '''
WORKER_MODULE = "p.w"


@task(name="t", version="{task}", kind="transform")
class T:
    """A task."""

    inputs = {{}}


@backend(task="t", name="b", version="{backend}", worker=WORKER_MODULE)
class B:
    """A backend."""
'''


def _files(task: str = "1", backend: str = "1", worker: str = "def handle():\n    return 1\n") -> dict[str, str]:
    return {MODULE: SOURCE.format(task=task, backend=backend), WORKER: worker}


def _flagged(before: dict[str, str], after: dict[str, str]) -> list[str]:
    flagged: list[str] = digests.changed_without_bump(digests.digests_of(after), digests.digests_of(before))
    return flagged


def test_source_edit_without_bump_fails() -> None:
    """An edit to the worker module or to the declaration, with the version unchanged, is named."""
    assert _flagged(_files(), _files(worker="def handle():\n    return 2\n")) == ["backend:t/b"]
    edited = _files()
    edited[MODULE] = edited[MODULE].replace("inputs = {}", "inputs = {'x': 1}")
    assert _flagged(_files(), edited) == ["task:t"]


def test_a_separately_defined_params_default_is_part_of_the_digest() -> None:
    """A params class named by `params=`, in the module or imported from another, is hashed with its backend."""
    declared = SOURCE.replace("worker=WORKER_MODULE)", "worker=WORKER_MODULE, params=BParams)")

    def files(default: int, imported: bool) -> dict[str, str]:
        params = f"class BParams:\n    n: int = {default}\n"
        if imported:
            return {MODULE: "from p.q import BParams\n" + declared.format(task="1", backend="1"), PARAMS: params}
        return {MODULE: params + declared.format(task="1", backend="1")}

    for imported in (False, True):
        assert _flagged(files(1, imported), files(2, imported)) == ["backend:t/b"]
        assert _flagged(files(1, imported), files(1, imported)) == []


def test_source_edit_with_bump_passes() -> None:
    """The same edits with the declaring version raised are not named."""
    assert _flagged(_files(), _files(backend="2", worker="def handle():\n    return 2\n")) == []
    edited = _files(task="2")
    edited[MODULE] = edited[MODULE].replace("inputs = {}", "inputs = {'x': 1}")
    assert _flagged(_files(), edited) == []


def test_a_docstring_a_comment_a_bump_and_a_new_declaration_change_nothing() -> None:
    """Docstrings, comments and the version argument are outside the digest, and a new key has no baseline."""
    edited = _files(task="2")
    edited[MODULE] = edited[MODULE].replace('"""A task."""', '"""Changed."""  # note')
    before = digests.digests_of(_files())
    after = digests.digests_of(edited)
    assert after["task:t"].digest == before["task:t"].digest
    assert digests.changed_without_bump(after, {}) == []


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, timeout=60)


def _commit(repo: Path, files: dict[str, str], message: str) -> None:
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    _git(repo, "add", ".")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-m", message)


def test_baseline_is_merge_base_before_first_release(tmp_path: Path) -> None:
    """With no released digest file the baseline is the tree where the branch left 2.x; the file wins once it exists."""
    _git(tmp_path, "init", "-b", "2.x")
    _commit(tmp_path, _files(), "base")
    _git(tmp_path, "checkout", "-b", "feature")
    _commit(tmp_path, {WORKER: "def handle():\n    return 2\n"}, "edit")
    _commit(tmp_path, {MODULE: _files(backend="2")[MODULE]}, "bump")
    released = tmp_path / "released.yaml"
    base, origin = digests.baseline(tmp_path, released)
    assert origin.startswith("merge-base") and base == digests.digests_of(_files())
    here = digests.digests_of(digests.sources_at(tmp_path, None))
    assert digests.changed_without_bump(here, base) == []
    released.write_text("format: source-digests@1\ndeclarations:\n  task:t: {version: '1', digest: abc}\n")
    base, origin = digests.baseline(tmp_path, released)
    assert origin == "released file" and base == {"task:t": digests.Declared("1", "abc")}
    released.write_text("format: other@1\ndeclarations: {}\n")
    with pytest.raises(ValueError, match="source-digests@1"):
        digests.baseline(tmp_path, released)


def test_no_line_branch_is_refused(tmp_path: Path) -> None:
    """Without a 2.x ref the check fails with a message naming it, instead of passing."""
    _git(tmp_path, "init", "-b", "other")
    _commit(tmp_path, _files(), "base")
    with pytest.raises(RuntimeError, match="2.x"):
        digests.merge_base(tmp_path)
