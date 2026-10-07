"""Tests for pinned checkouts: bare mirrors, cleared outputs, verified patches, detached worktrees.

Every repository here is built in ``tmp_path``; nothing reaches a network.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from senselab.utils.reproduce import source
from senselab.utils.reproduce.locks import ReproduceLockError
from senselab.utils.reproduce.source import ReproduceSourceError
from senselab.utils.reproduce.spec import FileRef, SourceSpec

FIXTURE_ENV = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_AUTHOR_NAME": "fixture",
    "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
    "GIT_COMMITTER_NAME": "fixture",
    "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
    "GIT_AUTHOR_DATE": "@1000000000 +0000",
    "GIT_COMMITTER_DATE": "@1000000000 +0000",
}

NOTEBOOK = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# title\n"]},
        {
            "cell_type": "code",
            "execution_count": 7,
            "metadata": {},
            "outputs": [{"output_type": "stream", "name": "stdout", "text": ["committed output\n"]}],
            "source": ["print('committed output')\n"],
        },
    ],
    "metadata": {},
    "nbformat": 4,
    "nbformat_minor": 5,
}

PATCH_A = "--- a/run.py\n+++ b/run.py\n@@ -1 +1 @@\n-print(1)\n+print(2)\n"
PATCH_B = "--- a/run.py\n+++ b/run.py\n@@ -1 +1 @@\n-print(1)\n+print(3)\n"
PATCH_A2 = "--- a/run.py\n+++ b/run.py\n@@ -1 +1 @@\n-print(2)\n+print(4)\n"
PATCH_BAD = "--- a/run.py\n+++ b/run.py\n@@ -1 +1 @@\n-print(99)\n+print(2)\n"


def _git(cwd: Path, *args: str) -> str:
    env = {**FIXTURE_ENV, "HOME": str(cwd)}
    argv = ["git", "-c", "protocol.file.allow=always", "-c", "init.defaultBranch=main", "-C", str(cwd), *args]
    return subprocess.run(argv, env=env, check=True, capture_output=True, text=True).stdout.strip()


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def repos(tmp_path: Path) -> dict[str, object]:
    """An upstream with a notebook, a script and one submodule, as local paths."""
    sub = tmp_path / "upstream" / "lib"
    sub.mkdir(parents=True)
    _git(sub, "init", "-q")
    (sub / "lib.py").write_text("VALUE = 1\n")
    sub_sha = _commit_all(sub, "lib")

    top = tmp_path / "upstream" / "analysis"
    top.mkdir(parents=True)
    _git(top, "init", "-q")
    (top / "run.py").write_text("print(1)\n")
    (top / "nb.ipynb").write_text(json.dumps(NOTEBOOK, indent=1) + "\n")
    _git(top, "submodule", "add", "-q", str(sub), "ext/lib")
    commit = _commit_all(top, "analysis")
    return {"top": top, "sub": sub, "commit": commit, "sub_sha": sub_sha}


@pytest.fixture()
def manifest_dir(tmp_path: Path) -> Path:
    """A manifest repository holding the patch files."""
    base = tmp_path / "manifests"
    (base / "patches").mkdir(parents=True)
    for name, text in {"a.patch": PATCH_A, "b.patch": PATCH_B, "a2.patch": PATCH_A2, "bad.patch": PATCH_BAD}.items():
        (base / "patches" / name).write_text(text)
    return base


def _ref(base: Path, name: str) -> FileRef:
    data = (base / "patches" / name).read_bytes()
    return FileRef(file=f"patches/{name}", sha256=hashlib.sha256(data).hexdigest())


def _url(repos: dict[str, object]) -> str:
    return str(repos["top"])


def test_mirror_is_bare_and_mirrors_submodules(repos: dict[str, object], tmp_path: Path) -> None:
    """The repository and every submodule are mirrored bare under ``<cache>/src``."""
    cache = tmp_path / "cache"
    mirror = source.mirror(_url(repos), str(repos["commit"]), cache=cache)
    assert mirror == cache / "src" / "analysis"
    assert _git(mirror, "rev-parse", "--is-bare-repository") == "true"
    sub_mirror = cache / "src" / "lib"
    assert _git(sub_mirror, "rev-parse", "--is-bare-repository") == "true"
    assert _git(sub_mirror, "cat-file", "-t", str(repos["sub_sha"])) == "commit"


def test_checkout_returns_submodule_shas(repos: dict[str, object], tmp_path: Path) -> None:
    """The submodule is checked out from its mirror at the recorded gitlink."""
    co = source.checkout(_url(repos), str(repos["commit"]), [], cache=tmp_path / "cache")
    assert co.submodules == {"ext/lib": repos["sub_sha"]}
    assert (co.worktree / "ext" / "lib" / "lib.py").read_text() == "VALUE = 1\n"
    assert _git(co.worktree / "ext" / "lib", "rev-parse", "HEAD") == repos["sub_sha"]
    assert co.worktree.name == f"analysis-{str(repos['commit'])[:8]}-{source.EMPTY_PATCH_SET_SHA256[:8]}"
    assert co.worktree.parent == tmp_path / "cache" / "wt"


def test_clearing_outputs_is_a_commit(repos: dict[str, object], tmp_path: Path) -> None:
    """Outputs are emptied and execution counts nulled, committed as ``repro: clear outputs``."""
    co = source.checkout(_url(repos), str(repos["commit"]), [], cache=tmp_path / "cache")
    nb = json.loads((co.worktree / "nb.ipynb").read_text())
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(c["outputs"] == [] and c["execution_count"] is None for c in code)
    assert code[0]["source"] == ["print('committed output')\n"]
    assert _git(co.worktree, "log", "-1", "--format=%s") == "repro: clear outputs"
    assert _git(co.worktree, "rev-parse", "HEAD^") == repos["commit"]
    assert _git(co.worktree, "status", "--porcelain", "--untracked-files=all") == ""
    assert co.tree_sha == _git(co.worktree, "rev-parse", "HEAD^{tree}")


def test_clear_notebook_outputs_plain_json() -> None:
    """The JSON-level clear drops outputs and nulls counts, and reports whether anything changed."""
    cleared, changed = source.clear_notebook_outputs(json.dumps(NOTEBOOK))
    assert changed
    cell = json.loads(cleared)["cells"][1]
    assert cell["outputs"] == [] and cell["execution_count"] is None
    again, changed_again = source.clear_notebook_outputs(cleared)
    assert not changed_again and again == cleared


def test_unreadable_notebook_raises_with_code(tmp_path: Path) -> None:
    """A notebook that is not JSON stops the checkout with a fixed code."""
    with pytest.raises(ReproduceSourceError) as info:
        source.clear_notebook_outputs("{not json")
    assert info.value.code == source.NOTEBOOK_UNREADABLE


def test_patches_are_committed_in_order(repos: dict[str, object], manifest_dir: Path, tmp_path: Path) -> None:
    """Each patch becomes one ``repro: <name>`` commit on top of the clear commit."""
    refs = [_ref(manifest_dir, "a.patch"), _ref(manifest_dir, "a2.patch")]
    co = source.checkout(_url(repos), str(repos["commit"]), refs, base_dir=manifest_dir, cache=tmp_path / "cache")
    assert (co.worktree / "run.py").read_text() == "print(4)\n"
    log = _git(co.worktree, "log", "--first-parent", "--format=%s", "-3").splitlines()
    assert log == ["repro: a2.patch", "repro: a.patch", "repro: clear outputs"]
    assert _git(co.worktree, "rev-parse", "HEAD~3") == repos["commit"]
    assert co.patches == [(r.file, r.sha256) for r in refs]


def test_two_patch_sets_give_two_worktrees(repos: dict[str, object], manifest_dir: Path, tmp_path: Path) -> None:
    """Different patch sets on one commit never share a worktree."""
    cache = tmp_path / "cache"
    commit = str(repos["commit"])
    a = source.checkout(_url(repos), commit, [_ref(manifest_dir, "a.patch")], base_dir=manifest_dir, cache=cache)
    b = source.checkout(_url(repos), commit, [_ref(manifest_dir, "b.patch")], base_dir=manifest_dir, cache=cache)
    assert a.worktree != b.worktree and a.patch_set_sha256 != b.patch_set_sha256
    assert a.commit == b.commit == commit
    assert a.tree_sha != b.tree_sha
    assert (a.worktree / "run.py").read_text() == "print(2)\n"
    assert (b.worktree / "run.py").read_text() == "print(3)\n"


def test_patch_sha_mismatch_raises_before_apply(repos: dict[str, object], manifest_dir: Path, tmp_path: Path) -> None:
    """A patch whose bytes differ from the manifest sha256 stops before any worktree exists."""
    cache = tmp_path / "cache"
    ref = FileRef(file="patches/a.patch", sha256="0" * 64)
    with pytest.raises(ReproduceLockError):
        source.checkout(_url(repos), str(repos["commit"]), [ref], base_dir=manifest_dir, cache=cache)
    assert not (cache / "wt").exists() or not any((cache / "wt").iterdir())


def test_patch_that_does_not_apply_raises_with_code(
    repos: dict[str, object], manifest_dir: Path, tmp_path: Path
) -> None:
    """``git apply --check`` failing gives a fixed code, leaves no worktree and moves no ref."""
    cache = tmp_path / "cache"
    ref = _ref(manifest_dir, "bad.patch")
    with pytest.raises(ReproduceSourceError) as info:
        source.checkout(_url(repos), str(repos["commit"]), [ref], base_dir=manifest_dir, cache=cache)
    assert info.value.code == source.PATCH_DOES_NOT_APPLY
    assert "print(99)" not in str(info.value)
    tag = f"{str(repos['commit'])[:8]}-{source.patch_set_sha256([ref])[:8]}"
    assert not (cache / "wt" / f"analysis-{tag}").exists()
    assert _git(cache / "src" / "analysis", "for-each-ref", f"refs/heads/repro/{tag}") == ""


def test_rerun_is_idempotent_and_shas_are_reproducible(
    repos: dict[str, object], manifest_dir: Path, tmp_path: Path
) -> None:
    """A second checkout reuses the worktree; a fresh cache produces the same commit and tree shas."""
    refs = [_ref(manifest_dir, "a.patch")]
    commit = str(repos["commit"])
    first = source.checkout(_url(repos), commit, refs, base_dir=manifest_dir, cache=tmp_path / "c1")
    marker = first.worktree.with_name(first.worktree.name + ".json")
    inode = first.worktree.stat().st_ino
    second = source.checkout(_url(repos), commit, refs, base_dir=manifest_dir, cache=tmp_path / "c1")
    assert second == first and second.worktree.stat().st_ino == inode and marker.is_file()
    other = source.checkout(_url(repos), commit, refs, base_dir=manifest_dir, cache=tmp_path / "c2")
    assert (other.head, other.tree_sha) == (first.head, first.tree_sha)


def test_head_mismatch_raises(repos: dict[str, object], tmp_path: Path) -> None:
    """A worktree moved off its recorded head is refused, not silently reused."""
    cache = tmp_path / "cache"
    co = source.checkout(_url(repos), str(repos["commit"]), [], cache=cache)
    _git(co.worktree, "checkout", "-q", "--detach", str(repos["commit"]))
    with pytest.raises(ReproduceSourceError) as info:
        source.checkout(_url(repos), str(repos["commit"]), [], cache=cache)
    assert info.value.code == source.HEAD_MISMATCH


def test_first_parent_mismatch_raises(repos: dict[str, object], tmp_path: Path) -> None:
    """The first-parent chain must end at the pinned commit."""
    top = repos["top"]
    assert isinstance(top, Path)
    (top / "run.py").write_text("print(5)\n")
    later = _commit_all(top, "later")
    co = source.checkout(str(top), later, [], cache=tmp_path / "cache")
    with pytest.raises(ReproduceSourceError) as info:
        source.verify_checkout(co._replace(commit=str(repos["commit"])))
    assert info.value.code == source.CHAIN_MISMATCH


def test_dirty_worktree_raises(repos: dict[str, object], tmp_path: Path) -> None:
    """An edited worktree fails verification."""
    co = source.checkout(_url(repos), str(repos["commit"]), [], cache=tmp_path / "cache")
    (co.worktree / "run.py").write_text("print('edited')\n")
    with pytest.raises(ReproduceSourceError) as info:
        source.verify_checkout(co)
    assert info.value.code == source.WORKTREE_DIRTY


def test_missing_commit_raises(repos: dict[str, object], tmp_path: Path) -> None:
    """A commit the upstream does not have gives a fixed code."""
    with pytest.raises(ReproduceSourceError) as info:
        source.checkout(_url(repos), "f" * 40, [], cache=tmp_path / "cache")
    assert info.value.code == source.COMMIT_MISSING


def test_repro_refs_exist_only_in_the_mirror(repos: dict[str, object], manifest_dir: Path, tmp_path: Path) -> None:
    """The upstream never gains a ``repro/*`` ref; the mirror holds one per patch set."""
    cache = tmp_path / "cache"
    commit = str(repos["commit"])
    co = source.checkout(_url(repos), commit, [_ref(manifest_dir, "a.patch")], base_dir=manifest_dir, cache=cache)
    top = repos["top"]
    assert isinstance(top, Path)
    assert _git(top, "for-each-ref", "refs/heads/repro") == ""
    assert _git(cache / "src" / "analysis", "rev-parse", co.ref) == co.head
    assert co.ref == f"refs/heads/repro/{commit[:8]}-{co.patch_set_sha256[:8]}"
    detached = subprocess.run(
        ["git", "-C", str(co.worktree), "symbolic-ref", "-q", "HEAD"], env=FIXTURE_ENV, capture_output=True
    )
    assert detached.returncode != 0


def test_mirror_refresh_keeps_repro_refs(repos: dict[str, object], tmp_path: Path) -> None:
    """Fetching a newer upstream commit never prunes the throwaway refs."""
    cache = tmp_path / "cache"
    co = source.checkout(_url(repos), str(repos["commit"]), [], cache=cache)
    top = repos["top"]
    assert isinstance(top, Path)
    (top / "run.py").write_text("print(6)\n")
    later = _commit_all(top, "later")
    source.mirror(str(top), later, cache=cache)
    assert _git(cache / "src" / "analysis", "rev-parse", co.ref) == co.head


def test_authoring_worktree_is_detached_off_the_cleared_ref(repos: dict[str, object], tmp_path: Path) -> None:
    """The authoring copy is a separate detached worktree at the empty-patch-set ref."""
    cache = tmp_path / "cache"
    co = source.checkout(_url(repos), str(repos["commit"]), [], cache=cache)
    dest = tmp_path / "author" / "row"
    path = source.authoring_worktree(_url(repos), str(repos["commit"]), dest, cache=cache)
    assert path == dest and path != co.worktree
    assert _git(path, "rev-parse", "HEAD") == co.head
    detached = subprocess.run(["git", "-C", str(path), "symbolic-ref", "-q", "HEAD"], env=FIXTURE_ENV)
    assert detached.returncode != 0
    assert json.loads((path / "nb.ipynb").read_text())["cells"][1]["outputs"] == []


def test_authoring_worktree_needs_the_cleared_ref(repos: dict[str, object], tmp_path: Path) -> None:
    """Without a prefetch there is no cleared ref to author from."""
    cache = tmp_path / "cache"
    source.mirror(_url(repos), str(repos["commit"]), cache=cache)
    with pytest.raises(ReproduceSourceError) as info:
        source.authoring_worktree(_url(repos), str(repos["commit"]), tmp_path / "author", cache=cache)
    assert info.value.code == source.REF_MISSING


def test_sources_only_checkout(repos: dict[str, object], tmp_path: Path) -> None:
    """A draft source spec checks out with the empty patch set."""
    spec = SourceSpec(name="row", repo=_url(repos), commit=str(repos["commit"]), submodules=False)
    result = source.checkout_sources([spec], cache=tmp_path / "cache")
    assert list(result) == ["row"]
    assert result["row"].patch_set_sha256 == source.EMPTY_PATCH_SET_SHA256
    assert result["row"].submodules == {}


def test_caller_git_config_and_hooks_are_ignored(
    repos: dict[str, object], manifest_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """User hooks, identity and ``GIT_*`` variables never reach the mirror's commits."""
    home = tmp_path / "home"
    hooks = home / "hooks"
    hooks.mkdir(parents=True)
    for hook in ("pre-commit", "commit-msg", "post-checkout", "post-commit"):
        (hooks / hook).write_text("#!/bin/sh\nexit 1\n")
        (hooks / hook).chmod(0o755)
    (home / ".gitconfig").write_text(f"[core]\n\thooksPath = {hooks}\n[user]\n\tname = caller\n\temail = c@x.invalid\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "nowhere"))
    monkeypatch.setenv("GIT_AUTHOR_NAME", "caller")
    co = source.checkout(
        _url(repos), str(repos["commit"]), [_ref(manifest_dir, "a.patch")], base_dir=manifest_dir, cache=tmp_path / "c"
    )
    monkeypatch.delenv("GIT_DIR")
    ident = _git(co.worktree, "log", "-1", "--format=%an <%ae> %at|%cn <%ce> %ct")
    assert ident == source.REPRO_IDENTITY_LOG


def test_resolve_submodule_url() -> None:
    """Relative submodule URLs resolve against the superproject URL."""
    assert source.resolve_submodule_url("https://h/org/top.git", "../lib.git") == "https://h/org/lib.git"
    assert source.resolve_submodule_url("git@h:org/top.git", "../lib.git") == "git@h:org/lib.git"
    assert source.resolve_submodule_url("/srv/top", "./lib") == "/srv/top/lib"
    assert source.resolve_submodule_url("/srv/top", "/abs/lib") == "/abs/lib"


OUTPUTS_HUNK = """--- a/nb.ipynb
+++ b/nb.ipynb
@@ -10,6 +10,8 @@
    "execution_count": null,
    "metadata": {},
-   "outputs": [],
+   "outputs": [
+    {"name": "stdout", "output_type": "stream", "text": ["x\\n"]}
+   ],
    "source": [
     "print('x')\\n"
    ]
"""

MID_OUTPUTS_HUNK = """--- a/nb.ipynb
+++ b/nb.ipynb
@@ -20,3 +20,3 @@
       "text": [
-       "old line\\n"
+       "new line\\n"
       ]
@@ -30,3 +30,3 @@
     {
-     "output_type": "stream",
+     "output_type": "display_data",
     }
"""

NEW_CELL_HUNK = """--- a/nb.ipynb
+++ b/nb.ipynb
@@ -10,2 +10,8 @@
   {
+   "cell_type": "code",
+   "execution_count": null,
+   "metadata": {},
+   "outputs": [],
+   "source": ["x = 1\\n"]
+  },
+  {
    "cell_type": "code",
"""


HEADER_LOOKALIKE_HUNK = """--- a/nb.ipynb
+++ b/nb.ipynb
@@ -1,3 +1,3 @@
--- x
+++ y.py
    "outputs": [
-     "a"
+     "b"
"""


def test_lint_counts_hunk_lines_so_body_lines_are_not_headers() -> None:
    """Body lines shaped like file headers do not end the notebook hunk."""
    findings = source.lint_patch(HEADER_LOOKALIKE_HUNK, [])
    assert [(f.code, f.line) for f in findings] == [
        (source.LINT_NOTEBOOK_OUTPUTS, 7),
        (source.LINT_NOTEBOOK_OUTPUTS, 8),
    ]


def test_lint_rejects_hunks_inside_outputs() -> None:
    """A patch that adds or edits notebook outputs is refused."""
    findings = source.lint_patch(OUTPUTS_HUNK, [])
    assert findings and {f.code for f in findings} == {source.LINT_NOTEBOOK_OUTPUTS}
    assert {f.code for f in source.lint_patch(MID_OUTPUTS_HUNK, [])} == {source.LINT_NOTEBOOK_OUTPUTS}


def test_lint_accepts_a_new_cell_with_empty_outputs_and_plain_code() -> None:
    """Empty outputs in a new cell, and a patch to a script, pass."""
    assert source.lint_patch(NEW_CELL_HUNK, []) == []
    assert source.lint_patch(PATCH_A, []) == []


def test_lint_reports_deny_pattern_hits_without_content(tmp_path: Path) -> None:
    """Deny patterns come from a file of extended regexes; findings carry line numbers, not text."""
    patterns_file = tmp_path / "deny"
    patterns_file.write_text("# comment\n\nZZQ[0-9]{3}\n")
    patterns = source.load_deny_patterns(patterns_file)
    assert patterns == ["ZZQ[0-9]{3}"]
    patch = PATCH_A.replace("print(2)", "print('ZZQ123')")
    findings = source.lint_patch(patch, patterns)
    assert [(f.code, f.line, f.pattern) for f in findings] == [(source.LINT_DENY_PATTERN, 5, 0)]
    assert "ZZQ123" not in repr(findings)
    assert source.lint_patch(PATCH_A, patterns) == []


def test_checkout_lints_patches(repos: dict[str, object], manifest_dir: Path, tmp_path: Path) -> None:
    """A deny-pattern hit in a step patch stops the checkout before any worktree."""
    cache = tmp_path / "cache"
    with pytest.raises(ReproduceSourceError) as info:
        source.checkout(
            _url(repos),
            str(repos["commit"]),
            [_ref(manifest_dir, "a.patch")],
            base_dir=manifest_dir,
            cache=cache,
            deny_patterns=["print\\(2\\)"],
        )
    assert info.value.code == source.PATCH_LINT
    assert not (cache / "wt").exists() or not any((cache / "wt").iterdir())


COMPACT_OUTPUT_CELL = (
    '{"cells": [{"cell_type": "code", "execution_count": null, "metadata": {}, '
    '"outputs": [{"text": "x"}], "source": []}], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
)
QUOTED_NOTEBOOK_PATCH = (
    'diff --git "a/caf\\303\\251.ipynb" "b/caf\\303\\251.ipynb"\n'
    "new file mode 100644\n"
    "--- /dev/null\n"
    '+++ "b/caf\\303\\251.ipynb"\n'
    "@@ -0,0 +1 @@\n"
    f"+{COMPACT_OUTPUT_CELL}\n"
)
OUTPUTS_TEXT_ARRAY_HUNK = """--- a/nb.ipynb
+++ b/nb.ipynb
@@ -8,3 +8,3 @@
      "text": [
-      "committed output\\n"
+      "edited output\\n"
      ]
"""


def _write_patch(base: Path, rel: str, text: str) -> FileRef:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode())
    return FileRef(file=rel, sha256=hashlib.sha256(text.encode()).hexdigest())


def test_unquote_c_path() -> None:
    """Git's C-quoted paths decode to UTF-8; plain paths pass through."""
    assert source.unquote_c_path('"b/caf\\303\\251.ipynb"') == "b/caf\u00e9.ipynb"
    assert source.unquote_c_path("b/plain.ipynb") == "b/plain.ipynb"


def test_lint_sees_quoted_notebook_paths() -> None:
    """A quoted non-ASCII notebook path is still recognised as a notebook."""
    patch = QUOTED_NOTEBOOK_PATCH.replace('"outputs": [{"text"', '"outputs": [{"output_type": "stream", "text"')
    assert {f.code for f in source.lint_patch(patch, [])} == {source.LINT_NOTEBOOK_OUTPUTS}


def test_patch_leaving_notebook_outputs_is_refused_after_apply(repos: dict[str, object], tmp_path: Path) -> None:
    """Outputs that slip past the line lint (quoted path, compact JSON) are caught after apply."""
    base = tmp_path / "manifests"
    ref = _write_patch(base, "patches/quoted.patch", QUOTED_NOTEBOOK_PATCH)
    assert source.lint_patch(QUOTED_NOTEBOOK_PATCH, []) == []
    cache = tmp_path / "cache"
    with pytest.raises(ReproduceSourceError) as info:
        source.checkout(_url(repos), str(repos["commit"]), [ref], base_dir=base, cache=cache)
    assert info.value.code == source.PATCH_LINT
    tag = f"{str(repos['commit'])[:8]}-{source.patch_set_sha256([ref])[:8]}"
    assert not (cache / "wt" / f"analysis-{tag}").exists()
    assert _git(cache / "src" / "analysis", "for-each-ref", f"refs/heads/repro/{tag}") == ""


def test_hunk_inside_an_outputs_text_array_never_lands(repos: dict[str, object], tmp_path: Path) -> None:
    """A hunk written against committed outputs cannot apply to the cleared notebook."""
    base = tmp_path / "manifests"
    ref = _write_patch(base, "patches/text.patch", OUTPUTS_TEXT_ARRAY_HUNK)
    cache = tmp_path / "cache"
    with pytest.raises(ReproduceSourceError) as info:
        source.checkout(_url(repos), str(repos["commit"]), [ref], base_dir=base, cache=cache)
    assert info.value.code in (source.PATCH_DOES_NOT_APPLY, source.PATCH_LINT)
    assert not (cache / "wt").exists() or not [p for p in (cache / "wt").iterdir() if p.is_dir()]


def test_lint_rejects_symlinks() -> None:
    """Creating a symlink, or turning a file into one, is refused."""
    new_link = "diff --git a/l b/l\nnew file mode 120000\n--- /dev/null\n+++ b/l\n@@ -0,0 +1 @@\n+/etc\n"
    mode_change = "diff --git a/run.py b/run.py\nold mode 100644\nnew mode 120000\n"
    assert [f.code for f in source.lint_patch(new_link, [])] == [source.LINT_SYMLINK]
    assert [f.code for f in source.lint_patch(mode_change, [])] == [source.LINT_SYMLINK]


def test_same_patch_bytes_at_two_paths_are_two_patch_sets(repos: dict[str, object], tmp_path: Path) -> None:
    """``patches/a/x.patch`` and ``patches/b/x.patch`` with identical bytes do not collide."""
    base = tmp_path / "manifests"
    a = _write_patch(base, "patches/a/x.patch", PATCH_A)
    b = _write_patch(base, "patches/b/x.patch", PATCH_A)
    assert source.patch_set_sha256([a]) != source.patch_set_sha256([b])
    cache = tmp_path / "cache"
    first = source.checkout(_url(repos), str(repos["commit"]), [a], base_dir=base, cache=cache)
    second = source.checkout(_url(repos), str(repos["commit"]), [b], base_dir=base, cache=cache)
    assert first.worktree != second.worktree and first.tree_sha == second.tree_sha
    assert source.checkout(_url(repos), str(repos["commit"]), [a], base_dir=base, cache=cache) == first


def test_clear_drops_widget_state_and_attachments() -> None:
    """Notebook widget state and markdown attachments are removed with the outputs."""
    nb = json.loads(json.dumps(NOTEBOOK))
    nb["metadata"]["widgets"] = {"state": {}}
    nb["cells"][0]["attachments"] = {"img.png": {"image/png": "AAAA"}}
    cleared = json.loads(source.clear_notebook_outputs(json.dumps(nb))[0])
    assert "widgets" not in cleared["metadata"] and "attachments" not in cleared["cells"][0]


@pytest.fixture()
def repos_with_notebook_submodule(tmp_path: Path) -> dict[str, object]:
    """An upstream whose submodule tracks a notebook with outputs."""
    sub = tmp_path / "upstream" / "nblib"
    sub.mkdir(parents=True)
    _git(sub, "init", "-q")
    (sub / "inner").mkdir()
    (sub / "inner" / "sub.ipynb").write_text(json.dumps(NOTEBOOK, indent=1) + "\n")
    _commit_all(sub, "nblib")
    top = tmp_path / "upstream" / "outer"
    top.mkdir(parents=True)
    _git(top, "init", "-q")
    (top / "run.py").write_text("print(1)\n")
    _git(top, "submodule", "add", "-q", str(sub), "ext/nblib")
    return {"top": top, "commit": _commit_all(top, "outer")}


def test_submodule_notebook_outputs_fail_closed(
    repos_with_notebook_submodule: dict[str, object], tmp_path: Path
) -> None:
    """A submodule notebook with outputs is refused unless allowed, and then recorded."""
    url, commit = str(repos_with_notebook_submodule["top"]), str(repos_with_notebook_submodule["commit"])
    cache = tmp_path / "cache"
    with pytest.raises(ReproduceSourceError) as info:
        source.checkout(url, commit, [], cache=cache)
    assert info.value.code == source.SUBMODULE_NOTEBOOK_OUTPUTS
    assert "ext/nblib/inner/sub.ipynb" in info.value.detail
    assert not (cache / "wt" / f"outer-{commit[:8]}-{source.EMPTY_PATCH_SET_SHA256[:8]}").exists()
    co = source.checkout(url, commit, [], cache=cache, allow_submodule_notebook_outputs=True)
    assert co.submodule_notebooks_with_outputs == ["ext/nblib/inner/sub.ipynb"]
    with pytest.raises(ReproduceSourceError) as reused:
        source.checkout(url, commit, [], cache=cache)
    assert reused.value.code == source.SUBMODULE_NOTEBOOK_OUTPUTS
