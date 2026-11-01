"""Tests for `.github/scripts/docs-deploy.sh` against a local bare repository."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / ".github" / "scripts" / "docs-deploy.sh"

pytestmark = pytest.mark.skipif(shutil.which("git") is None or shutil.which("bash") is None, reason="needs git and bash")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        {
            "DOCS_DEPLOY_BACKOFF": "0",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            **extra,
        }
    )
    return env


@pytest.fixture()
def origin(tmp_path: Path) -> Path:
    """An empty bare remote standing in for the GitHub repository."""
    bare = tmp_path / "origin.git"
    _git(tmp_path, "init", "--quiet", "--bare", str(bare))
    return bare


def _clone(origin: Path, dest: Path) -> Path:
    _git(origin.parent, "init", "--quiet", str(dest))
    _git(dest, "remote", "add", "origin", str(origin))
    return dest


def _site(path: Path, marker: str) -> Path:
    (path / "sub").mkdir(parents=True)
    (path / "index.html").write_text(marker)
    (path / "sub" / "page.html").write_text(marker)
    (path / ".nojekyll").write_text("")
    return path


def _deploy(clone: Path, *args: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT), *args], cwd=clone, env=_env(**env), capture_output=True, text=True, check=False
    )


def _tree(origin: Path) -> set[str]:
    return set(_git(origin, "ls-tree", "-r", "--name-only", "docs").splitlines())


def _commit_count(origin: Path) -> int:
    return int(_git(origin, "rev-list", "--count", "docs"))


def _show(origin: Path, path: str) -> str:
    return _git(origin, "show", f"docs:{path}")


def test_preview_release_cleanup_round_trip(origin: Path, tmp_path: Path) -> None:
    """Each mode replaces only its own part of the tree and leaves one commit."""
    clone = _clone(origin, tmp_path / "clone")
    for pr in ("7", "8"):
        result = _deploy(clone, "preview", str(_site(tmp_path / f"pr{pr}", f"pr {pr}")), pr)
        assert result.returncode == 0, result.stderr
    result = _deploy(clone, "release", str(_site(tmp_path / "rel", "release")))
    assert result.returncode == 0, result.stderr
    assert _show(origin, "index.html") == "release"
    assert _show(origin, "pr-7/index.html") == "pr 7"
    assert _show(origin, "pr-8/sub/page.html") == "pr 8"
    assert _commit_count(origin) == 1

    result = _deploy(clone, "preview", str(_site(tmp_path / "pr7b", "pr 7 again")), "7")
    assert result.returncode == 0, result.stderr
    assert _show(origin, "pr-7/index.html") == "pr 7 again"
    assert _show(origin, "index.html") == "release"

    result = _deploy(clone, "cleanup", "8", "99")
    assert result.returncode == 0, result.stderr
    names = _tree(origin)
    assert not any(n.startswith("pr-7/") for n in names)
    assert "pr-8/index.html" in names
    assert "index.html" in names
    assert _commit_count(origin) == 1


def test_release_replaces_old_site_files(origin: Path, tmp_path: Path) -> None:
    """A file the new release no longer builds is removed, previews are kept."""
    clone = _clone(origin, tmp_path / "clone")
    old = _site(tmp_path / "old", "old")
    (old / "gone.html").write_text("x")
    assert _deploy(clone, "release", str(old)).returncode == 0
    assert _deploy(clone, "preview", str(_site(tmp_path / "pr", "pr")), "3").returncode == 0
    assert _deploy(clone, "release", str(_site(tmp_path / "new", "new"))).returncode == 0
    names = _tree(origin)
    assert "gone.html" not in names
    assert "pr-3/index.html" in names


def test_concurrent_previews_both_survive(origin: Path, tmp_path: Path) -> None:
    """Two deploys that read the same tip: one loses the lease, retries, and both previews survive."""
    seed = _clone(origin, tmp_path / "seed")
    assert _deploy(seed, "release", str(_site(tmp_path / "rel", "release"))).returncode == 0
    sync = tmp_path / "sync"
    sync.mkdir()
    # On the first attempt each deployer announces it has built its tree on the shared tip, then
    # waits for the other, so both push against the same lease.
    hook = (
        'if [ "$DOCS_DEPLOY_ATTEMPT" = 1 ]; then touch "$SYNC/$ME"; '
        'for _ in $(seq 200); do [ -e "$SYNC/$OTHER" ] && break; sleep 0.05; done; fi'
    )
    procs = []
    for me, other in (("1", "2"), ("2", "1")):
        clone = _clone(origin, tmp_path / f"clone{me}")
        site = _site(tmp_path / f"site{me}", f"pr {me}")
        procs.append(
            subprocess.Popen(
                ["bash", str(SCRIPT), "preview", str(site), me],
                cwd=clone,
                env=_env(DOCS_DEPLOY_BEFORE_PUSH=hook, SYNC=str(sync), ME=me, OTHER=other),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    outputs = [p.communicate(timeout=60) for p in procs]
    assert [p.returncode for p in procs] == [0, 0], outputs
    assert any("retrying" in err for _, err in outputs), outputs
    assert _show(origin, "pr-1/index.html") == "pr 1"
    assert _show(origin, "pr-2/index.html") == "pr 2"
    assert _show(origin, "index.html") == "release"
    assert _commit_count(origin) == 1


def test_many_concurrent_deploys(origin: Path, tmp_path: Path) -> None:
    """Six previews and a release racing freely all land in one commit."""
    procs = []
    for i in range(1, 7):
        clone = _clone(origin, tmp_path / f"clone{i}")
        site = _site(tmp_path / f"site{i}", f"pr {i}")
        procs.append(
            subprocess.Popen(
                ["bash", str(SCRIPT), "preview", str(site), str(i)],
                cwd=clone,
                env=_env(DOCS_DEPLOY_ATTEMPTS="20"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        )
    rel_clone = _clone(origin, tmp_path / "rel_clone")
    procs.append(
        subprocess.Popen(
            ["bash", str(SCRIPT), "release", str(_site(tmp_path / "rel", "release"))],
            cwd=rel_clone,
            env=_env(DOCS_DEPLOY_ATTEMPTS="20"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    )
    outputs = [p.communicate(timeout=120) for p in procs]
    assert all(p.returncode == 0 for p in procs), outputs
    for i in range(1, 7):
        assert _show(origin, f"pr-{i}/index.html") == f"pr {i}"
    assert _show(origin, "index.html") == "release"
    assert _commit_count(origin) == 1


def test_push_failure_other_than_lease_is_not_retried(origin: Path, tmp_path: Path) -> None:
    """A rejected push that is not a moved branch fails at once instead of retrying."""
    clone = _clone(origin, tmp_path / "clone")
    hooks = origin / "hooks"
    hooks.mkdir(exist_ok=True)
    pre_receive = hooks / "pre-receive"
    pre_receive.write_text("#!/bin/sh\necho denied >&2\nexit 1\n")
    pre_receive.chmod(0o755)
    result = _deploy(clone, "preview", str(_site(tmp_path / "pr", "pr")), "5")
    assert result.returncode != 0
    assert "retrying" not in result.stderr


def test_namespaces_are_cleaned_separately(origin: Path, tmp_path: Path) -> None:
    """A root cleanup never removes `2x/` previews, and a `2x` cleanup never removes root previews."""
    clone = _clone(origin, tmp_path / "clone")
    assert _deploy(clone, "preview", str(_site(tmp_path / "root7", "root 7")), "7").returncode == 0
    for pr in ("7", "9"):
        result = _deploy(clone, "--namespace", "2x", "preview", str(_site(tmp_path / f"ns{pr}", f"2x {pr}")), pr)
        assert result.returncode == 0, result.stderr
    assert _show(origin, "2x/pr-7/index.html") == "2x 7"
    assert _show(origin, "pr-7/index.html") == "root 7"

    assert _deploy(clone, "cleanup").returncode == 0
    names = _tree(origin)
    assert not any(n.startswith("pr-7/") for n in names)
    assert {"2x/pr-7/index.html", "2x/pr-9/index.html"} <= names

    assert _deploy(clone, "preview", str(_site(tmp_path / "root8", "root 8")), "8").returncode == 0
    assert _deploy(clone, "--namespace", "2x", "cleanup", "9").returncode == 0
    names = _tree(origin)
    assert not any(n.startswith("2x/pr-7/") for n in names)
    assert {"2x/pr-9/index.html", "pr-8/index.html"} <= names
    assert _commit_count(origin) == 1


def test_release_keeps_namespace_previews(origin: Path, tmp_path: Path) -> None:
    """A release replaces the root site and keeps every namespace directory, even one the site also builds."""
    clone = _clone(origin, tmp_path / "clone")
    assert _deploy(clone, "--namespace", "2x", "preview", str(_site(tmp_path / "ns", "2x")), "4").returncode == 0
    rel = _site(tmp_path / "rel", "release")
    (rel / "2x").mkdir()
    (rel / "2x" / "stray.html").write_text("x")
    assert _deploy(clone, "release", str(rel)).returncode == 0
    names = _tree(origin)
    assert "2x/pr-4/index.html" in names
    assert "2x/stray.html" not in names
    assert _show(origin, "index.html") == "release"


def test_unknown_namespace_and_namespaced_release_are_refused(origin: Path, tmp_path: Path) -> None:
    """Only listed namespaces are accepted, and a release takes none."""
    clone = _clone(origin, tmp_path / "clone")
    assert _deploy(clone, "--namespace", "zz", "cleanup").returncode != 0
    assert _deploy(clone, "--namespace", "2x", "release", str(_site(tmp_path / "rel", "r"))).returncode != 0
