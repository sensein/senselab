"""One ruff: the version uv.lock pins is the one pyproject requires and the one pre-commit runs."""

import tomllib
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[4]
RUFF_HOOKS = {"ruff", "ruff-check", "ruff-format"}
LOCKED_ENTRY = "uv run --frozen --no-sync ruff "


def locked_version(lock: dict, name: str) -> str | None:
    """Return the version of `name` that a parsed uv.lock pins, or None if it pins none."""
    versions = {p["version"] for p in lock.get("package", []) if p.get("name") == name}
    return versions.pop() if len(versions) == 1 else None


def toolchain_divergences(pyproject: dict, lock: dict, precommit: dict) -> list[str]:
    """Return every way the ruff of pyproject, uv.lock and pre-commit can differ; empty when they are one."""
    found = []
    version = locked_version(lock, "ruff")
    if version is None:
        found.append("uv.lock does not pin exactly one ruff")
    requirements = [r for group in pyproject.get("dependency-groups", {}).values() for r in group if isinstance(r, str)]
    ruff = [
        r.replace(" ", "") for r in requirements if r.replace(" ", "").startswith(("ruff=", "ruff~", "ruff>", "ruff<"))
    ]
    if ruff != [f"ruff=={version}"]:
        found.append(f"pyproject requires {ruff or 'no ruff'}, uv.lock pins ruff {version}")
    for repo in precommit.get("repos", []):
        for hook in repo.get("hooks", []):
            if hook.get("id") not in RUFF_HOOKS and "ruff" not in str(hook.get("entry", "")):
                continue
            if repo.get("repo") != "local":
                found.append(f"hook {hook.get('id')} comes from {repo.get('repo')}, which pins its own ruff")
            elif not str(hook.get("entry", "")).startswith(LOCKED_ENTRY) or hook.get("language") != "system":
                found.append(f"hook {hook.get('id')} does not run the locked ruff: {hook.get('entry')!r}")
    return found


def _repo_files() -> tuple[dict, dict, dict]:
    return (
        tomllib.loads((REPO / "pyproject.toml").read_text()),
        tomllib.loads((REPO / "uv.lock").read_text()),
        yaml.safe_load((REPO / ".pre-commit-config.yaml").read_text()),
    )


@pytest.mark.logic
def test_pre_commit_and_the_workspace_run_one_ruff() -> None:
    """The pyproject pins ruff to the version uv.lock holds, and every ruff hook runs that locked tool."""
    assert toolchain_divergences(*_repo_files()) == []


def _remote(rev: str) -> dict:
    return {"repo": "https://github.com/astral-sh/ruff-pre-commit", "rev": rev, "hooks": [{"id": "ruff-format"}]}


@pytest.mark.contract
@pytest.mark.parametrize(
    "plant",
    [
        lambda py, lock, pc: pc["repos"].append(_remote("v0.14.5")),
        lambda py, lock, pc: pc["repos"][0]["hooks"][0].update(entry="ruff format"),
        lambda py, lock, pc: pc["repos"][0]["hooks"][0].update(language="python"),
        lambda py, lock, pc: py["dependency-groups"]["dev"].append("ruff~=0.3"),
        lambda py, lock, pc: py["dependency-groups"].update(dev=["ruff==0.14.5"]),
        lambda py, lock, pc: py["dependency-groups"].update(dev=[]),
        lambda py, lock, pc: lock["package"].append({"name": "ruff", "version": "0.14.5"}),
    ],
)
def test_a_planted_divergence_is_reported(plant: object) -> None:
    """A remote ruff hook, a hook off the locked tool, or a pyproject or lock that disagree, is each reported."""
    pyproject = {"dependency-groups": {"dev": ["ruff==0.15.11"]}}
    lock = {"package": [{"name": "ruff", "version": "0.15.11"}]}
    hook = {"id": "ruff-format", "entry": LOCKED_ENTRY + "format", "language": "system"}
    precommit = {"repos": [{"repo": "local", "hooks": [hook]}]}
    assert toolchain_divergences(pyproject, lock, precommit) == []
    plant(pyproject, lock, precommit)  # type: ignore[operator]
    assert toolchain_divergences(pyproject, lock, precommit) != []
