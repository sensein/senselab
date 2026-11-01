"""One ruff: the version uv.lock pins is the one pyproject requires and the one pre-commit runs."""

import fnmatch
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

REPO = Path(__file__).resolve().parents[4]
RUFF_HOOKS = {"ruff", "ruff-check", "ruff-format"}
LOCKED_ENTRY = "uv run --frozen --no-sync ruff "


def locked_version(lock: dict[str, Any], name: str) -> str | None:
    """Return the version of `name` that a parsed uv.lock pins, or None if it pins none."""
    versions = {p["version"] for p in lock.get("package", []) if p.get("name") == name}
    return versions.pop() if len(versions) == 1 else None


def toolchain_divergences(pyproject: dict[str, Any], lock: dict[str, Any], precommit: dict[str, Any]) -> list[str]:
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


def _repo_files() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return (
        tomllib.loads((REPO / "pyproject.toml").read_text()),
        tomllib.loads((REPO / "uv.lock").read_text()),
        yaml.safe_load((REPO / ".pre-commit-config.yaml").read_text()),
    )


@pytest.mark.logic
def test_pre_commit_and_the_workspace_run_one_ruff() -> None:
    """The pyproject pins ruff to the version uv.lock holds, and every ruff hook runs that locked tool."""
    assert toolchain_divergences(*_repo_files()) == []


def _remote(rev: str) -> dict[str, Any]:
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


def workflow_ruff_pins(text: str) -> set[str]:
    """Every `ruff==<version>` a workflow installs."""
    return set(re.findall(r"\bruff==([0-9][0-9A-Za-z.]*)", text))


@pytest.mark.logic
def test_ci_installs_the_locked_ruff() -> None:
    """The ruff the core-only job installs (inline or by the core test group) is the version uv.lock pins."""
    lock = tomllib.loads((REPO / "uv.lock").read_text())
    pins = workflow_ruff_pins((REPO / ".github" / "workflows" / "tests.yaml").read_text())
    pins |= workflow_ruff_pins((REPO / "packages" / "senselab-core" / "pyproject.toml").read_text())
    assert pins == {locked_version(lock, "ruff")}


PLATFORM_CHECKED = "packages/senselab-core/src/senselab_core/runtime/worker_client.py"
PLATFORMS = ("linux", "darwin")


def platform_checked(precommit: dict[str, Any], path: str) -> set[str]:
    """The `--platform` values of the mypy hooks whose `files` pattern selects `path`."""
    found = set()
    for repo in precommit.get("repos", []):
        for hook in repo.get("hooks", []):
            args = [str(a) for a in hook.get("args", [])]
            if hook.get("id") == "mypy" and "--platform" in args and re.search(hook.get("files", "$^"), path):
                found.add(args[args.index("--platform") + 1])
    return found


@pytest.mark.logic
def test_the_platform_specific_module_is_type_checked_on_both_platforms() -> None:
    """A module that calls kqueue or waitid is checked by mypy under `--platform linux` and `--platform darwin`."""
    assert platform_checked(_repo_files()[2], PLATFORM_CHECKED) == set(PLATFORMS)


@pytest.mark.logic
def test_a_missing_platform_hook_is_found() -> None:
    """A configuration that checks one platform only is reported as covering one."""
    config = {"repos": [{"hooks": [{"id": "mypy", "args": ["--platform", "linux"], "files": r"^a/b\.py$"}]}]}
    assert platform_checked(config, "a/b.py") == {"linux"}
    assert platform_checked(config, "a/c.py") == set()


FORMATTER_REPO = "https://github.com/macisamuele/language-formatters-pre-commit-hooks"
FORMATTER_LIBRARIES = {"pretty-format-yaml": {"ruamel.yaml"}, "pretty-format-toml": {"toml-sort", "tomlrt"}}
"""The libraries whose version decides each formatter hook's output, which the hook repository leaves unpinned."""


def unpinned_formatter_libraries(precommit: dict[str, Any]) -> list[str]:
    """Each formatter hook whose output-deciding library is not pinned `==` in its `additional_dependencies`."""
    found = []
    for repo in precommit.get("repos", []):
        if repo.get("repo") != FORMATTER_REPO:
            continue
        hooks = {h.get("id"): h for h in repo.get("hooks", [])}
        for hook_id, libraries in FORMATTER_LIBRARIES.items():
            pinned = {
                m.group(1).lower()
                for dep in hooks.get(hook_id, {}).get("additional_dependencies", [])
                if (m := re.fullmatch(r"([A-Za-z0-9_.-]+)==[0-9][0-9A-Za-z.]*", str(dep)))
            }
            found += [f"{hook_id} does not pin {lib}" for lib in sorted(libraries) if lib.lower() not in pinned]
    return found


@pytest.mark.logic
def test_the_formatter_hooks_pin_the_libraries_that_decide_their_output() -> None:
    """A release of toml-sort, tomlrt or ruamel.yaml cannot change what CI's formatter hooks rewrite."""
    assert unpinned_formatter_libraries(_repo_files()[2]) == []


@pytest.mark.logic
def test_an_unpinned_formatter_library_is_found() -> None:
    """A hook with no pin, a range, or a different library is reported for each library it leaves free."""
    deps = {
        "pretty-format-yaml": ["ruamel.yaml==0.19.1"],
        "pretty-format-toml": ["toml-sort==0.25.0", "tomlrt==2.2.13"],
    }

    def config(**override: list[str]) -> dict[str, Any]:
        merged = {**deps, **override}
        hooks = [{"id": k, "additional_dependencies": v} for k, v in merged.items()]
        return {"repos": [{"repo": FORMATTER_REPO, "hooks": hooks}]}

    assert unpinned_formatter_libraries(config()) == []
    assert unpinned_formatter_libraries(config(**{"pretty-format-toml": ["toml-sort==0.25.0"]})) == [
        "pretty-format-toml does not pin tomlrt"
    ]
    assert unpinned_formatter_libraries(config(**{"pretty-format-toml": ["toml-sort>=0.22", "tomlrt==2.2.13"]})) == [
        "pretty-format-toml does not pin toml-sort"
    ]
    assert unpinned_formatter_libraries(config(**{"pretty-format-yaml": []})) == [
        "pretty-format-yaml does not pin ruamel.yaml"
    ]


KNOWN_SOURCE = "packages/senselab-core/src/senselab_core/kernel/identity.py"
KNOWN_FILES = (KNOWN_SOURCE, "packages/senselab-worker/scripts/unicode_table.py")
"""Package files every lint and format exclusion must leave checked."""


def _ruff(path: str, *args: str) -> str:
    done = subprocess.run(
        [sys.executable, "-m", "ruff", *args, "--force-exclude", path],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout + done.stderr


@pytest.mark.logic
@pytest.mark.parametrize("path", KNOWN_FILES)
def test_ruff_check_covers_package_files(path: str) -> None:
    """`ruff check` lists a package file even when pre-commit passes it with `--force-exclude`."""
    assert path in _ruff(path, "check", "--show-files")


@pytest.mark.logic
@pytest.mark.parametrize("path", KNOWN_FILES)
def test_ruff_format_covers_package_files(path: str) -> None:
    """`ruff format` reaches a package file rather than reporting no files."""
    assert "1 file already formatted" in _ruff(path, "format", "--check")


def exclusions_matching(path: str, pyproject: dict[str, Any], precommit: dict[str, Any]) -> list[str]:
    """Every codespell, mypy or pre-commit exclusion that matches `path`."""
    found = []
    parts = Path(path).parts
    for glob in pyproject["tool"]["codespell"]["skip"]:
        prefixes = ["./" + "/".join(parts[: i + 1]) for i in range(len(parts))]
        if any(fnmatch.fnmatch(c, glob) for c in [path, *parts, *prefixes]):
            found.append(f"codespell skip {glob!r}")
    found += [f"mypy exclude {r!r}" for r in pyproject["tool"]["mypy"]["exclude"] if re.search(r, path)]
    if re.search(precommit.get("exclude", "$^"), path):
        found.append(f"pre-commit exclude {precommit['exclude']!r}")
    for repo in precommit.get("repos", []):
        for hook in repo.get("hooks", []):
            if "ruff" in str(hook.get("entry", "")) and re.search(hook.get("exclude", "$^"), path):
                found.append(f"hook {hook.get('id')} exclude {hook['exclude']!r}")
    return found


@pytest.mark.logic
@pytest.mark.parametrize("path", KNOWN_FILES)
def test_no_exclusion_matches_package_files(path: str) -> None:
    """The 1.x-pool exclusions are anchored to the top-level tree and match nothing under `packages/`."""
    pyproject, _, precommit = _repo_files()
    assert exclusions_matching(path, pyproject, precommit) == []


@pytest.mark.logic
def test_an_unanchored_exclusion_is_found() -> None:
    """A bare `src` skip or an unanchored `src/` regex is reported against a package source path."""
    pyproject = {"tool": {"codespell": {"skip": ["src"]}, "mypy": {"exclude": ["src/"]}}}
    assert exclusions_matching(KNOWN_SOURCE, pyproject, {"exclude": "src/"}) == [
        "codespell skip 'src'",
        "mypy exclude 'src/'",
        "pre-commit exclude 'src/'",
    ]
