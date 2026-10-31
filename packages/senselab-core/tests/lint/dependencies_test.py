"""Every distribution under ``packages/`` is a root package and imports only what it declares."""

import configparser
import re
import sys
import tomllib
from importlib import metadata
from pathlib import Path

import grimp
import pytest

REPO = Path(__file__).resolve().parents[4]
PACKAGES = REPO / "packages"
IMPORTLINTER = REPO / ".importlinter"


def import_names(packages_dir: Path) -> set[str]:
    """The top-level import names under ``<packages_dir>/*/src/*``."""
    return {
        p.name for p in packages_dir.glob("*/src/*") if p.is_dir() and p.name != "__pycache__" and "." not in p.name
    }


def root_packages(text: str) -> set[str]:
    """The ``root_packages`` of an import-linter configuration."""
    parser = configparser.ConfigParser()
    parser.read_string(text)
    return {line.strip() for line in parser["importlinter"]["root_packages"].splitlines() if line.strip()}


def declared_dependencies(pyproject: dict[str, object]) -> set[str]:
    """Normalised distribution names in ``[project.dependencies]``, or in the custom metadata hook that fills them."""
    project = pyproject["project"]
    assert isinstance(project, dict)
    hook = pyproject.get("tool", {}).get("hatch", {}).get("metadata", {}).get("hooks", {}).get("custom", {})  # type: ignore[attr-defined]
    specs = [*project.get("dependencies", []), *hook.get("dependencies", [])]
    if hook.get("pin"):
        specs.append(hook["pin"])
    return {_normalise(re.split(r"[\s<>=!~;\[]", spec, maxsplit=1)[0]) for spec in specs}


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def import_names_of(distribution: str) -> set[str]:
    """The top-level import names an installed distribution provides.

    Raises:
        LookupError: if the distribution provides none, which means it is not installed.
    """
    found = {
        top
        for top, dists in metadata.packages_distributions().items()
        if distribution in {_normalise(d) for d in dists}
    }
    if not found:
        raise LookupError(f"distribution {distribution!r} is declared but provides no importable module here")
    return found


def undeclared_imports(graph: grimp.ImportGraph, roots: set[str], allowed: set[str]) -> set[str]:
    """The external top-level modules ``graph`` imports that are neither in ``roots`` nor in ``allowed``."""
    externals = {m.partition(".")[0] for m in graph.modules} - roots
    return externals - allowed - set(sys.stdlib_module_names)


REPOSITORY_ROOTS = {"senselab", "scripts"}
"""Root packages outside ``packages/``: the 1.x tree and the repository scripts, which the digest contract covers."""


@pytest.mark.logic
def test_every_package_is_a_root_package() -> None:
    """The import-linter ``root_packages`` are the import names under ``packages/*/src/*`` and the repository roots."""
    assert root_packages(IMPORTLINTER.read_text()) == import_names(PACKAGES) | REPOSITORY_ROOTS


@pytest.mark.logic
def test_an_unlisted_package_is_not_a_root_package(tmp_path: Path) -> None:
    """A package directory absent from ``root_packages`` makes the two sets differ."""
    for name in ("alpha", "beta"):
        (tmp_path / f"dist-{name}" / "src" / name).mkdir(parents=True)
    text = "[importlinter]\nroot_packages =\n    alpha\ninclude_external_packages = True\n"
    assert import_names(tmp_path) == {"alpha", "beta"} != root_packages(text)


OPTIONAL_IMPORTS = {"senselab_core.runtime.retry": frozenset({"requests", "httpx"})}
"""Modules a single module imports only inside ``try: ... except ImportError``; not dependencies."""


def external_imports(graph: grimp.ImportGraph, own: set[str]) -> dict[str, set[str]]:
    """``{module: top-level modules it imports directly}`` for every module of the root packages ``own``."""
    out: dict[str, set[str]] = {}
    for module in graph.modules:
        if module.partition(".")[0] in own:
            imported = {i.partition(".")[0] for i in graph.find_modules_directly_imported_by(module)} - own
            if imported:
                out[module] = imported
    return out


def undeclared_by_module(
    imports: dict[str, set[str]], allowed: set[str], optional: dict[str, frozenset[str]]
) -> dict[str, set[str]]:
    """The imports of each module that are neither allowed, stdlib, nor optional for that module."""
    stdlib = set(sys.stdlib_module_names)
    bad = {m: i - allowed - stdlib - optional.get(m, frozenset()) for m, i in imports.items()}
    return {m: i for m, i in bad.items() if i}


@pytest.mark.contract
def test_packages_import_only_declared_dependencies() -> None:
    """Each package's external imports are its declared dependencies, the stdlib and its sibling packages."""
    roots = import_names(PACKAGES)
    graph = grimp.build_graph(*sorted(roots), include_external_packages=True)
    workspace = {_normalise(tomllib.loads(p.read_text())["project"]["name"]) for p in PACKAGES.glob("*/pyproject.toml")}
    for pyproject in sorted(PACKAGES.glob("*/pyproject.toml")):
        data = tomllib.loads(pyproject.read_text())
        own = {p.name for p in (pyproject.parent / "src").iterdir() if p.is_dir() and p.name != "__pycache__"}
        allowed = set(roots)
        for dist in declared_dependencies(data) - workspace:
            allowed |= import_names_of(dist)
        assert undeclared_by_module(external_imports(graph, own), allowed, OPTIONAL_IMPORTS) == {}


@pytest.mark.logic
def test_optional_imports_are_still_imported() -> None:
    """Every optional import allowance names a module that still imports those names."""
    roots = import_names(PACKAGES)
    graph = grimp.build_graph(*sorted(roots), include_external_packages=True)
    imports = external_imports(graph, roots)
    for module, names in OPTIONAL_IMPORTS.items():
        assert names <= imports.get(module, set()), module


@pytest.mark.logic
def test_an_optional_import_is_allowed_only_in_its_own_module() -> None:
    """The same import in another module of the package is reported."""
    imports = {"pkg.retry": {"requests"}, "pkg.other": {"requests"}}
    bad = undeclared_by_module(imports, set(), {"pkg.retry": frozenset({"requests"})})
    assert bad == {"pkg.other": {"requests"}}


@pytest.mark.logic
def test_an_undeclared_third_party_import_is_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A package importing a module it does not declare is reported; declared and stdlib imports are not."""
    pkg = tmp_path / "plantedpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "mod.py").write_text("import os\nimport json\nimport requests\nimport numpy\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    graph = grimp.build_graph("plantedpkg", include_external_packages=True)
    imported = {i.partition(".")[0] for m in graph.modules for i in graph.find_modules_directly_imported_by(m)} - {
        "plantedpkg"
    }
    assert imported == {"os", "json", "requests", "numpy"}
    assert undeclared_imports(graph, {"plantedpkg"}, {"numpy"}) == {"requests"}
    assert undeclared_by_module(external_imports(graph, {"plantedpkg"}), {"numpy"}, {}) == {
        "plantedpkg.mod": {"requests"}
    }
