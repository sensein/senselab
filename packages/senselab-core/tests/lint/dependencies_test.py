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


@pytest.mark.logic
def test_every_package_is_a_root_package() -> None:
    """The import names under ``packages/*/src/*`` equal the import-linter ``root_packages``."""
    assert import_names(PACKAGES) == root_packages(IMPORTLINTER.read_text())


@pytest.mark.logic
def test_an_unlisted_package_is_not_a_root_package(tmp_path: Path) -> None:
    """A package directory absent from ``root_packages`` makes the two sets differ."""
    for name in ("alpha", "beta"):
        (tmp_path / f"dist-{name}" / "src" / name).mkdir(parents=True)
    text = "[importlinter]\nroot_packages =\n    alpha\ninclude_external_packages = True\n"
    assert import_names(tmp_path) == {"alpha", "beta"} != root_packages(text)


OPTIONAL_IMPORTS = {"senselab-core": frozenset({"requests", "httpx"})}
"""Modules a package imports only inside ``try: ... except ImportError`` (``runtime/retry.py``); not dependencies."""


def external_imports(graph: grimp.ImportGraph, own: set[str]) -> set[str]:
    """The top-level modules imported directly by any module of the root packages ``own``."""
    return {
        imported.partition(".")[0]
        for module in graph.modules
        if module.partition(".")[0] in own
        for imported in graph.find_modules_directly_imported_by(module)
    } - own


@pytest.mark.contract
def test_packages_import_only_declared_dependencies() -> None:
    """Each package's external imports are its declared dependencies, the stdlib and its sibling packages."""
    roots = import_names(PACKAGES)
    graph = grimp.build_graph(*sorted(roots), include_external_packages=True)
    workspace = {_normalise(tomllib.loads(p.read_text())["project"]["name"]) for p in PACKAGES.glob("*/pyproject.toml")}
    for pyproject in sorted(PACKAGES.glob("*/pyproject.toml")):
        data = tomllib.loads(pyproject.read_text())
        own = {p.name for p in (pyproject.parent / "src").iterdir() if p.is_dir() and p.name != "__pycache__"}
        allowed = set(roots) | OPTIONAL_IMPORTS.get(data["project"]["name"], frozenset())
        for dist in declared_dependencies(data) - workspace:
            allowed |= import_names_of(dist)
        undeclared = external_imports(graph, own) - allowed - set(sys.stdlib_module_names)
        assert undeclared == set(), f"{data['project']['name']} imports undeclared {sorted(undeclared)}"


@pytest.mark.logic
def test_optional_imports_are_still_imported() -> None:
    """Every optional import allowance names a module its package still imports."""
    roots = import_names(PACKAGES)
    graph = grimp.build_graph(*sorted(roots), include_external_packages=True)
    for pyproject in sorted(PACKAGES.glob("*/pyproject.toml")):
        data = tomllib.loads(pyproject.read_text())
        own = {p.name for p in (pyproject.parent / "src").iterdir() if p.is_dir() and p.name != "__pycache__"}
        optional = OPTIONAL_IMPORTS.get(data["project"]["name"], frozenset())
        assert optional <= external_imports(graph, own)


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
