"""Guard: modules moved into ``senselab_core`` leave no reference to their old paths."""

import ast
import importlib.util
import json
import re
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCAN_ROOTS = ("src", "scripts", "packages")
IMPORT_SCAN_ROOTS = (*SCAN_ROOTS, "tutorials")


def _fixture_budget_suffixes() -> tuple[str, ...]:
    spec = importlib.util.spec_from_file_location("_budget_for_suffixes", REPO / "scripts" / "check_fixture_budget.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.CODE_SUFFIXES)


CODE_SUFFIXES = _fixture_budget_suffixes()
SELF = Path(__file__).resolve()

MOVED = {
    "senselab.utils.file_lock",
    "senselab.utils.model_revision",
    "senselab/utils/file_lock",
    "senselab/utils/model_revision",
    "senselab.utils.prov_store",
    "senselab.utils.prov_bep028",
    "senselab/utils/prov_store",
    "senselab/utils/prov_bep028",
    "senselab.utils.cuda_probe",
    "senselab/utils/cuda_probe",
}

TOMBSTONES = {
    "senselab.utils.file_lock": "src/senselab/utils/file_lock.py",
    "senselab.utils.model_revision": "src/senselab/utils/model_revision.py",
    "senselab.utils.prov_store": "src/senselab/utils/prov_store.py",
    "senselab.utils.prov_bep028": "src/senselab/utils/prov_bep028.py",
    "senselab.utils.cuda_probe": "src/senselab/utils/cuda_probe.py",
}
"""Each moved module and the file it lived in; neither the file nor an importable module may exist there."""

DYNAMIC_ALLOWED: dict[str, frozenset[str]] = {
    "src/senselab/utils/venv_lock.py": frozenset({"<dynamic:module_name>"}),
    "src/senselab/audio/workflows/audio_analysis/__init__.py": frozenset({"<dynamic:f'{__name__}.{submodule}'>"}),
    "packages/senselab-core/src/senselab_core/testing/values.py": frozenset({"<dynamic:module.name>"}),
    "packages/senselab-core/src/senselab_core/testing/contracts.py": frozenset({"<dynamic:module>"}),
    "packages/senselab-core/tests/lint/destructive_ops_test.py": frozenset({"<dynamic:module>"}),
    "packages/senselab-worker/src/senselab_worker/launch.py": frozenset({"<dynamic:argv[1]>"}),
    "src/tests/conftest.py": frozenset({"<dynamic:module>"}),
    "src/tests/audio/tasks/audio_understanding_test.py": frozenset({"<dynamic:name>"}),
    "src/tests/audio/workflows/audio_analysis/run_config_test.py": frozenset(
        {"<dynamic:f'senselab.audio.workflows.audio_analysis.{module_name}'>"}
    ),
    "src/tests/audio/workflows/audio_analysis/adaptive/evidence_weight_test.py": frozenset({"<dynamic:module_path>"}),
}
"""Call sites of ``importlib.import_module`` / ``__import__`` whose argument cannot be folded to a constant."""

# Moved module basenames, for the `from senselab.utils import <name>` form.
MOVED_NAMES = {"file_lock", "model_revision", "prov_store", "prov_bep028", "cuda_probe"}

# Names that now live in senselab_core and must not be imported from `senselab.utils.dependencies`.
MOVED_HELPERS = {"retry_on_transient_error", "_is_transient", "_http_status", "get_cached_commit_hash"}

_IMPORT_FROM = re.compile(r"from\s+{module}\s+import\s+(\(?[^)\n]*(?:\n[^)\n]*)*?\)|[^\n]*)", re.M)


def _imported_names(text: str, module: str) -> set[str]:
    """Names brought in by every ``from <module> import ...`` statement in ``text``."""
    names: set[str] = set()
    for m in re.finditer(_IMPORT_FROM.pattern.format(module=re.escape(module)), text):
        names.update(n.strip().split(" as ")[0].strip("() \n") for n in m.group(1).replace("\n", ",").split(","))
    return {n for n in names if n}


# Extended by later tasks: patterns (not paths) that must not appear in any scanned file.
FORBIDDEN_PATTERNS: tuple[re.Pattern[str], ...] = ()


def _scanned_files() -> list[Path]:
    """Every text source under the scan roots, excluding this file."""
    files: list[Path] = []
    for root in SCAN_ROOTS:
        for suffix in ("*.py", "*.md"):
            files.extend(p for p in (REPO / root).rglob(suffix) if p.resolve() != SELF)
    return files


@pytest.mark.logic
def test_no_moved_module_paths_remain() -> None:
    """No scanned file mentions an old module path, worker-script strings included."""
    hits = []
    for path in _scanned_files():
        text = path.read_text(errors="replace")
        hits += [f"{path.relative_to(REPO)}: {m}" for m in sorted(MOVED) if m in text]
        hits += [f"{path.relative_to(REPO)}: {p.pattern}" for p in FORBIDDEN_PATTERNS if p.search(text)]
        hits += [
            f"{path.relative_to(REPO)}: from senselab.utils import {n}"
            for n in sorted(MOVED_NAMES & _imported_names(text, "senselab.utils"))
        ]
        hits += [
            f"{path.relative_to(REPO)}: {n} imported from senselab.utils.dependencies"
            for n in sorted(MOVED_HELPERS & _imported_names(text, "senselab.utils.dependencies"))
        ]
    assert hits == []


_SHA_LITERAL = re.compile(r"\b\w*SHA256\s*=\s*[\"'][0-9a-f]{64}[\"']")


@pytest.mark.logic
def test_no_weight_sha256_literals_remain() -> None:
    """Weight digests live in `weights.yaml`; no `*_SHA256 = "<64 hex>"` literal is left outside `reproduce/`."""
    hits = [
        str(path.relative_to(REPO))
        for path in (REPO / "src" / "senselab").rglob("*.py")
        if "reproduce" not in path.relative_to(REPO / "src" / "senselab").parts
        and _SHA_LITERAL.search(path.read_text())
    ]
    assert hits == []


# --- import resolution ------------------------------------------------------------------------------


def _module_name(path: Path) -> tuple[str, bool]:
    """The dotted module name of ``path`` (found by walking up through package markers) and whether it is a package."""
    parts = [] if path.stem == "__init__" else [path.stem]
    parent = path.parent
    while (parent / "__init__.py").exists() or (parent / "__init__.pyi").exists():
        parts.insert(0, parent.name)
        parent = parent.parent
    return ".".join(parts), path.stem == "__init__"


def _sources(path: Path) -> Iterator[str]:
    """Python source texts in ``path``: the file, or each code cell of a notebook with IPython lines blanked."""
    text = path.read_text(errors="replace")
    if path.suffix != ".ipynb":
        yield text
        return
    try:
        cells = json.loads(text).get("cells", [])
    except (ValueError, AttributeError):
        return
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        lines = (source if isinstance(source, str) else "".join(source)).splitlines()
        yield "\n".join("" if ln.lstrip().startswith(("%", "!", "?")) else ln for ln in lines)


def _fold(node: ast.AST) -> str | None:
    """The string ``node`` evaluates to when built from literals, ``+`` and f-strings of constants, else ``None``."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _fold(node.left), _fold(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        pieces = []
        for value in node.values:
            piece = (
                _fold(value.value)
                if isinstance(value, ast.FormattedValue) and value.format_spec is None
                else (_fold(value) if isinstance(value, ast.Constant) else None)
            )
            if piece is None:
                return None
            pieces.append(piece)
        return "".join(pieces)
    return None


def _importer_aliases(tree: ast.AST) -> set[str]:
    """Names bound to ``import_module`` or ``__import__`` by ``from importlib import ... as`` or by assignment."""
    names = {"import_module", "__import__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            names |= {a.asname for a in node.names if a.name in ("import_module", "__import__") and a.asname}
        elif isinstance(node, ast.Assign):
            value = node.value
            called = value.attr if isinstance(value, ast.Attribute) else value.id if isinstance(value, ast.Name) else ""
            if called in names:
                names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    return names


def _dynamic_target(node: ast.Call, importers: set[str] | None = None) -> ast.expr | None:
    """The module argument of ``importlib.import_module(x)``, ``import_module(x)`` or ``__import__(x)``, or ``None``."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    if name not in (importers or {"import_module", "__import__"}):
        return None
    if node.args:
        return node.args[0]
    return next((k.value for k in node.keywords if k.arg == "name"), None)


def _chain(node: ast.AST) -> list[str] | None:
    """``[root, attr, ...]`` of a ``Name.attr.attr`` expression, else ``None``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.insert(0, node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return [node.id, *parts]
    return None


def _imports_of(tree: ast.AST, module: str, is_package: bool, label: str) -> set[str]:
    package = module if is_package else module.rpartition(".")[0]
    found: set[str] = set()
    bound: dict[str, str] = {}
    importers = _importer_aliases(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name)
                bound[alias.asname or alias.name.partition(".")[0]] = (
                    alias.name if alias.asname else alias.name.partition(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".") if package else []
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                source = ".".join([*base, *([node.module] if node.module else [])])
            else:
                source = node.module or ""
            if source:
                found.add(source)
            for alias in node.names:
                if alias.name == "*":
                    continue
                full = f"{source}.{alias.name}" if source else alias.name
                found.add(full)
                bound[alias.asname or alias.name] = full
        elif isinstance(node, ast.Call):
            target = _dynamic_target(node, importers)
            if target is None:
                continue
            folded = _fold(target)
            found.add(folded if folded is not None else f"<dynamic:{ast.unparse(target)}>")
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and not isinstance(getattr(node, "ctx", None), ast.Store):
            chain = _chain(node)
            if chain and chain[0] in bound:
                dotted = bound[chain[0]].split(".") + chain[1:]
                found.update(".".join(dotted[:i]) for i in range(1, len(dotted) + 1))
    return found


def _imports(path: Path) -> set[str]:
    """Absolute module names that ``path`` imports, resolving relative imports against its package.

    Sources are ``.py`` and ``.pyi`` files and the code cells of ``.ipynb`` files. It covers ``import a.b``,
    ``from a import b`` (as ``a.b`` and ``a``), ``from a import *``, ``importlib.import_module(x)`` and
    ``__import__(x)`` when ``x`` folds to a constant, and attribute chains rooted at an imported name. A
    call whose argument does not fold appears as ``<dynamic:SOURCE>``.
    """
    module, is_package = _module_name(path)
    found: set[str] = set()
    for text in _sources(path):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            found.add(f"<unparseable:{path.name}>")
            continue
        found |= _imports_of(tree, module, is_package, path.name)
    return found


def _moved_dotted() -> set[str]:
    return {m for m in MOVED if "." in m}


def _hits(names: set[str]) -> list[str]:
    return sorted(n for n in names if any(n == m or n.startswith(m + ".") for m in _moved_dotted()))


def _import_scanned_files() -> list[Path]:
    files: list[Path] = []
    for root in IMPORT_SCAN_ROOTS:
        base = REPO / root
        if base.is_dir():
            files.extend(p for p in base.rglob("*") if p.suffix in CODE_SUFFIXES and p.resolve() != SELF)
    return files


@pytest.mark.logic
def test_no_file_imports_a_moved_module() -> None:
    """No source or notebook imports a moved module, by any statement form, relative import or constant call."""
    hits: list[str] = []
    for path in _import_scanned_files():
        rel = path.relative_to(REPO).as_posix()
        names = _imports(path)
        hits += [f"{rel}: {n}" for n in _hits(names)]
        hits += [
            f"{rel}: {n}"
            for n in sorted(names)
            if n.startswith(("<dynamic:", "<unparseable")) and n not in DYNAMIC_ALLOWED.get(rel, frozenset())
        ]
    assert hits == []


def dead_dynamic_entries(allowed: dict[str, frozenset[str]]) -> list[str]:
    """Return `file: argument` for each allowance whose file is gone or no longer has that dynamic import."""
    dead = []
    for rel, arguments in sorted(allowed.items()):
        path = REPO / rel
        found = _imports(path) if path.is_file() else set()
        dead += [f"{rel}: {a}" for a in sorted(arguments - found)]
    return dead


@pytest.mark.logic
def test_every_dynamic_allowance_still_matches_a_site() -> None:
    """Each `DYNAMIC_ALLOWED` entry names a file that still has that dynamic import, so no entry is dead."""
    assert dead_dynamic_entries(DYNAMIC_ALLOWED) == []


@pytest.mark.contract
@pytest.mark.parametrize(
    "planted",
    [
        {"src/senselab/utils/venv_lock.py": frozenset({"<dynamic:other_name>"})},
        {"src/senselab/utils/no_such_module.py": frozenset({"<dynamic:x>"})},
        {"src/senselab/__init__.py": frozenset({"<dynamic:module>"})},
    ],
)
def test_a_dead_dynamic_allowance_is_reported(planted: dict[str, frozenset[str]]) -> None:
    """An allowance for a site that does not exist, or in a file that does not exist, is reported by name."""
    ((rel, arguments),) = planted.items()
    expected = [f"{rel}: {a}" for a in sorted(arguments)]
    assert dead_dynamic_entries({**DYNAMIC_ALLOWED, **planted}) == expected


@pytest.mark.logic
@pytest.mark.parametrize("module", sorted(TOMBSTONES))
def test_tombstones_hold(module: str) -> None:
    """A moved module has no file at its old path and no importable spec."""
    old = REPO / TOMBSTONES[module]
    assert not old.exists()
    assert not old.with_suffix("").exists()
    try:
        spec = importlib.util.find_spec(module)
    except ModuleNotFoundError:
        spec = None
    assert spec is None


def _write(base: Path, rel: str, text: str) -> Path:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _tree(base: Path) -> None:
    for pkg in ("senselab", "senselab/utils", "senselab/audio"):
        _write(base, f"{pkg}/__init__.py", "")


@pytest.mark.logic
def test_relative_import_of_moved_module_is_found(tmp_path: Path) -> None:
    """``from .. import x`` and ``from ..utils.x import y`` resolve against the file's package."""
    _tree(tmp_path)
    one = _write(tmp_path, "senselab/audio/a.py", "from ..utils import file_lock\n")
    two = _write(tmp_path, "senselab/audio/b.py", "from ..utils.model_revision import resolve\n")
    three = _write(tmp_path, "senselab/utils/c.py", "from . import prov_store\n")
    assert _hits(_imports(one)) == ["senselab.utils.file_lock"]
    assert _hits(_imports(two)) == ["senselab.utils.model_revision", "senselab.utils.model_revision.resolve"]
    assert _hits(_imports(three)) == ["senselab.utils.prov_store"]


@pytest.mark.logic
def test_importlib_concat_is_found(tmp_path: Path) -> None:
    """A constant-folded ``import_module`` / ``__import__`` argument is found; a dynamic one is reported."""
    _tree(tmp_path)
    src = (
        "import importlib\n"
        "importlib.import_module('senselab.utils.' + 'file_' + 'lock')\n"
        "__import__(f'senselab.utils.{kind}_probe')\n"
        "__import__('senselab.utils.' f'prov_store')\n"
        "importlib.import_module(name)\n"
    )
    names = _imports(_write(tmp_path, "senselab/audio/a.py", src))
    assert {"senselab.utils.file_lock", "senselab.utils.prov_store", "<dynamic:name>"} <= names
    assert "<dynamic:f'senselab.utils.{kind}_probe'>" in names


@pytest.mark.logic
def test_aliased_import_module_is_found(tmp_path: Path) -> None:
    """``import_module`` bound to another name, by ``from ... as`` or by assignment, is still followed."""
    src = (
        "from importlib import import_module as im\n"
        "import importlib\n"
        "load = importlib.import_module\n"
        "im('senselab.utils.' + 'file_lock')\n"
        "load('senselab.utils.' + 'prov_store')\n"
        "load(name)\n"
    )
    names = _imports(_write(tmp_path, "a.py", src))
    assert {"senselab.utils.file_lock", "senselab.utils.prov_store", "<dynamic:name>"} <= names


@pytest.mark.logic
def test_notebook_cell_import_is_found(tmp_path: Path) -> None:
    """A code cell's import is found, a markdown cell's is not, and an IPython magic does not break parsing."""
    cells = [
        {"cell_type": "code", "source": ["%matplotlib inline\n", "import senselab.utils.file_lock\n"]},
        {"cell_type": "markdown", "source": ["import senselab.utils.prov_store"]},
    ]
    path = _write(tmp_path, "nb.ipynb", json.dumps({"cells": cells}))
    assert _hits(_imports(path)) == ["senselab.utils.file_lock"]


@pytest.mark.logic
def test_star_import_is_found(tmp_path: Path) -> None:
    """``from senselab.utils.file_lock import *`` is found, and so is an attribute chain off an imported package."""
    star = _write(tmp_path, "a.py", "from senselab.utils.file_lock import *\n")
    chain = _write(tmp_path, "b.py", "import senselab.utils\nsenselab.utils.cuda_probe.X\n")
    alias = _write(tmp_path, "c.py", "from senselab import utils as u\nu.prov_bep028.run()\n")
    assert "senselab.utils.file_lock" in _hits(_imports(star))
    assert "senselab.utils.cuda_probe.X" in _hits(_imports(chain))
    assert "senselab.utils.prov_bep028.run" in _hits(_imports(alias))


@pytest.mark.logic
def test_pyi_is_scanned(tmp_path: Path) -> None:
    """A stub file is a source."""
    path = _write(tmp_path, "stub.pyi", "from senselab.utils.model_revision import X\n")
    assert "senselab.utils.model_revision" in _hits(_imports(path))
    assert Path(path).suffix in CODE_SUFFIXES


@pytest.mark.logic
def test_clean_file_has_no_moved_hits(tmp_path: Path) -> None:
    """The scanner does not flag unrelated imports."""
    path = _write(
        tmp_path, "a.py", "import os\nfrom senselab.utils import fastio\nimport senselab_core.runtime.locks\n"
    )
    assert _hits(_imports(path)) == []
