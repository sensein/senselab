"""Guard: modules moved into ``senselab_core`` leave no reference to their old paths."""

import ast
import importlib.util
import json
import re
import subprocess
import types
from collections import Counter
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

DYNAMIC_ALLOWED: dict[str, dict[str, int]] = {
    "src/senselab/utils/venv_lock.py": {"<dynamic:module_name>": 1},
    "src/senselab/audio/workflows/audio_analysis/__init__.py": {"<dynamic:f'{__name__}.{submodule}'>": 1},
    "packages/senselab-core/src/senselab_core/testing/values.py": {"<dynamic:module.name>": 1},
    "packages/senselab-core/src/senselab_core/testing/contracts.py": {"<dynamic:module>": 1},
    "packages/senselab-core/tests/lint/destructive_ops_test.py": {"<dynamic:module>": 1},
    "packages/senselab-core/tests/lint/digest_contract_test.py": {"<dynamic:n>": 1},
    "packages/senselab-worker/src/senselab_worker/launch.py": {"<dynamic:argv[1]>": 1},
    "src/tests/conftest.py": {"<dynamic:module>": 2},
    "src/tests/audio/tasks/audio_understanding_test.py": {"<dynamic:name>": 1},
    "src/tests/audio/workflows/audio_analysis/run_config_test.py": {
        "<dynamic:f'senselab.audio.workflows.audio_analysis.{module_name}'>": 1
    },
    "src/tests/audio/workflows/audio_analysis/adaptive/evidence_weight_test.py": {"<dynamic:module_path>": 1},
}
"""Call sites of ``importlib.import_module`` / ``__import__`` whose argument cannot be folded, counted per argument."""

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


TEXT_EXCLUDED = ("specs/", ".specify/", ".claude/")
"""Tracked trees of records and agent configuration, which name moved paths as history."""


def _scanned_files() -> list[Path]:
    """Every tracked text file of any suffix outside `TEXT_EXCLUDED`, excluding this file."""
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True).stdout
    files = []
    for rel in listed.decode().split("\0"):
        path = REPO / rel
        if rel and not rel.startswith(TEXT_EXCLUDED) and path.is_file() and path.resolve() != SELF:
            if b"\0" not in path.read_bytes()[:8192]:
                files.append(path)
    return files


def _text_hits(path: Path) -> list[str]:
    """Each moved path, forbidden pattern or moved name the text of ``path`` mentions."""
    text = path.read_text(errors="replace")
    hits = [m for m in sorted(MOVED) if m in text]
    hits += [p.pattern for p in FORBIDDEN_PATTERNS if p.search(text)]
    hits += [f"from senselab.utils import {n}" for n in sorted(MOVED_NAMES & _imported_names(text, "senselab.utils"))]
    hits += [
        f"{n} imported from senselab.utils.dependencies"
        for n in sorted(MOVED_HELPERS & _imported_names(text, "senselab.utils.dependencies"))
    ]
    return hits


@pytest.mark.logic
def test_no_moved_module_paths_remain() -> None:
    """No tracked text file mentions an old module path, worker-script strings and non-Python files included."""
    hits = [f"{path.relative_to(REPO)}: {h}" for path in _scanned_files() for h in _text_hits(path)]
    assert hits == []


@pytest.mark.logic
def test_xg12_files_of_every_suffix_are_text_scanned(tmp_path: Path) -> None:
    """Shell, YAML and TOML files are scanned, and a `python -m` of a moved module in one is found."""
    suffixes = {p.suffix for p in _scanned_files()}
    assert {".sh", ".yaml", ".toml", ".py", ".md"} <= suffixes
    for name in ("run.sh", "job.yaml", "pyproject.toml"):
        planted = _write(tmp_path, name, "python -m senselab.utils.cuda_probe\n")
        assert _text_hits(planted) == ["senselab.utils.cuda_probe"]


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


# --- import resolution: the shared resolver ------------------------------------------------------


def _load_resolver() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_import_resolver", Path(__file__).with_name("import_resolver_fixture.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resolver = _load_resolver()
_imports = resolver.imports
_module_name = resolver.module_name


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
        hits += [f"{rel}: {n}" for n in sorted(names) if n.startswith("<unparseable")]
        hits += dynamic_site_hits(rel, resolver.dynamic_sites(path), DYNAMIC_ALLOWED)
    assert hits == []


def dynamic_site_hits(rel: str, sites: "Counter[str]", allowed: dict[str, dict[str, int]]) -> list[str]:
    """Each dynamic import argument of ``rel`` used at more sites than its allowance counts."""
    permitted = allowed.get(rel, {})
    return [
        f"{rel}: {arg}" + ("" if arg not in permitted else f" at {count} sites, {permitted[arg]} allowed")
        for arg, count in sorted(sites.items())
        if count > permitted.get(arg, 0)
    ]


def dead_dynamic_entries(allowed: dict[str, dict[str, int]]) -> list[str]:
    """Return `file: argument` for each allowance whose file is gone or has fewer such sites than it counts."""
    dead = []
    for rel, arguments in sorted(allowed.items()):
        path = REPO / rel
        found = resolver.dynamic_sites(path) if path.is_file() else Counter()
        dead += [f"{rel}: {a}" for a, count in sorted(arguments.items()) if found[a] < count]
    return dead


@pytest.mark.logic
def test_every_dynamic_allowance_still_matches_a_site() -> None:
    """Each `DYNAMIC_ALLOWED` entry names a file that still has that dynamic import, so no entry is dead."""
    assert dead_dynamic_entries(DYNAMIC_ALLOWED) == []


@pytest.mark.contract
@pytest.mark.parametrize(
    "planted",
    [
        {"src/senselab/utils/venv_lock.py": {"<dynamic:other_name>": 1}},
        {"src/senselab/utils/no_such_module.py": {"<dynamic:x>": 1}},
        {"src/senselab/__init__.py": {"<dynamic:module>": 1}},
        {"src/senselab/utils/venv_lock.py": {"<dynamic:module_name>": 2}},
    ],
)
def test_a_dead_dynamic_allowance_is_reported(planted: dict[str, dict[str, int]]) -> None:
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


@pytest.mark.logic
@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import importlib\nimportlib.import_module('.file_lock', 'senselab.utils')\n", "senselab.utils.file_lock"),
        (
            "from importlib import import_module\nimport_module(name='.model_revision', package='senselab.utils')\n",
            "senselab.utils.model_revision",
        ),
        (
            "import importlib\nimportlib.import_module('..utils.cuda_probe', 'senselab.audio')\n",
            "senselab.utils.cuda_probe",
        ),
        ("import senselab.utils\ngetattr(senselab.utils, 'prov_store')\n", "senselab.utils.prov_store"),
        ("import senselab\ngetattr(getattr(senselab, 'utils'), 'prov_bep028').run\n", "senselab.utils.prov_bep028"),
        ("from senselab import utils\ngetattr(utils, 'file_' + 'lock')\n", "senselab.utils.file_lock"),
    ],
)
def test_xg12_relative_keyword_and_getattr_imports_are_found(tmp_path: Path, source: str, expected: str) -> None:
    """A relative name resolved against its ``package`` argument, by position or keyword, and ``getattr`` are found."""
    assert expected in _hits(_imports(_write(tmp_path, "a.py", source)))


@pytest.mark.logic
def test_a_relative_import_against_an_unfoldable_package_is_dynamic(tmp_path: Path) -> None:
    """A relative name whose package does not fold is reported as dynamic, not dropped."""
    names = _imports(_write(tmp_path, "a.py", "import importlib\nimportlib.import_module('.file_lock', pkg)\n"))
    assert "<dynamic:'.file_lock', pkg>" in names


@pytest.mark.logic
def test_m5_a_second_site_of_an_allowed_dynamic_import_is_reported(tmp_path: Path) -> None:
    """An allowance covers one site: the same unfoldable argument at a second site in the file is reported."""
    one = _write(tmp_path, "one.py", "import importlib\nimportlib.import_module(module_name)\n")
    two = _write(
        tmp_path, "two.py", "import importlib\nimportlib.import_module(module_name)\n__import__(module_name)\n"
    )
    allowed = {"x.py": {"<dynamic:module_name>": 1}}
    assert dynamic_site_hits("x.py", resolver.dynamic_sites(one), allowed) == []
    assert dynamic_site_hits("x.py", resolver.dynamic_sites(two), allowed) == [
        "x.py: <dynamic:module_name> at 2 sites, 1 allowed"
    ]


@pytest.mark.logic
def test_a_moved_module_reached_across_notebook_cells_is_found(tmp_path: Path) -> None:
    """A notebook's cells share one namespace: a chain in one cell on a package imported in another is found."""
    cells = [
        {"cell_type": "code", "source": ["from senselab import utils\n"]},
        {"cell_type": "code", "source": ["utils.file_lock.FileLock\n"]},
    ]
    path = _write(tmp_path, "nb.ipynb", json.dumps({"cells": cells}))
    assert "senselab.utils.file_lock" in _hits(_imports(path))
