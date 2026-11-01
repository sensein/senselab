"""No production file reaches a digest module except along an allowed edge, by any spelling the resolver folds.

The forbidden modules and the allowed edges are read from `.importlinter` (`digests-are-tagged`, `raw-digest-owners`).
"""

import configparser
import importlib.util
import json
import subprocess
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CONTRACTS = ("importlinter:contract:digests-are-tagged", "importlinter:contract:raw-digest-owners")
PINNED = {
    "packages/senselab-worker/scripts/unicode_table.py": frozenset({"hashlib"}),
    "packages/senselab-core/hatch_pin_loader.py": frozenset(),
    "src/senselab/utils/data/network_guard/sitecustomize.py": frozenset(),
    "hatch_build_pins.py": frozenset(),
}
"""Production files outside every import-linter root, with the digest modules each may reach."""


def _load_resolver() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(
        "_import_resolver", Path(__file__).with_name("import_resolver_fixture.py")
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resolver = _load_resolver()


def _lines(value: str) -> list[str]:
    return [ln.strip() for ln in value.splitlines() if ln.strip() and not ln.strip().startswith("#")]


def contract() -> tuple[frozenset[str], dict[str, frozenset[str]]]:
    """The forbidden module names and, per importing module, the forbidden names it may reach."""
    parser = configparser.ConfigParser()
    parser.read(REPO / ".importlinter")
    forbidden: set[str] = set()
    allowed: dict[str, set[str]] = {}
    for name in CONTRACTS:
        section = parser[name]
        forbidden |= {m.rpartition(".")[2] for m in _lines(section["forbidden_modules"])}
        for edge in _lines(section.get("ignore_imports", "")):
            importer, _, imported = (part.strip() for part in edge.partition("->"))
            allowed.setdefault(importer, set()).add(imported.rpartition(".")[2])
    return frozenset(forbidden), {k: frozenset(v) for k, v in allowed.items()}


def reached(path: Path, forbidden: frozenset[str]) -> set[str]:
    """The forbidden names any import, attribute chain or folded dynamic import in ``path`` reaches.

    A source the resolver cannot parse reaches ``<unparseable>``, which no allowance names.
    """
    names = resolver.imports(path)
    unparseable = {"<unparseable>"} if any(n.startswith("<unparseable") for n in names) else set()
    return unparseable | {segment for name in names for segment in name.split(".") if segment in forbidden}


def _module(rel: str, path: Path) -> str:
    if rel.startswith("scripts/"):
        return "scripts." + path.stem
    return str(resolver.module_name(path)[0])


def production_files() -> list[str]:
    """Every production source.

    That is each `.py`/`.pyi` under `packages/`, `src/` and `scripts/` outside test trees, each at the root, and each
    tracked notebook, read through its code cells.
    """
    found = {
        p.relative_to(REPO).as_posix()
        for root in ("packages", "src", "scripts")
        for p in (REPO / root).rglob("*")
        if p.suffix in (".py", ".pyi")
        and "tests" not in p.relative_to(REPO).parts
        and not p.name.endswith("_test.py")
        and p.name != "conftest.py"
        and ".venv" not in p.parts
    }
    return sorted(found | {p.name for p in REPO.glob("*.py")} | set(tracked_notebooks()))


def tracked_notebooks() -> list[str]:
    """Every notebook git tracks in this repository."""
    listed = subprocess.run(["git", "ls-files", "-z", "*.ipynb"], cwd=REPO, capture_output=True, check=True).stdout
    return [p for p in listed.decode().split("\0") if p]


def violations(rel: str, path: Path, forbidden: frozenset[str], allowed: dict[str, frozenset[str]]) -> list[str]:
    """Each forbidden name ``path`` reaches beyond its allowance, or beyond its pinned set if it is out of graph."""
    names = reached(path, forbidden)
    if rel in PINNED:
        return [f"{rel}: reaches {sorted(names)}, pinned {sorted(PINNED[rel])}"] if names != PINNED[rel] else []
    extra = names - allowed.get(_module(rel, path), frozenset())
    return [f"{rel}: {name}" for name in sorted(extra)]


@pytest.mark.logic
def test_no_production_file_reaches_a_digest_module_off_an_allowed_edge() -> None:
    """Imports, re-exports, attribute chains and literal dynamic imports of a digest module follow the contract."""
    forbidden, allowed = contract()
    hits = [h for rel in production_files() for h in violations(rel, REPO / rel, forbidden, allowed)]
    assert hits == []


@pytest.mark.logic
def test_the_out_of_graph_files_are_production_files() -> None:
    """Every pinned file exists, so no pin is dead."""
    assert set(PINNED) <= set(production_files())


PLANTED = {
    "literal_dunder_import": "__import__('hashlib').sha256(json.dumps(x).encode())",
    "builtins_dunder_import": "import builtins\nbuiltins.__import__('hashlib')",
    "import_module_literal": "import importlib\nimportlib.import_module('hmac')",
    "urllib_chain": "import urllib.request\nurllib.request.hashlib.sha256(json.dumps(x).encode())",
    "smtplib_chain": "import smtplib\nsmtplib.hmac.new(b'k', json.dumps(x).encode(), 'sha256')",
    "from_urllib": "from urllib.request import hashlib",
    "first_party_reexport": "from senselab.utils.venv_lock import hashlib",
    "first_party_chain": "import senselab.utils.tasks.cached_inference\nsenselab.utils.tasks.cached_inference.hashlib",
    "random_sha512": "import random\nrandom._sha512(json.dumps(x).encode())",
    "from_random_sha512": "from random import _sha512",
    "requests_chain": "import requests.auth\nrequests.auth.hashlib.md5(b'')",
    "urllib3_chain": "from urllib3.util import ssl_\nssl_.hmac",
    "torch_hub_chain": "import torch.hub\ntorch.hub.hashlib",
    "joblib_hashing": "from joblib import hashing\nhashing.hashlib",
    "getattr_literal": "import urllib.request\ngetattr(urllib.request, 'hashlib')",
    "raw_form_from_non_owner": "from senselab_worker import _raw_digest\n_raw_digest.bytes_digest('npy@1', b'')",
    "raw_form_reexport": "from senselab_worker.codec import _raw_digest",
    "underscore_sha2": "import _sha2",
    "xxhash": "import xxhash",
    "blake3": "import blake3",
}
"""Each spelling of the H9e review (I-1, I-2), planted in a 1.x module no allowance names."""


@pytest.mark.logic
@pytest.mark.parametrize("spelling", sorted(PLANTED))
@pytest.mark.parametrize("rel", ["src/senselab/zz/probe.py", "scripts/zz_probe.py", "hatch_build_pins.py"])
def test_i2_every_planted_spelling_is_reported(tmp_path: Path, spelling: str, rel: str) -> None:
    """A planted spelling in 1.x, a script or an out-of-graph root file is a violation."""
    forbidden, allowed = contract()
    path = tmp_path / Path(rel).name
    path.write_text(f"import json\n{PLANTED[spelling]}\n")
    assert violations(rel, path, forbidden, allowed), PLANTED[spelling]


@pytest.mark.logic
def test_i2_an_allowed_edge_is_not_reported(tmp_path: Path) -> None:
    """An owner reaching what its edge allows passes, and the same file reaching more does not."""
    forbidden, allowed = contract()
    pkg = tmp_path / "senselab_worker"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    owner = pkg / "codec.py"
    owner.write_text("from senselab_worker import _raw_digest\n_raw_digest.bytes_digest('npy@1', b'')\n")
    assert violations("packages/senselab-worker/src/senselab_worker/codec.py", owner, forbidden, allowed) == []
    owner.write_text(owner.read_text() + "import hashlib\n")
    assert violations("packages/senselab-worker/src/senselab_worker/codec.py", owner, forbidden, allowed) == [
        "packages/senselab-worker/src/senselab_worker/codec.py: hashlib"
    ]


@pytest.mark.logic
@pytest.mark.parametrize("rel", sorted(PINNED))
def test_m4_a_new_digest_import_in_an_out_of_graph_file_is_reported(tmp_path: Path, rel: str) -> None:
    """An out-of-graph file reaching a digest module beyond its pinned set is a violation."""
    forbidden, allowed = contract()
    path = tmp_path / Path(rel).name
    path.write_text((REPO / rel).read_text() + "\nimport hmac\n")
    assert violations(rel, path, forbidden, allowed)


@pytest.mark.logic
def test_every_tracked_notebook_is_a_production_file() -> None:
    """The code cells of each tracked notebook are scanned like a module that no allowance names."""
    notebooks = tracked_notebooks()
    assert notebooks and set(notebooks) <= set(production_files())


def _notebook(path: Path, *cells: str) -> Path:
    body = [{"cell_type": "code", "source": c.splitlines(keepends=True), "metadata": {}} for c in cells]
    body.insert(1, {"cell_type": "markdown", "source": ["import hashlib\n"], "metadata": {}})
    path.write_text(json.dumps({"cells": body, "nbformat": 4}))
    return path


@pytest.mark.logic
@pytest.mark.parametrize("spelling", sorted(PLANTED))
def test_notebook_cells_are_scanned_for_every_planted_spelling(tmp_path: Path, spelling: str) -> None:
    """A planted spelling in a notebook's code cell, with an IPython magic beside it, is a violation."""
    forbidden, allowed = contract()
    path = _notebook(tmp_path / "t.ipynb", "%matplotlib inline\nimport json\n", PLANTED[spelling])
    assert violations("tutorials/t.ipynb", path, forbidden, allowed), PLANTED[spelling]


@pytest.mark.logic
def test_a_binding_from_an_earlier_cell_is_followed(tmp_path: Path) -> None:
    """Cells share one namespace: a chain in one cell on a module imported in another is resolved."""
    forbidden, allowed = contract()
    path = _notebook(tmp_path / "t.ipynb", "import urllib.request\n", "urllib.request.hashlib.sha256(b'')\n")
    assert violations("tutorials/t.ipynb", path, forbidden, allowed) == ["tutorials/t.ipynb: hashlib"]


@pytest.mark.logic
@pytest.mark.parametrize("text", ["not json", "[]", '{"cells": 3}', '{"nbformat": 4}'])
def test_an_unreadable_notebook_is_a_violation(tmp_path: Path, text: str) -> None:
    """A notebook whose cells cannot be read is reported, not passed as empty."""
    forbidden, allowed = contract()
    path = tmp_path / "t.ipynb"
    path.write_text(text)
    assert violations("tutorials/t.ipynb", path, forbidden, allowed) == ["tutorials/t.ipynb: <unparseable>"]


@pytest.mark.logic
def test_a_notebook_with_an_unparseable_cell_is_scanned_cell_by_cell(tmp_path: Path) -> None:
    """One cell that is not Python marks the notebook unparseable, and the other cells are still resolved."""
    forbidden, allowed = contract()
    path = _notebook(tmp_path / "t.ipynb", "import hmac\n", "def broken(:\n")
    assert violations("tutorials/t.ipynb", path, forbidden, allowed) == [
        "tutorials/t.ipynb: <unparseable>",
        "tutorials/t.ipynb: hmac",
    ]
