"""No untagged digest API exists: the import contract, the banned names, and the digest module's surface (ADR-006)."""

import ast
import configparser
import importlib
import inspect
import json
import os
import pkgutil
import re
import shutil
import subprocess
import sys
import sysconfig
import textwrap
import types
from pathlib import Path

import numpy
import pytest
import senselab_core
import senselab_worker
from senselab_worker import _raw_digest, digest

REPO = Path(__file__).resolve().parents[4]
IMPORTLINTER = REPO / ".importlinter"
CONTRACT = "importlinter:contract:digests-are-tagged"
RAW_CONTRACT = "importlinter:contract:raw-digest-owners"
MEMBER_RULES = ("TID251", "B009", "S102", "S307", "F403")
OUT_OF_GRAPH = {
    "packages/senselab-core/hatch_pin_loader.py",
    "packages/senselab-worker/scripts/unicode_table.py",
    "src/senselab/utils/data/network_guard/sitecustomize.py",
}
"""Production files no import-linter root reaches: a build hook, the floor-interpreter table generator, a data file."""


def _contract() -> configparser.SectionProxy:
    parser = configparser.ConfigParser()
    parser.read(IMPORTLINTER)
    return parser[CONTRACT]


def _lines(value: str) -> list[str]:
    return [line.strip() for line in value.splitlines() if line.strip() and not line.strip().startswith("#")]


CAPABILITIES = frozenset(_lines(_contract()["forbidden_modules"]))


def _write(base: Path, rel: str, text: str) -> None:
    path = base / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


def _lint_imports(base: Path, *, extra_ignore: str = "", alerting: str = "none") -> subprocess.CompletedProcess[str]:
    """Run the repository's two digest contracts, alone, over the stub packages planted in `base`."""
    parser = configparser.ConfigParser()
    parser.read(IMPORTLINTER)
    config = configparser.ConfigParser()
    config["importlinter"] = dict(parser["importlinter"])
    for name in (CONTRACT, RAW_CONTRACT):
        config[name] = {**parser[name], "unmatched_ignore_imports_alerting": alerting}
    if extra_ignore:
        config[CONTRACT]["ignore_imports"] += f"\n{extra_ignore}"
    path = base / "contract.ini"
    with path.open("w") as f:
        config.write(f)
    code = "from importlinter.cli import lint_imports_command; lint_imports_command()"
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(
        [sys.executable, "-c", code, "--config", str(path), "--no-cache"],
        cwd=base,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _broken(output: str) -> set[str]:
    """`importer -> imported` for each reported import."""
    return {m.group(1) for m in re.finditer(r"^-\s+(\S+ -> \S+) \(l\.\d+\)", output, re.M)}


def _stubs(base: Path) -> None:
    for pkg in ("senselab", "senselab/zz", "senselab_core", "senselab_core/runtime"):
        _write(base, f"{pkg}/__init__.py", "")
    _write(base, "scripts/zz_clean.py", "import json\n")


XG1_IMPORTS = {
    "senselab/zz/probe_a.py": (
        """
        import hashlib, json
        def record_id(x):
            return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
        """,
        "hashlib",
    ),
    "scripts/zz_probe.py": ("import hmac\nhmac.new(b'k', b'm', 'sha256')\n", "hmac"),
    "senselab/zz/raw.py": (
        "import json\nfrom senselab_worker import _raw_digest\n"
        "_raw_digest.bytes_digest('npy@1', json.dumps(1).encode())\n",
        "senselab_worker._raw_digest",
    ),
    "senselab_core/runtime/fetch.py": (
        "from senselab_worker._raw_digest import bytes_digest\n",
        "senselab_worker._raw_digest",
    ),
}
"""XG-1: a module outside the owner hashes serialised data; the raw primitive it used no longer exists."""

XG2_SPELLINGS = {
    "sha384": "import hashlib\nhashlib.sha384(json.dumps(x).encode())",
    "sha224": "import hashlib\nhashlib.sha224(json.dumps(x).encode())",
    "sha3_512": "import hashlib\nhashlib.sha3_512(json.dumps(x).encode())",
    "shake_256": "import hashlib\nhashlib.shake_256(json.dumps(x).encode()).hexdigest(8)",
    "file_digest": "import hashlib, io\nhashlib.file_digest(io.BytesIO(json.dumps(x).encode()), 'sha256')",
    "hmac": "import hmac\nhmac.digest(b'k', json.dumps(x).encode(), 'sha256')",
    "ann_assign": "from hashlib import sha256\nd: bytes = json.dumps(x).encode()\nsha256(d)",
    "tuple_assign": "from hashlib import sha256\nd, e = json.dumps(x).encode(), 1\nsha256(d)",
    "aug_assign": "from hashlib import sha256\nd = b''\nd += json.dumps(x).encode()\nsha256(d)",
    "helper": "import hashlib\ndef ser(x):\n    return json.dumps(x).encode()\nhashlib.sha256(ser(x))",
    "model_dump_json": "import hashlib\nhashlib.sha256(m.model_dump_json().encode())",
    "fstring": "import hashlib\nhashlib.sha256(f'{x}'.encode())",
    "join": "import hashlib\nhashlib.sha256(','.join(x).encode())",
    "hasher_param": "import hashlib\ndef feed(h, x):\n    h.update(json.dumps(x).encode())\nfeed(hashlib.sha256(), x)",
    "rebound": "import hashlib\nm = hashlib.sha256()\nn = m\nn.update(json.dumps(x).encode())",
    "factory": "import hashlib\ndef mk():\n    return hashlib.sha256()\nmk().update(json.dumps(x).encode())",
    "underscore": "import _hashlib\n_hashlib.openssl_sha256(json.dumps(x).encode())",
    "blake3": "import blake3\nblake3.blake3(json.dumps(x).encode())",
    "xxhash": "import xxhash\nxxhash.xxh64(json.dumps(x).encode())",
}
"""XG-2: serialised data reaching a digest by spellings the data-flow lint missed; each needs a digest module."""


@pytest.mark.contract
def test_xg1_xg2_every_planted_digest_import_breaks_the_contract(tmp_path: Path) -> None:
    """Each XG-1 and XG-2 spelling, planted in 1.x, in a former 2.x owner or in a script, is reported by import."""
    _stubs(tmp_path)
    expected = set()
    for rel, (source, module) in XG1_IMPORTS.items():
        _write(tmp_path, rel, source)
        expected.add(f"{rel.removesuffix('.py').replace('/', '.')} -> {module}")
    for name, body in XG2_SPELLINGS.items():
        _write(tmp_path, f"senselab/zz/xg2_{name}.py", f"import json\nx = m = None\n{body}\n")
        imported = re.match(r"(?:from |import )(\w+)", body)
        assert imported
        expected.add(f"senselab.zz.xg2_{name} -> {imported.group(1)}")
    _write(tmp_path, "senselab_core/runtime/provenance.py", "import hashlib\nhashlib.sha256(b'')\n")
    expected.add("senselab_core.runtime.provenance -> hashlib")
    result = _lint_imports(tmp_path)
    assert result.returncode != 0, result.stdout
    assert _broken(result.stdout) == expected, result.stdout


@pytest.mark.contract
def test_the_contract_passes_a_tree_with_no_digest_import(tmp_path: Path) -> None:
    """The planted tree without a digest import keeps the contract, so the reports above are the plants."""
    _stubs(tmp_path)
    _write(tmp_path, "senselab/zz/clean.py", "import json\nfrom senselab_worker.digest import identity_digest\n")
    result = _lint_imports(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.contract
def test_an_allowance_that_matches_no_import_fails_the_contract(tmp_path: Path) -> None:
    """An `ignore_imports` entry naming an import that does not exist is an error, so no allowance is dead."""
    _stubs(tmp_path)
    result = _lint_imports(tmp_path, extra_ignore="senselab.zz.gone -> hashlib", alerting="error")
    assert result.returncode != 0
    assert "senselab.zz.gone -> hashlib" in result.stdout + result.stderr


def _ruff() -> str:
    found = shutil.which("ruff", path=sysconfig.get_path("scripts")) or shutil.which("ruff")
    if found is None:
        if os.environ.get("CI"):
            pytest.fail("ruff is not installed, and CI must run the banned-name tests")
        pytest.skip("ruff is not installed in this environment")
    return str(found)


def _ruff_codes(source: str, filename: str) -> list[str]:
    """The banned-name codes ruff reports, under the repository's configuration, for `source` as `filename`."""
    result = subprocess.run(
        [
            _ruff(),
            "check",
            "--no-cache",
            "--config",
            str(REPO / "pyproject.toml"),
            "--output-format",
            "json",
            "--stdin-filename",
            str(REPO / filename),
            "-",
        ],
        input=textwrap.dedent(source),
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    return [d["code"] for d in json.loads(result.stdout) if d["code"] in MEMBER_RULES]


XG3_SPELLINGS = {
    "keyword": "import importlib\nimportlib.import_module(name='hashlib')",
    "tuple_alias": "from importlib import import_module\nim, z = import_module, 1\nim('hashlib')",
    "annotated_alias": "from importlib import import_module\nim: object = import_module",
    "partial": "import functools, importlib\nfunctools.partial(importlib.import_module)('hashlib')",
    "getattr": "import importlib\ngetattr(importlib, 'import_module')('hashlib')",
    "dict_held": "import importlib\nloaders = {'f': importlib.import_module}\nloaders['f']('hashlib')",
    "sys_modules": "import sys\nsys.modules['hashlib'].sha256(b'')",
    "find_spec": "import importlib.util\nimportlib.util.find_spec('hashlib')",
    "module_from_spec": "from importlib.util import module_from_spec\nmodule_from_spec(spec)",
    "exec": "exec('import hashlib')",
    "eval": "eval('__import__')",
    "importlib_dunder": "import importlib\nimportlib.__import__('hashlib')",
    "builtins_dunder": "import builtins\nbuiltins.__import__('hashlib')",
    "attrgetter": "import operator, zlib\noperator.attrgetter('crc32')(zlib)",
    "methodcaller": "import operator, uuid\noperator.methodcaller('uuid5', ns, 'x')(uuid)",
    "star_import": "from zlib import *\ncrc32(b'')",
}
"""XG-3: re-spellings of a dynamic import that the capability-import lint missed."""

XG1_MEMBERS = {
    "uuid5": "import uuid\nuuid.uuid5(uuid.NAMESPACE_URL, json.dumps(x))",
    "uuid3": "from uuid import uuid3\nuuid3(ns, json.dumps(x))",
    "joblib": "import joblib\njoblib.hash(x)",
    "crc32": "import zlib\nzlib.crc32(json.dumps(x).encode())",
    "adler32": "from zlib import adler32\nadler32(b'')",
    "binascii": "import binascii\nbinascii.crc32(b'')",
    "murmurhash": "from sklearn.utils import murmurhash3_32\nmurmurhash3_32('x')",
    "dask": "import dask\ndask.tokenize(x)",
    "dask_base": "from dask.base import tokenize\ntokenize(x)",
}
"""XG-1: digests that need no digest module: name-based UUIDs and other libraries' hashes."""


@pytest.mark.contract
@pytest.mark.parametrize("filename", ["src/senselab/zz/probe.py", "packages/senselab-core/src/senselab_core/zz.py"])
@pytest.mark.parametrize("spelling", sorted({**XG3_SPELLINGS, **XG1_MEMBERS}))
def test_xg1_xg3_every_planted_member_spelling_is_banned(spelling: str, filename: str) -> None:
    """Each dynamic-import, name-based UUID or third-party hash spelling is reported in 1.x and 2.x code alike."""
    source = {**XG3_SPELLINGS, **XG1_MEMBERS}[spelling]
    assert _ruff_codes(f"import json\n{source}\n", filename), source


@pytest.mark.contract
def test_xg1_xg3_member_bans_cover_scripts() -> None:
    """`scripts/` is held to the banned names although it is exempt from the style rules."""
    assert _ruff_codes("import uuid\nuuid.uuid5(uuid.NAMESPACE_URL, 'x')\n", "scripts/zz_probe.py") == ["TID251"]


@pytest.mark.contract
def test_xg3_a_second_dynamic_import_in_an_allowed_file_is_reported() -> None:
    """Allowances are per site: a new `import_module` call beside the marked one in `launch.py` is reported."""
    rel = "packages/senselab-worker/src/senselab_worker/launch.py"
    source = (REPO / rel).read_text()
    assert _ruff_codes(source, rel) == []
    assert _ruff_codes(source + "\nimportlib.import_module('hashlib')\n", rel) == ["TID251"]


@pytest.mark.contract
def test_every_marked_member_site_still_needs_its_mark() -> None:
    """No `noqa` for a banned-name rule is unused, so no allowance is dead."""
    result = subprocess.run(
        [
            _ruff(),
            "check",
            "--no-cache",
            "--select",
            ",".join(("RUF100", *MEMBER_RULES)),
            "--output-format",
            "json",
            "packages",
            "src",
            "scripts",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=300,
    )
    dead = [
        f"{d['filename']}:{d['location']['row']}: {d['message']}"
        for d in json.loads(result.stdout)
        if d["code"] == "RUF100" and re.search(r"unused: [^)]*(" + "|".join(MEMBER_RULES) + ")", d["message"])
    ]
    assert dead == []


def _graph_files() -> set[str]:
    """Repository-relative files of every module in the import-linter graph."""
    parser = configparser.ConfigParser()
    parser.read(IMPORTLINTER)
    roots = _lines(parser["importlinter"]["root_packages"])
    code = textwrap.dedent(
        """
        import importlib.util, json, sys
        from pathlib import Path
        import grimp
        roots = sys.argv[1:]
        graph = grimp.build_graph(*roots, cache_dir=None)
        files = []
        for module in graph.modules:
            top, *rest = module.split(".")
            for base in importlib.util.find_spec(top).submodule_search_locations:
                path = Path(base).joinpath(*rest)
                files += [str(p.resolve()) for p in (path.with_suffix(".py"), path / "__init__.py") if p.is_file()]
        print(json.dumps(files))
        """
    )
    source_roots = (REPO, REPO / "src", *sorted((REPO / "packages").glob("*/src")))
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(str(p) for p in source_roots)}
    result = subprocess.run(
        [sys.executable, "-c", code, *roots], cwd=REPO, env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    files = {Path(p).relative_to(REPO.resolve()).as_posix() for p in json.loads(result.stdout)}
    assert not {f for f in files if ".venv" in Path(f).parts or "site-packages" in Path(f).parts}
    return files


@pytest.mark.contract
def test_every_production_file_is_in_the_contract_graph() -> None:
    """Every production `.py` under `packages/`, `src/` and `scripts/` is a module the contract sees, or listed."""
    production = {
        p.relative_to(REPO).as_posix()
        for root in ("packages", "src", "scripts")
        for p in (REPO / root).rglob("*.py")
        if "tests" not in p.relative_to(REPO).parts
        and not p.name.endswith("_test.py")
        and p.name != "conftest.py"
        and ".venv" not in p.parts
    }
    assert production - _graph_files() == OUT_OF_GRAPH


def _modules(package: types.ModuleType) -> list[types.ModuleType]:
    found = pkgutil.walk_packages(package.__path__, f"{package.__name__}.")
    names = [package.__name__] + [m.name for m in found if not m.name.endswith("__main__")]
    return [importlib.import_module(n) for n in names]


def _capability(obj: object) -> bool:
    if isinstance(obj, types.ModuleType):
        return obj.__name__.partition(".")[0] in CAPABILITIES
    module = getattr(obj, "__module__", None) or ""
    return module.partition(".")[0] in CAPABILITIES or module == _raw_digest.__name__


@pytest.mark.contract
@pytest.mark.parametrize("package", [senselab_worker, senselab_core], ids=lambda p: p.__name__)
def test_xg1_xg3_no_module_exposes_a_digest_capability(package: types.ModuleType) -> None:
    """No namespace holds a digest module or a function from one or from the raw form, so none is re-exported."""
    hits = [
        f"{module.__name__}.{name}"
        for module in _modules(package)
        for name, obj in vars(module).items()
        if _capability(obj) and module is not _raw_digest
    ]
    assert hits == []


@pytest.mark.contract
@pytest.mark.parametrize(
    "statement",
    [
        "from senselab_worker.hashing import sha256_bytes",
        "from senselab_worker.hashing import sha256_file",
        "from senselab_worker.hashing import tagged_digest",
        "from senselab_worker.canonical import tagged_digest",
        "from senselab_worker.hashing import hashlib",
        "from senselab_worker.digest import hashlib",
        "from senselab_worker.digest import blob_digest",
        "from senselab_worker._raw_digest import hashlib",
    ],
)
def test_xg1_xg3_the_raw_primitives_and_reexports_do_not_exist(statement: str) -> None:
    """The XG-1 helper and the XG-3 re-export no longer import."""
    with pytest.raises(ImportError):
        exec(statement, {})  # noqa: S102


DIGEST_PROBES: dict[str, object] = {"identity_digest": 0, "file_digest": "/nonexistent", "npy_digest": numpy.zeros(1)}
RAW_PROBES: dict[str, object] = {"bytes_digest": b""}
"""Each function an owner module may define, with a valid input to call it on under an unregistered tag."""


def untagged_surface(module: types.ModuleType, probes: dict[str, object], source: str | None = None) -> list[str]:
    """Return each function `source` defines outside `probes`, and each that does not lead with a checked tag.

    The functions are the top-level `def`s of the source text, so a name the interpreter adds to a module (3.14's
    `__annotate__`) is never one of them. `module` supplies the objects to call; `source` defaults to its file.
    """
    text = source if source is not None else Path(str(module.__file__)).read_text()
    defined = {n.name for n in ast.parse(text).body if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)}
    found = []
    for name, obj in vars(module).items():
        if name in defined and inspect.isfunction(obj):
            if name not in probes:
                found.append(f"{name}: not a tagged digest")
                continue
            params = list(inspect.signature(obj).parameters)
            if params[:1] != ["tag"]:
                found.append(f"{name}: first parameter is not the tag")
            try:
                obj("unregistered@0", probes[name])
            except ValueError:
                pass
            else:
                found.append(f"{name}: accepts an unregistered tag")
    return found


@pytest.mark.contract
def test_the_digest_module_defines_only_tagged_digests() -> None:
    """`digest` defines exactly the three typed digests and `_raw_digest` the owners' raw form, each tag-checked."""
    assert digest.__all__ == [
        "BLOB_TAGS",
        "FILE_TAGS",
        "IDENTITY_TAGS",
        "NPY_TAGS",
        "file_digest",
        "identity_digest",
        "npy_digest",
    ]
    assert untagged_surface(digest, DIGEST_PROBES) == []
    assert untagged_surface(_raw_digest, RAW_PROBES) == []


@pytest.mark.contract
@pytest.mark.parametrize(
    ("addition", "expected"),
    [
        ("def _sha(b):\n    import hashlib\n    return hashlib.sha256(b).hexdigest()\n", "_sha: not a tagged digest"),
        ("def sha384_json(x):\n    return x\n", "sha384_json: not a tagged digest"),
        (
            "def identity_digest(tag, value):\n    import hashlib\n    return hashlib.sha256(b'').hexdigest()\n",
            "identity_digest: accepts an unregistered tag",
        ),
        ("def file_digest(data, tag='file@1'):\n    raise ValueError\n", "file_digest: first parameter is not the tag"),
        ("def bytes_digest(tag, data):\n    return ''\n", "bytes_digest: not a tagged digest"),
    ],
)
def test_xg2_a_planted_function_in_the_owner_is_reported(addition: str, expected: str) -> None:
    """A helper added to the owner module, or a digest that stops checking its tag, is reported."""
    planted = types.ModuleType(digest.__name__)
    source = Path(digest.__file__).read_text() + "\n" + addition
    exec(compile(source, digest.__file__, "exec"), vars(planted))  # noqa: S102
    assert expected in untagged_surface(planted, DIGEST_PROBES, source)


@pytest.mark.logic
def test_the_tag_registries_are_disjoint_and_prefix_free() -> None:
    """No identity or blob tag is a prefix of another, and no tag is in both registries."""
    tags = sorted(digest.IDENTITY_TAGS | digest.BLOB_TAGS)
    assert not digest.IDENTITY_TAGS & digest.BLOB_TAGS
    assert all(not b.startswith(a) for a in tags for b in tags if a != b)


@pytest.mark.contract
def test_an_interpreter_generated_function_is_not_part_of_the_surface() -> None:
    """A module-level function the source does not define, as 3.14 adds `__annotate__`, is not reported."""
    planted = types.ModuleType("planted")
    exec("def __annotate__(format):\n    return {}\n", vars(planted))  # noqa: S102
    assert untagged_surface(planted, {}, "x = 1\n") == []
    assert untagged_surface(planted, {}, "def __annotate__(format):\n    return {}\n") == [
        "__annotate__: not a tagged digest"
    ]
