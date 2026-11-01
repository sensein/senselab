"""No untagged digest API exists: the import contract and the digest module's surface (ADR-006)."""

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
    for pkg in ("senselab_core", "senselab_core/zz", "senselab_core/runtime"):
        _write(base, f"{pkg}/__init__.py", "")
    _write(base, "scripts/zz_clean.py", "import json\n")


XG1_IMPORTS = {
    "senselab_core/zz/probe_a.py": (
        """
        import hashlib, json
        def record_id(x):
            return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()
        """,
        "hashlib",
    ),
    "scripts/zz_probe.py": ("import hmac\nhmac.new(b'k', b'm', 'sha256')\n", "hmac"),
    "senselab_core/zz/raw.py": (
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
    """Each XG-1 and XG-2 spelling, planted in a package module, a former owner or a script, is reported."""
    _stubs(tmp_path)
    expected = set()
    for rel, (source, module) in XG1_IMPORTS.items():
        _write(tmp_path, rel, source)
        expected.add(f"{rel.removesuffix('.py').replace('/', '.')} -> {module}")
    for name, body in XG2_SPELLINGS.items():
        _write(tmp_path, f"senselab_core/zz/xg2_{name}.py", f"import json\nx = m = None\n{body}\n")
        imported = re.match(r"(?:from |import )(\w+)", body)
        assert imported
        expected.add(f"senselab_core.zz.xg2_{name} -> {imported.group(1)}")
    _write(tmp_path, "senselab_core/runtime/provenance.py", "import hashlib\nhashlib.sha256(b'')\n")
    expected.add("senselab_core.runtime.provenance -> hashlib")
    result = _lint_imports(tmp_path)
    assert result.returncode != 0, result.stdout
    assert _broken(result.stdout) == expected, result.stdout


@pytest.mark.contract
def test_the_contract_passes_a_tree_with_no_digest_import(tmp_path: Path) -> None:
    """The planted tree without a digest import keeps the contract, so the reports above are the plants."""
    _stubs(tmp_path)
    _write(tmp_path, "senselab_core/zz/clean.py", "import json\nfrom senselab_worker.digest import identity_digest\n")
    result = _lint_imports(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.contract
def test_an_allowance_that_matches_no_import_fails_the_contract(tmp_path: Path) -> None:
    """An `ignore_imports` entry naming an import that does not exist is an error, so no allowance is dead."""
    _stubs(tmp_path)
    result = _lint_imports(tmp_path, extra_ignore="senselab_core.zz.gone -> hashlib", alerting="error")
    assert result.returncode != 0
    assert "senselab_core.zz.gone -> hashlib" in result.stdout + result.stderr


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
