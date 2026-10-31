"""Digest constructors are imported only by allowlisted modules, which never hash canonical JSON untagged."""

import ast
import re
from collections.abc import Callable
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
ROOTS = ("packages", "src", "scripts")
CAPABILITIES = {
    "hashlib",
    "_hashlib",
    "hmac",
    "blake3",
    "xxhash",
    "mmh3",
    "Crypto",
    "Cryptodome",
    "cryptography",
    "_md5",
    "_sha1",
    "_sha2",
    "_sha3",
    "_blake2",
}
IMPORTERS = {"import_module", "__import__"}
CANONICAL = "packages/senselab-worker/src/senselab_worker/canonical.py"
HELPER = "tagged_digest"
HASH_NAMES = {"sha256", "sha256_bytes", "sha1", "sha3_256", "sha512", "blake2b", "blake2s", "md5", "new"}
"""Hash constructors and one-shot digest functions; ``.update`` counts only on a receiver one of them made."""
HASH_TYPE = re.compile(r"\b(_Hash|HASH|hashlib\.\w+)\b")
"""An annotation naming a hash object's type: ``hashlib._Hash``, ``_hashlib.HASH`` or another hashlib name."""

ALLOWED_BY_REASON: dict[str, tuple[str, ...]] = {
    "the tagged digest helper": (CANONICAL,),
    "worker content hashing of raw bytes and content roots": (
        "packages/senselab-worker/src/senselab_worker/hashing.py",
    ),
    "file checksums of provenance records": ("packages/senselab-core/src/senselab_core/runtime/provenance.py",),
    "reproduce package: digests of files, inputs, locks and weights": tuple(
        f"src/senselab/utils/reproduce/{m}.py"
        for m in ("build cache cli digest engine env inputs lock lockgen locks run sandbox source weights").split()
    ),
    "1.x subprocess-venv and work-pool code: hashes of files, requirement text and cache keys": (
        "src/senselab/utils/subprocess_venv.py",
        "src/senselab/utils/uv_tools.py",
        "src/senselab/utils/venv_lock.py",
        "src/senselab/utils/venv_worker.py",
        "src/senselab/utils/vllm_server.py",
        "src/senselab/utils/work_pool.py",
        "src/senselab/utils/tasks/cached_inference.py",
    ),
    "1.x audio and text task code: hashes of audio, text and config bytes": (
        "src/senselab/audio/tasks/speech_to_text/crisperwhisper.py",
        "src/senselab/text/tasks/decision_model/ollama.py",
        "src/senselab/text/tasks/pii_detection/redaction_review.py",
        "src/senselab/text/tasks/pii_detection/subprocess_backend.py",
    ),
    "1.x audio_analysis and triage workflows: hashes of recordings, configs and vectors": (
        "src/senselab/audio/workflows/audio_analysis/adaptive/belief.py",
        "src/senselab/audio/workflows/audio_analysis/adaptive/policy.py",
        "src/senselab/audio/workflows/audio_analysis/fuse.py",
        "src/senselab/audio/workflows/audio_analysis/run_config.py",
        "src/senselab/audio/workflows/triage/cohort_stage.py",
        "src/senselab/audio/workflows/triage/config.py",
        "src/senselab/audio/workflows/triage/nodes/second_opinion.py",
        "src/senselab/audio/workflows/triage/recording_vectors.py",
        "src/senselab/audio/workflows/triage/review_page/page.py",
        "src/senselab/audio/workflows/triage/speaker_vectors.py",
        "src/senselab/audio/workflows/triage/task_audio.py",
        "src/senselab/audio/workflows/triage/task_squim.py",
    ),
    "repository scripts: hashes of files and run inputs": (
        "scripts/analyze_audio.py",
        "scripts/build_classifier_ontology_map.py",
        "scripts/check_fixture_budget.py",
        "scripts/extend_cohort.py",
        "scripts/free_speech_review_page.py",
        "scripts/release_gate.py",
        "scripts/verify_grid_unification.py",
    ),
}
ALLOWED = {path: reason for reason, paths in ALLOWED_BY_REASON.items() for path in paths}


DYNAMIC_ALLOWED: dict[str, frozenset[str]] = {
    "packages/senselab-core/src/senselab_core/testing/values.py": frozenset({"<dynamic:module.name>"}),
    "packages/senselab-core/src/senselab_core/testing/contracts.py": frozenset({"<dynamic:module>"}),
    "packages/senselab-worker/src/senselab_worker/launch.py": frozenset({"<dynamic:argv[1]>"}),
    "src/senselab/utils/venv_lock.py": frozenset({"<dynamic:module_name>"}),
    "src/senselab/audio/workflows/audio_analysis/__init__.py": frozenset({"<dynamic:f'{__name__}.{submodule}'>"}),
}
"""Call sites of ``import_module`` / ``__import__`` whose argument is not a literal, by file and argument source."""


def _real(found: set[str]) -> set[str]:
    return {f for f in found if not f.startswith("<dynamic:")}


def _name(func: ast.expr) -> str:
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""


def _top(name: str) -> str:
    return name.partition(".")[0]


def capability_imports(source: str) -> set[str]:
    """Digest-capability modules ``source`` imports, by any import statement, ``import_module`` or ``__import__``.

    An alias of ``import_module`` or ``__import__`` is followed. A string-literal argument is read; any other argument
    is reported as ``<dynamic:SOURCE>``, the form the moved-module guard uses.
    """
    tree = ast.parse(source)
    importers = set(IMPORTERS)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            importers |= {a.asname for a in node.names if a.name in IMPORTERS and a.asname}
        elif isinstance(node, ast.Assign) and _name(node.value) in importers:
            importers |= {t.id for t in node.targets if isinstance(t, ast.Name)}
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names if _top(a.name) in CAPABILITIES}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level and _top(node.module) in CAPABILITIES:
            found.add(node.module)
        elif isinstance(node, ast.Call) and _name(node.func) in importers and node.args:
            arg = node.args[0]
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                found.add(f"<dynamic:{ast.unparse(arg)}>")
            elif _top(arg.value) in CAPABILITIES:
                found.add(arg.value)
    return found


SERIALISERS = {"dumps", "dump", "repr", "str", "pformat", "format"}
"""Calls that turn a value into text or bytes; with ``strict``, data derived from one counts as serialised."""


def _jcs_aliases(tree: ast.AST, strict: bool) -> set[str]:
    aliases = {"jcs_dumps"} | (SERIALISERS if strict else set())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            aliases |= {a.asname for a in node.names if a.name == "jcs_dumps" and a.asname}
    return aliases


def untagged_digests(source: str, module: str, strict: bool = False) -> list[str]:
    """``module:line`` of each digest call whose data is, or comes from a name assigned from, a serialisation.

    The serialisation is ``jcs_dumps(...)``; with ``strict`` also ``json.dumps``, ``pickle.dumps``, ``repr``, ``str``
    and ``format``, with ``.encode()`` of any of them followed through.

    Hash calls are ``sha256``-family constructors, ``hashlib.new`` and ``.update``, under any import alias. The
    body of ``tagged_digest`` in ``canonical.py`` is exempt.
    """
    tree = ast.parse(source)
    jcs = _jcs_aliases(tree, strict)
    hashers = set(HASH_NAMES)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in CAPABILITIES:
            hashers |= {a.asname for a in node.names if a.asname}
    found: list[str] = []
    hash_objects = {
        ast.unparse(t)
        for n in ast.walk(tree)
        if isinstance(n, (ast.Assign, ast.AnnAssign))
        and isinstance(n.value, ast.Call)
        and _name(n.value.func) in hashers
        for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
    }
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            positional = [*n.args.posonlyargs, *n.args.args]
            defaults = dict(zip(reversed(positional), reversed(n.args.defaults)))
            defaults.update({a: d for a, d in zip(n.args.kwonlyargs, n.args.kw_defaults) if d is not None})
            for arg in [*positional, *n.args.kwonlyargs]:
                default = defaults.get(arg)
                annotated = arg.annotation is not None and HASH_TYPE.search(ast.unparse(arg.annotation))
                if annotated or (isinstance(default, ast.Call) and _name(default.func) in hashers):
                    hash_objects.add(arg.arg)

    def is_hash_call(call: ast.Call) -> bool:
        func = call.func
        if isinstance(func, ast.Attribute) and func.attr == "update":
            receiver = func.value
            return ast.unparse(receiver) in hash_objects or (
                isinstance(receiver, ast.Call) and _name(receiver.func) in hashers
            )
        return _name(func) in hashers

    def tainted_names(scope_node: ast.AST) -> set[str]:
        names: set[str] = set()
        for n in ast.walk(scope_node):
            if isinstance(n, ast.Assign) and any(
                isinstance(c, ast.Call) and _name(c.func) in jcs for c in ast.walk(n.value)
            ):
                names |= {t.id for t in n.targets if isinstance(t, ast.Name)}
        return names

    def visit(node: ast.AST, scope: tuple[str, ...], tainted: set[str]) -> None:
        for child in ast.iter_child_nodes(node):
            inner, inner_taint = scope, tainted
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                inner = (*scope, child.name)
                inner_taint = tainted | tainted_names(child)
            if isinstance(child, ast.Call) and is_hash_call(child):
                data = (*child.args, *(k.value for k in child.keywords))
                hit = any(
                    (isinstance(n, ast.Call) and _name(n.func) in jcs) or (isinstance(n, ast.Name) and n.id in tainted)
                    for arg in data
                    for n in ast.walk(arg)
                )
                if hit and not (module == CANONICAL and HELPER in scope):
                    found.append(f"{module}:{child.lineno}")
            visit(child, inner, inner_taint)

    module_level = ast.Module(body=[n for n in tree.body if isinstance(n, ast.Assign)], type_ignores=[])
    visit(tree, (), tainted_names(module_level))
    return found


def _sources() -> list[Path]:
    """Every production source file; test trees are excluded."""
    return [
        p
        for root in ROOTS
        for p in (REPO / root).rglob("*.py")
        if "tests" not in p.relative_to(REPO).parts and not p.name.endswith("_test.py") and ".venv" not in p.parts
    ]


@pytest.mark.logic
def test_only_allowlisted_modules_import_a_digest_capability() -> None:
    """No production module under ``packages/``, ``src/`` or ``scripts/`` imports a digest module unlisted."""
    sources = {p.relative_to(REPO).as_posix(): p for p in _sources()}
    assert CANONICAL in sources
    hits = []
    for module, path in sorted(sources.items()):
        found = capability_imports(path.read_text())
        if module not in ALLOWED and _real(found):
            hits.append(f"{module}: {sorted(_real(found))}")
        hits += [f"{module}: {f}" for f in sorted(found - _real(found)) if f not in DYNAMIC_ALLOWED.get(module, ())]
    assert hits == []


def dead_dynamic_entries(allowed: dict[str, frozenset[str]], sites: Callable[[str], set[str]]) -> list[str]:
    """Return `file: argument` for each allowance whose file is gone or no longer has that dynamic import site."""
    dead = []
    for module, arguments in sorted(allowed.items()):
        path = REPO / module
        found = sites(path.read_text()) if path.is_file() else set()
        dead += [f"{module}: {a}" for a in sorted(arguments - found)]
    return dead


@pytest.mark.logic
def test_every_dynamic_allowance_still_matches_a_site() -> None:
    """Each `DYNAMIC_ALLOWED` entry names a file that still has that dynamic import, so no entry is dead."""
    assert dead_dynamic_entries(DYNAMIC_ALLOWED, capability_imports) == []


@pytest.mark.contract
@pytest.mark.parametrize(
    "planted",
    [
        {"packages/senselab-worker/src/senselab_worker/launch.py": frozenset({"<dynamic:argv[2]>"})},
        {"packages/senselab-worker/src/senselab_worker/no_such_module.py": frozenset({"<dynamic:x>"})},
        {CANONICAL: frozenset({"<dynamic:module>"})},
    ],
)
def test_a_dead_dynamic_allowance_is_reported(planted: dict[str, frozenset[str]]) -> None:
    """An allowance for a site that does not exist, or in a file that does not exist, is reported by name."""
    ((module, arguments),) = planted.items()
    assert dead_dynamic_entries({**DYNAMIC_ALLOWED, **planted}, capability_imports) == [
        f"{module}: {a}" for a in sorted(arguments)
    ]


@pytest.mark.logic
def test_every_allowlist_entry_still_imports_a_capability() -> None:
    """An allowlisted module exists and still imports a digest module, so the list cannot hold dead entries."""
    for module in ALLOWED:
        assert _real(capability_imports((REPO / module).read_text())), module


@pytest.mark.logic
def test_allowlisted_modules_hash_canonical_json_only_through_the_helper() -> None:
    """In every allowlisted module, no digest takes ``jcs_dumps`` bytes, directly or through a name.

    Modules under ``packages/`` are held to the strict form: no serialisation at all feeds a digest.
    """
    hits = [h for m in ALLOWED for h in untagged_digests((REPO / m).read_text(), m, m.startswith("packages/"))]
    assert hits == []


@pytest.mark.logic
def test_the_helper_is_the_one_allowed_site() -> None:
    """``tagged_digest`` hashes JCS bytes in ``canonical.py``, and the same code anywhere else is reported."""
    text = (REPO / CANONICAL).read_text()
    assert "jcs_dumps" in text and untagged_digests(text, CANONICAL) == []
    assert len(untagged_digests(text, "other/canonical.py")) == 1


@pytest.mark.logic
@pytest.mark.parametrize(
    "source",
    [
        "import hashlib",
        "import hashlib as h",
        "import hashlib.sha256",
        "from hashlib import sha256",
        "from hashlib import sha256 as h",
        "from hashlib import *",
        "import hmac",
        "import _hashlib",
        "import blake3",
        "import importlib\nimportlib.import_module('hashlib')",
        "__import__('hashlib')",
        "from importlib import import_module as im\nim('hashlib')",
        "from importlib import import_module\nimport_module('hmac')",
        "def f():\n    import hashlib\n",
        "try:\n    import blake3\nexcept ImportError:\n    pass\n",
    ],
)
def test_a_planted_capability_import_is_found(source: str) -> None:
    """Every import spelling of a digest module is found, at module level or nested."""
    assert capability_imports(source)


@pytest.mark.logic
def test_other_imports_are_not_capabilities() -> None:
    """Unrelated modules and literal imports of them are not reported."""
    src = "import os, json\nfrom pathlib import Path\nimport importlib\nimportlib.import_module('json')"
    assert capability_imports(src) == set()


@pytest.mark.logic
@pytest.mark.parametrize(
    "body",
    [
        "hashlib.sha256(jcs_dumps(x)).hexdigest()",
        "sha256(jcs_dumps(x)).hexdigest()",
        "sha256_bytes(jcs_dumps(x))",
        "hashlib.sha256(b'tag' + jcs_dumps(x)).hexdigest()",
        "d = jcs_dumps(x)\n    return hashlib.sha256(d).hexdigest()",
        "d = jcs_dumps(x)\n    return sha256_bytes(d)",
        "return h(jcs_dumps(x))",
        "return j(x) and hashlib.sha256(j(x)).hexdigest()",
        "return hashlib.new('sha256', jcs_dumps(x)).hexdigest()",
        "m = hashlib.sha256()\n    m.update(jcs_dumps(x))\n    return m.hexdigest()",
        "d = jcs_dumps(x)\n    m = hashlib.blake2b()\n    m.update(d)",
        "hashlib.sha256().update(jcs_dumps(x))",
        "self.m = h()\n    self.m.update(jcs_dumps(x))",
        "return hashlib.sha3_256(jcs_dumps(x)).hexdigest()",
    ],
)
def test_a_planted_untagged_digest_is_found(body: str) -> None:
    """Direct, aliased, variable-carried, ``new``, ``update`` and other-family digests of JCS bytes are found."""
    header = "from hashlib import sha256 as h\nfrom senselab_worker.canonical import jcs_dumps as j\n"
    source = f"{header}def f(x):\n    {body}\n"
    assert untagged_digests(source, "pkg/mod.py")
    if "j(x)" not in body:
        assert untagged_digests(source.replace("jcs_dumps(x)", "j(x)"), "pkg/mod.py")


@pytest.mark.logic
@pytest.mark.parametrize(
    "body",
    [
        "return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()",
        "d = json.dumps(x)\n    return hashlib.sha256(d.encode()).hexdigest()",
        "return hashlib.sha256(pickle.dumps(x)).hexdigest()",
        "return hashlib.sha256(repr(x).encode()).hexdigest()",
        "return hashlib.sha256(str(x).encode()).hexdigest()",
        "m = hashlib.sha256()\n    m.update(json.dumps(x).encode())\n    return m.hexdigest()",
        "t = repr(x)\n    return sha256_bytes(t.encode('utf-8'))",
    ],
)
def test_strict_mode_finds_any_serialisation_fed_to_a_digest(body: str) -> None:
    """Under ``strict``, ``json.dumps``, ``pickle.dumps``, ``repr`` and ``str`` are found, even via ``.encode()``."""
    source = f"import hashlib\ndef f(x):\n    {body}\n"
    assert untagged_digests(source, "pkg/mod.py", strict=True)
    assert untagged_digests(source, "pkg/mod.py") == []


@pytest.mark.contract
@pytest.mark.parametrize(
    ("body", "hashed"),
    [
        ("d = {}\n    d.update({str(k): v for k, v in x.items()})", False),
        ("row = dict(x)\n    row.update(name=str(x))", False),
        ("m = hashlib.sha256()\n    other = {}\n    other.update(str(x))", False),
        ("x.update(str(x))", False),
        ("m = hashlib.sha256()\n    m.update(str(x).encode())", True),
        ("m = hashlib.blake2b()\n    m.update(json.dumps(x).encode())", True),
        ("hashlib.sha256().update(repr(x).encode())", True),
        ("pass\ndef g(h: hashlib._Hash, x):\n    h.update(str(x).encode())", True),
        ("pass\ndef g(h: '_hashlib.HASH', x):\n    h.update(json.dumps(x).encode())", True),
        ("pass\ndef g(x, h=hashlib.sha256()):\n    h.update(repr(x).encode())", True),
        ("pass\ndef g(d: dict, x):\n    d.update(str(x))", False),
        ("pass\ndef g(x, d=dict()):\n    d.update(str(x))", False),
    ],
)
def test_strict_mode_counts_update_only_on_a_hash_object(body: str, hashed: bool) -> None:
    """``.update`` fed by a serialisation is a digest only when its receiver was made by a hash constructor."""
    source = f"import hashlib, json\ndef f(x):\n    {body}\n"
    assert bool(untagged_digests(source, "src/mod.py", strict=True)) is hashed


@pytest.mark.logic
def test_strict_mode_passes_digests_of_raw_bytes() -> None:
    """Hashing file chunks or a ``bytes`` value is not a serialisation."""
    src = "def f(h, chunk, b):\n    h.update(chunk)\n    return sha256_bytes(b)\n"
    assert untagged_digests(src, "pkg/mod.py", strict=True) == []


@pytest.mark.logic
def test_provenance_ids_are_tagged_digests() -> None:
    """The provenance store derives ids from `tagged_digest("provenance-id@2", ...)`, not a bare hash."""
    from senselab_core.runtime import provenance
    from senselab_worker.canonical import tagged_digest

    assert provenance._digest(["r", "span"], ("a", "b")) == tagged_digest("provenance-id@2", ["r", "span"])[:16]


@pytest.mark.logic
def test_a_module_level_untagged_digest_is_found() -> None:
    """A digest of a module-level name assigned from ``jcs_dumps`` is found."""
    assert untagged_digests("d = jcs_dumps(x)\nsha256(d)\n", "pkg/mod.py") == ["pkg/mod.py:2"]


@pytest.mark.logic
def test_hashing_other_bytes_and_serialising_without_hashing_pass() -> None:
    """A digest of non-JCS bytes, and ``jcs_dumps`` used for something other than a digest, are not reported."""
    src = (
        "def f(a, x):\n    sha256(b'x')\n    sha256_bytes(canonical_npy_bytes(a))\n    write(jcs_dumps(x))\n"
        "    tagged_digest('t', x)\n    d = jcs_dumps(x)\n    write(d)\n"
    )
    assert untagged_digests(src, "pkg/mod.py") == []
