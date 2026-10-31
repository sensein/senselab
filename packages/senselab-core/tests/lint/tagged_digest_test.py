"""Digest constructors are imported only by allowlisted modules, which never hash canonical JSON untagged."""

import ast
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
HASH_NAMES = {"sha256", "sha256_bytes", "sha1", "sha3_256", "sha512", "blake2b", "blake2s", "md5", "new", "update"}

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
        "scripts/verify_grid_unification.py",
    ),
}
ALLOWED = {path: reason for reason, paths in ALLOWED_BY_REASON.items() for path in paths}


def _name(func: ast.expr) -> str:
    return func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""


def _top(name: str) -> str:
    return name.partition(".")[0]


def capability_imports(source: str) -> set[str]:
    """Digest-capability modules ``source`` imports, by any import statement, ``import_module`` or ``__import__``.

    An alias of ``import_module`` or ``__import__`` is followed. Only a string-literal argument is read.
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
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and _top(arg.value) in CAPABILITIES:
                found.add(arg.value)
    return found


def _jcs_aliases(tree: ast.AST) -> set[str]:
    aliases = {"jcs_dumps"}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            aliases |= {a.asname for a in node.names if a.name == "jcs_dumps" and a.asname}
    return aliases


def untagged_digests(source: str, module: str) -> list[str]:
    """``module:line`` of each digest call whose data is, or comes from a name assigned from, ``jcs_dumps(...)``.

    Hash calls are ``sha256``-family constructors, ``hashlib.new`` and ``.update``, under any import alias. The
    body of ``tagged_digest`` in ``canonical.py`` is exempt.
    """
    tree = ast.parse(source)
    jcs = _jcs_aliases(tree)
    hashers = set(HASH_NAMES)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in CAPABILITIES:
            hashers |= {a.asname for a in node.names if a.asname}
    found: list[str] = []

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
            if isinstance(child, ast.Call) and _name(child.func) in hashers:
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
    hits = [m for m, p in sorted(sources.items()) if m not in ALLOWED and capability_imports(p.read_text())]
    assert hits == []


@pytest.mark.logic
def test_every_allowlist_entry_still_imports_a_capability() -> None:
    """An allowlisted module exists and still imports a digest module, so the list cannot hold dead entries."""
    for module in ALLOWED:
        assert capability_imports((REPO / module).read_text()), module


@pytest.mark.logic
def test_allowlisted_modules_hash_canonical_json_only_through_the_helper() -> None:
    """In every allowlisted module, no digest takes ``jcs_dumps`` bytes, directly or through a name."""
    hits = [h for m in ALLOWED for h in untagged_digests((REPO / m).read_text(), m)]
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
    """Unrelated modules, and a dynamic import with no literal, are not reported."""
    assert (
        capability_imports("import os, json\nfrom pathlib import Path\nimport importlib\nimportlib.import_module(n)")
        == set()
    )


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
