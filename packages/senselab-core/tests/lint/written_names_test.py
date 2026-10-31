"""Every file written under a storage root is named by a listed grammar, and nothing writes by another API.

The modules that write under a storage root are `SCANNED`: the store, fetch cache, locks, storage, provenance, BEP-028
and run-manifest modules, and 1.x `reproduce.build` and `reproduce.cache`. The runtime modules' `WRITTEN_NAMES`
(with `storage.WRITTEN_NAMES`) are the list of grammars. In every scanned module, each call to a `StorageDir` write
method passes `grammar=` naming one of those grammars, and no other call writes, renames, links, removes, makes
directories, changes modes or times, or opens a file for writing (`os`, `io`, `shutil`, `tempfile`, builtin `open`
and the `pathlib.Path` methods, under any import alias, assignment or `getattr` spelling the shared import resolver
folds) outside `StorageDir` itself, except the counted, reasoned `ALLOWED_RAW` sites (H-R40, H-R42).
"""

import ast
import importlib.util
import re
from collections import Counter
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from senselab_core.runtime import fetch, locks, revision, storage, store

REPO = Path(__file__).resolve().parents[4]
RESOLVER_PATH = REPO / "src" / "tests" / "utils" / "import_resolver_fixture.py"
RUNTIME = REPO / "packages" / "senselab-core" / "src" / "senselab_core" / "runtime"
REPRODUCE = REPO / "src" / "senselab" / "utils" / "reproduce"
GRAMMAR_MODULES: dict[str, ModuleType] = {
    "storage": storage,
    "store": store,
    "fetch": fetch,
    "locks": locks,
    "revision": revision,
}
SCANNED: dict[str, Path] = {
    **{name: RUNTIME / f"{name}.py" for name in GRAMMAR_MODULES},
    "provenance": RUNTIME / "provenance.py",
    "bep028": RUNTIME / "bep028.py",
    "reproduce.build": REPRODUCE / "build.py",
    "reproduce.cache": REPRODUCE / "cache.py",
}
WRITE_METHODS = frozenset({"open_new", "write_new", "write_atomically", "symlink", "link", "rename", "open_or_create"})
RAW_FUNCTIONS = frozenset(
    "os." + name
    for name in (
        "open",
        "rename",
        "renames",
        "replace",
        "link",
        "symlink",
        "mkdir",
        "makedirs",
        "chmod",
        "fchmod",
        "lchmod",
        "chown",
        "fchown",
        "lchown",
        "utime",
        "truncate",
        "ftruncate",
        "mkfifo",
        "mknod",
        "remove",
        "unlink",
        "rmdir",
        "removedirs",
    )
)
READ_ONLY = frozenset({"shutil.which", "shutil.disk_usage", "shutil.get_terminal_size", "tempfile.gettempdir"})
OPENERS = frozenset({"open", "io.open", "builtins.open"})
PATH_WRITES = frozenset(
    {"write_text", "write_bytes", "touch", "mkdir", "unlink", "rmdir", "chmod", "lchmod", "symlink_to", "hardlink_to"}
)
"""`pathlib.Path` methods that write; `open`, `rename` and `replace` are judged by their arguments."""
ALLOWED_GRAMMARS = {("storage", "write_atomically"): {"_CALLER_FILE"}}
"""Writers outside any storage root: `storage.write_atomically` writes the caller's own file."""
ALLOWED_RAW: dict[tuple[str, str], dict[str, int]] = {
    ("storage", "set_mode"): {"os.fchmod": 1},
    ("bep028", "write_bep028_files"): {".mkdir": 1, ".write_text": 1},
    ("fetch", "_urlopen"): {".open": 1},
    ("reproduce.build", "_copy_verified"): {".mkdir": 1, "shutil.copyfile": 1},
    ("reproduce.build", "_place"): {
        "os.chmod": 1,
        "os.link": 1,
        "os.replace": 1,
        "os.unlink": 1,
        "tempfile.mkstemp": 1,
    },
    ("reproduce.build", "_store"): {".mkdir": 1},
    ("reproduce.build", "_unpack"): {".chmod": 1, ".mkdir": 1},
    ("reproduce.build", "_write_atomic"): {"os.chmod": 1, "os.replace": 1, "os.unlink": 1, "tempfile.mkstemp": 1},
    ("reproduce.build", "base_environment"): {".mkdir": 2},
    ("reproduce.build", "build_wheel"): {".mkdir": 1, "tempfile.TemporaryDirectory": 1},
    ("reproduce.build", "fetch_sdist"): {".mkdir": 1, "os.replace": 1, "os.unlink": 1, "tempfile.mkstemp": 1},
    ("reproduce.build", "safe_extract"): {".mkdir": 1},
    ("reproduce.build", "stage_wheel"): {".mkdir": 1, ".write_bytes": 1},
    ("reproduce.cache", "_fsync_dir"): {"os.open": 1},
    ("reproduce.cache", "_remove"): {"os.unlink": 1, "shutil.rmtree": 1},
    ("reproduce.cache", "_wipe_keeping"): {".mkdir": 1, "os.rename": 2},
    ("reproduce.cache", "claim"): {".mkdir": 1},
    ("reproduce.cache", "link_run"): {".mkdir": 1, ".unlink": 1, "os.replace": 1, "os.symlink": 1},
    ("reproduce.cache", "reset_result_dir"): {".mkdir": 1, "shutil.rmtree": 1},
    ("reproduce.cache", "write_summary"): {".unlink": 1, "os.open": 1, "os.replace": 1},
}
"""Raw calls allowed outside `StorageDir`, by module and function, each with how many sites it may have.

- `storage.set_mode` changes a mode through the descriptor it is given.
- `fetch._urlopen`: `.open` is the HTTPS opener's request, not a file.
- `bep028.write_bep028_files`: BEP-028 sidecars in the caller's output tree by path; M-6, deferred to the 1.x
  cleanup by H-R41.
- `reproduce.build.*` and `reproduce.cache.*`: 1.x writes by path in the reproduce trees (sdist and wheel caches,
  venv bases, results roots and run links); moving them onto `StorageDir` is the same 1.x cleanup. Their locks
  already go through `SharedFileLock` (`build.file_lock`, `cache.claim`).

A new raw write anywhere in `SCANNED`, or one more site of a listed call, fails the check.
"""


def _resolver() -> ModuleType:
    spec = importlib.util.spec_from_file_location("import_resolver_fixture", RESOLVER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RESOLVER = _resolver()


def written_names() -> frozenset[re.Pattern[str]]:
    """Return the union of the runtime modules' `WRITTEN_NAMES`."""
    found: set[re.Pattern[str]] = set()
    for module in GRAMMAR_MODULES.values():
        found |= module.WRITTEN_NAMES
    return frozenset(found)


def _scopes(tree: ast.Module) -> dict[ast.AST, tuple[str, ...]]:
    """Map every node to the qualified name of the function or class it is in."""
    where: dict[ast.AST, tuple[str, ...]] = {}

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            inner = (*scope, child.name) if isinstance(child, (ast.FunctionDef, ast.ClassDef)) else scope
            where[child] = inner
            visit(child, inner)

    visit(tree, ())
    return where


def _mode(call: ast.Call, position: int) -> str | None:
    """The literal mode of an `open` call, `"r"` when it has none, None when it is not a literal."""
    node = (
        call.args[position]
        if len(call.args) > position
        else next((k.value for k in call.keywords if k.arg == "mode"), None)
    )
    if node is None:
        return "r"
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _writes(mode: str | None) -> bool:
    return mode is None or any(c in mode for c in "wax+")


def raw_write(call: ast.Call, bound: dict[str, str]) -> str | None:
    """Return the write `call` makes by an API other than `StorageDir`, as `<dotted name>` or `.<method>`, else None."""
    func = call.func
    target = RESOLVER.resolved(func, bound)
    if target is None and isinstance(func, ast.Name) and func.id == "open" and "open" not in bound:
        target = "open"
    if target is not None:
        if target in OPENERS:
            return target if _writes(_mode(call, 1)) else None
        if target in RAW_FUNCTIONS:
            return target
        if target.startswith(("shutil.", "tempfile.")) and target not in READ_ONLY:
            return target
        return None
    if not isinstance(func, ast.Attribute):
        return None
    attr = func.attr
    if attr in PATH_WRITES:
        return "." + attr
    if attr == "open" and _writes(_mode(call, 0)):
        return ".open"
    if attr in ("rename", "replace") and len(call.args) == 1 and not call.keywords:
        return "." + attr
    return None


def violations(
    source: str, module: str, path: Path, grammars: frozenset[re.Pattern[str]], namespace: object
) -> list[str]:
    """Return `module:line message` for each write by another API, or a `StorageDir` write without a listed grammar."""
    tree = ast.parse(source)
    where = _scopes(tree)
    bound = RESOLVER.bindings(tree, path)
    found: list[str] = []
    counted: Counter[tuple[str, str]] = Counter()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        scope = where.get(node, ())
        qualname = ".".join(scope)
        if module == "storage" and scope[:1] == ("StorageDir",):
            continue
        raw = raw_write(node, bound)
        if raw is not None:
            counted[(qualname, raw)] += 1
            if counted[(qualname, raw)] > ALLOWED_RAW.get((module, qualname), {}).get(raw, 0):
                found.append(f"{module}:{node.lineno} {qualname or '<module>'} {raw}")
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in WRITE_METHODS:
            continue
        if RESOLVER.resolved(func, bound) is not None:
            continue
        grammar = next((k.value for k in node.keywords if k.arg == "grammar"), None)
        if not isinstance(grammar, ast.Name):
            found.append(f"{module}:{node.lineno} .{func.attr} without a named grammar=")
            continue
        if grammar.id in ALLOWED_GRAMMARS.get((module, qualname), set()):
            continue
        if getattr(namespace, grammar.id, None) not in grammars:
            found.append(f"{module}:{node.lineno} .{func.attr} grammar={grammar.id} is not in WRITTEN_NAMES")
    return found


def _namespace(module: str) -> object:
    return GRAMMAR_MODULES.get(module, SimpleNamespace())


@pytest.mark.logic
def test_every_write_names_a_listed_grammar_and_goes_through_storage_dir() -> None:
    """In every module writing under a storage root, each write passes a listed grammar or is an allowed raw site."""
    grammars = written_names()
    found = [
        hit
        for name, path in SCANNED.items()
        for hit in violations(path.read_text(), name, path, grammars, _namespace(name))
    ]
    assert found == []


@pytest.mark.logic
def test_every_raw_allowance_still_matches_its_sites() -> None:
    """Each `ALLOWED_RAW` entry is used exactly as many times as it allows, so a removed site shrinks the list."""
    unused: list[str] = []
    for (module, qualname), calls in ALLOWED_RAW.items():
        path = SCANNED[module]
        tree = ast.parse(path.read_text())
        where = _scopes(tree)
        bound = RESOLVER.bindings(tree, path)
        seen: Counter[str] = Counter(
            raw
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ".".join(where.get(node, ())) == qualname
            and (raw := raw_write(node, bound)) is not None
        )
        unused.extend(
            f"{module} {qualname} {call}: {seen[call]} of {n}" for call, n in calls.items() if seen[call] != n
        )
    assert unused == []


@pytest.mark.logic
@pytest.mark.parametrize(
    "line",
    [
        "d.write_new(n, b'', 0o600)",
        "d.write_new(n, b'', 0o600, grammar=re.compile('.*'))",
        "d.rename(a, d, b, grammar=UNLISTED)",
        "d.write_atomically(n, b'', grammar=UNLISTED)",
        "os.rename(a, b)",
        "os.chmod(p, 0o600)",
        "os.utime(p)",
        "os.open(p, os.O_CREAT)",
        "shutil.copy(a, b)",
        "tempfile.mkstemp(dir=p)",
        "tempfile.NamedTemporaryFile(dir=p)",
        "io.open(p, 'w')",
        "open(p, 'w')",
        "open(p, mode)",
        "Path(p).write_text('x')",
        "Path(p).write_bytes(b'x')",
        "Path(p).touch()",
        "Path(p).mkdir()",
        "Path(p).unlink()",
        "Path(p).open('wb')",
        "Path(p).replace(q)",
        "Path(p).rename(q)",
        "o.rename(a, b)",
        "r(a, b)",
        "getattr(os, 'rename')(a, b)",
        "alias(a, b)",
    ],
)
def test_the_check_finds_every_unlisted_or_raw_write(line: str) -> None:
    """A write without a grammar, with an unlisted one, or through a raw API under any spelling is reported."""
    listed_grammar = re.compile("y")
    namespace = SimpleNamespace(UNLISTED=re.compile("x"), LISTED=listed_grammar)
    header = (
        "import io, os, re, shutil, tempfile\nimport os as o\nfrom os import rename as r\nfrom pathlib import Path\n"
        "alias = os.replace\n"
    )
    source = f"{header}def f(d, n, a, b, p, q, mode):\n    {line}\n"
    assert violations(source, "m", Path("m.py"), frozenset({listed_grammar}), namespace), line


@pytest.mark.logic
@pytest.mark.parametrize(
    "line",
    [
        "d.write_new(n, b'', 0o600, grammar=LISTED)",
        "open(p)",
        "open(p, 'rb')",
        "Path(p).open()",
        "Path(p).read_text()",
        "s.replace('a', 'b')",
        "tempfile.gettempdir()",
        "shutil.which('x')",
    ],
)
def test_the_check_passes_reads_and_listed_writes(line: str) -> None:
    """Reads, string methods and writes through a listed grammar are not reported."""
    listed_grammar = re.compile("y")
    namespace = SimpleNamespace(LISTED=listed_grammar)
    source = f"import os, shutil, tempfile\nfrom pathlib import Path\ndef f(d, n, p, s):\n    {line}\n"
    assert violations(source, "m", Path("m.py"), frozenset({listed_grammar}), namespace) == [], line
