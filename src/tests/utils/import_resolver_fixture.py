"""One import resolver for the repository guards: what a source file imports, by any static spelling.

Loaded by path (`load()`), so a guard copied into a miniature tree with this file beside it resolves the same way.
"""

import ast
import importlib.util
import json
from collections import Counter
from collections.abc import Iterator
from pathlib import Path


def module_name(path: Path) -> tuple[str, bool]:
    """The dotted module name of ``path`` (found by walking up through package markers) and whether it is a package."""
    parts = [] if path.stem == "__init__" else [path.stem]
    parent = path.parent
    while (parent / "__init__.py").exists() or (parent / "__init__.pyi").exists():
        parts.insert(0, parent.name)
        parent = parent.parent
    return ".".join(parts), path.stem == "__init__"


def sources(path: Path) -> Iterator[str]:
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


def fold(node: ast.AST) -> str | None:
    """The string ``node`` evaluates to when built from literals, ``+`` and f-strings of constants, else ``None``."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = fold(node.left), fold(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        pieces = []
        for value in node.values:
            piece = (
                fold(value.value)
                if isinstance(value, ast.FormattedValue) and value.format_spec is None
                else (fold(value) if isinstance(value, ast.Constant) else None)
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


def _argument(node: ast.Call, position: int, keyword: str) -> ast.expr | None:
    if len(node.args) > position:
        return node.args[position]
    return next((k.value for k in node.keywords if k.arg == keyword), None)


def _dynamic_target(node: ast.Call, importers: set[str] | None = None) -> ast.expr | None:
    """The module argument of ``importlib.import_module(x)``, ``import_module(x)`` or ``__import__(x)``, or ``None``."""
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else func.id if isinstance(func, ast.Name) else ""
    if name not in (importers or {"import_module", "__import__"}):
        return None
    return _argument(node, 0, "name")


def _dynamic_module(node: ast.Call, target: ast.expr) -> str:
    """The absolute module ``node`` imports, resolving a relative name against its ``package`` argument."""
    folded = fold(target)
    if folded is None:
        return f"<dynamic:{ast.unparse(target)}>"
    if not folded.startswith("."):
        return folded
    package = _argument(node, 1, "package")
    anchor = fold(package) if package is not None else None
    if anchor is None:
        return f"<dynamic:{ast.unparse(target)}, {ast.unparse(package) if package is not None else None}>"
    try:
        return importlib.util.resolve_name(folded, anchor)
    except ImportError:
        return f"<dynamic:{ast.unparse(target)}, {anchor}>"


def _chain(node: ast.AST) -> list[str] | None:
    """``[root, attr, ...]`` of a ``Name.attr`` chain, where ``getattr(x, "attr")`` with a folded name is ``x.attr``."""
    parts: list[str] = []
    while True:
        if isinstance(node, ast.Attribute):
            parts.insert(0, node.attr)
            node = node.value
            continue
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and (attr := fold(node.args[1])) is not None
        ):
            parts[:0] = attr.split(".")
            node = node.args[0]
            continue
        break
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
            found.add(_dynamic_module(node, target))
    for node in ast.walk(tree):
        is_getattr = isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr"
        if is_getattr or (isinstance(node, ast.Attribute) and not isinstance(getattr(node, "ctx", None), ast.Store)):
            chain = _chain(node)
            if chain and chain[0] in bound:
                dotted = bound[chain[0]].split(".") + chain[1:]
                found.update(".".join(dotted[:i]) for i in range(1, len(dotted) + 1))
    return found


def imports(path: Path) -> set[str]:
    """Absolute module names that ``path`` imports, resolving relative imports against its package.

    Sources are ``.py`` and ``.pyi`` files and the code cells of ``.ipynb`` files. It covers ``import a.b``,
    ``from a import b`` (as ``a.b`` and ``a``), ``from a import *``, ``importlib.import_module(x)`` and
    ``__import__(x)`` when ``x`` folds to a constant, and attribute chains rooted at an imported name. A
    call whose argument does not fold appears as ``<dynamic:SOURCE>``.
    """
    module, is_package = module_name(path)
    found: set[str] = set()
    for text in sources(path):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            found.add(f"<unparseable:{path.name}>")
            continue
        found |= _imports_of(tree, module, is_package, path.name)
    return found


def dynamic_sites(path: Path) -> Counter[str]:
    """Each ``import_module`` / ``__import__`` call in ``path`` whose argument does not fold, counted per argument."""
    sites: Counter[str] = Counter()
    for text in sources(path):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        importers = _importer_aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (target := _dynamic_target(node, importers)) is not None:
                name = _dynamic_module(node, target)
                if name.startswith("<dynamic:"):
                    sites[name] += 1
    return sites
