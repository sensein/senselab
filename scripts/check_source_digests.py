"""Fail when a task's or backend's source changed without a bump of its declared version.

Each ``@task`` class and each ``@backend`` class under ``packages/*/src`` has a source digest: its class body
(docstrings and the ``version`` argument left out), the class named by ``params=``, and, for a backend, the
source of its worker module. The digests
are compared with a baseline: ``packages/senselab/src/senselab/data/source_digests.yaml`` once it exists, else the
tree at the merge-base of ``HEAD`` with the ``2.x`` branch. A declaration whose digest changed and whose version did
not is named.

Usage::

    uv run python scripts/check_source_digests.py
"""

from __future__ import annotations

import ast
import copy
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import NamedTuple

import yaml
from senselab_worker.digest import identity_digest

ROOT = Path(__file__).resolve().parents[1]
RELEASED = ROOT / "packages" / "senselab" / "src" / "senselab" / "data" / "source_digests.yaml"
RELEASED_FORMAT = "source-digests@1"
SOURCE_TAG = "source@1"
LINE_BASE_REFS = ("origin/2.x", "2.x")
DECLARATIONS = {"task": "task", "backend": "backend"}


class Declared(NamedTuple):
    """One declaration: its declared version and its source digest."""

    version: str
    digest: str


def _call_name(call: ast.Call) -> str | None:
    f = call.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def _constant(node: ast.expr | None, module: ast.Module) -> str | None:
    """The string `node` is, or the string the module-level name it is assigned to."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        for stmt in module.body:
            if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == node.id for t in stmt.targets):
                return _constant(stmt.value, ast.Module(body=[], type_ignores=[]))
    return None


def _normalised(node: ast.AST) -> str:
    """`node` unparsed with every docstring removed, so a formatting or docstring edit does not change it."""
    tree = copy.deepcopy(node)
    for part in ast.walk(tree):
        if isinstance(part, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = part.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                if isinstance(body[0].value.value, str):
                    part.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def _worker_source(module_name: str | None, files: Mapping[str, str]) -> str | None:
    if module_name is None:
        return None
    rel = module_name.replace(".", "/")
    for path, text in files.items():
        if path.endswith((f"/src/{rel}.py", f"/src/{rel}/__init__.py")):
            return _normalised(ast.parse(text))
    return None


def _class_source(name: str, module: ast.Module, files: Mapping[str, str]) -> str | None:
    """The normalised class `name`, defined in `module` or imported into it from a module in `files`."""
    for stmt in module.body:
        if isinstance(stmt, ast.ClassDef) and stmt.name == name:
            return _normalised(stmt)
        if isinstance(stmt, ast.ImportFrom) and stmt.module and any((a.asname or a.name) == name for a in stmt.names):
            source = _worker_source(stmt.module, files)
            if source is not None:
                return _class_source(name, ast.parse(source), {})
    return None


def digests_of(files: Mapping[str, str]) -> dict[str, Declared]:
    """Map ``task:<name>`` and ``backend:<task>/<name>`` to the version and source digest of each declaration."""
    found: dict[str, Declared] = {}
    for path in sorted(files):
        module = ast.parse(files[path], filename=path)
        for node in ast.walk(module):
            if not isinstance(node, ast.ClassDef):
                continue
            for deco in node.decorator_list:
                if not isinstance(deco, ast.Call) or _call_name(deco) not in DECLARATIONS:
                    continue
                kind = DECLARATIONS[str(_call_name(deco))]
                args = {k.arg: k.value for k in deco.keywords if k.arg}
                name, version = _constant(args.get("name"), module), _constant(args.get("version"), module)
                if name is None or version is None:
                    continue
                key = f"task:{name}" if kind == "task" else f"backend:{_constant(args.get('task'), module)}/{name}"
                bare = copy.deepcopy(node)
                for d in bare.decorator_list:
                    if isinstance(d, ast.Call):
                        d.keywords = [k for k in d.keywords if k.arg != "version"]
                worker = _worker_source(_constant(args.get("worker"), module), files) if kind == "backend" else None
                params = args.get("params")
                params_source = _class_source(params.id, module, files) if isinstance(params, ast.Name) else None
                digest = identity_digest(
                    SOURCE_TAG, {"declaration": _normalised(bare), "worker": worker, "params": params_source}
                )
                found[key] = Declared(version, digest)
    return found


def changed_without_bump(current: Mapping[str, Declared], baseline: Mapping[str, Declared]) -> list[str]:
    """The keys in both whose digest changed and whose version did not."""
    return sorted(
        key
        for key, now in current.items()
        if key in baseline and baseline[key].digest != now.digest and baseline[key].version == now.version
    )


def _git(root: Path, *args: str) -> str:
    done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False, timeout=120)
    if done.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {done.stderr.strip()}")
    return done.stdout


def merge_base(root: Path) -> str:
    """The merge-base of ``HEAD`` with the 2.x line.

    Raises:
        RuntimeError: if no ref of the 2.x line exists here.
    """
    for ref in LINE_BASE_REFS:
        try:
            return _git(root, "merge-base", "HEAD", ref).strip()
        except RuntimeError:
            continue
    raise RuntimeError(f"none of {LINE_BASE_REFS} exists; fetch the 2.x branch")


def sources_at(root: Path, ref: str | None) -> dict[str, str]:
    """The Python sources under ``packages/*/src``, from the working tree or from the tree at `ref`."""
    if ref is None:
        return {p.relative_to(root).as_posix(): p.read_text() for p in sorted(root.glob("packages/*/src/**/*.py"))}
    names = _git(root, "ls-tree", "-r", "--name-only", ref, "packages").splitlines()
    return {
        n: _git(root, "show", f"{ref}:{n}")
        for n in names
        if n.endswith(".py") and n.startswith("packages/") and n.split("/")[2:3] == ["src"]
    }


def baseline(root: Path, released: Path) -> tuple[dict[str, Declared], str]:
    """The baseline digests and a word for where they came from: the released file when it exists, else the merge-base.

    Raises:
        ValueError: if the released file is not ``source-digests@1``.
    """
    if released.exists():
        data = yaml.safe_load(released.read_text())
        if data.get("format") != RELEASED_FORMAT:
            raise ValueError(f"{released} is not {RELEASED_FORMAT}")
        return {k: Declared(v["version"], v["digest"]) for k, v in data["declarations"].items()}, "released file"
    ref = merge_base(root)
    return digests_of(sources_at(root, ref)), f"merge-base {ref[:12]}"


def main() -> int:
    """Check the working tree against the baseline; print the offenders; return 1 if any."""
    base, origin = baseline(ROOT, RELEASED)
    bad = changed_without_bump(digests_of(sources_at(ROOT, None)), base)
    for key in bad:
        print(f"{key}: source changed since the {origin} without a version bump")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
