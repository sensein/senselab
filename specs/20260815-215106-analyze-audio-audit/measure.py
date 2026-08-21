"""Re-derive the audit's layer measurements.

The audit slices `analyze_audio` by layer rather than by module, and every sweep's scope is
defined by this split, so the numbers have to come from one place that anyone can re-run. A
sweep that recomputed them differently would silently audit a different surface than the one
the design describes.

Run from the repository root:

    uv run --no-sync python specs/20260815-215106-analyze-audio-audit/measure.py

Never run it with a working directory inside ``audio_analysis/``: that package contains a
``types.py`` which shadows the stdlib module of the same name, and every import then fails with
a circular-import error that names ``weakref`` rather than the real cause.
"""

from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path("src/senselab/audio/workflows/audio_analysis")


def _counts(path: pathlib.Path) -> tuple[int, int, int, bool]:
    """Return ``(code, docstring, comment, imports_a_task)`` for one file.

    Docstrings are counted via the AST rather than by matching quotes, because a triple-quoted
    string used as a value is not a docstring and a regex cannot tell the difference.
    """
    src = path.read_text(encoding="utf-8")
    lines = src.splitlines()
    tree = ast.parse(src)
    doc = 0
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            text = ast.get_docstring(node, clean=False)
            if text:
                doc += text.count("\n") + 1
    blank = sum(1 for line in lines if not line.strip())
    comment = sum(1 for line in lines if line.strip().startswith("#"))
    code = len(lines) - blank - comment - doc
    imports_task = "senselab.audio.tasks" in src or "senselab.utils.tasks" in src
    return code, doc, comment, imports_task


def main() -> int:
    """Print the layer table and the per-file breakdown."""
    if not ROOT.is_dir():
        print(f"not found: {ROOT} (run from the repository root)", file=sys.stderr)
        return 1

    orch: list[tuple[int, str]] = []
    comp: list[tuple[int, str]] = []
    total_doc = total_comment = 0
    for path in sorted(ROOT.rglob("*.py")):
        if "__pycache__" in str(path):
            continue
        code, doc, comment, imports_task = _counts(path)
        total_doc += doc
        total_comment += comment
        rel = str(path.relative_to(ROOT))
        (orch if imports_task else comp).append((code, rel))

    orch.sort(reverse=True)
    comp.sort(reverse=True)
    orch_code = sum(c for c, _ in orch)
    comp_code = sum(c for c, _ in comp)
    prose = total_doc + total_comment

    print(f"orchestration : {len(orch):3d} files  {orch_code:6d} code")
    print(f"computation   : {len(comp):3d} files  {comp_code:6d} code")
    print(f"prose         :              {prose:6d}  ({total_doc} docstring + {total_comment} comment)")
    print(f"prose:code    : {prose / max(orch_code + comp_code, 1):.2f} : 1")
    print()
    print("orchestration files:")
    for code, name in orch:
        print(f"  {code:6d}  {name}")
    print()
    print("computation files (top 25):")
    for code, name in comp[:25]:
        print(f"  {code:6d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
