"""Shared AST helper for the promotion-candidate reproductions (F-142, F-148, F-152, F-153,
F-160): confirms a module/function has zero import-level coupling to
`senselab.audio.workflows.audio_analysis`-specific types, which is the operational meaning of
"pure computation belongs in utils/tasks, not the workflow" for these findings.
"""

from __future__ import annotations

import ast
from pathlib import Path


def workflow_imports(path: Path) -> list[str]:
    """Every name this module imports from `senselab.audio.workflows.audio_analysis` (any submodule)."""
    tree = ast.parse(path.read_text())
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and "audio_analysis" in node.module:
            found.append(node.module + "." + ",".join(a.name for a in node.names))
        if isinstance(node, ast.Import):
            for a in node.names:
                if "audio_analysis" in a.name:
                    found.append(a.name)
    return found


def function_param_types(path: Path, func_name: str) -> list[str]:
    """Source text of `func_name`'s parameter annotations, for a function-level check."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return [ast.unparse(a.annotation) for a in node.args.args if a.annotation is not None]
    raise ValueError(f"{func_name!r} not found in {path}")
