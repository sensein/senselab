"""The cluster-execution utilities stay generic: no site paths, partitions or project names."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULES = ("venv_worker", "vllm_server", "work_pool", "uv_tools")
# Built from fragments so this file never matches itself.
FORBIDDEN = re.compile("|".join(["/" + "orcd", "pi" + "_satra", "sai" + "ls"]), re.IGNORECASE)


def _files() -> list[Path]:
    package_dir, test_dir = ROOT / "src" / "senselab" / "utils", ROOT / "src" / "tests" / "utils"
    files = []
    for name in MODULES:
        for path in (package_dir / f"{name}.py", test_dir / f"{name}_test.py"):
            if path.is_file():
                files.append(path)
        for directory in (package_dir / name, test_dir / name):
            if directory.is_dir():
                files.extend(p for p in directory.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    return files


def test_no_site_names_in_cluster_exec_modules() -> None:
    """No cluster-execution module or its tests names a cluster path, a partition or a project."""
    files = _files()
    assert any("tests" not in p.relative_to(ROOT).parts for p in files), "no cluster-execution module found"
    hits = [str(p.relative_to(ROOT)) for p in files if FORBIDDEN.search(p.read_text(errors="replace"))]
    assert hits == []
