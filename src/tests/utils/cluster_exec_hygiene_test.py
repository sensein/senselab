"""The cluster-execution utilities stay generic: no site paths, partitions or project names.

The forbidden patterns are not in this repository. ``SENSELAB_HYGIENE_DENY_FILE`` names one or more files
(separated by ``os.pathsep``), each holding one regular expression per line; blank lines and lines starting with
``#`` are ignored. Without it the scan is skipped.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULES = ("venv_worker", "vllm_server", "work_pool", "uv_tools")
DENY_FILE_ENV = "SENSELAB_HYGIENE_DENY_FILE"


def _deny_patterns(value: str) -> list[re.Pattern[str]]:
    """The patterns in every file ``value`` names."""
    patterns = []
    for name in value.split(os.pathsep):
        if not name:
            continue
        for line in Path(name).read_text().splitlines():
            if line.strip() and not line.startswith("#"):
                patterns.append(re.compile(line))
    return patterns


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


def _hits(files: list[Path], patterns: list[re.Pattern[str]]) -> list[str]:
    """The files any pattern matches; only file names and pattern indexes are reported, never the text."""
    hits = []
    for path in files:
        text = path.read_text(errors="replace")
        hits += [f"{path.name}: pattern {i}" for i, pattern in enumerate(patterns) if pattern.search(text)]
    return hits


def test_deny_patterns_are_read_from_every_named_file(tmp_path: Path) -> None:
    """Every named file contributes its patterns; blank and comment lines do not."""
    first, second, module = tmp_path / "first.txt", tmp_path / "second.txt", tmp_path / "module.py"
    first.write_text("# a comment\n\nCANARY-7f3a\n")
    second.write_text("(?i)gpu-[ab]\n")
    module.write_text("partition = 'GPU-A'\n")
    patterns = _deny_patterns(os.pathsep.join([str(first), "", str(second)]))
    assert [p.pattern for p in patterns] == ["CANARY-7f3a", "(?i)gpu-[ab]"]
    assert _hits([module], patterns) == ["module.py: pattern 1"]


def test_no_site_names_in_cluster_exec_modules() -> None:
    """No cluster-execution module or its tests matches a pattern of the deny files."""
    value = os.environ.get(DENY_FILE_ENV)
    if not value:
        pytest.skip(f"{DENY_FILE_ENV} is not set: the forbidden patterns live outside this repository")
    patterns = _deny_patterns(value)
    assert patterns, f"{DENY_FILE_ENV} names no pattern"
    files = _files()
    assert any("tests" not in p.relative_to(ROOT).parts for p in files), "no cluster-execution module found"
    assert _hits(files, patterns) == []
