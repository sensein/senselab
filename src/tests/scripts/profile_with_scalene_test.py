"""Smoke test for scripts/profile_with_scalene.py.

Skipped automatically when Scalene is not installed (the default `uv sync` install
path), so the existing CI test suite is unaffected by this addition.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
WRAPPER = REPO_ROOT / "scripts" / "profile_with_scalene.py"

scalene_available = importlib.util.find_spec("scalene") is not None


@pytest.mark.skipif(not scalene_available, reason="scalene not installed")
def test_wrapper_produces_html_report(tmp_path: Path) -> None:
    """The wrapper produces a standalone HTML report for a trivial script."""
    target = tmp_path / "tiny.py"
    target.write_text(
        "import time\ntime.sleep(0.05)\n_ = [i * i for i in range(2000)]\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "reports"

    result = subprocess.run(
        [
            sys.executable,
            str(WRAPPER),
            "--output-dir",
            str(out_dir),
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert result.returncode == 0, f"wrapper exit={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    htmls = list(out_dir.glob("tiny_*.html"))
    assert len(htmls) == 1, f"expected one HTML report, found: {htmls}"
    assert htmls[0].stat().st_size > 1000, "HTML report is suspiciously small"


@pytest.mark.skipif(not scalene_available, reason="scalene not installed")
def test_wrapper_json_format(tmp_path: Path) -> None:
    """With --format json, the wrapper produces a JSON profile and skips the view step."""
    target = tmp_path / "tiny.py"
    target.write_text("x = sum(i * i for i in range(1000))\n", encoding="utf-8")
    out_dir = tmp_path / "reports"

    result = subprocess.run(
        [
            sys.executable,
            str(WRAPPER),
            "--format",
            "json",
            "--output-dir",
            str(out_dir),
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    jsons = list(out_dir.glob("tiny_*.json"))
    assert len(jsons) == 1, f"expected one JSON profile, found: {jsons}"


@pytest.mark.skipif(not scalene_available, reason="scalene not installed")
def test_wrapper_handles_notebook_with_ipython_imports(tmp_path: Path) -> None:
    """The wrapper profiles a notebook whose cells use IPython imports + magics.

    Verifies that the injected IPython stub covers `from IPython.display import display`
    (a common nbconvert output pattern) and `get_ipython().run_line_magic(...)`
    (how nbconvert translates `%matplotlib inline` and similar).
    """
    nb_path = tmp_path / "needs_ipython.ipynb"
    nb_path.write_text(
        "{\n"
        ' "cells": [{\n'
        '  "cell_type": "code",\n'
        '  "metadata": {},\n'
        '  "source": [\n'
        '    "from IPython.display import display\\n",\n'
        '    "get_ipython().run_line_magic(\\"matplotlib\\", \\"inline\\")\\n",\n'
        '    "import time\\n",\n'
        '    "time.sleep(0.05)\\n",\n'
        '    "display(\\"hello\\")\\n",\n'
        '    "_ = [i * i for i in range(1000)]\\n"\n'
        "  ],\n"
        '  "outputs": [],\n'
        '  "execution_count": null\n'
        " }],\n"
        ' "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},\n'
        '              "language_info": {"name": "python"}},\n'
        ' "nbformat": 4, "nbformat_minor": 4\n'
        "}\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "reports"

    result = subprocess.run(
        [
            sys.executable,
            str(WRAPPER),
            "--output-dir",
            str(out_dir),
            str(nb_path),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert result.returncode == 0, f"wrapper exit={result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
    htmls = list(out_dir.glob("needs_ipython_*.html"))
    assert len(htmls) == 1, f"expected one HTML report, found: {htmls}"


@pytest.mark.skipif(not scalene_available, reason="scalene not installed")
def test_wrapper_rejects_missing_target(tmp_path: Path) -> None:
    """The wrapper exits with code 4 when the target does not exist."""
    result = subprocess.run(
        [
            sys.executable,
            str(WRAPPER),
            str(tmp_path / "does_not_exist.py"),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 4, f"expected exit 4 for missing target, got {result.returncode}"
    assert "does not exist" in result.stderr.lower()


@pytest.mark.skipif(not scalene_available, reason="scalene not installed")
def test_wrapper_rejects_conflicting_scope_flags(tmp_path: Path) -> None:
    """The mutually-exclusive group rejects --scope + --no-thirdparty together."""
    target = tmp_path / "tiny.py"
    target.write_text("pass\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            str(WRAPPER),
            "--scope",
            "foo",
            "--no-thirdparty",
            str(target),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0, "expected non-zero exit for conflicting flags"
