"""Tests for ``scripts/triage_review_serve.py``: the source route reads only under its allowed roots."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("triage_review_serve", ROOT / "scripts" / "triage_review_serve.py")
assert _spec is not None and _spec.loader is not None
serve = importlib.util.module_from_spec(_spec)
sys.modules["triage_review_serve"] = serve
_spec.loader.exec_module(serve)


def test_the_source_route_maps_an_absolute_path_under_an_allowed_root(tmp_path: Path) -> None:
    """A raw recording outside the corpus is reachable by its absolute path, URL-decoded."""
    raw = tmp_path / "data" / "sub a" / "rec.wav"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"RIFF")
    roots = [(tmp_path / "data").resolve()]
    url = "/_source" + str(raw.resolve()).replace(" ", "%20")
    assert serve.source_path(url, roots) == raw.resolve()


def test_the_source_route_refuses_paths_outside_its_roots(tmp_path: Path) -> None:
    """Anything outside the allowed roots, including a climb out of one, is refused."""
    roots = [(tmp_path / "data").resolve()]
    assert serve.source_path("/_source/etc/passwd", roots) is None
    assert serve.source_path("/_source" + str(tmp_path / "data" / ".." / "secret"), roots) is None
    assert serve.source_path("/review/index.html", roots) is None
