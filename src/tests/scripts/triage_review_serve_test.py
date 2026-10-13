"""Tests for ``scripts/triage_review_serve.py``: the source route reads only under its allowed roots, and files seek."""

from __future__ import annotations

import functools
import http.client
import http.server
import importlib.util
import sys
import threading
from pathlib import Path
from typing import Iterator

import pytest

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


@pytest.fixture
def served(tmp_path: Path) -> Iterator[tuple[str, bytes, Path]]:
    """The server on a corpus root and one source root, in a thread; its base URL, a body and the raw file."""
    corpus = tmp_path / "corpus"
    (corpus / "out").mkdir(parents=True)
    body = bytes(range(256)) * 40
    (corpus / "out" / "enhanced.flac").write_bytes(body)
    (tmp_path / "secret.txt").write_bytes(b"secret")
    raw = tmp_path / "data" / "sub a" / "rec.wav"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(body[::-1])
    serve.Handler.source_roots = ((tmp_path / "data").resolve(),)
    handler = functools.partial(serve.Handler, directory=str(corpus.resolve()))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"127.0.0.1:{server.server_address[1]}", body, raw
    finally:
        server.shutdown()
        server.server_close()
        serve.Handler.source_roots = ()


def _get(host: str, path: str, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    connection = http.client.HTTPConnection(host, timeout=10)
    try:
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, response.read()
    finally:
        connection.close()


def test_a_range_request_gets_206_with_exactly_those_bytes(served: tuple[str, bytes, Path]) -> None:
    """``bytes=a-b``, ``a-`` and ``-n`` each answer 206 with the matching Content-Range and body."""
    host, body, _ = served
    size = len(body)
    for header, first, last in [("bytes=100-199", 100, 199), (f"bytes={size - 10}-", size - 10, size - 1),
                                ("bytes=-25", size - 25, size - 1), ("bytes=5-999999", 5, size - 1)]:
        status, headers, data = _get(host, "/out/enhanced.flac", {"Range": header})
        assert status == 206, header
        assert data == body[first : last + 1], header
        assert headers["content-range"] == f"bytes {first}-{last}/{size}"
        assert headers["content-length"] == str(last - first + 1)
        assert headers["accept-ranges"] == "bytes"


def test_a_whole_file_response_advertises_byte_ranges(served: tuple[str, bytes, Path]) -> None:
    """Without a Range header the file comes back whole, saying ranges are accepted."""
    host, body, _ = served
    status, headers, data = _get(host, "/out/enhanced.flac")
    assert (status, data, headers["accept-ranges"]) == (200, body, "bytes")


def test_the_source_route_honours_ranges(served: tuple[str, bytes, Path]) -> None:
    """A raw recording read in place seeks the same way as a file under the corpus root."""
    host, _, raw = served
    url = "/_source" + str(raw.resolve()).replace(" ", "%20")
    status, headers, data = _get(host, url, {"Range": "bytes=1000-1099"})
    assert status == 206
    assert data == raw.read_bytes()[1000:1100]
    assert headers["content-range"] == f"bytes 1000-1099/{raw.stat().st_size}"


def test_a_traversal_out_of_the_roots_is_refused(served: tuple[str, bytes, Path]) -> None:
    """Neither the static tree nor the source route reads a file outside its roots."""
    host, _, raw = served
    secret = raw.parents[2] / "secret.txt"
    for url in ["/../secret.txt", "/out/../../secret.txt", "/_source" + str(secret), "/_source" + str(raw.parents[1]) + "/../secret.txt"]:
        status, _, data = _get(host, url, {"Range": "bytes=0-3"})
        assert status == 404, url
        assert b"secret" not in data


def test_a_range_past_the_end_is_416(served: tuple[str, bytes, Path]) -> None:
    """A range starting beyond the file answers 416 naming the file's length."""
    host, body, _ = served
    status, headers, data = _get(host, "/out/enhanced.flac", {"Range": f"bytes={len(body)}-"})
    assert status == 416
    assert headers["content-range"] == f"bytes */{len(body)}"
    assert data == b""


def test_byte_range_parses_the_single_range_forms() -> None:
    """Multi-range and malformed headers fall back to the whole file; empty suffixes are unsatisfiable."""
    assert serve.byte_range(None, 10) is None
    assert serve.byte_range("bytes=0-1,4-5", 10) is None
    assert serve.byte_range("bytes=5-2", 10) is None
    assert serve.byte_range("bytes=-0", 10) is False
    assert serve.byte_range("bytes=-50", 10) == (0, 9)
