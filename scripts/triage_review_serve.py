r"""Serve a triage review page and the audio it plays, including raw recordings outside the corpus.

The page addresses a stream under the corpus root by its relative path, and a stream outside it (a
raw recording read in place, such as ``/orcd/data/...``) by its absolute path under ``/_source``.
This server serves the corpus root as a static tree and answers ``/_source/<absolute path>`` from the
filesystem, only for files under one of the ``--source-root`` directories. Files are served with single
byte ranges (``206 Partial Content``), which an ``<audio>`` element needs to seek::

    python3 scripts/triage_review_serve.py CORPUS --port 8765 --bind "$(hostname)" \
        --source-root /orcd/data/satra/002/datasets/b2ai

Standard library only, so it runs under any ``python3`` on a compute node.
"""

from __future__ import annotations

import argparse
import email.utils
import functools
import http.server
import os
import re
import sys
import urllib.parse
from pathlib import Path
from typing import BinaryIO, Sequence

SOURCE_ROUTE = "/_source"
_RANGE = re.compile(r"^\s*bytes\s*=\s*(\d*)\s*-\s*(\d*)\s*$")


def source_path(url_path: str, roots: Sequence[Path]) -> Path | None:
    """The file a ``/_source/<absolute path>`` URL names, where it lies under an allowed root.

    Args:
        url_path: The request path, query stripped.
        roots: The directories the route may read from, resolved.

    Returns:
        The resolved file path, or None where the URL is not on the route or leaves every root.
    """
    if not url_path.startswith(SOURCE_ROUTE + "/"):
        return None
    target = Path(os.path.normpath(urllib.parse.unquote(url_path[len(SOURCE_ROUTE) :]))).resolve()
    return target if any(target.is_relative_to(root) for root in roots) else None


def byte_range(header: str | None, size: int) -> tuple[int, int] | None | bool:
    """The inclusive byte span a ``Range`` header asks for in a file of ``size`` bytes.

    Args:
        header: The ``Range`` header value, or None.
        size: The file's length in bytes.

    Returns:
        ``(first, last)`` for one satisfiable range; None where there is no header, or it is not a single
        ``bytes=`` range (the whole file is then served); False where the range cannot be satisfied.
    """
    if header is None:
        return None
    match = _RANGE.match(header)
    if match is None:
        return None
    first_text, last_text = match.groups()
    if not first_text and not last_text:
        return None
    if not first_text:
        suffix = int(last_text)
        if suffix == 0 or size == 0:
            return False
        return max(0, size - suffix), size - 1
    first = int(first_text)
    last = int(last_text) if last_text else size - 1
    if last_text and last < first:
        return None
    if first >= size:
        return False
    return first, min(last, size - 1)


class _Slice:
    """A read-only view of ``length`` bytes of an open file, from its current position."""

    def __init__(self, handle: BinaryIO, length: int) -> None:
        self._handle = handle
        self._left = length

    def read(self, n: int = -1) -> bytes:
        """Up to ``n`` bytes, never past the slice's end."""
        if self._left <= 0:
            return b""
        chunk = self._handle.read(self._left if n < 0 else min(n, self._left))
        self._left -= len(chunk)
        return chunk

    def close(self) -> None:
        """Close the underlying file."""
        self._handle.close()


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static files from the corpus root, plus the ``/_source`` route onto the allowed roots."""

    protocol_version = "HTTP/1.1"
    source_roots: tuple[Path, ...] = ()
    _accept_ranges = False

    def translate_path(self, path: str) -> str:
        """The filesystem path a request names.

        Args:
            path: The request path.

        Returns:
            The file under an allowed source root for a ``/_source`` request (a path that cannot exist
            where the route leaves every root), else the static tree's own translation.
        """
        bare = urllib.parse.urlsplit(path).path
        if bare.startswith(SOURCE_ROUTE + "/"):
            found = source_path(bare, self.source_roots)
            return str(found) if found is not None else os.devnull + ".refused"
        return super().translate_path(path)

    def end_headers(self) -> None:
        """Advertise byte ranges on a file response, then end the header block."""
        if self._accept_ranges:
            self.send_header("Accept-Ranges", "bytes")
            self._accept_ranges = False
        super().end_headers()

    def send_head(self) -> BinaryIO | _Slice | None:  # type: ignore[override]
        """Send the status and headers for a GET or HEAD, honouring one byte range on a file.

        Returns:
            The body to copy (a slice of the file for a 206), or None where there is none.
        """
        path = self.translate_path(self.path)
        if not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        wanted = byte_range(self.headers.get("Range"), size)
        if_range = self.headers.get("If-Range")
        if wanted is not None and if_range is not None and if_range != self._last_modified(path):
            wanted = None
        if wanted is None:
            self._accept_ranges = True
            return super().send_head()
        if wanted is False:
            self.send_response(http.HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self._accept_ranges = True
            self.end_headers()
            return None
        assert isinstance(wanted, tuple)
        first, last = wanted
        try:
            handle = open(path, "rb")
        except OSError:
            self.send_error(http.HTTPStatus.NOT_FOUND, "File not found")
            return None
        handle.seek(first)
        self.send_response(http.HTTPStatus.PARTIAL_CONTENT)
        self.send_header("Content-type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {first}-{last}/{size}")
        self.send_header("Content-Length", str(last - first + 1))
        self.send_header("Last-Modified", self._last_modified(path))
        self._accept_ranges = True
        self.end_headers()
        return _Slice(handle, last - first + 1)

    def copyfile(self, source: BinaryIO, outputfile: BinaryIO) -> None:  # type: ignore[override]
        """Copy a body, closing the connection quietly where the client hung up mid-body."""
        try:
            super().copyfile(source, outputfile)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    @staticmethod
    def _last_modified(path: str) -> str:
        return email.utils.formatdate(os.path.getmtime(path), usegmt=True)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: The command line, or None for ``sys.argv``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", type=Path, help="the corpus root the page's relative stream paths start from")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--source-root", type=Path, action="append", default=[], help="a directory /_source may read")
    arguments = parser.parse_args(argv)
    handler = functools.partial(Handler, directory=str(arguments.root.resolve()))
    Handler.source_roots = tuple(root.resolve() for root in arguments.source_root)
    with http.server.ThreadingHTTPServer((arguments.bind, arguments.port), handler) as server:
        print(f"serving {arguments.root} on {arguments.bind}:{arguments.port}", flush=True)
        server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
