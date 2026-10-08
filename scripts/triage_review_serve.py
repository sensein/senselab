r"""Serve a triage review page and the audio it plays, including raw recordings outside the corpus.

The page addresses a stream under the corpus root by its relative path, and a stream outside it (a
raw recording read in place, such as ``/orcd/data/...``) by its absolute path under ``/_source``.
This server serves the corpus root as a static tree and answers ``/_source/<absolute path>`` from the
filesystem, only for files under one of the ``--source-root`` directories::

    python3 scripts/triage_review_serve.py CORPUS --port 8765 --bind "$(hostname)" \
        --source-root /orcd/data/satra/002/datasets/b2ai

Standard library only, so it runs under any ``python3`` on a compute node.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import sys
import urllib.parse
from pathlib import Path
from typing import Sequence

SOURCE_ROUTE = "/_source"


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


class Handler(http.server.SimpleHTTPRequestHandler):
    """Static files from the corpus root, plus the ``/_source`` route onto the allowed roots."""

    source_roots: tuple[Path, ...] = ()

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
