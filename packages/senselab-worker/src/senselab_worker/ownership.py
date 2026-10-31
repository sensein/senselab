"""The one place that deletes runtime files: `remove_owned`, and the name patterns of what the runtime writes."""

import os
import re
import shutil
import stat
import sys
from pathlib import Path

ARRAY_NAME = re.compile(r"^[0-9a-f]{64}\.npy$")
"""An array file: the sha256 of its bytes, then `.npy`."""

ARRAY_TEMP_NAME = re.compile(r"^\.[0-9a-f]{32}\.tmp$")
"""An array file being written: a dot, a uuid4 hex, then `.tmp`."""

UUID_HEX = re.compile(r"^[0-9a-f]{32}$")
"""A staging file or directory named by a uuid4 hex."""


def remove_owned(root: Path, path: Path, *, pattern: re.Pattern[str]) -> bool:
    """Remove `path` if it is a direct child of `root` whose name fully matches `pattern`; return whether it did.

    The child is examined with `lstat` relative to an open descriptor of `root`. A directory is
    removed as a tree without following symlinks; anything else, a symlink included, is unlinked and
    its target is never touched. `.`, `..` and `root` itself are never children. A missing `root` or
    child returns False.

    Raises:
        OSError: if the child exists but cannot be removed.
    """
    target = os.path.abspath(path)
    name = os.path.basename(target)
    if os.path.dirname(target) != os.path.abspath(root) or name in ("", ".", "..") or not pattern.fullmatch(name):
        return False
    try:
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    except FileNotFoundError:
        return False
    try:
        try:
            mode = os.lstat(name, dir_fd=root_fd).st_mode
        except FileNotFoundError:
            return False
        try:
            if not stat.S_ISDIR(mode):
                os.unlink(name, dir_fd=root_fd)
            elif sys.version_info >= (3, 11):
                shutil.rmtree(name, dir_fd=root_fd)
            else:
                shutil.rmtree(target)
        except FileNotFoundError:
            return False
        return True
    finally:
        os.close(root_fd)
