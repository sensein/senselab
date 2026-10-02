# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Adapted from TensorFlow's fastio.walk, as published at
# https://gist.github.com/satra/0e02cd7554672120cf8073df3986f302: scandir instead of listdir and
# isdir, per-file work inside the workers, symlink and top-level filters, error reporting, and a
# shutdown that leaves no thread running when the consumer stops early.
"""Multithreaded directory walks.

:func:`walk` is :func:`os.walk` run by a fixed pool of threads, so many directory listings are in
flight at once; results arrive in no particular order. :func:`map_files` applies a function to every
matching file inside those threads, and :func:`find` returns the matching paths sorted.

Errors listing a directory are not printed and dropped. By default the walk finishes and then raises
:class:`WalkError` carrying every ``(path, error)`` pair; pass ``on_error`` to receive each pair as it
happens instead, in which case nothing is raised for them. An exception raised by the function given to
:func:`map_files` stops the walk and is re-raised in the consumer.

Example:
    >>> from senselab.utils.fastio import find
    >>> stores = find("/data/run_tree", "store.jsonl", threads=64)  # doctest: +SKIP
"""

from __future__ import annotations

import fnmatch
import os
import queue
import threading
from pathlib import Path
from typing import Callable, Iterator, TypeVar

R = TypeVar("R")

DEFAULT_THREADS = 32
"""Pool size when the caller names none."""

_DONE = object()


class WalkError(OSError):
    """Some directories could not be listed.

    Attributes:
        errors: Each ``(path, error)`` pair, in the order the workers met them.
    """

    def __init__(self, errors: list[tuple[str, OSError]]) -> None:
        """Hold the errors.

        Args:
            errors: Each ``(path, error)`` pair.
        """
        self.errors = errors
        first = f"{errors[0][0]}: {errors[0][1]}" if errors else ""
        super().__init__(f"{len(errors)} director{'y' if len(errors) == 1 else 'ies'} could not be listed; {first}")


def _pool(
    top: str | os.PathLike[str],
    emit: Callable[[str, list[str], list[os.DirEntry[str]]], list[object]],
    *,
    threads: int,
    follow_symlinks: bool,
    keep: Callable[[str], bool] | None,
    on_error: Callable[[str, OSError], None] | None,
) -> Iterator[object]:
    """Run the threaded walk, yielding whatever ``emit`` returns for each directory.

    Args:
        top: The directory to walk.
        emit: Called in a worker with a directory's path, its subdirectory names and its file
            entries; returns the items to hand to the consumer.
        threads: Size of the pool.
        follow_symlinks: Whether a symlink to a directory is descended into.
        keep: When given, a predicate on the names directly under ``top``; the others are skipped.
        on_error: Receives each listing error; None collects them and raises at the end.

    Yields:
        The items ``emit`` returned, in no particular order.

    Raises:
        WalkError: When directories could not be listed and ``on_error`` is None.
        ValueError: When ``threads`` is below 1.
    """
    if threads < 1:
        raise ValueError(f"threads must be at least 1, got {threads}")
    root = os.fspath(top)
    if not os.path.isdir(root):
        return
    lock = threading.Lock()
    has_work = threading.Condition(lock)
    pending = [root]
    state = {"tasks": 1, "stop": False}
    out: queue.SimpleQueue[object] = queue.SimpleQueue()
    errors: list[tuple[str, OSError]] = []

    def worker() -> None:
        try:
            while True:
                with lock:
                    while not pending and state["tasks"] and not state["stop"]:
                        has_work.wait()
                    if state["stop"] or not state["tasks"]:
                        return
                    path = pending.pop()
                try:
                    dirs: list[str] = []
                    files: list[os.DirEntry[str]] = []
                    try:
                        with os.scandir(path) as listing:
                            entries = list(listing)
                    except OSError as error:
                        if on_error is None:
                            with lock:
                                errors.append((path, error))
                        else:
                            on_error(path, error)
                        continue
                    for entry in entries:
                        if path == root and keep is not None and not keep(entry.name):
                            continue
                        try:
                            is_dir = entry.is_dir(follow_symlinks=True)
                        except OSError:
                            is_dir = False
                        if is_dir:
                            dirs.append(entry.name)
                            if follow_symlinks or not entry.is_symlink():
                                with lock:
                                    state["tasks"] += 1
                                    pending.append(entry.path)
                                    has_work.notify()
                        else:
                            files.append(entry)
                    for item in emit(path, sorted(dirs), sorted(files, key=lambda e: e.name)):
                        out.put(item)
                except BaseException as error:  # noqa: BLE001 — handed to the consumer, which re-raises
                    out.put(_Failure(error))
                    with lock:
                        state["stop"] = True
                        has_work.notify_all()
                finally:
                    with lock:
                        state["tasks"] -= 1
                        if not state["tasks"]:
                            has_work.notify_all()
        finally:
            out.put(_DONE)

    pool = [threading.Thread(target=worker, name=f"fastio.walk {i} {root}", daemon=True) for i in range(threads)]
    for thread in pool:
        thread.start()
    finished = 0
    try:
        while finished < len(pool):
            item = out.get()
            if item is _DONE:
                finished += 1
            elif isinstance(item, _Failure):
                raise item.error
            else:
                yield item
    finally:
        with lock:
            state["stop"] = True
            has_work.notify_all()
        for thread in pool:
            thread.join()
    if errors:
        raise WalkError(errors)


class _Failure:
    """An exception a worker raised, carried to the consumer."""

    def __init__(self, error: BaseException) -> None:
        self.error = error


def walk(
    top: str | os.PathLike[str],
    *,
    threads: int = DEFAULT_THREADS,
    follow_symlinks: bool = False,
    keep: Callable[[str], bool] | None = None,
    on_error: Callable[[str, OSError], None] | None = None,
) -> Iterator[tuple[str, list[str], list[str]]]:
    """Walk a tree with a pool of threads, like :func:`os.walk` but in no particular order.

    Args:
        top: The directory to walk.
        threads: Size of the pool.
        follow_symlinks: Whether a symlink to a directory is descended into. Either way it is listed
            among its parent's ``dirs``, as :func:`os.walk` does.
        keep: When given, a predicate on the names directly under ``top``; the others are skipped.
        on_error: Receives each ``(path, error)`` for a directory that could not be listed; None
            collects them and raises :class:`WalkError` once the walk is done.

    Yields:
        ``(path, dirs, files)`` for each directory, ``dirs`` and ``files`` sorted by name.

    Raises:
        WalkError: When directories could not be listed and ``on_error`` is None.
    """

    def emit(path: str, dirs: list[str], files: list[os.DirEntry[str]]) -> list[object]:
        return [(path, dirs, [entry.name for entry in files])]

    for item in _pool(top, emit, threads=threads, follow_symlinks=follow_symlinks, keep=keep, on_error=on_error):
        yield item  # type: ignore[misc]


def map_files(
    top: str | os.PathLike[str],
    fn: Callable[[str], R],
    *,
    threads: int = DEFAULT_THREADS,
    pattern: str | None = None,
    follow_symlinks: bool = False,
    keep: Callable[[str], bool] | None = None,
    on_error: Callable[[str, OSError], None] | None = None,
) -> Iterator[R]:
    """Apply a function to every matching file under a tree, inside the walk's threads.

    Args:
        top: The directory to walk.
        fn: Called with each matching file's path; its result is yielded. An exception it raises
            stops the walk and is re-raised here.
        threads: Size of the pool.
        pattern: An :mod:`fnmatch` pattern on the file's name; None matches every file.
        follow_symlinks: Whether a symlink to a directory is descended into.
        keep: When given, a predicate on the names directly under ``top``; the others are skipped.
        on_error: As in :func:`walk`.

    Yields:
        ``fn``'s result for each matching file, in no particular order.

    Raises:
        WalkError: When directories could not be listed and ``on_error`` is None.
    """

    def emit(path: str, dirs: list[str], files: list[os.DirEntry[str]]) -> list[object]:
        return [fn(entry.path) for entry in files if pattern is None or fnmatch.fnmatchcase(entry.name, pattern)]

    for item in _pool(top, emit, threads=threads, follow_symlinks=follow_symlinks, keep=keep, on_error=on_error):
        yield item  # type: ignore[misc]


def find(
    top: str | os.PathLike[str],
    pattern: str,
    *,
    threads: int = DEFAULT_THREADS,
    follow_symlinks: bool = False,
    on_error: Callable[[str, OSError], None] | None = None,
) -> list[Path]:
    """Every file under a tree whose name matches a pattern, sorted by path.

    Args:
        top: The directory to walk.
        pattern: An :mod:`fnmatch` pattern on the file's name.
        threads: Size of the pool.
        follow_symlinks: Whether a symlink to a directory is descended into.
        on_error: As in :func:`walk`.

    Returns:
        The matching paths, sorted.

    Raises:
        WalkError: When directories could not be listed and ``on_error`` is None.
    """
    return sorted(
        map_files(top, Path, threads=threads, pattern=pattern, follow_symlinks=follow_symlinks, on_error=on_error)
    )
