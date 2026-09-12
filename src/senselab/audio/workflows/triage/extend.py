"""Reading, writing and re-exporting a finished triage run, for the drivers that extend one.

A run root holds ``run/store.jsonl`` and ``prov/``. Every extend driver needs the same four
operations over that layout — derive the root from a path inside it, read the store under the run's
own id, replace the store atomically, re-export BEP028 — and the layout is the workflow's, not any
one driver's, so it is stated once here rather than copied per script.

Manifest reading and slicing live here for the same reason: a Slurm array shards a JSONL the same
way whatever it is extending.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Sequence

from senselab.utils.prov_bep028 import to_bep028_graph, write_bep028_files
from senselab.utils.prov_store import ProvStore

RUN_SUBDIR = "run"
STORE_FILE = "store.jsonl"
PROV_SUBDIR = "prov"
PROV_LABEL = "triage"
SLICES_SUBDIR = "slices"


def read_manifest(path: Path, *, required: Sequence[str] = ("stem",)) -> list[dict[str, Any]]:
    """Read a corpus manifest into rows.

    Args:
        path: The JSONL file, one object per line.
        required: Keys every row must carry a truthy value for.

    Returns:
        One dict per non-blank line, in file order.

    Raises:
        ValueError: If a line is not a JSON object, or is missing one of ``required``.
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            missing = [key for key in required if not payload.get(key)]
            if missing:
                raise ValueError(f"{path}:{number} carries no {' or '.join(missing)}")
            rows.append(payload)
    return rows


def take_slice(rows: Sequence[dict[str, Any]], index: int, count: int) -> list[dict[str, Any]]:
    """The stride of a manifest one array task owns.

    Args:
        rows: Every manifest row.
        index: This task's 0-based index.
        count: How many tasks the array has.

    Returns:
        ``rows[index::count]``.

    Raises:
        ValueError: If ``count`` is not positive, or ``index`` is outside it.
    """
    if count < 1:
        raise ValueError(f"--slice-count must be at least 1, got {count}")
    if not 0 <= index < count:
        raise ValueError(f"--slice-index must be in [0, {count}), got {index}")
    return list(rows[index::count])


def batches(rows: Sequence[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    """Split rows into fixed-size batches, the last one as short as it needs to be.

    Args:
        rows: The rows to split.
        size: Rows per batch.

    Yields:
        Each batch, in order.

    Raises:
        ValueError: If ``size`` is not positive.
    """
    if size < 1:
        raise ValueError(f"--batch-size must be at least 1, got {size}")
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def run_root_of(stream_path: Path) -> Path:
    """The run root holding one recording's store, from the path of a stream inside it.

    Args:
        stream_path: ``<run_root>/run/streams/<stream>``.

    Returns:
        ``<run_root>``.

    Raises:
        ValueError: If the path is not that shape.
    """
    parents = stream_path.parents
    if len(parents) < 3 or parents[0].name != "streams" or parents[1].name != RUN_SUBDIR:
        raise ValueError(f"{stream_path} is not <run_root>/{RUN_SUBDIR}/streams/<stream>; no run root to extend")
    return parents[2]


def read_store(run_root: Path) -> ProvStore:
    """Read one run's store under the run's own id, so re-derived entity ids match the run's.

    Args:
        run_root: The run root.

    Returns:
        The store, with ``run_id`` set to the run root's own name.

    Raises:
        FileNotFoundError: If the run holds no store.
    """
    store_path = run_root / RUN_SUBDIR / STORE_FILE
    if not store_path.is_file():
        raise FileNotFoundError(f"no store at {store_path}")
    return ProvStore.read_jsonl(store_path, run_id=run_root.name)


def write_store(store: ProvStore, run_root: Path) -> Path:
    """Replace one run's store atomically, so a killed task never leaves a truncated one.

    Args:
        store: The merged store.
        run_root: The run root.

    Returns:
        The store's path.
    """
    store_path = run_root / RUN_SUBDIR / STORE_FILE
    partial = store_path.with_suffix(store_path.suffix + ".partial")
    store.write_jsonl(partial)
    partial.replace(store_path)
    return store_path


def export_prov(store: ProvStore, run_root: Path) -> list[Path]:
    """Re-export the BEP028 files from the merged store, so ``prov/`` agrees with it.

    Args:
        store: The merged store.
        run_root: The run root ``prov/`` is a directory of.

    Returns:
        The files written.
    """
    graph = to_bep028_graph(store, label=PROV_LABEL)
    return write_bep028_files(graph, run_root / PROV_SUBDIR, label=PROV_LABEL)
