r"""Build the triage review page: every recording's decision and evidence, faceted, for a reviewer.

Two phases, because the corpus lives on a cluster and the page is read on a laptop.

``extract`` walks a run tree and writes one JSON line per recording: the decision and every evidence
item (from the decision tables' reader), overlays, stream paths, a quantised spectrogram, and for a
speech task its transcript, PII and redactions (from the free-speech review page's reader)::

    uv run python scripts/triage_review_page.py extract CORPUS --out review.jsonl --workers 16

``render`` turns those lines into ``index.html`` plus side files under ``data/``::

    uv run python scripts/triage_review_page.py render review.jsonl --out CORPUS/review \
        --audio-base ../

Open ``index.html`` from ``file://`` to see quantised spectrograms. To hear the audio, serve the
corpus root from the cluster and tunnel to it, then open http://localhost:8765/review/index.html::

    ssh orcd 'cd CORPUS && python3 -m http.server 8765 --bind 127.0.0.1'   # on the cluster
    ssh -N -L 8765:127.0.0.1:8765 orcd                                     # on the laptop

``--audio-base`` is the URL path from the page's directory to the corpus root; the stream paths are
relative to that root. Both outputs carry transcribed speech and PII: keep them local.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from senselab.audio.workflows.triage.nodes.branches import SPEECH_EXPECTATIONS  # noqa: E402
from senselab.audio.workflows.triage.recording_vectors import recording_dirs  # noqa: E402
from senselab.audio.workflows.triage.review_page import review_record, write_page  # noqa: E402
from senselab.utils import fastio  # noqa: E402

EXTRACT_SCHEMA = "senselab.triage.review.extract"
EXTRACT_VERSION = 1


def _free_speech_page() -> Any:  # noqa: ANN401 -- a module
    """The free-speech review page's module, whose transcript reader and mark rules this page shares."""
    name = "free_speech_review_page"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SPEECH_FAMILIES = frozenset(SPEECH_EXPECTATIONS)


def speech_view(run_root: Path) -> dict[str, Any] | None:
    """A speech task's transcript with its marks, PII findings and what decided its release.

    Args:
        run_root: The run root.

    Returns:
        The view, or None where the recording is not a speech task.
    """
    fs = _free_speech_page()
    row = fs.recording_record(run_root, SPEECH_FAMILIES)
    if row is None:
        return None
    determination = row.get("d") or {}
    return {
        "html": fs.paragraph(row.get("w") or [], row.get("f") or []),
        "pii": row.get("pii") or [],
        "release_ground": row.get("rg"),
        "why": row.get("why"),
        "redact_why": row.get("rwhy"),
        "condition_kind": row.get("hk"),
        "names_proposed": row.get("np") or [],
        "llm": determination.get("llm") or {},
        "language": row.get("lang"),
    }


def _worker(payload: tuple[str, str]) -> dict[str, Any] | None:
    """Pool entry point: one recording's record, an error record, or None."""
    run_root, root = payload
    try:
        return review_record(Path(run_root), Path(root), speech_view)
    except Exception as error:  # noqa: BLE001 -- one unreadable store must not stop the sweep
        return {"error": f"{type(error).__name__}: {error}", "run_dir": run_root}


def _rows(payloads: Sequence[tuple[str, str]], workers: int) -> Iterator[dict[str, Any] | None]:
    if workers <= 1:
        yield from (_worker(payload) for payload in payloads)
        return
    with Pool(workers) as pool:
        yield from pool.imap_unordered(_worker, list(payloads), chunksize=8)


def extract(
    corpus: Path,
    out: Path,
    workers: int,
    *,
    manifest: Path | None = None,
    walk_threads: int = fastio.DEFAULT_THREADS,
) -> dict[str, Any]:
    """Write one JSON line per recording.

    Args:
        corpus: The run tree, ``<root>/sub-*/ses-*/<stem>_<timestamp>/``; paths are relative to it.
        out: The JSONL to write.
        workers: Processes reading stores.
        manifest: A file of run roots, one per line, to read instead of walking the tree.
        walk_threads: Participants listed at once while walking.

    Returns:
        The sweep's counts.
    """
    if manifest is not None:
        run_roots = [Path(line.strip()) for line in manifest.read_text().splitlines() if line.strip()]
    else:
        run_roots = list(recording_dirs(corpus, threads=walk_threads))
    counts: Counter[str] = Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        handle.write(json.dumps({"schema": EXTRACT_SCHEMA, "version": EXTRACT_VERSION}) + "\n")
        for row in _rows([(str(r), str(corpus)) for r in run_roots], workers):
            if row is None:
                counts["no_fold"] += 1
                continue
            if "error" in row:
                counts["errors"] += 1
                handle.write(json.dumps(row) + "\n")
                continue
            counts["recordings"] += 1
            counts[f"verdict:{row['verdict']}"] += 1
            handle.write(json.dumps(row, separators=(",", ":"), default=str) + "\n")
    return {"schema": EXTRACT_SCHEMA, "version": EXTRACT_VERSION, "run_roots": len(run_roots), **dict(counts)}


def load(path: Path) -> list[Mapping[str, Any]]:
    """The records of an extract, without its header and error lines.

    Args:
        path: The JSONL.

    Returns:
        The records.

    Raises:
        ValueError: When the header names another schema or version.
    """
    lines = path.read_text().splitlines()
    header = json.loads(lines[0]) if lines else {}
    if header.get("schema") != EXTRACT_SCHEMA or header.get("version") != EXTRACT_VERSION:
        raise ValueError(f"{path} is not a {EXTRACT_SCHEMA} v{EXTRACT_VERSION} extract")
    return [record for record in map(json.loads, lines[1:]) if "error" not in record]


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: The command line, or None for ``sys.argv``.

    Returns:
        The exit status.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    extractor = sub.add_parser("extract", help="walk a corpus and write one JSON line per recording")
    extractor.add_argument("corpus", type=Path)
    extractor.add_argument("--out", type=Path, required=True)
    extractor.add_argument("--workers", type=int, default=8)
    extractor.add_argument("--manifest", type=Path, default=None, help="run roots, one per line, instead of a walk")
    extractor.add_argument("--walk-threads", type=int, default=fastio.DEFAULT_THREADS)
    renderer = sub.add_parser("render", help="turn an extract into index.html and its side files")
    renderer.add_argument("data", type=Path)
    renderer.add_argument("--out", type=Path, required=True)
    renderer.add_argument("--title", default="Triage review")
    renderer.add_argument("--audio-base", default="../", help="URL path from the page's directory to the corpus root")
    arguments = parser.parse_args(argv)
    if arguments.command == "extract":
        summary = extract(
            arguments.corpus,
            arguments.out,
            arguments.workers,
            manifest=arguments.manifest,
            walk_threads=arguments.walk_threads,
        )
        print(json.dumps(summary, indent=2))
        return 0
    written = write_page(
        load(arguments.data),
        arguments.out,
        title=arguments.title,
        audio_base=arguments.audio_base,
        mark_style=_free_speech_page().MARK_STYLE,
    )
    print(json.dumps(written, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
