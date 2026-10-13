#!/usr/bin/env python3
r"""Run SECOND_OPINION over a finished triage corpus, in place, without replaying the graph.

    uv run python scripts/extend_second_opinion.py MANIFEST --slice-index I --slice-count N \
        --config ENABLE.yaml --ollama-binary BIN --ollama-models DIR --hints DIR \
        [--log-dir DIR] [--verified-dir DIR] [--force] [--workers N]

``MANIFEST`` is the JSONL every ``extend_*`` driver takes: one object per line carrying ``stem``
and ``enhanced``, the absolute path of that recording's ``run/streams/enhanced.flac``. A row may
also carry ``source``, the recording the hint is built from.

``--slice-index`` / ``--slice-count`` shard the manifest for a Slurm array: task *i* of *n* takes
``rows[i::n]``.

The node reads the store and the recording's declaration (``--hints``), which supplies the task
context. One pinned ``ollama serve`` is started per task, the first time a row needs it, and stopped
when the task ends. ``--workers`` rows are asked at once (default ``second_opinion.workers``), the
server answering as many requests in parallel; each store is read and written by one worker.
``--config`` must set ``second_opinion.enabled: true``; the driver refuses to run otherwise rather
than writing ``disabled`` into every store.

SECOND_OPINION runs after REVIEW: it reads the final mask plan, the reviewer's releases included. A
store without REVIEW's ``redaction_llm_annotation`` is an error row, and an opinion asked over an
earlier annotation than the store's live one is asked again.

The driver writes the ``second_opinion_answers`` and nothing else: no verdict is decided again here. Follow
the run with ``scripts/extend_refold.py`` over the same corpus, which re-decides every VERDICT over
the opinions it finds. A store already holding a live ``ok`` ``second_opinion_answers`` that a
SECOND_OPINION activity generated is ``present`` and not asked again, so a preempted task is resumed
by resubmitting it. ``--force`` asks again and retires the opinion it replaces.

The design is in ``specs/20261003-clef-second-opinion/design.md``.

Install:
    uv sync --all-extras --group dev
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Sequence

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.extend import (
    ERROR,
    OK,
    PRESENT,
    SLICES_SUBDIR,
    SliceLog,
    export_prov,
    load_hint_builder,
    logged,
    read_manifest,
    read_store,
    run_root_of,
    source_of,
    supersede,
    take_slice,
    write_store,
)
from senselab.audio.workflows.triage.nodes.common import describe_exception, find_measurement, software_agent
from senselab.audio.workflows.triage.nodes.second_opinion import ABSENT, NODE, Ask, pin_of, second_opinion, settings
from senselab.audio.workflows.triage.vocabulary import REDACTION_LLM_ANNOTATION, SECOND_OPINION_ANSWERS
from senselab.text.tasks.decision_model.ollama import OllamaServer, PinMismatchError, ask_decisions, verify_pin
from senselab.text.tasks.decision_model.second_opinion import QUESTION_SET_VERSION
from senselab.utils.prov_store import Entity, ProvStore

OPINION_SUPERSEDED = "opinion_superseded"
_SUPERSEDED_REASON = "asked again under a later configuration"
_DEFAULT_VERIFIED = Path.home() / ".cache" / "senselab" / "ollama-verified"

OpenAsk = Callable[[], ContextManager[Ask]]
"""Opens whatever answers the questions for the length of a task."""


def build_parser() -> argparse.ArgumentParser:
    """The CLI: a manifest, which shard of it this task takes, and where the pinned model is.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", maxsplit=1)[0] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("manifest", type=Path, help="Manifest JSONL: one object per recording")
    parser.add_argument("--slice-index", type=int, required=True, help="This task's index, 0-based")
    parser.add_argument("--slice-count", type=int, required=True, help="How many tasks the array has")
    parser.add_argument("--log-dir", type=Path, default=None, help=f"Where {SLICES_SUBDIR}/ goes (default: beside it)")
    parser.add_argument("--config", type=Path, default=None, help="Partial YAML deep-merged over the packaged config")
    parser.add_argument("--ollama-binary", type=Path, default=None, help="The ollama executable")
    parser.add_argument("--ollama-models", type=Path, default=None, help="The Ollama model store (OLLAMA_MODELS)")
    parser.add_argument(
        "--verified-dir", type=Path, default=_DEFAULT_VERIFIED, help="Where a completed blob hash is remembered"
    )
    parser.add_argument("--force", action="store_true", help="Ask again where an opinion already stands")
    parser.add_argument(
        "--hints", type=Path, required=True, help="Directory holding hints.py; the task context comes from it"
    )
    parser.add_argument(
        "--workers", type=int, default=None, help="Rows asked at once (default: second_opinion.workers)"
    )
    return parser


def ollama_asker(
    config: TriageConfig,
    *,
    binary: Path,
    models_dir: Path,
    verified_dir: Path | None,
    log_path: Path | None,
    num_parallel: int = 1,
) -> OpenAsk:
    """An :data:`OpenAsk` that serves the configured pin and asks it over HTTP.

    Args:
        config: The triage configuration, which names the pin, the seed and the timeout.
        binary: The ``ollama`` executable.
        models_dir: The model store.
        verified_dir: Where a completed blob hash is remembered.
        log_path: Where the server's output goes.
        num_parallel: How many requests the server answers at once.

    Returns:
        The opener.
    """
    held = settings(config)
    pin = pin_of(config)

    @contextlib.contextmanager
    def open_ask() -> Iterator[Ask]:
        with OllamaServer(
            binary,
            models_dir,
            pin,
            verified_dir=verified_dir,
            log_path=log_path,
            num_parallel=num_parallel,
            load_timeout_s=float(held["load_timeout_s"]),
        ) as server:
            yield lambda state, questions: ask_decisions(
                server.host,
                f"{pin.name}:{pin.tag}",
                state,
                questions,
                seed=int(held["seed"]),
                timeout_s=float(held["timeout_s"]),
            )

    return open_ask


class SliceAbortedError(Exception):
    """The slice cannot go on asking: the server did not start, or kept failing."""


class LazyAsk:
    """Opens the asker on its first call and holds it until :meth:`close`; safe to call from threads.

    A failed start, or ``max_consecutive_errors`` failed asks in a row, raises
    :class:`SliceAbortedError` from that call and every later one.

    Args:
        open_ask: The opener.
        max_consecutive_errors: Failed asks in a row that end the slice; 0 never ends it.
    """

    def __init__(self, open_ask: OpenAsk, *, max_consecutive_errors: int = 0) -> None:
        """Hold the opener; nothing starts until the first call."""
        self._open_ask = open_ask
        self._max_errors = int(max_consecutive_errors)
        self._errors = 0
        self._stack = contextlib.ExitStack()
        self._ask: Ask | None = None
        self._failure: SliceAbortedError | None = None
        self._lock = threading.Lock()

    def __call__(self, state: Any, questions: Any) -> Any:  # noqa: ANN401 — the Ask signature
        """Ask, starting the server first if this is the first question.

        Raises:
            SliceAbortedError: If the server did not start, or this ask makes the run of failures
                reach ``max_consecutive_errors``.
        """
        with self._lock:
            if self._failure is not None:
                raise self._failure
            if self._ask is None:
                try:
                    self._ask = self._stack.enter_context(self._open_ask())
                except (OSError, RuntimeError) as error:
                    self._failure = SliceAbortedError(f"the decision-model server did not start: {error}")
                    raise self._failure from error
            ask = self._ask
        try:
            answer = ask(state, questions)
        except Exception as error:
            with self._lock:
                self._errors += 1
                if self._failure is None and self._max_errors and self._errors >= self._max_errors:
                    self._failure = SliceAbortedError(
                        f"{self._errors} decision-model requests failed in a row; the last: "
                        f"{type(error).__name__}: {error}"
                    )
                if self._failure is not None:
                    raise self._failure from error
            raise
        with self._lock:
            self._errors = 0
        return answer

    @property
    def opened(self) -> bool:
        """Whether the asker was ever started."""
        return self._ask is not None

    def close(self) -> None:
        """Stop the asker if it was started."""
        self._stack.close()
        self._ask = None


def standing(store: ProvStore, config: TriageConfig | None = None) -> Entity | None:
    """The opinion SECOND_OPINION has already left here, if any, to the current questions.

    Args:
        store: The run's store.
        config: The triage configuration whose pinned model the opinion must come from; any model
            where None.

    Returns:
        The live ``ok`` ``second_opinion_answers`` a SECOND_OPINION activity generated under the current
        :data:`~senselab.text.tasks.decision_model.second_opinion.QUESTION_SET_VERSION` (and, with
        ``config``, from its pinned weights) over the store's live REVIEW annotation, or None.
    """
    opinion = find_measurement(store, SECOND_OPINION_ANSWERS)
    if opinion is None or opinion.attributes.get("status") != "ok":
        return None
    if opinion.attributes.get("question_set_version") != QUESTION_SET_VERSION:
        return None
    if config is not None and opinion.attributes.get("blob_digest") != pin_of(config).blob_digest:
        return None
    annotation = find_measurement(store, REDACTION_LLM_ANNOTATION)
    if annotation is None or opinion.attributes.get("review_annotation_id") != annotation.id:
        return None
    activity_id = store.generated_by(opinion.id)
    if activity_id is None:
        return None
    try:
        return opinion if store.get_activity(activity_id).node == NODE else None
    except KeyError:
        return None


def live_opinions(store: ProvStore) -> list[str]:
    """Every live ``second_opinion_answers`` in the store.

    Args:
        store: The run's store.

    Returns:
        The entity ids, in the store's own order.
    """
    return [
        entity.id
        for entity in store.entities("measurement")
        if entity.attributes.get("name") == SECOND_OPINION_ANSWERS and not store.is_invalidated(entity.id)
    ]


def extend_one(
    run_root: Path,
    config: TriageConfig,
    ask: Ask,
    *,
    force: bool,
    build_hint: Callable[[Path], Any] | None,
    source: Path | None,
    num_parallel: int = 1,
) -> dict[str, str]:
    """Ask about one finished run and write the opinion into its store.

    Args:
        run_root: The run root.
        config: The triage configuration.
        ask: Sends a state and questions to the model.
        force: Whether to ask again where an opinion stands.
        build_hint: The hint populator, or None to ask without the declaration's task context.
        source: The recording the run was over, for the hint.
        num_parallel: How many requests the server answers at once, recorded on the opinion.

    Returns:
        ``{status, SECOND_OPINION}`` -- ``ok`` when an opinion landed, ``present`` when one already
        stood or the store came out unchanged, ``error`` when the store would not open, REVIEW has not
        written its annotation, or the model was asked and did not answer. An unanswered ask is not written.
    """
    try:
        store = read_store(run_root)
    except (OSError, ValueError) as error:
        return {"status": ERROR, NODE: describe_exception(error)}
    held = standing(store, config)
    if held is not None and not force:
        return {"status": PRESENT, NODE: str(held.attributes.get("status") or "")}
    if find_measurement(store, REDACTION_LLM_ANNOTATION) is None:
        return {"status": ERROR, NODE: f"REVIEW has not run: no {REDACTION_LLM_ANNOTATION} in the store"}
    replaced = live_opinions(store)
    hint: AudioHints | None = None
    if build_hint is not None and source is not None:
        try:
            hint = build_hint(source)[0]
        except Exception as error:  # noqa: BLE001 — a hint that cannot be built is this row's error
            return {"status": ERROR, NODE: f"hint: {describe_exception(error)}"}
    before = store.fingerprint()
    try:
        outcome = second_opinion(store, config, hint, ask, num_parallel=num_parallel)
    except (OSError, ValueError, LookupError, RuntimeError) as error:
        return {"status": ERROR, NODE: describe_exception(error)}
    if outcome.status == ABSENT:
        failure = store.get_entity(outcome.measurement_id).attributes.get("failure")
        return {"status": ERROR, NODE: f"{ABSENT}: {failure}"}
    for entity_id in replaced:
        if entity_id != outcome.measurement_id:
            supersede(
                store,
                entity_id,
                node=NODE,
                step=OPINION_SUPERSEDED,
                reason=_SUPERSEDED_REASON,
                software=software_agent(store),
            )
    if store.fingerprint() == before:
        return {"status": PRESENT, NODE: outcome.status}
    write_store(store, run_root)
    export_prov(store, run_root)
    return {"status": OK, NODE: outcome.status}


def process(
    rows: Sequence[dict[str, Any]],
    config: TriageConfig,
    ask: Ask,
    *,
    force: bool,
    build_hint: Callable[[Path], Any] | None = None,
    workers: int = 1,
    log: SliceLog | None = None,
) -> list[dict[str, Any]]:
    """Ask about every run named by these rows, ``workers`` at a time.

    Args:
        rows: The manifest rows this task owns.
        config: The triage configuration.
        ask: Sends a state and questions to the model; called from ``workers`` threads at once.
        force: Whether to ask again where an opinion stands.
        build_hint: The hint populator, or None to ask without the declaration's task context.
        workers: How many rows are asked at once, which is also the server's parallelism.
        log: Where each row's record is appended as it lands, or None.

    Returns:
        One outcome record per input row, in order.
    """

    def one(row: dict[str, Any]) -> dict[str, Any]:
        try:
            run_root = run_root_of(Path(row["enhanced"]))
        except ValueError as error:
            return {**row, "status": ERROR, NODE: describe_exception(error)}
        source: Path | None = None
        if build_hint is not None:
            try:
                source = Path(str(row["source"])) if row.get("source") else source_of(run_root)
            except (OSError, ValueError, KeyError) as error:
                return {**row, "status": ERROR, NODE: f"source: {describe_exception(error)}"}
        return {
            **row,
            **extend_one(
                run_root, config, ask, force=force, build_hint=build_hint, source=source, num_parallel=workers
            ),
        }

    return logged(rows, one, log, workers=workers)


def run_slice(
    manifest: Path,
    *,
    slice_index: int,
    slice_count: int,
    config: TriageConfig,
    log_dir: Path,
    open_ask: OpenAsk,
    force: bool = False,
    hints: Path | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Ask about every run in one array task's stride of the manifest.

    Args:
        manifest: The manifest JSONL.
        slice_index: This task's 0-based index.
        slice_count: How many tasks the array has.
        config: The triage configuration.
        log_dir: Where this task's ``slices/`` log goes.
        open_ask: Opens the asker; called at most once, on the first row that needs it.
        force: Whether to ask again where an opinion stands.
        hints: The directory holding ``hints.py``, or None to ask without the declaration's task context.
        workers: How many rows are asked at once; the asker must answer as many in parallel.

    Returns:
        The task's summary: its counts, its parameters, and where its log went.

    Raises:
        SliceAbortedError: If the server did not start or kept failing; the rows logged so far stand.
    """
    started = time.time()
    mine = take_slice(read_manifest(manifest, required=("stem", "enhanced")), slice_index, slice_count)
    print(f"[slice {slice_index}/{slice_count}] {len(mine)} rows, {workers} at once", flush=True)
    build_hint = load_hint_builder(hints) if hints is not None else None
    slices_dir = log_dir / SLICES_SUBDIR
    label = f"second-opinion-slice-{slice_index}-of-{slice_count}"
    log_path = slices_dir / f"{label}.jsonl"
    rows_log = SliceLog(log_path, slice_index=slice_index, slice_count=slice_count, total=len(mine), workers=workers)
    ask = LazyAsk(open_ask, max_consecutive_errors=int(settings(config)["max_consecutive_errors"]))
    try:
        log = process(mine, config, ask, force=force, build_hint=build_hint, workers=workers, log=rows_log)
    finally:
        rows_log.close()
        server_started = ask.opened
        ask.close()

    counts: dict[str, int] = {}
    readings: dict[str, int] = {}
    for record in log:
        counts[str(record["status"])] = counts.get(str(record["status"]), 0) + 1
        readings[str(record[NODE])] = readings.get(str(record[NODE]), 0) + 1
    pin = pin_of(config)
    summary = {
        "manifest": str(manifest),
        "slice_index": slice_index,
        "slice_count": slice_count,
        "config_hash": config.config_hash,
        "model_id": pin.model_id,
        "blob_digest": pin.blob_digest,
        "server_started": server_started,
        "rows": len(mine),
        "counts": counts,
        "readings": readings,
        "force": force,
        "hints": str(hints) if hints is not None else None,
        "workers": workers,
        "host": os.uname().nodename,
        "elapsed_s": time.time() - started,
        "log": str(log_path),
    }
    (slices_dir / f"{label}.summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Ask the second-opinion model about one array task's stride of a finished corpus.

    Args:
        argv: The command line, or None for ``sys.argv``.

    Returns:
        0 where every row landed or already stood, 1 where any row is ``error``, 2 where the
        arguments could not be resolved and nothing was asked, 3 where the slice stopped because
        the server did not start or kept failing.
    """
    args = build_parser().parse_args(argv)
    if not args.manifest.exists():
        print(f"ERROR: manifest not found: {args.manifest}", file=sys.stderr)
        return 2
    binary = args.ollama_binary or (Path(os.environ["OLLAMA_BINARY"]) if os.environ.get("OLLAMA_BINARY") else None)
    models = args.ollama_models or (Path(os.environ["OLLAMA_MODELS"]) if os.environ.get("OLLAMA_MODELS") else None)
    if binary is None or models is None:
        print(
            "ERROR: --ollama-binary and --ollama-models (or OLLAMA_BINARY, OLLAMA_MODELS) are required.",
            file=sys.stderr,
        )
        return 2
    try:
        config = load_triage_config(args.config) if args.config else load_triage_config()
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    held = settings(config)
    if not held["enabled"]:
        print("ERROR: second_opinion.enabled is false; pass a --config that enables it.", file=sys.stderr)
        return 2
    workers = int(args.workers if args.workers is not None else held["workers"])
    if workers < 1:
        print(f"ERROR: --workers must be at least 1, got {workers}.", file=sys.stderr)
        return 2
    try:
        verify_pin(models, pin_of(config), verified_dir=args.verified_dir)
    except PinMismatchError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    log_dir = args.log_dir if args.log_dir is not None else args.manifest.parent
    server_log = log_dir / SLICES_SUBDIR / f"second-opinion-slice-{args.slice_index}-of-{args.slice_count}.ollama.log"
    server_log.parent.mkdir(parents=True, exist_ok=True)
    try:
        summary = run_slice(
            args.manifest,
            slice_index=args.slice_index,
            slice_count=args.slice_count,
            config=config,
            log_dir=log_dir,
            open_ask=ollama_asker(
                config,
                binary=binary,
                models_dir=models,
                verified_dir=args.verified_dir,
                log_path=server_log,
                num_parallel=workers,
            ),
            force=args.force,
            hints=args.hints,
            workers=workers,
        )
    except SliceAbortedError as error:
        print(f"ABORTED: {error}", file=sys.stderr)
        return 3
    except (OSError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(f"Rows:    {summary['rows']}")
    print(f"Log:     {summary['log']}")
    for status, number in sorted(summary["counts"].items()):
        print(f"  {status:<9} {number}")
    for reading, number in sorted(summary["readings"].items()):
        print(f"  read {reading:<16} {number}")
    return 1 if summary["counts"].get(ERROR) else 0


if __name__ == "__main__":
    raise SystemExit(main())
