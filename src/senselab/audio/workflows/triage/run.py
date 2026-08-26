"""The triage runner: one recording driven through the graph over one provenance store.

Holds no thresholds and decides nothing. It builds the run's directory layout, calls each node in
the graph's order, records whether each one completed, was skipped or raised, and hands that mapping
to VERDICT — which is the only place ``errored`` can come from, since a node that raised wrote no
verdict and the store cannot tell it from one never asked to run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.enrollment import Enrollment
from senselab.audio.workflows.triage.nodes.admit import admit
from senselab.audio.workflows.triage.nodes.airway import airway
from senselab.audio.workflows.triage.nodes.common import NodeResult, describe_exception
from senselab.audio.workflows.triage.nodes.preprocess import preprocess
from senselab.audio.workflows.triage.nodes.redact import redact
from senselab.audio.workflows.triage.nodes.report import report
from senselab.audio.workflows.triage.nodes.routing import routing
from senselab.audio.workflows.triage.nodes.speech import speech
from senselab.audio.workflows.triage.nodes.taxonomy import taxonomy
from senselab.audio.workflows.triage.nodes.verdict import verdict
from senselab.audio.workflows.triage.nodes.voice import voice
from senselab.audio.workflows.triage.vocabulary import (
    BRANCHES,
    GRAPH_ORDER,
    FileVerdict,
    NodeVerdict,
    Outcome,
    RunState,
)
from senselab.utils.prov_store import ProvStore

REPORT_NODE = "REPORT"

STORE_FILE = "store.jsonl"
LOG_FILE = "run.json"
RUN_SUBDIR = "run"
RELEASE_SUBDIR = "released"
SUMMARY_SUBDIR = "summary"
SIDECAR_SUBDIRS = ("streams", "derivatives")

_RUN_STAMP = "%Y%m%d-%H%M%S"
_CONDITIONED_STREAM = "plain"
_SOURCE_STREAM = "recording"

_R = TypeVar("_R", bound=NodeResult)


@dataclass(frozen=True)
class NodeOutcome:
    """What one node did on one run.

    Attributes:
        node: The node's name.
        state: Whether it completed, was skipped, or raised.
        verdict: Its conclusion, or None when it did not conclude.
        error: The exception's type and message when it raised, else None. The runner's own record —
            never a store fact, because a crash is not a finding about the recording.
    """

    node: str
    state: RunState
    verdict: NodeVerdict | None = None
    error: str | None = None


@dataclass(frozen=True)
class TriageRunResult:
    """What one run of the graph produced.

    Attributes:
        file_verdict: The graph's conclusion on both axes, or None when VERDICT itself raised.
        nodes: Per-node outcome, in graph order.
        run_dir: The run directory holding the store and every sidecar.
        artifacts_dir: The release directory REDACT was given; disjoint from ``run_dir``.
        store_path: The persisted store.
        summary_dir: Where REPORT's two products go. Beside the store, never under ``artifacts_dir``:
            both carry element ids, which are a join key back into the store.
        released: REDACT's released pair, empty unless it cleared one.
        summary: REPORT's two products, empty when REPORT itself raised.
    """

    file_verdict: FileVerdict | None
    nodes: dict[str, NodeOutcome]
    run_dir: Path
    artifacts_dir: Path
    store_path: Path
    summary_dir: Path
    released: dict[str, Path] = field(default_factory=dict)
    summary: dict[str, Path] = field(default_factory=dict)

    @property
    def ran(self) -> dict[str, RunState]:
        """Whether each node ran, keyed by node name.

        Returns:
            The run state per node, in graph order.
        """
        return {node: outcome.state for node, outcome in self.nodes.items()}


@dataclass(frozen=True)
class RunLayout:
    """One run's directories.

    Attributes:
        root: The per-run root the two trees below are siblings in.
        run_dir: Where the store and every sidecar go.
        artifacts_dir: Where REDACT may release a pair. Fresh and empty on every run.
        store_path: Where the store is persisted.
        summary_dir: Where REPORT's two products go, a sibling of the store tree and of the release
            tree alike.
    """

    root: Path
    run_dir: Path
    artifacts_dir: Path
    store_path: Path
    summary_dir: Path


def prepare_run_layout(out_dir: Path, stem: str) -> RunLayout:
    """Create a fresh run root under ``out_dir``, with the store tree and the release tree apart.

    The release directory is created empty on every run and is never reused: a directory still
    holding an earlier run's released pair would let a withheld run appear to have published one.
    Two runs that land in the same second are separated by a numeric suffix rather than merged.

    Args:
        out_dir: Where run roots are created.
        stem: The recording's file stem, used to name the run root.

    Returns:
        The run's directories, all of them created.
    """
    stamp = datetime.now(timezone.utc).strftime(_RUN_STAMP)
    attempt = 0
    while True:
        suffix = "" if attempt == 0 else f"-{attempt}"
        root = out_dir / f"{stem}_{stamp}{suffix}"
        try:
            root.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            attempt += 1
            continue
        break
    run_dir = root / RUN_SUBDIR
    for subdir in SIDECAR_SUBDIRS:
        (run_dir / subdir).mkdir(parents=True)
    artifacts_dir = root / RELEASE_SUBDIR
    artifacts_dir.mkdir()
    return RunLayout(
        root=root,
        run_dir=run_dir,
        artifacts_dir=artifacts_dir,
        store_path=run_dir / STORE_FILE,
        summary_dir=root / SUMMARY_SUBDIR,
    )


def _attempt(outcomes: dict[str, NodeOutcome], node: str, call: Callable[[], _R]) -> _R | None:
    """Call one node, recording what happened instead of propagating a failure.

    Args:
        outcomes: The per-node record this call is added to.
        node: The node's name.
        call: The node call, already bound to its arguments.

    Returns:
        The node's result, or None when it raised or returned no result.
    """
    try:
        result = call()
        if result is None:
            raise RuntimeError(f"{node} returned no result")
        node_verdict = result.verdict
    except Exception as error:  # noqa: BLE001 — any failure is an operational fact about the run
        outcomes[node] = NodeOutcome(node=node, state=RunState.ERRORED, error=describe_exception(error))
        return None
    outcomes[node] = NodeOutcome(node=node, state=RunState.COMPLETED, verdict=node_verdict)
    return result


def _speech_found_pii(store: ProvStore) -> bool:
    """Whether SPEECH's scan over the consensus transcript found anything.

    Args:
        store: The provenance store.

    Returns:
        True when at least one live ``pii`` entity is in the store.
    """
    return bool([finding for finding in store.entities("pii") if not store.is_invalidated(finding.id)])


def _drive_branches(
    store: ProvStore,
    audio: Audio,
    config: TriageConfig,
    hint: AudioHints | None,
    *,
    run_dir: Path,
    artifacts_dir: Path,
    outcomes: dict[str, NodeOutcome],
    enrollment: Enrollment | None,
) -> dict[str, Path]:
    """Run PREPROCESS, TAXONOMY and routing, then exactly the branches routing selected.

    A branch routing declined is recorded ``SKIPPED`` and never called, which is what lets VERDICT
    tell a branch that found nothing from a branch that never looked. A failed ROUTING call likewise
    leaves every branch ``SKIPPED``: no branch has an authorised decision to act on. A branch that
    raises is still recorded ``ERRORED`` and its siblings still run: none of them reads another's
    output. REDACT is a step of SPEECH and runs only when SPEECH ran and its scan found PII.

    Args:
        store: The provenance store, already holding ADMIT's ``recording`` stream.
        audio: The audio ADMIT decoded.
        config: The triage configuration.
        hint: What the recording was declared to contain.
        run_dir: The run directory sidecar paths are relative to.
        artifacts_dir: The release directory handed to REDACT.
        outcomes: The per-node record each call is added to.
        enrollment: The target speaker's enrollment, when the caller supplied one.

    Returns:
        REDACT's released pair, empty unless it cleared one.
    """
    _attempt(outcomes, "PREPROCESS", lambda: preprocess(store, audio, config, hint, run_dir=run_dir))
    _attempt(outcomes, "TAXONOMY", lambda: taxonomy(store, _CONDITIONED_STREAM, config, hint, run_dir=run_dir))
    routed = _attempt(outcomes, "routing", lambda: routing(store, None, config, hint, run_dir=run_dir))
    selected = set(routed.runs) if routed is not None else set()
    branches: dict[str, Callable[[], NodeResult]] = {
        "AIRWAY": lambda: airway(store, _CONDITIONED_STREAM, config, hint, run_dir=run_dir),
        "SPEECH": lambda: speech(store, _CONDITIONED_STREAM, config, hint, run_dir=run_dir, enrollment=enrollment),
        "VOICE": lambda: voice(store, _CONDITIONED_STREAM, config, hint, run_dir=run_dir),
    }
    for branch in BRANCHES:
        if branch in selected:
            _attempt(outcomes, branch, branches[branch])
        else:
            outcomes[branch] = NodeOutcome(node=branch, state=RunState.SKIPPED)
    if "SPEECH" in selected and _speech_found_pii(store):
        redacted = _attempt(
            outcomes,
            "REDACT",
            lambda: redact(store, _SOURCE_STREAM, config, hint, run_dir=run_dir, artifacts_dir=artifacts_dir),
        )
        return dict(redacted.artifacts) if redacted is not None else {}
    outcomes["REDACT"] = NodeOutcome(node="REDACT", state=RunState.SKIPPED)
    return {}


def _attempt_artifacts(
    outcomes: dict[str, NodeOutcome], node: str, call: Callable[[], dict[str, Path]]
) -> dict[str, Path]:
    """Call one node that renders rather than concludes, recording what happened.

    REPORT writes no verdict — a rendering is not evidence — so its outcome carries no conclusion,
    and its failure changes nothing about the file: the store was already written.

    Args:
        outcomes: The per-node record this call is added to.
        node: The node's name.
        call: The node call, already bound to its arguments.

    Returns:
        The artifacts it produced. A failure that carried artifacts on the exception keeps them —
        REPORT writes its JSON before it draws, and losing a complete product because the picture
        beside it could not be drawn would be a second failure, not a consequence of the first.
    """
    try:
        artifacts = call()
    except Exception as error:  # noqa: BLE001 — any failure is an operational fact about the run
        outcomes[node] = NodeOutcome(node=node, state=RunState.ERRORED, error=describe_exception(error))
        salvaged = getattr(error, "artifacts", None)
        return dict(salvaged) if isinstance(salvaged, dict) else {}
    outcomes[node] = NodeOutcome(node=node, state=RunState.COMPLETED)
    return artifacts


def _write_log(path: Path, source: Path, config: TriageConfig, result: TriageRunResult) -> None:
    """Write the runner's own record of the run beside the store.

    Args:
        path: Where the log goes.
        source: The recording that was triaged.
        config: The configuration the run used.
        result: What the run produced.
    """
    payload: dict[str, Any] = {
        "source": str(source),
        "config": {"name": config.name, "version": config.version, "config_hash": config.config_hash},
        "run_dir": str(result.run_dir),
        "artifacts_dir": str(result.artifacts_dir),
        "store": str(result.store_path),
        "triage": result.file_verdict.triage.value if result.file_verdict is not None else None,
        "release": result.file_verdict.release.value if result.file_verdict is not None else None,
        "ran": {node: outcome.state.value for node, outcome in result.nodes.items()},
        "errors": {node: outcome.error for node, outcome in result.nodes.items() if outcome.error is not None},
        "released": {name: str(released) for name, released in result.released.items()},
        "summary_dir": str(result.summary_dir),
        "summary": {name: str(product) for name, product in result.summary.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_triage(
    source: Path,
    out_dir: Path,
    config: TriageConfig,
    hint: AudioHints | None = None,
    enrollment: Enrollment | None = None,
) -> TriageRunResult:
    """Triage one recording: the whole graph, one store, one fresh run directory.

    ADMIT is the initial gate. A ``fail`` there means the recording was never measured, so every other
    node is recorded as ``skipped`` and only VERDICT runs, which is what makes "could not measure"
    distinguishable from "measured, and found nothing". Every other node's failure is captured rather
    than propagated and the store is persisted either way. ROUTING is the only later dependency gate:
    when it fails, its dependent branches are recorded ``skipped`` because no execution decision was
    written; other nodes' independent siblings still run.

    Args:
        source: The recording to triage.
        out_dir: Where the run root is created.
        config: The triage configuration.
        hint: What the recording was declared to contain, if anything.
        enrollment: The target speaker's enrollment, handed to SPEECH when the caller supplied one.

    Returns:
        The file verdict, the per-node outcomes and errors, the run's paths, REDACT's released pair
        and REPORT's two products. REPORT runs after VERDICT on every outcome and writes no elements,
        so its own failure is recorded beside every other node's and changes no verdict.
    """
    source = Path(source)
    layout = prepare_run_layout(Path(out_dir), source.stem)
    store = ProvStore(run_id=layout.root.name)
    outcomes: dict[str, NodeOutcome] = {}

    admitted = _attempt(outcomes, "ADMIT", lambda: admit(store, source, config, hint, run_dir=layout.run_dir))
    measurable = admitted is not None and admitted.verdict.outcome is not Outcome.FAIL and admitted.audio is not None

    released: dict[str, Path] = {}
    if admitted is not None and measurable and admitted.audio is not None:
        released = _drive_branches(
            store,
            admitted.audio,
            config,
            hint,
            run_dir=layout.run_dir,
            artifacts_dir=layout.artifacts_dir,
            outcomes=outcomes,
            enrollment=enrollment,
        )
    else:
        for node in GRAPH_ORDER[1:-1]:
            outcomes[node] = NodeOutcome(node=node, state=RunState.SKIPPED)

    ran = {node: outcome.state for node, outcome in outcomes.items()}
    folded = _attempt(
        outcomes,
        "VERDICT",
        lambda: verdict(store, None, config, hint, run_dir=layout.run_dir, ran=ran),
    )
    store.write_jsonl(layout.store_path)

    summary = _attempt_artifacts(
        outcomes,
        REPORT_NODE,
        lambda: report(store, layout.summary_dir, config, run_dir=layout.run_dir),
    )

    result = TriageRunResult(
        file_verdict=folded.file_verdict if folded is not None else None,
        nodes={node: outcomes[node] for node in (*GRAPH_ORDER, REPORT_NODE) if node in outcomes},
        run_dir=layout.run_dir,
        artifacts_dir=layout.artifacts_dir,
        store_path=layout.store_path,
        summary_dir=layout.summary_dir,
        released=released,
        summary=summary,
    )
    _write_log(layout.run_dir / LOG_FILE, source, config, result)
    return result
