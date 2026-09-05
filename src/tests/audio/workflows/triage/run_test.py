"""Behavioural tests for the triage runner and its CLI.

The runner's contract is ordering, error capture, gating and paths — not what any node measures. So
every node is replaced at the module boundary by a fake that records its call and writes a real
``verdict`` entity, and VERDICT folds those for real. The whole-graph integration over real models
lives in the per-node suites.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any, Callable

import pytest
import torch

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.workflows.triage import run as run_module
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.admit import AdmitResult
from senselab.audio.workflows.triage.nodes.common import NodeResult, software_agent, write_verdict
from senselab.audio.workflows.triage.nodes.preprocess import PreprocessResult
from senselab.audio.workflows.triage.nodes.redact import RedactResult
from senselab.audio.workflows.triage.nodes.report import ReportRenderError
from senselab.audio.workflows.triage.nodes.routing import routing as real_routing
from senselab.audio.workflows.triage.nodes.taxonomy import TaxonomyResult
from senselab.audio.workflows.triage.run import run_triage
from senselab.audio.workflows.triage.vocabulary import (
    FileVerdict,
    NodeVerdict,
    Outcome,
    Release,
    RunState,
    Triage,
)
from senselab.utils.prov_store import ProvStore

REPO_ROOT = Path(__file__).resolve().parents[5]
CLI = REPO_ROOT / "scripts" / "triage_audio.py"

GRAPH = ("ADMIT", "PREPROCESS", "TAXONOMY", "routing", "AIRWAY", "SPEECH", "VOICE", "REDACT", "VERDICT")

_MISSING = object()


@pytest.fixture
def config() -> TriageConfig:
    """The packaged configuration, unmodified."""
    return load_triage_config()


def _conclude(store: ProvStore, node: str, outcome: Outcome, kind: str | None) -> tuple[str, NodeVerdict]:
    """Write one fake node's verdict entity so the real VERDICT can fold it."""
    activity = store.activity(node=node, step=None, parameters={})
    agent = software_agent(store)
    store.was_associated_with(activity, agent)
    return write_verdict(
        store, activity, agent, node=node, outcome=outcome, kind=kind, why="a fake node concluded", detail={}
    )


def _tone() -> Audio:
    """A short non-constant waveform, standing in for what ADMIT decoded."""
    return Audio(waveform=torch.linspace(-0.5, 0.5, 16000).unsqueeze(0), sampling_rate=16000)


def _fakes(
    calls: list[str],
    *,
    admit_outcome: Outcome = Outcome.PASS,
    raising: str | None = None,
    released: dict[str, Path] | None = None,
    kinds: dict[str, str] | None = None,
    pii: bool = True,
    routing_outcome: str | None = None,
) -> dict[str, Callable[..., Any]]:
    """Fake node functions, one per graph node, recording their calls into ``calls``.

    ``routing`` is the exception: its entry calls the **real** node over the store the fake TAXONOMY
    seeded, so the gate under test is the production one and only the branches around it are faked.

    Args:
        calls: The list every fake appends its node name to, in call order.
        admit_outcome: What the fake ADMIT concludes; ``FAIL`` returns no audio.
        raising: The node whose fake raises ``RuntimeError`` instead of concluding.
        released: What the fake REDACT reports as its released pair.
        kinds: The classification the fake TAXONOMY writes as ``kind`` entities. None writes none,
            which is what a run where TAXONOMY never concluded leaves behind.
        pii: Whether the fake SPEECH writes a live ``pii`` entity, which is REDACT's whole gate.
        routing_outcome: ``"raise"`` makes the routing entry raise; ``"none"`` makes it return no
            result instead of calling the real node.

    Returns:
        The fakes, keyed by the attribute name they replace on the runner's module.
    """

    def _record(node: str) -> None:
        calls.append(node)
        if node == raising:
            raise RuntimeError(f"{node} could not run")

    def _admit(
        store: ProvStore, source: str | Path, config: TriageConfig, hint: AudioHints | None = None, *, run_dir: Path
    ) -> AdmitResult:
        _record("ADMIT")
        entity_id, verdict = _conclude(store, "ADMIT", admit_outcome, None)
        audio = None if admit_outcome is Outcome.FAIL else _tone()
        return AdmitResult(verdict=verdict, view=(entity_id,), verdict_entity_id=entity_id, audio=audio)

    def _preprocess(
        store: ProvStore, source: Audio, config: TriageConfig, hint: AudioHints | None = None, *, run_dir: Path
    ) -> PreprocessResult:
        _record("PREPROCESS")
        entity_id, verdict = _conclude(store, "PREPROCESS", Outcome.PASS, None)
        return PreprocessResult(verdict=verdict, view=(entity_id,), verdict_entity_id=entity_id, absent=())

    def _taxonomy(
        store: ProvStore, source: str, config: TriageConfig, hint: AudioHints | None = None, *, run_dir: Path
    ) -> TaxonomyResult:
        _record("TAXONOMY")
        fold = store.activity(node="TAXONOMY", step="fold", parameters={})
        for kind, state in (kinds or {}).items():
            kind_id = store.entity(
                prov_type="kind",
                extent=None,
                attributes={"kind": kind, "state": state, "lines": {}, "stream": "plain"},
            )
            store.was_generated_by(kind_id, fold)
        entity_id, verdict = _conclude(store, "TAXONOMY", Outcome.PASS, None)
        return TaxonomyResult(verdict=verdict, view=(entity_id,), verdict_entity_id=entity_id, kinds=dict(kinds or {}))

    def _routing(
        store: ProvStore, source: str | None, config: TriageConfig, hint: AudioHints | None = None, *, run_dir: Path
    ) -> Any:  # noqa: ANN401
        _record("routing")
        if routing_outcome == "raise":
            raise RuntimeError("routing could not run")
        if routing_outcome == "none":
            return None
        return real_routing(store, source, config, hint, run_dir=run_dir)

    def _airway(
        store: ProvStore, source: str, config: TriageConfig, hint: AudioHints | None = None, *, run_dir: Path
    ) -> NodeResult:
        _record("AIRWAY")
        entity_id, verdict = _conclude(store, "AIRWAY", Outcome.PASS, "airway")
        return NodeResult(verdict=verdict, view=(entity_id,), verdict_entity_id=entity_id)

    def _speech(
        store: ProvStore,
        source: str,
        config: TriageConfig,
        hint: AudioHints | None = None,
        *,
        run_dir: Path,
        enrollment: Any = None,  # noqa: ANN401
    ) -> NodeResult:
        _record("SPEECH")
        if pii:
            scan = store.activity(node="SPEECH", step="pii", parameters={})
            finding = store.entity(prov_type="pii", extent=(0.0, 1.0), attributes={"category": "name"})
            store.was_generated_by(finding, scan)
        entity_id, verdict = _conclude(store, "SPEECH", Outcome.PASS, "speech")
        return NodeResult(verdict=verdict, view=(entity_id,), verdict_entity_id=entity_id)

    def _voice(
        store: ProvStore, source: str, config: TriageConfig, hint: AudioHints | None = None, *, run_dir: Path
    ) -> NodeResult:
        _record("VOICE")
        entity_id, verdict = _conclude(store, "VOICE", Outcome.PASS, "voice")
        return NodeResult(verdict=verdict, view=(entity_id,), verdict_entity_id=entity_id)

    def _redact(
        store: ProvStore,
        source: str,
        config: TriageConfig,
        hint: AudioHints | None = None,
        *,
        run_dir: Path,
        artifacts_dir: Path,
    ) -> RedactResult:
        _record("REDACT")
        entity_id, verdict = _conclude(store, "REDACT", Outcome.PASS, None)
        artifacts: dict[str, Path] = {}
        for name, payload in (released or {}).items():
            path = artifacts_dir / Path(payload).name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(str(payload))
            artifacts[name] = path
        return RedactResult(verdict=verdict, view=(entity_id,), verdict_entity_id=entity_id, artifacts=artifacts)

    return {
        "admit": _admit,
        "preprocess": _preprocess,
        "taxonomy": _taxonomy,
        "routing": _routing,
        "airway": _airway,
        "speech": _speech,
        "voice": _voice,
        "redact": _redact,
    }


@pytest.fixture
def graph(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[str]]:
    """Install the fake graph on the runner's module and return the list recording call order."""

    def _install(**kwargs: Any) -> list[str]:  # noqa: ANN401
        calls: list[str] = []
        for name, fake in _fakes(calls, **kwargs).items():
            monkeypatch.setattr(run_module, name, fake)
        real_verdict = run_module.verdict

        def _verdict(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            calls.append("VERDICT")
            return real_verdict(*args, **kwargs)

        monkeypatch.setattr(run_module, "verdict", _verdict)
        return calls

    return _install


class TestHappyPath:
    """Every node runs, in order, and the fold is reported."""

    def test_calls_all_nine_nodes_in_graph_order(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The runner drives the DAG in the graph's declared order, VERDICT last."""
        calls = graph()
        run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert tuple(calls) == GRAPH

    def test_returns_the_file_verdict_with_every_node_completed(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A graph in which nothing raised reports ``COMPLETED`` for all nine nodes."""
        graph()
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert result.file_verdict is not None
        assert result.file_verdict.triage is Triage.PASS
        assert result.ran == {**dict.fromkeys(GRAPH, RunState.COMPLETED), "REPORT": RunState.COMPLETED}

    def test_the_layout_is_written_and_the_release_dir_is_disjoint(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """run_dir carries the store and its three sidecar trees; artifacts_dir contains neither."""
        graph(released={"audio": "audio.wav", "transcript": "transcript.txt"})
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert result.store_path == result.run_dir / "store.jsonl"
        assert result.store_path.is_file()
        for sub in ("streams", "derivatives"):
            assert (result.run_dir / sub).is_dir()
        run_resolved, release_resolved = result.run_dir.resolve(), result.artifacts_dir.resolve()
        assert not run_resolved.is_relative_to(release_resolved)
        assert not release_resolved.is_relative_to(run_resolved)
        assert sorted(result.released) == ["audio", "transcript"]
        assert all(path.parent == result.artifacts_dir for path in result.released.values())

    def test_the_summary_sits_beside_the_store_and_never_under_released(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """It carries element ids and marked words' extents, so it inherits the store's sensitivity."""
        graph(released={"audio": "audio.wav", "transcript": "transcript.txt"})
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert result.summary_dir.parent == result.run_dir.parent
        assert not result.summary_dir.resolve().is_relative_to(result.artifacts_dir.resolve())
        assert not result.artifacts_dir.resolve().is_relative_to(result.summary_dir.resolve())

    def test_both_products_are_emitted_on_every_run(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """REPORT runs after VERDICT, on every outcome, and its products land under summary/."""
        graph()
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert sorted(result.summary) == ["json", "summary"]
        assert all(product.parent == result.summary_dir for product in result.summary.values())
        assert all(product.is_file() for product in result.summary.values())

    def test_a_refused_file_still_gets_both_products(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The file that most needs a report is the one ADMIT refused (V24)."""
        graph(admit_outcome=Outcome.FAIL)
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert sorted(result.summary) == ["json", "summary"]
        assert json.loads(result.summary["json"].read_text())["verdict"]["triage"] == "discard"

    def test_a_report_failure_is_recorded_and_changes_no_verdict(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The store was already written; a rendering that failed is an operational fact, not a finding."""
        graph()

        def _raise(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            raise RuntimeError("the renderer could not run")

        monkeypatch.setattr(run_module, "report", _raise)
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert result.summary == {}
        assert result.ran["REPORT"] is RunState.ERRORED
        assert result.file_verdict is not None and result.file_verdict.triage is Triage.PASS
        assert result.store_path.is_file()

    def test_a_drawing_failure_keeps_the_json_it_had_already_written(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A consumer reading many files must not lose one because a page could not be drawn."""
        graph()
        written = tmp_path / "salvaged.json"
        written.write_text("{}")

        def _raise(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            raise ReportRenderError("the canvas is gone", {"json": written})

        monkeypatch.setattr(run_module, "report", _raise)
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert result.summary == {"json": written}
        assert result.ran["REPORT"] is RunState.ERRORED
        assert "the canvas is gone" in (result.nodes["REPORT"].error or "")
        assert json.loads((result.run_dir / "run.json").read_text())["summary"] == {"json": str(written)}

    def test_the_release_axis_follows_redact(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A passing REDACT clears its own pair, so the fold reads ``releasable``."""
        graph()
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert result.file_verdict is not None
        assert result.file_verdict.release is Release.RELEASABLE

    def test_the_hint_reaches_every_node(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The caller's hint is handed to each node rather than dropped at the runner."""
        seen: list[AudioHints | None] = []
        graph()
        for name in ("admit", "preprocess", "taxonomy", "routing", "airway", "speech", "voice", "redact", "verdict"):
            original = getattr(run_module, name)

            def _spy(*args: Any, _original: Any = original, **kwargs: Any) -> Any:  # noqa: ANN401
                seen.append(args[3] if len(args) > 3 else kwargs.get("hint"))
                return _original(*args, **kwargs)

            monkeypatch.setattr(run_module, name, _spy)
        hint = AudioHints(may_contain=["cough"])
        run_triage(tmp_path / "recording.wav", tmp_path / "out", config, hint=hint)
        assert seen == [hint] * len(GRAPH)


class TestConditionalExecution:
    """run.py runs the branches routing selected, and records the rest as skipped."""

    def test_a_skipped_branch_is_not_called_and_is_recorded_skipped(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A branch with will_run false never runs, and RunState.SKIPPED says so."""
        calls = graph(kinds={"speech": "present", "airway": "absent", "voice": "absent"})
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert "AIRWAY" not in calls
        assert result.ran["AIRWAY"] is RunState.SKIPPED
        assert result.ran["SPEECH"] is RunState.COMPLETED

    def test_redact_runs_only_when_speech_ran_and_found_pii(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """REDACT is a step of SPEECH; no speech branch means no REDACT verdict at all."""
        calls = graph(kinds={"speech": "absent", "airway": "present", "voice": "absent"})
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert "REDACT" not in calls
        assert result.ran["REDACT"] is RunState.SKIPPED

    def test_speech_running_without_a_finding_still_skips_redact(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """redact.md: SPEECH ran and found no PII, so the release axis reads not_assessed."""
        calls = graph(kinds={"speech": "present", "airway": "absent", "voice": "absent"}, pii=False)
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert "SPEECH" in calls and "REDACT" not in calls
        assert result.file_verdict is not None
        assert result.file_verdict.release is Release.NOT_ASSESSED

    def test_speech_running_with_a_finding_reaches_redact(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """One live pii entity is the whole gate."""
        calls = graph(kinds={"speech": "present", "airway": "absent", "voice": "absent"}, pii=True)
        run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert "REDACT" in calls

    def test_an_empty_execution_set_still_reaches_verdict(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The file reaches the fold with no branch conclusions, and the fold discards it as empty.

        End to end over the real ROUTING and the real VERDICT: this is the whole point of routing
        recording the empty set rather than flagging it, and only the runner exercises both nodes.
        """
        calls = graph(kinds={"speech": "absent", "airway": "absent", "voice": "absent"})
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert result.ran["VERDICT"] is RunState.COMPLETED
        assert {"AIRWAY", "SPEECH", "VOICE"}.isdisjoint(calls)
        assert result.file_verdict is not None
        assert result.file_verdict.triage is Triage.DISCARD
        assert result.file_verdict.discard_ground == "acoustically_empty"

    @pytest.mark.parametrize("routing_outcome", ["raise", "none"])
    def test_a_failed_routing_skips_dependent_branches_and_flags_the_file(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path, routing_outcome: str
    ) -> None:
        """Branches do not run without ROUTING's decisions, and VERDICT records why they did not."""
        calls = graph(routing_outcome=routing_outcome)
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert tuple(calls) == ("ADMIT", "PREPROCESS", "TAXONOMY", "routing", "VERDICT")
        assert result.ran["routing"] is RunState.ERRORED
        assert all(result.ran[branch] is RunState.SKIPPED for branch in ("AIRWAY", "SPEECH", "VOICE", "REDACT"))
        assert result.file_verdict is not None
        assert result.file_verdict.triage is Triage.FLAG
        assert any(
            "routing failed; branch execution was withheld" in reason.why for reason in result.file_verdict.reasons
        )


class TestNodeErrorsAreCaptured:
    """A node that raises is recorded, and the rest of the graph still runs."""

    def test_a_raising_voice_errors_without_stopping_the_graph(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """VOICE raising leaves it ``ERRORED``; REDACT and VERDICT still run."""
        calls = graph(raising="VOICE")
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert tuple(calls) == GRAPH
        assert result.ran["VOICE"] is RunState.ERRORED
        assert result.ran["REDACT"] is RunState.COMPLETED
        assert result.ran["VERDICT"] is RunState.COMPLETED
        assert result.file_verdict is not None

    def test_the_error_is_kept_in_the_result_and_never_in_the_store(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The exception's type and message are the runner's own record, not a store fact."""
        graph(raising="VOICE")
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        error = result.nodes["VOICE"].error
        assert error is not None
        assert "RuntimeError" in error and "VOICE could not run" in error
        assert "VOICE could not run" not in result.store_path.read_text()

    def test_the_store_is_persisted_and_reads_back(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A run with an errored node still writes a store that reads back under its invariants."""
        graph(raising="VOICE")
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        reread = ProvStore.read_jsonl(result.store_path)
        concluded = {entity.attributes["node"] for entity in reread.entities("verdict")}
        assert "VOICE" not in concluded
        assert {"ADMIT", "PREPROCESS", "TAXONOMY", "AIRWAY", "SPEECH", "REDACT", "VERDICT"} <= concluded

    def test_the_errored_state_reaches_the_folded_verdict(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Only the runner can report ``errored``, so its mapping wins over what the store derives."""
        graph(raising="VOICE")
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert result.file_verdict is not None
        assert result.file_verdict.ran["VOICE"] is RunState.ERRORED

    def test_a_raising_preprocess_skips_every_node_that_reads_its_output(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Every later node reads what PREPROCESS measured, so a raise there leaves them SKIPPED."""
        calls = graph(raising="PREPROCESS")
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert tuple(calls) == ("ADMIT", "PREPROCESS", "VERDICT")
        assert result.ran["PREPROCESS"] is RunState.ERRORED
        assert result.ran["TAXONOMY"] is RunState.SKIPPED


class TestPreprocessFailShortCircuits:
    """PREPROCESS is the second gate: a raise skips every node that depends on its measurements."""

    def test_taxonomy_routing_and_every_branch_are_skipped(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """None of them has any evidence to act on, so none of them is attempted."""
        calls = graph(raising="PREPROCESS")
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        skipped = ("TAXONOMY", "routing", "AIRWAY", "SPEECH", "VOICE", "REDACT")
        assert set(skipped).isdisjoint(calls)
        assert [result.ran[node] for node in skipped] == [RunState.SKIPPED] * len(skipped)

    def test_verdict_still_runs_and_flags_the_file_with_a_reason(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The file reaches VERDICT rather than reading as a silent, evidence-free pass."""
        graph(raising="PREPROCESS")
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert result.ran["VERDICT"] is RunState.COMPLETED
        assert result.file_verdict is not None
        assert result.file_verdict.triage is Triage.FLAG
        assert any("preprocess failed" in reason.why for reason in result.file_verdict.reasons)

    def test_the_store_is_still_persisted_and_report_still_runs(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """REPORT runs on every outcome, including one this bare — the near-empty-store case."""
        graph(raising="PREPROCESS")
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert result.store_path.is_file()
        assert result.ran["REPORT"] is RunState.COMPLETED
        assert sorted(result.summary) == ["json", "summary"]


class TestAdmitFailShortCircuits:
    """ADMIT is the one gate: nothing downstream runs, and VERDICT still folds."""

    def test_only_admit_and_verdict_run(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A file that will not decode is not measured, and the skipped nodes say so."""
        calls = graph(admit_outcome=Outcome.FAIL)
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert tuple(calls) == ("ADMIT", "VERDICT")
        assert result.ran["ADMIT"] is RunState.COMPLETED
        assert result.ran["VERDICT"] is RunState.COMPLETED
        skipped = [node for node in GRAPH[1:-1]]
        assert [result.ran[node] for node in skipped] == [RunState.SKIPPED] * len(skipped)

    def test_the_file_verdict_discards_and_nothing_is_released(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """An unmeasurable recording discards on triage and is never assessed for release."""
        graph(admit_outcome=Outcome.FAIL)
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert result.file_verdict is not None
        assert result.file_verdict.triage is Triage.DISCARD
        assert result.file_verdict.release is Release.NOT_ASSESSED
        assert result.released == {}
        assert result.store_path.is_file()


class TestFreshArtifactsDir:
    """A re-run never publishes into a directory holding an earlier pair."""

    def test_a_rerun_gets_a_fresh_empty_artifacts_dir(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Two runs of one source into one out_dir hold their released pairs apart."""
        source, out_dir = tmp_path / "recording.wav", tmp_path / "out"
        graph(released={"audio": "audio.wav", "transcript": "transcript.txt"})
        first = run_triage(source, out_dir, config)
        second = run_triage(source, out_dir, config)
        assert first.artifacts_dir != second.artifacts_dir
        assert first.run_dir != second.run_dir
        assert sorted(p.name for p in first.artifacts_dir.iterdir()) == ["audio.wav", "transcript.txt"]
        assert sorted(p.name for p in second.artifacts_dir.iterdir()) == ["audio.wav", "transcript.txt"]
        assert not second.artifacts_dir.is_relative_to(first.artifacts_dir)

    def test_a_rerun_that_releases_nothing_leaves_no_earlier_pair_behind(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The second run's release directory is empty when its REDACT released nothing."""
        source, out_dir = tmp_path / "recording.wav", tmp_path / "out"
        graph(released={"audio": "audio.wav"})
        first = run_triage(source, out_dir, config)
        graph()
        second = run_triage(source, out_dir, config)
        assert list(second.artifacts_dir.iterdir()) == []
        assert (first.artifacts_dir / "audio.wav").is_file()

    def test_the_runner_log_names_the_paths_and_the_states(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The runner's own log carries what the store cannot: the per-node run states."""
        graph(raising="VOICE")
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        log = json.loads((result.run_dir / "run.json").read_text())
        assert log["run_dir"] == str(result.run_dir)
        assert log["artifacts_dir"] == str(result.artifacts_dir)
        assert log["ran"]["VOICE"] == RunState.ERRORED.value
        assert "RuntimeError" in log["errors"]["VOICE"]


def _cli() -> types.ModuleType:
    """Import ``scripts/triage_audio.py`` as a module so its helpers can be driven directly."""
    spec = importlib.util.spec_from_file_location("triage_audio_under_test", CLI)
    assert spec is not None and spec.loader is not None, f"could not load {CLI}"
    module = importlib.util.module_from_spec(spec)
    sys.modules["triage_audio_under_test"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_result(tmp_path: Path) -> Callable[..., Any]:
    """A ``TriageRunResult`` standing in for a graph run, so the CLI can be driven without one.

    It defaults to the clean run — VERDICT concluded and nothing errored — so a test that cares
    about a failure states it.
    """

    def _make(file_verdict: Any = _MISSING, nodes: Any = None) -> Any:  # noqa: ANN401
        run_dir = tmp_path / "run"
        run_dir.mkdir(exist_ok=True)
        return run_module.TriageRunResult(
            file_verdict=FileVerdict(triage=Triage.PASS, release=Release.NOT_ASSESSED)
            if file_verdict is _MISSING
            else file_verdict,
            nodes=nodes or {},
            run_dir=run_dir,
            artifacts_dir=tmp_path / "released",
            store_path=run_dir / "store.jsonl",
            summary_dir=tmp_path / "summary",
        )

    return _make


class TestCli:
    """The CLI is two arguments over one versioned config."""

    def test_it_parses_its_arguments_and_calls_the_runner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config: TriageConfig
    ) -> None:
        """Source, out dir, config override and hint all reach ``run_triage``."""
        cli = _cli()
        source = tmp_path / "recording.wav"
        source.write_bytes(b"")
        override = tmp_path / "override.yaml"
        override.write_text("redaction:\n  padding_ms: 50\n")
        hint_file = tmp_path / "hint.yaml"
        hint_file.write_text("may_contain: [cough]\nenvironment: clinic\n")
        seen: dict[str, Any] = {}

        def _fake_run(source: Path, out_dir: Path, config: TriageConfig, **kwargs: Any) -> Any:  # noqa: ANN401
            seen.update(source=source, out_dir=out_dir, config=config, **kwargs)
            run_dir = tmp_path / "run"
            run_dir.mkdir(exist_ok=True)
            return run_module.TriageRunResult(
                file_verdict=FileVerdict(triage=Triage.PASS, release=Release.NOT_ASSESSED),
                nodes={},
                run_dir=run_dir,
                artifacts_dir=tmp_path / "released",
                store_path=run_dir / "store.jsonl",
                summary_dir=tmp_path / "summary",
            )

        monkeypatch.setattr(cli, "run_triage", _fake_run)
        code = cli.main(
            [str(source), "--out", str(tmp_path / "out"), "--config", str(override), "--hint", str(hint_file)]
        )
        assert code == 0
        assert seen["source"] == source
        assert seen["out_dir"] == tmp_path / "out"
        assert seen["config"].config_hash != config.config_hash
        assert seen["hint"] == AudioHints(may_contain=["cough"], environment="clinic")

    def test_it_defaults_the_out_dir_and_the_hint(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With only a recording, the packaged config runs and no hint is declared."""
        cli = _cli()
        source = tmp_path / "recording.wav"
        source.write_bytes(b"")
        seen: dict[str, Any] = {}

        def _fake_run(source: Path, out_dir: Path, config: TriageConfig, **kwargs: Any) -> Any:  # noqa: ANN401
            seen.update(source=source, out_dir=out_dir, config=config, **kwargs)
            run_dir = tmp_path / "run"
            run_dir.mkdir(exist_ok=True)
            return run_module.TriageRunResult(
                file_verdict=FileVerdict(triage=Triage.PASS, release=Release.NOT_ASSESSED),
                nodes={},
                run_dir=run_dir,
                artifacts_dir=tmp_path / "released",
                store_path=run_dir / "store.jsonl",
                summary_dir=tmp_path / "summary",
            )

        monkeypatch.setattr(cli, "run_triage", _fake_run)
        assert cli.main([str(source)]) == 0
        assert seen["hint"] is None
        assert seen["out_dir"] == cli.DEFAULT_OUT_DIR
        assert seen["config"].config_hash == load_triage_config().config_hash

    def test_a_missing_recording_is_refused_before_anything_runs(self, tmp_path: Path) -> None:
        """A path that does not exist is a caller error, not a recording finding."""
        cli = _cli()
        assert cli.main([str(tmp_path / "absent.wav")]) == 2

    def test_it_has_no_per_knob_flags(self) -> None:
        """Every value but the four paths lives in the config file."""
        cli = _cli()
        parser = cli.build_parser()
        options = {action.dest for action in parser._actions if action.dest != "help"}
        assert options == {"audio", "out", "config", "hint", "enrollment"}

    def test_a_hints_table_is_refused_rather_than_read_as_an_empty_hint(self, tmp_path: Path) -> None:
        """AudioHints ignores unknown keys, so a filename-keyed hints table validated to may_contain=[].

        Every absence path then concluded fail with nothing anywhere saying the hint had been
        dropped — the worst of the three outcomes, because a dropped hint looks like a finding.
        """
        cli = _cli()
        table = tmp_path / "hints.json"
        table.write_text(json.dumps({"recording-001.wav": {"may_contain": ["cough"]}}))
        with pytest.raises(ValueError, match="recording-001.wav"):
            cli.load_hint(table)

    def test_the_refusal_says_a_table_needs_per_file_extraction(self, tmp_path: Path) -> None:
        """The caller's next move belongs in the message: the file they have is usually the right file."""
        cli = _cli()
        table = tmp_path / "hints.json"
        table.write_text(json.dumps({"recording-001.wav": {"may_contain": ["cough"]}}))
        with pytest.raises(ValueError, match="per-file"):
            cli.load_hint(table)

    def test_a_hint_naming_only_real_fields_still_loads(self, tmp_path: Path) -> None:
        """The control: refusing unknown keys must not refuse the hints the nodes actually read."""
        cli = _cli()
        hint_file = tmp_path / "hint.yaml"
        hint_file.write_text("may_contain: [cough]\nenvironment: clinic\n")
        assert cli.load_hint(hint_file) == AudioHints(may_contain=["cough"], environment="clinic")


ENROLLMENT_YAML = f"""\
subject_id: sub-01
task: sustained-vowel
vector: [1.0, 0.0]
provenance:
  model_id: speechbrain/spkrec-ecapa-voxceleb
  model_commit_sha: {"a" * 40}
  source_files: [a.wav, b.wav]
  n_windows_used: 12
  n_windows_dropped: 1
"""


class TestTheEnrollmentDriver:
    """SPEECH identifies the target by enrollment, so the CLI must be able to supply one."""

    def test_an_enrollment_file_reaches_the_runner(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_result: Callable[..., Any]
    ) -> None:
        """Without this flag the enrollment path was unreachable from the shipped entry point."""
        cli = _cli()
        source = tmp_path / "recording.wav"
        source.write_bytes(b"")
        enrollment_file = tmp_path / "enrollment.yaml"
        enrollment_file.write_text(ENROLLMENT_YAML)
        seen: dict[str, Any] = {}

        def _fake_run(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            seen.update(kwargs)
            return fake_result()

        monkeypatch.setattr(cli, "run_triage", _fake_run)
        assert cli.main([str(source), "--enrollment", str(enrollment_file)]) == 0
        enrollment = seen["enrollment"]
        assert enrollment.subject_id == "sub-01"
        assert enrollment.vector == [1.0, 0.0]
        assert enrollment.provenance.model_id == "speechbrain/spkrec-ecapa-voxceleb"
        assert enrollment.provenance.model_commit_sha == "a" * 40
        assert enrollment.sources == ["a.wav", "b.wav"]

    def test_no_enrollment_flag_hands_the_runner_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_result: Callable[..., Any]
    ) -> None:
        """The control: an unenrolled run is the ordinary one, and must stay a run."""
        cli = _cli()
        source = tmp_path / "recording.wav"
        source.write_bytes(b"")
        seen: dict[str, Any] = {}

        def _fake_run(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            seen.update(kwargs)
            return fake_result()

        monkeypatch.setattr(cli, "run_triage", _fake_run)
        assert cli.main([str(source)]) == 0
        assert seen["enrollment"] is None

    def test_json_is_read_the_same_way(self, tmp_path: Path) -> None:
        """An estimator writing JSON must not need a converter to feed this flag."""
        cli = _cli()
        path = tmp_path / "enrollment.json"
        path.write_text(
            json.dumps(
                {
                    "subject_id": "sub-01",
                    "vector": [1.0, 0.0],
                    "provenance": {"model_id": "m", "model_commit_sha": "b" * 40, "source_files": ["a.wav"]},
                }
            )
        )
        assert cli.load_enrollment(path).subject_id == "sub-01"

    def test_an_unknown_top_level_key_is_refused_loudly(self, tmp_path: Path) -> None:
        """Pydantic ignores an extra key, so a misspelled field would enrol a different vector silently."""
        cli = _cli()
        path = tmp_path / "enrollment.yaml"
        path.write_text(ENROLLMENT_YAML.replace("subject_id:", "subjectId:"))
        with pytest.raises(ValueError, match="subjectId"):
            cli.load_enrollment(path)

    def test_an_unknown_provenance_key_is_refused_loudly(self, tmp_path: Path) -> None:
        """The commit lives in the nested block, and an ignored key there is an unpinned enrollment."""
        cli = _cli()
        path = tmp_path / "enrollment.yaml"
        path.write_text(ENROLLMENT_YAML.replace("  model_commit_sha:", "  revision:"))
        with pytest.raises(ValueError, match="revision"):
            cli.load_enrollment(path)

    def test_a_table_keyed_by_subject_is_refused_rather_than_read_as_one_enrollment(self, tmp_path: Path) -> None:
        """A per-subject table is the shape an estimator naturally writes; it is not this file."""
        cli = _cli()
        path = tmp_path / "enrollments.json"
        path.write_text(json.dumps({"sub-01": {"subject_id": "sub-01", "vector": [1.0]}}))
        with pytest.raises(ValueError, match="sub-01"):
            cli.load_enrollment(path)

    def test_a_file_that_is_not_a_mapping_is_refused(self, tmp_path: Path) -> None:
        """A list of enrollments is the other shape, and is refused with the same message."""
        cli = _cli()
        path = tmp_path / "enrollments.json"
        path.write_text(json.dumps([{"subject_id": "sub-01"}]))
        with pytest.raises(ValueError, match="mapping"):
            cli.load_enrollment(path)

    def test_a_malformed_enrollment_is_an_invocation_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A bad enrollment is the caller's error, so nothing is measured and the exit code says so."""
        cli = _cli()
        source = tmp_path / "recording.wav"
        source.write_bytes(b"")
        path = tmp_path / "enrollment.yaml"
        path.write_text("subject_id: sub-01\n")

        def _never(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            raise AssertionError("the graph must not run against an enrollment that did not load")

        monkeypatch.setattr(cli, "run_triage", _never)
        assert cli.main([str(source), "--enrollment", str(path)]) == 2


class TestTheExitCodeSaysWhetherTheGraphRanClean:
    """0 was returned unconditionally, so a scheduler could not tell a clean run from a broken one."""

    def _drive(
        self,
        cli: types.ModuleType,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        result: Any,  # noqa: ANN401
    ) -> int:
        """Run the CLI over a canned result."""
        source = tmp_path / "recording.wav"
        source.write_bytes(b"")
        monkeypatch.setattr(cli, "run_triage", lambda *a, **k: result)
        return int(cli.main([str(source)]))

    def test_a_clean_run_is_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_result: Callable[..., Any]
    ) -> None:
        """VERDICT concluded and every node completed."""
        nodes = {node: run_module.NodeOutcome(node=node, state=RunState.COMPLETED) for node in GRAPH}
        assert self._drive(_cli(), monkeypatch, tmp_path, fake_result(nodes=nodes)) == 0

    def test_a_skipped_node_is_still_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_result: Callable[..., Any]
    ) -> None:
        """A branch ROUTING declined is the graph working, not the graph failing."""
        nodes = {node: run_module.NodeOutcome(node=node, state=RunState.COMPLETED) for node in GRAPH}
        nodes["VOICE"] = run_module.NodeOutcome(node="VOICE", state=RunState.SKIPPED)
        assert self._drive(_cli(), monkeypatch, tmp_path, fake_result(nodes=nodes)) == 0

    def test_an_errored_node_is_one_even_though_the_verdict_was_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_result: Callable[..., Any]
    ) -> None:
        """The run produced a verdict over a graph that did not fully run; the caller must be told."""
        nodes = {node: run_module.NodeOutcome(node=node, state=RunState.COMPLETED) for node in GRAPH}
        nodes["VOICE"] = run_module.NodeOutcome(
            node="VOICE", state=RunState.ERRORED, error="RuntimeError: VOICE could not run"
        )
        assert self._drive(_cli(), monkeypatch, tmp_path, fake_result(nodes=nodes)) == 1

    def test_verdict_itself_not_running_is_one(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_result: Callable[..., Any]
    ) -> None:
        """There is no file verdict at all, which is the one outcome that cannot read as success."""
        assert self._drive(_cli(), monkeypatch, tmp_path, fake_result(file_verdict=None)) == 1

    def test_a_discard_verdict_is_still_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_result: Callable[..., Any]
    ) -> None:
        """The exit code reports whether the graph ran, never what it concluded about the recording."""
        nodes = {node: run_module.NodeOutcome(node=node, state=RunState.COMPLETED) for node in GRAPH}
        discarded = FileVerdict(triage=Triage.DISCARD, release=Release.WITHHELD, discard_ground="acoustically_empty")
        assert self._drive(_cli(), monkeypatch, tmp_path, fake_result(file_verdict=discarded, nodes=nodes)) == 0

    def test_the_errored_node_is_named_on_stderr(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        fake_result: Callable[..., Any],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """An exit code nobody can act on is half a signal; the node and its error go with it."""
        nodes = {"VOICE": run_module.NodeOutcome(node="VOICE", state=RunState.ERRORED, error="RuntimeError: boom")}
        self._drive(_cli(), monkeypatch, tmp_path, fake_result(nodes=nodes))
        assert "VOICE" in capsys.readouterr().err
