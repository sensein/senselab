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
from senselab.audio.workflows.triage.nodes.airway import AirwayResult
from senselab.audio.workflows.triage.nodes.common import NodeResult, software_agent, write_verdict
from senselab.audio.workflows.triage.nodes.preprocess import PreprocessResult
from senselab.audio.workflows.triage.nodes.redact import RedactResult
from senselab.audio.workflows.triage.nodes.taxonomy import TaxonomyResult
from senselab.audio.workflows.triage.run import run_triage
from senselab.audio.workflows.triage.vocabulary import NodeVerdict, Outcome, Release, RunState
from senselab.utils.prov_store import ProvStore

REPO_ROOT = Path(__file__).resolve().parents[5]
CLI = REPO_ROOT / "scripts" / "triage_audio.py"

GRAPH = ("ADMIT", "PREPROCESS", "TAXONOMY", "AIRWAY", "SPEECH", "VOICE", "REDACT", "VERDICT")


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
) -> dict[str, Callable[..., Any]]:
    """Fake node functions, one per graph node, recording their calls into ``calls``.

    Args:
        calls: The list every fake appends its node name to, in call order.
        admit_outcome: What the fake ADMIT concludes; ``FAIL`` returns no audio.
        raising: The node whose fake raises ``RuntimeError`` instead of concluding.
        released: What the fake REDACT reports as its released pair.

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
        entity_id, verdict = _conclude(store, "TAXONOMY", Outcome.PASS, None)
        return TaxonomyResult(verdict=verdict, view=(entity_id,), verdict_entity_id=entity_id, kinds={})

    def _airway(
        store: ProvStore, source: str, config: TriageConfig, hint: AudioHints | None = None, *, run_dir: Path
    ) -> AirwayResult:
        _record("AIRWAY")
        entity_id, verdict = _conclude(store, "AIRWAY", Outcome.PASS, "airway")
        return AirwayResult(verdict=verdict, view=(entity_id,), verdict_entity_id=entity_id, figure_path=None)

    def _speech(
        store: ProvStore, source: str, config: TriageConfig, hint: AudioHints | None = None, *, run_dir: Path
    ) -> NodeResult:
        _record("SPEECH")
        entity_id, verdict = _conclude(store, "SPEECH", Outcome.PASS, "speech")
        return NodeResult(verdict=verdict, view=(entity_id,), verdict_entity_id=entity_id)

    def _voice(
        store: ProvStore, source: str, config: TriageConfig, hint: AudioHints | None = None, *, run_dir: Path
    ) -> NodeResult:
        _record("VOICE")
        entity_id, verdict = _conclude(store, "VOICE", Outcome.PASS, "voice_no_words")
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

    def test_calls_all_eight_nodes_in_graph_order(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """The runner drives the DAG in the graph's declared order, VERDICT last."""
        calls = graph()
        run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert tuple(calls) == GRAPH

    def test_returns_the_file_verdict_with_every_node_completed(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """A graph in which nothing raised reports ``COMPLETED`` for all eight nodes."""
        graph()
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert result.file_verdict is not None
        assert result.file_verdict.triage is Outcome.PASS
        assert result.ran == dict.fromkeys(GRAPH, RunState.COMPLETED)

    def test_the_layout_is_written_and_the_release_dir_is_disjoint(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """run_dir carries the store and its three sidecar trees; artifacts_dir contains neither."""
        graph(released={"audio": "audio.wav", "transcript": "transcript.txt"})
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert result.store_path == result.run_dir / "store.jsonl"
        assert result.store_path.is_file()
        for sub in ("streams", "derivatives", "figures"):
            assert (result.run_dir / sub).is_dir()
        run_resolved, release_resolved = result.run_dir.resolve(), result.artifacts_dir.resolve()
        assert not run_resolved.is_relative_to(release_resolved)
        assert not release_resolved.is_relative_to(run_resolved)
        assert sorted(result.released) == ["audio", "transcript"]
        assert all(path.parent == result.artifacts_dir for path in result.released.values())

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
        for name in ("admit", "preprocess", "taxonomy", "airway", "speech", "voice", "redact", "verdict"):
            original = getattr(run_module, name)

            def _spy(*args: Any, _original: Any = original, **kwargs: Any) -> Any:  # noqa: ANN401
                seen.append(args[3] if len(args) > 3 else kwargs.get("hint"))
                return _original(*args, **kwargs)

            monkeypatch.setattr(run_module, name, _spy)
        hint = AudioHints(may_contain=["cough"])
        run_triage(tmp_path / "recording.wav", tmp_path / "out", config, hint=hint)
        assert seen == [hint] * len(GRAPH)


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

    def test_a_raising_preprocess_does_not_stop_the_branches(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """Downstream nodes read the store, so an errored PREPROCESS does not skip them."""
        calls = graph(raising="PREPROCESS")
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert tuple(calls) == GRAPH
        assert result.ran["PREPROCESS"] is RunState.ERRORED
        assert result.ran["TAXONOMY"] is RunState.COMPLETED


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

    def test_the_file_verdict_is_fail_and_nothing_is_released(
        self, graph: Callable[..., list[str]], config: TriageConfig, tmp_path: Path
    ) -> None:
        """An unmeasurable recording fails on triage and is never assessed for release."""
        graph(admit_outcome=Outcome.FAIL)
        result = run_triage(tmp_path / "recording.wav", tmp_path / "out", config)
        assert result.file_verdict is not None
        assert result.file_verdict.triage is Outcome.FAIL
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

        def _fake_run(source: Path, out_dir: Path, config: TriageConfig, hint: AudioHints | None = None) -> Any:  # noqa: ANN401
            seen.update(source=source, out_dir=out_dir, config=config, hint=hint)
            run_dir = tmp_path / "run"
            run_dir.mkdir(exist_ok=True)
            return run_module.TriageRunResult(
                file_verdict=None,
                nodes={},
                run_dir=run_dir,
                artifacts_dir=tmp_path / "released",
                store_path=run_dir / "store.jsonl",
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

        def _fake_run(source: Path, out_dir: Path, config: TriageConfig, hint: AudioHints | None = None) -> Any:  # noqa: ANN401
            seen.update(source=source, out_dir=out_dir, config=config, hint=hint)
            run_dir = tmp_path / "run"
            run_dir.mkdir(exist_ok=True)
            return run_module.TriageRunResult(
                file_verdict=None,
                nodes={},
                run_dir=run_dir,
                artifacts_dir=tmp_path / "released",
                store_path=run_dir / "store.jsonl",
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
        assert options == {"audio", "out", "config", "hint"}
