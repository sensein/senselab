"""The ruleset run inside the graph reads the same evidence the analysis reads outside it.

Every test here builds one live :class:`~senselab.utils.prov_store.ProvStore`, evaluates the ruleset
over it both ways — through the in-pipeline reader and through the analysis reader over the same
store written to disk — and asserts the two agree. That equivalence is the whole point of the stage;
a narrowing of either path that broke it would be two definitions of one gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from senselab.audio.tasks.features_extraction.ppg import PHONEME_LABELS
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.live_evidence import (
    EVIDENCE_PREFIX,
    UNDECLARED,
    evaluate_live_routes,
    failed_route_attributes,
    read_live_features,
    recording_stem,
    required_sources,
    route_attributes,
)
from senselab.audio.workflows.triage.routing_analysis.features import (
    SILENT_PHONEME,
    extract_features,
    onomatopoeic_vocabulary,
    span_label_memberships,
)
from senselab.audio.workflows.triage.routing_analysis.ruleset import (
    GateOutcome,
    RouteEvaluation,
    RouteState,
    Ruleset,
    evaluate_routes,
    load_ruleset,
)
from senselab.utils.prov_store import ProvStore

STEM = "sub-01_ses-01_task-free-speech"
"""The stem the synthetic recordings carry; its ``task-`` id is a declaration and is never read."""


@pytest.fixture(scope="module")
def config() -> TriageConfig:
    """The packaged triage configuration.

    Returns:
        The configuration as ``data/config/default.yaml`` declares it.
    """
    return load_triage_config()


@pytest.fixture(scope="module")
def ruleset(config: TriageConfig) -> Ruleset:
    """The packaged ruleset.

    Args:
        config: The packaged configuration.

    Returns:
        The ruleset ``taxonomy.ruleset`` declares.
    """
    return load_ruleset(config)


def _run_dir(tmp_path: Path) -> Path:
    """A run directory with the sidecar subdirectories the graph creates.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The run directory.
    """
    run_dir = tmp_path / "run"
    (run_dir / "derivatives").mkdir(parents=True, exist_ok=True)
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    return run_dir


def _base_store(run_dir: Path) -> ProvStore:
    """A live store standing for a recording with speech, one airway token and a repeated word.

    Every family of evidence the packaged gates read is represented, either present or deliberately
    absent: three gates fire, five stay silent and three cannot be read at all.

    Args:
        run_dir: The run directory the store's stem path is placed beside.

    Returns:
        The store.
    """
    store = ProvStore(run_id="live-evidence-test")
    activity = store.activity(node="PREPROCESS", step=None, parameters={})
    store.entity(
        prov_type="stream",
        extent=(0.0, 4.0),
        attributes={"name": "recording", "path": str(run_dir.parent / f"{STEM}.wav")},
    )
    for index, (text, outcome, bracketed) in enumerate(
        [
            ("buttercup", "agreement", False),
            ("buttercup", "agreement", False),
            ("buttercup", "variant", False),
            ("[BREATH]", "agreement", True),
        ]
    ):
        store.entity(
            prov_type="word",
            extent=(0.1 * index, 0.1 * index + 0.05),
            attributes={"text": text, "outcome": outcome, "bracketed": bracketed, "index": index},
        )
    withdrawn = store.entity(
        prov_type="word",
        extent=(3.0, 3.2),
        attributes={"text": "[Throat Clearing]", "outcome": "agreement", "bracketed": True, "index": 4},
    )
    store.was_invalidated_by(withdrawn, activity)
    store.entity(
        prov_type="measurement",
        extent=None,
        attributes={"name": "consensus_transcript", "signal": "plain", "text": "buttercup buttercup buttercup"},
    )
    store.entity(
        prov_type="measurement",
        extent=None,
        attributes={"name": "residual", "signal": "residual", "energy_fraction": 0.01, "speech_present": True},
    )
    store.entity(
        prov_type="measurement",
        extent=None,
        attributes={
            "name": "yamnet_label_summary",
            "signal": "plain",
            "classifier": "yamnet",
            "labels": {"Speech": {"peak": 0.97}, "Chant": {"peak": 0.01}},
        },
    )
    store.entity(prov_type="span", extent=(0.0, 1.5), attributes={"measure": "amplitude"})
    store.entity(prov_type="span", extent=(2.0, 2.5), attributes={"measure": "continuity"})
    return store


def _empty_store(run_dir: Path) -> ProvStore:
    """A live store standing for a recording nothing routes and the bypass calls empty.

    Args:
        run_dir: The run directory the store's stem path is placed beside.

    Returns:
        The store.
    """
    store = ProvStore(run_id="live-evidence-empty")
    store.entity(
        prov_type="stream",
        extent=(0.0, 6.0),
        attributes={"name": "recording", "path": str(run_dir.parent / f"{STEM}.wav")},
    )
    store.entity(
        prov_type="measurement",
        extent=None,
        attributes={"name": "consensus_transcript", "signal": "plain", "text": ""},
    )
    store.entity(
        prov_type="measurement",
        extent=None,
        attributes={"name": "residual", "signal": "residual", "energy_fraction": 0.0, "speech_present": False},
    )
    for stream in ("enhanced", "residual"):
        store.entity(
            prov_type="measurement",
            extent=None,
            attributes={
                "name": f"{stream}_yamnet_summary_all",
                "signal": stream,
                "classifier": "yamnet",
                "labels": {"Speech": {"peak": 0.01}, "Cough": {"peak": 0.02}},
            },
        )
    return store


def _write_posteriorgram(run_dir: Path, store: ProvStore, silent_frames: int, voiced_frames: int) -> None:
    """Write a posteriorgram sidecar and the measurement naming it, by relative path.

    Args:
        run_dir: The run directory the sidecar goes under.
        store: The store the measurement is written to.
        silent_frames: How many frames take the inventory's silence label.
        voiced_frames: How many frames take a single other phoneme.
    """
    silent = PHONEME_LABELS.index(SILENT_PHONEME)
    other = next(index for index, label in enumerate(PHONEME_LABELS) if label != SILENT_PHONEME)
    frames = silent_frames + voiced_frames
    posteriorgram = np.zeros((frames, len(PHONEME_LABELS)), dtype=np.float16)
    posteriorgram[:silent_frames, silent] = 1.0
    posteriorgram[silent_frames:, other] = 1.0
    relative = "derivatives/ppg_posteriorgram.npz"
    np.savez(
        run_dir / relative,
        posteriorgram=posteriorgram,
        phonemes=np.asarray(PHONEME_LABELS, dtype=np.str_),
        seconds_per_frame=np.float64(0.01),
        duration_s=np.float64(frames * 0.01),
        sampling_rate=np.int64(16000),
    )
    store.entity(
        prov_type="measurement",
        extent=(0.0, frames * 0.01),
        attributes={
            "name": "ppg_posteriorgram",
            "signal": "enhanced",
            "path": relative,
            "frames": frames,
            "n_phonemes": len(PHONEME_LABELS),
            "seconds_per_frame": 0.01,
        },
    )


def _from_disk(store: ProvStore, config: TriageConfig, run_dir: Path) -> RouteEvaluation:
    """The analysis path: the store written out, streamed back, and the ruleset evaluated over it.

    Args:
        store: The store to write.
        config: The triage configuration.
        run_dir: The run directory the store and its sidecars sit in.

    Returns:
        The evaluation the analysis module would make of a finished run.
    """
    path = run_dir / "store.jsonl"
    store.write_jsonl(path)
    features = extract_features(
        path,
        stem=recording_stem(store),
        run_root=str(run_dir.parent),
        task_id=UNDECLARED,
        family=UNDECLARED,
        memberships=span_label_memberships(config),
        onomatopoeic=onomatopoeic_vocabulary(config),
    )
    return evaluate_routes(features, load_ruleset(config))


def _assert_same(live: RouteEvaluation, on_disk: RouteEvaluation) -> None:
    """Assert two evaluations of one store agree on everything content decides.

    Args:
        live: The in-pipeline evaluation.
        on_disk: The analysis evaluation.
    """
    assert live.gate_outcomes == on_disk.gate_outcomes
    assert live.routed == on_disk.routed
    assert live.state is on_disk.state
    assert live.unavailable == on_disk.unavailable
    assert live.flags == on_disk.flags


class TestTheTwoPathsAgree:
    """One store, two readers, one set of gate outcomes."""

    def test_every_gate_outcome_matches(self, config: TriageConfig, tmp_path: Path) -> None:
        """The equivalence this stage exists to pin, over a store exercising all three outcomes."""
        run_dir = _run_dir(tmp_path)
        store = _base_store(run_dir)
        _assert_same(evaluate_live_routes(store, config, run_dir=run_dir), _from_disk(store, config, run_dir))

    def test_the_store_exercises_all_three_outcomes(self, config: TriageConfig, tmp_path: Path) -> None:
        """An all-unavailable store would make the equality above pass for the wrong reason."""
        run_dir = _run_dir(tmp_path)
        outcomes = set(evaluate_live_routes(_base_store(run_dir), config, run_dir=run_dir).gate_outcomes.values())
        assert outcomes == {GateOutcome.FIRED, GateOutcome.SILENT, GateOutcome.UNAVAILABLE}

    def test_a_sidecar_measurement_is_read_the_same_way(self, config: TriageConfig, tmp_path: Path) -> None:
        """A measurement naming a sidecar by relative path resolves under both readers."""
        run_dir = _run_dir(tmp_path)
        store = _base_store(run_dir)
        _write_posteriorgram(run_dir, store, silent_frames=90, voiced_frames=10)
        live = evaluate_live_routes(store, config, run_dir=run_dir)
        _assert_same(live, _from_disk(store, config, run_dir))
        assert live.gate_outcomes["airway.ppg_silent_fraction"] is GateOutcome.FIRED

    def test_an_unrouted_recording_agrees_on_emptiness(self, config: TriageConfig, tmp_path: Path) -> None:
        """The bypass is consulted only where no gate fired, and reads the same both ways."""
        run_dir = _run_dir(tmp_path)
        store = _empty_store(run_dir)
        live = evaluate_live_routes(store, config, run_dir=run_dir)
        _assert_same(live, _from_disk(store, config, run_dir))
        assert live.routed == ()
        assert live.state is RouteState.EMPTY

    def test_an_unreadable_bypass_leaves_the_recording_unexplained(self, config: TriageConfig, tmp_path: Path) -> None:
        """A stream summary absent from the store is not a stream that scored zero."""
        run_dir = _run_dir(tmp_path)
        store = ProvStore(run_id="live-evidence-bare")
        store.entity(prov_type="stream", extent=(0.0, 1.0), attributes={"name": "recording", "path": f"{STEM}.wav"})
        live = evaluate_live_routes(store, config, run_dir=run_dir)
        _assert_same(live, _from_disk(store, config, run_dir))
        assert live.state is RouteState.UNEXPLAINED


class TestWhatTheReaderReads:
    """The live reader's own contract, beside the equivalence."""

    def test_an_invalidated_word_is_dropped_by_both_paths(self, config: TriageConfig, tmp_path: Path) -> None:
        """A withdrawn ``[Throat Clearing]`` must not be counted as an airway bracket token."""
        run_dir = _run_dir(tmp_path)
        features = read_live_features(
            _base_store(run_dir),
            run_dir=run_dir,
            memberships=span_label_memberships(config),
            onomatopoeic=onomatopoeic_vocabulary(config),
            stem=STEM,
        )
        assert features.bracketed_types == {"breath": 1}
        assert features.words["lexical"] == 3

    def test_the_evaluation_carries_no_declaration(self, config: TriageConfig, tmp_path: Path) -> None:
        """TAXONOMY reads no hint, and a ``task-`` id is a hint; nothing is declared here."""
        run_dir = _run_dir(tmp_path)
        evaluation = evaluate_live_routes(_base_store(run_dir), config, run_dir=run_dir)
        assert evaluation.family == UNDECLARED
        assert evaluation.declared == ()
        assert evaluation.stem == STEM

    def test_the_serialisation_is_removed(self, config: TriageConfig, tmp_path: Path) -> None:
        """The reduction writes into the run directory and leaves nothing behind."""
        run_dir = _run_dir(tmp_path)
        evaluate_live_routes(_base_store(run_dir), config, run_dir=run_dir)
        assert list(run_dir.glob(f"{EVIDENCE_PREFIX}*")) == []

    def test_the_stem_comes_off_the_recording_stream(self, tmp_path: Path) -> None:
        """An identifier, read off ADMIT's own entity rather than plumbed through the node shape."""
        run_dir = _run_dir(tmp_path)
        assert recording_stem(_base_store(run_dir)) == STEM
        assert recording_stem(ProvStore(run_id="bare")) == ""

    def test_the_required_sources_come_off_the_configured_gates(self, ruleset: Ruleset) -> None:
        """What a configuration consumes is knowable from the gates, not from the feature surface."""
        sources = required_sources(ruleset)
        assert sources == tuple(sorted(set(sources)))
        assert "stream_peak_max" in sources
        assert {"words", "ppg", "transcript_repeat", "bracketed_set"} <= set(sources)


class TestTheRecordedAttributes:
    """What TAXONOMY writes, in the one shape a reader handles."""

    def test_the_two_shapes_carry_the_same_keys(self, config: TriageConfig, ruleset: Ruleset, tmp_path: Path) -> None:
        """A failed evaluation is discriminated by ``state`` and ``error``, never by a missing key."""
        run_dir = _run_dir(tmp_path)
        recorded = route_attributes(evaluate_live_routes(_base_store(run_dir), config, run_dir=run_dir), ruleset)
        failed = failed_route_attributes("ValueError: no ruleset")
        assert recorded.keys() == failed.keys()
        assert recorded["error"] is None
        assert failed["state"] is None

    def test_nothing_recorded_here_is_authoritative(
        self, config: TriageConfig, ruleset: Ruleset, tmp_path: Path
    ) -> None:
        """The ruleset selects nothing while ``kind_state`` still decides what runs."""
        run_dir = _run_dir(tmp_path)
        recorded = route_attributes(evaluate_live_routes(_base_store(run_dir), config, run_dir=run_dir), ruleset)
        assert recorded["authoritative"] is False
        assert failed_route_attributes("boom")["authoritative"] is False

    def test_the_routed_branches_survive_serialisation(
        self, config: TriageConfig, ruleset: Ruleset, tmp_path: Path
    ) -> None:
        """Every recorded field is a JSON scalar or container, so the store can carry it."""
        run_dir = _run_dir(tmp_path)
        recorded = route_attributes(evaluate_live_routes(_base_store(run_dir), config, run_dir=run_dir), ruleset)
        assert set(recorded["routed"]) == {"AIRWAY", "SPEECH", "DDK"}
        assert all(isinstance(value, str) for value in recorded["gate_outcomes"].values())
        assert recorded["state"] == RouteState.ROUTED.value
