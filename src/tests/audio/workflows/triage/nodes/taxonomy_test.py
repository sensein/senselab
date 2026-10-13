"""TAXONOMY: a consolidation over PREPROCESS's stored classifier scores.

No models, no hints, no decisions. The per-classifier label summaries and the consensus taxonomy
have their own suites; this one pins what the node as a whole runs, writes and concludes.
"""

from pathlib import Path
from typing import Callable

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import taxonomy as taxonomy_module
from senselab.audio.workflows.triage.nodes.common import find_measurement, live_entities
from senselab.audio.workflows.triage.nodes.taxonomy import taxonomy
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore


def _config(tmp_path: Path) -> TriageConfig:
    """The packaged config, with the speech family named so SPEECH's own key is not null.

    Args:
        tmp_path: Where the override YAML is written.

    Returns:
        The resolved configuration.
    """
    path = tmp_path / "taxonomy.yaml"
    path.write_text("taxonomy:\n  speech_labels: [Speech, Narration, monologue, Conversation]\n")
    return load_triage_config(path)


class TestItRunsNoModels:
    """Every classifier call belongs to PREPROCESS; this node consolidates what is already there."""

    def test_the_module_imports_no_classifier(self) -> None:
        """A model function reachable from this module is a boundary violation, not a convenience."""
        for name in ("classify_audios", "detect_health_acoustic_events", "transcribe_audios"):
            assert not hasattr(taxonomy_module, name)

    def test_it_writes_no_activity_that_names_a_model(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Every step re-reads what PREPROCESS already measured; none of them runs a model."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=["one", "two"])
        taxonomy(store, "plain", _config(tmp_path), run_dir=tmp_path)
        # Named rather than counted: a genuinely new step must be reviewed against the class's
        # invariant, and re-reading a stored score distribution is not running a classifier.
        assert [a.step for a in store.activities("TAXONOMY")] == ["yamnet_label_summary", "conclude"]
        assert not [
            agent
            for activity in store.activities("TAXONOMY")
            for agent in store.associated_with(activity.id)
            if store.get_agent(agent).agent_type == "model"
        ]


class TestItDecidesNothing:
    """The node that measures content must not also select what runs on it."""

    def test_it_writes_no_branch_decision_and_no_ruleset_reading(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Both belong to ``routing``; a measurement node writing either is the boundary breaking."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=["one", "two"])
        taxonomy(store, "plain", _config(tmp_path), run_dir=tmp_path)
        assert not live_entities(store, "branch_decision")
        assert find_measurement(store, "ruleset_routing") is None

    def test_it_localises_nothing(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """No new span, no interval, no other extent-bearing element."""
        seed_preprocess_store(
            store, yamnet_labels=[["Speech"]], words=["one", "two", "three"], phonation=[(0.0, 2.0, "voiced")]
        )
        before = {e.id for e in live_entities(store, "span")}
        taxonomy(store, "plain", _config(tmp_path), run_dir=tmp_path)
        assert {e.id for e in live_entities(store, "span")} == before
        assert not live_entities(store, "interval")


class TestHintsAreNotAnInput:
    """A measurement that reads the declaration cannot disagree with it."""

    def test_a_hint_changes_nothing_this_node_writes(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The same store consolidates the same way whether or not a declaration came with it."""
        seed_preprocess_store(
            store,
            yamnet_labels=[["Speech"]],
            words=["one"],
            spans=[(0.0, 1.0, 9.0)],
            span_hear_labels=[["Cough"]],
        )
        config = _config(tmp_path)
        without = taxonomy(store, "plain", config, run_dir=tmp_path)
        with_hint = taxonomy(store, "plain", config, AudioHints(may_contain=["speech"]), run_dir=tmp_path)
        assert without.classifiers == with_hint.classifiers
        assert without.n_labels == with_hint.n_labels
        assert without.verdict.why == with_hint.verdict.why


class TestTheOutcome:
    """It reports whether the consolidation had anything to work from, and nothing more."""

    def test_a_consolidation_over_real_scores_passes(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A per-span classifier that produced scores is something to consolidate."""
        seed_preprocess_store(
            store,
            yamnet_labels=[["Speech"]],
            words=["one"],
            spans=[(0.0, 1.0, 9.0)],
            span_hear_labels=[["Cough"]],
        )
        result = taxonomy(store, "plain", _config(tmp_path), run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert result.classifiers == ("hear",)
        assert result.n_labels >= 1
        assert "hear" in result.verdict.why

    def test_nothing_to_consolidate_flags(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """No per-span classifier ran, so every downstream label gate will read unavailable."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=["one", "two"])
        result = taxonomy(store, "plain", _config(tmp_path), run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FLAG
        assert result.classifiers == ()
        assert result.n_labels == 0
        assert find_measurement(store, "consensus_taxonomy") is None

    def test_the_verdict_carries_no_kind(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """This node concludes about the recording's evidence, never about a branch's subject."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=["one"])
        result = taxonomy(store, "plain", _config(tmp_path), run_dir=tmp_path)
        assert result.verdict.kind is None
