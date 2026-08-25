"""TAXONOMY v2: a fold over PREPROCESS's stored derivatives. No models, no hints, no localisation."""

from pathlib import Path
from typing import Callable

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import taxonomy as taxonomy_module
from senselab.audio.workflows.triage.nodes.common import live_entities
from senselab.audio.workflows.triage.nodes.taxonomy import SCREENED_KINDS, taxonomy
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore


def _floors(tmp_path: Path, **extra: str) -> TriageConfig:
    """The packaged config with every TAXONOMY floor supplied and the speech family named.

    Args:
        tmp_path: Where the override YAML is written.
        **extra: Further YAML fragments appended to the override, for a test that needs one.

    Returns:
        The resolved configuration.
    """
    body = (
        "taxonomy:\n"
        "  presence_floor:\n"
        "    speech: {acoustic: 1, lexical: 1}\n"
        "    airway: {health_acoustic: 1, acoustic: 1}\n"
        "  voice_min_duration_s: 1.0\n"
        "  voice_uncertain_duration_s: 0.3\n"
        "  speech_labels: [Speech, Narration, monologue, Conversation]\n"
    )
    path = tmp_path / "floors.yaml"
    path.write_text(body + "".join(extra.values()))
    return load_triage_config(path)


class TestItRunsNoModels:
    """Every classifier call belongs to PREPROCESS; this node folds what is already there."""

    def test_the_module_imports_no_classifier(self) -> None:
        """A model function reachable from this module is a boundary violation, not a convenience."""
        for name in ("classify_audios", "detect_health_acoustic_events", "transcribe_audios"):
            assert not hasattr(taxonomy_module, name)

    def test_it_writes_no_activity_that_names_a_model(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The only activity is the fold."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=["one", "two"])
        taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert [a.step for a in store.activities("TAXONOMY")] == ["fold"]
        assert not [
            agent
            for activity in store.activities("TAXONOMY")
            for agent in store.associated_with(activity.id)
            if store.get_agent(agent).agent_type == "model"
        ]


class TestTheThreeKinds:
    """speech, airway and voice, each with its own rule and its own evidence."""

    def test_speech_needs_both_lines(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Acoustic windows and lexical words both clearing their floors makes speech present."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=["one", "two", "three"])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["speech"] == "present"

    def test_speech_with_windows_but_no_words_is_uncertain(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """One line present and one absent is disagreement, which is uncertain, not present."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=[])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["speech"] == "uncertain"

    def test_speech_with_neither_line_is_absent(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Both lines below their floors is absence."""
        seed_preprocess_store(store, yamnet_labels=[["Music"]], words=[])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["speech"] == "absent"

    def test_a_bracketed_event_carries_no_lexical_evidence(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """PREPROCESS wrote it as an event, so nothing here counts it toward the word floor."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=[], events=["[COUGH]", "[COUGH]", "ahem"])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["speech"] == "uncertain"

    def test_airway_needs_hear_and_audioset(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The health-acoustic and acoustic lines both carrying evidence makes airway present."""
        seed_preprocess_store(store, hear_labels=[["Cough"]], yamnet_labels=[["Cough"]])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["airway"] == "present"

    def test_ast_windows_serve_the_acoustic_line_beside_yamnet(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The acoustic line reads either grid; a label on AST alone is still acoustic evidence."""
        seed_preprocess_store(store, hear_labels=[["Cough"]], ast_labels=[["Cough"]])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["airway"] == "present"

    def test_voice_is_classified_from_phonation_span_duration_alone(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A 2 s sustain makes voice present, whatever else is in the recording."""
        seed_preprocess_store(store, phonation=[(0.0, 2.0, "voiced")])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["voice"] == "present"

    def test_an_unvoiced_sustain_makes_voice_present_too(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A disordered voice sustaining without periodicity is phonation."""
        seed_preprocess_store(store, phonation=[(0.0, 2.0, "unvoiced")])
        result = taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert result.kinds["voice"] == "present"

    def test_a_short_span_is_uncertain_and_a_shorter_one_is_absent(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Between the two floors is uncertain; below the shorter floor there is nothing to be sure of."""
        seed_preprocess_store(store, phonation=[(0.0, 0.5, "voiced")])
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).kinds["voice"] == "uncertain"
        other = ProvStore(run_id="short")
        seed_preprocess_store(other, phonation=[(0.0, 0.1, "voiced")])
        assert taxonomy(other, "plain", _floors(tmp_path), run_dir=tmp_path).kinds["voice"] == "absent"

    def test_no_phonation_span_is_absent(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The pass ran and found nothing, which is absence."""
        seed_preprocess_store(store, phonation=[])
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).kinds["voice"] == "absent"

    def test_no_phonation_pass_at_all_is_uncertain(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A pass that did not run leaves the line unavailable; that is not evidence of absence."""
        seed_preprocess_store(store, phonation=None)
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).kinds["voice"] == "uncertain"


class TestAMissingDerivativeIsNotAbsence:
    """A line whose derivative never reached the store is unavailable, and unavailable is uncertain."""

    def test_a_null_threshold_leaves_every_kind_uncertain(
        self, store: ProvStore, config: TriageConfig, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The packaged config folds no windows, so nothing can be absent."""
        seed_preprocess_store(store, yamnet_labels=None, hear_labels=None, ast_labels=None, phonation=None)
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        assert set(result.kinds.values()) == {"uncertain"}
        assert result.verdict.outcome is Outcome.FLAG

    def test_the_unavailable_line_says_so_on_the_kind_element(
        self, store: ProvStore, config: TriageConfig, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """A reader must see why a kind is uncertain, not only that it is."""
        seed_preprocess_store(store, yamnet_labels=None, hear_labels=None, ast_labels=None, phonation=None)
        taxonomy(store, "plain", config, run_dir=tmp_path)
        speech = next(e for e in live_entities(store, "kind") if e.attributes["kind"] == "speech")
        assert speech.attributes["lines"]["acoustic"]["state"] == "unavailable"


class TestHintsAreNotAnInput:
    """A classification that reads the declaration cannot disagree with it."""

    def test_a_hint_declaring_speech_does_not_move_the_classification(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """The same store classifies the same way with and without a hint."""
        seed_preprocess_store(store, yamnet_labels=[["Music"]], words=[])
        hinted = taxonomy(store, "plain", _floors(tmp_path), AudioHints(may_contain=["speech"]), run_dir=tmp_path)
        other = ProvStore(run_id="unhinted")
        seed_preprocess_store(other, yamnet_labels=[["Music"]], words=[])
        plain = taxonomy(other, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert hinted.kinds == plain.kinds


class TestTheOutcome:
    """fail on all-absent, flag on any-uncertain, pass otherwise."""

    def test_all_absent_fails(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Nothing is classified present."""
        seed_preprocess_store(
            store, yamnet_labels=[["Music"]], hear_labels=[[]], ast_labels=[[]], words=[], phonation=[]
        )
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).verdict.outcome is Outcome.FAIL

    def test_any_uncertain_flags(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """One kind the lines disagree about is enough."""
        seed_preprocess_store(
            store, yamnet_labels=[["Speech"]], hear_labels=[[]], ast_labels=[[]], words=[], phonation=[]
        )
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).verdict.outcome is Outcome.FLAG

    def test_present_and_absent_together_pass(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """Speech present, airway and voice absent, nothing uncertain."""
        seed_preprocess_store(
            store,
            yamnet_labels=[["Speech"]],
            hear_labels=[[]],
            ast_labels=[["Speech"]],
            words=["one", "two", "three"],
            phonation=[],
        )
        assert taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path).verdict.outcome is Outcome.PASS

    def test_exactly_three_kind_elements_and_no_residual(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """voice_no_words and not_screened are gone; nothing is a kind by virtue of the others."""
        seed_preprocess_store(store, yamnet_labels=[["Speech"]], words=["one", "two", "three"])
        taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        kinds = {e.attributes["kind"] for e in live_entities(store, "kind")}
        assert kinds == set(SCREENED_KINDS) == {"airway", "speech", "voice"}
        assert not [e for e in live_entities(store, "kind") if e.attributes["state"] == "not_screened"]

    def test_it_localises_nothing(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None], tmp_path: Path
    ) -> None:
        """No span, no interval, no extent-bearing element is authored by this node."""
        seed_preprocess_store(
            store, yamnet_labels=[["Speech"]], words=["one", "two", "three"], phonation=[(0.0, 2.0, "voiced")]
        )
        before = {e.id for e in live_entities(store, "span")}
        taxonomy(store, "plain", _floors(tmp_path), run_dir=tmp_path)
        assert {e.id for e in live_entities(store, "span")} == before
        assert not live_entities(store, "interval")
