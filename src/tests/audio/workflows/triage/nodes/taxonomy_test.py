"""TAXONOMY predicts kinds by family agreement. Advisory: it gates nothing and runs on every path.

AST and HeAR are monkeypatched on the node module; YAMNet's windows come from the seeded store.
"""

from pathlib import Path
from typing import Any, Callable

import pytest

from senselab.audio.data_structures import AudioClassificationResult
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import taxonomy as node
from senselab.audio.workflows.triage.nodes.taxonomy import taxonomy
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import Entity, ProvStore


class _FakeModel:
    """A model spec stub carrying path_or_uri and a resolved commit."""

    def __init__(self, path_or_uri: str) -> None:
        """Stub a resolved model."""
        self.path_or_uri = path_or_uri
        self.commit_sha = "b" * 40


def _yamnet_window(start: float, end: float, scores: dict[str, float]) -> dict[str, Any]:
    """One YAMNet-shaped window."""
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    return {
        "start": start,
        "end": end,
        "label_scores": [{label: score} for label, score in ranked],
        "win_length": 0.96,
        "hop_length": 0.48,
    }


@pytest.fixture
def detector_calls() -> dict[str, list]:
    """Captured AST and HeAR call arguments."""
    return {"ast": [], "hear": []}


@pytest.fixture
def mock_detectors(monkeypatch: pytest.MonkeyPatch, detector_calls: dict[str, list]) -> dict[str, Any]:
    """Replace AST and HeAR with controllable fakes; return the mutable score dicts.

    AST's payload mirrors classify_audios' whole-audio return (AudioClassificationResult with
    parallel labels/scores); HeAR's mirrors detect_health_acoustic_events (windowed dicts with
    descending single-key label_scores over the eight labels).
    """
    scores = {"ast": {"Cough": 0.1, "Speech": 0.1}, "hear": {"Cough": 0.1, "Speech": 0.01}}
    monkeypatch.setattr(node, "_ast_model", lambda: _FakeModel(node.AST_ID))

    def fake_ast(audios: list, model: object, top_k: int | None = None, **kwargs: object) -> list:
        """AST, whole-audio mode."""
        detector_calls["ast"].append({"top_k": top_k, **kwargs})
        labels = list(scores["ast"])
        return [AudioClassificationResult(labels=labels, scores=[scores["ast"][label] for label in labels])]

    def fake_hear(
        audios: list,
        model: str = "hear-event-detector",
        device: object = None,
        hop_length: float = 0.25,
        top_k: int | None = None,
    ) -> list:
        """HeAR's sliding detector."""
        detector_calls["hear"].append({"top_k": top_k, "hop_length": hop_length})
        ranked = sorted(scores["hear"].items(), key=lambda kv: -kv[1])
        window = {
            "start": 0.0,
            "end": 2.0,
            "label_scores": [{label: score} for label, score in ranked],
            "win_length": 2.0,
            "hop_length": hop_length,
        }
        return [[window] for _ in audios]

    monkeypatch.setattr(node, "classify_audios", fake_ast)
    monkeypatch.setattr(node, "detect_health_acoustic_events", fake_hear)
    return scores


def _kind(store: ProvStore, name: str) -> Entity:
    """The one kind entity for this kind."""
    [entity] = [e for e in store.entities("kind") if e.attributes["kind"] == name]
    return entity


class TestEligibility:
    """Who may vote, per kind."""

    def test_hear_is_barred_from_the_speech_kind(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """A strong HeAR Speech score contributes nothing: speech is folded from families A and B."""
        mock_detectors["hear"]["Speech"] = 0.99
        seed_store(store, yamnet_windows=[_yamnet_window(0.0, 0.96, {"Speech": 0.1})], words=())
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        speech = _kind(store, "speech")
        assert "C_health" not in speech.attributes["families"]
        assert result.kinds["speech"] == "absent"

    def test_lexical_airway_vote_reads_bracketed_tokens_only(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """[cough] votes airway-present; the plain word "cough" is lexical content, not an event."""
        seed_store(
            store,
            yamnet_windows=[_yamnet_window(0.0, 0.96, {"Speech": 0.1})],
            words=({"text": "[cough]", "start": 1.0, "end": 1.2},),
        )
        taxonomy(store, "plain", config, run_dir=tmp_path)
        airway = _kind(store, "airway")
        assert airway.attributes["families"]["B_lexical"]["state"] == "present"
        speech = _kind(store, "speech")
        assert speech.attributes["families"]["B_lexical"]["state"] == "absent"


class TestTheFold:
    """Presence needs agreement, absence needs unanimity, and the unmeasured count stays honest."""

    def test_unanimous_presence_is_present_while_min_families_is_unmeasured(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """All three eligible families agree, so any legal min_families would agree too."""
        mock_detectors["hear"]["Cough"] = 0.9
        seed_store(
            store,
            yamnet_windows=[_yamnet_window(0.0, 0.96, {"Cough": 0.9})],
            words=({"text": "[cough]", "start": 1.0, "end": 1.2},),
        )
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        assert result.kinds["airway"] == "present"
        assert _kind(store, "airway").attributes["min_families"] == "unmeasured"

    def test_disagreement_without_min_families_is_undecided_and_flags(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """One family present and two absent cannot be adjudicated without the count."""
        seed_store(store, yamnet_windows=[_yamnet_window(0.0, 0.96, {"Cough": 0.9})], words=())
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        assert result.kinds["airway"] == "undecided"
        assert result.verdict.outcome is Outcome.FLAG

    def test_two_of_three_present_without_min_families_is_undecided_and_flags(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """Two families present and one absent is not unanimity, so the unmeasured count stays honest."""
        mock_detectors["hear"]["Cough"] = 0.9
        seed_store(store, yamnet_windows=[_yamnet_window(0.0, 0.96, {"Cough": 0.9})], words=())
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        airway = _kind(store, "airway")
        assert airway.attributes["families"]["A_audioset"]["state"] == "present"
        assert airway.attributes["families"]["C_health"]["state"] == "present"
        assert airway.attributes["families"]["B_lexical"]["state"] == "absent"
        assert result.kinds["airway"] == "undecided"
        assert result.verdict.outcome is Outcome.FLAG

    def test_two_family_a_members_agreeing_count_as_one_family(
        self,
        store: ProvStore,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """YAMNet and AST both present is one family's vote, so min_families = 2 is not met."""
        override = tmp_path / "override.yaml"
        override.write_text("taxonomy:\n  presence_floor:\n    ast: 0.5\n  min_families:\n    airway: 2\n")
        config = load_triage_config(override)
        mock_detectors["ast"]["Cough"] = 0.99
        seed_store(store, yamnet_windows=[_yamnet_window(0.0, 0.96, {"Cough": 0.9})], words=())
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        family_a = _kind(store, "airway").attributes["families"]["A_audioset"]
        assert family_a["members"]["yamnet"]["state"] == "present"
        assert family_a["members"]["ast"]["state"] == "present"
        assert family_a["state"] == "present"
        assert result.kinds["airway"] == "undecided"

    def test_a_min_families_override_applies_the_design_rule(
        self,
        store: ProvStore,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """With min_families.airway = 2, two present families out of three decide presence."""
        override = tmp_path / "override.yaml"
        override.write_text("taxonomy:\n  min_families:\n    airway: 2\n")
        config = load_triage_config(override)
        mock_detectors["hear"]["Cough"] = 0.9
        seed_store(store, yamnet_windows=[_yamnet_window(0.0, 0.96, {"Cough": 0.9})], words=())
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        assert result.kinds["airway"] == "present"
        assert _kind(store, "airway").attributes["min_families"] == 2

    def test_an_out_of_range_override_raises(
        self,
        store: ProvStore,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """min_families beyond the eligible family count is a configuration error, not a fold."""
        override = tmp_path / "override.yaml"
        override.write_text("taxonomy:\n  min_families:\n    airway: 5\n")
        config = load_triage_config(override)
        seed_store(store, yamnet_windows=[], words=())
        with pytest.raises(ValueError, match="min_families"):
            taxonomy(store, "plain", config, run_dir=tmp_path)

    def test_absence_needs_unanimity_and_all_absent_fails(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """Every eligible family says absent for both screened kinds: the prediction is fail."""
        seed_store(store, yamnet_windows=[_yamnet_window(0.0, 0.96, {"Speech": 0.1, "Cough": 0.1})], words=())
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        assert result.kinds == {"airway": "absent", "speech": "absent", "voice_no_words": "not_screened"}
        assert result.verdict.outcome is Outcome.FAIL

    def test_speech_present_with_airway_absent_passes(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """Present + absent with no undecided kind is a pass."""
        seed_store(
            store,
            yamnet_windows=[_yamnet_window(0.0, 0.96, {"Speech": 0.9})],
            words=({"text": "hello", "start": 1.0, "end": 1.2},),
        )
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        assert result.kinds["speech"] == "present"
        assert result.kinds["airway"] == "absent"
        assert result.verdict.outcome is Outcome.PASS


class TestMembersAndArguments:
    """Member-level honesty and the explicit model arguments."""

    def test_ast_abstains_while_its_floor_is_null(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """AST's presence floor ships unmeasured; its member abstains and the record says why."""
        mock_detectors["ast"]["Cough"] = 0.99
        seed_store(store, yamnet_windows=[_yamnet_window(0.0, 0.96, {"Cough": 0.9})], words=())
        taxonomy(store, "plain", config, run_dir=tmp_path)
        family_a = _kind(store, "airway").attributes["families"]["A_audioset"]
        assert family_a["members"]["ast"]["state"] == "abstained"
        assert family_a["state"] == "present"  # YAMNet's vote carries the family

    def test_an_abstained_ast_member_still_records_what_it_measured(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """AST's inference ran, so the abstained member carries its score, label and analysis frame."""
        mock_detectors["ast"]["Cough"] = 0.99
        seed_store(store, yamnet_windows=[_yamnet_window(0.0, 0.96, {"Cough": 0.9})], words=())
        taxonomy(store, "plain", config, run_dir=tmp_path)
        member = _kind(store, "airway").attributes["families"]["A_audioset"]["members"]["ast"]
        assert member["state"] == "abstained"
        assert member["max_score"] == pytest.approx(0.99)
        assert member["label"] == "Cough"
        assert member["frame_s"] == config.require("taxonomy.ast_frame_s") == 10.24

    def test_model_arguments_are_explicit(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
        detector_calls: dict[str, list],
    ) -> None:
        """AST runs with sigmoid and no top-k truncation; HeAR keeps all eight labels."""
        seed_store(store, yamnet_windows=[], words=())
        taxonomy(store, "plain", config, run_dir=tmp_path)
        [ast_call] = detector_calls["ast"]
        assert ast_call["function_to_apply"] == "sigmoid"
        assert ast_call["top_k"] is None
        [hear_call] = detector_calls["hear"]
        assert hear_call["top_k"] is None

    def test_advisory_on_fail_everything_is_still_written(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_store: Callable[..., dict],
        mock_detectors: dict[str, Any],
    ) -> None:
        """A fail is a prediction, not a gate: three kind entities and a verdict exist regardless."""
        seed_store(store, yamnet_windows=[], words=())
        result = taxonomy(store, "plain", config, run_dir=tmp_path)
        assert len(store.entities("kind")) == 3
        assert _kind(store, "voice_no_words").attributes["state"] == "not_screened"
        assert store.get_entity(result.verdict_entity_id).attributes["kinds"] == result.kinds
