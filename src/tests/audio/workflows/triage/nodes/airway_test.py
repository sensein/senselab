"""AIRWAY interprets PREPROCESS's spans.

HeAR labels whole spans in a silent buffer, YAMNet confirms from its own native windows, ASR words
are presence-only evidence, a hint changes only what an absence means. HeAR is monkeypatched on the
node module; nothing here loads weights.
"""

import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
import pytest

from senselab.audio.data_structures import Audio, AudioHints
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes import airway as node
from senselab.audio.workflows.triage.nodes.airway import airway
from senselab.audio.workflows.triage.vocabulary import Outcome
from senselab.utils.prov_store import ProvStore

matplotlib.use("Agg", force=True)


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
def hear_calls() -> list[dict[str, Any]]:
    """Captured HeAR call payloads, in call order."""
    return []


@pytest.fixture
def hear_scores() -> dict[str, float]:
    """The mutable label scores the fake detector returns for every window."""
    return {"Cough": 0.97, "Breathe": 0.2, "Speech": 0.01}


@pytest.fixture
def mock_hear(monkeypatch: pytest.MonkeyPatch, hear_calls: list[dict[str, Any]], hear_scores: dict[str, float]) -> None:
    """Replace the HeAR detector; the payload mirrors detect_health_acoustic_events' return."""

    def fake_hear(
        audios: list,
        model: str = "hear-event-detector",
        device: object = None,
        hop_length: float = 0.25,
        top_k: int | None = None,
    ) -> list:
        """One window per 2 s of each input, all carrying the fixture scores."""
        hear_calls.append(
            {
                "hop_length": hop_length,
                "lengths": [int(a.waveform.shape[-1]) for a in audios],
                "waveforms": [a.waveform.clone() for a in audios],
            }
        )
        ranked = sorted(hear_scores.items(), key=lambda kv: -kv[1])
        window = {
            "start": 0.0,
            "end": 2.0,
            "label_scores": [{label: score} for label, score in ranked],
            "win_length": 2.0,
            "hop_length": hop_length,
        }
        return [[dict(window)] for _ in audios]

    monkeypatch.setattr(node, "detect_health_acoustic_events", fake_hear)


def _labels(store: ProvStore) -> list:
    """Every label assertion in the store."""
    return [a for a in store.entities("assertion") if a.attributes.get("verb") == "label"]


def _answers(store: ProvStore, verb: str) -> list:
    """Every confirm/contest/abstain assertion of one verb."""
    return [a for a in store.entities("assertion") if a.attributes.get("verb") == verb]


class TestHearClassification:
    """Step 1: the whole span, buffered by the module's own function."""

    def test_the_whole_span_is_buffered_by_the_module_function(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_airway_store: Callable[..., dict],
        mock_hear: None,
        hear_calls: list[dict[str, Any]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The buffer comes from span_to_hear_buffer at the configured placement; one window scored."""
        from senselab.audio.tasks.health_acoustics.hear import span_to_hear_buffer

        buffer_calls: list[dict[str, Any]] = []

        def recording_buffer(audio: Audio, start_s: float, end_s: float, *, placement: str) -> Audio:
            """The real function, with its arguments captured."""
            buffer_calls.append({"start_s": start_s, "end_s": end_s, "placement": placement})
            return span_to_hear_buffer(audio, start_s, end_s, placement=placement)

        monkeypatch.setattr(node, "span_to_hear_buffer", recording_buffer)
        ids = seed_airway_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=[])
        result = airway(store, "plain", config, run_dir=tmp_path)
        [buffer_call] = buffer_calls
        assert buffer_call == {"start_s": 1.5, "end_s": 1.65, "placement": "centre"}
        window_s = float(config.require("hear.window_s"))
        [call] = hear_calls
        assert call["hop_length"] == window_s
        assert call["lengths"] == [int(window_s * 16000)]  # the function's whole-window buffer
        [label] = _labels(store)
        assert label.attributes["label"] == "Cough"
        assert label.attributes["input"] == "buffered"
        assert ids["spans"][0] in store.derived_from(label.id)
        assert result.verdict.outcome in (Outcome.PASS, Outcome.FLAG)

    def test_a_span_longer_than_the_window_takes_the_sliding_path(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_airway_store: Callable[..., dict],
        mock_hear: None,
        hear_calls: list[dict[str, Any]],
    ) -> None:
        """span_to_hear_buffer refuses a 3 s span; its own audio is scanned and the assertion says so."""
        seed_airway_store(store, spans=((0.5, 3.5, 40.0),), yamnet_windows=[])
        airway(store, "plain", config, run_dir=tmp_path)
        [call] = hear_calls
        assert call["hop_length"] == 0.25  # the detector's own default, not the buffer hop
        assert call["lengths"] == [int(3.0 * 16000)]
        [label] = _labels(store)
        assert label.attributes["input"] == "sliding"

    def test_a_best_label_below_the_floor_leaves_the_span_unlabelled(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_airway_store: Callable[..., dict],
        mock_hear: None,
        hear_scores: dict[str, float],
    ) -> None:
        """No label assertion, no substitute record, and the branch flags."""
        hear_scores.update({"Cough": 0.3, "Breathe": 0.2})
        seed_airway_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=[])
        result = airway(store, "plain", config, run_dir=tmp_path)
        assert _labels(store) == []
        assert result.verdict.outcome is Outcome.FLAG
        verdict = store.get_entity(result.verdict_entity_id)
        assert verdict.attributes["labelled_n"] == 0

    def test_airway_has_no_path_to_yamnet_as_a_model(self) -> None:
        """YAMNet is read from the store's native windows; the module cannot classify with it."""
        assert not hasattr(node, "classify_audios")


class TestYamnetConfirmation:
    """Step 2: coverage over native windows; confirm, contest or abstain — never relabel."""

    def test_matching_coverage_confirms(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_airway_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """Cough windows over a Cough span confirm it, with the coverage recorded."""
        windows = [
            _yamnet_window(1.44, 2.40, {"Cough": 0.9}),
            _yamnet_window(0.96, 1.92, {"Cough": 0.8}),
        ]
        seed_airway_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=windows)
        result = airway(store, "plain", config, run_dir=tmp_path)
        [confirm] = _answers(store, "confirm")
        assert confirm.attributes["winner"] == "Cough"
        assert confirm.attributes["coverage"] == 1.0
        assert confirm.attributes["n_windows"] == 2
        [label] = _labels(store)
        assert label.id in store.derived_from(confirm.id)
        assert result.verdict.outcome is Outcome.PASS

    def test_a_confident_outside_label_contests_without_relabelling(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_airway_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """Speech coverage against a Cough label contests and flags; the label stands."""
        windows = [_yamnet_window(1.44, 2.40, {"Speech": 0.9, "Cough": 0.1})]
        seed_airway_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=windows)
        result = airway(store, "plain", config, run_dir=tmp_path)
        [contest] = _answers(store, "contest")
        assert contest.attributes["winner"] == "Speech"
        [label] = _labels(store)
        assert label.attributes["label"] == "Cough"
        assert result.verdict.outcome is Outcome.FLAG
        assert store.get_entity(result.verdict_entity_id).attributes["contested_n"] == 1

    def test_nothing_confident_anywhere_abstains_single_source(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_airway_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """No window reaches the coverage threshold: the label stands, marked single-source."""
        windows = [_yamnet_window(1.44, 2.40, {"Cough": 0.2, "Speech": 0.1})]
        seed_airway_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=windows)
        result = airway(store, "plain", config, run_dir=tmp_path)
        [abstain] = _answers(store, "abstain")
        assert abstain.attributes["best_coverage"] == 0.0
        assert abstain.attributes["n_windows"] == 1
        assert result.verdict.outcome is Outcome.PASS

    def test_breathe_is_confirmed_by_sigh(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_airway_store: Callable[..., dict],
        mock_hear: None,
        hear_scores: dict[str, float],
    ) -> None:
        """The confirmation map sends Breathe to {Breathing, Sigh, Gasp}."""
        hear_scores.update({"Cough": 0.1, "Breathe": 0.95})
        windows = [_yamnet_window(1.44, 2.40, {"Sigh": 0.8})]
        seed_airway_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=windows)
        result = airway(store, "plain", config, run_dir=tmp_path)
        [confirm] = _answers(store, "confirm")
        assert confirm.attributes["winner"] == "Sigh"
        assert confirm.attributes["mapped_to"] == "Breathe"
        assert result.verdict.outcome is Outcome.PASS


class TestLexicalContamination:
    """Step 3: the interval spans the gaps; brackets and out-of-interval words do not count."""

    def test_a_word_in_the_gap_between_labelled_spans_flags_by_id_only(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_airway_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """The interval covers first-start to last-end; the flag names word ids, never text."""
        ids = seed_airway_store(
            store,
            spans=((1.0, 1.2, 40.0), (2.5, 2.7, 40.0)),
            yamnet_windows=[],
            words=(
                {"text": "Marisol", "start": 1.8, "end": 1.9},
                {"text": "[cough]", "start": 1.85, "end": 1.95},
                {"text": "later", "start": 3.5, "end": 3.6},
            ),
        )
        result = airway(store, "plain", config, run_dir=tmp_path)
        [flag] = [a for a in store.entities("assertion") if a.attributes.get("verb") == "flag"]
        assert flag.attributes["reason"] == "lexical_contamination"
        assert flag.attributes["word_ids"] == [ids["words"][0]]
        assert "Marisol" not in json.dumps(flag.attributes)
        [interval] = store.entities("interval")
        assert interval.extent == (1.0, 2.7)
        assert result.verdict.outcome is Outcome.FLAG

    def test_a_word_outside_the_interval_does_not_flag(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_airway_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """Unlabelled spans never extend the interval and later words never enter it."""
        seed_airway_store(
            store,
            spans=((1.0, 1.2, 40.0),),
            yamnet_windows=[],
            words=({"text": "later", "start": 3.5, "end": 3.6},),
        )
        result = airway(store, "plain", config, run_dir=tmp_path)
        assert [a for a in store.entities("assertion") if a.attributes.get("verb") == "flag"] == []
        assert result.verdict.outcome is Outcome.PASS


class TestOutcomeAndHint:
    """Step 4: a hint conditions only what an absence means."""

    def test_no_spans_is_fail_and_a_hint_makes_it_flag(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_airway_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """Nothing proposed: fail without a hint, flag with one — never a pass."""
        seed_airway_store(store, spans=(), yamnet_windows=[], no_contrast_k=18.0)
        result = airway(store, "plain", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert "no_contrast" in result.verdict.why
        hinted_store = ProvStore(run_id="hinted")
        seed_airway_store(hinted_store, spans=(), yamnet_windows=[], no_contrast_k=18.0)
        hinted = airway(hinted_store, "plain", config, hint=AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert hinted.verdict.outcome is Outcome.FLAG

    def test_no_contrast_at_another_k_is_not_this_readers_no_contrast(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_airway_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """no_contrast is a (K, recording) finding; a 12 dB finding says nothing at 18 dB."""
        seed_airway_store(store, spans=(), yamnet_windows=[], no_contrast_k=12.0)
        result = airway(store, "plain", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert "no_contrast" not in result.verdict.why

    def test_a_hint_changes_nothing_when_spans_are_labelled(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_airway_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """With labelled spans the hint is inert: same pass either way."""
        seed_airway_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=[])
        result = airway(store, "plain", config, hint=AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS


class TestFigure:
    """The figure is an artifact; its failure changes no verdict."""

    def test_the_figure_is_written(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_airway_store: Callable[..., dict],
        mock_hear: None,
    ) -> None:
        """One aligned figure per recording, under run_dir/figures."""
        seed_airway_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=[])
        result = airway(store, "plain", config, run_dir=tmp_path)
        assert result.figure_path is not None
        assert result.figure_path.exists()

    def test_a_figure_failure_changes_no_verdict(
        self,
        store: ProvStore,
        config: TriageConfig,
        tmp_path: Path,
        seed_airway_store: Callable[..., dict],
        mock_hear: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Rendering raising leaves the same outcome with figure_path None."""

        def broken_plot(*args: object, **kwargs: object) -> object:
            """A renderer crash."""
            raise RuntimeError("no display")

        monkeypatch.setattr(node, "plot_aligned_panels", broken_plot)
        seed_airway_store(store, spans=((1.5, 1.65, 40.0),), yamnet_windows=[])
        result = airway(store, "plain", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS
        assert result.figure_path is None
