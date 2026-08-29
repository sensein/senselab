"""AIRWAY re-evaluates PREPROCESS's candidate spans with HeAR.

HeAR confirms whether an isolated candidate carries cough or breath, YAMNet may contest it only
from inside the HeAR window that carried the label, the gate is per task with a band around it, and
a hint conditions only what an absence means. The seeder writes the PREPROCESS-shaped provenance
surface the branch consumes.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes import airway as airway_module
from senselab.audio.workflows.triage.nodes.airway import airway
from senselab.audio.workflows.triage.nodes.common import live_entities
from senselab.audio.workflows.triage.nodes.routing import routing
from senselab.audio.workflows.triage.nodes.verdict import verdict
from senselab.audio.workflows.triage.vocabulary import Outcome, Triage
from senselab.utils.prov_store import Entity, ProvStore

_CLASSIFIER_GRID = {"hear": (2.0, 2.0), "yamnet": (0.96, 0.48)}
_HEAR_CODES = {(): 0.0, ("Breathe",): 0.2, ("Cough",): 0.1, ("Laugh",): 0.3, ("Breathe", "Cough"): 0.4}
_HEAR_LABELS_BY_CODE = {code: labels for labels, code in _HEAR_CODES.items()}


@pytest.fixture(autouse=True)
def fake_span_hear(monkeypatch: pytest.MonkeyPatch) -> None:
    """Return the label code embedded in each isolated test span instead of loading HeAR."""

    def _detect(audios: list, **_: object) -> list[list[dict[str, Any]]]:
        results = []
        for audio in audios:
            code = round(float(np.abs(audio.waveform.detach().cpu().numpy()).max()), 1)
            labels = _HEAR_LABELS_BY_CODE.get(code, ())
            results.append(
                [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "label_scores": [{label: 0.9} for label in labels],
                        "win_length": 2.0,
                        "hop_length": 2.0,
                    }
                ]
            )
        return results

    monkeypatch.setattr(airway_module, "detect_health_acoustic_events", _detect)


def _override(tmp_path: Path, body: str) -> TriageConfig:
    """The packaged configuration with one partial YAML deep-merged over it.

    Args:
        tmp_path: Where the override is written.
        body: The partial YAML.

    Returns:
        The merged configuration.
    """
    path = tmp_path / f"airway-{hashlib.sha256(body.encode()).hexdigest()[:12]}.yaml"
    path.write_text("windows:\n  hear:\n    default_threshold: 0.5\n    label_thresholds: {}\n" + body)
    return load_triage_config(path)


@pytest.fixture
def airway_config(tmp_path: Path) -> TriageConfig:
    """The packaged configuration with this branch's gate, its band and its contest set supplied.

    Args:
        tmp_path: Where the override is written.

    Returns:
        The merged configuration. The three values are a fixture, not a fit: the packaged file leaves
        each of them null.
    """
    return _override(tmp_path, "airway:\n  k_db: 18.0\n  k_margin_db: 2.0\n  contest_labels: [Speech]\n")


def _seed_airway_store(  # noqa: C901 — one independent block per derivative, as PREPROCESS has
    store: ProvStore,
    tmp_path: Path,
    *,
    spans: list[tuple[float, float, float]] | None = None,
    hear_windows: list[tuple[tuple[float, float], list[str]]] | None = None,
    yamnet_windows: list[tuple[tuple[float, float], list[str]]] | None = None,
    words: list[tuple[str, tuple[float, float]]] | None = None,
    events: list[tuple[str, tuple[float, float]]] | None = None,
    silence_windows: list[dict[str, Any]] | None = None,
    no_contrast_k: float | None = None,
    span_k_db: float = 18.0,
    span_merged: int = 1,
    duration_s: float = 5.0,
) -> dict[str, Any]:
    """Write the store surface AIRWAY reads, in the shapes PREPROCESS ships.

    Window classifications are placed at the extents the caller names rather than on a generated
    grid, because co-location is the property under test and it is a relation between two extents.

    Args:
        store: The store to seed.
        tmp_path: The run directory; the stream WAV goes under ``streams/``.
        spans: ``[(start, end, peak_over_floor_db), ...]`` envelope spans at ``span_k_db``.
        hear_windows: ``[((start, end), [label, ...]), ...]`` HeAR windows. None writes no HeAR
            record at all; ``[]`` writes the scores and the fold with no window.
        yamnet_windows: The same for YAMNet.
        words: ``[(text, (start, end)), ...]`` consensus words.
        events: ``[(bracketed, (start, end)), ...]`` bracketed non-words.
        silence_windows: YAMNet's graded windows, as ``{start, end, score, is_silence}`` dicts.
        no_contrast_k: The K a ``spans_no_contrast`` finding was made at.
        span_k_db: The K the seeded spans were proposed at.
        span_merged: The ``merged_proposals`` count every seeded span carries.
        duration_s: The stream's duration.

    Returns:
        The ids of what was written, keyed ``plain``/``spans``/``words``/``events``/``hear``/
        ``yamnet``/``silence``.
    """
    (tmp_path / "streams").mkdir(exist_ok=True)
    name = f"plain-{store.run_id}.wav"
    samples = np.zeros(int(duration_s * 16000), dtype=np.float32)
    activity = store.activity(node="PREPROCESS", step="seed", parameters={})
    agent = store.agent(agent_type="software", version="senselab test-seed")
    store.was_associated_with(activity, agent)
    ids: dict[str, Any] = {"spans": [], "words": [], "events": [], "hear": [], "yamnet": []}

    def _write(prov_type: str, extent: tuple[float, float] | None, attributes: dict[str, Any]) -> str:
        """One seeded entity, generated by the seed activity and attributed to the seed agent."""
        entity_id = store.entity(prov_type=prov_type, extent=extent, attributes=attributes)  # type: ignore[arg-type]
        store.was_generated_by(entity_id, activity)
        store.was_attributed_to(entity_id, agent)
        return entity_id

    for stream in ("recording", "plain"):
        stream_id = _write(
            "stream",
            (0.0, duration_s),
            {"name": stream, "path": f"streams/{name}", "sampling_rate": 16000, "channels": 1},
        )
        ids[stream] = stream_id

    for classifier, windows in (("hear", hear_windows), ("yamnet", yamnet_windows)):
        if windows is None:
            continue
        win_s, hop_s = _CLASSIFIER_GRID[classifier]
        model = store.agent(agent_type="model", model_id=f"seeded/{classifier}", unresolved_reason="seeded fixture")
        scores_id = store.entity(
            prov_type="measurement",
            extent=None,
            attributes={
                "name": f"{classifier}_scores",
                "classifier": classifier,
                "signal": "plain",
                "path": f"derivatives/{classifier}_scores.json",
                "n_windows": len(windows),
                "win_length_s": win_s if windows else None,
                "hop_s": hop_s if windows else None,
            },
        )
        store.was_generated_by(scores_id, activity)
        store.was_attributed_to(scores_id, model)
        windows_by_label: dict[str, list[str]] = {}
        for extent, labels in windows:
            window_id = _write(
                "measurement",
                extent,
                {
                    "name": f"{classifier}_window",
                    "classifier": classifier,
                    "signal": "plain",
                    "labels": list(labels),
                    "scores": {label: 0.9 for label in labels},
                },
            )
            store.was_derived_from(window_id, scores_id)
            ids[classifier].append(window_id)
            for label in labels:
                windows_by_label.setdefault(label, []).append(window_id)
        _write(
            "measurement",
            None,
            {
                "name": f"{classifier}_windows",
                "classifier": classifier,
                "signal": "plain",
                "labels": sorted(windows_by_label),
                "windows_by_label": windows_by_label,
                "n_windows": len(windows),
                "win_length_s": win_s if windows else None,
                "hop_s": hop_s if windows else None,
                "default_threshold": 0.5,
                "label_thresholds": {},
            },
        )

    for start, end, peak in spans if spans is not None else []:
        ids["spans"].append(
            _write(
                "span",
                (start, end),
                {
                    "peak_over_floor_db": peak,
                    "k_db": span_k_db,
                    "signal": "preemphasised",
                    "merged_proposals": span_merged,
                },
            )
        )

        labels = sorted(
            {
                label
                for (window_start, window_end), window_labels in hear_windows or []
                if window_start < end and window_end > start
                for label in window_labels
            }
        )
        code = _HEAR_CODES.get(tuple(labels), 0.0)
        samples[int(start * 16000) : int(end * 16000)] = code

    sf.write(str(tmp_path / "streams" / name), samples, 16000)

    for text, extent in events if events is not None else []:
        ids["events"].append(
            _write(
                "event",
                extent,
                {"bracketed": text, "raw": text, "origin": "bracketed", "recognizers": ["seeded/asr"]},
            )
        )

    for index, (text, extent) in enumerate(words if words is not None else []):
        ids["words"].append(
            _write(
                "word",
                extent,
                {
                    "text": text,
                    "confidence": 0.9,
                    "existence_confidence": 0.9,
                    "temporal_confidence": 0.9,
                    "coverage": 1.0,
                    "recognizers": ["seeded/asr"],
                    "timing_sources": 2,
                    "index": index,
                },
            )
        )

    if no_contrast_k is not None:
        ids["no_contrast"] = _write(
            "measurement",
            None,
            {"name": "spans_no_contrast", "signal": "preemphasised", "k_db": no_contrast_k, "reason": "seeded"},
        )
    if silence_windows is not None:
        ids["silence"] = _write(
            "measurement",
            None,
            {"name": "silence", "signal": "plain", "threshold": 0.5, "windows": silence_windows},
        )
    return ids


def _verdict_entity(store: ProvStore, node: str) -> Entity:
    """The verdict one node wrote.

    Args:
        store: The provenance store.
        node: The node's name.

    Returns:
        Its verdict entity.
    """
    return next(e for e in live_entities(store, "verdict") if e.attributes["node"] == node)


def _assertions(store: ProvStore, verb: str) -> list[Entity]:
    """Every live assertion carrying one verb.

    Args:
        store: The provenance store.
        verb: ``"label"``, ``"confirm"``, ``"contest"``, ``"abstain"`` or ``"flag"``.

    Returns:
        The assertions, oldest first.
    """
    return [e for e in live_entities(store, "assertion") if e.attributes.get("verb") == verb]


class TestItReevaluatesEachCandidate:
    """AIRWAY runs the event detector over the isolated candidate, not PREPROCESS's labels."""

    def test_the_module_has_the_span_detector(self) -> None:
        """The branch owns its candidate-level HeAR pass."""
        assert hasattr(airway_module, "detect_health_acoustic_events")
        assert hasattr(airway_module, "span_to_hear_buffer")

    def test_a_fresh_span_result_overrides_an_old_whole_file_label(
        self,
        monkeypatch: pytest.MonkeyPatch,
        store: ProvStore,
        airway_config: TriageConfig,
        tmp_path: Path,
    ) -> None:
        """A PREPROCESS HeAR label is not AIRWAY evidence after the candidate re-evaluation."""
        inputs: list[Any] = []

        def _fresh(audios: list, **_: object) -> list[list[dict[str, Any]]]:
            """Return a Cough independently of the old whole-file window."""
            inputs.extend(audios)
            return [[{"start": 0.0, "end": 2.0, "label_scores": [{"Cough": 0.9}]}] for _ in audios]

        monkeypatch.setattr(airway_module, "detect_health_acoustic_events", _fresh)
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Laugh"])])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["by_label"] == {"Cough": 1}
        assert inputs[0].waveform.shape[-1] / inputs[0].sampling_rate == pytest.approx(2.0)

    def test_it_records_the_model_agent_it_ran(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The fresh HeAR pass is visible in the branch's provenance."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough"])])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert {a.step for a in store.activities("AIRWAY")} <= {"classify", "confirm", "lexical"}
        associated = {agent_id for a in store.activities("AIRWAY") for agent_id in store.associated_with(a.id)}
        assert {store.get_agent(agent_id).agent_type for agent_id in associated} == {"software", "model"}

    def test_long_candidate_windows_keep_native_times_and_all_raw_scores(
        self,
        monkeypatch: pytest.MonkeyPatch,
        store: ProvStore,
        airway_config: TriageConfig,
        tmp_path: Path,
    ) -> None:
        """A long candidate maps each relative HeAR window back onto the recording timeline."""

        def _long_result(_: list, **__: object) -> list[list[dict[str, Any]]]:
            return [
                [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "label_scores": [
                            {"Cough": 0.9},
                            {"Snore": 0.1},
                            {"Baby Cough": 0.2},
                            {"Breathe": 0.4},
                            {"Sneeze": 0.3},
                            {"Throat Clear": 0.05},
                            {"Laugh": 0.15},
                            {"Speech": 0.25},
                        ],
                    },
                    {
                        "start": 2.0,
                        "end": 4.0,
                        "label_scores": [
                            {"Cough": 0.2},
                            {"Snore": 0.1},
                            {"Baby Cough": 0.05},
                            {"Breathe": 0.8},
                            {"Sneeze": 0.3},
                            {"Throat Clear": 0.4},
                            {"Laugh": 0.15},
                            {"Speech": 0.25},
                        ],
                    },
                ]
            ]

        monkeypatch.setattr(airway_module, "detect_health_acoustic_events", _long_result)
        ids = _seed_airway_store(store, tmp_path, spans=[(10.0, 14.0, 30.0)], duration_s=15.0)
        airway(store, "plain", airway_config, run_dir=tmp_path)

        windows = [
            entity
            for entity in live_entities(store, "measurement")
            if entity.attributes.get("name") == "hear_span_window"
        ]
        assert [window.extent for window in windows] == [(10.0, 12.0), (12.0, 14.0)]
        assert all(window.attributes["span_id"] == ids["spans"][0] for window in windows)
        assert all(window.attributes["isolated_span"] is True for window in windows)
        assert windows[0].attributes["labels"] == ["Cough"]
        assert windows[1].attributes["labels"] == ["Breathe"]
        assert windows[0].attributes["scores"] == {"Cough": 0.9}
        assert windows[0].attributes["raw_scores"] == {
            "Cough": 0.9,
            "Snore": 0.1,
            "Baby Cough": 0.2,
            "Breathe": 0.4,
            "Sneeze": 0.3,
            "Throat Clear": 0.05,
            "Laugh": 0.15,
            "Speech": 0.25,
        }


class TestHearConfirmsRatherThanFinds:
    """The candidate is the span; HeAR says whether that extent carries cough or breath."""

    def test_a_span_whose_windows_carry_the_label_is_labelled(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Membership in the window's set is the evidence; no score is compared here."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough"])])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["by_label"] == {"Cough": 1}

    def test_the_label_names_the_window_behind_it(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The evidence is a stored window, and the assertion is derived from it and from the span."""
        ids = _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough"])])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [label] = _assertions(store, "label")
        [hear] = [
            entity
            for entity in live_entities(store, "measurement")
            if entity.attributes.get("name") == "hear_span_window"
        ]
        assert label.attributes["hear_window_ids"] == [hear.id]
        assert set(store.derived_from(label.id)) == {ids["spans"][0], hear.id}

    def test_a_window_carrying_two_labels_of_interest_labels_the_span_twice(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A window's product is a set, so a span may carry more than one label and by_label counts each."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough", "Breathe"])])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        verdict = _verdict_entity(store, "AIRWAY")
        assert verdict.attributes["by_label"] == {"Breathe": 1, "Cough": 1}
        assert verdict.attributes["labelled_n"] == 1, "one span, labelled twice"

    def test_a_hear_window_without_a_span_labels_nothing(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """HeAR does not find a span; with no candidate there is nothing to confirm."""
        _seed_airway_store(store, tmp_path, spans=[], hear_windows=[((0.0, 2.0), ["Cough"])])
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert _verdict_entity(store, "AIRWAY").attributes["labelled_n"] == 0

    def test_a_transcribed_span_is_not_offered_to_hear(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A span overlapping consensus words is transcribed content, not an airway candidate."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"])],
            words=[("hello", (1.0, 1.2))],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["labelled_n"] == 0

    def test_a_span_carrying_only_events_stays_eligible(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Bracketed and onomatopoeic events are exactly what this branch is looking for."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"])],
            events=[("[COUGH]", (1.0, 1.2))],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["labelled_n"] == 1

    def test_a_span_whose_windows_carry_no_member_of_interest_is_unlabelled(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A span without a label is simply a span without a label assertion."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Laugh"])])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert not [e for e in live_entities(store, "assertion") if e.attributes.get("verb") == "label"]

    def test_a_window_that_does_not_overlap_the_span_is_not_its_evidence(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A HeAR window elsewhere in the recording describes a different two seconds."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((2.0, 4.0), ["Cough"])])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["labelled_n"] == 0

    def test_an_invalidated_span_is_not_labelled(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """This node's reads follow the store's shared rule; a withdrawn span is not a candidate."""
        ids = _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0), (2.5, 2.8, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"]), ((2.0, 4.0), ["Cough"])],
        )
        withdraw = store.activity(node="PREPROCESS", step="withdraw", parameters={})
        store.was_invalidated_by(ids["spans"][1], withdraw)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["labelled_n"] == 1


class TestContestRequiresColocation:
    """A label a window away is a different event, not a disagreement about this one (V21)."""

    def test_a_contest_label_in_the_same_hear_window_contests(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Both inside the HeAR window whose set carried the label."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"])],
            yamnet_windows=[((0.96, 1.92), ["Speech"])],
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["contested_n"] == 1
        assert result.verdict.outcome is Outcome.FLAG

    def test_a_contest_names_the_window_the_colocation_was_found_in(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A reader must be able to reach both windows the co-location was read from."""
        ids = _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"])],
            yamnet_windows=[((0.96, 1.92), ["Speech"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [contest] = _assertions(store, "contest")
        assert contest.attributes["yamnet_window_ids"] == ids["yamnet"]
        assert all(
            store.get_entity(window_id).attributes["name"] == "hear_span_window"
            for window_id in contest.attributes["hear_window_ids"]
        )

    def test_a_contest_label_outside_that_window_does_not(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The YAMNet window is outside the HeAR window, so it describes a different event."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"])],
            yamnet_windows=[((2.88, 3.84), ["Speech"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["contested_n"] == 0

    def test_a_yamnet_window_straddling_the_hear_boundary_does_not_contest(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Overlapping the HeAR window is not being inside it; half the evidence is elsewhere."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"])],
            yamnet_windows=[((1.44, 2.40), ["Speech"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["contested_n"] == 0

    def test_a_label_outside_contest_labels_does_not_contest(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The eligible set is declared, not all 521."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"])],
            yamnet_windows=[((0.96, 1.92), ["Rain"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["contested_n"] == 0

    def test_a_mapped_label_in_the_same_window_confirms(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The confirmation map sends the HeAR label to the AudioSet labels that corroborate it."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"])],
            yamnet_windows=[((0.96, 1.92), ["Cough"])],
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        [confirm] = _assertions(store, "confirm")
        assert confirm.attributes["label"] == "Cough"
        assert confirm.attributes["yamnet_labels"] == ["Cough"]
        assert result.verdict.outcome is Outcome.PASS

    def test_any_member_of_the_confirmation_set_confirms_not_only_the_identical_label(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Breathe's set is {Breathing, Sigh, Gasp}: Sigh corroborates a breath and is not the same word.

        Every other confirmation here reads Cough against Cough, which a node comparing the HeAR label
        to itself would also pass. Sigh is a member of Breathe's set and nothing else, so shrinking the
        map to {Breathing} leaves the span abstaining instead.
        """
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_windows=[((0.0, 2.0), ["Breathe"])],
            yamnet_windows=[((0.96, 1.92), ["Sigh"])],
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        [confirm] = _assertions(store, "confirm")
        assert confirm.attributes["label"] == "Breathe"
        assert confirm.attributes["yamnet_labels"] == ["Sigh"]
        assert _assertions(store, "abstain") == []
        assert result.verdict.outcome is Outcome.PASS

    def test_no_colocated_window_abstains(self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path) -> None:
        """Nothing co-located either way: the label stands, marked single-source."""
        _seed_airway_store(
            store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough"])], yamnet_windows=[]
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        [abstain] = _assertions(store, "abstain")
        assert abstain.attributes["colocated_windows_n"] == 0
        assert result.verdict.outcome is Outcome.PASS

    def test_a_contest_never_relabels(self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path) -> None:
        """Flag the span; the label stands and the assertion is not invalidated."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"])],
            yamnet_windows=[((0.96, 1.92), ["Speech"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        label = next(e for e in live_entities(store, "assertion") if e.attributes.get("verb") == "label")
        assert label.attributes["label"] == "Cough"

    def test_intersecting_label_sets_are_refused_at_load(self, store: ProvStore, tmp_path: Path) -> None:
        """A label cannot both support and contest the same conclusion."""
        config = _override(tmp_path, "airway:\n  contest_labels: [Speech, Cough]\n  k_db: 18.0\n")
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)])
        before = len(store.entities())
        with pytest.raises(ValueError, match="disjoint"):
            airway(store, "plain", config, run_dir=tmp_path)
        assert len(store.entities()) == before


class TestTheGateIsAdjustableAndItsEdgeFlags:
    """K is per task, and a span that only just cleared it is a decision a human should see."""

    def test_airway_k_db_overrides_the_shared_gate(self, store: ProvStore, tmp_path: Path) -> None:
        """An airway event is level-limited; one value fitted on coughs does not serve quiet breaths."""
        config = _override(tmp_path, "airway:\n  k_db: 12.0\n  k_margin_db: 2.0\n  contest_labels: [Speech]\n")
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], span_k_db=12.0)
        airway(store, "plain", config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["k_db"] == 12.0

    def test_a_span_at_another_k_is_not_this_branchs_span(self, store: ProvStore, tmp_path: Path) -> None:
        """The gate selects the spans as well as naming them; an 18 dB span is not a 12 dB reader's."""
        config = _override(tmp_path, "airway:\n  k_db: 12.0\n  k_margin_db: 2.0\n  contest_labels: [Speech]\n")
        _seed_airway_store(
            store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough"])], span_k_db=18.0
        )
        result = airway(store, "plain", config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert "no span was proposed at this K" in result.verdict.why

    def test_a_declared_task_overrides_it_again(self, store: ProvStore, tmp_path: Path) -> None:
        """airway.k_db_by_task is the per-task gate."""
        config = _override(
            tmp_path,
            "airway:\n  k_db: 18.0\n  k_db_by_task: {breath: 8.0}\n  k_margin_db: 2.0\n  contest_labels: [Speech]\n",
        )
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], span_k_db=8.0)
        hint = AudioHints(metadata={"task": "breath"})
        airway(store, "plain", config, hint, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["k_db"] == 8.0

    def test_an_undeclared_task_falls_back_to_the_branch_gate(self, store: ProvStore, tmp_path: Path) -> None:
        """A task with no entry in the map is not a reason to invent a gate for it."""
        config = _override(
            tmp_path,
            "airway:\n  k_db: 18.0\n  k_db_by_task: {breath: 8.0}\n  k_margin_db: 2.0\n  contest_labels: [Speech]\n",
        )
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], span_k_db=18.0)
        hint = AudioHints(metadata={"task": "cough"})
        airway(store, "plain", config, hint, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["k_db"] == 18.0

    def test_a_span_inside_the_margin_flags_with_its_margin(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Any span the gate would have kept out under a slightly different setting is visible."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 19.0)], hear_windows=[((0.0, 2.0), ["Cough"])])
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["near_gate_n"] == 1
        assert result.verdict.outcome is Outcome.FLAG
        label = next(e for e in live_entities(store, "assertion") if e.attributes.get("verb") == "label")
        assert label.attributes["margin_over_k_db"] == pytest.approx(1.0)

    def test_a_span_clear_of_the_margin_does_not_flag(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The band is a band; a span 12 dB over the gate is not a borderline decision."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough"])])
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["near_gate_n"] == 0
        assert result.verdict.outcome is Outcome.PASS

    def test_a_null_margin_leaves_the_band_inert(self, store: ProvStore, tmp_path: Path) -> None:
        """Nobody derived how close is too close, so no span is near the gate."""
        config = _override(tmp_path, "airway:\n  k_db: 18.0\n  contest_labels: [Speech]\n")
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 19.0)], hear_windows=[((0.0, 2.0), ["Cough"])])
        airway(store, "plain", config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["near_gate_n"] == 0
        label = next(e for e in live_entities(store, "assertion") if e.attributes.get("verb") == "label")
        assert "margin_over_k_db" not in label.attributes

    def test_the_merge_rate_is_reported(self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path) -> None:
        """A span covering several events must be legible as one."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.9, 30.0)],
            span_merged=3,
            hear_windows=[((0.0, 2.0), ["Cough"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["merged_n"] == 3

    def test_the_merge_rate_counts_a_span_once_however_many_labels_it_carries(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """merged_n is a count of absorbed proposals, so a second label must not double it."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.9, 30.0)],
            span_merged=3,
            hear_windows=[((0.0, 2.0), ["Cough", "Breathe"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["merged_n"] == 3


class TestCertifiedSilence:
    """The label records whether its span lies inside YAMNet-certified silence."""

    def test_a_span_inside_all_silent_windows_reads_true(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Every overlapping graded window certified silent: the label says True."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"])],
            silence_windows=[{"start": 1.0, "end": 2.0, "score": 0.8, "is_silence": True}],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [label] = _assertions(store, "label")
        assert label.attributes["in_certified_silence"] is True

    def test_a_span_over_mixed_windows_reads_false(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """One overlapping window graded not-silent is enough: the label says False."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"])],
            silence_windows=[
                {"start": 0.8, "end": 1.1, "score": 0.8, "is_silence": True},
                {"start": 1.1, "end": 1.7, "score": 0.2, "is_silence": False},
            ],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [label] = _assertions(store, "label")
        assert label.attributes["in_certified_silence"] is False

    def test_a_span_no_graded_window_overlaps_reads_none(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Graded windows exist but none overlaps the span: the question has no answer, None."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"])],
            silence_windows=[{"start": 0.0, "end": 0.5, "score": 0.9, "is_silence": True}],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [label] = _assertions(store, "label")
        assert label.attributes["in_certified_silence"] is None


class TestLexicalContamination:
    """The interval spans the gaps; an event is not a word and a word outside it is not contamination."""

    def test_a_word_in_the_gap_between_labelled_spans_flags_by_id_only(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The interval covers first-start to last-end; the flag names word ids, never text."""
        ids = _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.2, 30.0), (2.5, 2.7, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"]), ((2.0, 4.0), ["Cough"])],
            words=[("Marisol", (1.8, 1.9)), ("later", (3.5, 3.6))],
            events=[("[COUGH]", (1.85, 1.95))],
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        [flag] = _assertions(store, "flag")
        assert flag.attributes["reason"] == "lexical_contamination"
        assert flag.attributes["word_ids"] == [ids["words"][0]]
        assert "Marisol" not in json.dumps(flag.attributes)
        [interval] = live_entities(store, "interval")
        assert interval.extent == (1.0, 2.7)
        assert interval.attributes["name"] == "airway_labelled_interval"
        assert result.verdict.outcome is Outcome.FLAG

    def test_a_word_outside_the_interval_does_not_flag(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Unlabelled spans never extend the interval and later words never enter it."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.2, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"])],
            words=[("later", (3.5, 3.6))],
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _assertions(store, "flag") == []
        assert result.verdict.outcome is Outcome.PASS

    def test_an_invalidated_word_does_not_contaminate(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A withdrawn word is not evidence, on this read as on every other."""
        ids = _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.2, 30.0), (2.5, 2.7, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"]), ((2.0, 4.0), ["Cough"])],
            words=[("Marisol", (1.8, 1.9))],
        )
        withdraw = store.activity(node="PREPROCESS", step="withdraw", parameters={})
        store.was_invalidated_by(ids["words"][0], withdraw)
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _assertions(store, "flag") == []
        assert result.verdict.outcome is Outcome.PASS


class TestOutcomeAndHint:
    """An absence is a fail whatever the hint said; the mismatch is the fold's to name."""

    def test_no_spans_is_fail_with_or_without_a_hint(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Nothing proposed: this branch found no airway, and a declaration does not supply one."""
        _seed_airway_store(store, tmp_path, spans=[], no_contrast_k=18.0)
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert "no_contrast" in result.verdict.why
        hinted_store = ProvStore(run_id="hinted")
        _seed_airway_store(hinted_store, tmp_path, spans=[], no_contrast_k=18.0)
        hinted = airway(hinted_store, "plain", airway_config, hint=AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert hinted.verdict.outcome is Outcome.FAIL
        assert "hint" not in hinted.verdict.why
        assert _verdict_entity(hinted_store, "AIRWAY").attributes["flags"] == []

    def test_no_contrast_at_another_k_is_not_this_readers_no_contrast(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """no_contrast is a (K, recording) finding; a 12 dB finding says nothing at 18 dB."""
        _seed_airway_store(store, tmp_path, spans=[], no_contrast_k=12.0)
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert "no_contrast" not in result.verdict.why

    def test_spans_that_carry_no_label_fail_like_no_spans_at_all(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Both routes to no airway established mean the same thing, and neither is a hint's to change."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Laugh"])])
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert _verdict_entity(store, "AIRWAY").attributes["flags"] == []
        hinted_store = ProvStore(run_id="hinted-unlabelled")
        _seed_airway_store(hinted_store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Laugh"])])
        hinted = airway(hinted_store, "plain", airway_config, hint=AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert hinted.verdict.outcome is Outcome.FAIL
        assert _verdict_entity(hinted_store, "AIRWAY").attributes["flags"] == []

    def test_a_hint_that_does_not_declare_airway_leaves_an_absence_a_fail(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The control: an unrelated tag and a declaring tag reach the same fail."""
        _seed_airway_store(store, tmp_path, spans=[], no_contrast_k=18.0)
        result = airway(store, "plain", airway_config, hint=AudioHints(may_contain=["music"]), run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL

    def test_a_hint_changes_nothing_when_spans_are_labelled(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """With labelled spans the hint is inert: same pass either way."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Cough"])])
        result = airway(store, "plain", airway_config, hint=AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.PASS

    def test_the_verdict_is_generated_by_the_step_that_concluded(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Walking generated_by from the verdict must reach the last step, not the first."""
        ids = _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.2, 30.0), (2.5, 2.7, 30.0)],
            hear_windows=[((0.0, 2.0), ["Cough"]), ((2.0, 4.0), ["Cough"])],
            words=[("Marisol", (1.8, 1.9))],
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        concluding = store.generated_by(result.verdict_entity_id)
        assert concluding is not None
        assert store.get_activity(concluding).step == "lexical"
        assert set(ids["words"]) <= set(store.uses_of(concluding)), "the concluding step is the one that read words"

    def test_the_verdict_of_an_unlabelled_run_is_generated_by_the_confirm_step(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """With no label there is no lexical step, so confirm concludes."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_windows=[((0.0, 2.0), ["Laugh"])])
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        concluding = store.generated_by(result.verdict_entity_id)
        assert concluding is not None
        assert store.get_activity(concluding).step == "confirm"


class TestTheFoldNamesTheHintMismatchThisBranchDoesNot:
    """The end-to-end pin: a declared kind no branch found reaches the file verdict as a mismatch."""

    def _seed_kinds(self, store: ProvStore) -> None:
        """Write TAXONOMY's classification: every kind absent, which is what an empty file screens as."""
        seed = store.activity(node="TAXONOMY", step="seed-kinds", parameters={})
        agent = store.agent(agent_type="software", version="senselab test-seed")
        store.was_associated_with(seed, agent)
        for kind_name in ("airway", "speech", "voice"):
            kind_id = store.entity(
                prov_type="kind",
                extent=None,
                attributes={"kind": kind_name, "state": "absent", "lines": {}, "stream": "plain"},
            )
            store.was_generated_by(kind_id, seed)

    def test_a_declared_kind_the_branch_did_not_find_flags_the_file_and_records_the_kind_absent(
        self, store: ProvStore, tmp_path: Path
    ) -> None:
        """AIRWAY fails, ROUTING recorded the claim, and the fold flags without resolving the kind present."""
        hint_config = _override(
            tmp_path,
            "airway:\n  k_db: 18.0\n  k_margin_db: 2.0\n  contest_labels: [Speech]\n"
            "routing:\n  hint_kind_map:\n    cough: airway\n",
        )
        hint = AudioHints(may_contain=["cough"])
        _seed_airway_store(store, tmp_path, spans=[], no_contrast_k=18.0)
        self._seed_kinds(store)
        routing(store, "plain", hint_config, hint, run_dir=tmp_path)
        branch = airway(store, "plain", hint_config, hint, run_dir=tmp_path)
        assert branch.verdict.outcome is Outcome.FAIL

        folded = verdict(store, None, hint_config, hint, run_dir=tmp_path).file_verdict
        assert folded.triage is Triage.FLAG
        assert folded.kinds["airway"] == "absent"
        assert folded.hints["airway"] == "claimed_not_found"
        assert folded.discard_ground is None
        assert any(
            reason.why == "hint mismatch: airway was declared and AIRWAY did not find it" for reason in folded.reasons
        ), [reason.why for reason in folded.reasons]

    def test_the_same_file_with_no_declaration_discards_as_acoustically_empty(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The control the upgrade made unreachable: nothing found and nothing claimed is an empty file."""
        _seed_airway_store(store, tmp_path, spans=[], no_contrast_k=18.0)
        self._seed_kinds(store)
        routing(store, "plain", airway_config, None, run_dir=tmp_path)
        airway(store, "plain", airway_config, None, run_dir=tmp_path)

        folded = verdict(store, None, airway_config, None, run_dir=tmp_path).file_verdict
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "acoustically_empty"
