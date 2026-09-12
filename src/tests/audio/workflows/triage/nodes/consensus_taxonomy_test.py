"""``consensus_taxonomy`` merges its classifiers on ontology identity, not on the label string.

HeAR and AudioSet spell the same event differently -- ``Throat Clear``/``Throat clearing``,
``Snore``/``Snoring``, ``Baby Cough``/``Cough`` -- so an exact-string merge left five of HeAR's
eight labels unable to reach ``n_classifiers: 2``. See
``specs/20260910-classifier-ontology-mapping/design.md``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from pathlib import Path

import pytest

from senselab.audio.workflows.triage.classifier_ontology import canonical_names
from senselab.audio.workflows.triage.config import TriageConfig
from senselab.audio.workflows.triage.nodes.common import find_measurement, software_agent
from senselab.audio.workflows.triage.nodes.taxonomy import PER_SPAN_CLASSIFIERS, _write_consensus_taxonomy
from senselab.utils.prov_store import ProvStore


def _seed_spans(store: ProvStore, per_classifier: Mapping[str, Sequence[Mapping[str, float]]]) -> None:
    """Write one span per scored entry, and each classifier's per-span measurement over it.

    Args:
        store: The store to write into.
        per_classifier: ``{classifier: [{label: score} per span]}``. Every classifier's sequence
            must be the same length; span ``i`` is shared by all of them.
    """
    agent = software_agent(store)
    lengths = {len(scores) for scores in per_classifier.values()}
    assert len(lengths) == 1, "every classifier must score the same spans"
    span_ids = []
    activity = store.activity(node="PREPROCESS", step="spans", parameters={})
    store.was_associated_with(activity, agent)
    for index in range(lengths.pop()):
        span_id = store.entity(
            prov_type="span", extent=(float(index), float(index) + 0.5), attributes={"signal": "preemphasised"}
        )
        store.was_generated_by(span_id, activity)
        span_ids.append(span_id)
    for classifier, per_span in per_classifier.items():
        step = PER_SPAN_CLASSIFIERS[classifier]
        span_activity = store.activity(node="PREPROCESS", step=step, parameters={})
        store.was_associated_with(span_activity, agent)
        for span_id, scores in zip(span_ids, per_span, strict=True):
            entity_id = store.entity(
                prov_type="measurement",
                extent=store.get_entity(span_id).extent,
                attributes={
                    "name": step,
                    "classifier": classifier,
                    "signal": "plain",
                    "span_id": span_id,
                    "raw_scores": dict(scores),
                    "labelled": True,
                },
            )
            store.was_generated_by(entity_id, span_activity)
            store.was_attributed_to(entity_id, agent)


def _rows(store: ProvStore, config: TriageConfig) -> dict[str, dict[str, Any]]:
    """Run the consensus fold and return its rows by label.

    Args:
        store: A store already carrying the per-span measurements.
        config: The configuration the fold reads.

    Returns:
        ``{label: row}``.
    """
    written = _write_consensus_taxonomy(store, config, software_agent(store))
    assert written, "the fold wrote nothing"
    measurement = find_measurement(store, "consensus_taxonomy")
    assert measurement is not None
    return {str(row["label"]): row for row in measurement.attributes["labels"]}


class TestTheTwoSpellingsOfOneNodeMerge:
    """The defect: HeAR and AudioSet name the same ontology node differently."""

    @pytest.mark.parametrize(
        ("hear_label", "audioset_label"),
        [("Throat Clear", "Throat clearing"), ("Snore", "Snoring"), ("Baby Cough", "Cough")],
    )
    def test_the_pair_is_one_row_reaching_two_classifiers(
        self, store: ProvStore, config: TriageConfig, hear_label: str, audioset_label: str
    ) -> None:
        """One node, one row, both classifiers on it."""
        _seed_spans(store, {"hear": [{hear_label: 0.8}], "yamnet": [{audioset_label: 0.6}]})

        rows = _rows(store, config)

        assert hear_label not in rows or hear_label == audioset_label
        assert rows[audioset_label]["n_classifiers"] == 2
        assert rows[audioset_label]["classifiers"] == ["hear", "yamnet"]
        assert rows[audioset_label]["peak_by_classifier"] == {"hear": 0.8, "yamnet": 0.6}

    def test_the_merged_row_is_named_by_the_ontology(self, store: ProvStore, config: TriageConfig) -> None:
        """The AudioSet display name is the row's name; the HeAR spelling survives beside it."""
        _seed_spans(store, {"hear": [{"Throat Clear": 0.8}], "yamnet": [{"Throat clearing": 0.6}]})

        row = _rows(store, config)["Throat clearing"]

        assert row["label"] == "Throat clearing"
        assert row["labels_by_classifier"] == {"hear": ["Throat Clear"], "yamnet": ["Throat clearing"]}

    def test_a_hear_only_label_is_still_renamed(self, store: ProvStore, config: TriageConfig) -> None:
        """Identity does not depend on a second classifier having voted."""
        _seed_spans(store, {"hear": [{"Snore": 0.7}]})

        rows = _rows(store, config)

        assert set(rows) == {"Snoring"}
        assert rows["Snoring"]["n_classifiers"] == 1
        assert rows["Snoring"]["labels_by_classifier"] == {"hear": ["Snore"]}


class TestOnlyTheMappedNodeMerges:
    """Corroboration is a subtree test; identity is not, or every subtree would collapse to a row."""

    def test_a_descendant_stays_its_own_row(self, store: ProvStore, config: TriageConfig) -> None:
        """``Throat clearing`` corroborates ``Cough`` but is not the same node, so it is not merged."""
        _seed_spans(store, {"hear": [{"Cough": 0.9}], "yamnet": [{"Throat clearing": 0.5}]})

        rows = _rows(store, config)

        assert set(rows) == {"Cough", "Throat clearing"}
        assert rows["Cough"]["n_classifiers"] == 1
        assert rows["Throat clearing"]["n_classifiers"] == 1


class TestAnOverlapIsCountedOnce:
    """A node two HeAR roots both reach must not enter ``n_classifiers`` twice."""

    def test_snoring_is_one_row_however_many_hear_roots_reach_it(self, store: ProvStore, config: TriageConfig) -> None:
        """``Snoring`` is in ``Breathe``'s subtree and is ``Snore``'s own node."""
        _seed_spans(store, {"hear": [{"Snore": 0.7, "Breathe": 0.4}], "yamnet": [{"Snoring": 0.6}]})

        rows = _rows(store, config)

        assert rows["Snoring"]["n_classifiers"] == 2
        assert rows["Snoring"]["peak_by_classifier"] == {"hear": 0.7, "yamnet": 0.6}
        assert rows["Breathing"]["n_classifiers"] == 1

    def test_two_hear_labels_on_one_node_are_one_classifier(self, store: ProvStore, config: TriageConfig) -> None:
        """``Cough`` and ``Baby Cough`` are both AudioSet ``Cough``; HeAR still voted once."""
        _seed_spans(store, {"hear": [{"Cough": 0.3, "Baby Cough": 0.8}], "yamnet": [{"Cough": 0.5}]})

        row = _rows(store, config)["Cough"]

        assert row["n_classifiers"] == 2
        assert row["peak_by_classifier"] == {"hear": 0.8, "yamnet": 0.5}
        assert row["labels_by_classifier"] == {"hear": ["Baby Cough", "Cough"], "yamnet": ["Cough"]}


class TestLabelsOutsideTheMapping:
    """Most AudioSet labels have no HeAR counterpart, and nothing may drop them."""

    def test_an_unmapped_audioset_label_survives(self, store: ProvStore, config: TriageConfig) -> None:
        """``Silence`` is an AudioSet class no HeAR label denotes."""
        _seed_spans(store, {"yamnet": [{"Silence": 0.95}]})

        rows = _rows(store, config)

        assert rows["Silence"]["n_classifiers"] == 1
        assert rows["Silence"]["peak_by_classifier"] == {"yamnet": 0.95}

    def test_a_label_the_ontology_does_not_hold_keeps_its_own_spelling(
        self, store: ProvStore, config: TriageConfig
    ) -> None:
        """A spelling absent from the profile is carried through rather than dropped."""
        assert "Not an AudioSet class" not in canonical_names()
        _seed_spans(store, {"yamnet": [{"Not an AudioSet class": 0.4}]})

        rows = _rows(store, config)

        assert rows["Not an AudioSet class"]["n_classifiers"] == 1


class TestTheClassifierSetIsUnchanged:
    """AST is whole-file in this pipeline; it writes no per-span measurement to fold."""

    def test_only_yamnet_and_hear_are_per_span(self) -> None:
        """Widening this set is a separate change with its own measurement behind it."""
        assert PER_SPAN_CLASSIFIERS == {"yamnet": "span_yamnet", "hear": "span_hear"}


class TestTheFeatureShardKeepsEachClassifiersOwnSpelling:
    """An ontology-named row must still reach a detector that knows only HeAR's vocabulary."""

    def test_a_merged_row_is_emitted_under_the_spelling_its_classifier_used(self, tmp_path: Path) -> None:
        """`Throat clearing` is the row; `airway.hear_peak.consensus` looks for `Throat Clear`."""
        from senselab.audio.workflows.triage.routing_analysis.features import RecordingFeatures, _absorb_measurement

        features = RecordingFeatures(stem="s", run_root="r", task_id="t", family="f")
        _absorb_measurement(
            features,
            {
                "name": "consensus_taxonomy",
                "labels": [
                    {
                        "label": "Throat clearing",
                        "peak_by_classifier": {"hear": 0.7, "yamnet": 0.4},
                        "labels_by_classifier": {"hear": ["Throat Clear"], "yamnet": ["Throat clearing"]},
                    }
                ],
            },
            {},
            tmp_path,
        )
        assert features.peaks["consensus|hear|Throat Clear"] == pytest.approx(0.7)
        assert features.peaks["consensus|yamnet|Throat clearing"] == pytest.approx(0.4)
        assert "consensus|hear|Throat clearing" not in features.peaks

    def test_a_row_without_native_spellings_falls_back_to_its_own_label(self, tmp_path: Path) -> None:
        """A row the merge never touched carries no spelling map, and must still be read."""
        from senselab.audio.workflows.triage.routing_analysis.features import RecordingFeatures, _absorb_measurement

        features = RecordingFeatures(stem="s", run_root="r", task_id="t", family="f")
        _absorb_measurement(
            features,
            {"name": "consensus_taxonomy", "labels": [{"label": "Cough", "peak_by_classifier": {"yamnet": 0.9}}]},
            {},
            tmp_path,
        )
        assert features.peaks["consensus|yamnet|Cough"] == pytest.approx(0.9)
