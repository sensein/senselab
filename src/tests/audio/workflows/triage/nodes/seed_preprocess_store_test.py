"""The one shared node-test fixture, pinned: every task after T1 builds its own seeder on this one.

The distinction these tests exist for is ``None`` against ``[]`` — nothing written against a
derivative that ran and found nothing. A consumer reads the first as ``unavailable`` and the second
as ``absent``, so a seeder that collapsed them would make six downstream test suites agree with a
node that was wrong.
"""

from typing import Any, Callable

from senselab.audio.workflows.triage.nodes.common import find_measurement, find_measurements, live_entities
from senselab.utils.prov_store import ProvStore


class TestSeeder:
    """The shared seeder writes what the contract says it writes."""

    def test_nothing_by_default(self, store: ProvStore, seed_preprocess_store: Callable[..., None]) -> None:
        """No derivative argument means no derivative written; the streams are always there."""
        seed_preprocess_store(store)
        assert len(live_entities(store, "stream")) == 2
        assert find_measurement(store, "yamnet_windows") is None
        assert find_measurement(store, "consensus_transcript") is None
        assert live_entities(store, "word") == []

    def test_empty_lists_write_the_derivative(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None]
    ) -> None:
        """An empty list is a derivative that ran and found nothing, unlike None."""
        seed_preprocess_store(store, words=[], yamnet_labels=[], phonation=[])
        consensus = find_measurement(store, "consensus_transcript")
        assert consensus is not None
        assert consensus.attributes["words"] == []
        pooled = find_measurement(store, "yamnet_windows")
        assert pooled is not None
        assert pooled.attributes["n_windows"] == 0
        assert [a.step for a in store.activities("PREPROCESS")].count("phonation_spans") == 1

    def test_the_full_surface(self, store: ProvStore, seed_preprocess_store: Callable[..., None]) -> None:
        """Every argument at once, in both the bare and the timed shape."""
        seed_preprocess_store(
            store,
            duration_s=6.0,
            yamnet_labels=[["Speech"], [], ["Cough", "Speech"]],
            ast_labels=[["Speech"]],
            hear_labels=[["Cough"], []],
            words=["hello", ("world", (2.0, 2.4))],
            events=["[COUGH]"],
            phonation=[(1.0, 3.0, "voiced"), (4.0, 4.5, "unvoiced")],
            spans=[(1.0, 1.2, 30.0)],
            span_merged=2,
            disruptions_file=True,
        )
        assert len(find_measurements(store, "yamnet_window")) == 3
        pooled = find_measurement(store, "yamnet_windows")
        assert pooled is not None
        assert pooled.attributes["labels"] == ["Cough", "Speech"]
        assert len(pooled.attributes["windows_by_label"]["Speech"]) == 2
        assert len(find_measurements(store, "hear_window")) == 2
        words: list[Any] = live_entities(store, "word")
        assert [w.attributes["text"] for w in words] == ["hello", "world"]
        assert words[1].extent == (2.0, 2.4)
        events = live_entities(store, "event")
        assert events[0].attributes["bracketed"] == "[COUGH]"
        phonation = [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]
        assert [e.attributes["production"] for e in phonation] == ["voiced", "unvoiced"]
        assert [e.attributes["duration_s"] for e in phonation] == [2.0, 0.5]
        envelope = [e for e in live_entities(store, "span") if e.attributes.get("family") is None]
        assert envelope[0].attributes["merged_proposals"] == 2
        assert find_measurement(store, "disruptions_file") is not None
