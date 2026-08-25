"""The one shared node-test fixture, pinned: every task after T1 builds its own seeder on this one.

The distinction these tests exist for is ``None`` against ``[]`` — nothing written against a
derivative that ran and found nothing. A consumer reads the first as ``unavailable`` and the second
as ``absent``, so a seeder that collapsed them would make six downstream test suites agree with a
node that was wrong.
"""

from typing import Any, Callable

from senselab.audio.workflows.triage.nodes.common import find_measurement, find_measurements, live_entities
from senselab.utils.prov_store import ProvStore


def _steps(store: ProvStore) -> list[str | None]:
    """Every PREPROCESS activity step the seeder recorded."""
    return [activity.step for activity in store.activities("PREPROCESS")]


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
        assert _steps(store).count("phonation_spans") == 1

    def test_every_documented_argument_tells_none_from_empty(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None]
    ) -> None:
        """The whole point of the fixture: `[]` leaves a mark on the store and `None` leaves none.

        ``spans`` and ``events`` used to collapse the two through an ``or []``, which made a seeded
        "the pass ran and proposed nothing" indistinguishable from "the pass never ran" — the exact
        distinction six downstream suites read as ``absent`` against ``unavailable``.
        """
        seed_preprocess_store(store, spans=[], events=[], ast_labels=[], hear_labels=[])
        assert _steps(store).count("spans") == 1
        assert _steps(store).count("consensus") == 1
        assert live_entities(store, "span") == []
        assert live_entities(store, "event") == []
        assert find_measurement(store, "ast_windows") is not None
        assert find_measurement(store, "hear_windows") is not None
        assert find_measurement(store, "consensus_transcript") is None

    def test_none_writes_no_marker_for_any_argument(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None]
    ) -> None:
        """The other half of the same distinction, asserted on the same store shape."""
        seed_preprocess_store(store)
        steps = _steps(store)
        assert "spans" not in steps
        assert "consensus" not in steps
        assert "phonation_spans" not in steps
        for name in ("ast_windows", "hear_windows", "ast_scores", "hear_scores", "yamnet_scores"):
            assert find_measurement(store, name) is None, name

    def test_scores_only_seeds_the_packaged_configs_own_state(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None]
    ) -> None:
        """Model ran, threshold null, so the scores exist and the fold does not.

        This is what the shipped configuration actually produces, and until the seeder could write it
        no downstream test could exercise the case its own node will meet first.
        """
        seed_preprocess_store(
            store,
            yamnet_labels=[["Speech"]],
            ast_labels=[["Speech"]],
            hear_labels=[["Cough"]],
            scores_only=("ast", "hear"),
        )
        for name in ("yamnet_scores", "ast_scores", "hear_scores"):
            assert find_measurement(store, name) is not None, name
        assert find_measurement(store, "yamnet_windows") is not None
        for name in ("ast_windows", "hear_windows"):
            assert find_measurement(store, name) is None, name
        assert find_measurements(store, "ast_window") == []

    def test_a_phonation_span_may_be_seeded_as_a_glide(
        self, store: ProvStore, seed_preprocess_store: Callable[..., None]
    ) -> None:
        """T5 and T6 both read the glide member, so the seeder has to be able to write one."""
        seed_preprocess_store(store, phonation=[(1.0, 2.0, "voiced"), (3.0, 3.4, "voiced", "glide")])
        spans = [e for e in live_entities(store, "span") if e.attributes.get("family") == "phonation"]
        assert [e.attributes["member"] for e in spans] == ["sustained", "glide"]
        assert spans[0].attributes["glide_direction"] is None
        assert spans[1].attributes["glide_direction"] == "rising"
        assert spans[1].attributes["glide_extent_cents"] > 0.0
        assert spans[1].attributes["offset_criterion"] == "monotonicity"

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
