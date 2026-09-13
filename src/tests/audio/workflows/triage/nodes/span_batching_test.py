"""Per-span classification runs in one call, and consolidates into one file-level taxonomy."""

from typing import Any, Callable

import numpy as np
import pytest

from senselab.audio.data_structures import Audio
from senselab.audio.workflows.triage.nodes.preprocess import _classify_spans_in_batch
from senselab.audio.workflows.triage.nodes.taxonomy import _consolidate, _per_span_label_scores
from senselab.utils.prov_store import ProvStore


def _audio(seconds: float, value: float) -> Audio:
    """One flat mono clip, distinguishable by its own value.

    Args:
        seconds: Duration.
        value: The sample value, so a stub can tell one input from another.

    Returns:
        The clip at 16 kHz.
    """
    return Audio(waveform=np.full((1, int(seconds * 16000)), value, dtype="float32"), sampling_rate=16000)


def _scoring_stub(calls: list[int]) -> Callable[[list[Audio]], list[list[dict[str, Any]]]]:
    """A classifier that scores each input from its own first sample, recording each call's size.

    Args:
        calls: Appended to with the batch size of every call.

    Returns:
        The stub.
    """

    def classify(batch: list[Audio]) -> list[list[dict[str, Any]]]:
        calls.append(len(batch))
        return [
            [{"start": 0.0, "end": 0.96, "label_scores": [{"Speech": float(item.waveform[0, 0])}]}] for item in batch
        ]

    return classify


class TestTheBatchCall:
    """One call for every span, with each span's own outcome still its own."""

    def test_every_span_is_classified_in_a_single_call(self) -> None:
        """The whole point: 91 spans must not be 91 subprocess spawns."""
        calls: list[int] = []
        inputs = [_audio(1.0, 0.1), _audio(1.0, 0.2), _audio(1.0, 0.3)]

        results = _classify_spans_in_batch(inputs, _scoring_stub(calls))

        assert calls == [3], "the classifier must be called once with every span, not once per span"
        assert len(results) == 3
        assert all(failure is None for _, failure in results)

    def test_the_batch_result_matches_classifying_one_span_at_a_time(self) -> None:
        """Batching may not change a score, only the number of calls."""
        inputs = [_audio(1.0, 0.1), _audio(1.0, 0.2), _audio(1.0, 0.3)]

        batched = [windows for windows, _ in _classify_spans_in_batch(inputs, _scoring_stub([]))]
        looped = [_scoring_stub([])([item])[0] for item in inputs]

        assert batched == looped

    def test_no_inputs_makes_no_call(self) -> None:
        """A recording with no spans must not reach the model at all."""
        calls: list[int] = []

        assert _classify_spans_in_batch([], _scoring_stub(calls)) == []
        assert calls == []


class TestWhenTheBatchFails:
    """A batch that dies as a whole must not blame every span for it."""

    def test_it_falls_back_to_one_call_per_span(self) -> None:
        """Each span's own failure stays its own fact, rather than one misleading reason for all."""
        calls: list[int] = []

        def classify(batch: list[Audio]) -> list[list[dict[str, Any]]]:
            calls.append(len(batch))
            if len(batch) > 1:
                raise RuntimeError("out of memory")
            if abs(float(batch[0].waveform[0, 0]) - 0.2) < 1e-6:
                raise ValueError("this span alone is bad")
            return [[{"start": 0.0, "end": 0.96, "label_scores": [{"Speech": 0.5}]}]]

        results = _classify_spans_in_batch([_audio(1.0, 0.1), _audio(1.0, 0.2), _audio(1.0, 0.3)], classify)

        assert calls == [3, 1, 1, 1], "the batch is tried first, then each span on its own"
        assert results[0][1] is None and results[2][1] is None, "a good span survives its neighbour's failure"
        assert results[1][0] is None
        assert "ValueError" in str(results[1][1]), "the span's own error is named"
        assert "RuntimeError" in str(results[1][1]), "and so is the batch failure that forced the retry"

    def test_a_short_result_is_treated_as_a_batch_failure(self) -> None:
        """A classifier returning fewer results than inputs must not silently misalign spans."""
        calls: list[int] = []

        def classify(batch: list[Audio]) -> list[list[dict[str, Any]]]:
            calls.append(len(batch))
            if len(batch) > 1:
                return [[{"start": 0.0, "end": 0.96, "label_scores": [{"Speech": 0.5}]}]]
            return [[{"start": 0.0, "end": 0.96, "label_scores": [{"Speech": 0.5}]}]]

        results = _classify_spans_in_batch([_audio(1.0, 0.1), _audio(1.0, 0.2)], classify)

        assert calls == [2, 1, 1], "a mismatched length must fall back rather than align by luck"
        assert len(results) == 2
        assert all(failure is None for _, failure in results)


def _store_with_span_windows(name: str, per_span: dict[str, dict[str, float]]) -> ProvStore:
    """A store carrying one per-span classifier window per span.

    Args:
        name: ``"span_yamnet"`` or ``"span_hear"``.
        per_span: The scores to write, keyed by span id.

    Returns:
        The store.
    """
    store = ProvStore(run_id="test")
    activity = store.activity(node="PREPROCESS", step=name, parameters={})
    for index, (span_id, scores) in enumerate(per_span.items()):
        window = store.entity(
            prov_type="measurement",
            extent=(float(index), float(index) + 1.0),
            attributes={"name": name, "span_id": span_id, "raw_scores": dict(scores)},
        )
        store.was_generated_by(window, activity)
    return store


class TestTheConsolidation:
    """PREPROCESS provides labels; TAXONOMY consolidates them over the file."""

    def test_a_label_is_reduced_to_its_peak_median_and_reach(self) -> None:
        """The consensus is over spans, so each label carries how it behaved across them."""
        per_span = {"a": {"Speech": 0.9}, "b": {"Speech": 0.5}, "c": {"Speech": 0.1}}

        consolidated = _consolidate(per_span, floor=None)

        assert consolidated["Speech"]["peak"] == pytest.approx(0.9)
        assert consolidated["Speech"]["median"] == pytest.approx(0.5)
        assert consolidated["Speech"]["n_spans"] == pytest.approx(3.0)

    def test_the_floor_drops_a_label_no_span_ever_scored(self) -> None:
        """521 labels mean a silent span still contributes its four highest, all near zero."""
        per_span = {"a": {"Speech": 0.9, "Tick": 0.004}, "b": {"Speech": 0.8, "Tick": 0.001}}

        consolidated = _consolidate(per_span, floor=0.1)

        assert "Speech" in consolidated
        assert "Tick" not in consolidated, "a label whose peak never reaches the floor is not consolidated"

    def test_a_label_at_the_floor_is_kept(self) -> None:
        """The floor is a floor, not a strict inequality."""
        assert "Tick" in _consolidate({"a": {"Tick": 0.1}}, floor=0.1)

    def test_scores_are_read_from_the_store_per_span(self) -> None:
        """The consensus reads the model's own output, whatever the labelling threshold said."""
        store = _store_with_span_windows("span_yamnet", {"a": {"Speech": 0.9}, "b": {"Speech": 0.4, "Music": 0.2}})

        per_span = _per_span_label_scores(store, "span_yamnet")

        assert per_span == {"a": {"Speech": 0.9}, "b": {"Speech": 0.4, "Music": 0.2}}
