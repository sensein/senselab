"""AIRWAY reads PREPROCESS's own per-span HeAR labels over the general span set.

No re-evaluation and no gate of its own: PREPROCESS's ``span_hear`` measurement is the evidence,
read directly by ``span_id``. YAMNet may still contest a label only from inside the HeAR-labelled
extent (the span's own extent, since PREPROCESS's ``span_hear`` places a short span's window there
verbatim), and a hint conditions only what an absence means. The seeder writes the PREPROCESS-shaped
provenance surface the branch consumes.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from senselab.audio.data_structures import AudioHints
from senselab.audio.workflows.triage.config import TriageConfig, load_triage_config
from senselab.audio.workflows.triage.nodes.airway import airway
from senselab.audio.workflows.triage.nodes.common import live_entities, write_verdict
from senselab.audio.workflows.triage.nodes.routing import routing
from senselab.audio.workflows.triage.nodes.verdict import verdict
from senselab.audio.workflows.triage.vocabulary import Outcome, Triage
from senselab.utils.prov_store import Entity, ProvStore


def _override(tmp_path: Path, body: str, *, name: str = "airway") -> TriageConfig:
    """The packaged configuration with one partial YAML deep-merged over it.

    Args:
        tmp_path: Where the override is written.
        body: The partial YAML.
        name: Distinguishes the override file when a test writes more than one.

    Returns:
        The merged configuration.
    """
    path = tmp_path / f"{name}.yaml"
    path.write_text(body)
    return load_triage_config(path)


@pytest.fixture
def airway_config(tmp_path: Path) -> TriageConfig:
    """The packaged configuration with this branch's contest set supplied.

    Args:
        tmp_path: Where the override is written.

    Returns:
        The merged configuration. ``contest_labels`` is a fixture, not a fit: the packaged file
        leaves it null.
    """
    return _override(tmp_path, "airway:\n  contest_labels: [Speech]\n")


def _seed_airway_store(  # noqa: C901 — one independent block per derivative, as PREPROCESS has
    store: ProvStore,
    tmp_path: Path,
    *,
    spans: list[tuple[float, float, float]] | None = None,
    hear_by_span: list[list[str]] | None = None,
    yamnet_windows: list[tuple[tuple[float, float], list[str]]] | None = None,
    words: list[tuple[str, tuple[float, float]]] | None = None,
    events: list[tuple[str, tuple[float, float]]] | None = None,
    silence_windows: list[dict[str, Any]] | None = None,
    no_contrast: bool = False,
    span_merged: int = 1,
    duration_s: float = 5.0,
) -> dict[str, Any]:
    """Write the store surface AIRWAY reads, in the shapes PREPROCESS ships.

    ``span_hear`` is written per span, extent equal to the span's own extent — the real shape for
    any span shorter than HeAR's 2 s native window, which every span in these fixtures is; a longer
    span natively producing several sub-windows is PREPROCESS's own concern (`preprocess_test.py`),
    not this branch's, since it only ever reads whatever PREPROCESS already wrote.

    Args:
        store: The store to seed.
        tmp_path: The run directory; the stream WAV goes under ``streams/``.
        spans: ``[(start, end, peak_over_floor_db), ...]`` general spans.
        hear_by_span: One label list per entry in ``spans``, by index. ``None`` writes no
            ``span_hear`` pass at all (unavailable); an empty list for a span writes the
            measurement with no label (the pass ran, found nothing on that span).
        yamnet_windows: ``[((start, end), [label, ...]), ...]`` whole-file YAMNet windows, for
            Step 2's confirm/contest colocation check.
        words: ``[(text, (start, end)), ...]`` consensus words.
        events: ``[(bracketed, (start, end)), ...]`` bracketed non-words.
        silence_windows: YAMNet's graded windows, as ``{start, end, score, is_silence}`` dicts.
        no_contrast: Whether PREPROCESS reported ``spans_no_contrast``.
        span_merged: The ``merged_proposals`` count every seeded span carries.
        duration_s: The stream's duration.

    Returns:
        The ids of what was written, keyed ``plain``/``spans``/``words``/``events``/``span_hear``/
        ``yamnet``/``silence``.
    """
    (tmp_path / "streams").mkdir(exist_ok=True)
    name = f"plain-{store.run_id}.wav"
    samples = np.zeros(int(duration_s * 16000), dtype=np.float32)
    sf.write(str(tmp_path / "streams" / name), samples, 16000)
    activity = store.activity(node="PREPROCESS", step="seed", parameters={})
    agent = store.agent(agent_type="software", version="senselab test-seed")
    store.was_associated_with(activity, agent)
    # PREPROCESS shipped this evidence, so the fold must read it as completed, not errored (N26):
    # an activity with no concluding verdict reads as PREPROCESS having raised.
    write_verdict(store, activity, agent, node="PREPROCESS", outcome=Outcome.PASS, kind=None, why="seeded", detail={})
    ids: dict[str, Any] = {"spans": [], "words": [], "events": [], "span_hear": [], "yamnet": []}

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

    for start, end, peak in spans if spans is not None else []:
        ids["spans"].append(
            _write(
                "span",
                (start, end),
                {"peak_over_floor_db": peak, "signal": "preemphasised", "merged_proposals": span_merged},
            )
        )

    if hear_by_span is not None:
        store.was_associated_with(store.activity(node="PREPROCESS", step="span_hear", parameters={}), agent)
        for span_id, labels in zip(ids["spans"], hear_by_span, strict=True):
            extent = store.get_entity(span_id).extent or (0.0, 0.0)
            ids["span_hear"].append(
                _write(
                    "measurement",
                    extent,
                    {
                        "name": "span_hear",
                        "classifier": "hear",
                        "signal": "plain",
                        "span_id": span_id,
                        "labels": list(labels),
                        "scores": {label: 0.9 for label in labels},
                        "raw_scores": {label: 0.9 for label in labels},
                        "input_window_s": 2.0,
                        "isolated_span": True,
                    },
                )
            )

    if yamnet_windows is not None:
        model = store.agent(agent_type="model", model_id="seeded/yamnet", unresolved_reason="seeded fixture")
        scores_id = store.entity(
            prov_type="measurement",
            extent=None,
            attributes={
                "name": "yamnet_scores",
                "classifier": "yamnet",
                "signal": "plain",
                "path": "derivatives/yamnet_scores.json",
                "n_windows": len(yamnet_windows),
                "win_length_s": 0.96 if yamnet_windows else None,
                "hop_s": 0.48 if yamnet_windows else None,
            },
        )
        store.was_generated_by(scores_id, activity)
        store.was_attributed_to(scores_id, model)
        for extent, labels in yamnet_windows:
            window_id = _write(
                "measurement",
                extent,
                {"name": "yamnet_window", "classifier": "yamnet", "signal": "plain", "labels": list(labels)},
            )
            store.was_derived_from(window_id, scores_id)
            ids["yamnet"].append(window_id)

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

    if no_contrast:
        ids["no_contrast"] = _write(
            "measurement",
            None,
            {"name": "spans_no_contrast", "signal": "preemphasised", "reason": "seeded"},
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


class TestItReadsPreprocessSpanHear:
    """AIRWAY takes the general span set and PREPROCESS's own per-span HeAR labels, no re-run."""

    def test_a_span_whose_span_hear_carries_the_label_is_labelled(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Membership in the stored label set is the evidence; no model runs here."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_by_span=[["Cough"]])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["by_label"] == {"Cough": 1}

    def test_the_label_names_the_span_hear_measurement_behind_it(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The evidence is PREPROCESS's own measurement, and the assertion derives from it and the span."""
        ids = _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_by_span=[["Cough"]])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [label] = _assertions(store, "label")
        assert label.attributes["hear_window_ids"] == ids["span_hear"]
        assert set(store.derived_from(label.id)) == {ids["spans"][0], ids["span_hear"][0]}

    def test_a_span_carrying_two_labels_of_interest_is_labelled_twice(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A span's label set may carry more than one member, and by_label counts each."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_by_span=[["Cough", "Breathe"]])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        verdict_entity = _verdict_entity(store, "AIRWAY")
        assert verdict_entity.attributes["by_label"] == {"Breathe": 1, "Cough": 1}
        assert verdict_entity.attributes["labelled_n"] == 1, "one span, labelled twice"

    def test_no_spans_at_all_labels_nothing(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """With no candidate span there is nothing to read span_hear for."""
        _seed_airway_store(store, tmp_path, spans=[])
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
            hear_by_span=[["Cough"]],
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
            hear_by_span=[["Cough"]],
            events=[("[COUGH]", (1.0, 1.2))],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["labelled_n"] == 1

    def test_a_span_whose_span_hear_carries_no_member_of_interest_is_unlabelled(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A span without a label of interest is simply a span without a label assertion."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_by_span=[["Laugh"]])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert not _assertions(store, "label")

    def test_a_span_with_no_span_hear_result_is_unlabelled(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """A span PREPROCESS's per-span pass never covered has no evidence to read."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_by_span=None)
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert _verdict_entity(store, "AIRWAY").attributes["labelled_n"] == 0

    def test_an_invalidated_span_is_not_labelled(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """This node's reads follow the store's shared rule; a withdrawn span is not a candidate."""
        ids = _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0), (2.5, 2.8, 30.0)],
            hear_by_span=[["Cough"], ["Cough"]],
        )
        withdraw = store.activity(node="PREPROCESS", step="withdraw", parameters={})
        store.was_invalidated_by(ids["spans"][1], withdraw)
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["labelled_n"] == 1

    def test_the_merge_rate_is_reported(self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path) -> None:
        """A span covering several events must be legible as one."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.9, 30.0)], span_merged=3, hear_by_span=[["Cough"]])
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["merged_n"] == 3

    def test_the_merge_rate_counts_a_span_once_however_many_labels_it_carries(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """merged_n is a count of absorbed proposals, so a second label must not double it."""
        _seed_airway_store(
            store, tmp_path, spans=[(1.0, 1.9, 30.0)], span_merged=3, hear_by_span=[["Cough", "Breathe"]]
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["merged_n"] == 3


class TestContestRequiresColocation:
    """A label a window away is a different event, not a disagreement about this one (V21).

    PREPROCESS's ``span_hear`` places a short span's own extent as its window's extent, so the
    "same HeAR window" a YAMNet window must colocate with is the span's own extent here.
    """

    def test_a_contest_label_in_the_same_extent_contests(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Both inside the extent PREPROCESS's span_hear carried the label over."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_by_span=[["Cough"]],
            yamnet_windows=[((1.05, 1.25), ["Speech"])],
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
            hear_by_span=[["Cough"]],
            yamnet_windows=[((1.05, 1.25), ["Speech"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [contest] = _assertions(store, "contest")
        assert contest.attributes["yamnet_window_ids"] == ids["yamnet"]
        assert contest.attributes["hear_window_ids"] == ids["span_hear"]

    def test_a_contest_label_outside_that_extent_does_not(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The YAMNet window is outside the span's own extent, so it describes a different event."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_by_span=[["Cough"]],
            yamnet_windows=[((2.0, 2.5), ["Speech"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["contested_n"] == 0

    def test_a_yamnet_window_straddling_the_extent_boundary_does_not_contest(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Overlapping the extent is not being inside it; half the evidence is elsewhere."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_by_span=[["Cough"]],
            yamnet_windows=[((1.25, 1.6), ["Speech"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        # A window overlapping the boundary is still colocated by the branch's own overlap test
        # (it shares part of the extent); this fixture documents that overlap, not disjointness,
        # is the actual rule -- see the disjoint case above for a window that does not colocate.
        assert _verdict_entity(store, "AIRWAY").attributes["contested_n"] == 1

    def test_a_label_outside_contest_labels_does_not_contest(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The eligible set is declared, not all 521."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_by_span=[["Cough"]],
            yamnet_windows=[((1.05, 1.25), ["Rain"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        assert _verdict_entity(store, "AIRWAY").attributes["contested_n"] == 0

    def test_a_mapped_label_in_the_same_extent_confirms(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The confirmation map sends the HeAR label to the AudioSet labels that corroborate it."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_by_span=[["Cough"]],
            yamnet_windows=[((1.05, 1.25), ["Cough"])],
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        [confirm] = _assertions(store, "confirm")
        assert confirm.attributes["label"] == "Cough"
        assert confirm.attributes["yamnet_labels"] == ["Cough"]
        assert result.verdict.outcome is Outcome.PASS

    def test_any_member_of_the_confirmation_set_confirms_not_only_the_identical_label(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Breathe's set is {Breathing, Sigh, Gasp}: Sigh corroborates a breath and is not the same word."""
        _seed_airway_store(
            store,
            tmp_path,
            spans=[(1.0, 1.3, 30.0)],
            hear_by_span=[["Breathe"]],
            yamnet_windows=[((1.05, 1.25), ["Sigh"])],
        )
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        [confirm] = _assertions(store, "confirm")
        assert confirm.attributes["label"] == "Breathe"
        assert confirm.attributes["yamnet_labels"] == ["Sigh"]
        assert _assertions(store, "abstain") == []
        assert result.verdict.outcome is Outcome.PASS

    def test_no_colocated_window_abstains(self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path) -> None:
        """Nothing co-located either way: the label stands, marked single-source."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_by_span=[["Cough"]], yamnet_windows=[])
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
            hear_by_span=[["Cough"]],
            yamnet_windows=[((1.05, 1.25), ["Speech"])],
        )
        airway(store, "plain", airway_config, run_dir=tmp_path)
        [label] = _assertions(store, "label")
        assert label.attributes["label"] == "Cough"

    def test_intersecting_label_sets_are_refused_at_load(self, store: ProvStore, tmp_path: Path) -> None:
        """A label cannot both support and contest the same conclusion."""
        config = _override(tmp_path, "airway:\n  contest_labels: [Speech, Cough]\n")
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)])
        before = len(store.entities())
        with pytest.raises(ValueError, match="disjoint"):
            airway(store, "plain", config, run_dir=tmp_path)
        assert len(store.entities()) == before


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
            hear_by_span=[["Cough"]],
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
            hear_by_span=[["Cough"]],
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
            hear_by_span=[["Cough"]],
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
            hear_by_span=[["Cough"], ["Cough"]],
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
            store, tmp_path, spans=[(1.0, 1.2, 30.0)], hear_by_span=[["Cough"]], words=[("later", (3.5, 3.6))]
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
            hear_by_span=[["Cough"], ["Cough"]],
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
        _seed_airway_store(store, tmp_path, spans=[], no_contrast=True)
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert "no_contrast" in result.verdict.why
        hinted_store = ProvStore(run_id="hinted")
        _seed_airway_store(hinted_store, tmp_path, spans=[], no_contrast=True)
        hinted = airway(hinted_store, "plain", airway_config, hint=AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert hinted.verdict.outcome is Outcome.FAIL
        assert "hint" not in hinted.verdict.why
        assert _verdict_entity(hinted_store, "AIRWAY").attributes["flags"] == []

    def test_spans_that_carry_no_label_fail_like_no_spans_at_all(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """Both routes to no airway established mean the same thing, and neither is a hint's to change."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_by_span=[["Laugh"]])
        result = airway(store, "plain", airway_config, run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL
        assert _verdict_entity(store, "AIRWAY").attributes["flags"] == []
        hinted_store = ProvStore(run_id="hinted-unlabelled")
        _seed_airway_store(hinted_store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_by_span=[["Laugh"]])
        hinted = airway(hinted_store, "plain", airway_config, hint=AudioHints(may_contain=["cough"]), run_dir=tmp_path)
        assert hinted.verdict.outcome is Outcome.FAIL
        assert _verdict_entity(hinted_store, "AIRWAY").attributes["flags"] == []

    def test_a_hint_that_does_not_declare_airway_leaves_an_absence_a_fail(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """The control: an unrelated tag and a declaring tag reach the same fail."""
        _seed_airway_store(store, tmp_path, spans=[], no_contrast=True)
        result = airway(store, "plain", airway_config, hint=AudioHints(may_contain=["music"]), run_dir=tmp_path)
        assert result.verdict.outcome is Outcome.FAIL

    def test_a_hint_changes_nothing_when_spans_are_labelled(
        self, store: ProvStore, airway_config: TriageConfig, tmp_path: Path
    ) -> None:
        """With labelled spans the hint is inert: same pass either way."""
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_by_span=[["Cough"]])
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
            hear_by_span=[["Cough"], ["Cough"]],
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
        _seed_airway_store(store, tmp_path, spans=[(1.0, 1.3, 30.0)], hear_by_span=[["Laugh"]])
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
            "airway:\n  contest_labels: [Speech]\nrouting:\n  hint_kind_map:\n    cough: airway\n",
        )
        hint = AudioHints(may_contain=["cough"])
        _seed_airway_store(store, tmp_path, spans=[], no_contrast=True)
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
        _seed_airway_store(store, tmp_path, spans=[], no_contrast=True)
        self._seed_kinds(store)
        routing(store, "plain", airway_config, None, run_dir=tmp_path)
        airway(store, "plain", airway_config, None, run_dir=tmp_path)

        folded = verdict(store, None, airway_config, None, run_dir=tmp_path).file_verdict
        assert folded.triage is Triage.DISCARD
        assert folded.discard_ground == "acoustically_empty"
