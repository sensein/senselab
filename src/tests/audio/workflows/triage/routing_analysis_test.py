"""Tests for the routing-evidence analysis, over small synthetic provenance stores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from senselab.audio.workflows.triage.routing_analysis.detectors import (
    CONSOLIDATION_FLOOR,
    DETECTORS,
    GATE_CLOSED,
    Detector,
    detector_value,
)
from senselab.audio.workflows.triage.routing_analysis.families import declared_kinds, task_family, task_id_of
from senselab.audio.workflows.triage.routing_analysis.features import (
    RecordingFeatures,
    bracket_type,
    extract_features,
)
from senselab.audio.workflows.triage.routing_analysis.report import (
    REFERENCE_STANDARDS,
    Confusion,
    disagreements,
    label_prevalence,
    prevalence,
    score_detector,
    taxonomy_as_run,
    write_report,
)


def _entity(prov_type: str, entity_id: str, attributes: dict[str, Any], extent: list[float] | None = None) -> str:
    """One store entity line, with the key order the real store writes.

    Args:
        prov_type: The entity type.
        entity_id: Its id.
        attributes: Its attributes.
        extent: Its extent.

    Returns:
        The JSON line.
    """
    return json.dumps(
        {"attributes": attributes, "extent": extent, "id": entity_id, "prov_type": prov_type, "record": "entity"},
        sort_keys=True,
    )


def _relation(relation: str, source: str, target: str) -> str:
    """One store relation line.

    Args:
        relation: The PROV relation.
        source: The source id.
        target: The target id.

    Returns:
        The JSON line.
    """
    return json.dumps({"record": "relation", "relation": relation, "source": source, "target": target}, sort_keys=True)


def _write_store(path: Path, lines: list[str]) -> Path:
    """Write a synthetic store.

    Args:
        path: Where to write it.
        lines: The JSONL lines.

    Returns:
        The path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    return path


def _speech_store(path: Path) -> Path:
    """A store standing for a recording someone spoke in.

    Args:
        path: Where to write it.

    Returns:
        The path.
    """
    return _write_store(
        path,
        [
            _entity("stream", "stream-1", {"name": "recording"}, [0.0, 4.0]),
            _entity("word", "word-1", {"outcome": "agreement", "bracketed": False, "index": 0}, [0.1, 0.4]),
            _entity("word", "word-2", {"outcome": "variant", "bracketed": False, "index": 1}, [0.5, 0.8]),
            _entity(
                "word",
                "word-3",
                {"text": "[BREATH]", "outcome": "agreement", "bracketed": True, "index": 2},
                [0.9, 1.1],
            ),
            _entity(
                "word",
                "word-4",
                {"text": "[Throat Clearing]", "outcome": "agreement", "bracketed": True, "index": 3},
                [1.2, 1.5],
            ),
            _relation("wasInvalidatedBy", "word-4", "activity-1"),
            _entity("measurement", "measurement-1", {"name": "consensus_transcript", "signal": "plain"}),
            _entity(
                "measurement",
                "measurement-2",
                {"name": "residual", "signal": "residual", "energy_fraction": 0.01, "speech_present": True},
            ),
            _entity(
                "measurement",
                "measurement-3",
                {
                    "name": "yamnet_label_summary",
                    "classifier": "yamnet",
                    "labels": {"Speech": {"peak": 0.97, "median": 0.5}, "Cough": {"peak": 0.02, "median": 0.0}},
                },
            ),
            _entity(
                "measurement",
                "measurement-4",
                {
                    "name": "hear_label_summary",
                    "classifier": "hear",
                    "labels": {"Speech": {"peak": 0.8}, "Snore": {"peak": 0.3}, "Breathe": {"peak": 0.1}},
                },
            ),
            _entity(
                "measurement",
                "measurement-5",
                {"name": "span_hear", "raw_scores": {"Breathe": 0.4, "Cough": 0.05}, "span_id": "span-1"},
            ),
            _entity(
                "measurement",
                "measurement-6",
                {
                    "name": "consensus_taxonomy",
                    "consolidation_floor": CONSOLIDATION_FLOOR,
                    "labels": [{"label": "Speech", "peak_by_classifier": {"yamnet": 0.99, "hear": 0.79}}],
                },
            ),
            _entity("span", "span-1", {"measure": "amplitude"}, [0.0, 1.5]),
            _entity("span", "span-2", {"measure": "amplitude"}, [2.0, 2.5]),
            _entity("span", "span-3", {"measure": "continuity"}, [0.0, 0.6]),
            _entity("kind", "kind-1", {"kind": "speech", "state": "uncertain"}),
            _entity("kind", "kind-2", {"kind": "airway", "state": "uncertain"}),
            _entity("verdict", "verdict-1", {"node": "TAXONOMY", "outcome": "flag"}),
        ],
    )


def _silent_store(path: Path) -> Path:
    """A store standing for a recording with no agreed word and a loud residual.

    Args:
        path: Where to write it.

    Returns:
        The path.
    """
    return _write_store(
        path,
        [
            _entity("stream", "stream-1", {"name": "recording"}, [0.0, 6.0]),
            _entity("measurement", "measurement-1", {"name": "consensus_transcript", "signal": "plain"}),
            _entity(
                "measurement",
                "measurement-2",
                {"name": "residual", "signal": "residual", "energy_fraction": 0.6, "speech_present": False},
            ),
            _entity(
                "measurement",
                "measurement-3",
                {
                    "name": "yamnet_label_summary",
                    "classifier": "yamnet",
                    "labels": {"Speech": {"peak": 0.03}, "Cough": {"peak": 0.7}, "Chant": {"peak": 0.4}},
                },
            ),
            _entity(
                "measurement",
                "measurement-4",
                {"name": "hear_label_summary", "classifier": "hear", "labels": {"Cough": {"peak": 0.66}}},
            ),
            _entity("span", "span-1", {"measure": "amplitude"}, [0.5, 5.5]),
            _entity("kind", "kind-1", {"kind": "speech", "state": "uncertain"}),
        ],
    )


def _features(tmp_path: Path) -> tuple[RecordingFeatures, RecordingFeatures]:
    """One spoken and one silent synthetic recording, extracted.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The two records.
    """
    spoken = extract_features(
        _speech_store(tmp_path / "a" / "run" / "store.jsonl"),
        "sub-1_ses-1_task-rainbow-passage",
        str(tmp_path / "a"),
        "rainbow-passage",
        "rainbow-passage",
    )
    silent = extract_features(
        _silent_store(tmp_path / "b" / "run" / "store.jsonl"),
        "sub-2_ses-1_task-voluntary-cough",
        str(tmp_path / "b"),
        "voluntary-cough",
        "voluntary-cough",
    )
    return spoken, silent


def test_task_family_collapses_only_trailing_indices() -> None:
    """Repeat and Harvard-list indices collapse; a v2 marker does not."""
    assert task_family("harvard-sentences-list-10-3") == "harvard-sentences-list"
    assert task_family("cape-v-sentences-v2-4") == "cape-v-sentences-v2"
    assert task_family("free-speech-1") == "free-speech"
    assert task_family("respiration-and-cough-v2-threebreathsnose") == "respiration-and-cough-v2-threebreathsnose"
    assert task_family("glides-high-to-low") == "glides-high-to-low"


def test_task_id_reads_the_bids_entity() -> None:
    """The task id comes from the stem's trailing ``task-`` entity, lowercased."""
    assert task_id_of("sub-A_ses-B_task-Prolonged-Vowel") == "prolonged-vowel"
    assert task_id_of("sub-A_ses-B") == "unknown"


def test_declared_kinds_partition_the_families() -> None:
    """Airway carries a breath/cough sub-kind, because the two are detected by different evidence."""
    assert declared_kinds("respiration-and-cough-fivebreaths") == frozenset({"airway", "breath"})
    assert declared_kinds("respiration-and-cough-cough") == frozenset({"airway", "cough"})
    assert declared_kinds("respiration-and-cough-v2-hardcough") == frozenset({"airway", "cough"})
    assert declared_kinds("voluntary-cough") == frozenset({"airway", "cough"})
    assert declared_kinds("maximum-phonation-time") == frozenset({"voice", "sustained"})
    assert declared_kinds("harvard-sentences-list") == frozenset({"speech", "lexical_speech"})
    assert declared_kinds("diadochokinesis-ka") == frozenset({"speech"})


def test_extract_counts_live_words_by_outcome(tmp_path: Path) -> None:
    """Invalidated words are dropped and the remaining ones are counted by outcome."""
    spoken, _ = _features(tmp_path)
    assert spoken.words["total"] == 3
    assert spoken.words["agreement"] == 2
    assert spoken.words["variant"] == 1
    assert spoken.words["bracketed"] == 1
    assert spoken.words["lexical"] == 2
    assert spoken.words["agreement_lexical"] == 1
    assert spoken.consensus_present is True


def test_extract_types_the_bracketed_tokens_off_the_word_entities(tmp_path: Path) -> None:
    """The type is normalised, and an invalidated bracket contributes nothing."""
    spoken, _ = _features(tmp_path)
    assert spoken.bracketed_types == {"breath": 1}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("[BREATH]", "breath"),
        ("[Throat Clearing]", "throatclearing"),
        (" [throat-clearing] ", "throatclearing"),
        ("[um]", "um"),
        ("breath", None),
        ("[]", None),
        ("[ ]", None),
    ],
)
def test_bracket_type_normalises_one_token(text: str, expected: str | None) -> None:
    """Casing, spacing and inner punctuation do not split one bracket type into several."""
    assert bracket_type(text) == expected


def test_extract_reads_spans_streams_and_kinds(tmp_path: Path) -> None:
    """Span durations, the recording's extent and TAXONOMY's own states come back."""
    spoken, _ = _features(tmp_path)
    assert spoken.duration_s == pytest.approx(4.0)
    assert spoken.span_count["amplitude"] == 2
    assert spoken.span_longest_s["amplitude"] == pytest.approx(1.5)
    assert spoken.span_total_s["amplitude"] == pytest.approx(2.0)
    assert spoken.span_longest_s["gap"] == 0.0
    assert spoken.kind_state == {"speech": "uncertain", "airway": "uncertain", "voice": "missing"}
    assert spoken.verdicts["TAXONOMY"] == "flag"


def test_extract_keeps_tracked_peaks_per_stream_and_classifier(tmp_path: Path) -> None:
    """Peaks are keyed by stream, classifier and label, and untracked labels are dropped."""
    spoken, _ = _features(tmp_path)
    assert spoken.peaks["plain|yamnet|Speech"] == pytest.approx(0.97)
    assert spoken.peaks["plain|hear|Speech"] == pytest.approx(0.8)
    assert spoken.peaks["span|hear|Breathe"] == pytest.approx(0.4)
    assert spoken.peaks["consensus|yamnet|Speech"] == pytest.approx(0.99)
    assert sorted(spoken.classifier_streams) == ["plain|hear", "plain|yamnet"]
    assert spoken.residual["energy_fraction"] == pytest.approx(0.01)


def test_detector_value_returns_none_when_the_stream_never_ran(tmp_path: Path) -> None:
    """A classifier stream absent from the store excludes the recording, it does not score zero."""
    spoken, _ = _features(tmp_path)
    plain = Detector("d", "speech", ("peak", "plain", "yamnet", "speech"), "score", (0.5,))
    enhanced = Detector("d", "speech", ("peak", "enhanced", "yamnet", "speech"), "score", (0.5,))
    assert detector_value(spoken, plain) == pytest.approx(0.97)
    assert detector_value(spoken, enhanced) is None


def test_detector_value_rejects_an_unknown_source(tmp_path: Path) -> None:
    """A reader naming a source the dispatcher does not implement raises rather than scoring."""
    spoken, _ = _features(tmp_path)
    with pytest.raises(ValueError, match="unknown detector source"):
        detector_value(spoken, Detector("d", "speech", ("nowhere",), "score", (0.5,)))


def test_confusion_rates() -> None:
    """Sensitivity, specificity and Youden's J, and None where a margin is empty."""
    table = Confusion(tp=3, fp=1, tn=4, fn=2)
    assert table.sensitivity == pytest.approx(0.6)
    assert table.specificity == pytest.approx(0.8)
    assert table.youden == pytest.approx(0.4)
    assert Confusion(tp=0, fp=2, tn=3, fn=0).sensitivity is None
    assert Confusion(tp=1, fp=0, tn=0, fn=1).specificity is None


def test_score_detector_sweeps_and_marks_the_consolidation_floor(tmp_path: Path) -> None:
    """Every threshold gets its own 2x2, and 0.2 is marked as the configured floor."""
    records = list(_features(tmp_path))
    reference = next(r for r in REFERENCE_STANDARDS if r.name == "agreed_asr")
    detector = next(d for d in DETECTORS if d.name == "speech.yamnet_peak.plain")
    scored = score_detector(records, detector, reference)
    assert scored["n_scored"] == 2
    assert scored["n_reference_positive"] == 1
    rows = {row["threshold"]: row for row in scored["rows"]}
    assert rows[0.2]["marker"] == "taxonomy.consolidation_floor"
    assert (rows[0.5]["tp"], rows[0.5]["fp"], rows[0.5]["tn"], rows[0.5]["fn"]) == (1, 0, 1, 0)
    assert (rows[0.01]["tp"], rows[0.01]["fp"], rows[0.01]["tn"], rows[0.01]["fn"]) == (1, 1, 0, 0)


def test_score_detector_excludes_recordings_with_no_evidence(tmp_path: Path) -> None:
    """A detector reading a stream neither recording carries scores nothing."""
    records = list(_features(tmp_path))
    reference = next(r for r in REFERENCE_STANDARDS if r.name == "agreed_asr")
    detector = next(d for d in DETECTORS if d.name == "speech.yamnet_peak.residual")
    scored = score_detector(records, detector, reference)
    assert scored["n_scored"] == 0
    assert scored["n_unavailable"] == 2


def test_disagreements_are_enumerated_not_averaged(tmp_path: Path) -> None:
    """A spoken non-speech task and a silent speech task are each listed with their evidence."""
    spoken, silent = _features(tmp_path)
    spoken.family = "voluntary-cough"
    silent.family = "rainbow-passage"
    report = disagreements([spoken, silent])
    spoke = report["spoke_in_non_speech_task"]
    quiet = report["silent_in_speech_task"]
    assert spoke["voluntary-cough"]["n"] == 1
    assert spoke["voluntary-cough"]["cases"][0]["words"]["agreement"] == 2
    assert quiet["rainbow-passage"]["n"] == 1
    assert quiet["rainbow-passage"]["cases"][0]["residual_energy_fraction"] == pytest.approx(0.6)


def test_taxonomy_as_run_reports_the_branch_that_would_run(tmp_path: Path) -> None:
    """Anything but ``absent`` runs the branch, so an all-uncertain run has zero specificity."""
    records = list(_features(tmp_path))
    baseline = taxonomy_as_run(records)
    table = baseline["routing_would_run"]["agreed_asr"]
    assert table["sensitivity"] == pytest.approx(1.0)
    assert table["specificity"] == pytest.approx(0.0)
    assert baseline["kind_states"]["voice"]["missing"] == 2


def test_prevalence_counts_each_family_once(tmp_path: Path) -> None:
    """Every recording lands in exactly one family, and the counts sum to the corpus."""
    records = list(_features(tmp_path))
    report = prevalence(records)
    assert report["n_recordings"] == 2
    assert sum(row["n"] for row in report["families"].values()) == 2
    assert report["families"]["rainbow-passage"]["agreed_asr"] == 1
    assert report["families"]["voluntary-cough"]["declared_airway"] == 1


def test_write_report_is_idempotent(tmp_path: Path) -> None:
    """Two runs over the same records write byte-identical outputs."""
    records = list(_features(tmp_path))
    first = tmp_path / "out1"
    second = tmp_path / "out2"
    write_report(records, first)
    write_report(records, second)
    for name in (
        "sweeps.json",
        "sweeps_by_family.json",
        "label_prevalence.json",
        "prevalence.json",
        "disagreements.json",
        "index.json",
        "summary.md",
    ):
        assert (first / name).read_text() == (second / name).read_text()
    index = json.loads((first / "index.json").read_text())
    assert index["n_recordings"] == 2


def test_gated_detector_reads_the_primary_only_when_the_gate_fires(tmp_path: Path) -> None:
    """A closed gate reads below every threshold; an open one reads the primary unchanged.

    The spoken fixture's HeAR ``Snore`` peak is 0.3, which clears an airway gate at 0.2, so the
    gate here is set at 0.5.
    """
    spoken, silent = _features(tmp_path)
    gate = Detector(
        "d",
        "airway",
        ("gated", ("residual", "energy_fraction"), ("peak", "plain", "hear", "airway"), 0.5, "above"),
        "fraction",
        (0.1,),
    )
    assert detector_value(silent, gate) == pytest.approx(0.6)
    assert detector_value(spoken, gate) == pytest.approx(GATE_CLOSED)
    below = Detector(
        "d",
        "voice",
        ("gated", ("span_longest", "amplitude"), ("words", "agreement"), 1, "below"),
        "seconds",
        (1.0,),
    )
    assert detector_value(silent, below) == pytest.approx(5.0)
    assert detector_value(spoken, below) == pytest.approx(GATE_CLOSED)


def test_label_prevalence_counts_only_the_streams_that_ran(tmp_path: Path) -> None:
    """A family's label prevalence counts a recording only where that stream's summary exists."""
    records = list(_features(tmp_path))
    report = label_prevalence(records)
    assert report["floor"] == pytest.approx(0.2)
    snore = report["labels"]["plain|hear|Snore"]
    assert snore["rainbow-passage"] == {"n": 1, "n_over_floor": 1, "fraction": 1.0}
    assert snore["voluntary-cough"] == {"n": 1, "n_over_floor": 0, "fraction": 0.0}
    assert "residual|yamnet|Cough" not in report["labels"]


def _labelled_span_store(path: Path) -> Path:
    """A store whose spans carry per-span YAMNet and HeAR scores.

    Args:
        path: Where to write it.

    Returns:
        The path.
    """
    return _write_store(
        path,
        [
            _entity("stream", "stream-1", {"name": "recording"}, [0.0, 10.0]),
            _entity("span", "span-1", {"measure": "amplitude", "peak_over_floor_db": 40.0}, [0.0, 1.0]),
            _entity("span", "span-2", {"measure": "amplitude", "peak_over_floor_db": 30.0}, [2.0, 3.0]),
            _entity("span", "span-3", {"measure": "amplitude", "peak_over_floor_db": 55.0}, [4.0, 5.0]),
            _entity("span", "span-4", {"measure": "amplitude", "peak_over_floor_db": 99.0}, [6.0, 7.0]),
            _relation("wasInvalidatedBy", "span-4", "activity-1"),
            _entity(
                "measurement",
                "yam-1",
                {"name": "span_yamnet", "span_id": "span-1", "raw_scores": {"Cough": 0.8, "Speech": 0.1}},
                [0.0, 1.0],
            ),
            _entity(
                "measurement",
                "yam-2",
                {"name": "span_yamnet", "span_id": "span-2", "raw_scores": {"Cough": 0.6, "Speech": 0.2}},
                [2.0, 3.0],
            ),
            _entity(
                "measurement",
                "yam-3",
                {"name": "span_yamnet", "span_id": "span-3", "raw_scores": {"Silence": 0.9, "Cough": 0.1}},
                [4.0, 5.0],
            ),
            _entity(
                "measurement",
                "yam-4",
                {"name": "span_yamnet", "span_id": "span-4", "raw_scores": {"Cough": 0.95}},
                [6.0, 7.0],
            ),
            _entity(
                "measurement",
                "hear-1",
                {"name": "span_hear", "span_id": "span-1", "raw_scores": {"Cough": 0.3, "Breathe": 0.5}},
                [0.0, 0.5],
            ),
            _entity(
                "measurement",
                "hear-2",
                {"name": "span_hear", "span_id": "span-1", "raw_scores": {"Cough": 0.7, "Breathe": 0.2}},
                [0.5, 1.0],
            ),
        ],
    )


def _labelled(tmp_path: Path) -> RecordingFeatures:
    """The labelled-span fixture, extracted.

    Args:
        tmp_path: The test's temporary directory.

    Returns:
        The record.
    """
    return extract_features(
        _labelled_span_store(tmp_path / "c" / "run" / "store.jsonl"),
        "sub-3_ses-1_task-voluntary-cough",
        str(tmp_path / "c"),
        "voluntary-cough",
        "voluntary-cough",
    )


def test_span_label_stats_aggregate_the_spans_carrying_one_label(tmp_path: Path) -> None:
    """Two live spans whose best YAMNet label is Cough form one distribution together."""
    record = _labelled(tmp_path)
    assert record.span_label_stats["yamnet.Cough.span_count"] == pytest.approx(2.0)
    assert record.span_label_stats["yamnet.Cough.peak_over_floor_db_n"] == pytest.approx(2.0)
    assert record.span_label_stats["yamnet.Cough.peak_over_floor_db_min"] == pytest.approx(30.0)
    assert record.span_label_stats["yamnet.Cough.peak_over_floor_db_max"] == pytest.approx(40.0)
    assert record.span_label_stats["yamnet.Cough.peak_over_floor_db_mean"] == pytest.approx(35.0)


def test_span_label_stats_drop_a_span_whose_best_label_is_untracked(tmp_path: Path) -> None:
    """``Silence`` is outside TRACKED_LABELS, so the 55 dB span it labels contributes nothing."""
    record = _labelled(tmp_path)
    assert not [key for key in record.span_label_stats if key.startswith("yamnet.Silence.")]
    assert record.span_label_stats["yamnet.Cough.peak_over_floor_db_max"] == pytest.approx(40.0)


def test_span_label_stats_exclude_an_invalidated_span(tmp_path: Path) -> None:
    """The invalidated 99 dB Cough span is dropped exactly as ``live_spans`` drops it elsewhere."""
    record = _labelled(tmp_path)
    assert record.span_count["amplitude"] == 3
    assert record.span_label_stats["yamnet.Cough.span_count"] == pytest.approx(2.0)
    assert record.span_label_stats["yamnet.Cough.peak_over_floor_db_max"] == pytest.approx(40.0)


def test_span_label_stats_take_the_best_window_of_each_span(tmp_path: Path) -> None:
    """A span's label is the best score over every window its classifier placed on it."""
    record = _labelled(tmp_path)
    assert record.span_label_stats["hear.Cough.span_count"] == pytest.approx(1.0)
    assert record.span_label_stats["hear.Cough.peak_over_floor_db_max"] == pytest.approx(40.0)
    assert not [key for key in record.span_label_stats if key.startswith("hear.Breathe.")]
    assert record.peaks["span|hear|Breathe"] == pytest.approx(0.5)
    assert record.peaks["span|hear|Cough"] == pytest.approx(0.7)


def test_span_label_stats_are_empty_when_no_span_carries_a_label(tmp_path: Path) -> None:
    """A recording whose store holds no per-span classifier window costs no bytes."""
    _, silent = _features(tmp_path)
    assert silent.span_label_stats == {}


def test_label_conditioned_detectors_read_the_cough_spans(tmp_path: Path) -> None:
    """The three catalogued cough detectors read the YAMNet-Cough distribution, or nothing."""
    record = _labelled(tmp_path)
    _, silent = _features(tmp_path)
    names = (
        "cough.yamnet_cough_span_peak_over_floor_db_max",
        "cough.yamnet_cough_span_peak_over_floor_db_p75",
        "cough.yamnet_cough_span_peak_over_floor_db_p90",
    )
    detectors = {name: next(d for d in DETECTORS if d.name == name) for name in names}
    assert detector_value(record, detectors[names[0]]) == pytest.approx(40.0)
    assert detector_value(record, detectors[names[1]]) == pytest.approx(37.5)
    assert detector_value(record, detectors[names[2]]) == pytest.approx(39.0)
    for detector in detectors.values():
        assert detector_value(silent, detector) is None
        assert max(detector.thresholds) >= 55.0
