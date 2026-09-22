"""The compact recording vectors: the encoding contract, and null where nothing was measured."""

import json
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile

from senselab.audio.workflows.triage import recording_vectors as rv
from senselab.audio.workflows.triage.nodes import figure as figure_module
from senselab.audio.workflows.triage.vocabulary import BRANCHES, Release, Triage

MEASURE_STATS = Path(__file__).parents[5] / "specs" / "20260817-triage-workflow-dag" / "measure-stats-20260921.json"


# ------------------------------------------------------------------------------ a synthetic store


def _entity(entity_id: str, prov_type: str, extent: Any, **attributes: Any) -> str:  # noqa: ANN401 -- store values
    return json.dumps(
        {"record": "entity", "id": entity_id, "prov_type": prov_type, "extent": extent, "attributes": attributes}
    )


def _relation(relation: str, source: str, target: str) -> str:
    return json.dumps({"record": "relation", "relation": relation, "source": source, "target": target})


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def build_recording(
    root: Path,
    stem: str = "sub-a1_ses-b2_task-diadochokinesis-pa",
    timestamp: str = "20260920-030735",
    duration_s: float = 4.0,
    sampling_rate: int = 16000,
    *,
    with_preprocess: bool = True,
    with_transcript: bool = True,
    with_pii_scan: bool = True,
    pii_findings: int = 1,
    with_routing: bool = True,
    with_fold: bool = True,
    with_derivatives: bool = True,
    with_stream_file: bool = True,
    measurements: tuple[tuple[str, Any], ...] = (("voiced_duration_s", 2.5),),
) -> Path:
    """A run directory carrying only what one test needs.

    Args:
        root: Where the ``sub-*/ses-*/<stem>_<timestamp>`` tree goes.
        stem: The BIDS stem.
        timestamp: The launch timestamp the run directory's name carries.
        duration_s: The recording's duration.
        sampling_rate: Its sampling rate.
        with_preprocess: Whether PREPROCESS's verdict and its spans exist.
        with_transcript: Whether a ``consensus_transcript`` measurement and words exist.
        with_pii_scan: Whether SPEECH's report says the PII scan ran.
        pii_findings: How many ``pii`` entities to write.
        with_routing: Whether ``branch_decision`` entities exist.
        with_fold: Whether VERDICT's fold exists.
        with_derivatives: Whether the envelope and continuity npz files exist.
        with_stream_file: Whether the conditioned flac exists.
        measurements: ``(name, value)`` pairs to write as measurement entities.

    Returns:
        The run root, the directory holding ``run/``.
    """
    participant, session = stem.split("_")[0], stem.split("_")[1]
    run_root = root / participant / session / f"{stem}_{timestamp}"
    run_dir = run_root / "run"
    (run_dir / "streams").mkdir(parents=True, exist_ok=True)
    (run_dir / "derivatives").mkdir(parents=True, exist_ok=True)
    samples = int(duration_s * sampling_rate)
    lines: list[str] = []

    lines.append(_entity("stream-rec", "stream", [0.0, duration_s], name="recording", sampling_rate=sampling_rate))
    lines.append(
        _entity(
            "stream-pre",
            "stream",
            [0.0, duration_s],
            name="preemphasised",
            path="streams/preemphasised.flac",
            sampling_rate=sampling_rate,
        )
    )
    if with_stream_file:
        tone = 0.5 * np.sin(2 * np.pi * 220 * np.arange(samples) / sampling_rate)
        soundfile.write(run_dir / "streams" / "preemphasised.flac", tone.astype("float32"), sampling_rate)

    if with_derivatives:
        _write_npz(
            run_dir / "derivatives" / "energy_envelope.npz",
            envelope_dbfs=np.linspace(-80.0, -10.0, samples),
            floor_dbfs=np.full(samples, -73.25),
        )
        _write_npz(run_dir / "derivatives" / "continuity_trace.npz", continuity=np.linspace(0.0, 1.0, samples))
        lines.append(
            _entity(
                "measurement-env",
                "measurement",
                None,
                name="energy_envelope",
                path="derivatives/energy_envelope.npz",
                signal="preemphasised",
            )
        )
        lines.append(
            _entity(
                "measurement-cont",
                "measurement",
                None,
                name="continuity_trace",
                path="derivatives/continuity_trace.npz",
                signal="preemphasised",
            )
        )

    for index, (name, value) in enumerate(measurements):
        lines.append(_entity(f"measurement-m{index}", "measurement", None, name=name, value=value))

    if with_preprocess:
        lines.append(_entity("verdict-pre", "verdict", None, node="PREPROCESS", outcome="pass", absent={}))
        lines.append(_entity("span-0", "span", [0.5, 1.5], measure="amplitude", signal="preemphasised"))
        lines.append(_entity("span-1", "span", [2.0, 3.0], measure="amplitude", signal="normalized"))
        lines.append(
            _entity(
                "measurement-y0",
                "measurement",
                [0.5, 1.5],
                name="span_yamnet",
                classifier="yamnet",
                span_id="span-0",
                signal="plain",
                raw_scores={"Speech": 0.75, "Snore": 0.10},
            )
        )
        lines.append(
            _entity(
                "assertion-sq0", "assertion", [0.5, 1.5], verb="measure", name="squim", stoi=0.5, pesq=2.75, si_sdr=10.0
            )
        )
        lines.append(_relation("wasDerivedFrom", "assertion-sq0", "span-0"))

    if with_transcript:
        lines.append(_entity("measurement-tr", "measurement", None, name="consensus_transcript", n_words=2))
        lines.append(_entity("word-0", "word", [0.5, 0.9], index=0, text="Hello", outcome="agreement"))
        lines.append(_entity("word-1", "word", [1.0, 1.4], index=1, text="Jane", outcome="variant"))

    pii_block = {"n": pii_findings, "categories": ["PERSON"], "failed": [], "missing": []} if with_pii_scan else None
    lines.append(
        _entity(
            "branch_report-speech",
            "branch_report",
            None,
            node="SPEECH",
            kind="speech",
            conformance=True,
            conformance_of="task",
            deviations=[],
            unmeasured=[],
            **({"pii": pii_block} if pii_block is not None else {}),
        )
    )
    for index in range(pii_findings if with_pii_scan else 0):
        lines.append(_entity(f"pii-{index}", "pii", [1.0, 1.4], category="PERSON", source="presidio"))

    if with_routing:
        for branch in BRANCHES:
            lines.append(
                _entity(
                    f"branch_decision-{branch}",
                    "branch_decision",
                    None,
                    branch=branch,
                    will_run=branch == "SPEECH",
                    route_state="routed" if branch == "SPEECH" else "declined",
                    flag_gates=[],
                )
            )
        lines.append(_entity("span-speech", "span", [0.4, 3.2], family="speech", role="speech_run_12"))
        lines.append(_entity("span-redact", "span", [1.0, 1.4], name="redaction", category="PERSON"))
        lines.append(_relation("wasDerivedFrom", "span-speech", "span-0"))

    if with_fold:
        lines.append(
            _entity(
                "verdict-file",
                "verdict",
                None,
                node="VERDICT",
                outcome=Triage.FLAG.value,
                triage=Triage.FLAG.value,
                release=Release.WITHHELD.value,
                discard_ground=None,
                declared_family="diadochokinesis-pa",
                conformance={"SPEECH": True, "VOICE": "UNDETERMINED"},
                routes={"SPEECH": "routed", "VOICE": "routed", "AIRWAY": "declined"},
                route_state="routed",
                reasons=[
                    {"node": "ADMIT", "outcome": "pass", "kind": None, "why": "decodes"},
                    {"node": "SPEECH", "outcome": "flag", "kind": "speech", "why": "pii found"},
                ],
            )
        )
    (run_dir / "store.jsonl").write_text("\n".join(lines) + "\n")
    return run_root


@pytest.fixture
def one_row(tmp_path: Path) -> dict[str, Any]:
    """The row a fully populated synthetic recording yields.

    Args:
        tmp_path: pytest's temporary directory.

    Returns:
        The extracted row.
    """
    run_root = build_recording(tmp_path)
    row = rv.extract(run_root, tmp_path)
    assert row is not None
    return row


# ------------------------------------------------------------------------------- the quantisers


@pytest.mark.parametrize("seconds", [0.0, 0.5, 3.9999, 30.0])
def test_time_round_trips_within_half_a_millisecond(seconds: float) -> None:
    """A time survives the uint16 round trip to well under what the graph measures."""
    duration = 30.0
    code = rv.quantise_time(seconds, duration)
    assert 0 <= code <= rv.TIME_SCALE
    assert abs(rv.dequantise_time(code, duration) - min(seconds, duration)) < 0.0005


@pytest.mark.parametrize(("value", "low", "high"), [(0.5, 0.0, 1.0), (2.75, 1.0, 4.5), (-10.0, -10.0, 30.0)])
def test_value_round_trips_within_one_step(value: float, low: float, high: float) -> None:
    """A value survives the uint8 round trip to within half a quantiser step."""
    code = rv.quantise_value(value, low, high)
    assert 0 <= code <= 255
    assert abs(rv.dequantise_value(code, low, high) - value) <= (high - low) / 255 / 2 + 1e-9


def test_out_of_range_values_clamp_rather_than_wrap() -> None:
    """A reading outside the declared range saturates at the range's end."""
    assert rv.quantise_value(-500.0, *rv.ENVELOPE_DBFS_RANGE) == 0
    assert rv.quantise_value(500.0, *rv.ENVELOPE_DBFS_RANGE) == 255
    assert rv.quantise_time(-1.0, 4.0) == 0
    assert rv.quantise_time(99.0, 4.0) == rv.TIME_SCALE


def test_non_finite_values_do_not_escape_the_quantisers() -> None:
    """NaN and infinity never become a byte outside the range."""
    assert rv.quantise_value(math.nan, 0.0, 1.0) == 0
    assert rv.quantise_time(math.inf, 4.0) == 0


# -------------------------------------------------------------------------------- the byte layout


@pytest.mark.parametrize(
    ("layout", "records"),
    [
        (rv.SPANS_LAYOUT, [(0, 1, 65535), (4, 100, 200)]),
        (rv.SPAN_LABELS_LAYOUT, [(0, 1, 191), (65535, 255, 0)]),
        (rv.SPAN_SQUIM_LAYOUT, [(3, 127, 200, 51)]),
        (rv.ASR_WORDS_LAYOUT, [(1092, 1966, 0), (2184, 3058, 1)]),
        (rv.PII_MARKS_LAYOUT, [(2184, 3058)]),
        (rv.BRANCH_LANES_LAYOUT, [(1, 874, 6990)]),
    ],
)
def test_records_round_trip_through_the_declared_layout(layout: str, records: list[tuple[int, ...]]) -> None:
    """Every block packs and unpacks at its declared width, little-endian."""
    blob = rv.encode_records(records, layout)
    assert len(blob) == len(records) * struct.calcsize("<" + layout)
    assert rv.decode_records(blob, layout) == records


def test_a_blob_that_is_not_a_whole_number_of_records_is_refused() -> None:
    """A truncated block raises rather than silently decoding one record short."""
    with pytest.raises(ValueError, match="whole number"):
        rv.decode_records(b"\x00\x01\x02", rv.SPANS_LAYOUT)


def test_an_empty_block_and_an_absent_block_both_decode_to_no_records() -> None:
    """Empty is zero records; the distinction from absent lives in the column, not the bytes."""
    assert rv.decode_records(b"", rv.SPANS_LAYOUT) == []
    assert rv.decode_records(None, rv.SPANS_LAYOUT) == []


def test_the_worked_example_in_the_schema_document_decodes_as_written() -> None:
    """One span, row ``A``, 1.0 s to 2.0 s of a 4 s recording, is the nine bytes the doc quotes."""
    blob = rv.encode_records([(rv.SPAN_ROWS.index("A"), rv.quantise_time(1.0, 4.0), rv.quantise_time(2.0, 4.0))], "BHH")
    assert blob == bytes([0x02, 0x00, 0x40, 0x00, 0x80])
    (row, t0, t1) = rv.decode_records(blob, rv.SPANS_LAYOUT)[0]
    assert rv.SPAN_ROWS[row] == "A"
    assert rv.dequantise_time(t0, 4.0) == pytest.approx(1.0, abs=1e-4)
    assert rv.dequantise_time(t1, 4.0) == pytest.approx(2.0, abs=1e-4)


# ------------------------------------------------------------------------------------ the traces


def test_a_trace_is_always_exactly_the_declared_number_of_points() -> None:
    """Short and long inputs both decimate to ``TRACE_POINTS``."""
    for length in (3, 256, 100311):
        blob = rv.encode_trace(np.linspace(-80.0, -10.0, length), *rv.ENVELOPE_DBFS_RANGE, how="max")
        assert blob is not None
        assert len(blob) == rv.TRACE_POINTS


def test_a_decimated_ramp_decodes_back_to_a_ramp() -> None:
    """The shape survives: the decoded trace is monotone and hits both ends."""
    blob = rv.encode_trace(np.linspace(-80.0, -10.0, 10000), *rv.ENVELOPE_DBFS_RANGE, how="max")
    assert blob is not None
    decoded = [rv.dequantise_value(b, *rv.ENVELOPE_DBFS_RANGE) for b in blob]
    assert decoded == sorted(decoded)
    assert decoded[0] == pytest.approx(-80.0, abs=0.7)
    assert decoded[-1] == pytest.approx(-10.0, abs=0.7)


def test_an_absent_trace_is_none_and_an_all_nan_trace_is_none() -> None:
    """Absent stays absent; a trace with nothing finite in it is absent too, never a row of zeros."""
    assert rv.encode_trace(None, 0.0, 1.0, how="max") is None
    assert rv.encode_trace(np.full(100, np.nan), 0.0, 1.0, how="max") is None


def test_the_waveform_carries_a_min_and_a_max_per_bucket_scaled_by_its_own_peak() -> None:
    """A sine decodes to a symmetric envelope at the reported peak."""
    tone = 0.25 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000)
    blob, peak = rv.encode_waveform(tone)
    assert blob is not None and peak == pytest.approx(0.25, abs=1e-3)
    assert len(blob) == rv.TRACE_POINTS * 2
    lows = [rv.dequantise_value(blob[i], -peak, peak) for i in range(0, len(blob), 2)]
    highs = [rv.dequantise_value(blob[i], -peak, peak) for i in range(1, len(blob), 2)]
    assert all(low <= high for low, high in zip(lows, highs))
    assert min(lows) == pytest.approx(-0.25, abs=0.01)
    assert max(highs) == pytest.approx(0.25, abs=0.01)


# --------------------------------------------------------------------------------- the extraction


def test_the_three_identity_columns_come_first_and_in_the_directed_order() -> None:
    """``participant``, ``task``, ``verdict`` — owner-directed, and the schema leads with them."""
    assert [field.name for field in rv.schema()][:3] == ["participant", "task", "verdict"]


def test_identity_is_read_from_the_stem_and_the_verdict_from_the_fold(one_row: dict[str, Any]) -> None:
    """Participant and task come from the stem; the verdict is the fold's triage."""
    assert one_row["participant"] == "sub-a1"
    assert one_row["session"] == "ses-b2"
    assert one_row["task"] == "diadochokinesis-pa"
    assert one_row["verdict"] == Triage.FLAG.value
    assert one_row["release"] == Release.WITHHELD.value
    assert one_row["declared_family"] == "diadochokinesis-pa"


def test_the_decision_record_the_html_asked_for_is_present(one_row: dict[str, Any]) -> None:
    """Conformance per branch, the flag count, the grounds, duration and release."""
    assert one_row["conformance_speech"] == "true"
    assert one_row["conformance_voice"] == "undetermined"
    assert one_row["conformance_airway"] is None
    assert one_row["route_airway"] == "declined"
    assert one_row["flags_n"] == 1
    assert one_row["flag_nodes"] == ["SPEECH"]
    assert one_row["grounds"] is None
    assert one_row["duration_s"] == pytest.approx(4.0)
    assert one_row["time_scale_s"] == pytest.approx(4.0)


def test_the_spans_block_carries_one_record_per_live_general_span(one_row: dict[str, Any]) -> None:
    """Two spans, on the ``E`` and ``S`` rows, at their quantised extents."""
    records = rv.decode_records(one_row["spans"], rv.SPANS_LAYOUT)
    assert [rv.SPAN_ROWS[r] for r, _, _ in records] == ["E", "S"]
    assert rv.dequantise_time(records[0][1], 4.0) == pytest.approx(0.5, abs=1e-3)
    assert rv.dequantise_time(records[1][2], 4.0) == pytest.approx(3.0, abs=1e-3)


def test_a_span_label_names_its_classifier_and_its_strongest_label(one_row: dict[str, Any]) -> None:
    """The top label per span per classifier, with its score, and the name in the parallel column."""
    records = rv.decode_records(one_row["span_labels"], rv.SPAN_LABELS_LAYOUT)
    assert len(records) == 1
    span_index, classifier, score = records[0]
    assert span_index == 0
    assert rv.CLASSIFIERS[classifier] == "yamnet"
    assert rv.dequantise_value(score, *rv.SCORE_RANGE) == pytest.approx(0.75, abs=0.01)
    assert one_row["span_label_name"] == ["Speech"]


def test_squim_is_bound_to_its_span_through_the_derivation_edge(one_row: dict[str, Any]) -> None:
    """The three readings decode at their own declared ranges."""
    records = rv.decode_records(one_row["span_squim"], rv.SPAN_SQUIM_LAYOUT)
    assert len(records) == 1
    span_index, stoi, pesq, si_sdr = records[0]
    assert span_index == 0
    assert rv.dequantise_value(stoi, 0.0, 1.0) == pytest.approx(0.5, abs=0.01)
    assert rv.dequantise_value(pesq, 1.0, 4.5) == pytest.approx(2.75, abs=0.01)
    assert rv.dequantise_value(si_sdr, -10.0, 30.0) == pytest.approx(10.0, abs=0.1)


def test_the_asr_lane_carries_extents_and_the_words_in_index_order(one_row: dict[str, Any]) -> None:
    """Owner-directed 2026-09-22: the words stay, alongside their extents and outcomes."""
    records = rv.decode_records(one_row["asr_words"], rv.ASR_WORDS_LAYOUT)
    assert len(records) == 2 == len(one_row["asr_word_text"])
    assert one_row["asr_word_text"] == ["Hello", "Jane"]
    assert [rv.WORD_OUTCOMES[outcome] for _, _, outcome in records] == ["agreement", "variant"]
    assert rv.dequantise_time(records[0][0], 4.0) == pytest.approx(0.5, abs=1e-3)


def test_every_pii_finding_is_marked_with_its_category_and_extent(one_row: dict[str, Any]) -> None:
    """Marked rather than omitted, so the reader can see which category fired and where."""
    records = rv.decode_records(one_row["pii_marks"], rv.PII_MARKS_LAYOUT)
    assert len(records) == 1 == len(one_row["pii_category"])
    assert one_row["pii_category"] == ["PERSON"]
    assert rv.dequantise_time(records[0][0], 4.0) == pytest.approx(1.0, abs=1e-3)
    assert one_row["pii_findings_n"] == 1


def test_branch_lanes_carry_the_lane_the_role_kind_and_the_rectangle(one_row: dict[str, Any]) -> None:
    """A proposal's instance index is stripped; a redaction lands in the REDACT lane."""
    records = rv.decode_records(one_row["branch_lanes"], rv.BRANCH_LANES_LAYOUT)
    assert [rv.LANES[lane] for lane, _, _ in records] == ["SPEECH", "REDACT"]
    assert one_row["branch_lane_role"] == ["speech_run", "PERSON"]


def test_the_waveform_and_the_traces_are_the_declared_widths(one_row: dict[str, Any]) -> None:
    """256 envelope bytes, 256 continuity bytes, 512 waveform bytes, one floor scalar."""
    assert len(one_row["env_dbfs"]) == rv.TRACE_POINTS
    assert len(one_row["continuity"]) == rv.TRACE_POINTS
    assert len(one_row["wave_minmax"]) == rv.TRACE_POINTS * 2
    assert one_row["floor_dbfs"] == pytest.approx(-73.25)


# -------------------------------------------------------------------- null, never zero


def test_every_measurement_has_a_column_and_a_reading_count() -> None:
    """The 29 names the corpus scan enumerated, each with the count that explains its null."""
    names = set(json.loads(MEASURE_STATS.read_text()))
    assert set(rv.MEASUREMENTS) == names
    columns = {field.name for field in rv.schema()}
    for name in names:
        assert f"m_{name}" in columns
        assert f"m_{name}_n" in columns


def test_a_measure_absent_from_the_store_is_null_and_its_count_is_zero(one_row: dict[str, Any]) -> None:
    """The invariant the whole file turns on: null means not measured, and ``_n`` says so."""
    for name in rv.MEASUREMENTS:
        value, count = one_row[f"m_{name}"], one_row[f"m_{name}_n"]
        assert (value is None) == (count == 0), name
    assert one_row["m_voiced_duration_s"] == pytest.approx(2.5)
    assert one_row["m_voiced_duration_s_n"] == 1
    assert one_row["m_glide_extent_semitones"] is None


def test_a_measure_read_as_zero_is_zero_and_not_null(tmp_path: Path) -> None:
    """A reading of 0.0 is a measurement, and must not read as an absence."""
    run_root = build_recording(tmp_path, measurements=(("verbatim_overlap_fraction", 0.0),))
    row = rv.extract(run_root, tmp_path)
    assert row is not None
    assert row["m_verbatim_overlap_fraction"] == 0.0
    assert row["m_verbatim_overlap_fraction_n"] == 1


def test_repeated_readings_are_averaged_and_their_count_is_reported(tmp_path: Path) -> None:
    """Four of the numeric measures are written per span; the column says how many it folded."""
    run_root = build_recording(
        tmp_path,
        measurements=(("breath_peak_over_floor_db", 20.0), ("breath_peak_over_floor_db", 30.0)),
    )
    row = rv.extract(run_root, tmp_path)
    assert row is not None
    assert row["m_breath_peak_over_floor_db"] == pytest.approx(25.0)
    assert row["m_breath_peak_over_floor_db_n"] == 2


def test_a_recording_never_scanned_for_pii_is_null_not_clean(tmp_path: Path) -> None:
    """No scan is not a clean scan: both the count and the marks are null."""
    run_root = build_recording(tmp_path, with_pii_scan=False, pii_findings=0)
    row = rv.extract(run_root, tmp_path)
    assert row is not None
    assert row["pii_findings_n"] is None
    assert row["pii_marks"] is None
    assert row["pii_category"] is None


def test_a_recording_scanned_and_found_clean_is_zero_not_null(tmp_path: Path) -> None:
    """A scan that found nothing is an empty block and a count of zero."""
    run_root = build_recording(tmp_path, with_pii_scan=True, pii_findings=0)
    row = rv.extract(run_root, tmp_path)
    assert row is not None
    assert row["pii_findings_n"] == 0
    assert row["pii_marks"] == b""
    assert row["pii_category"] == []


def test_a_block_whose_producer_did_not_run_is_null_not_empty(tmp_path: Path) -> None:
    """Absent producer, absent block — for spans, the ASR lane and the branch lanes alike."""
    run_root = build_recording(
        tmp_path, with_preprocess=False, with_transcript=False, with_routing=False, with_derivatives=False
    )
    row = rv.extract(run_root, tmp_path)
    assert row is not None
    for column in ("spans", "span_labels", "span_squim", "asr_words", "branch_lanes", "env_dbfs", "continuity"):
        assert row[column] is None, column
    assert row["span_label_name"] is None
    assert row["asr_word_text"] is None
    assert row["floor_dbfs"] is None


def test_an_undecodable_stream_leaves_the_waveform_null(tmp_path: Path) -> None:
    """The waveform is null when no stream decodes, never a flat line of zeros."""
    run_root = build_recording(tmp_path, with_stream_file=False)
    row = rv.extract(run_root, tmp_path)
    assert row is not None
    assert row["wave_minmax"] is None
    assert row["wave_peak"] is None


def test_a_store_with_no_fold_yields_no_row(tmp_path: Path) -> None:
    """A slice still running is skipped, not written as a row of nulls."""
    run_root = build_recording(tmp_path, with_fold=False)
    assert rv.extract(run_root, tmp_path) is None


def test_an_invalidated_entity_is_not_read(tmp_path: Path) -> None:
    """``wasInvalidatedBy`` removes an entity from every block, as it does from the figure."""
    run_root = build_recording(tmp_path)
    store = run_root / "run" / "store.jsonl"
    store.write_text(store.read_text() + _relation("wasInvalidatedBy", "span-1", "activity-x") + "\n")
    row = rv.extract(run_root, tmp_path)
    assert row is not None
    assert [rv.SPAN_ROWS[r] for r, _, _ in rv.decode_records(row["spans"], rv.SPANS_LAYOUT)] == ["E"]


# ------------------------------------------------------------------------------------ the table


def test_a_missing_key_becomes_null_and_not_a_default(one_row: dict[str, Any]) -> None:
    """``to_table`` never substitutes a zero for a key a row does not carry."""
    stripped = {k: v for k, v in one_row.items() if k not in {"m_voiced_duration_s", "flags_n", "pii_findings_n"}}
    table = rv.to_table([stripped])
    assert table.column("m_voiced_duration_s")[0].as_py() is None
    assert table.column("flags_n")[0].as_py() is None
    assert table.column("pii_findings_n")[0].as_py() is None


def test_the_table_round_trips_every_block(one_row: dict[str, Any]) -> None:
    """What arrow stores is byte-identical to what the encoder produced."""
    table = rv.to_table([one_row])
    for column in ("wave_minmax", "env_dbfs", "continuity", "spans", "span_labels", "span_squim", "asr_words"):
        assert table.column(column)[0].as_py() == one_row[column], column
    assert table.column("asr_word_text")[0].as_py() == ["Hello", "Jane"]


def test_shards_written_from_the_same_schema_concatenate(tmp_path: Path) -> None:
    """Two shards' tables share a schema, so the merge is a concat and not a cast."""
    row = rv.extract(build_recording(tmp_path), tmp_path)
    assert row is not None
    first, second = rv.to_table([row]), rv.to_table([])
    assert first.schema == second.schema == rv.schema()


# ------------------------------------------------------------------------------------- the scan


def test_the_scan_shards_by_stem_and_covers_every_recording_exactly_once(tmp_path: Path) -> None:
    """Content-addressed sharding, so a tree that grows between shards does not shift a row."""
    stems = [f"sub-{i:02d}_ses-b_task-t{i}" for i in range(20)]
    for stem in stems:
        build_recording(tmp_path, stem=stem)
    seen: list[str] = []
    for index in range(4):
        rows, report = rv.scan(tmp_path, index, 4)
        seen += [row["stem"] for row in rows]
        assert report.written == len(rows)
    assert sorted(seen) == sorted(stems)


def test_a_growing_tree_does_not_move_a_recording_between_shards(tmp_path: Path) -> None:
    """A stem's shard depends on the stem alone, not on how many siblings exist."""
    assert rv.shard_of("sub-a1_ses-b2_task-x", 32) == rv.shard_of("sub-a1_ses-b2_task-x", 32)
    assignments = {stem: rv.shard_of(stem, 32) for stem in (f"sub-{i}_ses-b_task-t" for i in range(50))}
    assert all(0 <= value < 32 for value in assignments.values())
    assert len(set(assignments.values())) > 1


def test_a_rerun_supersedes_its_earlier_run(tmp_path: Path) -> None:
    """One stem, two timestamped run directories: the later one wins and the earlier is reported."""
    build_recording(tmp_path, timestamp="20260920-030735")
    build_recording(tmp_path, timestamp="20260921-101010")
    rows, report = rv.scan(tmp_path)
    assert len(rows) == 1
    assert rows[0]["run_dir"].endswith("20260921-101010")
    assert report.superseded == ["sub-a1_ses-b2_task-diadochokinesis-pa_20260920-030735"]


def test_an_unfinished_recording_is_counted_rather_than_written(tmp_path: Path) -> None:
    """One slice still running must show up as a number, not as a silently short table."""
    build_recording(tmp_path, stem="sub-a1_ses-b2_task-done")
    build_recording(tmp_path, stem="sub-a2_ses-b2_task-running", with_fold=False)
    rows, report = rv.scan(tmp_path)
    assert report.written == 1 == len(rows)
    assert len(report.incomplete) == 1


def test_a_measurement_the_schema_has_no_column_for_is_reported_not_dropped_silently(tmp_path: Path) -> None:
    """A new measurement name must surface as an anomaly rather than vanish."""
    build_recording(tmp_path, measurements=(("a_name_nobody_declared", 1.0),))
    _, report = rv.scan(tmp_path)
    assert report.anomalies == {"a_name_nobody_declared": 1}


# ------------------------------------------------------- the encoding agrees with what it encodes


def test_the_span_rows_are_the_figure_s_own() -> None:
    """The five-row lane's codes and its measure mapping are not restated by hand."""
    assert rv.SPAN_ROWS == figure_module._SPAN_ROWS
    assert rv.MEASURE_ROW == figure_module._MEASURE_CODE


def test_the_squim_ranges_are_the_figure_s_own() -> None:
    """A SQUIM byte decodes on the range the figure colours it over."""
    assert dict(rv.SQUIM_RANGES) == figure_module.FigureStyle().squim_ranges


def test_the_lanes_are_the_branches_plus_redact() -> None:
    """The lane byte indexes the same list the summary figure draws."""
    assert rv.LANES == (*BRANCHES, "REDACT")


def test_the_flag_outcomes_are_the_ones_the_report_counts() -> None:
    """``flags_n`` counts what ``report.py`` calls a flag, not a second definition."""
    source = (Path(figure_module.__file__).parent / "report.py").read_text()
    assert 'reason.get("outcome") in {"flag", "fail", "discard"}' in source
    assert rv.FLAG_OUTCOMES == {"flag", "fail", "discard"}
