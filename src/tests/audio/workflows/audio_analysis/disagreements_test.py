"""Disagreements index ranking + axis-priority tiebreak.

Ranks over the fused axes on ``triage_score`` — the column that exists for "where should budget
go?". No ``pass`` field: an axis is a fold across passes, so a disagreement belongs to a span of
the recording, not to one transform of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from senselab.audio.workflows.audio_analysis.disagreements import build_disagreements_index
from senselab.audio.workflows.audio_analysis.layout import estimates_dir
from senselab.audio.workflows.audio_analysis.types import FusedAxis, SignalResult, SignalRow


def _row(start: float, triage: float | None, **extra: object) -> dict[str, Any]:
    return {
        "start": start,
        "end": start + 0.5,
        "uncertainty": triage,
        "epistemic_uncertainty": triage,
        "triage_score": triage,
        "contributing_signals": ["m"],
        "contributing_passes": ["raw", "enhanced"],
        "signal_weights": {"m": 1.0},
        "weight_basis": {"m": {"stability": 1.0, "support": 1.0}},
        "round": 0,
        **extra,
    }


def _axes(**by_axis: list[dict[str, Any]]) -> dict[str, FusedAxis]:
    return {axis: FusedAxis(axis=axis, rows=rows) for axis, rows in by_axis.items()}  # type: ignore[arg-type]


def test_disagreements_index_ranks_by_triage_desc(tmp_path: Path) -> None:
    """Disagreements index ranks by triage_score desc."""
    idx = build_disagreements_index(
        fused_axes=_axes(
            speech_presence=[_row(0.0, 0.2), _row(1.0, 0.9)],
            asr=[_row(0.5, 0.5)],
        ),
        top_n=10,
        run_dir=tmp_path,
        config={"top_n": 10, "aggregator": "min", "phoneme_disagreement_threshold": 0.5},
        incomparable_reasons={},
    )
    assert [e["triage_score"] for e in idx["entries"]] == [0.9, 0.5, 0.2]
    assert idx["entries"][0]["rank"] == 1
    assert idx["entries"][2]["rank"] == 3


def test_entries_have_no_pass_field(tmp_path: Path) -> None:
    """A fold across passes cannot be attributed to one pass; it lists them instead."""
    idx = build_disagreements_index(
        fused_axes=_axes(speaker=[_row(0.0, 0.5)]),
        top_n=10,
        run_dir=tmp_path,
        config={},
        incomparable_reasons={},
    )
    entry = idx["entries"][0]
    assert "pass" not in entry
    assert entry["contributing_passes"] == ["raw", "enhanced"]
    assert "rows_by_pass" not in idx["totals"]


def test_parquet_path_points_at_the_file_that_exists(tmp_path: Path) -> None:
    """The index has to send a reader to a path the writer actually used.

    Resolved against a file on disk and against :func:`estimates_dir`, not against a literal. The
    literal this used to assert (``L2/round3/uncertainty/asr.parquet``) kept passing after both the
    directory name and the round nesting changed under it, so the test named the property while
    pinning the drift.
    """
    written = estimates_dir(tmp_path, 3) / "asr.parquet"
    written.parent.mkdir(parents=True)
    written.write_bytes(b"")

    idx = build_disagreements_index(
        fused_axes=_axes(asr=[_row(0.0, 0.5, round=3)]),
        top_n=10,
        run_dir=tmp_path,
        config={},
        incomparable_reasons={},
    )

    pointer = idx["entries"][0]["parquet"]
    assert pointer == "L2/round/3/estimates/asr.parquet"
    assert not Path(pointer).is_absolute(), "the pointer travels inside the run dir"
    assert (tmp_path / pointer) == written
    assert (tmp_path / pointer).is_file()
    assert idx["entries"][0]["ls_region_id"].startswith("uncertainty__asr__")


def test_disagreements_axis_priority_tiebreak(tmp_path: Path) -> None:
    """Same triage_score → asr > speaker > speech_presence."""
    idx = build_disagreements_index(
        fused_axes=_axes(
            speech_presence=[_row(0.0, 0.5)],
            speaker=[_row(0.0, 0.5)],
            asr=[_row(0.0, 0.5)],
        ),
        top_n=10,
        run_dir=tmp_path,
        config={"top_n": 10, "aggregator": "min", "phoneme_disagreement_threshold": 0.5},
        incomparable_reasons={},
    )
    assert [e["axis"] for e in idx["entries"]] == ["asr", "speaker", "speech_presence"]


def test_disagreements_top_n_truncates(tmp_path: Path) -> None:
    """Disagreements top n truncates."""
    idx = build_disagreements_index(
        fused_axes=_axes(speech_presence=[_row(float(i), 1.0 - i * 0.1) for i in range(5)]),
        top_n=2,
        run_dir=tmp_path,
        config={"top_n": 2, "aggregator": "min", "phoneme_disagreement_threshold": 0.5},
        incomparable_reasons={},
    )
    assert len(idx["entries"]) == 2
    assert idx["totals"]["total_rows"] == 5


def test_disagreements_top_n_zero_returns_empty_entries(tmp_path: Path) -> None:
    """top_n=0 → no entries listed; totals still populated."""
    idx = build_disagreements_index(
        fused_axes=_axes(speech_presence=[_row(0.0, 0.9)]),
        top_n=0,
        run_dir=tmp_path,
        config={"top_n": 0, "aggregator": "min", "phoneme_disagreement_threshold": 0.5},
        incomparable_reasons={},
    )
    assert idx["entries"] == []
    assert idx["totals"]["total_rows"] == 1


def test_disagreements_null_triage_sorts_last(tmp_path: Path) -> None:
    """A bucket where nothing spoke ranks last rather than first."""
    idx = build_disagreements_index(
        fused_axes=_axes(speech_presence=[_row(0.0, None), _row(1.0, 0.5)]),
        top_n=10,
        run_dir=tmp_path,
        config={"top_n": 10, "aggregator": "min", "phoneme_disagreement_threshold": 0.5},
        incomparable_reasons={},
    )
    assert idx["entries"][0]["triage_score"] == 0.5
    assert idx["entries"][1]["triage_score"] is None


def test_summary_reads_l1_measurements_not_a_second_fold(tmp_path: Path) -> None:
    """Per-signal detail comes from the L1 signal rows, joined on (bucket, signal)."""
    signal_results = {
        "raw": {
            "m": SignalResult(
                perturbation="raw",
                signal="m",
                rows=[SignalRow(start=0.0, end=0.5, signal="m", measurement={"covered_fraction": 1.0})],
            )
        }
    }
    idx = build_disagreements_index(
        fused_axes=_axes(speech_presence=[_row(0.0, 0.7, snr_brouhaha_db=12.0, src_dominant="speech")]),
        top_n=10,
        run_dir=tmp_path,
        config={},
        incomparable_reasons={},
        signal_results_by_pass=signal_results,
    )
    summary = idx["entries"][0]["summary"]
    assert "snr=12.0dB" in summary
    assert "src=speech" in summary
    assert "raw::m" in summary
