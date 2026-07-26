"""LS bundle integration tests (T025)."""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.labelstudio import (
    LABEL_VALUES,
    attach_uncertainty_tracks_to_ls,
    uncertainty_to_label_bin,
)
from senselab.audio.workflows.audio_analysis.types import AxisResult, UncertaintyRow


def test_uncertainty_to_label_bin_thresholds() -> None:
    """Uncertainty to label bin thresholds."""
    assert uncertainty_to_label_bin(0.0, "ok") == "low"
    assert uncertainty_to_label_bin(0.32, "ok") == "low"
    assert uncertainty_to_label_bin(0.33, "ok") == "medium"
    assert uncertainty_to_label_bin(0.65, "ok") == "medium"
    assert uncertainty_to_label_bin(0.66, "ok") == "high"
    assert uncertainty_to_label_bin(1.0, "ok") == "high"


def test_uncertainty_to_label_bin_status_overrides() -> None:
    """Uncertainty to label bin status overrides."""
    assert uncertainty_to_label_bin(0.5, "incomparable") == "incomparable"
    assert uncertainty_to_label_bin(0.5, "unavailable") == "unavailable"
    assert uncertainty_to_label_bin(0.5, "one_sided") == "incomparable"
    assert uncertainty_to_label_bin(None, "ok") == "incomparable"


def test_label_values_is_fixed_5_value_enum() -> None:
    """Label values is fixed 5 value enum."""
    assert LABEL_VALUES == ("low", "medium", "high", "incomparable", "unavailable")


def _row(start: float, end: float, axis: str, u: float | None, votes: dict) -> UncertaintyRow:
    return UncertaintyRow(
        start=start,
        end=end,
        axis=axis,  # type: ignore[arg-type]
        aggregated_uncertainty=u,
        contributing_models=sorted(votes.keys()),
        model_votes=votes,
        comparison_status="ok",
    )


def test_attach_uncertainty_tracks_adds_xml_blocks_and_regions() -> None:
    """Six Labels tracks (3 axes × 2 passes) + 3 raw_vs_enh + utterance TextArea siblings."""
    base_config = '<View>\n  <Audio name="audio" value="$audio"/>\n</View>'
    ls_tasks = [
        {
            "data": {"audio": "x.wav", "pass": "raw_16k"},
            "predictions": [{"result": []}],
        },
        {
            "data": {"audio": "x.wav", "pass": "enhanced_16k"},
            "predictions": [{"result": []}],
        },
    ]
    axis_results: dict = {}
    for pass_label in ("raw_16k", "enhanced_16k", "raw_vs_enhanced"):
        for axis in ("presence", "identity", "utterance"):
            row = _row(
                0.0,
                0.5,
                axis,
                0.7,
                (
                    {"whisper": {"text": "hello", "speaks": True}}
                    if axis != "identity"
                    else {"pyannote": {"speaker_label": "SPEAKER_00"}}
                ),
            )
            axis_results[(pass_label, axis)] = AxisResult(
                pass_label=pass_label,  # type: ignore[arg-type]
                axis=axis,  # type: ignore[arg-type]
                rows=[row],
            )

    out_tasks, out_config = attach_uncertainty_tracks_to_ls(
        ls_tasks=ls_tasks, ls_config=base_config, axis_results=axis_results
    )

    # XML check: 9 Labels tracks (3 axes × 3 pass-buckets) + 3 utterance TextArea.
    assert out_config.count('<Labels name="') == 9
    assert out_config.count("<TextArea") == 3
    # Track names.
    assert 'name="raw_16k__uncertainty__presence"' in out_config
    assert 'name="enhanced_16k__uncertainty__utterance"' in out_config
    assert 'name="pass_pair__uncertainty__identity"' in out_config

    # Tasks: each row produces a Labels region; utterance rows additionally produce a
    # TextArea region.
    raw_task = out_tasks[0]
    enhanced_task = out_tasks[1]
    raw_regions = raw_task["predictions"][0]["result"]
    enh_regions = enhanced_task["predictions"][0]["result"]
    # raw_16k carries 3 Labels (presence/identity/utterance) + 1 TextArea (utterance) +
    # the raw_vs_enhanced delta tracks (which fall back to raw_16k task by convention).
    # That's 3 (own) + 1 (own utterance text) + 3 (pass_pair labels) + 1 (pass_pair utterance text) = 8.
    assert len(raw_regions) == 8
    # enhanced_16k carries 3 Labels + 1 TextArea = 4.
    assert len(enh_regions) == 4
    # Bin label is "high" because aggregated_uncertainty=0.7 ≥ 0.66.
    label_regions = [r for r in raw_regions if r["type"] == "labels"]
    assert all(r["value"]["labels"] == ["high"] for r in label_regions)


# ── FR-024 (T040): scene tracks on per-pass presence results ────────────


def _presence_row_with_scene(start: float, *, quality_snr: float | None, src_dominant: str | None) -> UncertaintyRow:
    return UncertaintyRow(
        start=start,
        end=start + 0.5,
        axis="presence",
        aggregated_uncertainty=0.4,
        contributing_models=["m"],
        model_votes={"m": {"speaks": True}},
        comparison_status="ok",
        quality_snr=quality_snr,
        src_dominant=src_dominant,
    )


def test_scene_tracks_added_when_columns_present() -> None:
    """Quality + sources tracks appear (additive) only for passes carrying the columns."""
    from senselab.audio.workflows.audio_analysis.labelstudio import attach_uncertainty_tracks_to_ls

    axis_results = {
        ("raw_16k", "presence"): AxisResult(
            pass_label="raw_16k",
            axis="presence",
            rows=[
                _presence_row_with_scene(0.0, quality_snr=0.8, src_dominant="machine"),
                _presence_row_with_scene(0.5, quality_snr=None, src_dominant=None),
            ],
        ),
    }
    tasks = [{"data": {"pass": "raw_16k"}, "predictions": [{"result": []}]}]
    tasks_out, config = attach_uncertainty_tracks_to_ls(
        ls_tasks=tasks, ls_config="<View></View>", axis_results=axis_results
    )
    assert '<Labels name="raw_16k__presence__quality"' in config
    assert '<Labels name="raw_16k__presence__sources"' in config
    assert '<Label value="machine"/>' in config
    regions = tasks_out[0]["predictions"][0]["result"]
    q_regions = [r for r in regions if r["from_name"] == "raw_16k__presence__quality"]
    s_regions = [r for r in regions if r["from_name"] == "raw_16k__presence__sources"]
    # Only the row with non-null columns emits scene regions (no all-"unavailable" noise).
    assert len(q_regions) == 1 and q_regions[0]["value"]["labels"] == ["high"]  # 0.8 ≥ HIGH
    assert len(s_regions) == 1 and s_regions[0]["value"]["labels"] == ["machine"]
    # Existing presence track unchanged (still one region per row).
    base = [r for r in regions if r["from_name"] == "raw_16k__uncertainty__presence"]
    assert len(base) == 2


def test_scene_tracks_absent_without_columns_and_for_deltas() -> None:
    """Legacy bundles stay byte-identical: no scene tracks when columns are null / delta pass."""
    from senselab.audio.workflows.audio_analysis.labelstudio import attach_uncertainty_tracks_to_ls

    axis_results = {
        ("raw_16k", "presence"): AxisResult(
            pass_label="raw_16k",
            axis="presence",
            rows=[_presence_row_with_scene(0.0, quality_snr=None, src_dominant=None)],
        ),
        ("raw_vs_enhanced", "presence"): AxisResult(
            pass_label="raw_vs_enhanced",
            axis="presence",
            rows=[_presence_row_with_scene(0.0, quality_snr=0.9, src_dominant="speech")],
        ),
    }
    tasks = [{"data": {"pass": "raw_16k"}, "predictions": [{"result": []}]}]
    _, config = attach_uncertainty_tracks_to_ls(
        ls_tasks=tasks, ls_config="<View></View>", axis_results=axis_results
    )
    assert "__presence__quality" not in config
    assert "__presence__sources" not in config
