"""LS bundle integration tests (T025)."""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.labelstudio import (
    LABEL_VALUES,
    _classification_to_ls,
    _collect_classification_labels,
    attach_uncertainty_tracks_to_ls,
    uncertainty_to_label_bin,
)
from senselab.audio.workflows.audio_analysis.types import FusedAxis, SignalResult, SignalRow


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
    assert uncertainty_to_label_bin(None, "ok") == "incomparable"


def test_label_values_is_fixed_5_value_enum() -> None:
    """Label values is fixed 5 value enum."""
    assert LABEL_VALUES == ("low", "medium", "high", "incomparable", "unavailable")


def _row(start: float, end: float, u: float | None, **extra: object) -> dict:
    """One fused-axis row: no pass, no per-signal votes — those live on the L1 signal rows."""
    return {
        "start": start,
        "end": end,
        "uncertainty": u,
        "epistemic_uncertainty": u,
        "confidence": None if u is None else 1.0 - u,
        "variability": 0.0,
        "triage_score": u,
        "contributing_signals": ["m"],
        "contributing_passes": ["raw_16k", "enhanced_16k"],
        "signal_weights": {"m": 1.0},
        "weight_basis": {"m": {}},
        "round": 0,
        **extra,
    }


def _signal(pass_label: str, signal: str, *, rows: list) -> SignalResult:
    return SignalResult(pass_label=pass_label, signal=signal, rows=rows)  # type: ignore[arg-type]


def test_attach_uncertainty_tracks_adds_one_track_per_axis_no_pass_token() -> None:
    """Three axis tracks, attached once; per-pass evidence rides its own signal tracks."""
    base_config = '<View>\n  <Audio name="audio" value="$audio"/>\n</View>'
    ls_tasks = [
        {"data": {"audio": "x.wav", "pass": "raw_16k"}, "predictions": [{"result": []}]},
        {"data": {"audio": "x.wav", "pass": "enhanced_16k"}, "predictions": [{"result": []}]},
    ]
    fused_axes = {
        axis: FusedAxis(
            axis=axis,  # type: ignore[arg-type]
            rows=[_row(0.0, 0.5, 0.7, consensus_votes={"raw_16k::whisper": {"text": "hello"}})],
        )
        for axis in ("speech_presence", "speaker", "asr")
    }
    signal_results = {
        pl: {"m": _signal(pl, "m", rows=[SignalRow(start=0.0, end=0.5, signal="m", measurement={"x": 1.0})])}
        for pl in ("raw_16k", "enhanced_16k")
    }

    out_tasks, out_config = attach_uncertainty_tracks_to_ls(
        ls_tasks=ls_tasks,
        ls_config=base_config,
        fused_axes=fused_axes,
        signal_results_by_pass=signal_results,
    )

    # 3 axis tracks + 1 signal track per (pass, signal) = 5 Labels; 1 asr TextArea.
    assert out_config.count('<Labels name="') == 5
    assert out_config.count("<TextArea") == 1
    assert 'name="uncertainty__speech_presence"' in out_config
    assert 'name="uncertainty__asr"' in out_config
    # The vocabulary that made the category error feel natural is gone.
    assert "pass_pair__" not in out_config
    assert "raw_16k__uncertainty__" not in out_config
    assert 'name="raw_16k__signal__m"' in out_config

    raw_regions = out_tasks[0]["predictions"][0]["result"]
    enh_regions = out_tasks[1]["predictions"][0]["result"]
    # raw_16k carries the 3 axis Labels + 1 asr TextArea + its own 1 signal track.
    assert len(raw_regions) == 5
    # enhanced_16k carries only its own signal track: an axis is not per pass.
    assert len(enh_regions) == 1
    axis_labels = [r for r in raw_regions if r["from_name"].startswith("uncertainty__") and r["type"] == "labels"]
    assert all(r["value"]["labels"] == ["high"] for r in axis_labels)  # 0.7 >= HIGH


def test_binning_policy_travels_with_the_bundle() -> None:
    """A rendered "high" is a thresholded value, so the thresholds ride on the task."""
    tasks = [{"data": {"pass": "raw_16k"}, "predictions": [{"result": []}]}]
    out, _ = attach_uncertainty_tracks_to_ls(
        ls_tasks=tasks,
        ls_config="<View></View>",
        fused_axes={"speaker": FusedAxis(axis="speaker", rows=[_row(0.0, 0.5, 0.7)])},
    )
    policy = out[0]["data"]["uncertainty_bin_policy"]
    assert policy["low_threshold"] == 0.33 and policy["high_threshold"] == 0.66


def test_scene_tracks_read_l1_measurements_and_l2_scores() -> None:
    """Quality + source stripes are per-pass measurements; the anchored scores join from L2."""
    fused_axes = {
        "speech_presence": FusedAxis(
            axis="speech_presence",
            rows=[_row(0.0, 0.5, 0.4, quality_snr=0.8), _row(0.5, 1.0, 0.4)],
        )
    }
    signal_results = {
        "raw_16k": {
            "scene_quality": _signal(
                "raw_16k",
                "scene_quality",
                rows=[
                    SignalRow(start=0.0, end=0.5, signal="scene_quality", measurement={"snr_brouhaha_db": 6.0}),
                    SignalRow(start=0.5, end=1.0, signal="scene_quality", measurement={"snr_brouhaha_db": 30.0}),
                ],
            ),
            "sound_sources": _signal(
                "raw_16k",
                "sound_sources",
                rows=[SignalRow(start=0.0, end=0.5, signal="sound_sources", measurement={"src_dominant": "machine"})],
            ),
        }
    }
    tasks = [{"data": {"pass": "raw_16k"}, "predictions": [{"result": []}]}]
    tasks_out, config = attach_uncertainty_tracks_to_ls(
        ls_tasks=tasks,
        ls_config="<View></View>",
        fused_axes=fused_axes,
        signal_results_by_pass=signal_results,
    )
    assert '<Labels name="raw_16k__presence__quality"' in config
    assert '<Labels name="raw_16k__presence__sources"' in config
    assert '<Label value="machine"/>' in config
    regions = tasks_out[0]["predictions"][0]["result"]
    q_regions = [r for r in regions if r["from_name"] == "raw_16k__presence__quality"]
    s_regions = [r for r in regions if r["from_name"] == "raw_16k__presence__sources"]
    assert len(q_regions) == 1 and q_regions[0]["value"]["labels"] == ["high"]  # 0.8 >= HIGH
    # The max-over-four is a display fold and says so on the region it produced.
    assert q_regions[0]["value"]["fold"]["rule"].startswith("max over")
    assert len(s_regions) == 1 and s_regions[0]["value"]["labels"] == ["machine"]


def test_scene_tracks_absent_without_measurements() -> None:
    """No scene signal → no scene tracks, rather than an all-"unavailable" stripe."""
    tasks = [{"data": {"pass": "raw_16k"}, "predictions": [{"result": []}]}]
    _, config = attach_uncertainty_tracks_to_ls(
        ls_tasks=tasks,
        ls_config="<View></View>",
        fused_axes={"speech_presence": FusedAxis(axis="speech_presence", rows=[_row(0.0, 0.5, 0.4)])},
        signal_results_by_pass={"raw_16k": {}},
    )
    assert "__presence__quality" not in config
    assert "__presence__sources" not in config


# ── asr_has_timestamps consolidation (T051b) ──────────────────────────


def test_chunked_but_untimed_transcript_has_no_timestamps() -> None:
    """Chunks alone are not evidence of timing.

    analyze_audio.py carried a looser duplicate that returned True whenever
    ``chunks`` was non-empty. That made the alignment stage *skip* a
    chunked-but-untimed transcript — precisely the input alignment exists to fix —
    and it disagreed with ``resolve_asr_result``, which has always required a real
    timestamp. The strict semantics is the surviving one.
    """
    from senselab.audio.workflows.audio_analysis.harvesters import asr_has_timestamps
    from senselab.utils.data_structures import ScriptLine

    untimed = ScriptLine(text="hello world", chunks=[ScriptLine(text="hello"), ScriptLine(text="world")])
    assert asr_has_timestamps([untimed]) is False, "chunks without start times are not timestamps"


def test_timestamped_chunk_is_detected() -> None:
    """A chunk carrying a start time does count."""
    from senselab.audio.workflows.audio_analysis.harvesters import asr_has_timestamps
    from senselab.utils.data_structures import ScriptLine

    timed = ScriptLine(text="hi", chunks=[ScriptLine(text="hi", start=0.0, end=0.5)])
    assert asr_has_timestamps([timed]) is True


def test_line_level_timestamp_is_detected() -> None:
    """A line-level start counts even with no chunks."""
    from senselab.audio.workflows.audio_analysis.harvesters import asr_has_timestamps
    from senselab.utils.data_structures import ScriptLine

    assert asr_has_timestamps([ScriptLine(text="hi", start=0.0, end=1.0)]) is True


def test_empty_result_has_no_timestamps() -> None:
    """Nothing in, False out."""
    from senselab.audio.workflows.audio_analysis.harvesters import asr_has_timestamps

    assert asr_has_timestamps([]) is False
    assert asr_has_timestamps(None) is False


def test_ls_builders_are_importable_from_the_library() -> None:
    """T051b: the export builders now live here, not in the CLI script."""
    from senselab.audio.workflows.audio_analysis.labelstudio import (
        build_labelstudio_config,
        build_labelstudio_task,
    )

    assert callable(build_labelstudio_task) and callable(build_labelstudio_config)


# ── Moved from analyze_audio_test.py with the code they cover (T051b) ──


def test_collect_classification_labels_pulls_unique_labels() -> None:
    """The LS-config XML builder collects every distinct AudioSet label observed."""
    classify_result = [
        [
            {"start": 0.0, "end": 0.5, "label_scores": [{"Speech": 0.9}, {"Music": 0.1}]},
            {"start": 0.5, "end": 1.0, "label_scores": [{"Speech": 0.8}, {"Silence": 0.2}]},
        ]
    ]
    labels = _collect_classification_labels(classify_result)
    assert labels == {"Speech", "Music", "Silence"}


def test_classification_to_ls_emits_regions_for_dict_shape() -> None:
    """The LS conversion must skip empty entries but emit one region per dict window."""
    result = [
        [
            {"start": 0.0, "end": 0.5, "label_scores": [{"Speech": 0.95}]},
            {"start": 0.5, "end": 1.0, "label_scores": [{"Music": 0.62}]},
        ]
    ]
    regions = _classification_to_ls(result, prefix="raw__ast", win_length=0.5, hop_length=0.5)
    assert len(regions) == 2
    labels = [r["value"]["labels"][0] for r in regions]
    assert labels == ["Speech", "Music"]


# ── background mask + per-speaker speech_presence tracks (T106) ──────────────


def _bundle() -> tuple[list[dict], str]:
    task = {"data": {"pass": "raw_16k"}, "predictions": [{"result": []}]}
    return [task], "<View>\n<Audio name='audio' value='$audio'/>\n</View>"


def test_the_background_mask_reaches_the_annotation_bundle() -> None:
    """FR-033: a reviewer needs the intervals the machine actually trusted.

    The mask decides which findings are trustworthy, so a human checking those findings must
    see the same regions rather than inferring them.
    """
    from senselab.audio.workflows.audio_analysis.labelstudio import attach_scene_context_tracks_to_ls

    tasks, config = _bundle()
    tasks, config = attach_scene_context_tracks_to_ls(
        ls_tasks=tasks,
        ls_config=config,
        mask_rows=[
            {"start": 0.0, "end": 1.0, "state": "target_free"},
            {"start": 1.0, "end": 2.0, "state": "target_active"},
        ],
    )
    assert "raw_16k__background__mask" in config
    regions = tasks[0]["predictions"][0]["result"]
    assert [r["value"]["labels"] for r in regions] == [["target_free"], ["target_active"]]
    assert regions[0]["value"]["start"] == 0.0 and regions[0]["value"]["end"] == 1.0


def test_every_mask_state_is_declared_in_the_config() -> None:
    """An undeclared label value is rejected on import, losing the region silently."""
    from senselab.audio.workflows.audio_analysis.labelstudio import attach_scene_context_tracks_to_ls

    tasks, config = _bundle()
    _t, config = attach_scene_context_tracks_to_ls(
        ls_tasks=tasks, ls_config=config, mask_rows=[{"start": 0.0, "end": 1.0, "state": "indeterminate"}]
    )
    for state in ("target_free", "target_active", "indeterminate"):
        assert f'value="{state}"' in config


def test_per_speaker_presence_reaches_the_bundle_labelled_by_speaker() -> None:
    """Regions are labelled by speaker, not merged.

    The point of the per-speaker axis is knowing *who* is contested; a single merged track
    would put the same unreadable scalar in front of the annotator again.
    """
    from senselab.audio.workflows.audio_analysis.labelstudio import attach_scene_context_tracks_to_ls

    tasks, config = _bundle()
    tasks, config = attach_scene_context_tracks_to_ls(
        ls_tasks=tasks,
        ls_config=config,
        speaker_rows=[
            {
                "speaker_id": "S0",
                "start": 0.0,
                "end": 0.5,
                "speech_presence_confidence": 1.0,
                "speech_presence_uncertainty": 0.0,
            },
            {
                "speaker_id": "S1",
                "start": 0.0,
                "end": 0.5,
                "speech_presence_confidence": 0.5,
                "speech_presence_uncertainty": 1.0,
            },
        ],
    )
    assert "raw_16k__speaker__speech_presence" in config
    regions = [r for r in tasks[0]["predictions"][0]["result"] if r["from_name"] == "raw_16k__speaker__speech_presence"]
    assert {r["value"]["labels"][0] for r in regions} == {"S0", "S1"}


def test_a_speakers_own_uncertainty_travels_with_its_region() -> None:
    """Without it the annotator sees who was claimed but not how doubtful the claim was."""
    from senselab.audio.workflows.audio_analysis.labelstudio import attach_scene_context_tracks_to_ls

    tasks, config = _bundle()
    tasks, _c = attach_scene_context_tracks_to_ls(
        ls_tasks=tasks,
        ls_config=config,
        speaker_rows=[
            {
                "speaker_id": "S0",
                "start": 0.0,
                "end": 0.5,
                "speech_presence_confidence": 0.33,
                "speech_presence_uncertainty": 0.92,
                "contributing_sources": ["pyannote"],
            }
        ],
    )
    texts = [r for r in tasks[0]["predictions"][0]["result"] if r["type"] == "textarea"]
    assert texts and "0.92" in texts[0]["value"]["text"][0]
    assert "pyannote" in texts[0]["value"]["text"][0]


def test_nothing_is_added_when_there_is_nothing_to_show() -> None:
    """A run without a mask or per-speaker output must produce an unchanged bundle."""
    from senselab.audio.workflows.audio_analysis.labelstudio import attach_scene_context_tracks_to_ls

    tasks, config = _bundle()
    out_tasks, out_config = attach_scene_context_tracks_to_ls(ls_tasks=tasks, ls_config=config)
    assert out_config == config
    assert out_tasks[0]["predictions"][0]["result"] == []


def test_list_columns_read_back_from_parquet_do_not_break_attachment() -> None:
    """Parquet list columns come back as numpy arrays, not lists.

    ``value or []`` on an array raises rather than returning a default, which took down
    attachment mid-run on a real recording — after regions had been appended but before the
    config gained their track declarations.
    """
    import numpy as np

    from senselab.audio.workflows.audio_analysis.labelstudio import attach_scene_context_tracks_to_ls

    tasks, config = _bundle()
    tasks, config = attach_scene_context_tracks_to_ls(
        ls_tasks=tasks,
        ls_config=config,
        speaker_rows=[
            {
                "speaker_id": "S0",
                "start": 0.0,
                "end": 0.5,
                "speech_presence_confidence": 0.5,
                "speech_presence_uncertainty": 0.9,
                "contributing_sources": np.array(["pyannote", "sortformer"], dtype=object),
            }
        ],
    )
    texts = [r for r in tasks[0]["predictions"][0]["result"] if r["type"] == "textarea"]
    assert "pyannote, sortformer" in texts[0]["value"]["text"][0]


def test_a_failure_partway_through_leaves_the_bundle_untouched() -> None:
    """Regions referencing a track the config never declared are worse than no tracks.

    Label Studio drops such regions silently, so a half-applied attachment reads as a
    successful one with missing data. The rows are built before anything is mutated.
    """
    from senselab.audio.workflows.audio_analysis.labelstudio import attach_scene_context_tracks_to_ls

    tasks, config = _bundle()
    bad = [
        {
            "speaker_id": "S0",
            "start": 0.0,
            "end": 0.5,
            "speech_presence_confidence": 1.0,
            "speech_presence_uncertainty": 0.0,
        },
        {"speaker_id": "S1", "start": "not-a-time", "end": 1.0},
    ]
    with pytest.raises((ValueError, TypeError)):
        attach_scene_context_tracks_to_ls(ls_tasks=tasks, ls_config=config, speaker_rows=bad)
    assert tasks[0]["predictions"][0]["result"] == []
