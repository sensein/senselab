"""Timeline plot smoke test (T035)."""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.audio.workflows.audio_analysis.layout import evidence_dir, final_dir
from senselab.audio.workflows.audio_analysis.plot import build_aligned_timeline_plot
from senselab.audio.workflows.audio_analysis.types import FusedAxis


def _row(start: float, axis: str, u: float) -> dict:
    """One fused-axis row. There is no pass on it — an axis is a fold across passes."""
    return {
        "start": start,
        "end": start + 0.5,
        "uncertainty": u,
        "epistemic_uncertainty": u * 0.5,
        "confidence": 1.0 - u,
        "variability": 0.0,
        "triage_score": u,
        "contributing_signals": ["m"],
        "contributing_passes": ["raw", "enhanced"],
        "signal_weights": {"m": 1.0},
        "weight_basis": {"m": {}},
        "round": 0,
    }


def _axes(**by_axis: list) -> dict:
    return {axis: FusedAxis(axis=axis, rows=rows) for axis, rows in by_axis.items()}  # type: ignore[arg-type]


def test_build_aligned_timeline_plot_writes_png(tmp_path: Path) -> None:
    """6-row figure renders for a tiny 4 s synthetic axis_results dict + detail bundles."""
    from types import SimpleNamespace

    import numpy as np

    fused_axes = _axes(
        **{
            axis: [_row(0.0, axis, 0.2), _row(0.5, axis, 0.7), _row(1.0, axis, 0.4), _row(1.5, axis, 0.9)]
            for axis in ("speech_presence", "speaker", "asr")
        }
    )

    # Per-pass detail bundles drive the 3 detail rows.
    diar_segs = [
        SimpleNamespace(start=0.0, end=2.0, speaker="SPEAKER_00"),
        SimpleNamespace(start=2.0, end=4.0, speaker="SPEAKER_01"),
    ]
    asr_chunks = [
        SimpleNamespace(start=0.1, end=0.5, text="hello", avg_logprob=-0.2),
        SimpleNamespace(start=0.6, end=1.2, text="world", avg_logprob=-0.3),
    ]
    asr_line = SimpleNamespace(text="hello world", chunks=asr_chunks, start=0.1, end=1.2, avg_logprob=-0.25)

    from senselab.audio.workflows.audio_analysis.embeddings import WindowEmbedding

    detail_by_pass = {
        pl: {
            "diar_by_model": {"pyannote": diar_segs},
            "asr_by_model": {"whisper": [asr_line]},
            "per_window_embeddings": {
                "speechbrain/spkrec-ecapa-voxceleb": [
                    WindowEmbedding(start_s=0.0, end_s=2.0, vector=np.array([1.0, 0.0, 0.0])),
                    WindowEmbedding(start_s=1.0, end_s=3.0, vector=np.array([0.0, 1.0, 0.0])),
                    WindowEmbedding(start_s=2.0, end_s=4.0, vector=np.array([0.0, 0.0, 1.0])),
                ],
            },
        }
        for pl in ("raw", "enhanced")
    }

    out = build_aligned_timeline_plot(
        run_dir=tmp_path,
        fused_axes=fused_axes,
        duration_s=4.0,
        grid_hop=0.5,
        detail_by_pass=detail_by_pass,
        title="smoke test",
    )
    assert out is not None
    assert out.exists()
    assert out.stat().st_size > 5_000


def test_build_aligned_timeline_plot_minimal_no_detail(tmp_path: Path) -> None:
    """When detail_by_pass is None, the plot still renders the 3 uncertainty rows."""
    fused_axes = _axes(speech_presence=[_row(0.0, "speech_presence", 0.5)])
    out = build_aligned_timeline_plot(
        run_dir=tmp_path,
        fused_axes=fused_axes,
        duration_s=2.0,
        grid_hop=0.5,
        detail_by_pass=None,
    )
    assert out is not None
    assert out.exists()


def test_build_aligned_timeline_plot_returns_none_for_zero_duration(tmp_path: Path) -> None:
    """Zero-duration audio → no plot."""
    out = build_aligned_timeline_plot(
        run_dir=tmp_path,
        fused_axes={},
        duration_s=0.0,
        grid_hop=0.5,
    )
    assert out is None


def test_build_aligned_timeline_plot_renders_spectrogram_top_row(tmp_path: Path) -> None:
    """When ``audio_waveform`` is provided, a spectrogram row is added at the top."""
    import numpy as np

    sr = 16000
    t = np.linspace(0, 4.0, sr * 4, endpoint=False)
    wf = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    fused_axes = _axes(speech_presence=[_row(0.0, "speech_presence", 0.5)])
    out = build_aligned_timeline_plot(
        run_dir=tmp_path,
        fused_axes=fused_axes,
        duration_s=4.0,
        grid_hop=0.5,
        audio_waveform=wf,
        audio_sr=sr,
    )
    assert out is not None
    assert out.exists()
    # PNG with one extra (spectrogram) row should be larger than the no-detail / no-spec variant.
    assert out.stat().st_size > 5_000


def test_build_aligned_timeline_plot_chunks_long_audio(tmp_path: Path) -> None:
    """Audio longer than ``chunk_duration_s`` produces ``timeline_NNN.png`` per chunk."""
    import numpy as np

    sr = 16000
    duration_s = 50.0  # 50s @ default chunk_duration_s=20 → 3 chunks
    wf = (0.2 * np.random.RandomState(0).randn(int(sr * duration_s))).astype(np.float32)
    fused_axes = _axes(speech_presence=[_row(i * 0.5, "speech_presence", 0.5) for i in range(int(duration_s * 2))])
    first = build_aligned_timeline_plot(
        run_dir=tmp_path,
        fused_axes=fused_axes,
        duration_s=duration_s,
        grid_hop=0.5,
        audio_waveform=wf,
        audio_sr=sr,
        # Chunking is opt-in now: by default it produced timeline_001.png, timeline_002.png …
        # whose panels were mostly empty, and one figure per L2 round replaced it.
        chunk_duration_s=20.0,
    )
    assert first is not None
    assert first.name == "uncertainty_detail_001.png"
    # final/, not L1/. The chunks carry the same axis rows the single figure does, so they are
    # conclusions wherever they are written; putting them under L1 made the layer a function of
    # how long the recording happened to be.
    chunks = sorted(final_dir(tmp_path).glob("uncertainty_detail_*.png"))
    assert [p.name for p in chunks] == [
        "uncertainty_detail_001.png",
        "uncertainty_detail_002.png",
        "uncertainty_detail_003.png",
    ]
    assert not list(evidence_dir(tmp_path).glob("*.png"))


def test_axis_figure_never_lands_in_l1(tmp_path: Path) -> None:
    """The figure draws fused axes, so no output of it may be written under ``L1/``.

    The regression this pins: the default was ``evidence_dir(run_dir) / "timeline.png"``, chosen
    inside the renderer to dodge a filename collision with the adaptive ``final/timeline.png``.
    Relabelling a figure of L2 conclusions as "the evidence timeline" moved the violation into
    the one artifact class — a rendering, with no key and no producer — that the write-root
    capability cannot see. Asserted on the *output path* rather than against a declared contract,
    because the contract declared it and the declaration was what made it look legal.
    """
    fused_axes = _axes(speech_presence=[_row(i * 0.5, "speech_presence", 0.4) for i in range(8)])
    out = build_aligned_timeline_plot(run_dir=tmp_path, fused_axes=fused_axes, duration_s=4.0, grid_hop=0.5)

    assert out is not None
    assert out.parent == final_dir(tmp_path)
    assert out.name == "uncertainty_detail.png"
    assert not list(evidence_dir(tmp_path).rglob("*.png"))


def test_explicit_save_path_still_wins(tmp_path: Path) -> None:
    """A caller naming its own destination is obeyed — the default is a default, not a policy."""
    fused_axes = _axes(speech_presence=[_row(i * 0.5, "speech_presence", 0.4) for i in range(8)])
    dest = tmp_path / "elsewhere" / "figure.png"
    out = build_aligned_timeline_plot(
        run_dir=tmp_path, fused_axes=fused_axes, duration_s=4.0, grid_hop=0.5, save_path=dest
    )
    assert out == dest
    assert dest.exists()


def test_scene_quality_and_source_rows_render(tmp_path: Path) -> None:
    """Presence rows carrying quality_* / src_* add the scene-quality and sound-source rows."""
    rows = []
    for i in range(8):
        r = _row(i * 0.5, "speech_presence", 0.3)
        r.update(
            {
                "quality_snr": 0.2,
                "quality_clip": 0.05,
                "quality_reverb": 0.15,
                "quality_bandwidth": 0.1,
                "snr_brouhaha_db": 18.0,
                "src_speech": 0.6,
                "src_people": 0.15,
                "src_machine": 0.15,
                "src_environment": 0.10,
                "src_dominant": "speech",
            }
        )
        rows.append(r)
    fused_axes = _axes(
        speech_presence=rows,
        speaker=[_row(i * 0.5, "speaker", 0.3) for i in range(8)],
        asr=[_row(i * 0.5, "asr", 0.3) for i in range(8)],
    )
    out = build_aligned_timeline_plot(
        run_dir=tmp_path, fused_axes=fused_axes, duration_s=4.0, grid_hop=0.5, detail_by_pass=None
    )
    assert out is not None and out.exists() and out.stat().st_size > 0
