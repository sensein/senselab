"""Timeline plot smoke test (T035)."""

from __future__ import annotations

from pathlib import Path

import pytest

from senselab.audio.workflows.audio_analysis.plot import build_aligned_timeline_plot
from senselab.audio.workflows.audio_analysis.types import AxisResult, UncertaintyRow


def _row(start: float, axis: str, u: float) -> UncertaintyRow:
    return UncertaintyRow(
        start=start,
        end=start + 0.5,
        axis=axis,  # type: ignore[arg-type]
        aggregated_uncertainty=u,
        contributing_models=["m"],
        model_votes={"m": {"speaks": True}},
        comparison_status="ok",
    )


def test_build_aligned_timeline_plot_writes_png(tmp_path: Path) -> None:
    """6-row figure renders for a tiny 4 s synthetic axis_results dict + detail bundles."""
    from types import SimpleNamespace

    import numpy as np

    axis_results = {}
    for pass_label in ("raw_16k", "enhanced_16k", "raw_vs_enhanced"):
        for axis in ("presence", "identity", "utterance"):
            axis_results[(pass_label, axis)] = AxisResult(
                pass_label=pass_label,  # type: ignore[arg-type]
                axis=axis,  # type: ignore[arg-type]
                rows=[
                    _row(0.0, axis, 0.2),
                    _row(0.5, axis, 0.7),
                    _row(1.0, axis, 0.4),
                    _row(1.5, axis, 0.9),
                ],
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
        for pl in ("raw_16k", "enhanced_16k")
    }

    out = build_aligned_timeline_plot(
        run_dir=tmp_path,
        axis_results=axis_results,
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
    axis_results = {
        ("raw_16k", "presence"): AxisResult(pass_label="raw_16k", axis="presence", rows=[_row(0.0, "presence", 0.5)]),
    }
    out = build_aligned_timeline_plot(
        run_dir=tmp_path,
        axis_results=axis_results,
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
        axis_results={},
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
    axis_results = {
        ("raw_16k", "presence"): AxisResult(pass_label="raw_16k", axis="presence", rows=[_row(0.0, "presence", 0.5)]),
    }
    out = build_aligned_timeline_plot(
        run_dir=tmp_path,
        axis_results=axis_results,
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
    axis_results = {
        ("raw_16k", "presence"): AxisResult(
            pass_label="raw_16k",
            axis="presence",
            rows=[_row(i * 0.5, "presence", 0.5) for i in range(int(duration_s * 2))],
        ),
    }
    first = build_aligned_timeline_plot(
        run_dir=tmp_path,
        axis_results=axis_results,
        duration_s=duration_s,
        grid_hop=0.5,
        audio_waveform=wf,
        audio_sr=sr,
    )
    assert first is not None
    assert first.name == "timeline_001.png"
    chunks = sorted(tmp_path.glob("timeline_*.png"))
    assert [p.name for p in chunks] == ["timeline_001.png", "timeline_002.png", "timeline_003.png"]
