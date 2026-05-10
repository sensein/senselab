"""Smoke test — workflow module imports and exposes the documented entry points."""

from __future__ import annotations


def test_module_exports_public_api() -> None:
    """Every documented entry point in __init__.py is importable."""
    from senselab.audio.workflows import audio_analysis as wf

    assert callable(wf.compute_uncertainty_axes)
    assert callable(wf.build_disagreements_index)
    assert callable(wf.build_aligned_timeline_plot)
    assert callable(wf.attach_uncertainty_tracks_to_ls)
    assert callable(wf.write_axis_parquet)
    assert callable(wf.apply_aggregator)
    assert callable(wf.uncertainty_to_label_bin)
    assert callable(wf.extract_per_window_embeddings)
    assert tuple(wf.AGGREGATORS) == ("min", "mean", "harmonic_mean", "disagreement_weighted")
