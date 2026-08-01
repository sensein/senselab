"""``L2/round<N>/timeline.png`` — one figure per round, showing only fused quantities."""

from __future__ import annotations

from senselab.audio.workflows.audio_analysis.l2_plot import build_round_timeline


def _rows(values: list[float | None]) -> list[dict]:
    return [
        {
            "start": i * 0.5,
            "end": i * 0.5 + 0.5,
            "uncertainty": v,
            "epistemic_uncertainty": v,
            "confidence": None if v is None else 1.0 - v,
            "variability": v,
        }
        for i, v in enumerate(values)
    ]


def test_a_round_with_rows_gets_a_figure(tmp_path) -> None:  # noqa: ANN001
    """Per round, because a single end-state figure cannot show what the iteration did."""
    path = build_round_timeline(tmp_path, round_index=0, axis_rows={"speaker": _rows([0.1, 0.9])}, duration_s=1.0)
    assert path is not None and path.exists()
    assert path.parent.name == "round0"


def test_each_round_writes_its_own_file(tmp_path) -> None:  # noqa: ANN001
    """Later rounds must not overwrite earlier ones, or the comparison is impossible."""
    a = build_round_timeline(tmp_path, round_index=0, axis_rows={"speaker": _rows([0.5])}, duration_s=0.5)
    b = build_round_timeline(tmp_path, round_index=1, axis_rows={"speaker": _rows([0.2])}, duration_s=0.5)
    assert a is not None and b is not None
    assert a != b and a.exists() and b.exists()


def test_a_round_with_no_rows_writes_nothing(tmp_path) -> None:  # noqa: ANN001
    """An empty figure would suggest the round ran and found nothing, not that it never ran."""
    assert build_round_timeline(tmp_path, round_index=0, axis_rows={"speaker": []}, duration_s=1.0) is None


def test_unmeasured_buckets_leave_gaps_rather_than_interpolating(tmp_path) -> None:  # noqa: ANN001
    """A stretch nobody measured must read as absent, not as a value drawn across it."""
    import numpy as np

    from senselab.audio.workflows.audio_analysis.l2_plot import build_round_timeline as build

    path = build(tmp_path, round_index=0, axis_rows={"speaker": _rows([0.4, None, 0.4])}, duration_s=1.5)
    assert path is not None and path.exists()
    # The gap is expressed as NaN, which matplotlib breaks the line at.
    values = [r["uncertainty"] for r in _rows([0.4, None, 0.4])]
    assert np.isnan(np.array([np.nan if v is None else v for v in values])).any()


def test_the_figure_shows_only_fused_quantities() -> None:
    """Evidence rows live in L1/signals.png; mixing them invites reading one as the other."""
    import inspect

    from senselab.audio.workflows.audio_analysis import l2_plot

    source = inspect.getsource(l2_plot)
    for term in ("model_votes", "native_confidence", "speaks"):
        assert term not in source
