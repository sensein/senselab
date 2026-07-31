"""``L1/signals.png`` — the evidence plot: every signal, plus level, and no uncertainty.

L1 is evidence. A plot of it should show what each signal reported and how loud the audio was
at the time, because "the diarizer stopped here" and "the level dropped to -60 dBFS here" is
the pairing that explains most disagreements.

Uncertainty rows belong to L2 and are deliberately absent: a figure that mixes evidence with
the conclusions drawn from it invites reading the conclusion as another observation.
"""

from __future__ import annotations

import numpy as np
import pytest

from senselab.audio.workflows.audio_analysis.l1_plot import (
    build_l1_signal_plot,
    rms_dbfs_track,
)

SR = 16000


def test_the_level_track_is_in_dbfs() -> None:
    """dBFS, not raw RMS: a level plot is read against full scale, and 0 dBFS is the anchor."""
    full_scale = np.ones(SR, dtype=np.float64)
    times, levels = rms_dbfs_track(full_scale, SR, hop_s=0.1)
    assert len(times) == len(levels)
    assert levels[0] == pytest.approx(0.0, abs=0.1)


def test_a_half_amplitude_signal_reads_about_six_db_down() -> None:
    """The check that the scale is amplitude-referenced rather than power-referenced."""
    _t, levels = rms_dbfs_track(np.full(SR, 0.5), SR, hop_s=0.1)
    assert levels[0] == pytest.approx(-6.02, abs=0.1)


def test_digital_silence_is_floored_not_negative_infinity() -> None:
    """-inf cannot be plotted; a floor keeps the axis usable while staying visibly bottomed."""
    _t, levels = rms_dbfs_track(np.zeros(SR), SR, hop_s=0.1)
    assert np.isfinite(levels).all()
    assert levels[0] <= -90.0


def test_the_track_follows_the_requested_hop() -> None:
    """The level track is the one row that should be at native resolution, not bucket grid."""
    times, _levels = rms_dbfs_track(np.zeros(SR * 2), SR, hop_s=0.5)
    assert len(times) == pytest.approx(4, abs=1)


def test_the_plot_includes_every_signal_that_reported(tmp_path) -> None:  # noqa: ANN001
    """All of them: a plot that silently drops a signal makes its absence invisible."""
    signals = {
        "pyannote": [(0.0, 1.0), (2.0, 3.0)],
        "sortformer": [(0.0, 3.0)],
        "yamnet": [(1.0, 2.0)],
    }
    path = build_l1_signal_plot(
        tmp_path,
        signals=signals,
        duration_s=3.0,
        waveform=np.zeros(SR * 3),
        sampling_rate=SR,
    )
    assert path.exists() and path.stat().st_size > 0


def test_the_plot_carries_no_uncertainty_row(tmp_path) -> None:  # noqa: ANN001
    """L1 is evidence. Mixing in a conclusion invites reading it as another observation."""
    from senselab.audio.workflows.audio_analysis import l1_plot

    source = __import__("inspect").getsource(l1_plot)
    for term in ("aggregated_uncertainty", "within_pass_uncertainty", "epistemic"):
        assert term not in source


def test_the_plot_survives_a_signal_with_no_spans(tmp_path) -> None:  # noqa: ANN001
    """A model that ran and reported nothing must still get a row, or its silence is invisible."""
    path = build_l1_signal_plot(
        tmp_path,
        signals={"quiet": [], "loud": [(0.0, 1.0)]},
        duration_s=1.0,
        waveform=np.zeros(SR),
        sampling_rate=SR,
    )
    assert path.exists()


def test_the_plot_works_without_audio(tmp_path) -> None:  # noqa: ANN001
    """A run whose waveform is unavailable should still get its signal rows."""
    path = build_l1_signal_plot(tmp_path, signals={"a": [(0.0, 1.0)]}, duration_s=1.0, waveform=None, sampling_rate=SR)
    assert path.exists()
