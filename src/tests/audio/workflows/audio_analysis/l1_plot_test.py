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


# ── the richer evidence figure ─────────────────────────────────────────


def test_the_spectrogram_is_returned_in_db() -> None:
    """A spectrogram row is read in dB; linear magnitude hides everything below the loudest."""
    from senselab.audio.workflows.audio_analysis.l1_plot import spectrogram_db

    t = np.arange(0, 1.0, 1 / SR)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    spec, times, freqs = spectrogram_db(tone, SR)
    assert spec.shape == (freqs.size, times.size)
    assert spec.max() <= 0.0, "normalised so 0 dB is the loudest bin"
    peak_hz = freqs[int(np.argmax(spec[:, spec.shape[1] // 2]))]
    assert abs(peak_hz - 440.0) < 60.0


def test_a_silent_signal_yields_a_floored_spectrogram() -> None:
    """Digital silence must not produce -inf, which cannot be rendered."""
    from senselab.audio.workflows.audio_analysis.l1_plot import spectrogram_db

    spec, _t, _f = spectrogram_db(np.zeros(SR), SR)
    assert np.isfinite(spec).all()


def test_scene_composition_sums_to_one_where_any_label_fired() -> None:
    """A composition plot is read as shares, so the columns must be normalised."""
    from senselab.audio.workflows.audio_analysis.l1_plot import scene_composition

    windows = [
        {"start": 0.0, "end": 1.0, "labels": ["Speech", "Music"], "scores": [0.8, 0.2]},
        {"start": 1.0, "end": 2.0, "labels": ["Silence"], "scores": [0.9]},
    ]
    times, shares = scene_composition(windows, duration_s=2.0, hop_s=0.5)
    assert times.size == shares.shape[1]
    totals = shares.sum(axis=0)
    assert np.allclose(totals[totals > 0], 1.0)


def test_scene_composition_leaves_uncovered_time_empty() -> None:
    """A gap in classifier coverage must read as absent, not as an even split."""
    from senselab.audio.workflows.audio_analysis.l1_plot import scene_composition

    windows = [{"start": 0.0, "end": 1.0, "labels": ["Speech"], "scores": [0.9]}]
    _times, shares = scene_composition(windows, duration_s=3.0, hop_s=0.5)
    assert shares[:, -1].sum() == pytest.approx(0.0)


def test_the_plot_accepts_words_a_spectrogram_and_scene_rows(tmp_path) -> None:  # noqa: ANN001
    """All of it in one figure, which is the point: the rows explain each other."""
    path = build_l1_signal_plot(
        tmp_path,
        signals={"pyannote": [(0.0, 1.0)]},
        duration_s=2.0,
        waveform=np.zeros(SR * 2),
        sampling_rate=SR,
        words_by_model={"whisper": [{"start": 0.1, "end": 0.4, "text": "hello"}]},
        scene_by_classifier={
            "yamnet": [{"start": 0.0, "end": 1.0, "labels": ["Speech"], "scores": [0.9]}],
            "ast": [{"start": 0.0, "end": 2.0, "labels": ["Music"], "scores": [0.5]}],
        },
    )
    assert path.exists() and path.stat().st_size > 0


def test_the_figure_still_carries_no_uncertainty(tmp_path) -> None:  # noqa: ANN001
    """The invariant survives the extra rows: L1 shows evidence, never conclusions."""
    from senselab.audio.workflows.audio_analysis import l1_plot

    source = __import__("inspect").getsource(l1_plot)
    for term in ("within_pass_uncertainty", "epistemic", "triage_score"):
        assert term not in source


# ── grouping, display type, and honest absence ─────────────────────────


def test_signals_are_grouped_by_the_kind_of_evidence_they_are() -> None:
    """Alphabetical order interleaved a frame VAD, an acoustic proxy and a diarizer.

    Every row then looked identical, so a reader could not tell what kind of claim any of them
    was making. Grouping is what lets the eye compare like with like.
    """
    from senselab.audio.workflows.audio_analysis.l1_plot import classify_signal

    assert classify_signal("frame_brouhaha_vad") == "frame"
    assert classify_signal("acoustic_hnr") == "acoustic"
    assert classify_signal("yamnet") == "scene"
    assert classify_signal("pyannote/speaker-diarization-community-1") == "diarization"
    assert classify_signal("nvidia/diar_sortformer_4spk-v1") == "diarization"
    assert classify_signal("nyralabs/CrisperWhisper2.0_turbo") == "asr"
    assert classify_signal("nvidia/canary-qwen-2.5b") == "asr"


def test_a_continuous_signal_is_drawn_as_a_trace_not_a_bar(tmp_path) -> None:  # noqa: ANN001
    """Rendering a frame posterior as on/off discards everything it measured.

    Both VAD rows previously drew as solid full-width blocks — they fire in every bucket — which
    is exactly the information a trace preserves and a bar destroys.
    """
    path = build_l1_signal_plot(
        tmp_path,
        signals={},
        duration_s=1.0,
        series={"frame_brouhaha_vad": ([0.0, 0.5, 1.0], [0.1, 0.9, 0.2])},
        waveform=None,
    )
    assert path.exists()


def test_a_failed_signal_keeps_a_row(tmp_path) -> None:  # noqa: ANN001
    """YAMNet failed on a real run and vanished from the figure.

    Omitting it makes a failure indistinguishable from a signal that was never configured, and
    those call for different responses.
    """
    path = build_l1_signal_plot(
        tmp_path, signals={"ast": [(0.0, 1.0)]}, duration_s=1.0, failed=["yamnet"], waveform=None
    )
    assert path.exists()


def test_asr_words_are_drawn_in_their_own_model_row(tmp_path) -> None:  # noqa: ANN001
    """A shared words row collided every token into an unreadable smear, and attributed the
    transcript to no model in particular.
    """
    path = build_l1_signal_plot(
        tmp_path,
        signals={"whisper": [(0.0, 1.0)], "canary": [(0.0, 1.0)]},
        duration_s=1.0,
        words_by_model={
            "whisper": [{"start": 0.1, "end": 0.4, "text": "hello"}],
            "canary": [{"start": 0.1, "end": 0.4, "text": "hallo"}],
        },
        waveform=None,
    )
    assert path.exists()
