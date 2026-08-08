"""Each signal declares its own temporal resolution; fusion re-interpolates to a common one.

Forcing every signal onto one grid loses information in both directions. A frame posterior at
~17 ms collapsed onto 250 ms buckets saturates — the measured consequence was a VAD trace flat
at 1.0 across a conversation with four clear pauses. An AST window of 10.24 s spread over the
same buckets pretends to a precision it does not have, which is why its scene row came out
nearly constant.

So L1 declares, and L2 converts. The declaration has to travel with the signal: a resolution
inferred at fusion time is a guess about what the harvester did.
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.resolution import (
    NATIVE_RESOLUTION_S,
    declared_resolution_s,
    resample_series,
)


def test_a_frame_voter_declares_its_native_frame_rate() -> None:
    """~17 ms for segmentation-3.0 and brouhaha — the receptive-field step, not a bucket."""
    assert declared_resolution_s("frame_segmentation") == pytest.approx(0.017, abs=0.005)
    assert declared_resolution_s("frame_brouhaha_vad") == pytest.approx(0.017, abs=0.005)


def test_a_long_window_classifier_declares_its_window() -> None:
    """AST decides over 10.24 s. Reporting it at 250 ms claims precision it does not have."""
    assert declared_resolution_s("ast") == pytest.approx(10.24, abs=0.01)


def test_a_windowed_classifier_declares_its_hop_when_one_is_given() -> None:
    """Windowed AST is the whole point: a finer hop earns a finer declared resolution."""
    assert declared_resolution_s("ast", hop_s=1.0) == pytest.approx(1.0)


def test_an_unknown_signal_falls_back_to_the_grid() -> None:
    """A signal with no declaration must not silently claim the finest resolution available."""
    assert declared_resolution_s("something_new", grid_s=0.25) == pytest.approx(0.25)


def test_upsampling_a_coarse_signal_holds_its_value() -> None:
    """A 10 s decision applies across its whole window; interpolating it would invent detail."""
    times, values = resample_series([0.0, 10.0], [0.2, 0.8], target_hop_s=5.0, duration_s=20.0, kind="hold")
    assert list(times) == [0.0, 5.0, 10.0, 15.0]
    assert values[0] == pytest.approx(0.2)
    assert values[1] == pytest.approx(0.2), "held across its own window, not ramped"


def test_downsampling_a_fine_signal_integrates_rather_than_samples() -> None:
    """Going coarser must integrate, not sample.

    Point-sampling a 17 ms posterior at 250 ms throws away 14 of every 15 measurements and
    which survives is arbitrary; averaging keeps what they collectively said.
    """
    fine_times = [i * 0.05 for i in range(20)]
    fine_values = [1.0 if i < 10 else 0.0 for i in range(20)]
    _t, values = resample_series(fine_times, fine_values, target_hop_s=0.5, duration_s=1.0, kind="mean")
    assert values[0] == pytest.approx(1.0)
    assert values[1] == pytest.approx(0.0)


def test_averaging_does_not_saturate_a_partly_active_bucket() -> None:
    """The defect this fixes: a bucket containing one speech frame read as fully active."""
    fine_times = [i * 0.05 for i in range(20)]
    fine_values = [1.0] + [0.0] * 19
    _t, values = resample_series(fine_times, fine_values, target_hop_s=1.0, duration_s=1.0, kind="mean")
    assert values[0] < 0.2


def test_a_bucket_with_no_source_samples_is_not_measured() -> None:
    """A gap must stay a gap; zero would assert the signal reported absence there."""
    import math

    _t, values = resample_series([0.0], [0.5], target_hop_s=0.5, duration_s=1.5, kind="mean")
    assert math.isnan(values[-1])


def test_the_native_table_is_keyed_by_signal_family() -> None:
    """Resolutions belong with the extractor that produces them, not scattered at call sites."""
    assert "frame_" in " ".join(NATIVE_RESOLUTION_S)


# ── a constant posterior is not evidence ───────────────────────────────


def test_multilabel_activations_reduce_by_noisy_or_not_max() -> None:
    """Measured: the max over three speaker channels gave *exactly* 1.0000 in all 1070 buckets.

    A voice detector reporting speech everywhere across a conversation with four clear pauses
    is not a measurement. The max saturates the moment any one channel is confident; the
    noisy-or, P(at least one speaker), keeps quiet frames quiet.
    """
    import numpy as np

    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import collapse_to_speech_prob

    quiet = np.array([[0.02, 0.01, 0.03]])
    loud = np.array([[0.95, 0.10, 0.05]])
    assert collapse_to_speech_prob(quiet, channel_format="per_speaker")[0] < 0.1
    assert collapse_to_speech_prob(loud, channel_format="per_speaker")[0] > 0.9


def test_the_powerset_path_requires_a_declaration_not_a_row_sum() -> None:
    """Rows summing to 1 are no longer taken as evidence of powerset output.

    This test previously asserted the opposite — that a softmax-looking row selects the
    ``1 - P(no speaker)`` path. Measured against the real model, that inference is what produced
    the saturation: segmentation-3.0 returns per-speaker activations, and a single fully-active
    speaker makes those rows sum to exactly 1.0, so the row-sum test misread column 0 as the
    no-speaker class and reported P(speech) = 1.0000 in 100% of frames including silence.
    """
    import numpy as np

    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import (
        channel_format_for,
        collapse_to_speech_prob,
    )

    rows_sum_to_one = np.array([[0.7, 0.1, 0.1, 0.1]])
    # Three declared speakers, four columns: neither 3 (per-speaker) nor 7 (powerset), so the
    # safe reading is per-speaker rather than assuming a no-speaker column exists.
    fmt = channel_format_for(n_columns=4, declared_classes=["speaker#1", "speaker#2", "speaker#3"])
    assert fmt == "per_speaker"
    assert collapse_to_speech_prob(rows_sum_to_one, channel_format=fmt)[0] == pytest.approx(1.0 - 0.3 * 0.9 * 0.9 * 0.9)
    # A model that genuinely declares powerset width gets the powerset reduction.
    assert collapse_to_speech_prob(rows_sum_to_one, channel_format="powerset")[0] == pytest.approx(0.3)
