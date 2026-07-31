"""Tests for the absolute-scale acoustic speech_presence voters (D-3, register items 8-9).

``Loudness_sma3`` and ``spectralFlux_sma3`` were normalised against per-pass percentiles, which
made them ranks rather than levels: ~10% of frames pinned at 0 and ~25% at 1.0 by construction,
and a uniformly quiet recording still spread to fill ``[0, 1]``. They are replaced by ``lufs``
(BS.1770, absolute) and level-above-the-measured-noise-floor (gain-invariant).

The two are asymmetric on purpose, which these tests pin: LUFS can assert absence because -90 LUFS
is unambiguous, while a low excess above floor cannot, because a source running through the whole
recording is absorbed into its own floor estimate.
"""

from __future__ import annotations

import pytest

# ── absolute acoustic voters (D-3, register items 8-9) ───────────────────────


def test_absolute_voters_dissent_from_models_claiming_speech_over_silence() -> None:
    """A silent waveform must be able to contradict a diarizer that reports speech.

    This is the capability the percentile-ranked voters lacked. Because they normalised within the
    recording, a digitally silent file still spread its ranks across ``[0, 1]`` and could report
    "loud" frames -- so no acoustic voter could ever contradict a confident model. LUFS and
    dB-above-floor are absolute, so silence reads as silence and the disagreement surfaces.
    """
    import numpy as np

    from senselab.audio.workflows.audio_analysis.grid import BucketGrid
    from senselab.audio.workflows.audio_analysis.speech_presence import harvest_speech_presence_votes

    pass_summary = {
        "duration_s": 2.0,
        "diarization": {
            "by_model": {
                "pyannote": {
                    "status": "ok",
                    "result": [[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]],
                }
            }
        },
    }
    buckets = harvest_speech_presence_votes(
        pass_summary=pass_summary,
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
        speech_presence_labels=["Speech"],
        alignment_by_model={},
        waveform=np.zeros(16000 * 2, dtype=np.float64),
        sampling_rate=16000,
    )
    assert buckets
    votes = buckets[0]["votes"]
    assert votes["pyannote"]["speaks"] is True, "the mocked diarizer still claims speech"
    # LUFS can assert absence: -90 LUFS is unambiguous, so it contradicts the diarizer outright.
    assert votes["acoustic_lufs"]["speaks"] is False
    assert votes["acoustic_lufs"]["lufs"] < -60.0
    # level-above-floor abstains instead. A low excess is ambiguous between silence and a source
    # that runs through the whole recording and is therefore absorbed into its own floor.
    assert votes["acoustic_level_above_floor"]["native_confidence"] == pytest.approx(0.5)
    assert votes["acoustic_level_above_floor"]["excess_db"] < 3.0


def test_absolute_voters_agree_with_models_on_audible_speech() -> None:
    """The converse: at conversational level LUFS concurs, and the excess measure abstains."""
    import numpy as np

    from senselab.audio.workflows.audio_analysis.grid import BucketGrid
    from senselab.audio.workflows.audio_analysis.speech_presence import harvest_speech_presence_votes

    sr = 16000
    t = np.arange(sr * 2) / sr
    y = 0.15 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    buckets = harvest_speech_presence_votes(
        pass_summary={"duration_s": 2.0},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
        speech_presence_labels=["Speech"],
        alignment_by_model={},
        waveform=y,
        sampling_rate=sr,
    )
    votes = buckets[0]["votes"]
    assert votes["acoustic_lufs"]["speaks"] is True
    # A wall-to-wall tone is its own floor, so the excess measure abstains rather than asserting.
    # Recorded explicitly because it is a real limit of the measure, not an accident of this input.
    assert votes["acoustic_level_above_floor"]["native_confidence"] >= 0.5


def test_absolute_voters_absent_without_audio() -> None:
    """No waveform means no vote, rather than a vote from no measurement."""
    from senselab.audio.workflows.audio_analysis.grid import BucketGrid
    from senselab.audio.workflows.audio_analysis.speech_presence import harvest_speech_presence_votes

    buckets = harvest_speech_presence_votes(
        pass_summary={"duration_s": 1.0},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
        speech_presence_labels=["Speech"],
        alignment_by_model={},
    )
    for b in buckets:
        assert "acoustic_lufs" not in b["votes"]
        assert "acoustic_level_above_floor" not in b["votes"]


def test_level_above_floor_asserts_speech_presence_when_the_recording_has_quiet_stretches() -> None:
    """With a real floor to measure, the excess voter does assert -- which is when it can.

    The pairing with the previous test is the point: the same measure abstains on a continuous
    source and asserts when the recording contains both quiet and active stretches, because only
    then is there a floor distinct from the signal.
    """
    import numpy as np

    from senselab.audio.workflows.audio_analysis.grid import BucketGrid
    from senselab.audio.workflows.audio_analysis.speech_presence import harvest_speech_presence_votes

    sr = 16000
    t = np.arange(sr * 4) / sr
    y = np.zeros_like(t)
    y[sr * 2 :] = 0.2 * np.sin(2 * np.pi * 220 * t[sr * 2 :])  # silent first half, tone second
    buckets = harvest_speech_presence_votes(
        pass_summary={"duration_s": 4.0},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
        speech_presence_labels=["Speech"],
        alignment_by_model={},
        waveform=y,
        sampling_rate=sr,
    )
    quiet = [b for b in buckets if b["end"] <= 2.0]
    active = [b for b in buckets if b["start"] >= 2.0]
    assert quiet and active
    quiet_nc = [b["votes"]["acoustic_level_above_floor"]["native_confidence"] for b in quiet]
    active_speaks = [b["votes"]["acoustic_level_above_floor"]["speaks"] for b in active]
    assert all(nc == pytest.approx(0.5) for nc in quiet_nc), "quiet stretches abstain"
    assert all(active_speaks), "the tone is well above the measured floor"
