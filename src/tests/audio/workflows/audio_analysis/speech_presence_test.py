"""Tests for the absolute-scale acoustic speech-presence signals (D-3, register items 8-9).

These cross the L1/L2 boundary on purpose. The property being pinned — that LUFS may assert absence
while level-above-floor must abstain — is only observable once the measurement has been read as a
belief, so each test harvests evidence and then links it. Splitting them would test each half
against an expectation neither half is responsible for.

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
    from senselab.audio.workflows.audio_analysis.speech_presence import harvest_speech_presence_evidence
    from senselab.audio.workflows.audio_analysis.speech_presence_link import link_speech_presence

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
    evidence = harvest_speech_presence_evidence(
        pass_summary=pass_summary,
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
        speech_presence_labels=["Speech"],
        alignment_by_model={},
        waveform=np.zeros(16000 * 2, dtype=np.float64),
        sampling_rate=16000,
    )
    buckets = link_speech_presence(evidence)
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
    from senselab.audio.workflows.audio_analysis.speech_presence import harvest_speech_presence_evidence
    from senselab.audio.workflows.audio_analysis.speech_presence_link import link_speech_presence

    sr = 16000
    t = np.arange(sr * 2) / sr
    y = 0.15 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    evidence = harvest_speech_presence_evidence(
        pass_summary={"duration_s": 2.0},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
        speech_presence_labels=["Speech"],
        alignment_by_model={},
        waveform=y,
        sampling_rate=sr,
    )
    buckets = link_speech_presence(evidence)
    votes = buckets[0]["votes"]
    assert votes["acoustic_lufs"]["speaks"] is True
    # A wall-to-wall tone is its own floor, so the excess measure abstains rather than asserting.
    # Recorded explicitly because it is a real limit of the measure, not an accident of this input.
    assert votes["acoustic_level_above_floor"]["native_confidence"] >= 0.5


def test_absolute_voters_absent_without_audio() -> None:
    """No waveform means no vote, rather than a vote from no measurement."""
    from senselab.audio.workflows.audio_analysis.grid import BucketGrid
    from senselab.audio.workflows.audio_analysis.speech_presence import harvest_speech_presence_evidence
    from senselab.audio.workflows.audio_analysis.speech_presence_link import link_speech_presence

    evidence = harvest_speech_presence_evidence(
        pass_summary={"duration_s": 1.0},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
        speech_presence_labels=["Speech"],
        alignment_by_model={},
    )
    buckets = link_speech_presence(evidence)
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
    from senselab.audio.workflows.audio_analysis.speech_presence import harvest_speech_presence_evidence
    from senselab.audio.workflows.audio_analysis.speech_presence_link import link_speech_presence

    sr = 16000
    t = np.arange(sr * 4) / sr
    y = np.zeros_like(t)
    y[sr * 2 :] = 0.2 * np.sin(2 * np.pi * 220 * t[sr * 2 :])  # silent first half, tone second
    evidence = harvest_speech_presence_evidence(
        pass_summary={"duration_s": 4.0},
        grid=BucketGrid(win_length=0.5, hop_length=0.5),
        speech_presence_labels=["Speech"],
        alignment_by_model={},
        waveform=y,
        sampling_rate=sr,
    )
    buckets = link_speech_presence(evidence)
    quiet = [b for b in buckets if b["end"] <= 2.0]
    active = [b for b in buckets if b["start"] >= 2.0]
    assert quiet and active
    quiet_nc = [b["votes"]["acoustic_level_above_floor"]["native_confidence"] for b in quiet]
    active_speaks = [b["votes"]["acoustic_level_above_floor"]["speaks"] for b in active]
    assert all(nc == pytest.approx(0.5) for nc in quiet_nc), "quiet stretches abstain"
    assert all(active_speaks), "the tone is well above the measured floor"


# ── evidence carried alongside the vote (register items 1-5, 13-15) ──────────


def test_diar_coverage_distinguishes_a_grazing_segment_from_a_full_one() -> None:
    """``speaks`` is True for both, so the bool alone cannot tell them apart."""
    from senselab.audio.workflows.audio_analysis.harvesters import diar_covered_fraction

    full = [[{"start": 0.0, "end": 1.0, "speaker": "S"}]]
    grazing = [[{"start": 0.95, "end": 3.0, "speaker": "S"}]]
    assert diar_covered_fraction(full, 0.0, 1.0) == pytest.approx(1.0)
    assert diar_covered_fraction(grazing, 0.0, 1.0) == pytest.approx(0.05)


def test_diar_coverage_unions_overlapping_speakers() -> None:
    """Two speakers talking at once must not report more than a bucket's worth of coverage."""
    from senselab.audio.workflows.audio_analysis.harvesters import diar_covered_fraction

    both = [[{"start": 0.0, "end": 1.0, "speaker": "A"}, {"start": 0.0, "end": 1.0, "speaker": "B"}]]
    assert diar_covered_fraction(both, 0.0, 1.0) == pytest.approx(1.0)


def test_diar_coverage_is_none_without_segments() -> None:
    """An absent model is not a model reporting zero coverage."""
    from senselab.audio.workflows.audio_analysis.harvesters import diar_covered_fraction

    assert diar_covered_fraction(None, 0.0, 1.0) is None
    assert diar_covered_fraction([[]], 0.0, 1.0) is None


def test_frame_dispersion_is_reported_in_probability_units() -> None:
    """L1 reports the dispersion unrescaled; the [0, 1] mapping is L2's modelling choice.

    Previously L1 emitted ``clip(2 * std, 0, 1)``. Doubling turns a dispersion into something that
    reads like a probability, and the clip then hides where the rescale was wrong.
    """
    import numpy as np

    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import FramePosterior
    from senselab.audio.workflows.audio_analysis.grid import BucketGrid
    from senselab.audio.workflows.audio_analysis.speech_presence import harvest_speech_presence_evidence
    from senselab.audio.workflows.audio_analysis.speech_presence_link import link_speech_presence
    from senselab.audio.workflows.audio_analysis.votes import MAX_PROBABILITY_STD, _dispersion_to_instability

    # Half-0 / half-1 across the bucket: the maximal dispersion for a bounded value, std = 0.5.
    probs = np.concatenate([np.zeros(50), np.ones(50)])
    fp = FramePosterior(activations=probs[:, None], frame_hop_s=0.01, channel_format="single")
    evidence = harvest_speech_presence_evidence(
        pass_summary={"duration_s": 1.0},
        grid=BucketGrid(win_length=1.0, hop_length=1.0),
        speech_presence_labels=["Speech"],
        alignment_by_model={},
        frame_posteriors={"frame_segmentation": fp},
    )
    buckets = link_speech_presence(evidence)
    dispersion = buckets[0]["frame_dispersion"]
    assert dispersion == pytest.approx(MAX_PROBABILITY_STD, abs=0.02), "reported in probability units"
    # L2 maps it, and only there does it become a [0, 1] quantity.
    assert _dispersion_to_instability(dispersion) == pytest.approx(1.0, abs=0.05)
    assert _dispersion_to_instability(0.0) == pytest.approx(0.0)
    assert _dispersion_to_instability(None) is None


def test_frame_vote_keeps_per_speaker_channels() -> None:
    """D-5: which channel was active survives onto the vote, not just the pooled value."""
    import numpy as np

    from senselab.audio.tasks.voice_activity_detection.frame_posteriors import FramePosterior
    from senselab.audio.workflows.audio_analysis.grid import BucketGrid
    from senselab.audio.workflows.audio_analysis.speech_presence import harvest_speech_presence_evidence
    from senselab.audio.workflows.audio_analysis.speech_presence_link import link_speech_presence

    activations = np.tile(np.array([[0.0, 1.0, 0.0]]), (100, 1))
    fp = FramePosterior(
        activations=activations,
        frame_hop_s=0.01,
        channel_format="per_speaker",
        channel_labels=("speaker#1", "speaker#2", "speaker#3"),
    )
    evidence = harvest_speech_presence_evidence(
        pass_summary={"duration_s": 1.0},
        grid=BucketGrid(win_length=1.0, hop_length=1.0),
        speech_presence_labels=["Speech"],
        alignment_by_model={},
        frame_posteriors={"frame_segmentation": fp},
    )
    buckets = link_speech_presence(evidence)
    vote = buckets[0]["votes"]["frame_segmentation"]
    assert vote["channel_means"] == pytest.approx([0.0, 1.0, 0.0])
    assert vote["channel_labels"] == ["speaker#1", "speaker#2", "speaker#3"]
    assert vote["resolution_s"] == pytest.approx(0.01)
