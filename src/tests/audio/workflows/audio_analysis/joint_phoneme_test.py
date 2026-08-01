"""J7 — let the acoustics adjudicate where the ASR models disagree.

PPG posteriors are an independent witness: they come from the audio without going through any
language model, so where two transcripts read the same span differently the phoneme evidence can
favour one without simply repeating an ASR model's opinion. That is what makes J7 worth having over
the per-window PER the ASR axis already computes — the existing measure asks "does this model's
transcript match the audio", this one asks "which of the readings on the table does".
"""

from __future__ import annotations

import pytest

from senselab.audio.workflows.audio_analysis.harmonize import harmonize_transcripts
from senselab.audio.workflows.audio_analysis.joint import phoneme_transcript_agreement


def _frames(runs: list[tuple[str, int]]) -> list[str]:
    out: list[str] = []
    for label, n in runs:
        out.extend([label] * n)
    return out


def test_the_acoustics_pick_the_reading_they_support() -> None:
    """Two models disagree; the phoneme frames match one of them."""
    a = [(0.0, 0.5, "cat")]
    b = [(0.0, 0.5, "dog")]
    slots = harmonize_transcripts({"a": a, "b": b}).slots
    # /k ae t/ — the reading "cat", spoken over the slot.
    frames = _frames([("k", 10), ("ae", 20), ("t", 10)])
    result = phoneme_transcript_agreement(slots, ppg_per_frame=frames, ppg_frame_hop=0.0125)
    assert result is not None and len(result) == 1
    assert result[0]["acoustic_choice"] == "cat"
    assert result[0]["per"]["cat"] < result[0]["per"]["dog"]


def test_a_unanimous_slot_still_reports_whether_the_audio_backs_it() -> None:
    """Everyone can be wrong together, so agreement among models is not agreement with the audio."""
    words = [(0.0, 0.5, "cat")]
    slots = harmonize_transcripts({"a": words, "b": list(words)}).slots
    backed = phoneme_transcript_agreement(
        slots, ppg_per_frame=_frames([("k", 10), ("ae", 20), ("t", 10)]), ppg_frame_hop=0.0125
    )
    contradicted = phoneme_transcript_agreement(
        slots, ppg_per_frame=_frames([("s", 10), ("iy", 20), ("l", 10)]), ppg_frame_hop=0.0125
    )
    assert backed is not None and contradicted is not None
    assert backed[0]["per"]["cat"] < contradicted[0]["per"]["cat"]
    # One candidate means no choice to make, so the *selection* carries no doubt either way.
    assert backed[0]["uncertainty"] == pytest.approx(0.0)


def test_equally_supported_readings_leave_the_choice_open() -> None:
    """When the acoustics cannot separate the candidates, J7 must not pick one."""
    slots = harmonize_transcripts({"a": [(0.0, 0.5, "cat")], "b": [(0.0, 0.5, "cat")]}).slots
    # Force a two-candidate slot by hand with frames matching neither.
    slots[0].words = {"a": "cat", "b": "dog"}
    slots[0].consensus = None
    result = phoneme_transcript_agreement(slots, ppg_per_frame=_frames([("zh", 40)]), ppg_frame_hop=0.0125)
    assert result is not None
    assert result[0]["acoustic_choice"] is None
    assert result[0]["uncertainty"] > 0.9


def test_a_slot_with_no_phoneme_frames_makes_no_claim() -> None:
    """Outside the PPG's coverage there is no acoustic witness to consult."""
    slots = harmonize_transcripts({"a": [(9.0, 9.5, "cat")]}).slots
    result = phoneme_transcript_agreement(slots, ppg_per_frame=_frames([("k", 10)]), ppg_frame_hop=0.0125)
    assert result == []


def test_no_ppg_means_no_adjudication_rather_than_a_default_verdict() -> None:
    """An absent witness must not be read as agreement."""
    slots = harmonize_transcripts({"a": [(0.0, 0.5, "cat")]}).slots
    assert phoneme_transcript_agreement(slots, ppg_per_frame=[], ppg_frame_hop=0.0125) is None
    assert phoneme_transcript_agreement(slots, ppg_per_frame=["k"], ppg_frame_hop=0.0) is None
