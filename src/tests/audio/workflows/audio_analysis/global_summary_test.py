"""Tests for the speaker-profile extension of the single_speaker claim.

Covers the additive ``profile_other_voice`` parameter on
``compute_pass_global_summary``: when absent the claim is unchanged; when
present its sub-signals appear under ``single_speaker`` and the p95 other-voice
uncertainty folds into the headline via ``max()``.
"""

from __future__ import annotations

from typing import Any

from senselab.audio.workflows.audio_analysis.global_summary import compute_pass_global_summary

_PASS = "raw_16k"


def _call(profile_other_voice: dict[str, Any] | None) -> dict[str, Any]:
    return compute_pass_global_summary(
        pass_label=_PASS,
        pass_summary={"duration_s": 10.0},
        axis_results={},
        asr_resolved={},
        pii_report=None,
        expects_speech=True,
        profile_other_voice=profile_other_voice,
    )


def test_single_speaker_unchanged_without_profile() -> None:
    """No profile → the claim carries no profile_* keys (byte-identical path)."""
    ss = _call(None)["single_speaker"]
    assert not any(k.startswith("profile_") for k in ss)


def test_profile_subsignals_added_and_folded() -> None:
    """A profile rollup adds sub-signals and folds p95 into the headline via max()."""
    rollup = {
        "profile_other_voice_fraction": 1.0,
        "profile_other_voice_seconds": 2.8,
        "profile_peak_other_voice_uncertainty": 1.0,
        "profile_p95_other_voice_uncertainty": 0.9,
        "profile_speech_present_seconds": 2.8,
        "profile_confidence": "ok",
    }
    ss = _call(rollup)["single_speaker"]
    assert ss["profile_other_voice_fraction"] == 1.0
    assert ss["profile_confidence"] == "ok"
    # With no diar/identity signal the headline starts at None; the profile p95
    # becomes the headline.
    assert ss["uncertainty"] == 0.9


def test_profile_fold_takes_max_with_existing() -> None:
    """The fold never lowers an existing higher single_speaker uncertainty."""
    low = {
        "profile_other_voice_fraction": 0.0,
        "profile_other_voice_seconds": 0.0,
        "profile_peak_other_voice_uncertainty": 0.1,
        "profile_p95_other_voice_uncertainty": 0.05,
        "profile_speech_present_seconds": 5.0,
        "profile_confidence": "ok",
    }
    ss = _call(low)["single_speaker"]
    # No other signal present → headline is the (low) profile p95.
    assert ss["uncertainty"] == 0.05
