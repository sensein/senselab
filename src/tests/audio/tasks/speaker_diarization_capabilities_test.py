"""Declared capabilities for the diarization backends."""

import dataclasses

import pytest

from senselab.audio.tasks.speaker_diarization.capabilities import DiarizationCapabilities


def test_record_is_frozen() -> None:
    """A capability record is a declaration, not mutable state.

    If a caller could mutate one, two callers would disagree about what a backend
    can do, and the disagreement would depend on import order.
    """
    caps = DiarizationCapabilities(
        populates_text=False,
        speaker_label_kind="identity",
        labels_stable_across_files=False,
        max_speakers=None,
        honors_speaker_hints=False,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        caps.max_speakers = 4  # type: ignore[misc]


def test_max_speakers_none_is_allowed_and_means_unmeasured() -> None:
    """None is 'nobody has measured this', not 'unlimited'.

    Four of six backends have no published or measured ceiling. Encoding that as
    None keeps it distinguishable from a real limit, so the NeMo probe can fill it
    in later without anyone having guessed in the meantime.
    """
    caps = DiarizationCapabilities(
        populates_text=False,
        speaker_label_kind="identity",
        labels_stable_across_files=False,
        max_speakers=None,
        honors_speaker_hints=False,
    )
    assert caps.max_speakers is None


def test_max_speakers_must_be_at_least_one_when_given() -> None:
    """A ceiling of zero or less describes nothing that can diarize.

    Catches a typo or an off-by-one in a declaration at construction time rather
    than as a confusing empty result much later.
    """
    with pytest.raises(ValueError, match="max_speakers"):
        DiarizationCapabilities(
            populates_text=False,
            speaker_label_kind="identity",
            labels_stable_across_files=False,
            max_speakers=0,
            honors_speaker_hints=False,
        )


def test_speaker_label_kind_rejects_an_unknown_value() -> None:
    """Only 'identity' and 'role' are meaningful.

    The distinction decides whether labels may reach embedding clustering, so a
    third value silently defaulting to one branch would be a correctness bug.
    """
    with pytest.raises(ValueError, match="speaker_label_kind"):
        DiarizationCapabilities(
            populates_text=False,
            speaker_label_kind="cluster",  # type: ignore[arg-type]
            labels_stable_across_files=False,
            max_speakers=None,
            honors_speaker_hints=False,
        )
