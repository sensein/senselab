"""Declared hints on an Audio.

A hint is an assertion by whoever knows the acquisition protocol -- never a measurement, and
never consumed by any task in this change. These tests pin the two properties that make
"declared and carried" true: absent stays distinguishable from empty, and a hint cannot change
what a computation returns.
"""

import torch

from senselab.audio.data_structures import Audio
from senselab.audio.data_structures.audio_hints import (
    AudioHints,
    ExpectedSpeech,
    SpeakerEmbeddingProvenance,
    TargetSpeakerEmbedding,
)


def test_an_audio_carries_no_hints_by_default() -> None:
    """Absent must stay distinguishable from empty.

    An empty AudioHints would make "nobody declared anything" read the same as "declared
    nothing" -- the same collapse as reading a None confidence as 0.0, which pii_detection
    documents at length.
    """
    audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000)
    assert audio.hints is None


def test_hints_hold_every_declared_field() -> None:
    """Every hint in the request round-trips through the model."""
    hints = AudioHints(
        may_contain=["read-speech", "cough"],
        targeted_speaker_count=1,
        environment="quiet-room",
        expected_speech=[
            ExpectedSpeech(text="The quick brown fox.", prompt_id="harvard-01", reference="ieee-1969"),
            ExpectedSpeech(text="Rice is often served in round bowls.", prompt_id="harvard-02"),
        ],
    )
    audio = Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000, hints=hints)
    assert audio.hints is not None
    assert audio.hints.may_contain == ["read-speech", "cough"]
    assert audio.hints.targeted_speaker_count == 1
    assert audio.hints.environment == "quiet-room"
    assert len(audio.hints.expected_speech) == 2
    assert audio.hints.expected_speech[0].prompt_id == "harvard-01"


def test_expected_speech_preserves_order() -> None:
    """A file often holds several sentences read in sequence.

    Concatenating them would destroy the boundaries a matcher needs to say *which* sentence was
    skipped or reordered -- a different question from how close the whole thing was.
    """
    hints = AudioHints(
        expected_speech=[ExpectedSpeech(text="first"), ExpectedSpeech(text="second"), ExpectedSpeech(text="third")]
    )
    assert [e.text for e in hints.expected_speech] == ["first", "second", "third"]


def test_provenance_records_a_resolved_sha_or_says_why_not() -> None:
    """A ref in the commit-sha field would be provenance that is confidently wrong.

    #550 established that recording a ref while claiming a commit is worse than recording
    nothing, so an unresolved model must set unresolved_reason instead.
    """
    resolved = SpeakerEmbeddingProvenance(model_id="speechbrain/spkrec-ecapa-voxceleb", model_commit_sha="a" * 40)
    assert resolved.model_commit_sha == "a" * 40
    assert resolved.unresolved_reason is None

    unresolved = SpeakerEmbeddingProvenance(
        model_id="speechbrain/spkrec-ecapa-voxceleb",
        model_commit_sha=None,
        unresolved_reason="offline: hub unreachable and no cached ref",
    )
    assert unresolved.model_commit_sha is None
    assert unresolved.unresolved_reason


def test_a_non_sha_commit_value_is_rejected() -> None:
    """The field means "resolved commit". Anything ref-shaped in it defeats the point."""
    import pytest

    with pytest.raises(ValueError, match="40"):
        SpeakerEmbeddingProvenance(model_id="org/model", model_commit_sha="main")


def test_a_target_speaker_embedding_carries_its_provenance() -> None:
    """A vector with no provenance cannot be interpreted later, so provenance is required."""
    emb = TargetSpeakerEmbedding(
        vector=[0.1, 0.2, 0.3],
        provenance=SpeakerEmbeddingProvenance(model_id="org/model", model_commit_sha="b" * 40),
    )
    assert emb.provenance.model_id == "org/model"
    assert emb.distribution is None


def test_hints_do_not_change_the_cache_key() -> None:
    """A hint nothing consumes must not change what a computation returns.

    Not a backwards-compatibility concern -- alpha owes none -- but a correctness one: if a hint
    moved a cache key, "carried only" would be false.

    ``audio_signature`` is the real waveform-hashing helper (the brief's draft guessed
    ``audio_content_hash``, which does not exist). It is typed against the structural
    ``_HasWaveform`` protocol -- ``waveform`` and ``sampling_rate`` only -- so it cannot read
    ``hints`` even if a caller wanted it to; this test pins that behaviour rather than trusting
    the type alone.
    """
    from senselab.utils.tasks.cached_inference import audio_signature

    waveform = torch.rand(1, 16000)
    bare = Audio(waveform=waveform, sampling_rate=16000)
    hinted = Audio(
        waveform=waveform.clone(),
        sampling_rate=16000,
        hints=AudioHints(may_contain=["read-speech"], targeted_speaker_count=2),
    )
    assert audio_signature(bare) == audio_signature(hinted)
