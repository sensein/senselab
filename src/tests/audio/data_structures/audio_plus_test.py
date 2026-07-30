"""Unit tests for Audio+ construction from an incoming reference."""

from __future__ import annotations

import torch

from senselab.audio.data_structures import Audio
from senselab.audio.data_structures.audio_plus import (
    AudioPlus,
    AudioPlusMetadata,
    MetadataProvider,
    SpeakerInfo,
    TaskInfo,
    build_audio_plus,
)


def _fake_audio() -> Audio:
    """Return a tiny in-memory mono 16 kHz Audio (no file IO)."""
    return Audio(waveform=torch.zeros(1, 16000), sampling_rate=16000)


def test_build_audio_plus_defaults_to_empty_metadata() -> None:
    """With no provider, Audio+ carries the audio and empty task/speaker context."""
    seen: list[str] = []

    def loader(ref: str) -> Audio:
        seen.append(ref)
        return _fake_audio()

    result = build_audio_plus("s3://bucket/rec.wav", audio_loader=loader)

    assert isinstance(result, AudioPlus)
    assert result.ref == "s3://bucket/rec.wav"
    assert result.audio.sampling_rate == 16000
    assert result.task.name is None
    assert result.speaker.gsd is None
    assert result.related_audio_refs == []
    assert seen == ["s3://bucket/rec.wav"]  # loader was called with the incoming ref


def test_build_audio_plus_joins_provider_metadata() -> None:
    """A metadata provider's task/speaker/related-refs are joined onto Audio+."""

    class FakeProvider:
        """Returns fixed b2aiprep-like context for any ref."""

        def lookup(self, ref: str) -> AudioPlusMetadata:
            """Return canned metadata."""
            return AudioPlusMetadata(
                task=TaskInfo(name="Audio-Check", content="Say the days of the week"),
                speaker=SpeakerInfo(speaker_id="sub-01", gsd="MCI", age=71.0),
                related_audio_refs=["s3://bucket/sub-01/other-1.wav", "s3://bucket/sub-01/other-2.wav"],
            )

    result = build_audio_plus(
        "s3://bucket/sub-01/rec.wav",
        audio_loader=lambda _ref: _fake_audio(),
        metadata_provider=FakeProvider(),
    )

    assert result.task.name == "Audio-Check"
    assert result.speaker.gsd == "MCI"
    assert result.speaker.age == 71.0
    assert len(result.related_audio_refs) == 2


def test_load_related_audios_uses_loader_per_ref() -> None:
    """Related audios are materialized lazily, one loader call per reference."""
    result = build_audio_plus(
        "s3://bucket/sub-01/rec.wav",
        audio_loader=lambda _ref: _fake_audio(),
        metadata_provider=_provider_with_related(["a.wav", "b.wav", "c.wav"]),
    )

    calls: list[str] = []

    def loader(ref: str) -> Audio:
        calls.append(ref)
        return _fake_audio()

    related = result.load_related_audios(loader)
    assert len(related) == 3
    assert calls == ["a.wav", "b.wav", "c.wav"]


def _provider_with_related(refs: list[str]) -> MetadataProvider:
    """Build a metadata provider that only sets related_audio_refs."""

    class _Provider:
        def lookup(self, ref: str) -> AudioPlusMetadata:
            """Return metadata carrying only the related refs."""
            return AudioPlusMetadata(related_audio_refs=refs)

    return _Provider()


def test_related_refs_exclude_self_makes_enrollment_leave_one_out() -> None:
    """A provider that omits the queried ref is what makes a sibling-built profile LOO.

    Enrolling from ``related_audio_refs`` must not include the recording being analyzed —
    otherwise it contributes to the reference it is scored against, which biases every
    window toward "target" and hides contamination in exactly that file.
    """
    ref = "s3://bucket/sub-01/rec.wav"
    siblings = ["s3://bucket/sub-01/other-1.wav", "s3://bucket/sub-01/other-2.wav"]

    class _SelfExcludingProvider:
        def lookup(self, other: str) -> AudioPlusMetadata:
            """Return the speaker's recordings minus the queried one."""
            return AudioPlusMetadata(related_audio_refs=[r for r in [*siblings, other] if r != other])

    result = build_audio_plus(ref, audio_loader=lambda _r: _fake_audio(), metadata_provider=_SelfExcludingProvider())
    assert ref not in result.related_audio_refs
    assert result.related_audio_refs == siblings


def test_null_provider_satisfies_the_protocol() -> None:
    """NullMetadataProvider must be usable anywhere a MetadataProvider is expected."""
    from senselab.audio.data_structures.audio_plus import NullMetadataProvider

    assert isinstance(NullMetadataProvider(), MetadataProvider)
    assert NullMetadataProvider().lookup("anything") == AudioPlusMetadata()
