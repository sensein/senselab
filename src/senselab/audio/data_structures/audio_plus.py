"""Audio+ — an :class:`~senselab.audio.data_structures.Audio` plus its dataset context.

Many analyses need more than one file's waveform: which acquisition task it is, who the
speaker is, and — the reason this exists here — **which other recordings belong to that
same speaker**. Speaker-profile enrollment pools a subject's recordings, so it needs that
last piece before it can do anything.

The bundle is derived **on demand** from the reference an entry point receives (see
:func:`build_audio_plus`) rather than threaded through every call. Resolution is pluggable
via the :class:`MetadataProvider` protocol, so:

- the library never depends on any particular dataset layout,
- gated datasets stay behind an injectable boundary,
- tests inject a stub instead of standing up a corpus,
- code paths that have no metadata still work — :class:`NullMetadataProvider` yields an
  empty bundle, which is a waveform-only Audio+.

A concrete provider for the Bridge2AI-Voice BIDS layout lives in
:mod:`senselab.audio.metadata.b2ai`.

Related recordings are held as **references, not loaded audio**: a speaker can have
dozens of recordings, and eagerly reading them all to analyze one would be wasteful.
:meth:`AudioPlus.load_related_audios` materializes them when a caller actually needs them.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from senselab.audio.data_structures.audio import Audio


class TaskInfo(BaseModel):
    """The acquisition task a recording belongs to."""

    name: Optional[str] = None
    content: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpeakerInfo(BaseModel):
    """Speaker-level context joined from a dataset's metadata tables.

    ``gsd`` is a gold-standard diagnosis label where the dataset provides one; it is kept
    opaque (a string) because its vocabulary is dataset-specific.
    """

    speaker_id: Optional[str] = None
    gsd: Optional[str] = None
    age: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AudioPlusMetadata(BaseModel):
    """The non-waveform context resolved for a single audio reference."""

    recording_id: Optional[str] = None
    task: TaskInfo = Field(default_factory=TaskInfo)
    speaker: SpeakerInfo = Field(default_factory=SpeakerInfo)
    related_audio_refs: list[str] = Field(default_factory=list)


@runtime_checkable
class MetadataProvider(Protocol):
    """Resolves an audio reference to its dataset context."""

    def lookup(self, ref: str) -> AudioPlusMetadata:
        """Return the joined metadata for ``ref``."""
        ...


class NullMetadataProvider:
    """Fallback provider returning empty metadata (a waveform-only Audio+)."""

    def lookup(self, ref: str) -> AudioPlusMetadata:
        """Return empty :class:`AudioPlusMetadata` regardless of ``ref``."""
        return AudioPlusMetadata()


class AudioPlus(BaseModel):
    """One recording's waveform plus its resolved dataset context.

    Attributes:
        ref: The reference this bundle was built from (path, URL, or object key).
        audio: The loaded waveform.
        recording_id: The dataset's stable id for this recording, where it has one. Useful
            as the key when writing a result back into a dataset or annotation store.
        task: Acquisition-task context.
        speaker: Speaker-level context.
        related_audio_refs: References to the *same speaker's* other recordings — not
            loaded. Use :meth:`load_related_audios`. Excludes ``ref`` itself when the
            provider is well behaved, which is what makes a profile built from these
            leave-one-out by construction.
    """

    ref: str
    audio: Audio
    recording_id: Optional[str] = None
    task: TaskInfo = Field(default_factory=TaskInfo)
    speaker: SpeakerInfo = Field(default_factory=SpeakerInfo)
    related_audio_refs: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def load_related_audios(self, loader: Callable[[str], Audio]) -> list[Audio]:
        """Materialize the speaker's related recordings on demand.

        Args:
            loader: Maps a reference to an :class:`Audio`.

        Returns:
            The related recordings, in ``related_audio_refs`` order. A reference the loader
            raises on is not caught here — the caller decides whether a missing sibling is
            fatal or skippable.
        """
        return [loader(related_ref) for related_ref in self.related_audio_refs]


def build_audio_plus(
    ref: str,
    *,
    audio_loader: Callable[[str], Audio],
    metadata_provider: Optional[MetadataProvider] = None,
) -> AudioPlus:
    """Derive an :class:`AudioPlus` from an audio reference.

    Args:
        ref: The audio reference (local path, URL, or object key).
        audio_loader: Resolves ``ref`` to an :class:`Audio`.
        metadata_provider: Joins task / speaker / related-recording context. Defaults to
            :class:`NullMetadataProvider`, i.e. no external metadata.

    Returns:
        The bundle. Metadata resolution happens before the (more expensive) audio load, so
        a provider that raises on an unknown reference fails before any decoding.
    """
    provider = metadata_provider or NullMetadataProvider()
    meta = provider.lookup(ref)
    audio = audio_loader(ref)
    return AudioPlus(
        ref=ref,
        audio=audio,
        recording_id=meta.recording_id,
        task=meta.task,
        speaker=meta.speaker,
        related_audio_refs=meta.related_audio_refs,
    )
