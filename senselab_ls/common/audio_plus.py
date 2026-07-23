"""Audio+ -- an enriched audio bundle that any Label Studio backend can ingest.

An Audio+ wraps a senselab :class:`~senselab.audio.data_structures.Audio` with the joined
context our BIDS-like (b2aiprep) dataset provides: the acquisition task, the speaker's
phenotype (gold standard diagnosis, age), and references to the speaker's other recordings
for profile building.

The bundle is generated *on demand* from the audio reference a backend receives -- see
:func:`build_audio_plus`. Metadata resolution is pluggable via :class:`MetadataProvider` so
the gated b2aiprep records stay behind an injectable boundary and tests can mock them.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from senselab.audio.data_structures import Audio


class TaskInfo(BaseModel):
    """The acquisition task a recording belongs to (from the b2aiprep dataset)."""

    name: Optional[str] = None
    content: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SpeakerInfo(BaseModel):
    """Speaker phenotype joined from the b2aiprep-generated metadata."""

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
    """Resolves an incoming audio reference to its b2aiprep-derived context."""

    def lookup(self, ref: str) -> AudioPlusMetadata:
        """Return the joined metadata for ``ref``."""
        ...


class NullMetadataProvider:
    """Fallback provider that returns empty metadata (a bytes-only Audio+)."""

    def lookup(self, ref: str) -> AudioPlusMetadata:
        """Return empty :class:`AudioPlusMetadata` regardless of ``ref``."""
        return AudioPlusMetadata()


class AudioPlus(BaseModel):
    """Enriched audio bundle ingested by any backend.

    Holds the waveform (senselab ``Audio``) plus the joined task/speaker context and the
    *references* (not the loaded audio) of the speaker's related recordings. Related audios
    are materialized lazily via :meth:`load_related_audios` so a speaker's whole session is
    never eagerly loaded. ``recording_id`` is the dataset's stable id for this recording and is
    what a prediction is keyed by when written back into an annotation.
    """

    ref: str
    audio: Audio
    recording_id: Optional[str] = None
    task: TaskInfo = Field(default_factory=TaskInfo)
    speaker: SpeakerInfo = Field(default_factory=SpeakerInfo)
    related_audio_refs: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}

    def load_related_audios(self, loader: Callable[[str], Audio]) -> list[Audio]:
        """Materialize the speaker's related audios on demand.

        Args:
            loader: Maps a reference to a senselab ``Audio`` (e.g. ``audio_io.load_audio``).

        Returns:
            The related recordings as loaded ``Audio`` objects.
        """
        return [loader(related_ref) for related_ref in self.related_audio_refs]


def build_audio_plus(
    ref: str,
    *,
    audio_loader: Callable[[str], Audio],
    metadata_provider: Optional[MetadataProvider] = None,
) -> AudioPlus:
    """Generate or grab an :class:`AudioPlus` from the reference a backend received.

    This is the first step in any backend's ``predict``: it turns the bare audio reference
    the ML SDK sends in into the enriched bundle the analyzers consume.

    Args:
        ref: The incoming audio reference (``s3://`` key, URL, or local path).
        audio_loader: Resolves ``ref`` to a senselab ``Audio`` (see ``audio_io.load_audio``).
        metadata_provider: Joins the b2aiprep task / speaker (GSD, age) / related-audio
            context. Defaults to :class:`NullMetadataProvider` (no external metadata).

    Returns:
        The enriched :class:`AudioPlus` bundle.
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
