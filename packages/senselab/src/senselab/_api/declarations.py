"""The tasks shared by their backends, and the modules whose import declares the tasks and backends of the API."""

from typing import Literal

from pydantic import Field
from senselab_core.kernel.declare import REGISTRY, TaskParams, task
from senselab_core.kernel.results import Embedding, Intervals, Transcript
from senselab_core.kernel.signals import Audio

DECLARATION_MODULES: tuple[str, ...] = (
    "senselab._api.declarations",
    "senselab.signal.resample",
    "senselab.audio.tasks.speech_to_text.huggingface",
    "senselab.audio.tasks.speech_to_text.crisperwhisper_backend",
    "senselab.audio.tasks.speaker_diarization.pyannote_backend",
    "senselab.audio.tasks.speaker_embeddings.speechbrain_backend",
)
"""Importable modules declaring tasks and backends in `senselab_core.kernel.declare.REGISTRY`, in declaration order."""


class SpeechToTextParams(TaskParams):
    """Params of `speech_to_text`: `language` an ISO 639-1 code (detected when None), `timestamps` the finest level."""

    language: str | None = None
    timestamps: Literal["word", "segment", "none"] = "word"


@task(name="speech_to_text", version="1", kind="transform", registry=REGISTRY)
class SpeechToText:
    """Transcribe mono speech to a `Transcript`: segments, each dividing into words when `timestamps` is `word`."""

    inputs = {"audio": Audio}
    outputs = {"transcript": Transcript}
    Params = SpeechToTextParams


class DiarizeParams(TaskParams):
    """Params of `diarize`: the speaker count when known (`num_speakers`) or its bounds, each at least 1."""

    num_speakers: int | None = Field(default=None, ge=1)
    min_speakers: int | None = Field(default=None, ge=1)
    max_speakers: int | None = Field(default=None, ge=1)


@task(name="diarize", version="1", kind="transform", registry=REGISTRY)
class Diarize:
    """Find who speaks when in mono speech: speaker turns as `Intervals`, labelled by speaker, ordered by start."""

    inputs = {"audio": Audio}
    outputs = {"turns": Intervals}
    Params = DiarizeParams


@task(name="speaker_embeddings", version="1", kind="transform", registry=REGISTRY)
class SpeakerEmbeddings:
    """Embed mono speech as one speaker `Embedding`, in the space of the model that produced it."""

    inputs = {"audio": Audio}
    outputs = {"embedding": Embedding}
