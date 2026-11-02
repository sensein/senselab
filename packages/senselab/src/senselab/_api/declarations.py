"""The tasks shared by their backends, and the modules whose import declares the tasks and backends of the API."""

from typing import Literal

from pydantic import Field
from senselab_core.kernel.declare import TaskParams, task
from senselab_core.kernel.results import Embedding, Intervals, TextSpans, Transcript
from senselab_core.kernel.signals import Audio, Text

from senselab.text.tasks.pii_spans.profile import PiiProfile, packaged_profile

DECLARATION_MODULES: tuple[str, ...] = (
    "senselab._api.declarations",
    "senselab.signal.resample",
    "senselab.metrics.wer",
    "senselab.metrics.der",
    "senselab.metrics.eer",
    "senselab.metrics.span_f1",
    "senselab.metrics.resample_error",
    "senselab.audio.tasks.speech_to_text.huggingface",
    "senselab.audio.tasks.speech_to_text.crisperwhisper_backend",
    "senselab.audio.tasks.speaker_diarization.pyannote_backend",
    "senselab.audio.tasks.speaker_embeddings.speechbrain_backend",
    "senselab.text.tasks.pii_spans.transformers_backend",
    "senselab.text.tasks.pii_spans.vllm_backend",
)
"""Importable modules whose classes are decorated as tasks and backends, in declaration order."""


class SpeechToTextParams(TaskParams):
    """Params of `speech_to_text`: `language` an ISO 639-1 code (detected when None), `timestamps` the finest level."""

    language: str | None = None
    timestamps: Literal["word", "segment", "none"] = "word"


@task(name="speech_to_text", version="1", kind="transform")
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


@task(name="diarize", version="1", kind="transform")
class Diarize:
    """Find who speaks when in mono speech: speaker turns as `Intervals`, labelled by speaker, ordered by start."""

    inputs = {"audio": Audio}
    outputs = {"turns": Intervals}
    Params = DiarizeParams


@task(name="speaker_embeddings", version="1", kind="transform")
class SpeakerEmbeddings:
    """Embed mono speech as one speaker `Embedding`, in the space of the model that produced it."""

    inputs = {"audio": Audio}
    outputs = {"embedding": Embedding}


class PiiSpansParams(TaskParams):
    """Params of `pii_spans`: the `profile` slice the prompt is built from, and the task `family` selecting guidance."""

    profile: PiiProfile = Field(
        default_factory=packaged_profile,
        description=(
            "The slice the prompt is built from: an `instruction`, the `categories` a span may carry (each a "
            "`name` in capitals and a `description`) and optional `families` of guidance. Omitted, the packaged one."
        ),
        examples=[
            {
                "instruction": "Mark every span of personally identifying information in the text.",
                "categories": [
                    {"name": "PERSON", "description": "A person's name."},
                    {"name": "EMAIL", "description": "An email address."},
                ],
            }
        ],
    )
    family: str | None = Field(default=None, description="The task family selecting guidance from the profile.")


@task(name="pii_spans", version="1", kind="transform")
class PiiSpans:
    """Find the spans of a text that identify a person: `TextSpans` labelled by category, each with a score."""

    inputs = {"text": Text}
    outputs = {"spans": TextSpans}
    Params = PiiSpansParams
