"""The tasks shared by their backends, and the modules whose import declares the tasks and backends of the API."""

from typing import Literal

from senselab_core.kernel.declare import REGISTRY, TaskParams, task
from senselab_core.kernel.results import Transcript
from senselab_core.kernel.signals import Audio

DECLARATION_MODULES: tuple[str, ...] = (
    "senselab._api.declarations",
    "senselab.signal.resample",
    "senselab.audio.tasks.speech_to_text.huggingface",
    "senselab.audio.tasks.speech_to_text.crisperwhisper_backend",
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
