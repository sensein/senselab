"""Task classes and functions generated from the task declarations; do not edit.

Regenerate with `uv run python scripts/generate_task_api.py`.
"""

# ruff: noqa

import senselab._api.declarations
import senselab.signal.resample
import senselab.metrics.wer
import senselab.metrics.der
import senselab.metrics.eer
import senselab.metrics.span_f1
import senselab.metrics.resample_error
import senselab.audio.tasks.speech_to_text.huggingface
import senselab.audio.tasks.speech_to_text.crisperwhisper_backend
import senselab.audio.tasks.speaker_diarization.pyannote_backend
import senselab.audio.tasks.speaker_embeddings.speechbrain_backend
import senselab.text.tasks.pii_spans.transformers_backend
import senselab.text.tasks.pii_spans.vllm_backend

from collections.abc import Mapping
from os import PathLike
from senselab._api.call import ConfiguredTask, UNSET
from senselab.text.tasks.pii_spans.profile import PiiProfile
from senselab_core.kernel.dataset import Dataset
from senselab_core.kernel.identity import ModelRef
from senselab_core.kernel.results import Intervals, TextSpans, Transcript, Trials
from senselab_core.kernel.signals import Audio, RegularSeries, Text
from senselab_core.runtime.call import Result, TaskCall
from senselab_core.runtime.execution.run import ResultSet
from typing import Any, Literal

__all__ = [
    "Der",
    "Diarize",
    "Eer",
    "PiiSpans",
    "Resample",
    "ResampleError",
    "SpanF1",
    "SpeakerEmbeddings",
    "SpeechToText",
    "Wer",
    "der",
    "diarize",
    "eer",
    "pii_spans",
    "resample",
    "resample_error",
    "span_f1",
    "speaker_embeddings",
    "speech_to_text",
    "wer",
]


class SpeechToText(ConfiguredTask):
    """The configured, immutable task speech_to_text v1 (transform): audio -> transcript."""

    name = "speech_to_text"

    def __init__(
        self,
        model: ModelRef | str | None = None,
        *,
        language: str | None = None,
        timestamps: Literal["word", "segment", "none"] = "word",
        backend_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure the task; params are validated now."""
        super().__init__(
            model,
            {
                "language": language,
                "timestamps": timestamps,
            },
            backend_params,
        )

    def with_params(
        self,
        *,
        language: str | None = UNSET,
        timestamps: Literal["word", "segment", "none"] = UNSET,
    ) -> "SpeechToText":
        """Return a new task with these params changed; this one is unchanged."""
        return self._replace(
            {
                "language": language,
                "timestamps": timestamps,
            },
        )

    def call(
        self,
        audio: Audio | str | PathLike[str],
    ) -> TaskCall:
        """Return the validated call of this task on the inputs."""
        return self._call(
            {
                "audio": audio,
            },
        )

    def __call__(
        self,
        audio: Audio | str | PathLike[str],
    ) -> Result:
        """Run this task on the inputs with `InProcess` and return its one result."""
        return self._run(
            {
                "audio": audio,
            },
        )

    def map(
        self,
        dataset: Dataset,
        *,
        bind: Mapping[str, str] | None = None,
    ) -> ResultSet:
        """Run this task on every record of the dataset and return its `ResultSet`.

        A port reads the record input of its name, else `bind[port]`; the one port named for none reads the
        records' single input.
        """
        return self._map(dataset, bind)


def speech_to_text(
    audio: Audio | str | PathLike[str],
    *,
    model: ModelRef | str | None = None,
    language: str | None = None,
    timestamps: Literal["word", "segment", "none"] = "word",
    backend_params: Mapping[str, Any] | None = None,
) -> Result:
    """Run `SpeechToText` on the inputs: exactly `SpeechToText(model, **params)(...)`."""
    return SpeechToText(
        model,
        language=language,
        timestamps=timestamps,
        backend_params=backend_params,
    )(audio)


class Diarize(ConfiguredTask):
    """The configured, immutable task diarize v1 (transform): audio -> turns."""

    name = "diarize"

    def __init__(
        self,
        model: ModelRef | str | None = None,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
        backend_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure the task; params are validated now."""
        super().__init__(
            model,
            {
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            },
            backend_params,
        )

    def with_params(
        self,
        *,
        num_speakers: int | None = UNSET,
        min_speakers: int | None = UNSET,
        max_speakers: int | None = UNSET,
    ) -> "Diarize":
        """Return a new task with these params changed; this one is unchanged."""
        return self._replace(
            {
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            },
        )

    def call(
        self,
        audio: Audio | str | PathLike[str],
    ) -> TaskCall:
        """Return the validated call of this task on the inputs."""
        return self._call(
            {
                "audio": audio,
            },
        )

    def __call__(
        self,
        audio: Audio | str | PathLike[str],
    ) -> Result:
        """Run this task on the inputs with `InProcess` and return its one result."""
        return self._run(
            {
                "audio": audio,
            },
        )

    def map(
        self,
        dataset: Dataset,
        *,
        bind: Mapping[str, str] | None = None,
    ) -> ResultSet:
        """Run this task on every record of the dataset and return its `ResultSet`.

        A port reads the record input of its name, else `bind[port]`; the one port named for none reads the
        records' single input.
        """
        return self._map(dataset, bind)


def diarize(
    audio: Audio | str | PathLike[str],
    *,
    model: ModelRef | str | None = None,
    num_speakers: int | None = None,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    backend_params: Mapping[str, Any] | None = None,
) -> Result:
    """Run `Diarize` on the inputs: exactly `Diarize(model, **params)(...)`."""
    return Diarize(
        model,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
        backend_params=backend_params,
    )(audio)


class SpeakerEmbeddings(ConfiguredTask):
    """The configured, immutable task speaker_embeddings v1 (transform): audio -> embedding."""

    name = "speaker_embeddings"

    def __init__(
        self,
        model: ModelRef | str | None = None,
        *,
        backend_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure the task; params are validated now."""
        super().__init__(
            model,
            {},
            backend_params,
        )

    def with_params(
        self,
    ) -> "SpeakerEmbeddings":
        """Return a new task with these params changed; this one is unchanged."""
        return self._replace(
            {},
        )

    def call(
        self,
        audio: Audio | str | PathLike[str],
    ) -> TaskCall:
        """Return the validated call of this task on the inputs."""
        return self._call(
            {
                "audio": audio,
            },
        )

    def __call__(
        self,
        audio: Audio | str | PathLike[str],
    ) -> Result:
        """Run this task on the inputs with `InProcess` and return its one result."""
        return self._run(
            {
                "audio": audio,
            },
        )

    def map(
        self,
        dataset: Dataset,
        *,
        bind: Mapping[str, str] | None = None,
    ) -> ResultSet:
        """Run this task on every record of the dataset and return its `ResultSet`.

        A port reads the record input of its name, else `bind[port]`; the one port named for none reads the
        records' single input.
        """
        return self._map(dataset, bind)


def speaker_embeddings(
    audio: Audio | str | PathLike[str],
    *,
    model: ModelRef | str | None = None,
    backend_params: Mapping[str, Any] | None = None,
) -> Result:
    """Run `SpeakerEmbeddings` on the inputs: exactly `SpeakerEmbeddings(model, **params)(...)`."""
    return SpeakerEmbeddings(
        model,
        backend_params=backend_params,
    )(audio)


class PiiSpans(ConfiguredTask):
    """The configured, immutable task pii_spans v1 (transform): text -> spans."""

    name = "pii_spans"

    def __init__(
        self,
        model: ModelRef | str | None = None,
        *,
        profile: PiiProfile = UNSET,
        family: str | None = None,
        backend_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure the task; params are validated now."""
        super().__init__(
            model,
            {
                "profile": profile,
                "family": family,
            },
            backend_params,
        )

    def with_params(
        self,
        *,
        profile: PiiProfile = UNSET,
        family: str | None = UNSET,
    ) -> "PiiSpans":
        """Return a new task with these params changed; this one is unchanged."""
        return self._replace(
            {
                "profile": profile,
                "family": family,
            },
        )

    def call(
        self,
        text: Text | str | PathLike[str],
    ) -> TaskCall:
        """Return the validated call of this task on the inputs."""
        return self._call(
            {
                "text": text,
            },
        )

    def __call__(
        self,
        text: Text | str | PathLike[str],
    ) -> Result:
        """Run this task on the inputs with `InProcess` and return its one result."""
        return self._run(
            {
                "text": text,
            },
        )

    def map(
        self,
        dataset: Dataset,
        *,
        bind: Mapping[str, str] | None = None,
    ) -> ResultSet:
        """Run this task on every record of the dataset and return its `ResultSet`.

        A port reads the record input of its name, else `bind[port]`; the one port named for none reads the
        records' single input.
        """
        return self._map(dataset, bind)


def pii_spans(
    text: Text | str | PathLike[str],
    *,
    model: ModelRef | str | None = None,
    profile: PiiProfile = UNSET,
    family: str | None = None,
    backend_params: Mapping[str, Any] | None = None,
) -> Result:
    """Run `PiiSpans` on the inputs: exactly `PiiSpans(model, **params)(...)`."""
    return PiiSpans(
        model,
        profile=profile,
        family=family,
        backend_params=backend_params,
    )(text)


class Resample(ConfiguredTask):
    """The configured, immutable task signal.resample v1 (transform): series -> series."""

    name = "signal.resample"

    def __init__(
        self,
        model: ModelRef | str | None = None,
        *,
        target_rate: int,
        backend_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure the task; params are validated now."""
        super().__init__(
            model,
            {
                "target_rate": target_rate,
            },
            backend_params,
        )

    def with_params(
        self,
        *,
        target_rate: int = UNSET,
    ) -> "Resample":
        """Return a new task with these params changed; this one is unchanged."""
        return self._replace(
            {
                "target_rate": target_rate,
            },
        )

    def call(
        self,
        series: RegularSeries | str | PathLike[str],
    ) -> TaskCall:
        """Return the validated call of this task on the inputs."""
        return self._call(
            {
                "series": series,
            },
        )

    def __call__(
        self,
        series: RegularSeries | str | PathLike[str],
    ) -> Result:
        """Run this task on the inputs with `InProcess` and return its one result."""
        return self._run(
            {
                "series": series,
            },
        )

    def map(
        self,
        dataset: Dataset,
        *,
        bind: Mapping[str, str] | None = None,
    ) -> ResultSet:
        """Run this task on every record of the dataset and return its `ResultSet`.

        A port reads the record input of its name, else `bind[port]`; the one port named for none reads the
        records' single input.
        """
        return self._map(dataset, bind)


def resample(
    series: RegularSeries | str | PathLike[str],
    *,
    model: ModelRef | str | None = None,
    target_rate: int,
    backend_params: Mapping[str, Any] | None = None,
) -> Result:
    """Run `Resample` on the inputs: exactly `Resample(model, **params)(...)`."""
    return Resample(
        model,
        target_rate=target_rate,
        backend_params=backend_params,
    )(series)


class Wer(ConfiguredTask):
    """The configured, immutable task wer v1 (transform): reference, hypothesis -> metric."""

    name = "wer"

    def __init__(
        self,
        model: ModelRef | str | None = None,
        *,
        normalise: Literal["open_asr", "none"] = "open_asr",
        backend_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure the task; params are validated now."""
        super().__init__(
            model,
            {
                "normalise": normalise,
            },
            backend_params,
        )

    def with_params(
        self,
        *,
        normalise: Literal["open_asr", "none"] = UNSET,
    ) -> "Wer":
        """Return a new task with these params changed; this one is unchanged."""
        return self._replace(
            {
                "normalise": normalise,
            },
        )

    def call(
        self,
        reference: Transcript | str | PathLike[str],
        hypothesis: Transcript | str | PathLike[str],
    ) -> TaskCall:
        """Return the validated call of this task on the inputs."""
        return self._call(
            {
                "reference": reference,
                "hypothesis": hypothesis,
            },
        )

    def __call__(
        self,
        reference: Transcript | str | PathLike[str],
        hypothesis: Transcript | str | PathLike[str],
    ) -> Result:
        """Run this task on the inputs with `InProcess` and return its one result."""
        return self._run(
            {
                "reference": reference,
                "hypothesis": hypothesis,
            },
        )

    def map(
        self,
        dataset: Dataset,
        *,
        bind: Mapping[str, str] | None = None,
    ) -> ResultSet:
        """Run this task on every record of the dataset and return its `ResultSet`.

        A port reads the record input of its name, else `bind[port]`; the one port named for none reads the
        records' single input.
        """
        return self._map(dataset, bind)


def wer(
    reference: Transcript | str | PathLike[str],
    hypothesis: Transcript | str | PathLike[str],
    *,
    model: ModelRef | str | None = None,
    normalise: Literal["open_asr", "none"] = "open_asr",
    backend_params: Mapping[str, Any] | None = None,
) -> Result:
    """Run `Wer` on the inputs: exactly `Wer(model, **params)(...)`."""
    return Wer(
        model,
        normalise=normalise,
        backend_params=backend_params,
    )(reference, hypothesis)


class Der(ConfiguredTask):
    """The configured, immutable task der v1 (transform): reference, hypothesis -> metric."""

    name = "der"

    def __init__(
        self,
        model: ModelRef | str | None = None,
        *,
        collar: float = 0.0,
        skip_overlap: bool = False,
        backend_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure the task; params are validated now."""
        super().__init__(
            model,
            {
                "collar": collar,
                "skip_overlap": skip_overlap,
            },
            backend_params,
        )

    def with_params(
        self,
        *,
        collar: float = UNSET,
        skip_overlap: bool = UNSET,
    ) -> "Der":
        """Return a new task with these params changed; this one is unchanged."""
        return self._replace(
            {
                "collar": collar,
                "skip_overlap": skip_overlap,
            },
        )

    def call(
        self,
        reference: Intervals | str | PathLike[str],
        hypothesis: Intervals | str | PathLike[str],
    ) -> TaskCall:
        """Return the validated call of this task on the inputs."""
        return self._call(
            {
                "reference": reference,
                "hypothesis": hypothesis,
            },
        )

    def __call__(
        self,
        reference: Intervals | str | PathLike[str],
        hypothesis: Intervals | str | PathLike[str],
    ) -> Result:
        """Run this task on the inputs with `InProcess` and return its one result."""
        return self._run(
            {
                "reference": reference,
                "hypothesis": hypothesis,
            },
        )

    def map(
        self,
        dataset: Dataset,
        *,
        bind: Mapping[str, str] | None = None,
    ) -> ResultSet:
        """Run this task on every record of the dataset and return its `ResultSet`.

        A port reads the record input of its name, else `bind[port]`; the one port named for none reads the
        records' single input.
        """
        return self._map(dataset, bind)


def der(
    reference: Intervals | str | PathLike[str],
    hypothesis: Intervals | str | PathLike[str],
    *,
    model: ModelRef | str | None = None,
    collar: float = 0.0,
    skip_overlap: bool = False,
    backend_params: Mapping[str, Any] | None = None,
) -> Result:
    """Run `Der` on the inputs: exactly `Der(model, **params)(...)`."""
    return Der(
        model,
        collar=collar,
        skip_overlap=skip_overlap,
        backend_params=backend_params,
    )(reference, hypothesis)


class Eer(ConfiguredTask):
    """The configured, immutable task eer v1 (transform): trials -> metric."""

    name = "eer"

    def __init__(
        self,
        model: ModelRef | str | None = None,
        *,
        backend_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure the task; params are validated now."""
        super().__init__(
            model,
            {},
            backend_params,
        )

    def with_params(
        self,
    ) -> "Eer":
        """Return a new task with these params changed; this one is unchanged."""
        return self._replace(
            {},
        )

    def call(
        self,
        trials: Trials | str | PathLike[str],
    ) -> TaskCall:
        """Return the validated call of this task on the inputs."""
        return self._call(
            {
                "trials": trials,
            },
        )

    def __call__(
        self,
        trials: Trials | str | PathLike[str],
    ) -> Result:
        """Run this task on the inputs with `InProcess` and return its one result."""
        return self._run(
            {
                "trials": trials,
            },
        )

    def map(
        self,
        dataset: Dataset,
        *,
        bind: Mapping[str, str] | None = None,
    ) -> ResultSet:
        """Run this task on every record of the dataset and return its `ResultSet`.

        A port reads the record input of its name, else `bind[port]`; the one port named for none reads the
        records' single input.
        """
        return self._map(dataset, bind)


def eer(
    trials: Trials | str | PathLike[str],
    *,
    model: ModelRef | str | None = None,
    backend_params: Mapping[str, Any] | None = None,
) -> Result:
    """Run `Eer` on the inputs: exactly `Eer(model, **params)(...)`."""
    return Eer(
        model,
        backend_params=backend_params,
    )(trials)


class SpanF1(ConfiguredTask):
    """The configured, immutable task span_f1 v1 (transform): reference, hypothesis -> metric."""

    name = "span_f1"

    def __init__(
        self,
        model: ModelRef | str | None = None,
        *,
        backend_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure the task; params are validated now."""
        super().__init__(
            model,
            {},
            backend_params,
        )

    def with_params(
        self,
    ) -> "SpanF1":
        """Return a new task with these params changed; this one is unchanged."""
        return self._replace(
            {},
        )

    def call(
        self,
        reference: TextSpans | str | PathLike[str],
        hypothesis: TextSpans | str | PathLike[str],
    ) -> TaskCall:
        """Return the validated call of this task on the inputs."""
        return self._call(
            {
                "reference": reference,
                "hypothesis": hypothesis,
            },
        )

    def __call__(
        self,
        reference: TextSpans | str | PathLike[str],
        hypothesis: TextSpans | str | PathLike[str],
    ) -> Result:
        """Run this task on the inputs with `InProcess` and return its one result."""
        return self._run(
            {
                "reference": reference,
                "hypothesis": hypothesis,
            },
        )

    def map(
        self,
        dataset: Dataset,
        *,
        bind: Mapping[str, str] | None = None,
    ) -> ResultSet:
        """Run this task on every record of the dataset and return its `ResultSet`.

        A port reads the record input of its name, else `bind[port]`; the one port named for none reads the
        records' single input.
        """
        return self._map(dataset, bind)


def span_f1(
    reference: TextSpans | str | PathLike[str],
    hypothesis: TextSpans | str | PathLike[str],
    *,
    model: ModelRef | str | None = None,
    backend_params: Mapping[str, Any] | None = None,
) -> Result:
    """Run `SpanF1` on the inputs: exactly `SpanF1(model, **params)(...)`."""
    return SpanF1(
        model,
        backend_params=backend_params,
    )(reference, hypothesis)


class ResampleError(ConfiguredTask):
    """The configured, immutable task resample_error v1 (transform): reference, hypothesis -> metric."""

    name = "resample_error"

    def __init__(
        self,
        model: ModelRef | str | None = None,
        *,
        backend_params: Mapping[str, Any] | None = None,
    ) -> None:
        """Configure the task; params are validated now."""
        super().__init__(
            model,
            {},
            backend_params,
        )

    def with_params(
        self,
    ) -> "ResampleError":
        """Return a new task with these params changed; this one is unchanged."""
        return self._replace(
            {},
        )

    def call(
        self,
        reference: RegularSeries | str | PathLike[str],
        hypothesis: RegularSeries | str | PathLike[str],
    ) -> TaskCall:
        """Return the validated call of this task on the inputs."""
        return self._call(
            {
                "reference": reference,
                "hypothesis": hypothesis,
            },
        )

    def __call__(
        self,
        reference: RegularSeries | str | PathLike[str],
        hypothesis: RegularSeries | str | PathLike[str],
    ) -> Result:
        """Run this task on the inputs with `InProcess` and return its one result."""
        return self._run(
            {
                "reference": reference,
                "hypothesis": hypothesis,
            },
        )

    def map(
        self,
        dataset: Dataset,
        *,
        bind: Mapping[str, str] | None = None,
    ) -> ResultSet:
        """Run this task on every record of the dataset and return its `ResultSet`.

        A port reads the record input of its name, else `bind[port]`; the one port named for none reads the
        records' single input.
        """
        return self._map(dataset, bind)


def resample_error(
    reference: RegularSeries | str | PathLike[str],
    hypothesis: RegularSeries | str | PathLike[str],
    *,
    model: ModelRef | str | None = None,
    backend_params: Mapping[str, Any] | None = None,
) -> Result:
    """Run `ResampleError` on the inputs: exactly `ResampleError(model, **params)(...)`."""
    return ResampleError(
        model,
        backend_params=backend_params,
    )(reference, hypothesis)
