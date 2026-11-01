"""Task classes and functions generated from the task declarations; do not edit.

Regenerate with `uv run python scripts/generate_task_api.py`.
"""

# ruff: noqa

import senselab._api.declarations
import senselab.signal.resample
import senselab.audio.tasks.speech_to_text.huggingface

from collections.abc import Mapping
from os import PathLike
from senselab._api.call import ConfiguredTask, UNSET
from senselab_core.kernel.identity import ModelRef
from senselab_core.kernel.signals import Audio, RegularSeries
from senselab_core.runtime.call import Result, TaskCall
from typing import Any, Literal

__all__ = [
    "Resample",
    "SpeechToText",
    "resample",
    "speech_to_text",
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
        """Run this task on the inputs through `run_call` and return its one result."""
        return self._run(
            {
                "audio": audio,
            },
        )


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
        """Run this task on the inputs through `run_call` and return its one result."""
        return self._run(
            {
                "series": series,
            },
        )


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
