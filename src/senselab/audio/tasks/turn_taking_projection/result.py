"""The per-frame output of a Voice Activity Projection run."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator

FRAME_COLUMNS = (
    "time",
    "p_now_a",
    "p_now_b",
    "p_future_a",
    "p_future_b",
    "vad_a",
    "vad_b",
)
"""Columns every :class:`VAPResult` frame table carries. ``p_bc_a`` / ``p_bc_b`` are added when a
backchannel checkpoint was run."""


class VAPResult(BaseModel):
    """Voice Activity Projection frames for one dyad.

    Attributes:
        frame_rate: Frames per second.
        speakers: ``(speaker_a, speaker_b)``: the labels of channel ``a`` and channel ``b``.
        frames: One row per frame. ``time`` is the end of the audio the frame has seen, in seconds
            (frame ``i`` at ``(i + 1) / frame_rate``): the model is causal, so each row is what it
            projected at that instant. ``p_now_*`` / ``p_future_*``: probability that the speaker
            holds the floor 0-0.6 s / 0.6-2 s ahead, normalised so ``a + b = 1``. ``vad_*``: voice
            activity of each channel at the frame. ``p_bc_*``: probability that the speaker
            backchannels now while the other speaks.
        models: Provenance of each checkpoint run: repository, commit, file, licence.
        timing: Seconds spent, and the duration processed.
        input: How the two channels were obtained (e.g. a separation method and its parameters).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_rate: float
    speakers: Tuple[str, str]
    frames: pd.DataFrame
    models: Dict[str, Dict[str, str]] = {}
    timing: Dict[str, float] = {}
    input: Dict[str, Any] = {}

    @field_validator("frames")
    @classmethod
    def _has_columns(cls, frames: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in FRAME_COLUMNS if c not in frames.columns]
        if missing:
            raise ValueError(f"frames is missing columns {missing}")
        return frames

    def channel(self, speaker: str) -> str:
        """``"a"`` or ``"b"`` for a speaker label.

        Raises:
            KeyError: If the label is neither speaker.
        """
        if speaker == self.speakers[0]:
            return "a"
        if speaker == self.speakers[1]:
            return "b"
        raise KeyError(f"speaker {speaker!r} is not one of {self.speakers}")

    def series(self, column: str, speaker: str) -> np.ndarray:
        """One speaker's column (``"p_now"``, ``"p_future"``, ``"vad"``, ``"p_bc"``) as an array."""
        return self.frames[f"{column}_{self.channel(speaker)}"].to_numpy()

    @property
    def times(self) -> np.ndarray:
        """Frame times in seconds."""
        return self.frames["time"].to_numpy()

    @property
    def has_backchannel(self) -> bool:
        """Whether ``p_bc_a`` / ``p_bc_b`` are present."""
        return "p_bc_a" in self.frames.columns and "p_bc_b" in self.frames.columns
