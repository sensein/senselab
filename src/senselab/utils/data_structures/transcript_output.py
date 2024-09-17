"""This module contains the definition of the TranscriptOutput object."""

from dataclasses import dataclass

import pandas as pd


@dataclass
class TranscriptOutput:
    """Represents an output from an AI conversation transcript."""

    temp: float
    model: str
    prompt: str
    transcript: str
    data: pd.DataFrame

    def __str__(self) -> str:
        """Return a formatted string representation of the transcript.

        Returns:
            str: A formatted string representing the transcript.
        """
        output = ""
        for _, row in self.data.iterrows():
            output += f"Student:\t{row['student']}\n\n"
            output += f"Teacher:\t{row['teacher']}\n"
            output += f"AI:\t{row['AI']}\n\n"
        return output
