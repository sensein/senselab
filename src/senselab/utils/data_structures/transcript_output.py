"""This module contains the definition of the TranscriptOutput object."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class TranscriptOutput:
    """Represents an output from an AI conversation transcript."""

    temp: float
    model: str
    prompt: str
    transcript: str
    data: list[dict]  # list[dict[speaker, text, latency, in_tokens, out_tokens, metrics]]

    def __str__(self: "TranscriptOutput") -> str:
        """Return a formatted string representation of the transcript.

        Returns:
            str: A formatted string representing the transcript.
        """
        output = ""
        for item in self.data:
            output += f"{item['speaker']}: {item['text']}\n\n"
        return output

    def to_json(self: "TranscriptOutput") -> str:
        """Return a JSON representation of the transcript.

        Returns:
            str: A JSON representation of the transcript.
        """
        return json.dumps(self.__dict__)

    def save_to_json(self: "TranscriptOutput", path: Union[str, Path]) -> None:
        """Save the JSON representation of the transcript to a file.

        Args:
            path (str | Path): The path to save the JSON file.
        """
        with open(path, "w") as f:
            json.dump(self.__dict__, f)
