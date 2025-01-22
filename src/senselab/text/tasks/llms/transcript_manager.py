"""This module provides a data manager for handling interactions with a LLM."""

import json
from pathlib import Path
from typing import Dict, List

import tiktoken

from senselab.utils.data_structures.script_line import ScriptLine


class Transcript:
    """Manages message data for interactions with a LLM.

    Provides methods to load transcripts, convert JSON data to message objects,
    and generate data from a human conversation to query potential AI responses.

    Attributes:
        scriptlines (List[Scriptline]): A list of Scriptline objects.

    Methods:
        __init__(transcript_path: Path) -> None: Initializes the manager with a transcript file path.
        print_human_readable() -> None: Prints messages in a readable format.
        extract_response_opportunities() -> List[List[Scriptline]]: Extracts sublists ending with user input.
        get_num_tokens()-> int: total number of tokens in transcript
        _load_transcript(json_path: Path) -> Dict: Loads a JSON transcript from a file.
        convert_json_to_scriptlines(json_obj: Dict) -> List[ScriptLine]: Converts transcript format to LLM format.
    """

    def __init__(self: "Transcript", transcript_path: Path) -> None:
        """Initializes the manager with a transcript file path.

        Args:
            transcript_path (Path): The path to the JSON transcript file.
        """
        if not transcript_path.exists():
            raise ValueError("Transcript path not found!")
        json_obj = self._load_transcript(transcript_path)
        self.scriptlines = self.convert_json_to_scriptlines(json_obj)

    def print_human_readable(self: "Transcript") -> None:
        """Prints the stored scriptlines in a human-readable format."""
        for message in self.scriptlines:
            print(f"{message.speaker}:\t\t{message.text}\n")

    def get_num_tokens(self: "Transcript") -> int:
        """Returns the total number of OpenAI tokens in the conversation.

        Returns:
            int: number of tokens
        """
        c = 0
        encoding = tiktoken.encoding_for_model("gpt-4o")
        for message in self.scriptlines:
            if message.text:
                c += len(encoding.encode(message.text))
        return c

    def extract_response_opportunities(self: "Transcript") -> List[List[ScriptLine]]:
        """Extract consecutive sublists from the messages list, ending after every 'user' response.

        This is used to compare AI responses to a human's response
        over the course of a conversation, where the AI has the previous,
        natural conversation before making its own response.

        Returns:
            List[ScriptLine]: A list of sublists, each starting from the
                                         beginning of the messages list and ending with the next
                                         sequential message where the role is "user".
        """
        sublists = []

        for i, message in enumerate(self.scriptlines):
            if message.speaker == "user" and i >= 0:
                sublist = self.scriptlines[0 : i + 1]
                sublists.append(sublist)

        return sublists

    @staticmethod
    def _load_transcript(json_path: Path) -> Dict:
        """Load a JSON transcript from the specified file path.

        This static method reads a JSON file from the provided file path and
        returns the loaded JSON object.

        Args:
            json_path (Path): The file path to the JSON transcript file.

        Returns:
            Dict: The JSON object loaded from the file.
        """
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        return data

    @staticmethod
    def convert_json_to_scriptlines(json_obj: Dict) -> List[ScriptLine]:
        """Converts transcript segments to list of ScriptLine objects.

        The input JSON object should have the following structure:
        {
            "segments": [
                {
                    "start": <float>,
                    "end": <float>,
                    "text": <string>,
                    "words": [
                        {
                            "word": <string>,
                            "start": <float>,
                            "end": <float>,
                            "score": <float>,
                            "speaker": <string> [kid|teacher]
                        },
                        ...
                    ],
                    "speaker": <string> [kid|teacher]
                },
                ...
            ]
        }


        The conversion will map the "teacher" speaker role to "assistant" and the "kid" speaker
        role to "user".

        Args:
            json_obj (Dict): The input JSON object containing conversation segments.

        Returns:
            List[ScriptLine]: See src/senselab/utils/data_structures/script_line.py

        Raises:
            ValueError: If the input JSON structure is invalid or contains an unknown speaker role.
        """
        # Ensure valid JSON structure
        if not (isinstance(json_obj, dict) and isinstance(json_obj.get("segments"), list)):
            raise ValueError("Invalid JSON structure: must be a dictionary with a 'segments' list")

        scriptlines = []
        current_role: str = ""
        current_content: List[str] = []

        for segment in json_obj["segments"]:
            # Validate segment structure
            if not all(key in segment for key in ("words",)):
                raise ValueError(f"Invalid segment structure: {segment}")

            for word_obj in segment["words"]:
                if not all(key in word_obj for key in ("word", "speaker")):
                    raise ValueError(f"Invalid word structure: {word_obj}")

                word = word_obj["word"]
                speaker = word_obj["speaker"]

                if speaker == "teacher":
                    role = "assistant"
                elif speaker == "kid":
                    role = "user"
                else:
                    continue

                if role != current_role:
                    if current_content:
                        scriptlines.append(ScriptLine(text=" ".join(current_content), speaker=current_role))

                    current_role = role
                    current_content = [word]
                else:
                    current_content.append(word)

        if current_content:
            scriptlines.append(ScriptLine(text=" ".join(current_content), speaker=current_role))

        return scriptlines
