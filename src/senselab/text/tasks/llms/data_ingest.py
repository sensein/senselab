"""This module provides a data manager for handling interactions with a LLM."""

import json
from pathlib import Path
from typing import Dict, List


class MessagesManager:
    """Manages message data for interactions with a LLM.

    Provides methods to load transcripts, convert JSON data to message objects,
    and generate data from a human conversation to query potential AI responses.

    Attributes:
        messages (List[Dict[str, str]]): A list of message objects for the OpenAI API.

    Methods:
        __init__(transcript_path: Path) -> None: Initializes the manager with a transcript file path.
        print_human_readable(messages: List[Dict[str, str]]) -> None: Prints messages in a readable format.
        extract_response_opportunities() -> List[List[Dict[str, str]]]: Extracts sublists ending with user input.
        _load_transcript(json_path: Path) -> Dict: Loads a JSON transcript from a file.
        convert_json_to_messages(json_obj: Dict) -> List[Dict[str, str]]: Converts transcript format to LLM format.
    """

    def __init__(self, transcript_path: Path) -> None:
        """Initializes the manager with a transcript file path.

        Args:
            transcript_path (Path): The path to the JSON transcript file.
        """
        json_obj = self._load_transcript(transcript_path)
        self.messages = self.convert_json_to_messages(json_obj)

    @staticmethod
    def print_human_readable(messages: List[Dict[str, str]]) -> None:
        """Print a list of messages in a human-readable format.

        Args:
            messages (List[Dict[str, str]]): List of messages where each message is a dictionary
                                            with 'role' and 'content' keys.
        """
        for message in messages:
            print(f'{message["role"]}:\t\t{message["content"]}\n')

    def extract_response_opportunities(self) -> List[List[Dict[str, str]]]:
        """Extract consecutive sublists from the messages list, ending after every 'user' response.

        This is used to compare AI responses to a human's response
        over the course of a conversation, where the AI has the previous,
        natural conversation before making its own response.

        Returns:
            List[List[Dict[str, str]]]: A list of consecutive sublists, each starting from the
                                         beginning of the messages list and ending with a
                                         message where the role is "user".
        """
        sublists = []

        for i, message in enumerate(self.messages):
            if message["role"] == "user" and i > 0:
                sublist = self.messages[0 : i + 1]
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
            return json.load(file)

    @staticmethod
    def convert_json_to_messages(json_obj: Dict) -> List[Dict[str, str]]:
        """Converts transcript segments to list of message objects, excluding system messages.

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

        The output will be a list of message objects,
        suitable for OpenAI API, with the following structure:
        [
            {
                "role": "user",
                "content": "<user-input-string>"
            },
            {
                "role": "assistant",
                "content": "<assistant-response-string>"
            },
            ...
        ]

        The conversion will map the "teacher" speaker role to "assistant" and the "kid" speaker
        role to "user".

        Args:
            json_obj (Dict): The input JSON object containing conversation segments.

        Returns:
            List[Dict[str, str]]: A list of message objects in the format required by the OpenAI API.

        Raises:
            ValueError: If the input JSON structure is invalid or contains an unknown speaker role.
        """
        # Ensure valid JSON structure
        if not (isinstance(json_obj, dict) and isinstance(json_obj.get("segments"), list)):
            raise ValueError("Invalid JSON structure: must be a dictionary with a 'segments' list")

        messages = []
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
                    raise ValueError(f"Unknown speaker role: {speaker}")

                if role != current_role:
                    if current_content:
                        messages.append({"role": current_role, "content": " ".join(current_content)})
                    current_role = role
                    current_content = [word]
                else:
                    current_content.append(word)

        if current_content:
            messages.append({"role": current_role, "content": " ".join(current_content)})

        return messages
