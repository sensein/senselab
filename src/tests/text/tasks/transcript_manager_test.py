"""Test cases for the transcript manager module."""

import json
import os
from pathlib import Path
from typing import List

import pytest

from senselab.text.tasks.llms.transcript_manager import Transcript
from senselab.utils.data_structures.script_line import ScriptLine

if os.getenv("GITHUB_ACTIONS") != "true":

    @pytest.fixture
    def sample_json_obj() -> dict:
        """Fixture for a sample JSON object representing conversation segments."""
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "words": [
                        {"word": "uh", "start": 0.0, "end": 0.5, "score": 1.0, "speaker": "kid"},
                        {"word": "hello", "start": 0.6, "end": 1.0, "score": 1.0, "speaker": "teacher"},
                    ],
                    "speaker": "kid",
                },
                {
                    "start": 1.0,
                    "end": 2.0,
                    "words": [
                        {"word": "world", "start": 1.0, "end": 1.5, "score": 1.0, "speaker": "teacher"},
                        {"word": "namaste", "start": 1.6, "end": 2.0, "score": 1.0, "speaker": "teacher"},
                    ],
                    "speaker": "teacher",
                },
                {
                    "start": 2.0,
                    "end": 3.0,
                    "words": [
                        {"word": "kemosabe", "start": 2.0, "end": 2.5, "score": 1.0, "speaker": "teacher"},
                        {"word": "hi", "start": 2.6, "end": 2.8, "score": 1.0, "speaker": "kid"},
                        {"word": "there", "start": 2.9, "end": 3.0, "score": 1.0, "speaker": "kid"},
                    ],
                    "speaker": "kid",
                },
            ]
        }

    @pytest.fixture
    def sample_transcript(tmp_path: Path, sample_json_obj: dict) -> Path:
        """Fixture to create a sample transcript file."""
        transcript_file = tmp_path / "transcript.json"
        with transcript_file.open("w") as f:
            json.dump(sample_json_obj, f)
        return transcript_file

    @pytest.fixture
    def expected_messages() -> List[ScriptLine]:
        """Fixture for the expected list of message objects."""
        return [
            ScriptLine(speaker="user", text="uh"),
            ScriptLine(speaker="assistant", text="hello world namaste kemosabe"),
            ScriptLine(speaker="user", text="hi there"),
        ]

    def test_convert_json_to_messages(sample_json_obj: dict, expected_messages: List[ScriptLine]) -> None:
        """Test the conversion of JSON conversation segments to message objects."""
        result = Transcript.convert_json_to_scriptlines(sample_json_obj)
        assert result == expected_messages

    def test_missing_word_or_speaker_field() -> None:
        """Test behavior when word or speaker field is missing from the segment."""
        invalid_json = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "words": [{"word": "hello"}],  # Missing speaker
                    "speaker": "teacher",
                }
            ]
        }
        with pytest.raises(ValueError, match="Invalid word structure"):
            Transcript.convert_json_to_scriptlines(invalid_json)

    def test_get_num_tokens(sample_transcript: Path) -> None:
        """Test the ability of the program to return the correct number of expected tokens."""
        transcript = Transcript(sample_transcript)  # Initialize the transcript
        result = transcript.get_num_tokens()  # Get the token count
        assert result == 10

    def test_response_opportunities_extraction(sample_transcript: Path) -> None:
        """Test the extraction of response opportunities."""
        transcript = Transcript(sample_transcript)
        opportunities = transcript.extract_response_opportunities()

        assert len(opportunities) == 2, "Expected two response opportunities"
        assert opportunities[0][-1].speaker == "user", "Expected last message to be first message from user"
        assert opportunities[1][-1].speaker == "user", "Expected last message to be second message from 'user'"
