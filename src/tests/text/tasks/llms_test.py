"""This module is for testing the conversion of JSON conversation segments to message objects."""

import os
from typing import List

import pytest

from senselab.text.tasks.llms.data_ingest import MessagesManager

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
    def expected_messages() -> List[dict]:
        """Fixture for the expected list of message objects."""
        return [
            {"role": "user", "content": "uh"},
            {"role": "assistant", "content": "hello world namaste kemosabe"},
            {"role": "user", "content": "hi there"},
        ]

    def test_convert_json_to_messages(sample_json_obj: dict, expected_messages: List[dict]) -> None:
        """Test the conversion of JSON conversation segments to message objects."""
        result = MessagesManager.convert_json_to_messages(sample_json_obj)
        assert result == expected_messages
