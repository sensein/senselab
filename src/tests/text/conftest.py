"""This script includes some fixtures for pytest unit testing."""
from typing import List

import pytest


@pytest.fixture
def sample_texts() -> List[str]:
    """Fixture for some pieces of text to test our text utilities."""
    return ["Hello, world!", "Testing embeddings extraction."]

