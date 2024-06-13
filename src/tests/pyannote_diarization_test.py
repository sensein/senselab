"""Tests for the pyannote_diarize module."""

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from senselab.audio.tasks import _pyannote_diarize_batch, pyannote_diarize


@pytest.fixture
def sample_dataset() -> dict:
    """Fixture that provides a sample dataset for testing.

    Returns:
        dict: A dictionary representing a sample dataset with audio data.
    """
    return {
        "audio": [
            {
                "array": [0.1, 0.2, 0.3, 0.4],  # Example audio data
                "sampling_rate": 16000,
            }
        ]
    }


@pytest.fixture
def hf_token() -> str:
    """Fixture that provides a Hugging Face API token for testing.

    Returns:
        str: A string representing the Hugging Face API token.
    """
    return "your_hf_token"


@pytest.fixture
def cache_path(tmpdir: pytest.TempPathFactory) -> str:
    """Fixture that provides a temporary cache path for testing.

    Args:
        tmpdir (pytest.TempPathFactory): Temporary directory provided by pytest.

    Returns:
        str: A string representing the path to the cache directory.
    """
    return str(tmpdir.mkdir("cache"))


@patch("senselab.audio.tasks.Pipeline.from_pretrained")
@patch("senselab.audio.tasks._from_dict_to_hf_dataset")
@patch("senselab.audio.tasks._from_hf_dataset_to_dict")
def test_pyannote_diarize(
    mock_from_hf_dataset_to_dict: MagicMock,
    mock_from_dict_to_hf_dataset: MagicMock,
    mock_pipeline: MagicMock,
    sample_dataset: dict,
    hf_token: str,
    cache_path: str,
) -> None:
    """Test the `pyannote_diarize` function.

    Args:
        mock_from_hf_dataset_to_dict (MagicMock):
            Mock for `_from_hf_dataset_to_dict`.
        mock_from_dict_to_hf_dataset (MagicMock):
            Mock for `_from_dict_to_hf_dataset`.
        mock_pipeline (MagicMock): Mock for `Pipeline.from_pretrained`.
        sample_dataset (dict): The sample dataset fixture.
        hf_token (str): The Hugging Face API token fixture.
        cache_path (str): The cache path fixture.
    """
    mock_pipeline.return_value = MagicMock()
    mock_pipeline().return_value = MagicMock()
    mock_pipeline().return_value.itertracks.return_value = [
        (("0.0", "5.0"), None, "speaker_1"),
        (("5.0", "10.0"), None, "speaker_2"),
    ]

    mock_from_dict_to_hf_dataset.return_value = Dataset.from_dict(
        sample_dataset
    )
    mock_from_hf_dataset_to_dict.return_value = sample_dataset

    result = pyannote_diarize(
        sample_dataset,
        hf_token=hf_token,
        cache_path=cache_path,
    )

    mock_pipeline.assert_called_once_with(
        "pyannote/speaker-diarization-3.1", use_auth_token=hf_token
    )
    assert "pyannote31_diarizations" in result
    assert isinstance(result["pyannote31_diarizations"], list)


@patch("senselab.audio.tasks.Pipeline.from_pretrained")
def test_pyannote_diarize_batch(
    mock_pipeline: MagicMock,
    sample_dataset: dict,
    hf_token: str,
) -> None:
    """Test the `_pyannote_diarize_batch` function.

    Args:
        mock_pipeline (MagicMock): Mock for `Pipeline.from_pretrained`.
        sample_dataset (dict): The sample dataset fixture.
        hf_token (str): The Hugging Face API token fixture.
    """
    mock_pipeline.return_value = MagicMock()
    mock_pipeline().return_value = MagicMock()
    mock_pipeline().return_value.itertracks.return_value = [
        (("0.0", "5.0"), None, "speaker_1"),
        (("5.0", "10.0"), None, "speaker_2"),
    ]

    batch = Dataset.from_dict(sample_dataset)
    result = _pyannote_diarize_batch(
        batch, hf_token, "pyannote/speaker-diarization", "3.1"
    )

    assert "pyannote31_diarizations" in result
    assert isinstance(result["pyannote31_diarizations"], list)
    assert len(result["pyannote31_diarizations"]) == len(
        sample_dataset["audio"]
    )


if __name__ == "__main__":
    pytest.main()
