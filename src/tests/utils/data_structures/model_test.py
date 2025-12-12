"""Tests for HF models and functions."""

from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import HfApi

from senselab.utils.data_structures import HFModel, check_hf_repo_exists

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    TORCHAUDIO_AVAILABLE = False


@pytest.mark.skipif(TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_check_torchaudio_model_init() -> None:
    """Test torchaudio model initialization."""
    with pytest.raises(ModuleNotFoundError):
        from senselab.utils.data_structures.model import TorchAudioModel

        TorchAudioModel(path_or_uri="torchaudio_model", revision="main")


def test_check_hf_repo_exists_true() -> None:
    """Test HF repo exists."""
    with patch("huggingface_hub.HfApi.model_info") as mock_list_repo_commits:
        mock_list_repo_commits.return_value = True
        assert check_hf_repo_exists("valid_repo") is True


def test_check_hf_repo_exists_false() -> None:
    """Test HF repo does not exist."""
    with patch("huggingface_hub.HfApi.list_repo_commits", side_effect=Exception):
        assert check_hf_repo_exists("invalid_repo") is False


def test_hfmodel_valid_hf_repo_check() -> None:
    """Test valid HFModel repo check."""
    with patch("senselab.utils.data_structures.model.check_hf_repo_exists", return_value=True):
        model: HFModel = HFModel(path_or_uri="valid_repo")
        assert model.revision == "main"


def test_hfmodel_invalid_hf_repo_check() -> None:
    """Test invalid HFModel repo check."""
    with patch("senselab.utils.data_structures.model.check_hf_repo_exists", return_value=False):
        with pytest.raises(ValueError):
            HFModel(path_or_uri="invalid/repo")


@patch("huggingface_hub.HfApi.model_info")
def test_hfmodel_caches_hf_repo_check(mock_hf_api_model_info: MagicMock) -> None:
    """Test that we successfully cache HF repo checks and only make the check once."""
    mock_hf_api_model_info.return_value = True
    _ = HFModel(path_or_uri="unique_repo_name_1")

    mock_hf_api_model_info.assert_called_with(repo_id="unique_repo_name_1", revision="main")
    assert mock_hf_api_model_info.call_count == 1

    _ = HFModel(path_or_uri="unique_repo_name_1")
    assert mock_hf_api_model_info.call_count == 1

    _ = HFModel(path_or_uri="unique_repo_name_2")
    assert mock_hf_api_model_info.call_count == 2
