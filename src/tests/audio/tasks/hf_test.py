"""Tests for HF models and functions."""

from unittest.mock import patch

import pytest

from senselab.utils.hf import HFModel, check_hf_repo_exists


def test_check_hf_repo_exists_true() -> None:
    """Test HF repo exists."""
    with patch("huggingface_hub.HfApi.list_repo_commits") as mock_list_repo_commits:
        mock_list_repo_commits.return_value = True
        assert check_hf_repo_exists("valid_repo") is True


def test_check_hf_repo_exists_false() -> None:
    """Test HF repo does not exist."""
    with patch("huggingface_hub.HfApi.list_repo_commits", side_effect=Exception):
        assert check_hf_repo_exists("invalid_repo") is False


def test_hfmodel_valid_hf_repo_check() -> None:
    """Test valid HFModel repo check."""
    with patch("senselab.utils.hf.check_hf_repo_exists", return_value=True):
        model = HFModel(path_or_uri="valid_repo")
        assert model.revision == "main"


def test_hfmodel_invalid_hf_repo_check() -> None:
    """Test invalid HFModel repo check."""
    with patch("senselab.utils.hf.check_hf_repo_exists", return_value=False):
        with pytest.raises(ValueError):
            HFModel(path_or_uri="invalid/repo")
