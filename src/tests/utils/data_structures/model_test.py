"""Tests for HF models and functions."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from huggingface_hub import HfApi

from senselab.utils.data_structures import HFModel, check_hf_repo_exists
from senselab.utils.data_structures.model import get_huggingface_token
from senselab.utils.dependencies import torchaudio_available

TORCHAUDIO_AVAILABLE = torchaudio_available()


@pytest.mark.skipif(TORCHAUDIO_AVAILABLE, reason="torchaudio is not installed.")
def test_check_torchaudio_model_init() -> None:
    """Test torchaudio model initialization."""
    with pytest.raises(ModuleNotFoundError):
        from senselab.utils.data_structures.model import TorchAudioModel

        TorchAudioModel(path_or_uri="torchaudio_model", revision="main")


def test_check_hf_repo_exists_true() -> None:
    """Test HF repo exists."""
    with patch("senselab.utils.dependencies.ensure_hf_model", return_value="abc123"):
        assert check_hf_repo_exists("valid_repo") is True


def test_check_hf_repo_exists_false() -> None:
    """Test HF repo does not exist."""
    with patch("senselab.utils.dependencies.ensure_hf_model", side_effect=Exception("not found")):
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


def test_get_huggingface_token_from_env_file_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading a Hugging Face token from an explicit `.env` file path."""
    for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(env_var, raising=False)

    env_file = tmp_path / "hf.env"
    env_file.write_text("HF_TOKEN=hf_from_file\n")

    assert get_huggingface_token(env_file) == "hf_from_file"


def test_get_huggingface_token_from_local_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test loading a Hugging Face token from a local `.env` file in the cwd."""
    for env_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        monkeypatch.delenv(env_var, raising=False)

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HUGGING_FACE_HUB_TOKEN=hf_from_local_dotenv\n")

    assert get_huggingface_token() == "hf_from_local_dotenv"


def test_get_huggingface_token_prefers_environment_over_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test environment variables take precedence over `.env` values."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("HF_TOKEN=hf_from_file\n")
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")

    assert get_huggingface_token() == "hf_from_env"


@patch("senselab.utils.dependencies.ensure_hf_model", return_value="abc123")
def test_hfmodel_caches_hf_repo_check(mock_ensure: MagicMock) -> None:
    """Test that we successfully cache HF repo checks and only make the check once."""
    _ = HFModel(path_or_uri="unique_repo_name_1")
    assert mock_ensure.call_count == 1

    _ = HFModel(path_or_uri="unique_repo_name_1")
    # Second instantiation should use in-memory _hf_cache, not call ensure again
    assert mock_ensure.call_count == 1

    _ = HFModel(path_or_uri="unique_repo_name_2")
    assert mock_ensure.call_count == 2
