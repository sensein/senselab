"""This module implements some utilities for the model class."""

try:
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ModuleNotFoundError:
    TORCHAUDIO_AVAILABLE = False

try:
    from TTS.api import TTS

    TTS_AVAILABLE = True
except ModuleNotFoundError:
    TTS_AVAILABLE = False
    TTS = None  # This is to avoid errors during pdoc documentation generation

import os
from functools import lru_cache
from pathlib import Path
from typing import ClassVar, Dict, Generic, Optional, Tuple, TypeVar, Union

import requests
import torch
from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
from huggingface_hub.hf_api import ModelInfo
from pydantic import BaseModel, Field, PrivateAttr, ValidationInfo, field_validator
from typing_extensions import Annotated

# Define the TypeVar for provider types
PROVIDER_T = TypeVar("PROVIDER_T")


class SenselabModel(BaseModel, Generic[PROVIDER_T]):
    """Base configuration for SenselabModel class."""

    path_or_uri: Union[str, Path]
    revision: Optional[str] = None

    @field_validator("path_or_uri", mode="before")
    def validate_path_or_uri(cls, value: Union[str, Path]) -> Union[str, Path]:
        """Validate the path_or_uri.

        This check is only for files and not for remote resources.
        It does check if the path_or_uri is not empty and if it is an
        existing file.
        """
        if not value:
            raise ValueError("path_or_uri cannot be empty")

        if isinstance(value, Path) and not os.path.isfile(value):
            raise ValueError("path_or_uri is not an existing file")

        # If the value is a string and looks like an existing file path,
        # convert it to a Path object
        if isinstance(value, str) and os.path.isfile(value):
            value = Path(value)
            if not is_torch_model(value):
                raise ValueError("path_or_uri does not point to a valid torch model")

        return value


class HFModel(SenselabModel[PROVIDER_T]):
    """HuggingFace model.

    Note: For some HuggingFace models, HF_TOKEN may be required for access.
    """

    revision: Annotated[str, Field(validate_default=True)] = "main"
    info: Optional[ModelInfo] = None
    _hf_cache: ClassVar[Dict[Tuple[str, str], bool]] = {}

    @field_validator("revision")
    def validate_hf_model_id(cls, value: str, info: ValidationInfo) -> Union[str, Path]:
        """Validate the path_or_uri.

        This check is only for remote resources and not for files.
        It checks if the specified Hugging Face model ID and revision exist
        in the remote Hub.
        """
        path_or_uri = info.data["path_or_uri"]
        if not isinstance(path_or_uri, Path):
            # hf_result = cls._hf_cache.setdefault((str(path_or_uri),value),
            #        check_hf_repo_exists(repo_id=str(path_or_uri), revision=value, repo_type="model"))
            if (str(path_or_uri), value) not in cls._hf_cache:
                hf_result = check_hf_repo_exists(repo_id=str(path_or_uri), revision=value, repo_type="model")
                print("called hf repo:", path_or_uri, value)
                cls._hf_cache[(str(path_or_uri), value)] = hf_result

            # if not hf_result:
            if not cls._hf_cache[(str(path_or_uri), value)]:
                raise ValueError(
                    f"The huggingface model: path_or_uri ({path_or_uri}) or "
                    f"specified revision ({value}) cannot be found.\n"
                    "Please check the model ID and revision. If the model is "
                    "private or restricted access, make sure you have access "
                    "to it and have exported your huggingface token in your "
                    "environment variables."
                )
        return value

    def get_model_info(self) -> ModelInfo:
        """Gets the model info using the HuggingFace API and saves it as a property."""
        if isinstance(self.path_or_uri, Path):
            raise ValueError("Model info is only available for remote resources and not for files.")
        if not self.info:
            api = HfApi()
            self.info = api.model_info(repo_id=self.path_or_uri, revision=self.revision)
        return self.info


class SpeechBrainModel(HFModel[PROVIDER_T]):
    """SpeechBrain model."""

    pass


class PyannoteAudioModel(HFModel[PROVIDER_T]):
    """PyannoteAudioModel model."""

    pass


class SentenceTransformersModel(HFModel[PROVIDER_T]):
    """SentenceTransformersModel model."""

    pass


class CoquiTTSModel(SenselabModel[PROVIDER_T]):
    """CoquiTTSModel model."""

    _scope: Optional[str] = None

    @field_validator("path_or_uri", mode="before")
    def validate_path_or_uri(cls, value: Union[str, Path]) -> Union[str, Path]:
        """Validate the path_or_uri.

        This check is only for remote resources and not for files.
        It checks if the specified torch model ID and revision exist in the
        remote Hub.
        """
        if not TTS_AVAILABLE:
            raise ModuleNotFoundError(
                "`coqui-tts` is not installed. Please install senselab audio dependencies using `pip install senselab`."
            )
        if not isinstance(value, Path):
            model_ids = TTS().list_models()
            if value not in model_ids:
                raise ValueError(f"Model {value} not found. Available models: {model_ids}")
            cls._scope = value.split("/")[0]

        return value


class TorchModel(SenselabModel[PROVIDER_T]):
    """Generic torch model."""

    revision: Annotated[str, Field(validate_default=True)] = "main"

    @field_validator("revision", mode="before")
    def validate_torch_model_id(cls, value: str, info: ValidationInfo) -> Union[str, Path]:
        """Validate the path_or_uri.

        This check is only for remote resources and not for files.
        It checks if the specified torch model ID and revision exist in the
        remote Hub.
        """
        path_or_uri = info.data["path_or_uri"]
        if not isinstance(path_or_uri, Path):
            if not check_github_repo_exists(repo_id=str(path_or_uri), branch=value):
                raise ValueError("path_or_uri or specified revision is not a valid github repo")
        return value


class TorchAudioModel(SenselabModel[PROVIDER_T]):
    """TorchAudio model."""

    revision: Annotated[str, Field(validate_default=True)] = "main"

    @field_validator("revision", mode="before")
    def validate_torchaudio_model_id(cls, value: str, info: ValidationInfo) -> Union[str, Path]:
        """Validate the path_or_uri for torchaudio models.

        This check is only for remote resources and not for files.
        It checks if the specified torchaudio model ID exists.
        """
        path_or_uri = info.data["path_or_uri"]
        if not isinstance(path_or_uri, Path):
            if not check_torchaudio_model_exists(model_id=str(path_or_uri)):
                raise ValueError("path_or_uri is not a valid torchaudio model")
        return value


def check_torchaudio_model_exists(model_id: str) -> bool:
    """Private function to check if a torchaudio model exists."""
    if not TORCHAUDIO_AVAILABLE:
        raise ModuleNotFoundError(
            "`torchaudio` is not installed. Please install senselab audio dependencies using `pip install senselab`."
        )

    try:
        _ = getattr(torchaudio.pipelines, model_id)
        return True
    except AttributeError:
        return False


def is_torch_model(file_path: Path) -> bool:
    """Check if a file is a torch model."""
    try:
        _ = torch.load(file_path)
        return True
    except Exception:
        return False


def check_hf_repo_exists(repo_id: str, revision: str = "main", repo_type: str = "model") -> bool:
    """Private function to check if a Hugging Face repository exists."""
    api = HfApi()
    try:
        if repo_type == "model":
            api.model_info(repo_id=repo_id, revision=revision)
        else:
            api.list_repo_commits(repo_id=repo_id, revision=revision, repo_type=repo_type)
        return True
    except (RepositoryNotFoundError, RevisionNotFoundError):
        return False
    except Exception as e:
        raise RuntimeError(f"An error occurred checking HF repos not related to Repository and revisions existing: {e}")
        # return False


@lru_cache(maxsize=128)
def check_github_repo_exists(repo_id: str, branch: str = "main") -> bool:
    """Checks if a GitHub repository exists with caching and authentication."""
    url = f"https://api.github.com/repos/{repo_id}/branches/{branch}"
    token = os.getenv("GITHUB_TOKEN") or None

    headers = {}
    if token:
        headers = {"Authorization": f"token {token}"}

    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code == 200:
        return True
    elif response.status_code == 404:
        return False
    elif response.status_code == 403:  # Handle rate limit exceeded
        print("GitHub API rate limit exceeded. Please try again later.")
        return False
    else:
        response.raise_for_status()
        return False


# Rebuild the model classes to ensure proper generic type resolution
SenselabModel.model_rebuild()
HFModel.model_rebuild()
SpeechBrainModel.model_rebuild()
PyannoteAudioModel.model_rebuild()
SentenceTransformersModel.model_rebuild()
CoquiTTSModel.model_rebuild()
TorchModel.model_rebuild()
TorchAudioModel.model_rebuild()
