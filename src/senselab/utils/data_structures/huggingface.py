"""This module provides the implementation of Hugging Face utilities."""
from pathlib import Path
from typing import Union

from huggingface_hub import HfApi
from pydantic import Field, ValidationInfo, field_validator
from typing_extensions import Annotated

from senselab.utils.data_structures.model import SenselabModel


def check_hf_repo_exists(repo_id: str, revision: str = "main", repo_type: str = "model") -> bool:
    """Private function to check if a Hugging Face repository exists."""
    api = HfApi()
    try:
        api.list_repo_commits(repo_id=repo_id, revision=revision, repo_type=repo_type)
        return True
    except Exception:
        # raise RuntimeError(f"An error occurred: {e}")
        return False

class HFModel(SenselabModel):
    """Hugging Face model."""

    revision: Annotated[str, Field(validate_default=True)] = "main"

    @field_validator("revision", mode="before")
    def validate_hf_model_id(cls, value: str, info: ValidationInfo) -> Union[str, Path]:
        """Validate the path_or_uri.

        This check is only for remote resources and not for files.
        It checks if the specified Hugging Face model ID and revision exist in the remote Hub.
        """
        path_or_uri = info.data["path_or_uri"]
        if not isinstance(path_or_uri, Path):
            if not check_hf_repo_exists(repo_id=str(path_or_uri), revision=value, repo_type="model"):
                raise ValueError("path_or_uri or specified revision is not a valid Hugging Face model")
        return value
      