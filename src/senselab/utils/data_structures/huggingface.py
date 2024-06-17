"""This module provides the implementation of Hugging Face utilities."""

import os
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import HfApi
from pydantic import BaseModel, field_validator


class HFModel(BaseModel):
    """Hugging Face model."""

    hf_model_id: Union[str, Path]
    revision: str = "main"

    @field_validator("hf_model_id")
    def validate_hf_model_id(cls, value: str) -> str:
        """Validate the hf_model_id.

        # TODO: enabling using HF token
        """
        if not value:
            raise ValueError("hf_model_id cannot be empty")
        if not os.path.isfile(value) and not _check_hf_repo_exists(
            repo_id=value, revision="main", repo_type="model", token=None
        ):
            raise ValueError("hf_model_id is not a valid Hugging Face model")
        return value


def _check_hf_repo_exists(
    repo_id: str, revision: str = "main", repo_type: str = "model", token: Optional[str] = None
) -> bool:
    """Private function to check if a Hugging Face repository exists."""
    api = HfApi()
    try:
        api.list_repo_commits(repo_id=repo_id, revision=revision, repo_type=repo_type, token=token)
        return True
    except Exception:
        # raise RuntimeError(f"An error occurred: {e}")
        return False
