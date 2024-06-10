"""This module implements some utilities for the model class."""

import os
from pathlib import Path
from typing import Optional, Union

import torch
from pydantic import BaseModel, field_validator


class SenselabModel(BaseModel):
    """Base configuration for SenselabModel class."""

    path_or_uri: Union[str, Path]
    revision: Optional[str] = None

    @field_validator("path_or_uri", mode="before")
    def validate_path_or_uri(cls, value: Union[str, Path]) -> Union[str, Path]:
        """Validate the path_or_uri.

        This check is only for files and not for remote resources.
        It does check if the path_or_uri is not empty and if it is an existing file.
        """
        if not value:
            raise ValueError("path_or_uri cannot be empty")

        if isinstance(value, Path) and not os.path.isfile(value):
            raise ValueError("path_or_uri is not an existing file")

        # If the value is a string and looks like an existing file path, convert it to a Path object
        if isinstance(value, str) and os.path.isfile(value):
            value = Path(value)
            if not is_torch_model(value):
                raise ValueError("path_or_uri does not point to a valid torch model")

        return value

def is_torch_model(file_path: Path) -> bool:
    """Check if a file is a torch model."""
    try:
        _ = torch.load(file_path)
        return True
    except Exception:
        return False
