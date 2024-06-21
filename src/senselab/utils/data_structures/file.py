"""This module provides the implementation of file utilities."""

import os
from pathlib import Path
from typing import List

from pydantic import BaseModel, FilePath, field_validator

from senselab.utils.config import get_config


class File(BaseModel):
    """This class represents a file."""

    filepath: FilePath

    @property
    def type(self) -> str:
        """Determine the type of the file based on its extension."""
        extension = os.path.splitext(self.filepath)[1].lower()
        for type_name, type_info in get_config()["files"].items():
            if extension in type_info["extensions"]:
                return type_name
        raise ValueError("File type could not be determined from extension.")

    @field_validator("filepath")
    def validate_filepath(cls, v: FilePath) -> FilePath:
        """Validate the filepath."""
        if not os.path.exists(v):
            raise ValueError(f"File {v} does not exist")

        file_extension = os.path.splitext(v)[1].lower()
        valid_extensions = [ext for types in get_config()["files"].values() for ext in types["extensions"]]

        if file_extension not in valid_extensions:
            raise ValueError(
                f"Unsupported file extension: {file_extension}." f"Supported extensions: {valid_extensions}"
            )

        return v


def from_strings_to_files(list_of_files: List[str]) -> List[File]:
    """Create a list of `File` objects from a list of strings."""
    return [File(filepath=Path(file)) for file in list_of_files]


def get_common_directory(files: List[str]) -> str:
    """A function to get the common directory from a list of file paths.

    Parameters:
    - files: a list of file paths

    Returns:
    - the common directory among the file paths
    """
    if len(files) == 1:
        # Ensure the single path's directory ends with a separator
        common_path = os.path.dirname(files[0])
    else:
        # Use commonpath to find the common directory for multiple files
        common_path = os.path.commonpath(files)

    # Check if the path ends with the os separator, add if not
    if not common_path.endswith(os.sep):
        common_path += os.sep

    return common_path
