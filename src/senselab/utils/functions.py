"""Utility functions for senselab."""

import os
from typing import List


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
