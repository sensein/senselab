"""This module provides the implementation of config utilities."""

import json
import os
from typing import Any, Dict


def get_config() -> Dict[str, Any]:
    """Reads and returns the contents of the config file."""
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    # Build the path to the config file
    config_file_path = os.path.join(script_dir, "config.json")
    # Load the config file
    with open(config_file_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    return config
