"""This module contains functions to check if ffmpeg is installed and is compatible with senselab dependencies."""

import subprocess
from typing import Optional


def check_ffmpeg_version(min_version: Optional[float] = None, max_version: Optional[float] = None) -> None:
    """Check if ffmpeg is installed and its version is compatible with torchaudio."""
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        version_line = result.stdout.splitlines()[0]
        version_number = version_line.split()[2]  # Extract the version string
        version_float = float(".".join(version_number.split(".")[:2]))  # Convert to a float (e.g., "4.3" from "4.3.1")

        if (max_version is not None and version_float >= max_version) or (
            min_version is not None and version_float <= min_version
        ):
            raise RuntimeError(
                f"senselab requires ffmpeg version < {max_version} and > {min_version}. "
                "Please install a compatible version."
            )
    except FileNotFoundError as exc:
        raise FileNotFoundError("ffmpeg is not installed or not in PATH.") from exc
    except (IndexError, ValueError) as exc:
        raise RuntimeError("Failed to parse ffmpeg version. Ensure ffmpeg is correctly installed.") from exc
