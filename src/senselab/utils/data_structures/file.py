"""File utilities: validation, typing, and recursive discovery.

This module defines a `File` model that validates existence and supported
extensions against the Senselab configuration, along with helpers to convert
string paths to `File` objects, compute a common directory, and **recursively
list files** from a folder with optional filters.

Notes:
    - Paths are normalized to **absolute** paths via `Path.resolve()`.
    - Supported types and extensions are read from `get_config()["files"]`,
      where each type entry contains an `"extensions"` list (e.g., audio/video).
"""

import os
from pathlib import Path
from typing import List, Optional, Sequence, Union

from pydantic import BaseModel, FilePath, field_validator

from senselab.utils.config import get_config


class File(BaseModel):
    """Validated file reference (absolute path + supported extension).

    Attributes:
        filepath (FilePath): Absolute path to an existing file.

    Example:
        >>> f = File(filepath=Path("sample.wav").resolve())
        >>> f.type  # e.g., "audio"
        'audio'
    """

    filepath: FilePath

    @property
    def type(self) -> str:
        """Return the logical file type based on extension (e.g., "audio", "video").

        The mapping is taken from `get_config()["files"]`, where each entry
        declares `"extensions"` (lowercased with dot prefix).

        Returns:
            str: The matching type name.

        Raises:
            ValueError: If the extension does not match any configured type.
        """
        extension = os.path.splitext(self.filepath)[1].lower()
        for type_name, type_info in get_config()["files"].items():
            if extension in type_info["extensions"]:
                return type_name
        raise ValueError("File type could not be determined from extension.")

    @field_validator("filepath")
    def validate_filepath(cls, v: FilePath) -> FilePath:
        """Validate that the file exists and has a supported extension.

        Args:
            v (FilePath): Path to an existing file.

        Returns:
            FilePath: The same path after validation.

        Raises:
            ValueError: If the file does not exist or has an unsupported extension.
        """
        if not os.path.exists(v):
            raise ValueError(f"File {v} does not exist")

        file_extension = os.path.splitext(v)[1].lower()
        valid_extensions = [ext for types in get_config()["files"].values() for ext in types["extensions"]]

        if file_extension not in valid_extensions:
            raise ValueError(f"Unsupported file extension: {file_extension}. Supported extensions: {valid_extensions}")

        return v


def from_strings_to_files(list_of_files: List[str]) -> List[File]:
    """Convert a list of path strings to validated `File` objects (absolute paths).

    Args:
        list_of_files (list[str]):
            File paths (relative or absolute). Each is resolved to an absolute path.

    Returns:
        list[File]: Validated file objects.

    Raises:
        ValueError: If any file does not exist or has an unsupported extension.

    Example:
        >>> files = from_strings_to_files(["sample.wav", "movie.mp4"])
        >>> isinstance(files[0], File)
        True
    """
    return [File(filepath=Path(file).resolve()) for file in list_of_files]


def get_common_directory(files: List[str]) -> str:
    """Return the common directory among the given file paths (with trailing separator).

    Args:
        files (list[str]): One or more file paths.

    Returns:
        str: The common directory path ending with the OS path separator.

    Example:
        >>> get_common_directory(["/a/b/c.wav", "/a/b/d.mp3"])
        '/a/b/'
    """
    if len(files) == 1:
        common_path = os.path.dirname(files[0])
    else:
        common_path = os.path.commonpath(files)

    if not common_path.endswith(os.sep):
        common_path += os.sep

    return common_path


def list_files_recursively(
    root: Union[str, os.PathLike],
    *,
    types: Optional[Sequence[str]] = None,
    extensions: Optional[Sequence[str]] = None,
    include_hidden: bool = False,
    follow_symlinks: bool = False,
) -> List[File]:
    """Recursively discover files under a folder, optionally filtering by type/extension.

    This scans `root` (recursively) and returns validated `File` objects for any
    files whose extensions are supported by the Senselab config and match the
    provided filters.

    Filtering:
        - `types`: logical categories declared in `get_config()["files"]` (e.g., `"audio"`, `"video"`).
        - `extensions`: explicit file extensions (with or without leading dot), case-insensitive.
        - If **both** are provided, the allowed set is the **union** of the two.

    Args:
        root (str | os.PathLike):
            Directory to scan.
        types (Sequence[str] | None, optional):
            Restrict results to these configured types (e.g., `["audio"]`).
        extensions (Sequence[str] | None, optional):
            Restrict results to these extensions (e.g., `[".wav", "mp3"]`).
        include_hidden (bool, optional):
            Include files/dirs starting with ".". Defaults to `False`.
        follow_symlinks (bool, optional):
            Whether to follow directory symlinks. Defaults to `False`.

    Returns:
        list[File]: Validated file entries with absolute paths.

    Raises:
        ValueError:
            - If `root` does not exist or is not a directory.
            - If any requested `types` is not present in the config.

    Examples:
        List all supported files (any type) under a dataset folder:
            >>> from senselab.utils.data_structures.file import list_files_recursively
            >>> files = list_files_recursively("data/")
            >>> len(files) >= 0
            True

        Only audio files:
            >>> audio_files = list_files_recursively("data/", types=["audio"])

        Only certain extensions (case-insensitive, dot optional):
            >>> wavs = list_files_recursively("data/", extensions=["wav", ".flac"])

        Combine filters (union):
            >>> audio_and_mp4 = list_files_recursively("data/", types=["audio"], extensions=[".mp4"])
    """
    root_path = Path(root).resolve()
    if not root_path.exists() or not root_path.is_dir():
        raise ValueError(f"Root path must be an existing directory, got: {root}")

    cfg = get_config()
    cfg_files = cfg.get("files", {})

    # Collect allowed extensions from `types`
    allowed_from_types: set[str] = set()
    if types:
        for t in types:
            if t not in cfg_files:
                raise ValueError(f"Unknown file type '{t}'. Known types: {list(cfg_files.keys())}")
            for ext in cfg_files[t].get("extensions", []):
                allowed_from_types.add(ext.lower())

    # Normalize explicit `extensions`
    allowed_from_exts: set[str] = set()
    if extensions:
        for ext in extensions:
            ext_lc = ext.lower()
            if not ext_lc.startswith("."):
                ext_lc = "." + ext_lc
            allowed_from_exts.add(ext_lc)

    # If no filters given, allow any configured extension
    if not allowed_from_types and not allowed_from_exts:
        allowed_exts = {ext.lower() for t in cfg_files.values() for ext in t.get("extensions", [])}
    else:
        allowed_exts = allowed_from_types | allowed_from_exts  # union

    results: List[File] = []

    for dirpath, dirnames, filenames in os.walk(root_path, followlinks=follow_symlinks):
        # Optionally prune hidden directories
        if not include_hidden:
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for name in filenames:
            if not include_hidden and name.startswith("."):
                continue
            ext = os.path.splitext(name)[1].lower()
            if ext not in allowed_exts:
                continue

            full_path = Path(dirpath) / name
            try:
                results.append(File(filepath=full_path.resolve()))
            except Exception:
                # Skip files that fail validation (e.g., extension mismatch race condition)
                continue

    # Make output deterministic
    results.sort(key=lambda f: str(f.filepath))
    return results
