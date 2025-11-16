from typing import Set, List
from pathlib import Path


def get_audio_files_from_directory(
    directory_path: str,
    audio_extensions: Set[str] = {
        ".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".wma"
    },
    recursive: bool = True,
) -> List[str]:
    """Get a list of all audio files in a directory.

    Args:
        directory_path: Path to the directory to search for audio files
        audio_extensions: Set of audio file extensions to search for
                         (default includes common formats)
        recursive: If True, search recursively through subdirectories

    Returns:
        List of paths to audio files found in the directory
    """
    audio_files: List[Path] = []
    directory = Path(directory_path)

    if not directory.exists():
        print(f"Warning: Directory {directory_path} does not exist")
        return []

    if not directory.is_dir():
        print(f"Warning: {directory_path} is not a directory")
        return []

    # Convert extensions to lowercase for case-insensitive matching
    extensions_lower = {ext.lower() for ext in audio_extensions}

    # Search for audio files
    if recursive:
        # Find all files recursively
        for ext in extensions_lower:
            audio_files.extend(directory.rglob(f"*{ext}"))
            # Also search for uppercase extensions
            audio_files.extend(directory.rglob(f"*{ext.upper()}"))
    else:
        # Find files in current directory only
        for ext in extensions_lower:
            audio_files.extend(directory.glob(f"*{ext}"))
            # Also search for uppercase extensions
            audio_files.extend(directory.glob(f"*{ext.upper()}"))

    # Convert to strings, remove duplicates, and sort for consistent ordering
    audio_files_str = list(set(str(path) for path in audio_files))
    audio_files_str.sort()

    print(f"Found {len(audio_files_str)} audio files in {directory_path}")
    if audio_files_str:
        # Show summary of file types found
        extensions_found = set(Path(f).suffix.lower() for f in audio_files_str)
        print(f"File types found: {', '.join(sorted(extensions_found))}")

    return audio_files_str
