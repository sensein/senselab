"""Lazy, cached availability checks for optional dependencies."""

from functools import lru_cache


@lru_cache(maxsize=1)
def torchaudio_available() -> bool:
    """Return True if torchaudio can be imported without errors."""
    try:
        import torchaudio  # noqa: F401

        return True
    except (ImportError, RuntimeError):
        return False
