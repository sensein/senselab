"""This module provides the implementation of IO for videos."""

from .example_service import Service as ExampleService
from .torchaudio import Service as TorchaudioService

__all__ = ['ExampleService', 'TorchaudioService']
