"""This module provides the implementation of pipepal tasks for audio."""

from .example_task import Interface as ExampleTask
from .input_output import Interface as IOTask

__all__ = ['ExampleTask', 'IOTask']
