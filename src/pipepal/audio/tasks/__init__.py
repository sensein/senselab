"""This module provides the implementation of pipepal tasks for audio."""

from .example_task import Interface as ExampleTask
from .input_output import Interface as IOTask
from .raw_signal_processing import Interface as RawSignalProcessingTask

__all__ = ['ExampleTask', 'IOTask', 'RawSignalProcessingTask']
