"""This module defines a pydra API for the speech to text task."""
import pydra

from senselab.audio.tasks.speech_to_text import transcribe_dataset

transcribe_dataset_pt = pydra.mark.task(transcribe_dataset)