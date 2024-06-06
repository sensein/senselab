"""This module defines a pydra API for the speech to text task."""

import pydra

from senselab.audio.tasks.speech_to_text import transcribe_audios

transcribe_audios_pt = pydra.mark.task(transcribe_audios)
