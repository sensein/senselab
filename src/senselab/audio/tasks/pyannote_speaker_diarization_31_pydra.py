"""Pydra API for the Pyannote Speaker Diarization 3.1 task."""

import pydra

from senselab.audio.tasks.pyannote_speaker_diarization_31 import pyannote_diarize_31

transcribe_dataset_with_hf_pt = pydra.mark.task(pyannote_diarize_31)