"""Pydra API for the Pyannote Speaker Diarization 3.1 task."""

import pydra

from senselab.audio.tasks.pyannote_speaker_diarization import pyannote_diarize

transcribe_dataset_with_hf_pt = pydra.mark.task(pyannote_diarize)