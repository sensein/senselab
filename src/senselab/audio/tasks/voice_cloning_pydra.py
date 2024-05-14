"""This module defines a pydra API for the voice cloning task."""
import pydra

from senselab.audio.tasks.voice_cloning import clone_voice_in_dataset_with_KNNVC

clone_voice_in_dataset_with_KNNVC_pt = pydra.mark.task(clone_voice_in_dataset_with_KNNVC)