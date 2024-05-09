"""This module defines a pydra API for the preprocessing task."""
import pydra

from senselab.audio.tasks.preprocessing import resample_hf_dataset

resample_hf_dataset_pt = pydra.mark.task(resample_hf_dataset)
