
import torch
import json
import os
import shutil
from datasets import load_dataset, load_dataset_builder, Audio

from senselab.utils.tasks.input_output import read_dataset_from_hub
from senselab.audio.tasks.preprocessing import resample_hf_dataset


import sys
sys.path.append('/Users/isaacbevers/sensein/senselab/src/senselab')

from audio.tasks.pyannote_31 import pyannote_31_diarize

hf_token = "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"


model_cache_dir = os.path.expanduser("~/.cache/huggingface/transformers")
if os.path.exists(model_cache_dir):
    shutil.rmtree(model_cache_dir)

# cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
# if os.path.exists(cache_dir):
#     shutil.rmtree(cache_dir)

dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
dataset_first_two = dataset.select(range(2))
dataset_diarized = pyannote_31_diarize(dataset.select(range(4)))
print(json.dumps(dataset_diarized["pyannote31_diarization"], indent=4))

# dataset = read_dataset_from_hub(
#     "PolyAI/minds14",
#     split="train",
#     hf_token=hf_token
# )
# for diarization in diarizations:
#     print(diarization)
