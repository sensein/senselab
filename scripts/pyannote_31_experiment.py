"""Demonstrates use of pyannote_31.py with the PolyAI/minds14 dataset.

PolyAI/minds14 dataset:
https://huggingface.co/datasets/PolyAI/minds14
"""


import json

from datasets import load_dataset

from senselab.audio.tasks.preprocessing import resample_hf_dataset
from senselab.audio.tasks.pyannote_31 import pyannote_31_diarize
from senselab.utils.tasks.input_output import _from_hf_dataset_to_dict


hf_token = "YOUR HF TOKEN"
dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
dataset = dataset.select(range(4))

dataset = _from_hf_dataset_to_dict(dataset)

print("Resampling dataset...")
dataset = resample_hf_dataset(dataset, 16000)
print("Resampled dataset.")

print("Diarizing dataset...")
dataset_diarized = pyannote_31_diarize(dataset, batched=True, batch_size=2)
print("Diarized dataset.")

print(json.dumps(dataset_diarized, indent=4))
