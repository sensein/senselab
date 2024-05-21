
from datasets import load_dataset, load_dataset_builder, Audio

from senselab.audio.tasks.preprocessing import resample_hf_dataset
from senselab.utils.tasks.input_output import read_files_from_disk

from senselab.audio.tasks.pyannote_31 import pyannote_31_diarize
from senselab.utils.tasks.input_output import _from_hf_dataset_to_dict

data = {"files": 
            ["../src/tests/data_for_testing/audio_48khz_mono_16bits.wav", 
            "../src/tests/data_for_testing/audio_48khz_mono_16bits.wav" ]
        }


dataset = read_files_from_disk(data["files"])
print(f"Dataset loaded with {len(dataset)} records.")

print("Resampling dataset...")
dataset = resample_hf_dataset(dataset, 16000)
print("Resampled dataset.")

print("Diarizing dataset...")
dataset_diarized = pyannote_31_diarize(dataset)
print("Diarized dataset.")

print(dataset_diarized)


dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
dataset = dataset.select(range(2))

dataset = _from_hf_dataset_to_dict(dataset)

print("Resampling dataset...")
dataset = resample_hf_dataset(dataset, 16000)
print("Resampled dataset.")

print("Diarizing dataset...")
dataset_diarized = pyannote_31_diarize(dataset)
print("Diarized dataset.")

print(dataset_diarized)