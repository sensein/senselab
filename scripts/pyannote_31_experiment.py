import sys
import json

from datasets import load_dataset, load_dataset_builder, Audio



# import sys
# sys.path.append('/Users/isaacbevers/sensein/senselab/src/senselab')

#/Users/isaacbevers/sensein/senselab/scripts

from senselab.audio.tasks.preprocessing import resample_hf_dataset
from senselab.utils.tasks.input_output import read_files_from_disk

from senselab.audio.tasks.pyannote_31 import pyannote_31_diarize, pyannote_31_diarize_batch
from senselab.utils.tasks.input_output import _from_hf_dataset_to_dict




# data = {"files": 
#             ["../src/tests/data_for_testing/audio_48khz_mono_16bits.wav", 
#             "../src/tests/data_for_testing/audio_48khz_mono_16bits.wav" ]
#         }


# dataset = read_files_from_disk(data["files"])
# print(f"Dataset loaded with {len(dataset)} records.")

# print("Resampling dataset...")
# dataset = resample_hf_dataset(dataset, 16000)
# print("Resampled dataset.")

# print("Diarizing dataset...")
# dataset_diarized = pyannote_31_diarize(dataset)
# print("Diarized dataset.")

# print(dataset_diarized)


# dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
# dataset = dataset.select(range(3))

# dataset = _from_hf_dataset_to_dict(dataset)

# print("Resampling dataset...")
# dataset = resample_hf_dataset(dataset, 16000)
# print("Resampled dataset.")

# print("Diarizing dataset...")
# dataset_diarized = pyannote_31_diarize(dataset)
# print("Diarized dataset.")

# print(json.dumps(dataset_diarized, indent=4))


hf_token = "YOUR HF TOKEN"
dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
dataset = dataset.select(range(4))
# diarizations = []
# for row in dataset:
#     diarizations.append(pyannote_31_diarize_row(row, hf_token))
# print(json.dumps(diarizations, indent=4))

dataset = _from_hf_dataset_to_dict(dataset)

print("Resampling dataset...")
dataset = resample_hf_dataset(dataset, 16000)
print("Resampled dataset.")

print("Diarizing dataset...")
dataset_diarized = pyannote_31_diarize(dataset, batched=True, batch_size=2)
print("Diarized dataset.")

print(json.dumps(dataset_diarized, indent=4))



#run 
# Example usage:
# dataset = load_dataset("your_dataset")
# hf_token = "your_huggingface_api_token"
# diarizations = pyannote_31_diarize(dataset, hf_token=hf_token)


# def pyannote_31_diarize_row(audio, hf_token):
#     """
#     Diarizes an audio file with Pyannote 3.1.

#     :p audio: a dictionary containing 'array' and 'sampling_rate'
#     """
#     pipeline = Pipeline.from_pretrained(
#         "pyannote/speaker-diarization-3.1",
#         use_auth_token=hf_token)

#     #Possibly use Fabio's conversion code
#     waveform = torch.tensor(audio['array'], dtype=torch.float32)
#     if waveform.dim() == 1:
#         waveform = waveform.unsqueeze(0)  # Add channel dimension

#     if waveform.shape[0] > waveform.shape[1]:
#         waveform = waveform.T

#     #Possibly remove
#     with ProgressHook() as hook:
#         diarization = pipeline({"waveform": waveform, "sample_rate": audio['sampling_rate']}, hook=hook)

#     return diarization

# def pyannote_31_diarize(dataset, batch=False, hf_token=None):
#     """
#     Diarizes the audio files in a Hugging Face dataset.
#     TODO: mapping and batching
#     TODO: use Fabio's IO code
#     """
#     diarizations = []
#     # dataset[i]['audio']
#     #torch batching here
#     #variable batch sizes
#     diarizations = dataset.map(pyannote_31_diarize_row, dataset, hf_token=hf_token)
#     # for i in range(len(dataset['audio'])):
#     #     diarizations += [pyannote_31_diarize_row(dataset['audio'][i], hf_token)] #loading audio twice
#     return diarizations


# # diar = hf_dataset.map(lambda x: _resample_hf_row(x, resample_rate, rolloff))
# #apply row function to each row of dataset with map
# #create directory with atomic functions
# #merge  to get batching options
# #cuda
# #Fabio's IO
