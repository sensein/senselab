"""This script is used to test the voice cloning task."""
from senselab.audio.tasks.preprocessing import resample_hf_dataset
from senselab.audio.tasks.voice_cloning import clone_voice_in_dataset_with_KNNVC
from senselab.utils.tasks.input_output import read_files_from_disk

dataset = read_files_from_disk(["/Users/fabiocat/Documents/git/sensein/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav"])
    
print("Resampling dataset...")
dataset = resample_hf_dataset(dataset, 16000)
print("Resampled dataset.")

cloned_dataset = clone_voice_in_dataset_with_KNNVC(dataset, dataset)

print("cloned_dataset")
#print(cloned_dataset)