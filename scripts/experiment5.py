"""This script is used to test the audio tasks."""

from typing import Any, Dict

from torch_audiomentations import Compose, Gain, PolarityInversion

from senselab.audio.tasks.data_augmentation import augment_hf_dataset
from senselab.utils.decorators import get_response_time
from senselab.utils.tasks.input_output import read_files_from_disk


@get_response_time
def workflow(data: Dict[str, Any], augmentation: Compose) -> None:
    """This function reads audio files from disk, and transcribes them using Whisper."""
    print("Starting to read files from disk...")
    dataset = read_files_from_disk(data["files"])
    print(f"Dataset loaded with {len(dataset)} records.")

    print("Augmenting dataset...")
    dataset = augment_hf_dataset(dataset, augmentation)
    print("Augmented dataset.")



# Initialize augmentation callable
apply_augmentation = Compose(
    transforms=[
        Gain(
            min_gain_in_db=-15.0,
            max_gain_in_db=5.0,
            p=0.5,
        ),
        PolarityInversion(p=0.5)
    ]
)

data = {"files": 
            ["/Users/fabiocat/Documents/git/sensein/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav", 
            "/Users/fabiocat/Documents/git/sensein/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav"]
        }

workflow(data, apply_augmentation)