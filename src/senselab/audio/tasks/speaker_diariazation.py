
"""
Pyannote https://huggingface.co/pyannote/speaker-diarization-3.1

function accepting as input a HFDataset and returning the output of 
pyannote speaker-diarization model for each file. 

pyannote-audio's pipeline accept 
as an input an AudioFile object, that can be a map of "waveform" and "sample_rate" 
(https://github.com/pyannote/pyannote-audio/blob/develop/pyannote/audio/core/io.py#L43), 
hence it should be an easy adaptation of the code they have. you can look at the speech_to_text.py script to get inspired. 
you can do mapping and batching to speed up the processio.py
AudioFile = Union[Text, Path, IOBase, Mapping]
"""

import torch
from datasets import load_dataset, load_dataset_builder, Audio
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.audio import Pipeline

access_token = ""

def pyannote_31_diarize_row(audio):
    """
    Diarizes an audio file with Pyannote 3.1.

    :p audio: a dictionary containing 'array' and 'sampling_rate'
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=access_token)

    #Possibly use Fabio's conversion code
    waveform = torch.tensor(audio['array'], dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension

    if waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.T

    #Possibly remove
    with ProgressHook() as hook:
        diarization = pipeline({"waveform": waveform, "sample_rate": audio['sampling_rate']}, hook=hook)

    return diarization

def pyannote_31_diarize_hf_dataset(dataset, batch=False, ):
    """
    Diarizes the audio files in a Hugging Face dataset.
    TODO: mapping and batching
    TODO: use Fabio's IO code
    """
    diarizations = []
    # dataset[i]['audio']
    #torch batching here
    #variable batch sizes
    for i in range(len(dataset['audio'])):
        diarizations += [pyannote_31_diarize_row(dataset['audio'][i])] #loading audio twice
    return diarizations


dataset = load_dataset("PolyAI/minds14", "en-US", split="train")
diarizations = pyannote_31_diarize_hf_dataset(dataset[:2])
for diarization in diarizations:
    print(diarization)


#apply row function to each row of dataset with map
#create directory with atomic functions
#merge  to get batching options
#cuda