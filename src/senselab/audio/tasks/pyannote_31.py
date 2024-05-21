
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

# import torch
# from datasets import load_dataset, load_dataset_builder, Audio
from pyannote.audio.pipelines.utils.hook import ProgressHook
# from pyannote.audio import Pipeline

import torch
from datasets import load_dataset, load_dataset_builder, Audio
from pyannote.audio import Pipeline
# from pyannote.audio.utils.progress import ProgressHook  # Corrected import for ProgressHook


def annotation_to_dict(annotation):
    """
    Converts a pyannote.core.Annotation object to a dictionary.
    """
    result = {}
    for segment, _, label in annotation.itertracks(yield_label=True):
        result[str(segment)] = label
    return result

def pyannote_31_diarize_row(audio, hf_token):
    """
    Diarizes an audio file with Pyannote 3.1.
    :param audio: a dictionary containing 'array' and 'sampling_rate'
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token)
    
    waveform = torch.tensor(audio['array'], dtype=torch.float32)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension

    if waveform.shape[0] > waveform.shape[1]:
        waveform = waveform.T

    with ProgressHook() as hook:
        diarization = pipeline({"waveform": waveform, "sample_rate": audio['sampling_rate']}, hook=hook)
    
    diarization_dict = annotation_to_dict(diarization)
    return {"pyannote31_diarization":diarization_dict}

def pyannote_31_diarize(dataset, hf_token=None):
    """
    Diarizes the audio files in a Hugging Face dataset. The diarizations are added
    to the dataset as a new column 'pyannote31_diarization'
    """
    result =  dataset.map(lambda x: pyannote_31_diarize_row(x['audio'], hf_token))
    return result

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