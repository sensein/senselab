
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
from typing import Any, Dict
from datasets import load_dataset, load_dataset_builder, Audio

from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from senselab.utils.tasks.input_output import _from_dict_to_hf_dataset, _from_hf_dataset_to_dict


def annotation_to_dict(annotation):
    """
    Converts a pyannote.core.Annotation object to a dictionary.
    """
    result = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        result.append((str(segment), label))
    return result

def pyannote_31_diarize_batch(batch,  hf_token):
    """
    Diarizes an audio file with Pyannote 3.1.
    :param audio: a dictionary containing 'array' and 'sampling_rate'
    """
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token)
    
    diarizations = []
    for audio in batch['audio']:
        waveform = torch.tensor(audio['array'], dtype=torch.float32)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension

        if waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T

        with ProgressHook() as hook:
            diarization = pipeline({"waveform": waveform, "sample_rate": batch['audio'][0]['sampling_rate']}, hook=hook)
        
        diarization = annotation_to_dict(diarization)
        diarizations.append(diarization)
    results = {'pyannote31_diarizations': diarizations}
    return results    

def pyannote_31_diarize(dataset: Dict[str, Any], 
                        hf_token: str = None, 
                        batched: bool = False, 
                        batch_size: int = 1):
    """
    Diarizes the audio files in a Hugging Face dataset. The diarizations are added
    to the dataset as a new column 'pyannote31_diarization'
    """
    hf_dataset = _from_dict_to_hf_dataset(dataset)
    result =  hf_dataset.map(lambda x: pyannote_31_diarize_batch(x, hf_token), 
                             batched=batched, 
                             batch_size=batch_size,
                             cache_file_name="./cache",
                             remove_columns=["audio"])
    result = _from_hf_dataset_to_dict(result)
    return result