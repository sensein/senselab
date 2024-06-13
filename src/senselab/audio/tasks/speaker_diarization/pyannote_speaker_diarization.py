"""Diarizes a dataset with Pyannote speaker diarization 3.1.

If it runs very quickly with little output after running once, delete the
cache and re-run. Not tested with models other than speaker diarization 3.1.

Pyannote speaker diarization 3.1:
https://huggingface.co/pyannote/speaker-diarization-3.1
"""


import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation

from senselab.utils.tasks.input_output import (
    _from_dict_to_hf_dataset,
    _from_hf_dataset_to_dict,
)


def _annotation_to_dict(annotation: Annotation) -> List[Tuple]:
    """Convert a Pyannote annotation to a list of tuples.

    Args:
        annotation: The Pyannote annotation object.

    Returns:
        dirization_list:
          A list of tuples where each tuple contains a segment and a label.
    """
    dirization_list = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        dirization_list.append((str(segment), label))
    return dirization_list


def _pyannote_diarize_batch(
    batch: Dataset,
    hf_token: Optional[str],
    model_name: str,
    model_revision: str
) -> Dict[str, Any]:
    """Diarize a batch of audio files using the Pyannote diarization model.

    Args:
        batch: A batch of audio files from a Hugging Face dataset.
        hf_token: The Hugging Face API token.
        model_name: The model name used.
        model_revision: The model version used.

    Returns:
        A dictionary containing the diarizations for the batch.Becomes a
          column in the dataset when returned.
    """
    pipeline = Pipeline.from_pretrained(
        model_name + "-" + model_revision,
        use_auth_token=hf_token)

    diarizations = []
    for audio in batch['audio']:
        waveform = torch.tensor(audio['array'], dtype=torch.float32)

        # Add the channel dimension if the waveform is 1-dimensional.
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # Transpose if the number of channels exceeds the number of samples.
        if waveform.shape[0] > waveform.shape[1]:
            waveform = waveform.T

        # Apply the pipeline to the waveform.
        with ProgressHook() as hook:
            diarization = pipeline(
                {"waveform": waveform,
                    "sample_rate": batch['audio'][0]['sampling_rate']},
                hook=hook)

        diarization = _annotation_to_dict(diarization)
        diarizations.append(diarization)
    return {'pyannote31_diarizations': diarizations}


def pyannote_diarize(
    dataset: Dict[str, Any],
    hf_token: Optional[str] = None,
    batched: bool = False,
    batch_size: int = 1,
    cache_path: str = "scripts/cache/",
    model_name: str = "pyannote/speaker-diarization",
    model_revision: str = "3.1",  # 2.0, 3.0, or 3.1
) -> Dict[str, Any]:
    """Diarizes the audio files in a Hugging Face dataset.

    The diarizations are
    added to the dataset as a new column 'pyannote31_diarization'.

    Args:
        dataset: The dataset containing audio files to be diarized.
        hf_token: The Hugging Face API token, if required for access.
        batched: Whether to process the dataset in batches.
        batch_size: Number of samples to process in each batch,
          if batching is enabled.
        cache_path: The path to the cache directory.
        model_name: The model name used.
        model_revision: The model version used.

    Returns:
        hf_dataset_diarized: The dataset with an added column 
          'pyannote31_diarization' containing the diarizations.
    """
    if not os.path.exists(cache_path):
        os.mkdir(cache_path)

    hf_dataset = _from_dict_to_hf_dataset(dataset)
    hf_dataset_diarized = hf_dataset.map(
                            lambda x: _pyannote_diarize_batch(x, hf_token,
                                                              model_name,
                                                              model_revision),
                            batched=batched,
                            batch_size=batch_size,
                            cache_file_name=cache_path + "pyannote31_cache",
                            remove_columns=["audio"])
    hf_dataset_diarized = _from_hf_dataset_to_dict(hf_dataset_diarized)
    return hf_dataset_diarized
