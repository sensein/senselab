"""This module implements some utilities for the speech-to-text task."""
from typing import Any, Dict

from datasets import Dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from senselab.utils.functions import _select_device_and_dtype
from senselab.utils.hf import HFModel
from senselab.utils.tasks.input_output import _from_dict_to_hf_dataset, _from_hf_dataset_to_dict


def transcribe_dataset(dataset: Dict, model_id: str, audio_column: str = 'audio') -> Dict[str, Any]:
    """Transcribes all audio samples in the dataset."""
    _ = HFModel(model_id=model_id) # check HF model is valid
    hf_dataset = _from_dict_to_hf_dataset(dataset)
    transcript_dataset = _transcribe_dataset_with_hf(dataset=hf_dataset, model_id=model_id, audio_column=audio_column)
    return _from_hf_dataset_to_dict(transcript_dataset)

def _transcribe_dataset_with_hf(
    dataset: Dataset,
    model_id: str,
    low_cpu_mem_usage: bool = True,
    use_safetensors: bool = True,
    max_new_tokens: int = 128,
    chunk_length_s: int = 30,
    batch_size: int = 16,
    return_timestamps: bool = True,
    audio_column: str = 'audio'
) -> Dict[str, Any]:
    """Transcribes all audio samples in the dataset and adds the transcriptions as a new column.
    
    # TODO: optmizing the function replacing the loop with pydra wf
    """
    _ = HFModel(model_id=model_id) # check HF model is valid

    # Using batched map function to transcribe audio in batches for efficiency
    device, torch_dtype = _select_device_and_dtype(device_options=['cuda', 'cpu'])
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=low_cpu_mem_usage, use_safetensors=use_safetensors).to(device)
    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=max_new_tokens,
        chunk_length_s=chunk_length_s,
        batch_size=batch_size,
        return_timestamps=return_timestamps,
        torch_dtype=torch_dtype,
        device=device,
    )

    def _transcribe_row_with_hf(dataset_row: Dataset, pipe: pipeline, audio_column_name: str ="audio") -> Dict[str, Any]:
        """Transcribes a batch of audio samples and retains the original batch data."""
        # Perform transcription
        transcriptions = pipe(dataset_row[audio_column_name])
        return transcriptions
    
    transcripts = []
    for row in dataset:
        result = _transcribe_row_with_hf(row, pipe, audio_column_name=audio_column)
        transcripts.append(result)
    transcript_dataset = Dataset.from_dict({"asr": transcripts})
    return _from_hf_dataset_to_dict(transcript_dataset)
