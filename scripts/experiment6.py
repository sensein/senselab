"""This script is used to test the audio tasks."""

from typing import Any, Dict

from senselab.audio.tasks.preprocessing import resample_hf_dataset
from senselab.audio.tasks.speech_to_text import transcribe_dataset_with_hf
from senselab.text.tasks.sentence_transofmers_embeddings_extraction import extract_embeddings_from_hf_dataset
from senselab.utils.decorators import get_response_time
from senselab.utils.tasks.input_output import read_files_from_disk


@get_response_time
def workflow(data: Dict[str, Any]) -> None:
    """This function reads audio files from disk, and transcribes them using Whisper."""
    print("Starting to read files from disk...")
    dataset = read_files_from_disk(data["files"])
    print(f"Dataset loaded with {len(dataset)} records.")

    print("Resampling dataset...")
    dataset = resample_hf_dataset(dataset, 16000)
    print("Resampled dataset.")

    print("Transcribing dataset...")
    transcript_dataset = transcribe_dataset_with_hf(dataset=dataset, model_id="openai/whisper-tiny", language="en") # facebook/wav2vec2-base-960h
    print("Transcribed dataset.")

    print("Extracting embeddings...")
    _ = extract_embeddings_from_hf_dataset(transcript_dataset, model_id='sentence-transformers/paraphrase-MiniLM-L6-v2', text_column='asr')
    print("Extracted embeddings.")

data = {"files": 
            ["/Users/fabiocat/Documents/git/sensein/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav", 
            "/Users/fabiocat/Documents/git/sensein/senselab/src/tests/data_for_testing/audio_48khz_mono_16bits.wav"]
        }

workflow(data)
print("\n\n")