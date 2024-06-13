"""Forced alignment script using Wav2Vec2 from Hugging Face's transformers library."""

import numpy as np
import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Constants
MODEL_NAME = "facebook/wav2vec2-large-960h-lv60-self"
AUDIO_PATH = "path/to/audio/file.wav"
SAMPLING_RATE = 16000


def load_model_and_processor(model_name: str) -> tuple[Wav2Vec2Processor, Wav2Vec2ForCTC]:
    """Loads the Wav2Vec2 model and processor.

    Args:
        model_name (str): The name of the pre-trained Wav2Vec2 model.

    Returns:
        tuple: A tuple containing the Wav2Vec2 processor and model.
    """
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    return processor, model


def load_audio(audio_path: str) -> np.ndarray:
    """Loads an audio file.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        np.ndarray: The loaded audio data.
    """
    audio_input, _ = sf.read(audio_path)
    return audio_input


def preprocess_audio(processor: Wav2Vec2Processor, audio_input: np.ndarray, sampling_rate: int) -> torch.Tensor:
    """Preprocesses the audio input.

    Args:
        processor (Wav2Vec2Processor): The Wav2Vec2 processor.
        audio_input (np.ndarray): The audio input data.
        sampling_rate (int): The sampling rate of the audio.

    Returns:
        torch.Tensor: The preprocessed audio input values.
    """
    input_values = processor(audio_input, return_tensors="pt", sampling_rate=sampling_rate).input_values
    return input_values


def perform_inference(model: Wav2Vec2ForCTC, input_values: torch.Tensor) -> torch.Tensor:
    """Performs inference to get logits from the model.

    Args:
        model (Wav2Vec2ForCTC): The Wav2Vec2 model.
        input_values (torch.Tensor): The preprocessed audio input values.

    Returns:
        torch.Tensor: The logits output from the model.
    """
    with torch.no_grad():
        logits = model(input_values).logits
    return logits


def decode_predictions(processor: Wav2Vec2Processor, logits: torch.Tensor) -> list[str]:
    """Decodes the predictions to get the transcription.

    Args:
        processor (Wav2Vec2Processor): The Wav2Vec2 processor.
        logits (torch.Tensor): The logits output from the model.

    Returns:
        list: The decoded transcription.
    """
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription


def forced_alignment(
    logits: torch.Tensor, transcription: list[str], processor: Wav2Vec2Processor
) -> list[tuple[int, int]]:
    """Performs forced alignment between the audio and transcription.

    Args:
        logits (torch.Tensor): The logits output from the model.
        transcription (list[str]): The decoded transcription.
        processor (Wav2Vec2Processor): The Wav2Vec2 processor.

    Returns:
        list: A list of tuples representing the alignment between
        audio frames and transcription tokens.
    """
    predicted_ids = torch.argmax(logits, dim=-1).cpu().numpy().squeeze()
    transcription_ids = processor.tokenizer(transcription[0]).input_ids

    # Create cost matrix
    cost = np.zeros((len(predicted_ids), len(transcription_ids)))

    for i in range(len(predicted_ids)):
        for j in range(len(transcription_ids)):
            if predicted_ids[i] == transcription_ids[j]:
                cost[i, j] = 0
            else:
                cost[i, j] = 1

    # Perform dynamic programming
    alignment = []
    i, j = 0, 0
    while i < len(predicted_ids) and j < len(transcription_ids):
        alignment.append((i, j))
        if i == len(predicted_ids) - 1:
            j += 1
        elif j == len(transcription_ids) - 1:
            i += 1
        elif cost[i + 1, j] < cost[i, j + 1]:
            i += 1
        else:
            j += 1

    return alignment


def main() -> None:
    """Main function to perform forced alignment and print the results."""
    processor, model = load_model_and_processor(MODEL_NAME)
    audio_input = load_audio(AUDIO_PATH)
    input_values = preprocess_audio(processor, audio_input, SAMPLING_RATE)
    logits = perform_inference(model, input_values)
    transcription = decode_predictions(processor, logits)
    alignment = forced_alignment(logits, transcription, processor)

    for audio_idx, text_idx in alignment:
        print(f"Audio frame {audio_idx} aligned with transcription token {text_idx}")


if __name__ == "__main__":
    main()
