"""Script for detecting voice in long audio recordings using VAD and speaker diarization."""

from typing import List, Tuple

import torch
import torchaudio
from pyannote.audio import Pipeline

# Constants
AUDIO_PATH = "path/to/long/audio/file.wav"
SAMPLE_RATE = 16000


def load_audio(audio_path: str, sample_rate: int) -> torch.Tensor:
    """Loads an audio file.

    Args:
        audio_path (str): Path to the audio file.
        sample_rate (int): The sampling rate for the audio.

    Returns:
        torch.Tensor: The loaded audio tensor.
    """
    waveform, sr = torchaudio.load(audio_path)
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    return waveform


def perform_vad(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """Performs voice activity detection (VAD) on the audio.

    Args:
        waveform (torch.Tensor): The audio waveform tensor.
        sample_rate (int): The sampling rate of the audio.

    Returns:
        torch.Tensor: The VAD mask indicating speech segments.
    """
    vad = torchaudio.transforms.Vad(sample_rate=sample_rate)
    return vad(waveform)


def speaker_diarization(audio_path: str) -> List[Tuple[float, float, int]]:
    """Performs speaker diarization on the audio file.

    Args:
        audio_path (str): Path to the audio file.

    Returns:
        List[Tuple[float, float, int]]: List of tuples containing start time,
                                        end time, and speaker ID.
    """
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    diarization = pipeline({"uri": "filename", "audio": audio_path})

    result = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        result.append((turn.start, turn.end, speaker))
    return result


def main() -> None:
    """Main function to detect voice in long audio recordings and print results."""
    waveform = load_audio(AUDIO_PATH, SAMPLE_RATE)
    vad_mask = perform_vad(waveform, SAMPLE_RATE)

    # Save the VAD segments (optional)
    vad_segments = vad_mask.nonzero(as_tuple=False).squeeze()

    print("Voice Activity Detection (VAD) results:")
    print(vad_segments)

    diarization_results = speaker_diarization(AUDIO_PATH)

    print("\nSpeaker Diarization results:")
    for start, end, speaker in diarization_results:
        print(f"Start: {start:.2f}s, End: {end:.2f}s, Speaker: {speaker}")


if __name__ == "__main__":
    main()
