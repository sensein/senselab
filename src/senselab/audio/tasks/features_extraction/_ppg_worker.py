"""PPGs worker — runs inside the isolated subprocess venv.

This module is called by call_in_venv() and should NOT be imported
directly by senselab. It has access to ppgs, espnet, and its own torch.
"""

import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torchaudio


def run_ppg_extraction(
    audio_paths: List[str],
    device: str,
    output_dir: str,
) -> Dict[str, List[str]]:
    """Extract PPGs in the isolated ppgs venv.

    Args:
        audio_paths: Paths to input FLAC files.
        device: "cpu" or "cuda".
        output_dir: Directory for output .npy files.

    Returns:
        Dict with "output_paths" list of PPG numpy file paths.
    """
    import ppgs

    gpu = 0 if device == "cuda" else None

    output_paths = []
    for i, audio_path in enumerate(audio_paths):
        waveform, sr = torchaudio.load(audio_path)

        try:
            posteriorgram = ppgs.from_audio(
                torch.unsqueeze(waveform, dim=0),
                ppgs.SAMPLE_RATE,
                gpu=gpu,
            ).cpu()
        except RuntimeError as e:
            print(f"RuntimeError extracting PPGs for audio {i}: {e}")
            print(traceback.format_exc())
            posteriorgram = torch.tensor(float("nan"))

        out_path = str(Path(output_dir) / f"ppg_{i}.npy")
        np.save(out_path, posteriorgram.numpy())
        output_paths.append(out_path)

    return {"output_paths": output_paths}
