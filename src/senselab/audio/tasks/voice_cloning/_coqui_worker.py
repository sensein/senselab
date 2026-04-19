"""Coqui TTS worker — runs inside the isolated subprocess venv.

This module is called by call_in_venv() and should NOT be imported
directly by senselab. It has access to coqui-tts and its own torch.
"""

from pathlib import Path
from typing import Dict, List

import torch
import torchaudio
from TTS.api import TTS


def run_voice_cloning(
    source_paths: List[str],
    target_paths: List[str],
    model_id: str,
    device: str,
    output_dir: str,
) -> Dict[str, List[str]]:
    """Run voice cloning in the isolated coqui venv.

    Args:
        source_paths: Paths to source FLAC files.
        target_paths: Paths to target FLAC files.
        model_id: Coqui TTS model identifier.
        device: "cpu" or "cuda".
        output_dir: Directory for output FLAC files.

    Returns:
        Dict with "output_paths" list of cloned audio file paths.
    """
    tts = TTS(model_id).to(device=device)
    audio_config = tts.voice_converter.vc_config.audio

    expected_sample_rate = (
        audio_config.input_sample_rate
        if hasattr(audio_config, "input_sample_rate")
        else getattr(audio_config, "sample_rate", None)
    )

    output_sample_rate = (
        audio_config.output_sample_rate
        if hasattr(audio_config, "output_sample_rate")
        else getattr(audio_config, "sample_rate", expected_sample_rate)
    )

    output_paths = []
    for i, (src_path, tgt_path) in enumerate(zip(source_paths, target_paths)):
        src_wav, src_sr = torchaudio.load(src_path)
        tgt_wav, tgt_sr = torchaudio.load(tgt_path)

        # Validate sampling rates match model expectation
        if expected_sample_rate is not None:
            if src_sr != expected_sample_rate:
                raise ValueError(f"[Pair {i}] Source sample rate {src_sr} != model expected {expected_sample_rate}")
            if tgt_sr != expected_sample_rate:
                raise ValueError(f"[Pair {i}] Target sample rate {tgt_sr} != model expected {expected_sample_rate}")

        converted = tts.voice_conversion(
            source_wav=src_wav.squeeze(),
            target_wav=tgt_wav.squeeze(),
        )

        if not isinstance(converted, torch.Tensor):
            converted = torch.tensor(converted)
        if converted.dim() == 1:
            converted = converted.unsqueeze(0)

        # Fall back to source sample rate if output_sample_rate is None
        sr = output_sample_rate or src_sr
        out_path = str(Path(output_dir) / f"cloned_{i}.flac")
        torchaudio.save(out_path, converted, sr, format="flac")
        output_paths.append(out_path)

    return {"output_paths": output_paths}
